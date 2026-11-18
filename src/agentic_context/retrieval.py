# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Retrieval clients for the Context Service (Beacon) read-side REST API.

The read-side endpoints use three different header conventions — kept separate
from the ingestion clients (client.py / client_async.py) for that reason:

  POST (ingestion)       → headers: dataspace, owner_id, agent_id
  GET conversation/list  → headers: x-sfdc-dataspace, agent_id
  GET/POST context       → headers: ctx-dataspace-ds_name, agent_id

SessionManager composes these clients; application code should call the
manager methods rather than instantiating these classes directly.

Endpoint reference (verified against the live API):

  POST /api/v1/agentic/context
    Purpose : Retrieve in-session summary + cross-conversation user facts.
              Call this BEFORE every LLM turn and inject the result into the
              system prompt. This is what gives the agent memory — without it
              the LLM has no access to prior sessions even if data was written.
              The same endpoint serves two shapes, distinguished by body:
                • default read — base identifiers only (fast context);
                • pack-scoped read — base + context_pack + params (see below).
              Matches the ``GetContextRequest`` schema
              (``additionalProperties: false`` — only the fields below are
              accepted).
    Headers : Authorization: Bearer {token}
              ctx-dataspace-ds_name: {dataspace}
              agent_id: {agent_id}
    Body    : {conversation_id, owner_id, agent_id, owner_type}

  POST /api/v1/agentic/context  (pack-scoped)
    Purpose : Retrieve one named CONTEXT PACK — a server-advertised bundle of
              structured context (identity, enrollment, rules, matched document
              passages, ...) — instead of the default fast/deep read. The pack id
              comes from GET /api/v1/agentic/runtime-context-packs (the
              authoritative catalog of packs this agent may use — see below).
              PREVIEW: the pack-scoped shape is
              not deployed in every environment yet, so it may 404/405 where
              unavailable.
    Headers : Authorization: Bearer {token}
              ctx-dataspace-ds_name: {dataspace}
              agent_id: {agent_id}
    Body    : {conversation_id, owner_id, owner_type, agent_id, context_pack, params}

  GET /api/v1/agentic/runtime-context-packs
    Purpose : List the CONTEXT PACKs this agent is authorized to use.
              Only ACTIVE packs are returned (DRAFT/DISABLED are excluded); an
              unknown agent or one with no packs returns an empty page.
              PREVIEW: this GET op is not deployed in every environment yet.
              May 404 where unavailable or where context packs are disabled on tenant.
    Headers : Authorization: Bearer {token}
              ctx-dataspace-ds_name: {dataspace}
              agent_id: {agent_id}
    Params  : agentApiName, pageSize, page

  GET /api/v1/agentic/conversations/{conversation_id}
    Purpose : Retrieve the full message + trace history for one conversation.
              Use to rehydrate a conversation on load, or to verify what was
              written. Pass scope=all to get messages across all sessions.
    Headers : Authorization: Bearer {token}
              x-sfdc-dataspace: {dataspace}
              agent_id: {agent_id}
    Params  : owner_id, owner_type=individual, scope=all

  GET /api/v1/agentic/conversations
    Purpose : List all conversations for a given owner. Use to populate a
              conversation picker UI or to resume a prior conversation.
    Headers : Authorization: Bearer {token}
              x-sfdc-dataspace: {dataspace}
              agent_id: {agent_id}
    Params  : owner_id, owner_type=individual

  GET /api/v1/agentic/memory
    Purpose : List the stored semantic memory entries (e.g. PROFILE_FACT) for a
              profile. Paginated. Used to enumerate a profile's memories — the
              read half of a "delete all memory for this user" flow (the write
              half is DELETE below). Note: the memory endpoints take profile_id /
              profile_type query params (not owner_id/owner_type) — the SDK
              methods accept owner_id/owner_type and translate on the wire.
    Headers : Authorization: Bearer {token}
              x-sfdc-dataspace: {dataspace}
              agent_id: {agent_id}
    Params  : profile_id, profile_type, category (default PROFILE_FACT),
              page_size, page_token, is_unified, include_metadata

  DELETE /api/v1/agentic/memory
    Purpose : Delete the given memory ids for a profile (write-side endpoint on
              the ingestion-beacon service). The pipeline writes a tombstone
              asynchronously. There is no server-side "delete all" — the caller
              enumerates ids via the GET above and passes them here.
    Headers : Authorization: Bearer {token}
              agent_id: {agent_id}   (optional; resolves dataspace server-side)
    Params  : profile_id, profile_type, category
    Body    : {"memories": [{"memoryId": "..."}, ...]}  (minItems 1)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
import requests

from . import _retry
from .client import IngestionError, _json_or_raise, _sanitize_header

if TYPE_CHECKING:
    from .auth import ContextAuth

logger = logging.getLogger(__name__)


# VERSION guidance block, hoisted so the five adapter tool docstrings and the
# shared CONTEXT_PACK_INSTRUCTION read from a single source. Adapters splice
# this verbatim into their ``make_get_context_pack_tool`` descriptions so a
# future wording change needs exactly one edit instead of six.
CONTEXT_PACK_VERSION_GUIDANCE = """\
VERSION — when a pack entry in the "[Available context packs]" block includes a
`version`, you MUST pass it on every call to `get_context_pack` as
`context_pack_version`. This pins the call to the same revision you planned
against. Only omit `context_pack_version` when the catalog entry has no
`version` field. If the pack has been republished since you first saw the list,
the tool returns status "version_expired" with `latest_version` and a `latest`
block describing the new pack (id, description, version, dataNeeds). Compare
the new dataNeeds to your plan: if they still cover what you need, retry
`get_context_pack` once with `context_pack_version` set to `latest_version`
when it is an integer, or omit `context_pack_version` when `latest_version` is
null and treat the next result as the new baseline. If the new dataNeeds no
longer cover your plan, tell the user which piece is no longer available. Do
NOT loop — retry at most once per turn."""

# get_context_pack(context_pack_id, params) tool built on top of
# :meth:`get_context_pack`. It is the shared source of truth used by every
# adapter's ``make_get_context_pack_tool`` (Claude / Google ADK / LangGraph all
# re-export this exact object) and by raw QUICKSTART users who wire the pack POST
# by hand. Prepend it to your agent's instruction. The advertised packs, their
# data needs, and each need's parameters arrive in the "[Available context packs]"
# block; the model chooses a pack from that block and fills ``params`` from the
# conversation. Kept free of any domain vocabulary so it is reusable across
# agents. Ensure to update this instruction if CONTEXT_PACK_LIST_MARKER changes.
CONTEXT_PACK_INSTRUCTION = """\
CONTEXT PACKS — get_context_pack(context_pack_id, params)
You have a tool `get_context_pack` that retrieves one named context pack for the
current subject. That is the same read as getContext(contextPack).

AVAILABLE PACKS — look for an "[Available context packs]" block elsewhere in
your context. Each pack has a description and a dataNeeds list. Each data need
has a description and a parameters list (name and description). Read those
before you call the tool. If a pack's description, or one of its data needs,
would help answer the user's question, call `get_context_pack` with that pack
id. Pass EXACTLY one id that appears in the block. If the block is absent or
empty, no packs are available — do not guess an id or call the tool
speculatively.

PARAMETERS — before the call, read the parameters on the data needs you need.
Pass their values in `params`, a JSON object whose keys are those parameter
names. Use a value you can determine from the conversation. If you cannot
determine a value, ask the user for it, or omit that parameter. Do not invent a
value, and do not send an empty string. An omitted parameter is not sent. A
data need that declares that parameter may then be absent from the tool result.
Answer from the data needs that did return, and tell the user which part could
not be loaded because a parameter was missing. Do not pass the session owner
unless a listed parameter asks for it.

If the call returns status "invalid_pack", use "valid_pack_ids" and each pack's
description to choose the next pack.

FETCH ONCE, THEN REUSE — a pack's content does not change from one turn to the
next, so once you have pulled a pack in this conversation keep answering from
what it returned. Re-call a pack only when you need a data need the earlier pull
did not return, or when you now have a parameter value that was missing. Combine
every parameter into a SINGLE call per pack per turn.

""" + CONTEXT_PACK_VERSION_GUIDANCE
CONTEXT_PACK_LIST_MARKER = "[Available context packs]"


def version_mismatch_entry(payload: Any) -> dict | None:
    """Recognize Beacon's context-pack version-mismatch envelope.

    Beacon returns a standard ``SessionContextResponse`` with exactly one
    ``userContext`` entry when ``context_pack_version`` on the request does
    not match the pack's current revision: ``sourceType="contextPack"``, and
    ``entry`` carries the latest pack description as TOON. Everything else
    (normal pack payloads, non-``contextPack`` sourceTypes, zero or many
    entries) is NOT a mismatch — return ``None`` so the tool returns
    ``status="ok"`` with the payload as usual.

    Returns ``{"entry": <toon str>, "version": <int or None>}`` on match.
    ``version`` is parsed from the TOON ``version:`` line; when the line is
    absent or unparseable the caller surfaces the raw entry and the model
    reconciles from the text.
    """
    if not isinstance(payload, dict):
        return None
    items = payload.get("userContext")
    if not isinstance(items, list) or len(items) != 1:
        return None
    item = items[0]
    if not isinstance(item, dict) or item.get("sourceType") != "contextPack":
        return None
    entry = item.get("entry")
    version: int | None = None
    if isinstance(entry, str):
        for line in entry.splitlines():
            # ``partition`` is stricter than ``startswith("version:")``: it
            # only matches an exact ``version:`` key, not near-misses like
            # ``versionX:5`` that would otherwise slip through.
            key, sep, val = line.partition(":")
            if sep and key == "version":
                try:
                    version = int(val.strip())
                except ValueError:
                    version = None
                break
    return {"entry": entry, "version": version}


def patch_pack_version_in_catalog(
    catalog_block: str, pack_id: str, new_version: int | None,
) -> str:
    """Re-render an ``[Available context packs]`` block after a version_expired.

    Adapters cache the rendered catalog block in ``_allowed_context_pack_list``.
    When Beacon returns ``version_expired``, the cached block still carries the
    stale ``version`` for that pack, so a second model turn would re-send the
    stale pin and get another expiry — the "stale catalog cliff".

    Only ``version`` is touched. The cached block is shared by every session on
    the adapter, and only the session that hit the expiry received the ``latest``
    description, so ``description``/``dataNeeds`` must stay intact for the
    others. When ``new_version`` is an int, patch it in; when it is None (server
    reported no current revision), drop ``version`` so the retry goes unpinned.

    Returns the input unchanged when the block is missing / malformed / does
    not carry ``pack_id``: a failure to patch should not surface as an error
    from a successful retrieval path.
    """
    marker, sep, blob = catalog_block.partition("\n")
    if not sep or marker != CONTEXT_PACK_LIST_MARKER:
        return catalog_block
    try:
        entries = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return catalog_block
    if not isinstance(entries, dict) or pack_id not in entries:
        return catalog_block
    entry = entries[pack_id]
    if not isinstance(entry, dict):
        return catalog_block
    if isinstance(new_version, int) and not isinstance(new_version, bool):
        entry["version"] = new_version
    else:
        entry.pop("version", None)
    return f"{CONTEXT_PACK_LIST_MARKER}\n{json.dumps(entries)}"


# A pack POST is a DEEP fetch that fans out across multiple slow data sources, so
# it routinely outlasts the retrieval client's fast-read default (10s). Adapters
# give this one call a generous budget (via get_context_pack's ``timeout=``)
# without mutating the shared client timeout. 60s sits well under a typical
# serverless request timeout so a slow fan-out completes.
CONTEXT_PACK_TIMEOUT_S = 60.0


def context_pack_params(raw_params: Any, query: str, defaults: dict[str, Any]) -> dict[str, str]:
    """Build the getContext ``params`` object from a tool call.

    ``raw_params`` is the model's parameter object: a dict, or a JSON object
    string (the Claude tool schema only allows ``str``). Keys are the parameter
    names advertised on the pack's data needs. Blank values are dropped so a
    data need that requires that parameter is omitted by the server. A non-empty
    ``query`` is sent as ``params["query"]`` only when the model did not already
    set that key. ``defaults`` are applied first and the model's values override.

    ``owner_id`` is stripped from the model-supplied payload here — it is the
    authenticated session owner and must never be chosen by the LLM. Callers
    stamp the owner on the outgoing request themselves.
    """
    parsed: dict[str, Any] = {}
    if isinstance(raw_params, dict):
        parsed = raw_params
    elif isinstance(raw_params, str) and raw_params.strip():
        try:
            loaded = json.loads(raw_params)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded

    parsed = {k: v for k, v in parsed.items() if k != "owner_id"}

    params: dict[str, str] = {}
    for key, value in {**defaults, **parsed}.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            params[str(key)] = text
    q = (query or "").strip()
    if q and "query" not in params:
        params["query"] = q
    return params


class RetrievalClient:
    """Sync read-only client for the Context Service (Beacon) REST API.

    Handles all three GET endpoints. Pairs with IngestionClient (sync writes)
    and SessionManager.

    Use as a context manager so the connection pool closes cleanly::

        with RetrievalClient(base_url=..., api_key=..., dataspace=...) as rc:
            ctx = rc.get_context(conversation_id, owner_id, agent_id)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        dataspace: str | None = None,
        timeout: float = 10.0,
        session: requests.Session | None = None,
        auth: "ContextAuth | None" = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.startswith(("https://", "http://")):
            raise ValueError(
                f"base_url must start with 'https://' or 'http://'; got {base_url!r}"
            )
        if normalized.startswith("http://"):
            logger.warning(
                "RetrievalClient base_url uses http:// (not TLS); "
                "credentials will be sent in clear text"
            )
        if dataspace is None:
            logger.warning(
                "RetrievalClient dataspace not specified; falling back to 'default'"
            )
            dataspace = "default"
        self.base_url = normalized
        self.api_key = api_key
        self.dataspace = dataspace
        self.timeout = timeout
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._auth = auth
        # Guards ``self.api_key`` mutation during 401-refresh on the sync path.
        # Sync callers may be on different OS threads (asyncio.to_thread users,
        # manual thread pools); without this, two concurrent refreshes could
        # both call ``_auth.get_token()`` and interleave the attribute write.
        # Not a correctness bug (assignment is atomic) but it prevents wasted
        # token fetches.
        self._refresh_lock = threading.Lock()

    def _refresh_token_if_stale(self, sent_token: str) -> None:
        """Compare-and-refresh helper for sync 401 responses.

        Another thread may have already rotated ``self.api_key`` between our
        send and our 401. Only invalidate + refetch if we still hold the
        token we just sent — without this check N concurrent 401s serialize
        here and each discards a sibling thread's fresh token.
        """
        if self._auth is None:
            return
        with self._refresh_lock:
            if f"Bearer {self.api_key}" == sent_token:
                self._auth.invalidate()
                self.api_key, _ = self._auth.get_token()

    def __enter__(self) -> RetrievalClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def get_context(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/context

        Returns the live context payload for this conversation — in-session
        summary + cross-conversation user facts. Inject the result into the
        LLM system prompt on every user turn so the agent can recall prior
        sessions and preferences without re-asking.

        Returns an empty dict if no context exists yet (e.g. first turn of a
        brand-new conversation) — callers should handle that gracefully.
        """
        return self._send(
            "POST",
            "/api/v1/agentic/context",
            headers=self._context_headers(agent_id),
            json=_context_body(conversation_id, owner_id, agent_id, owner_type),
        )

    def get_context_pack(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        context_pack: str,
        *,
        params: dict[str, Any] | None = None,
        owner_type: str = "individual",
        context_pack_version: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/context — retrieve one named CONTEXT PACK.

        The pack-scoped counterpart to :meth:`get_context`: instead of the
        default fast/deep read, ask the Context Service for one named
        ``context_pack`` (an id advertised by :meth:`get_context_pack_list`,
        the authoritative runtime catalog of packs this agent may use) and
        return its structured payload. ``params`` carries the pack's declared
        data needs (e.g. a free-text ``query`` matched against published
        documents); ``owner_id`` is bound server-side from this argument and
        must NOT be taken from an LLM.

        ``context_pack_version`` — optional pin on the HTTP request. When set,
        the key is included in the POST body and Beacon validates it against
        the pack's current revision; on mismatch Beacon returns a standard
        ``SessionContextResponse`` whose single ``userContext`` entry has
        ``sourceType="contextPack"`` and ``entry`` carrying the latest pack
        description as TOON. When ``None``, the key is dropped from the body
        and Beacon serves whatever revision is live. This client does not
        enforce any LLM-side rule about when the pin must be passed — see
        ``CONTEXT_PACK_INSTRUCTION`` (and ``CONTEXT_PACK_VERSION_GUIDANCE``)
        for the tool contract the adapters surface to the model.

        PREVIEW: the POST operation is not
        deployed in every environment yet, so this may raise ``IngestionError``
        with a 404/405 where it is unavailable. Callers that want to degrade
        gracefully should catch that and fall back to :meth:`get_context`.

        A deep pack read fans out across multiple slow data sources, so it can
        outlast the fast-read ``timeout`` the client was constructed with — pass
        a larger ``timeout`` for this call without disturbing the shared one.
        """
        sent_params = dict(params or {})
        # owner_id is a trust boundary: always bind from the trusted argument,
        # never from caller-supplied params (which may originate in an LLM).
        sent_params["owner_id"] = owner_id
        body: dict[str, Any] = {
            "conversation_id": conversation_id,
            "owner_id": owner_id,
            "owner_type": owner_type,
            "agent_id": agent_id,
            "context_pack": context_pack,
            "params": sent_params,
        }
        if context_pack_version is not None:
            body["context_pack_version"] = context_pack_version
        return self._send(
            "POST",
            "/api/v1/agentic/context",
            headers=self._context_headers(agent_id),
            json=body,
            timeout=timeout,
        )

    def get_context_pack_list(
        self,
        agent_id: str,
        *,
        page_size: int = 10,
        page: int = 1,
        max_pages: int = 100,
    ) -> dict[str, Any]:
        """GET /api/v1/agentic/runtime-context-packs — every authorized pack.

        Takes the agent id (agentApiName) and returns all authorized context pack lists
        that the given agent can utilize. This calls the endpoint to fetch each page from
        the paginated api/v1/agentic/runtime-context-packs endpoint.

        Each pack has contextPackId and description (used to choose a pack), plus
        version, status, dates, and dataNeeds. A query need has sql, dataspaceName,
        and parameters. A builtin need has builtin information.

        Agents that are unknown or have no associated packs will return an empty page.
        """
        packs: list[dict[str, Any]] = []
        page_idx = page
        for _ in range(max_pages):
            result = self._get(
                "/api/v1/agentic/runtime-context-packs",
                params={
                    "agentApiName": agent_id,
                    "pageSize": str(page_size),
                    "page": str(page_idx),
                },
                headers=self._context_headers(agent_id),
            )
            # A 200 with ``{}`` (no agent published any packs) or a last
            # page that omits ``hasMore`` should degrade to an empty list
            # rather than raising KeyError at the caller — the docstring
            # says unknown agents return an empty page.
            packs.extend(result.get("contextPacks") or [])
            if not result.get("hasMore"):
                break
            page_idx += 1
        return {"contextPacks": packs}

    def get_system_prompt(self, prompt_type: str) -> str | None:
        """GET /api/v1/agentic/prompts/system/{prompt_type} — static system prompt.

        Returns the plain-text prompt the server publishes for ``prompt_type``
        (e.g. ``"context_pack_instruction"``), ``None`` when the server
        definitively has no prompt to serve, or raises for a transient
        failure. :class:`SessionPrompt` pins the bundled fallback for the
        session on either a miss or a failure.

        **Definitive miss** — returns ``None``:

          * 404 — prompt not published for this type;
          * 410 — endpoint deprecated/gone.

        (The actual list is ``_CACHEABLE_MISS_STATUSES``; keep this
        docstring in sync with that constant when expanding it.)

        **Transient failure** — raises after the shared retry budget:

          * 400 / 403 / 405 / 415 — treated as transient (misrouted
            traffic, deploy skew, policy propagation);
          * 401 that persists after one refresh — token scope race;
          * 200 with ``text/html`` content type — gateway served a login
            page, not the real prompt;
          * 408 / 429 / 5xx — retried per the shared policy, raises after
            the budget is exhausted;
          * connection resets / DNS blips / timeouts — same.

        A 401 that gets a refresh does NOT consume an attempt from the
        retry budget; it only counts once we've burned the refresh.

        A fresh ``Authorization`` header is composed on every attempt from
        ``self.api_key``, so a token refresh happening between retries (in
        another request or a background refresher) is picked up.

        These prompts are tenant-neutral; auth is enforced upstream at the
        gateway, so the call carries ``Authorization`` only (no dataspace /
        agent_id header).
        """
        path = f"/api/v1/agentic/prompts/system/{quote(prompt_type, safe='')}"
        refreshed = False
        attempt = 0
        error: Exception | None = None
        while attempt <= _retry._MAX_RETRIES:
            headers = self._auth_header()
            try:
                resp = self._session.get(
                    self.base_url + path, headers=headers, timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 — transport failure path
                if _retry.is_retryable(exc) and attempt < _retry._MAX_RETRIES:
                    delay = _retry.backoff(attempt)
                    logger.warning(
                        "system prompt %s transport error (%s); retry %d/%d in %.2fs",
                        path, exc, attempt + 1, _retry._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    error = exc
                    attempt += 1
                    continue
                raise
            if resp.status_code == 401 and self._auth is not None and not refreshed:
                refreshed = True
                self._refresh_token_if_stale(headers.get("Authorization", ""))
                # Don't advance attempt — the refresh gets its own try.
                continue
            ctype_header = resp.headers.get("content-type", "")
            if "charset" not in ctype_header.lower():
                # requests defaults charset-less text/* to ISO-8859-1 (httpx
                # uses UTF-8); prompts are UTF-8, so match the async path.
                resp.encoding = "utf-8"
            outcome = _classify_prompt_response(
                resp.status_code,
                ctype_header,
                resp.text,
                path,
            )
            if outcome.kind == "ok":
                return outcome.text
            if outcome.kind == "miss":
                return None
            # Remaining kind is "transient".
            error = outcome.error
            if _retry.is_retryable(error) and attempt < _retry._MAX_RETRIES:
                delay = _resolve_backoff(attempt, resp.headers)
                logger.warning(
                    "system prompt %s failed (%s); retry %d/%d in %.2fs",
                    path, resp.status_code, attempt + 1, _retry._MAX_RETRIES, delay,
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise error
        assert error is not None
        raise error

    def get_conversation(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        *,
        scope: str = "all",
        owner_type: str = "individual",
        session_id: str | None = None,
        participant_id: str | None = None,
        entry_type: str | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """GET /api/v1/agentic/conversations/{conversation_id}

        Returns the full message + trace history for one conversation.
        scope=all returns entries across all sessions under this conversation.
        scope=TRACES returns trace entries only (tool_call, tool_result,
        agent_reasoning) — note: requires server-side support, may return
        empty on some deployments. Optional filters: ``session_id`` scopes
        to one session, ``participant_id`` to one participant, ``entry_type``
        to one event type (e.g., ``"user_message"``), ``page_size`` paginates.
        """
        params: dict[str, Any] = {
            "owner_id": owner_id,
            "owner_type": owner_type,
            "scope": scope,
        }
        if session_id is not None:
            params["session_id"] = session_id
        if participant_id is not None:
            params["participant_id"] = participant_id
        if entry_type is not None:
            params["type"] = entry_type
        if page_size is not None:
            params["page_size"] = page_size
        return self._get(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}",
            params=params,
            headers=self._read_headers(agent_id),
        )

    def list_conversations(
        self,
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
        status: str | None = None,
        time_range: str | None = None,
        fields: list[str] | None = None,
        participant_id: str | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """GET /api/v1/agentic/conversations

        Lists all conversations for a given owner. Use to populate a
        conversation picker or to resume prior sessions. Optional filters:
        ``status`` (e.g., ``"ACTIVE"``), ``time_range`` (ISO interval
        ``"<start>/<end>"``), ``fields`` (projection list, serialized as
        repeated ``fields=…`` params), ``participant_id``, ``page_size``.

        Normalizes the API response to always return a flat list, regardless
        of whether Beacon wraps it in {"conversations": [...]}, {"data": [...]},
        or returns a bare list.
        """
        params: dict[str, Any] = {
            "owner_id": owner_id,
            "owner_type": owner_type,
        }
        if status is not None:
            params["status"] = status
        if time_range is not None:
            params["time_range"] = time_range
        if fields:
            params["fields"] = fields
        if participant_id is not None:
            params["participant_id"] = participant_id
        if page_size is not None:
            params["page_size"] = page_size
        result = self._get(
            "/api/v1/agentic/conversations",
            params=params,
            headers=self._read_headers(agent_id),
        )
        return _normalize_list_response(result)

    # ── Headers ───────────────────────────────────────────────────────────────

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _context_headers(self, agent_id: str) -> dict[str, str]:
        """Headers for GET/POST /context — uses ctx-dataspace-ds_name.

        Includes ``agent_id`` for parity with every other retrieval + ingestion
        call; Beacon uses it for authz + telemetry even on /context reads.
        """
        return {
            **self._auth_header(),
            "ctx-dataspace-ds_name": _sanitize_header(self.dataspace),
            "agent_id": _sanitize_header(agent_id),
        }

    def _read_headers(self, agent_id: str) -> dict[str, str]:
        """Headers for GET /conversations — uses x-sfdc-dataspace."""
        return {
            **self._auth_header(),
            "x-sfdc-dataspace": _sanitize_header(self.dataspace),
            "agent_id": _sanitize_header(agent_id),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(
        self,
        path: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        # Retry transient failures (5xx / connection / timeout) with the shared
        # backoff policy — same policy the ingestion + auth paths use. The read
        # path (GET /agentic/context, i.e. Fast Context) previously had none, so a
        # single transient 500 from the Context Service surfaced as "no context"
        # (a blank, non-personalised greeting). 401 still triggers one in-loop
        # re-auth (not counted as a retry attempt); non-retryable 4xx raise at once.
        for attempt in range(_retry._MAX_RETRIES + 1):
            resp = self._session.get(
                self.base_url + path,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 401 and self._auth is not None:
                self._refresh_token_if_stale(headers.get("Authorization", ""))
                headers = {**headers, "Authorization": f"Bearer {self.api_key}"}
                resp = self._session.get(
                    self.base_url + path,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            if resp.ok:
                return _json_or_raise(
                    resp.status_code, resp.text, resp.json,
                    resp.headers.get("content-type", ""),
                )
            error = IngestionError(resp.status_code, resp.text)
            if _retry.is_retryable(error) and attempt < _retry._MAX_RETRIES:
                delay = _resolve_backoff(attempt, resp.headers)
                logger.warning(
                    "context read %s failed (%s); retry %d/%d in %.2fs",
                    path, resp.status_code, attempt + 1, _retry._MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            raise error
        raise error  # pragma: no cover - loop always returns or raises above

    def _send(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Method-generic sibling of :meth:`_get` (any verb, optional JSON body).

        Same transient-retry + single 401 re-auth policy as ``_get``. ``timeout``
        overrides the client's default for this one call (the pack POST needs a
        longer budget than the fast read); ``None`` uses ``self.timeout``.
        """
        eff_timeout = timeout if timeout is not None else self.timeout
        error: IngestionError | None = None
        for attempt in range(_retry._MAX_RETRIES + 1):
            resp = self._session.request(
                method,
                self.base_url + path,
                params=params,
                headers=headers,
                json=json,
                timeout=eff_timeout,
            )
            if resp.status_code == 401 and self._auth is not None:
                self._refresh_token_if_stale(headers.get("Authorization", ""))
                headers = {**headers, "Authorization": f"Bearer {self.api_key}"}
                resp = self._session.request(
                    method,
                    self.base_url + path,
                    params=params,
                    headers=headers,
                    json=json,
                    timeout=eff_timeout,
                )
            if resp.ok:
                return _json_or_raise(
                    resp.status_code, resp.text, resp.json,
                    resp.headers.get("content-type", ""),
                )
            error = IngestionError(resp.status_code, resp.text)
            if _retry.is_retryable(error) and attempt < _retry._MAX_RETRIES:
                delay = _resolve_backoff(attempt, resp.headers)
                logger.warning(
                    "context %s %s failed (%s); retry %d/%d in %.2fs",
                    method, path, resp.status_code, attempt + 1, _retry._MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            raise error
        raise error  # pragma: no cover - loop always returns or raises above


class AsyncRetrievalClient:
    """Async read-only client for the Context Service (Beacon) REST API.

    Same three GET endpoints as RetrievalClient — only the I/O is non-blocking
    (httpx). Use inside an event loop. Pairs with AsyncIngestionClient and
    AsyncSessionManager::

        async with AsyncRetrievalClient(base_url=..., api_key=...) as rc:
            ctx = await rc.get_context(conversation_id, owner_id, agent_id)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        dataspace: str | None = None,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
        auth: "ContextAuth | None" = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.startswith(("https://", "http://")):
            raise ValueError(
                f"base_url must start with 'https://' or 'http://'; got {base_url!r}"
            )
        if normalized.startswith("http://"):
            logger.warning(
                "AsyncRetrievalClient base_url uses http:// (not TLS); "
                "credentials will be sent in clear text"
            )
        if dataspace is None:
            logger.warning(
                "AsyncRetrievalClient dataspace not specified; falling back to 'default'"
            )
            dataspace = "default"
        self.base_url = normalized
        self.api_key = api_key
        self.dataspace = dataspace
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._auth = auth
        # Guards ``self.api_key`` mutation during 401-refresh on the async
        # path. Mirrors the sync client's ``_refresh_lock`` so concurrent
        # 401s compare-and-swap rather than each discarding a sibling
        # task's fresh token. Created lazily on first contention so it
        # binds to whichever event loop is actually running.
        self._refresh_lock: asyncio.Lock | None = None

    def _get_or_make_refresh_lock(self) -> asyncio.Lock:
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    async def _refresh_token_if_stale(self, sent_token: str) -> None:
        """Async twin of :meth:`RetrievalClient._refresh_token_if_stale`.

        Another task may have already rotated ``self.api_key`` between our
        send and our 401; only invalidate + refetch if we still hold the
        token we just sent.
        """
        if self._auth is None:
            return
        async with self._get_or_make_refresh_lock():
            if f"Bearer {self.api_key}" == sent_token:
                self._auth.invalidate()
                self.api_key, _ = await self._auth.aget_token()

    async def __aenter__(self) -> AsyncRetrievalClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── Endpoints ─────────────────────────────────────────────────────────────

    async def get_context(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/context

        Returns the live context payload for this conversation — in-session
        summary + cross-conversation user facts. Inject the result into the
        LLM system prompt on every user turn so the agent can recall prior
        sessions and preferences without re-asking.

        Returns an empty dict if no context exists yet (e.g. first turn of a
        brand-new conversation) — callers should handle that gracefully.
        """
        return await self._send(
            "POST",
            "/api/v1/agentic/context",
            headers=self._context_headers(agent_id),
            json=_context_body(conversation_id, owner_id, agent_id, owner_type),
        )

    async def get_context_pack_list(
        self,
        agent_id: str,
        *,
        page_size: int = 10,
        page: int = 1,
        max_pages: int = 100,
    ) -> dict[str, Any]:
        """GET /api/v1/agentic/runtime-context-packs

        Takes the agent id (agentApiName) and returns all authorized context pack lists
        that the given agent can utilize. This calls the endpoint to fetch each page from
        the paginated api/v1/agentic/runtime-context-packs endpoint.

        Each pack has contextPackId and description (used to choose a pack), plus
        version, status, dates, and dataNeeds. A query need has sql, dataspaceName,
        and parameters. A builtin need has builtin information.

        Agents that are unknown or have no associated packs will return an empty page.
        """
        packs: list[dict[str, Any]] = []
        page_idx = page
        for _ in range(max_pages):
            result = await self._get(
                "/api/v1/agentic/runtime-context-packs",
                params={
                    "agentApiName": agent_id,
                    "pageSize": str(page_size),
                    "page": str(page_idx),
                },
                headers=self._context_headers(agent_id),
            )
            # A 200 with ``{}`` (no agent published any packs) or a last
            # page that omits ``hasMore`` should degrade to an empty list
            # rather than raising KeyError at the caller — the docstring
            # says unknown agents return an empty page.
            packs.extend(result.get("contextPacks") or [])
            if not result.get("hasMore"):
                break
            page_idx += 1
        return {"contextPacks": packs}

    async def get_system_prompt(self, prompt_type: str) -> str | None:
        """GET /api/v1/agentic/prompts/system/{prompt_type} — static system prompt.

        Async twin of :meth:`RetrievalClient.get_system_prompt`. See that
        method for the full cacheability contract (``None`` = cacheable
        miss, raise = transient).
        """
        path = f"/api/v1/agentic/prompts/system/{quote(prompt_type, safe='')}"
        refreshed = False
        attempt = 0
        error: Exception | None = None
        while attempt <= _retry._MAX_RETRIES:
            headers = self._auth_header()
            try:
                resp = await self._client.get(
                    self.base_url + path, headers=headers, timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 — transport failure path
                # httpx.ReadError / RemoteProtocolError (keep-alive resets) are
                # not OSErrors, unlike their requests twins; this GET is
                # idempotent, so retry every transport error like the sync path.
                retryable = _retry.is_retryable(exc) or isinstance(exc, httpx.TransportError)
                if retryable and attempt < _retry._MAX_RETRIES:
                    delay = _retry.backoff(attempt)
                    logger.warning(
                        "system prompt %s transport error (%s); retry %d/%d in %.2fs",
                        path, exc, attempt + 1, _retry._MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    error = exc
                    attempt += 1
                    continue
                raise
            if resp.status_code == 401 and self._auth is not None and not refreshed:
                refreshed = True
                await self._refresh_token_if_stale(headers.get("Authorization", ""))
                # Don't advance attempt — the refresh gets its own try.
                continue
            outcome = _classify_prompt_response(
                resp.status_code,
                resp.headers.get("content-type", ""),
                resp.text,
                path,
            )
            if outcome.kind == "ok":
                return outcome.text
            if outcome.kind == "miss":
                return None
            # Remaining kind is "transient".
            error = outcome.error
            if _retry.is_retryable(error) and attempt < _retry._MAX_RETRIES:
                delay = _resolve_backoff(attempt, resp.headers)
                logger.warning(
                    "system prompt %s failed (%s); retry %d/%d in %.2fs",
                    path, resp.status_code, attempt + 1, _retry._MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            raise error
        assert error is not None
        raise error

    async def get_conversation(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        *,
        scope: str = "all",
        owner_type: str = "individual",
        session_id: str | None = None,
        participant_id: str | None = None,
        entry_type: str | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """GET /api/v1/agentic/conversations/{conversation_id}

        Returns the full message + trace history for one conversation.
        scope=all returns entries across all sessions under this conversation.
        scope=TRACES returns trace entries only (tool_call, tool_result,
        agent_reasoning) — note: requires server-side support, may return
        empty on some deployments. Optional filters: ``session_id`` scopes
        to one session, ``participant_id`` to one participant, ``entry_type``
        to one event type (e.g., ``"user_message"``), ``page_size`` paginates.
        """
        params: dict[str, Any] = {
            "owner_id": owner_id,
            "owner_type": owner_type,
            "scope": scope,
        }
        if session_id is not None:
            params["session_id"] = session_id
        if participant_id is not None:
            params["participant_id"] = participant_id
        if entry_type is not None:
            params["type"] = entry_type
        if page_size is not None:
            params["page_size"] = page_size
        return await self._get(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}",
            params=params,
            headers=self._read_headers(agent_id),
        )

    async def list_conversations(
        self,
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
        status: str | None = None,
        time_range: str | None = None,
        fields: list[str] | None = None,
        participant_id: str | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """GET /api/v1/agentic/conversations

        Lists all conversations for a given owner. Use to populate a
        conversation picker or to resume prior sessions. Optional filters:
        ``status`` (e.g., ``"ACTIVE"``), ``time_range`` (ISO interval
        ``"<start>/<end>"``), ``fields`` (projection list, serialized as
        repeated ``fields=…`` params), ``participant_id``, ``page_size``.

        Normalizes the API response to always return a flat list, regardless
        of whether Beacon wraps it in {"conversations": [...]}, {"data": [...]},
        or returns a bare list.
        """
        params: dict[str, Any] = {
            "owner_id": owner_id,
            "owner_type": owner_type,
        }
        if status is not None:
            params["status"] = status
        if time_range is not None:
            params["time_range"] = time_range
        if fields:
            params["fields"] = fields
        if participant_id is not None:
            params["participant_id"] = participant_id
        if page_size is not None:
            params["page_size"] = page_size
        result = await self._get(
            "/api/v1/agentic/conversations",
            params=params,
            headers=self._read_headers(agent_id),
        )
        return _normalize_list_response(result)

    async def list_memory(
        self,
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
        category: str = "PROFILE_FACT",
        is_unified: bool = False,
        include_metadata: bool = False,
        page_size: int = 200,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        """GET /api/v1/agentic/memory — all memory entries for a profile+category.

        Walks every page (following ``nextPageToken``) and returns a flat list of
        entries, each ``{"memoryId": str, "content": str, "metadata"?: {...}}``.
        This is the enumeration half of a "forget me" flow: list the ids here,
        then pass them to :meth:`delete_memory`.

        ``owner_id`` / ``owner_type`` are the SDK's owner vocabulary; on the wire
        they map to the endpoint's ``profile_id`` / ``profile_type`` params
        (``individual`` → Individual id, ``employee`` → User id). ``category``
        defaults to ``PROFILE_FACT`` (the default semantic-memory category).
        ``max_pages`` is a safety bound against a mis-paginating server.
        """
        memories: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(max_pages):
            params: dict[str, str] = {
                "profile_id": owner_id,
                "profile_type": owner_type,
                "category": category,
                "is_unified": "true" if is_unified else "false",
                "include_metadata": "true" if include_metadata else "false",
                "page_size": str(page_size),
            }
            if page_token:
                params["page_token"] = page_token
            result = await self._send(
                "GET",
                "/api/v1/agentic/memory",
                params=params,
                headers=self._read_headers(agent_id),
            )
            if isinstance(result, dict):
                memories.extend(result.get("memories") or [])
                page_token = result.get("nextPageToken") or None
            else:
                memories.extend(_normalize_list_response(result))
                page_token = None
            if not page_token:
                break
        return memories

    async def delete_memory(
        self,
        memory_ids: list[str],
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
        category: str = "PROFILE_FACT",
    ) -> dict[str, Any]:
        """DELETE /api/v1/agentic/memory — delete the given ids for a profile.

        One HTTP DELETE per call. The request body carries the explicit ids
        (``{"memories": [{"memoryId": ...}]}``) — there is no server-side wildcard
        delete, so callers enumerate via :meth:`list_memory` first. An empty
        ``memory_ids`` is a no-op (the endpoint requires ``minItems: 1``), so the
        caller should skip the call rather than send an empty body. Chunk large id
        sets across multiple calls at the call site.

        Returns the write response ``{"statusCode": "deleted", "memories": [...]}``.
        """
        if not memory_ids:
            return {"statusCode": "deleted", "memories": []}
        return await self._send(
            "DELETE",
            "/api/v1/agentic/memory",
            params={
                "profile_id": owner_id,
                "profile_type": owner_type,
                "category": category,
            },
            headers={**self._auth_header(), "agent_id": _sanitize_header(agent_id)},
            json={"memories": [{"memoryId": mid} for mid in memory_ids]},
        )

    async def get_context_pack(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        context_pack: str,
        *,
        params: dict[str, Any] | None = None,
        owner_type: str = "individual",
        context_pack_version: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/context — retrieve one named CONTEXT PACK.

        The pack-scoped counterpart to :meth:`get_context`: instead of the
        default fast/deep read, ask the Context Service for one named
        ``context_pack`` (an id advertised by :meth:`get_context_pack_list`,
        the authoritative runtime catalog of packs this agent may use) and
        return its structured payload. ``params`` carries the pack's declared
        data needs (e.g. a free-text ``query`` matched against published
        documents); ``owner_id`` is bound server-side from this argument and
        must NOT be taken from an LLM.

        ``context_pack_version`` — optional pin on the HTTP request. When set,
        the key is included in the POST body and Beacon validates it against
        the pack's current revision; on mismatch Beacon returns a standard
        ``SessionContextResponse`` whose single ``userContext`` entry has
        ``sourceType="contextPack"`` and ``entry`` carrying the latest pack
        description as TOON. When ``None``, the key is dropped from the body
        and Beacon serves whatever revision is live. This client does not
        enforce any LLM-side rule about when the pin must be passed — see
        ``CONTEXT_PACK_INSTRUCTION`` (and ``CONTEXT_PACK_VERSION_GUIDANCE``)
        for the tool contract the adapters surface to the model.

        PREVIEW: the POST operation is not
        deployed in every environment yet, so this may raise ``IngestionError``
        with a 404/405 where it is unavailable. Callers that want to degrade
        gracefully should catch that and fall back to :meth:`get_context`.

        A deep pack read fans out across multiple slow data sources, so it can
        outlast the fast-read ``timeout`` the client was constructed with — pass
        a larger ``timeout`` for this call without disturbing the shared one.
        """
        sent_params = dict(params or {})
        # owner_id is a trust boundary: always bind from the trusted argument,
        # never from caller-supplied params (which may originate in an LLM).
        sent_params["owner_id"] = owner_id
        body: dict[str, Any] = {
            "conversation_id": conversation_id,
            "owner_id": owner_id,
            "owner_type": owner_type,
            "agent_id": agent_id,
            "context_pack": context_pack,
            "params": sent_params,
        }
        if context_pack_version is not None:
            body["context_pack_version"] = context_pack_version
        return await self._send(
            "POST",
            "/api/v1/agentic/context",
            headers=self._context_headers(agent_id),
            json=body,
            timeout=timeout,
        )

    # ── Headers ───────────────────────────────────────────────────────────────

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _context_headers(self, agent_id: str) -> dict[str, str]:
        """Headers for GET/POST /context — uses ctx-dataspace-ds_name.

        Includes ``agent_id`` for parity with every other retrieval + ingestion
        call; Beacon uses it for authz + telemetry even on /context reads.
        """
        return {
            **self._auth_header(),
            "ctx-dataspace-ds_name": _sanitize_header(self.dataspace),
            "agent_id": _sanitize_header(agent_id),
        }

    def _read_headers(self, agent_id: str) -> dict[str, str]:
        """Headers for GET /conversations — uses x-sfdc-dataspace."""
        return {
            **self._auth_header(),
            "x-sfdc-dataspace": _sanitize_header(self.dataspace),
            "agent_id": _sanitize_header(agent_id),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        # Retry transient failures (5xx / connection / timeout) with the shared
        # backoff policy — same policy the ingestion + auth paths use. This is the
        # path the ADK adapter's before-model Fast Context read goes through
        # (wrap_before_model → get_context); previously a single transient 500
        # from the Context Service surfaced as "no context" (a blank, non-
        # personalised greeting). 401 still triggers one in-loop re-auth (not
        # counted as a retry attempt); non-retryable 4xx raise at once.
        for attempt in range(_retry._MAX_RETRIES + 1):
            resp = await self._client.get(
                self.base_url + path,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 401 and self._auth is not None:
                await self._refresh_token_if_stale(headers.get("Authorization", ""))
                headers = {**headers, "Authorization": f"Bearer {self.api_key}"}
                resp = await self._client.get(
                    self.base_url + path,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            if resp.status_code < 400:
                return _json_or_raise(
                    resp.status_code, resp.text, resp.json,
                    resp.headers.get("content-type", ""),
                )
            error = IngestionError(resp.status_code, resp.text)
            if _retry.is_retryable(error) and attempt < _retry._MAX_RETRIES:
                delay = _resolve_backoff(attempt, resp.headers)
                logger.warning(
                    "context read %s failed (%s); retry %d/%d in %.2fs",
                    path, resp.status_code, attempt + 1, _retry._MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise error
        raise error  # pragma: no cover - loop always returns or raises above

    async def _send(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Method-generic sibling of :meth:`_get` for the memory + pack endpoints.

        Same transient-retry + single 401 re-auth policy as ``_get`` (GET reads),
        extended to any verb with an optional JSON body so the memory list (GET),
        delete (DELETE), and pack read (POST) share one code path. ``timeout``
        overrides the client's default for this one call (the pack POST needs a
        longer budget than the fast read); ``None`` uses ``self.timeout``. 401
        re-auth is not counted as a retry attempt; non-retryable 4xx raise at once.
        """
        eff_timeout = timeout if timeout is not None else self.timeout
        error: IngestionError | None = None
        for attempt in range(_retry._MAX_RETRIES + 1):
            resp = await self._client.request(
                method,
                self.base_url + path,
                params=params,
                headers=headers,
                json=json,
                timeout=eff_timeout,
            )
            if resp.status_code == 401 and self._auth is not None:
                await self._refresh_token_if_stale(headers.get("Authorization", ""))
                headers = {**headers, "Authorization": f"Bearer {self.api_key}"}
                resp = await self._client.request(
                    method,
                    self.base_url + path,
                    params=params,
                    headers=headers,
                    json=json,
                    timeout=eff_timeout,
                )
            if resp.status_code < 400:
                return _json_or_raise(
                    resp.status_code, resp.text, resp.json,
                    resp.headers.get("content-type", ""),
                )
            error = IngestionError(resp.status_code, resp.text)
            if _retry.is_retryable(error) and attempt < _retry._MAX_RETRIES:
                delay = _resolve_backoff(attempt, resp.headers)
                logger.warning(
                    "context %s %s failed (%s); retry %d/%d in %.2fs",
                    method, path, resp.status_code, attempt + 1, _retry._MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise error
        raise error  # pragma: no cover - loop always returns or raises above


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _context_body(
    conversation_id: str,
    owner_id: str,
    agent_id: str,
    owner_type: str,
) -> dict[str, Any]:
    """Build the POST /context JSON body, shared by the sync + async clients.

    Matches the ``GetContextRequest`` schema. The
    schema sets ``additionalProperties: false`` so only the four identifier
    fields may ride on the wire today; the backend currently serves FAST
    context only (deep mode / context packs land in later server PRs).
    """
    return {
        "conversation_id": conversation_id,
        "owner_id": owner_id,
        "agent_id": agent_id,
        "owner_type": owner_type,
    }


def _normalize_list_response(result: dict[str, Any] | list | None) -> list[dict[str, Any]]:
    """Normalize various Beacon response shapes to a flat list.

    The list_conversations API may return:
    - A bare list: [{"conversationId": ...}, ...]
    - Wrapped in "conversations": {"conversations": [...]}
    - Wrapped in "data": {"data": [...]}
    - Wrapped in "items": {"items": [...]}
    - Empty dict or None

    This normalizes all shapes so callers always get list[dict].
    """
    if result is None or result == {}:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("conversations", "data", "items", "results"):
            if key in result and isinstance(result[key], list):
                return result[key]
    return []


# ─── Server-side system-prompt resolution ──────────────────────
#
# The server publishes shared, tenant-neutral system prompts via
# GET /api/v1/agentic/prompts/system/{prompt_type}; the SDK prefers the
# server copy when available and falls back to the bundled client-side
# constant otherwise. Each adapter holds one :class:`SessionPrompt`, fetches
# on a session's first turn and injects the pinned text next to the pack
# catalog on every turn — customers don't call anything.

SYSTEM_PROMPT_TYPE_CONTEXT_PACK = "context_pack_instruction"

# Status codes that signal a stable "the server has no such prompt right now"
# answer: the server will keep saying the same thing until a redeploy. Only
# 404 (prompt not published) and 410 (endpoint deprecated) are definitive —
# 400/405/415 can originate at a gateway/WAF (malformed proxy request,
# header-size error) or during a rolling deploy and will self-heal, so we
# treat them as transient (retried, then raised). Anything else (401,
# 403, 429, 5xx, HTML-200, 204/empty-200) is either transient or an auth race
# we re-try.
_CACHEABLE_MISS_STATUSES = frozenset({404, 410})


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an integer ``Retry-After`` header into non-negative seconds.

    HTTP-date form (RFC 9110) is intentionally not supported — it is
    very rare in practice and a pathological value would fall through
    to the normal exponential backoff, which is the right behaviour.
    """
    if not value:
        return None
    try:
        secs = int(value.strip())
    except ValueError:
        return None
    return max(secs, 0.0)


def _resolve_backoff(attempt: int, headers: Any) -> float:
    """Pick the retry delay: ``Retry-After`` if the server sent one, else backoff.

    When the server advertises ``Retry-After`` (RFC 9110 §10.2.3 / §15.5.4)
    we honour it, clamped to the SDK's absolute 30 s backoff ceiling so a
    mis-set 24-hour ``Retry-After`` can't stall a short-lived process.
    Falling back to ``_retry.backoff(attempt)`` here would be wrong — that
    is a random draw from ``uniform(0, cap)``, so ``min(retry_after, draw)``
    would ignore the server's hint in almost every case.

    When no ``Retry-After`` is present, falls back to the shared
    exponential-with-full-jitter policy.
    """
    try:
        retry_after = _parse_retry_after(headers.get("Retry-After"))
    except AttributeError:
        retry_after = None
    if retry_after is not None:
        return min(retry_after, _retry._BACKOFF_MAX)
    return _retry.backoff(attempt)


class _PromptOutcome:
    """Tag + payload for ``_classify_prompt_response``.

    Three possible ``kind``s:
      * ``"ok"`` — 200 with a usable body; ``text`` is the prompt.
      * ``"miss"`` — server definitively has no prompt (see
        ``_CACHEABLE_MISS_STATUSES``).
      * ``"transient"`` — may be retryable (per :mod:`_retry`) or raised.
    """

    __slots__ = ("kind", "text", "error")

    def __init__(
        self,
        kind: str,
        *,
        text: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.kind = kind
        self.text = text
        self.error = error


def _classify_prompt_response(
    status_code: int,
    content_type: str,
    body: str,
    path: str,
) -> _PromptOutcome:
    """Classify a ``GET /prompts/system/{type}`` response.

    Centralises the 200/HTML/4xx/5xx policy so sync and async paths stay
    in lockstep. The caller handles the retry/refresh dance around it.
    """
    if status_code == 401:
        # A 401 reaching this classifier means refresh has already been
        # consumed (or no auth was attached). 401 is not retryable, so this
        # raises; SessionPrompt pins the fallback for the session.
        return _PromptOutcome(
            "transient",
            error=IngestionError(status_code, body),
        )
    if status_code == 403:
        # Token/scope races happen during policy propagation; don't pin.
        return _PromptOutcome(
            "transient",
            error=IngestionError(status_code, body),
        )
    if status_code in _CACHEABLE_MISS_STATUSES:
        return _PromptOutcome("miss")
    if 200 <= status_code < 300:
        ctype = (content_type or "").lower().split(";", 1)[0].strip()
        # Only reject the one shape that is reliably NOT a prompt: a
        # gateway/login HTML page masquerading as a 200. ``text/plain``,
        # ``application/octet-stream`` (proxy strips content-type), and an
        # absent content-type pass through verbatim — the server returns the
        # raw prompt as ``text/plain``.
        if ctype.startswith("text/html"):
            logger.warning(
                "system prompt %s returned %d with content-type %r — "
                "treating as transient (gateway login page?)",
                path, status_code, ctype,
            )
            return _PromptOutcome(
                "transient",
                error=IngestionError(
                    status_code,
                    f"non-prompt response (content-type={ctype!r})",
                ),
            )
        raw = (body or "").strip()
        if not raw:
            # 200 + empty body = nothing to inject this turn. Not cached.
            return _PromptOutcome(
                "transient",
                error=IngestionError(status_code, "empty response body"),
            )
        return _PromptOutcome("ok", text=raw)
    # Any other 4xx / 5xx: let the retry layer decide retryability via
    # is_retryable(IngestionError(...)).
    return _PromptOutcome("transient", error=IngestionError(status_code, body))


def _is_coroutine_function(fn: Any) -> bool:
    """Return True if ``fn`` is (or wraps) an ``async def``.

    ``asyncio.iscoroutinefunction`` returns ``False`` for ``functools.partial``
    and most decorator wrappers, so a wrapped async client's
    ``get_system_prompt`` gets misrouted onto the sync/thread dispatch path.
    ``inspect.iscoroutinefunction`` handles ``partial`` on 3.12+ and anything
    exposing ``__wrapped__`` via ``functools.wraps`` — walk a few layers
    defensively so decorator chains and Mock wrappers resolve correctly.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    current = fn
    for _ in range(5):
        inner = getattr(current, "__wrapped__", None) or getattr(current, "func", None)
        if inner is None or inner is current:
            return False
        if inspect.iscoroutinefunction(inner):
            return True
        current = inner
    return False


class SessionPrompt:
    """Server-published system prompt, fetched once per session and pinned.

    The first lookup for a ``session_id`` calls
    ``client.get_system_prompt(prompt_type)`` and pins the answer for that
    session, so the prompt never changes mid-conversation. A miss
    (``None``) or any fetch error pins ``fallback`` instead — no retry
    within the session; a later session tries again once the failure
    cooldown below has passed. With no client
    (built lazily by some adapters) the fallback is returned unpinned so
    a later turn can still fetch.

    After a fetch error, new sessions skip the fetch and pin ``fallback``
    for ``FAILURE_COOLDOWN_S`` so a Beacon outage doesn't make every new
    session's first turn wait out the full retry budget.

    At most ``max_sessions`` sessions are remembered; the least recently
    used is evicted first and simply re-fetches if it is seen again.
    """

    DEFAULT_MAX_SESSIONS = 10_000
    FAILURE_COOLDOWN_S = 10.0

    def __init__(
        self,
        fallback: str,
        prompt_type: str = SYSTEM_PROMPT_TYPE_CONTEXT_PACK,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self._fallback = fallback
        self._prompt_type = prompt_type
        self._max_sessions = max_sessions
        self._by_session: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._failed_at: float | None = None

    def present_in(self, prompt: str | None, text: str) -> bool:
        """True if ``text`` (or the bundled fallback) is already in ``prompt``.

        Lets an adapter skip injection when the customer already put the
        instruction in their own prompt.
        """
        return bool(prompt) and (text in prompt or self._fallback in prompt)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._by_session.pop(session_id, None)

    def clear(self) -> None:
        """Forget every session (adapter shutdown)."""
        with self._lock:
            self._by_session.clear()

    def _pinned(self, session_id: str) -> str | None:
        with self._lock:
            text = self._by_session.get(session_id)
            if text is not None:
                self._by_session.move_to_end(session_id)
            return text

    def _pin(self, session_id: str, text: Any) -> str:
        if not isinstance(text, str) or not text:
            text = self._fallback
        with self._lock:
            # A concurrent first turn of the same session may have pinned
            # already; keep that answer so the session stays consistent.
            existing = self._by_session.get(session_id)
            if existing is not None:
                return existing
            self._by_session[session_id] = text
            while len(self._by_session) > self._max_sessions:
                self._by_session.popitem(last=False)
        return text

    def _cooling_down(self) -> bool:
        failed_at = self._failed_at
        return (
            failed_at is not None
            and time.monotonic() - failed_at < self.FAILURE_COOLDOWN_S
        )

    def _on_failure(self, session_id: str, exc: Exception) -> str:
        self._failed_at = time.monotonic()
        logger.warning(
            "server-side %s fetch failed; using bundled fallback for session %s: %s",
            self._prompt_type, session_id[:8], exc,
        )
        return self._pin(session_id, None)

    async def get(self, session_id: str, client: Any) -> str:
        """Return the session's prompt, fetching on its first turn."""
        pinned = self._pinned(session_id)
        if pinned is not None:
            return pinned
        fn = getattr(client, "get_system_prompt", None) if client is not None else None
        if fn is None:
            return self._fallback
        if self._cooling_down():
            return self._pin(session_id, None)
        try:
            if _is_coroutine_function(fn):
                text = await fn(self._prompt_type)
            else:
                text = await asyncio.to_thread(fn, self._prompt_type)
                # ``_is_coroutine_function`` can miss deep wrapper chains;
                # never pin an awaitable as the prompt string.
                if inspect.isawaitable(text):
                    text = await text
        except Exception as exc:  # noqa: BLE001 — degrade, don't fail
            return self._on_failure(session_id, exc)
        return self._pin(session_id, text)

    def get_sync(self, session_id: str, client: Any) -> str:
        """Blocking twin of :meth:`get` for sync clients."""
        pinned = self._pinned(session_id)
        if pinned is not None:
            return pinned
        fn = getattr(client, "get_system_prompt", None) if client is not None else None
        if fn is None:
            return self._fallback
        if self._cooling_down():
            return self._pin(session_id, None)
        try:
            text = fn(self._prompt_type)
        except Exception as exc:  # noqa: BLE001 — degrade, don't fail
            return self._on_failure(session_id, exc)
        if inspect.isawaitable(text):
            text.close()  # type: ignore[union-attr]
            raise RuntimeError(
                "SessionPrompt.get_sync() was given an async client; "
                "use `await session_prompt.get(...)`."
            )
        return self._pin(session_id, text)
