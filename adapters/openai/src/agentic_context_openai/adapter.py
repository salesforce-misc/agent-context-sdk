# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""OpenAI adapter for agent-context-sdk.

Single adapter instance per agent, manages N sessions internally. Lifecycle is
lazy (triggered on first ingest). Mirrors ClaudeAdapter / GoogleAdkAdapter /
LangGraphAdapter so a team familiar with any of those can read this in minutes.

Primary runtime target: the OpenAI Agents SDK (``openai-agents``), the direct
analog of claude-agent-sdk / google-adk / langgraph. The adapter also accepts a
plain Chat Completions messages list — ``openai.chat.completions.create(messages
=...)`` — for the simple snapshot path, so teams that only use the raw API can
still ship transcripts without adopting the Agents SDK.

Two normalization inputs — both route through the same ``RecordTransformer``:

1. **Chat Completions messages** (dicts with ``role`` + ``content``):

       messages = [
           {"role": "user", "content": "What's the weather?"},
           {"role": "assistant", "tool_calls": [{"id": "call_1",
               "function": {"name": "get_weather", "arguments": "{\\"city\\":\\"SF\\"}"}}]},
           {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
           {"role": "assistant", "content": "It's 72F in SF."},
       ]
       records = adapter.snapshot(session_id, messages=messages)

2. **Responses API items** (``RunResult.to_input_list()`` from the Agents SDK):

       result = await Runner.run(agent, "What's the weather?")
       records = adapter.snapshot(session_id, items=result.to_input_list())

Usage (credentials — recommended; SDK manages token exchange + refresh)::

    adapter = OpenAIAdapter(
        agent_id="my-agent",
        owner_id="user@corp.com",
        private_key=Path("server.key").read_text(),
        consumer_key="3MVG...",
        subject="user@org.com",
        login_url="login.salesforce.com",
    )
    await adapter.push(session_id, messages=my_messages)
    # … end of session …
    await adapter.close()
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import warnings
from typing import Any

from agentic_context import (
    CONTEXT_PACK_INSTRUCTION,
    CONTEXT_PACK_LIST_MARKER,
    CONTEXT_PACK_TIMEOUT_S,
    ApiKeyAuth,
    AsyncIngestionClient,
    AsyncSessionManager,
    Capabilities,
    ContextAuth,
    IngestionClient,
    Record,
    SessionManager,
)
from agentic_context.retrieval import (
    CONTEXT_PACK_VERSION_GUIDANCE,
    AsyncRetrievalClient,
    SessionPrompt,
    RetrievalClient,
    context_pack_params,
    patch_pack_version_in_catalog,
    version_mismatch_entry,
)

from .transformer import RecordTransformer

logger = logging.getLogger(__name__)

# The OpenAI Agents SDK (`pip install openai-agents`) is the primary runtime
# this adapter targets, but it is NOT a hard dependency: the pure-ingest path
# (push/snapshot/ingest with a messages list) works without it. The on-demand
# recall tool (make_context_retrieval_tool) is the one entry point that needs
# the SDK, so it imports lazily — mirroring ClaudeAdapter._require_sdk.
_SDK_MODULE = "agents"
_SDK_INSTALL_HINT = "pip install openai-agents"


def _require_sdk(method: str) -> None:
    """Fail with a single actionable message if ``openai-agents`` is absent."""
    try:
        importlib.import_module(_SDK_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"OpenAIAdapter.{method}() requires the OpenAI Agents SDK, which is "
            f"not installed. Install it with:\n\n    {_SDK_INSTALL_HINT}\n"
        ) from exc


# Beacon ``AiAgentParticipant`` label for this runtime — picking "OpenAI" so
# analytics bucket this adapter's traffic apart from the default "Custom".
_AGENT_PARTICIPANT_VARIABLE = {
    "AiAgentType": "OpenAI",
    "AiAgentVersionApiName": "v1",
    "AiAgentTemplateApiName": "default",
}


class OpenAIAdapter:
    """OpenAI adapter — one instance per agent, manages N sessions.

    Internally caches a SessionManager per session_id. Lifecycle (create
    conversation + session + participants) is lazy — triggered on first
    ingestion.

    Args:
        agent_id: Beacon registered agent ID.
        owner_id: REQUIRED. Default owner ID bound to this adapter. In a
            multi-user app, pass the real end-user owner per call via
            ``push(... owner_id=...)`` or ``ingest(... owner_id=...)`` — the
            per-call owner scopes BOTH reads (recall) and writes (user_message
            participant_id). A ``session_id`` is pinned to the first owner it
            is seen with; a later call with a different owner for the same
            session is rejected.
        auth: Shared ``ContextAuth`` instance. Recommended when the same
            credentials back multiple adapters (OpenAI + Claude + ADK + LangGraph).
            Takes precedence over inline credential kwargs.
        private_key / consumer_key / subject / login_url: Inline connected-app
            credentials. If all four are supplied (and ``auth`` is None), the
            adapter builds a ``ContextAuth`` internally.
        secrets_api_key: API-key path — the SDK downloads the full connected-app
            credential set from the secrets server on first token fetch.
        secrets_url: Base URL of the secrets server. Falls back to the
            ``AGENTIC_CONTEXT_SECRET_SERVER_URL`` env var when empty.
        dataspace: Beacon dataspace name. Resolves from the ``CDP_DATASPACE``
            env var, then to ``"default"`` when unset.
        owner_type: Override the Beacon owner type. Defaults to the auth
            class's advertised value (``"individual"`` for service-account auth,
            ``"employee"`` for interactive user auth).
        use_async: Use async clients. Default True. The Agents SDK tool loop
            is async-native, so sync mode is only for the pure ingestion path.
        context_mode: ``"auto"`` seeds Beacon context at conversation-open
            (the next ``push``) and leaves subsequent reads to the agent via
            the recall tool. ``"manual"`` disables adapter-side injection —
            the tool is still registered. Runtime-reversible via
            :meth:`set_context_mode`.
        agent_participant_variable: Full-override for Beacon's
            ``AiAgentParticipant`` label. Defaults to
            ``AiAgentType="OpenAI"`` + the adapter's ``agent_id``.
    """

    capabilities = Capabilities(working=True, persistent=False)

    def __init__(
        self,
        agent_id: str,
        *,
        owner_id: str | None = None,
        auth: ContextAuth | None = None,
        private_key: str = "",
        consumer_key: str = "",
        subject: str = "",
        login_url: str = "",
        secrets_api_key: str = "",
        secrets_url: str = "",
        dataspace: str | None = None,
        owner_type: str | None = None,
        use_async: bool = True,
        context_mode: str = "auto",
        agent_participant_variable: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        if not owner_id:
            raise ValueError(
                "OpenAIAdapter requires owner_id. Pass the end-user id (or a "
                "stable service-account id for single-tenant deployments). "
                "There is no default — a shared adapter must not attribute "
                "transcripts to a placeholder owner."
            )
        # Soft nudge — the Agents SDK is only needed by the recall tool, so
        # construction stays non-fatal when it is missing. find_spec never
        # imports the module, so it is cheap and side-effect-free.
        if importlib.util.find_spec(_SDK_MODULE) is None:
            warnings.warn(
                f"{_SDK_MODULE} (openai-agents) is not installed. "
                f"OpenAIAdapter.make_context_retrieval_tool() will fail until "
                f"you install it:\n\n    {_SDK_INSTALL_HINT}\n\n"
                "The pure ingestion path (push/snapshot/ingest/get_context) "
                "works without it.",
                stacklevel=2,
            )
        self._owner_id = owner_id
        self._use_async = use_async
        dataspace = dataspace or os.environ.get("CDP_DATASPACE") or "default"
        self._dataspace = dataspace
        self._owner_type_override = owner_type
        self._context_mode = _normalize_context_mode(context_mode)
        # Two independent one-shot gates; must NOT share a set — ``push()``
        # and ``instructions_provider`` have different lifecycle semantics
        # and either-order wiring the two together loses whichever call
        # runs second.
        self._fast_context_seeded: set[str] = set()
        self._catalog_fetched_sessions: set[str] = set()
        # run_hooks turns with captured records not yet posted; close() flushes.
        self._unflushed_hooks: set[Any] = set()
        # Cache of the payload returned by the auto-seed ``get_context`` call so
        # the recall tool can serve it without a second Beacon round-trip. Keyed
        # by session_id; populated only when the seed succeeded.
        self._seeded_contexts: dict[str, dict[str, Any]] = {}
        # Pack ids the integrator declared via ``make_get_context_pack_tool`` /
        # ``default_tools(valid_pack_ids=…)``. Shared source of truth for the
        # instructions-provider catalog block AND the pack-tool allow set, so
        # a server-catalog miss degrades to a block that lists exactly what
        # the tool will accept. Populated when the tool is constructed.
        self._pinned_valid_pack_ids: list[str] = []
        # Server-side context_pack_instruction prompt, fetched on
        # a session's first ``instructions_provider`` turn and pinned for the
        # rest of that session.
        self._context_pack_prompt = SessionPrompt(CONTEXT_PACK_INSTRUCTION)

        # Resolve auth exactly like the other adapters: shared instance >
        # inline credentials > secrets API key. No static token path — token
        # exchange and 401-retry must be handled by the SDK.
        if auth is not None:
            self._auth: ContextAuth | ApiKeyAuth = auth
        elif private_key and consumer_key and subject and login_url:
            self._auth = ContextAuth(
                private_key=private_key,
                consumer_key=consumer_key,
                subject=subject,
                login_url=login_url,
            )
        elif secrets_api_key:
            self._auth = ApiKeyAuth(
                secrets_api_key, secret_server_url=secrets_url or None
            )
        else:
            raise ValueError(
                "OpenAIAdapter requires credentials. Pass auth=ContextAuth(...), "
                "inline private_key+consumer_key+subject+login_url, or "
                "secrets_api_key= to fetch them from the secrets server."
            )

        self._owner_type = (
            self._owner_type_override
            if self._owner_type_override is not None
            else getattr(self._auth, "default_owner_type", "individual")
        )

        # JWT → CDP exchange here; base_url is discovered from the exchange.
        api_key, base_url = self._auth.get_token()
        self._base_url = base_url

        # Beacon AiAgentParticipant label. If the caller didn't full-override,
        # splice the agent_id into the SDK default so analytics show this
        # agent, not "Custom".
        if agent_participant_variable is None:
            self._agent_participant_variable: dict[str, Any] = {
                **_AGENT_PARTICIPANT_VARIABLE,
                "AiAgentApiName": agent_id,
            }
        else:
            self._agent_participant_variable = dict(agent_participant_variable)

        self._transformer = RecordTransformer(agent_id, owner_id)
        self._sessions: dict[str, Any] = {}
        # Catalog state. ``_allowed_pack_ids`` is authoritative for the
        # ``get_context_pack`` tool; ``_allowed_context_pack_list`` is the
        # ``[Available context packs]\n<json>`` block the LLM sees. Both are
        # written together, and only via ``_publish_catalog`` — so the LLM
        # never sees an id the tool would reject, and the tool never accepts
        # an id the LLM wasn't told about. Writers: eager seed from
        # ``make_get_context_pack_tool`` (adapter construction time) and
        # ``_provider`` on each session's first turn.
        self._allowed_pack_ids: set[str] = set()
        self._allowed_context_pack_list: str = ""
        # Cached pack-id -> catalog version. Populated from the fetched catalog
        # (ints only, never nulls or bools) so the get_context_pack handler can
        # fill it in for the model when the model omits context_pack_version
        # (enforcing the MUST declared in CONTEXT_PACK_VERSION_GUIDANCE at the
        # SDK boundary). Patched on status=version_expired so subsequent calls
        # pin to the new revision without a catalog refresh.
        self._pack_versions: dict[str, int] = {}

        if use_async:
            self._ingestion_client = AsyncIngestionClient(
                base_url=base_url, api_key=api_key, dataspace=dataspace, auth=self._auth,
            )
            self._retrieval_client = AsyncRetrievalClient(
                base_url=base_url, api_key=api_key, dataspace=dataspace, auth=self._auth,
            )
        else:
            self._ingestion_client = IngestionClient(
                base_url=base_url, api_key=api_key, dataspace=dataspace, auth=self._auth,
            )
            self._retrieval_client = RetrievalClient(
                base_url=base_url, api_key=api_key, dataspace=dataspace, auth=self._auth,
            )

        self._auth.attach(self._ingestion_client)
        self._auth.attach(self._retrieval_client)

    # ─── Session Management (internal) ────────────────────────────

    def _get_or_create_manager(
        self, session_id: str, owner_id: str | None = None,
    ) -> Any:
        """Get or create a SessionManager for ``session_id`` (cached).

        A ``session_id`` is bound to one owner for the lifetime of this
        adapter. If a caller passes an explicit ``owner_id`` that disagrees
        with the cached binding, refuse the call — reusing a conversation id
        across owners would attribute one user's transcript to another.
        """
        if session_id in self._sessions:
            manager = self._sessions[session_id]
            cached_owner = getattr(manager, "_owner_id", None)
            if owner_id and cached_owner and owner_id != cached_owner:
                raise ValueError(
                    f"session_id {session_id!r} is already bound to owner "
                    f"{cached_owner!r}; refusing to serve owner {owner_id!r}. "
                    "Session ids must not be reused across owners."
                )
            return manager

        resolved_owner = owner_id or self._owner_id
        manager_kwargs = dict(
            agent_id=self.agent_id,
            owner_id=resolved_owner,
            owner_type=self._owner_type,
            session_id=session_id,
            retrieval_client=self._retrieval_client,
            agent_participant_variable=self._agent_participant_variable,
        )

        if self._use_async:
            manager = AsyncSessionManager(self._ingestion_client, **manager_kwargs)
        else:
            manager = SessionManager(self._ingestion_client, **manager_kwargs)

        self._sessions[session_id] = manager
        logger.debug("[OpenAI] created session manager for %s", session_id[:8])
        return manager

    # ─── Public API: ContextSource contract ───────────────────────

    def snapshot(
        self,
        session_id: str,
        *,
        messages: list[dict] | None = None,
        items: list[dict] | None = None,
        owner_id: str | None = None,
        interaction_id: str | None = None,
    ) -> list[Record]:
        """Normalize a full conversation into Records (ContextSource contract).

        Accepts either a Chat Completions messages list OR a Responses API
        items list (from ``RunResult.to_input_list()``). Both route through the
        same transformer, so a mix is also fine — the transformer sniffs each
        entry.

        ``owner_id`` is stamped as ``participant_id`` on user_message records
        (fallback: the adapter's default owner). ``interaction_id`` groups
        every emitted record under one Beacon turn (see
        :meth:`push` for the streaming callers that mint it).
        """
        entries = self._collect_entries(messages, items)
        return self._transformer.transform(
            entries, session_id,
            owner_id=owner_id, interaction_id=interaction_id,
        )

    # ─── Public API: Lifecycle ────────────────────────────────────

    async def start_session(
        self, session_id: str, *, owner_id: str | None = None, title: str = "",
    ) -> dict[str, str]:
        """Optional — eagerly create session in Beacon.

        409 (already exists) is treated as success — returning users just work.
        """
        manager = self._get_or_create_manager(session_id, owner_id)
        if self._use_async:
            return await manager.create_session(session_id, title=title)
        return manager.create_session(session_id, title=title)

    # ─── Public API: Ingestion ────────────────────────────────────

    async def push(
        self,
        session_id: str,
        *,
        messages: list[dict] | None = None,
        items: list[dict] | None = None,
        owner_id: str | None = None,
        interaction_id: str | None = None,
    ) -> None:
        """Push a conversation (full or incremental slice) to Beacon.

        Supply ``messages`` (Chat Completions) or ``items`` (Responses API) or
        both — both route through the transformer which sniffs each entry.
        Lifecycle is lazy; the first push triggers conversation + session
        creation.

        Snapshot mode: pass the full conversation. Do it once per session:
        ``event_id`` is not sent, so nothing is deduped and re-passing earlier
        entries on a later turn writes them again.

        Streaming mode: pass only the new entries this turn — the transformer
        emits one Record per logical event, and the manager POSTs one batch
        per call. One push per turn, never per node or token.

        Under ``context_mode="auto"`` the first push for a ``session_id``
        seeds Beacon fast context in the background (best-effort). This is a
        no-op under ``context_mode="manual"`` and on later pushes for the
        same session.
        """
        if not messages and not items:
            return
        manager = self._get_or_create_manager(session_id, owner_id)
        resolved_owner = (
            getattr(manager, "_owner_id", None) or owner_id or self._owner_id
        )
        await self._maybe_seed_fast_context(session_id, resolved_owner)

        records = self.snapshot(
            session_id, messages=messages, items=items,
            owner_id=resolved_owner, interaction_id=interaction_id,
        )
        if not records:
            return
        if self._use_async:
            await manager.ingest(records)
        else:
            manager.ingest(records)

    async def ingest(
        self,
        session_id: str,
        records: list[Record],
        *,
        owner_id: str | None = None,
    ) -> None:
        """Ingest pre-built Records directly — bypass the transformer."""
        if not records:
            return
        manager = self._get_or_create_manager(session_id, owner_id)
        if self._use_async:
            await manager.ingest(records)
        else:
            manager.ingest(records)

    # ─── Public API: Retrieval ────────────────────────────────────

    async def get_context(
        self, session_id: str | None = None, *, owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fast context read for ``session_id``.

        Best-effort — returns None on error. ``owner_id`` only matters for a
        cold session (one this adapter has not seen before).
        """
        sid = session_id or self._last_session_id
        if not sid:
            return None
        manager = self._get_or_create_manager(sid, owner_id)
        if self._use_async:
            return await manager.get_context()
        return manager.get_context()

    async def get_session(
        self, session_id: str | None = None, *, owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch full session data (messages, metadata) for ``session_id``."""
        sid = session_id or self._last_session_id
        if not sid:
            return None
        manager = self._get_or_create_manager(sid, owner_id)
        if self._use_async:
            return await manager.get_session()
        return manager.get_session()

    async def list_sessions(
        self, owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List conversations for an owner. Best-effort — returns [] on error."""
        resolved_owner = owner_id or self._owner_id
        try:
            if self._use_async:
                return await self._retrieval_client.list_conversations(
                    resolved_owner, self.agent_id, owner_type=self._owner_type,
                )
            return self._retrieval_client.list_conversations(
                resolved_owner, self.agent_id, owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "list_sessions failed for owner=%s (best-effort, returning []): %s",
                resolved_owner, e,
            )
            return []

    # ─── Public API: Context mode ─────────────────────────────────

    @property
    def context_mode(self) -> str:
        """Current context mode — ``"auto"`` or ``"manual"``."""
        return self._context_mode

    def set_context_mode(self, mode: str) -> None:
        """Switch context mode at runtime. Operator/developer knob — reversible."""
        previous = self._context_mode
        self._context_mode = _normalize_context_mode(mode)
        if previous != self._context_mode:
            logger.info(
                "[OpenAI] context_mode: %s → %s (runtime toggle)",
                previous,
                self._context_mode,
            )

    # ─── Public API: Server-side context_pack_instruction ─────────

    async def get_context_pack_instruction(self, session_id: str) -> str:
        """Return the context-pack instruction pinned for ``session_id``.

        :meth:`instructions_provider` already adds this to the agent's
        instructions on every turn, so most integrations never call it.
        Use it only when building ``instructions`` yourself. The first call
        for a session fetches the server-published prompt
        (``GET /api/v1/agentic/prompts/system/context_pack_instruction``);
        a miss or failure pins the bundled
        :data:`agentic_context.CONTEXT_PACK_INSTRUCTION` for that session.
        """
        return await self._context_pack_prompt.get(session_id, self._retrieval_client)

    def _should_inject_fast_context(self, session_id: str, *, is_open: bool) -> bool:
        """Mirror the cadence rule used by the other adapters."""
        if self._context_mode == "auto":
            return is_open
        return False

    def _update_context_pack_list(self, new_context_pack_list: dict) -> None:
        """Transform Beacon's pack catalog into the system-instruction block
        + populate the id allowlist used by the pack tool.

        Mirrors the Claude adapter's ``update_context_pack_list`` so the
        system-instruction block is byte-identical across adapters
        (``[Available context packs]\\n<json>``). An empty server response
        degrades to the pinned-catalog view so the block and the pack tool
        agree on which ids are usable this turn.
        """
        new_map: dict[str, dict] = {}
        new_pack_versions: dict[str, int] = {}
        for pack in new_context_pack_list.get("contextPacks", []):
            pack_id = pack["contextPackId"]
            entry: dict[str, Any] = {
                "description": pack.get("description") or "",
                "dataNeeds": [
                    {
                        "name": need["name"],
                        "dataNeedDescription": need.get("description") or "",
                        "parameters": need.get("parameters") or [],
                    }
                    for need in pack.get("dataNeeds") or []
                ],
            }
            # Surface `version` to the LLM when the server reports one so it
            # can echo it back on the call as ``context_pack_version`` and
            # recover from ``version_expired``. The pinned-fallback catalog
            # does not include a version — the integrator pinned by id, not
            # by revision.
            # Only int values are surfaced to the LLM and seed the SDK-side
            # default. ``null`` means the server has no revision for this
            # pack; bool is an int subclass so guard it explicitly. Leaking
            # null/bool into the catalog block would satisfy the "entry
            # includes a `version`" condition in CONTEXT_PACK_VERSION_GUIDANCE
            # and make the LLM echo a value the SDK would then reject.
            version = pack.get("version")
            if isinstance(version, int) and not isinstance(version, bool):
                entry["version"] = version
                new_pack_versions[pack_id] = version
            new_map[pack_id] = entry
        if not new_map:
            new_map = self._build_pinned_catalog()
        self._publish_catalog(new_map, new_pack_versions)

    def _build_pinned_catalog(self) -> dict[str, dict]:
        """Synthetic catalog from pinned ids — same shape as the server's,
        but with empty ``description``/``dataNeeds`` since those come from
        the server. Used when the server catalog is empty or unreachable so
        the LLM at least sees which pack ids it can call by name.
        """
        return {
            pid: {"description": "", "dataNeeds": []}
            for pid in self._pinned_valid_pack_ids
        }

    def _publish_catalog(
        self,
        source_map: dict[str, dict],
        pack_versions: dict[str, int] | None = None,
    ) -> None:
        """Single write-point for the catalog block + the allow set.

        Everything the pack tool accepts is listed in the block, and vice
        versa — so the LLM never sees an id the tool would reject, and the
        tool never accepts an id the LLM wasn't told about.
        """
        self._allowed_pack_ids = set(source_map)
        self._allowed_context_pack_list = (
            f"{CONTEXT_PACK_LIST_MARKER}\n{json.dumps(source_map)}"
        )
        self._pack_versions = dict(pack_versions or {})

    # ─── Public API: dynamic instructions (pack-list injection) ────

    def instructions_provider(
        self,
        base: str,
        session_id: str,
        owner_id: str | None = None,
    ):
        """Return an OpenAI Agents SDK ``instructions`` callable.

        The Agents SDK re-invokes the returned coroutine on every turn.
        On the first invocation per ``session_id`` it fetches the agent's
        authorized context-pack catalog from Beacon and caches two things
        on the adapter:

          - ``_allowed_pack_ids`` — the id set the ``get_context_pack``
            tool validates against (server catalog wins over the pinned
            ``valid_pack_ids`` fallback).
          - ``_allowed_context_pack_list`` — the formatted
            ``[Available context packs]\\n<json>`` block appended to
            ``base`` on every turn so the LLM sees which pack ids exist
            and which data needs each covers.

        Every turn also gets the context-pack instruction (server-published,
        pinned per session — see :meth:`get_context_pack_instruction`),
        placed between ``base`` and the catalog block. It is skipped if
        ``base`` already contains it, so there is no need to prepend
        :data:`CONTEXT_PACK_INSTRUCTION` yourself.

        Mirrors the Claude adapter's ``UserPromptSubmit`` injection — one
        pack-list fetch per session, cached for all subsequent turns of
        that session. ``_catalog_fetched_sessions`` is adapter-wide (and
        independent of the fast-context seed gate used by ``push()``), so
        one adapter can drive many concurrent sessions and each gets
        exactly one fetch.

        Usage::

            agent = Agent(
                name="support",
                instructions=adapter.instructions_provider(
                    base="You are a helpful assistant.",
                    session_id="conv-1",
                ),
                tools=adapter.default_tools(
                    "conv-1", valid_pack_ids=["PlanRules"],
                ),
            )

        A fetch failure (or an empty server catalog) degrades to the
        pinned view: if ``make_get_context_pack_tool`` was constructed with
        ``valid_pack_ids``, those ids are surfaced in the block with empty
        description/dataNeeds; otherwise the block is ``{}``. Either way
        the LLM sees the catalog was consulted and the pack tool's allow
        set matches the block (no silent disagreement). A warning is
        logged on fetch failure.

        Args:
            base: Base system instructions the customer owns. The block is
                always appended when ``instructions_provider`` runs; the
                map is server-, pinned-, or empty-{} depending on fetch
                outcome.
            session_id: Beacon session this Agent will drive. Pinned to
                this provider; reuse the adapter + call
                ``instructions_provider`` again for a new conversation.
            owner_id: Overrides the adapter-default owner for the
                session↔owner bind check. Pass when the end-user owner
                differs from the one passed at adapter construction.

        Raises:
            RuntimeError: If the adapter was constructed with
                ``use_async=False`` — the Agents SDK ``instructions``
                callable is async-only.
            ValueError: If the ``session_id`` is already bound to a
                different owner (propagated from
                ``_get_or_create_manager``).
        """
        if not self._use_async:
            raise RuntimeError(
                "OpenAIAdapter.instructions_provider() requires use_async=True; "
                "the Agents SDK instructions callable is async-only."
            )
        # Bind manager now so the session↔owner check fails fast at
        # Agent construction, not on the first turn.
        self._get_or_create_manager(session_id, owner_id)

        async def _provider(run_context: Any, agent: Any) -> str:
            # Snapshot-then-mark mirrors the Claude adapter: a failed fetch
            # does NOT retry. Uses its own gate (not shared with ``push()``'s
            # fast-context seed) so either call order works.
            is_open = session_id not in self._catalog_fetched_sessions
            self._catalog_fetched_sessions.add(session_id)
            if is_open:
                try:
                    fetched = await self._retrieval_client.get_context_pack_list(
                        agent_id=self.agent_id,
                    )
                except Exception as e:
                    logger.warning(
                        "instructions_provider pack list fetch failed "
                        "[session=%s]: %s: %s",
                        session_id[:8], type(e).__name__, e,
                    )
                    # Preserve "server catalog wins once seen": only degrade
                    # to the pinned view if no prior session has populated
                    # ``_allowed_pack_ids`` from a successful server fetch.
                    # Empty pinned → empty block, tool rejects everything.
                    if not self._allowed_pack_ids:
                        self._publish_catalog(self._build_pinned_catalog())
                else:
                    self._update_context_pack_list(fetched)
            parts = [base] if base else []
            # The instruction explains the catalog; with no packs it is only cost.
            if self._allowed_pack_ids:
                instruction = await self.get_context_pack_instruction(session_id)
                if not self._context_pack_prompt.present_in(base, instruction):
                    parts.append(instruction)
            if self._allowed_context_pack_list:
                parts.append(self._allowed_context_pack_list)
            return "\n\n".join(parts)

        return _provider

    # ─── Public API: On-demand memory tool ─────────────────────────

    def make_context_retrieval_tool(
        self,
        session_id: str | None = None,
        owner_id: str | None = None,
    ) -> Any:
        """Return an OpenAI Agents SDK ``@function_tool`` for cross-session memory.

        The returned callable is decorated with ``agents.function_tool`` so it
        plugs straight into an Agents SDK ``Agent(tools=[...])`` list. The tool
        reads cross-session profile context — the model passes an optional
        ``topic`` hint to signal what it is reaching for; the hint is logged
        for audit only (the Beacon read is scoped to owner + agent).

        Owner and session are bound server-side here so the model never passes
        a customer id. ``session_id`` is required for the tool to return a
        result — an unbound tool degrades to ``not_ready`` (see below).
        """
        _require_sdk("make_context_retrieval_tool")
        from agents import function_tool  # type: ignore[import-not-found]

        adapter = self
        bound_session = session_id
        bound_owner = owner_id

        @function_tool(
            name_override="recall_customer_context",
            description_override=(
                "Look up what we already know about this customer from past "
                "conversations (cross-session memory). Call this when "
                "answering would benefit from memory the current conversation "
                "hasn't covered yet — the customer's plan or account details, "
                "issues raised before, stated preferences, or anything they'd "
                "expect you to remember. You do NOT need to call it for small "
                "talk or when the current conversation already holds the answer."
            ),
        )
        async def recall_customer_context(topic: str = "") -> str:
            """Return cross-session customer context, scoped to this agent."""
            import json as _json
            if bound_session is None:
                # A tool built without a pinned session has no conversation to
                # scope the read to. Falling back to the adapter's most-recent
                # session leaks context between concurrent sessions sharing one
                # adapter — return not_ready instead.
                logger.info(
                    "[OpenAI] recall_customer_context: topic=%r → not_ready "
                    "(no session bound to tool)", topic,
                )
                return _json.dumps({
                    "status": "not_ready",
                    "message": (
                        "Context recall is not available for this turn — the "
                        "tool was built without a session binding."
                    ),
                }, indent=2)
            # Enforce the session ↔ owner binding BEFORE consulting the cache.
            # Otherwise a tool bound to (session=A, owner=B) would serve
            # owner-A's cached context when A seeded the session first —
            # get_context()'s manager lookup normally rejects that mismatch,
            # and the cache path must keep the same contract.
            try:
                adapter._get_or_create_manager(bound_session, bound_owner)
            except ValueError:
                logger.info(
                    "[OpenAI] recall_customer_context: owner=%r session_id=%r "
                    "topic=%r → not_ready (owner mismatch for session)",
                    bound_owner, bound_session, topic,
                )
                return _json.dumps({
                    "status": "not_ready",
                    "message": (
                        "Context recall is not available for this turn — the "
                        "session is bound to a different owner."
                    ),
                }, indent=2)
            # Prefer the payload the push() auto-seed already cached — avoids a
            # second Beacon round-trip for every turn. Falls back to a live read
            # when the cache is empty (manual mode, or the seed never ran).
            context = adapter._seeded_contexts.get(bound_session)
            if context is None:
                context = await adapter.get_context(
                    bound_session, owner_id=bound_owner,
                )
            has_context = bool(context and context.get("userContext"))
            logger.info(
                "[OpenAI] recall_customer_context: owner=%r session_id=%r "
                "topic=%r → has_context=%s",
                bound_owner or adapter._owner_id,
                bound_session,
                topic,
                has_context,
            )
            payload = context or {"userContext": []}
            return _json.dumps(payload, indent=2)

        return recall_customer_context

    def make_get_context_pack_tool(
        self,
        valid_pack_ids: list[str],
        session_id: str | None = None,
        owner_id: str | None = None,
        *,
        param_defaults: dict[str, Any] | None = None,
        timeout: float | None = CONTEXT_PACK_TIMEOUT_S,
    ) -> Any:
        """Return an OpenAI Agents SDK ``@function_tool`` for the pack-scoped read.

        This is the pack-scoped read (POST GetContext) — the
        agent asks for one server-advertised ``context_pack`` of structured
        detail instead of the fast/deep profile read
        :meth:`make_context_retrieval_tool` does. It is the OpenAI counterpart
        of the Claude/ADK/LangGraph ``make_get_context_pack_tool`` — same
        capability in every adapter, so any agent built from QUICKSTART can
        offer packs.

        One adapter supports at most one pack-tool pinned set: calling this
        method a second time with a different ``valid_pack_ids`` raises
        ``RuntimeError``. Different sets would race on the shared
        ``_allowed_pack_ids`` state and silently break one of the two
        tools. Build a second adapter instance for a second agent.

        **Side effect at construction time**: the pinned ids are published to
        the adapter immediately (``_allowed_pack_ids`` and
        ``_allowed_context_pack_list``) so a tools-only integrator (no
        ``instructions_provider`` wired) still gets a working pack tool. A
        successful server catalog fetch later via
        :meth:`instructions_provider` overrides this with the richer server
        metadata.

        Args:
            valid_pack_ids: the pack ids this agent advertises. Published to
                the adapter so :meth:`instructions_provider` can surface them
                in the ``[Available context packs]`` block when the server
                catalog is empty or unreachable — the block and this tool's
                allow set are then guaranteed to agree. Any id outside
                ``self._allowed_pack_ids`` is rejected with
                ``{"status": "invalid_pack"}`` without a server call.
            session_id / owner_id: bind this tool to a conversation / owner
                (as with :meth:`make_context_retrieval_tool`). ``session_id``
                is required for the tool to return a result — an unbound tool
                degrades to ``not_ready``.
            param_defaults: fixed pack ``params`` merged into every call. The
                model's ``params`` JSON overrides those keys. Blank values are
                omitted.
            timeout: per-call HTTP timeout for the (slow, deep) pack POST;
                defaults to :data:`agentic_context.CONTEXT_PACK_TIMEOUT_S`.

        The tool is best-effort: it NEVER raises. Beyond ``invalid_pack`` it
        returns ``{"status": "ok", "context": {...}}`` on success or
        ``{"status": "not_ready", "detail": ...}`` on any failure (endpoint
        not deployed, sync mode, auth/init failure, timeout) — matching the
        PREVIEW contract in :data:`agentic_context.CONTEXT_PACK_INSTRUCTION`
        so the agent degrades instead of failing.
        """
        _require_sdk("make_get_context_pack_tool")
        from agents import function_tool  # type: ignore[import-not-found]

        import json as _json

        adapter = self
        bound_session = session_id
        bound_owner = owner_id
        defaults = dict(param_defaults or {})
        # One adapter supports at most one pack-tool pinned set. A second call
        # with a different list would race on the shared ``_allowed_pack_ids``
        # state (last writer wins) and silently break one of the two tools.
        # Build a second adapter instance for a second agent instead.
        new_pinned = list(valid_pack_ids)
        if self._pinned_valid_pack_ids and self._pinned_valid_pack_ids != new_pinned:
            raise RuntimeError(
                "make_get_context_pack_tool was already called on this adapter "
                f"with pinned ids {self._pinned_valid_pack_ids!r}; a second "
                f"call with {new_pinned!r} would race on the shared catalog "
                "state. Build a second OpenAIAdapter instance for a second "
                "pack tool."
            )
        # Publish pinned ids to the adapter so ``instructions_provider`` can
        # fall back to them when the server catalog is empty or unreachable.
        # The pack tool reads ``_allowed_pack_ids`` only; the block and the
        # tool are kept consistent by ``_publish_catalog``.
        self._pinned_valid_pack_ids = new_pinned
        # Seed the catalog from pinned eagerly so a tools-only integrator
        # (no ``instructions_provider`` wired) still has a working pack tool.
        # A successful server fetch from ``_provider`` will override later.
        if not self._allowed_pack_ids:
            self._publish_catalog(self._build_pinned_catalog())

        @function_tool(
            name_override="get_context_pack",
            description_override=(
                "Retrieve one named context pack for the current subject. "
                "This is the same read as getContext(contextPack).\n\n"
                "The \"[Available context packs]\" block lists each pack's "
                "description and its data needs. If a pack or one of its data "
                "needs would help answer the user's question, call this tool "
                "with that pack's id as `context_pack_id`.\n\n"
                "Pass parameter values in `params`, a JSON object whose keys "
                "are the parameter names from those data needs. Use values "
                "you can determine from the conversation. If you cannot "
                "determine a value, ask the user, or omit that key. Do not "
                "invent a value and do not send an empty string. A data need "
                "that declares an omitted parameter may be absent from the "
                "result; answer from the data needs that did return.\n\n"
                + CONTEXT_PACK_VERSION_GUIDANCE
            ),
        )
        async def get_context_pack(
            context_pack_id: str,
            params: str = "",
            context_pack_version: int | None = None,
        ) -> str:
            def envelope(payload: dict) -> str:
                return _json.dumps(payload, indent=2)

            # Reject non-integer context_pack_version at the SDK boundary so a
            # bad value never masquerades as "omitted" (which would silently
            # bypass versioning).
            if context_pack_version is not None and (
                isinstance(context_pack_version, bool)
                or not isinstance(context_pack_version, int)
            ):
                logger.warning(
                    "[OpenAI] get_context_pack: %r rejected non-integer "
                    "context_pack_version=%r",
                    context_pack_id, context_pack_version,
                )
                return envelope({
                    "status": "invalid_version",
                    "context_pack_id": context_pack_id,
                    "message": (
                        "`context_pack_version` must be an integer from the "
                        "catalog's `version` field. Re-call with the integer "
                        "value, or omit it if the catalog entry has no version."
                    ),
                })

            # Enforce CONTEXT_PACK_VERSION_GUIDANCE at the SDK boundary: when
            # the model omits ``context_pack_version`` and the cached catalog
            # has one for this pack, fill it in.
            if context_pack_version is None:
                context_pack_version = adapter._pack_versions.get(context_pack_id)

            allowed = adapter._allowed_pack_ids

            if context_pack_id not in allowed:
                logger.info(
                    "[OpenAI] get_context_pack: invalid pack id %r",
                    context_pack_id,
                )
                return envelope({
                    "status": "invalid_pack",
                    "context_pack_id": context_pack_id,
                    "valid_pack_ids": sorted(allowed),
                    "message": (
                        f"'{context_pack_id}' is not an available context pack. "
                        f"Valid pack ids: {', '.join(sorted(allowed))}."
                    ),
                })

            def not_ready(reason: str) -> dict:
                logger.info(
                    "[OpenAI] get_context_pack: %r → not_ready (%s)",
                    context_pack_id, reason,
                )
                return {
                    "status": "not_ready",
                    "context_pack_id": context_pack_id,
                    "message": (
                        "Context pack retrieval is a preview capability and "
                        "is not available in this environment yet. Fall back "
                        "to the context you already have."
                    ),
                    "detail": reason,
                }

            # Pack POST is awaited — needs the async retrieval client. Sync
            # mode degrades to not_ready.
            if not adapter._use_async:
                return envelope(not_ready(
                    "pack read requires async mode (use_async=True)"
                ))

            # A tool built without a session binding cannot fall back to
            # ``adapter._last_session_id`` — that would serve concurrent
            # sessions sharing one adapter the wrong conversation's pack.
            sid = bound_session
            if not sid:
                return envelope(not_ready("no session bound to tool"))
            manager = adapter._get_or_create_manager(sid, bound_owner)
            resolved_owner = (
                getattr(manager, "_owner_id", None) or adapter._owner_id
            )
            if not resolved_owner:
                return envelope(not_ready("no owner for this turn"))

            # owner_id is bound server-side from the resolved owner — never
            # taken from the model. Blank parameter values are omitted so a
            # data need that requires them is skipped rather than queried empty.
            sent = context_pack_params(params, "", defaults)

            logger.info(
                "[OpenAI] get_context_pack: %r params=%s version=%s",
                context_pack_id, sorted(sent), context_pack_version,
            )
            try:
                payload = await adapter._retrieval_client.get_context_pack(
                    sid,
                    resolved_owner,
                    adapter.agent_id,
                    context_pack_id,
                    params=sent,
                    owner_type=adapter._owner_type,
                    context_pack_version=context_pack_version,
                    timeout=timeout,
                )
            except Exception as e:
                # Log the full exception server-side; return only its TYPE.
                # repr(e) can carry request URLs, tokens, or upstream error
                # text, which must not reach the model.
                logger.warning(
                    "[OpenAI] get_context_pack: %r request failed",
                    context_pack_id, exc_info=True,
                )
                return envelope(not_ready(
                    f"request failed: {type(e).__name__}"
                ))

            mismatch = version_mismatch_entry(payload)
            if mismatch is not None:
                logger.info(
                    "[OpenAI] get_context_pack: %r → version_expired "
                    "(requested=%s, latest=%s)",
                    context_pack_id,
                    context_pack_version,
                    mismatch["version"],
                )
                # Patch cached catalog state so the next call (and the next
                # catalog-block render) sees the fresh version. When ``latest``
                # is ``None`` (Beacon reports no current revision), drop the
                # pin from the cache and the catalog so the retry goes
                # unpinned instead of re-sending the stale version.
                latest = mismatch["version"]
                if isinstance(latest, int) and not isinstance(latest, bool):
                    adapter._pack_versions[context_pack_id] = latest
                else:
                    adapter._pack_versions.pop(context_pack_id, None)
                adapter._allowed_context_pack_list = patch_pack_version_in_catalog(
                    adapter._allowed_context_pack_list, context_pack_id, latest,
                )
                return envelope({
                    "status": "version_expired",
                    "context_pack_id": context_pack_id,
                    "requested_version": context_pack_version,
                    "latest_version": mismatch["version"],
                    "latest": mismatch["entry"],
                    "message": (
                        "The pack has been republished since this agent saw "
                        "the catalog. Review the `latest` description, then "
                        "retry get_context_pack with "
                        "context_pack_version=latest_version if the new data "
                        "needs still cover your plan."
                    ),
                })

            logger.info(
                "[OpenAI] get_context_pack: %r → ok", context_pack_id,
            )
            return envelope({
                "status": "ok",
                "context_pack_id": context_pack_id,
                "context": payload,
            })

        return get_context_pack

    def default_tools(
        self,
        session_id: str | None = None,
        owner_id: str | None = None,
        *,
        valid_pack_ids: list[str] | None = None,
    ) -> list[Any]:
        """SDK-provided Agents SDK tools for this adapter.

        Convenience helper that returns both on-demand memory tools in one
        call, matching the Claude/ADK/LangGraph ``default_tools`` shape::

            agent = Agent(
                ...,
                tools=[*adapter.default_tools(session_id), *my_tools],
            )

        Pass the same ``session_id`` used by
        :meth:`instructions_provider` / :meth:`run_hooks`; an unbound tool
        degrades to ``not_ready`` for every call.

        Includes:

        - ``recall_customer_context`` — fast/deep profile read.
        - ``get_context_pack`` — named-pack read for structured detail. The
          tool reads from ``self._allowed_pack_ids``. ``valid_pack_ids`` is
          published to the adapter at construction time (eager initial value
          — tools-only integrators get a working pack tool with no
          ``instructions_provider`` wired) AND acts as the degrade target if
          the server catalog fetch returns empty or fails. A successful
          server catalog fetch replaces both the allow set and the LLM-
          visible block.

        Kept separate from :meth:`make_context_retrieval_tool` /
        :meth:`make_get_context_pack_tool` so integrators who want a custom
        tool list can still reach the primitives.
        """
        return [
            self.make_context_retrieval_tool(session_id, owner_id),
            self.make_get_context_pack_tool(
                list(valid_pack_ids or []),
                session_id,
                owner_id,
            ),
        ]

    # ─── Public API: Streaming hooks ──────────────────────────────

    def run_hooks(
        self,
        session_id: str,
        owner_id: str | None = None,
        *,
        user_input: str | list[dict] | None = None,
        next_hooks: Any = None,
    ) -> Any:
        """Return an Agents SDK ``RunHooks`` that ingests this run as one turn.

        Create one per ``Runner.run`` and pass the turn's user input::

            hooks = adapter.run_hooks(session_id, user_input=user_text)
            try:
                result = await Runner.run(agent, user_text, hooks=hooks)
            finally:
                await hooks.flush()

        One hooks object is one turn: every Record it produces — the user
        message, then ``agent_response`` / ``agent_reasoning`` / ``tool_call``
        from ``on_llm_end`` and ``tool_result`` from ``on_tool_end`` — shares
        one ``interaction_id`` (exposed as ``hooks.interaction_id``) and is
        posted in **one** ``/entries`` batch when the run reaches its final
        output (``on_agent_end``).

        If the run ends without a final output (an exception, an input
        guardrail tripwire, an approval interruption, a cancelled stream),
        ``on_agent_end`` never fires; ``hooks.flush()`` posts what was
        captured. It is idempotent, so calling it in ``finally`` as above is
        always safe. Until then the turn is held in memory; :meth:`close`
        posts any turn left unflushed (with a warning). An output guardrail
        runs after ``on_agent_end``, so a blocked output is still ingested.

        With ``Runner.run_streamed`` put the ``stream_events()`` loop inside
        the ``try`` — ``run_streamed`` itself returns before the run executes,
        so flushing right after it would split the turn::

            hooks = adapter.run_hooks(session_id, user_input=user_text)
            result = Runner.run_streamed(agent, user_text, hooks=hooks)
            try:
                async for event in result.stream_events():
                    ...
            finally:
                await hooks.flush()

        ``user_input`` is the new input for this turn only — a string, or a
        list of message / Responses items. Don't pass prior history: it would
        be written again (``event_id`` is not sent, so nothing is deduped).
        Omit it if you already ``push`` the user message yourself, and when
        resuming an interrupted run from its ``RunState`` (the user message
        was already recorded by the first run).

        Under ``context_mode="auto"`` the first run for a session seeds Beacon
        fast context (``on_agent_start``), exactly as the first :meth:`push`
        does. The seed is awaited before the first model call; if it fails it
        is retried on the next agent start.

        Pass ``next_hooks=`` to compose with the customer's existing ``RunHooks``.
        Both will run on every lifecycle event; the adapter's work runs first
        and swallows its own exceptions so the customer's hook always fires.

        Streaming is async-only — raises ``RuntimeError`` under ``use_async=False``.
        """
        if not self._use_async:
            raise RuntimeError(
                "OpenAIAdapter.run_hooks() requires use_async=True; the "
                "Agents SDK RunHooks interface is async-only."
            )
        cls = _build_run_hooks_class()
        return cls(self, session_id, owner_id or self._owner_id, next_hooks, user_input)

    # ─── Public API: Shutdown ─────────────────────────────────────

    async def close(self) -> None:
        """Close all session managers and shared clients. Idempotent.

        Turns captured by ``run_hooks`` but never flushed are posted first.
        """
        for hooks in list(self._unflushed_hooks):
            logger.warning(
                "[OpenAI] run_hooks turn %s was never flushed; posting it at "
                "close(). Call `await hooks.flush()` in a finally after "
                "Runner.run so a run that ends without a final output is "
                "ingested right away.", hooks.interaction_id,
            )
            await hooks.flush()
        for sid, manager in list(self._sessions.items()):
            try:
                if self._use_async:
                    await manager.close()
                else:
                    manager.close()
            except Exception as e:
                logger.warning("Error closing session %s: %s", sid[:8], e)
        self._sessions.clear()
        self._context_pack_prompt.clear()

        if self._use_async:
            await self._retrieval_client.close()
            await self._ingestion_client.close()
        else:
            self._retrieval_client.close()
            self._ingestion_client.close()

    # ─── Properties ───────────────────────────────────────────────

    @property
    def session_id(self) -> str | None:
        """Most recently accessed session ID (backward-compat convenience)."""
        return self._last_session_id

    @property
    def _last_session_id(self) -> str | None:
        if self._sessions:
            return list(self._sessions.keys())[-1]
        return None

    # ─── Internal ─────────────────────────────────────────────────

    async def _maybe_seed_fast_context(self, session_id: str, owner_id: str | None) -> None:
        """Seed Beacon fast context on a session's first turn (``auto`` only).

        Shared by :meth:`push` and the ``run_hooks`` path, so whichever runs
        first for a session seeds it.
        """
        is_open = session_id not in self._fast_context_seeded
        if not self._should_inject_fast_context(session_id, is_open=is_open):
            self._fast_context_seeded.add(session_id)
            return
        try:
            payload = await self.get_context(session_id, owner_id=owner_id)
        except Exception as e:  # noqa: BLE001
            # Fast-context seed is best-effort — a Beacon failure must
            # never break the turn. Mirrors the Claude hook.
            # Leave the session un-seeded so the next turn retries.
            logger.debug(
                "[OpenAI] auto fast-context seed failed (best-effort): %s", e,
            )
        else:
            if payload:
                self._seeded_contexts[session_id] = payload
            self._fast_context_seeded.add(session_id)

    @staticmethod
    def _collect_entries(
        messages: list[dict] | None,
        items: list[dict] | None,
    ) -> list[dict]:
        """Combine the two accepted input shapes into one entry list."""
        entries: list[dict] = []
        if messages:
            entries.extend(messages)
        if items:
            entries.extend(items)
        return entries


# ─── Module-level helpers ──────────────────────────────────────────


_CONTEXT_MODES = ("auto", "manual")


def _normalize_context_mode(mode: str) -> str:
    """Validate + canonicalize a context_mode value — same shape as the Claude adapter."""
    normalized = str(mode).strip().lower().replace("_", "-").replace(" ", "-")
    if normalized not in _CONTEXT_MODES:
        raise ValueError(
            f"context_mode must be one of {_CONTEXT_MODES!r}; got {mode!r}."
        )
    return normalized


def _normalize_item(item: Any) -> dict:
    """Serialize a Responses API output item (pydantic model or dict) to a dict."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump()
        except Exception:  # noqa: BLE001 — fall through to attr dump
            pass
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {}


def _build_run_hooks_class() -> type:
    """Lazily build ``BeaconRunHooks`` — needs ``agents.lifecycle.RunHooks``.

    Deferred so the adapter module stays importable when ``openai-agents`` is
    absent. Mirrors ``make_context_retrieval_tool``'s lazy-import pattern.
    """
    _require_sdk("run_hooks")
    import json as _json
    from uuid import uuid4 as _uuid4

    from agents.lifecycle import RunHooks  # type: ignore[import-not-found]

    # openai-agents <0.3 ships RunHooks without on_llm_start / on_llm_end.
    # The streaming path depends on both — degrade loudly instead of silently
    # dropping every assistant response, reasoning, and tool_call.
    _required = ("on_llm_start", "on_llm_end")
    _missing = [m for m in _required if not hasattr(RunHooks, m)]
    if _missing:
        raise ImportError(
            "OpenAIAdapter.run_hooks() requires openai-agents >= 0.3 "
            f"(installed version is missing {_missing} on agents.lifecycle.RunHooks). "
            "Upgrade with:\n\n    pip install -U 'openai-agents>=0.3'\n"
        )

    # Items the Agents SDK Chat Completions converter can't give a real id —
    # see transformer._SDK_PLACEHOLDER_ID. Keeping the string out of
    # _seen_item_ids avoids dropping every message after the first.
    _PLACEHOLDER_ID = "__fake_id__"

    def _real_item_id(item: dict) -> str:
        """Native item id, treating __fake_id__ and empty as absent."""
        for key in ("call_id", "id"):
            raw = item.get(key)
            if raw and str(raw) != _PLACEHOLDER_ID:
                return str(raw)
        return ""

    class BeaconRunHooks(RunHooks):
        """Captures one Agents SDK run as one Beacon turn.

        One instance is one turn for one ``(session_id, owner_id)``. Create a
        fresh one per ``Runner.run`` call via :meth:`OpenAIAdapter.run_hooks`:
        the instance owns the turn's ``interaction_id``, its record buffer and
        its dedup state (``_seen_item_ids``).
        """

        def __init__(
            self,
            adapter: Any,
            session_id: str,
            owner_id: str | None,
            next_hooks: Any = None,
            user_input: str | list[dict] | None = None,
        ) -> None:
            super().__init__()
            self._adapter = adapter
            self._session_id = session_id
            self._owner_id = owner_id
            self._next = next_hooks
            # Dedupe across repeated on_llm_end calls within one agent run. The
            # SDK gives each Responses item a stable id (or call_id); items
            # without one (reasoning w/o id) are ingested every time, and
            # nothing dedupes them downstream (event_id is not sent).
            self._seen_item_ids: set[str] = set()
            # The whole run is one turn: every record shares this id and is
            # posted in one batch by flush().
            self.interaction_id = f"turn-{session_id}-{_uuid4().hex[:12]}"
            self._pending: list[Any] = []
            # Running fallback-id idx across this turn's _buffer calls, so two
            # id-less items (Chat Completions __fake_id__, reasoning without
            # an id) in one turn don't share a local event_id.
            self._fallback_offset = 0
            if user_input:
                entries = (
                    [{"role": "user", "content": user_input}]
                    if isinstance(user_input, str) else list(user_input)
                )
                self._buffer(entries)

        def _buffer(self, items: list[dict]) -> None:
            if not items:
                return
            transformer = self._adapter._transformer
            records = transformer.transform(
                items, self._session_id,
                owner_id=self._owner_id,
                interaction_id=self.interaction_id,
                fallback_offset=self._fallback_offset,
            )
            self._fallback_offset = transformer.fallback_count
            if records:
                self._pending.extend(records)
                self._adapter._unflushed_hooks.add(self)

        async def flush(self) -> None:
            """Post the records captured so far as one batch. Idempotent."""
            records, self._pending = self._pending, []
            self._adapter._unflushed_hooks.discard(self)
            if not records:
                return
            try:
                await self._adapter.ingest(
                    self._session_id, records, owner_id=self._owner_id,
                )
            except Exception as e:  # noqa: BLE001
                # Hooks are best-effort: a Beacon failure must never break the
                # agent's run. Mirror the IngestionCallbacks contract.
                logger.warning(
                    "[OpenAI] run_hooks ingest failed (best-effort): %s", e,
                )

        async def on_llm_end(
            self, context: Any, agent: Any, response: Any,
        ) -> None:
            output = getattr(response, "output", None) or []
            items: list[dict] = []
            for raw in output:
                item = _normalize_item(raw)
                if not item:
                    continue
                item_id = _real_item_id(item)
                if item_id and item_id in self._seen_item_ids:
                    continue
                if item_id:
                    self._seen_item_ids.add(item_id)
                items.append(item)
            self._buffer(items)
            if self._next is not None:
                await self._next.on_llm_end(context, agent, response)

        async def on_tool_end(
            self, context: Any, agent: Any, tool: Any, result: object,
        ) -> None:
            # Build a function_call_output pseudo-item. The transformer resolves
            # the tool name from _tool_names (populated by the matching
            # function_call in on_llm_end), so the Trace record carries Name.
            #
            # Computer and local-shell tool callbacks receive a bare
            # RunContextWrapper with no tool_call_id. The transformer's
            # fallback event_id is slot-based (idx=0 per on_tool_end call), so
            # every empty-id invocation would share one event_id.
            # Mint a per-invocation uuid when the SDK gives us nothing so each
            # tool_result gets a distinct event_id.
            raw_call_id = getattr(context, "tool_call_id", None) or ""
            if raw_call_id and raw_call_id != _PLACEHOLDER_ID:
                call_id = raw_call_id
            else:
                call_id = f"tool-{_uuid4().hex}"
            if isinstance(result, str):
                output_text = result
            else:
                try:
                    output_text = _json.dumps(result, default=str)
                except (TypeError, ValueError):
                    output_text = str(result)
            item = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_text,
            }
            self._buffer([item])
            if self._next is not None:
                await self._next.on_tool_end(context, agent, tool, result)

        # Pass-through for the other lifecycle methods — don't drop user hooks.
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            # First run for a session seeds fast context under "auto" (the
            # adapter gates it per session, so later agents/turns no-op).
            await self._adapter._maybe_seed_fast_context(
                self._session_id, self._owner_id,
            )
            if self._next is not None:
                await self._next.on_agent_start(context, agent)

        async def on_agent_end(
            self, context: Any, agent: Any, output: Any,
        ) -> None:
            # Final output reached: the turn is complete — one POST.
            await self.flush()
            if self._next is not None:
                await self._next.on_agent_end(context, agent, output)

        async def on_handoff(
            self, context: Any, from_agent: Any, to_agent: Any,
        ) -> None:
            if self._next is not None:
                await self._next.on_handoff(context, from_agent, to_agent)

        async def on_llm_start(
            self,
            context: Any,
            agent: Any,
            system_prompt: str | None,
            input_items: list[Any],
        ) -> None:
            if self._next is not None:
                await self._next.on_llm_start(
                    context, agent, system_prompt, input_items,
                )

        async def on_tool_start(
            self, context: Any, agent: Any, tool: Any,
        ) -> None:
            if self._next is not None:
                await self._next.on_tool_start(context, agent, tool)

    return BeaconRunHooks
