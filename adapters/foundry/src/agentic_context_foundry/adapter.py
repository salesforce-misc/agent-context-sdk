# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Microsoft Agent Framework adapter for the Agentic Context SDK.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

from agentic_context import (
    CONTEXT_PACK_INSTRUCTION,
    CONTEXT_PACK_LIST_MARKER,
    CONTEXT_PACK_TIMEOUT_S,
    ApiKeyAuth,
    AsyncIngestionClient,
    AsyncRetrievalClient,
    AsyncSessionManager,
    Capabilities,
    ContextAuth,
    Record,
)
from agentic_context.retrieval import (
    CONTEXT_PACK_VERSION_GUIDANCE,
    SessionPrompt,
    context_pack_params,
    patch_pack_version_in_catalog,
    version_mismatch_entry,
)

from .transformer import RecordTransformer

logger = logging.getLogger(__name__)

_SDK_MODULE = "agent_framework"


def session_messages(session: Any) -> list[Any]:
    """Read conversation history from MAF ``AgentSession.state``.

    Default ``InMemoryHistoryProvider`` stores messages at
    ``state['in_memory']['messages']`` (the provider ``source_id`` namespace).
    ``state['messages']`` is only a fallback for tests and manually populated
    sessions.
    """
    state = getattr(session, "state", None) or {}
    if not isinstance(state, dict):
        return []
    in_memory = state.get("in_memory")
    if isinstance(in_memory, dict) and "messages" in in_memory:
        return list(in_memory.get("messages") or [])
    top = state.get("messages")
    if top:
        return list(top)
    return []

_SDK_INSTALL_HINT = "pip install agent-framework-foundry"


# Beacon ``AiAgentParticipant`` label for this runtime. Without it
# every agent lands under the SDK
# default "Custom" and analytics can't tell the frameworks apart.
_AGENT_PARTICIPANT_VARIABLE = {
    "AiAgentType": "MicrosoftAgentFramework",
    "AiAgentVersionApiName": "v1",
    "AiAgentTemplateApiName": "default",
}

def _require_sdk(method: str) -> None:
    try:
        importlib.import_module(_SDK_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"FoundryAdapter.{method}() requires the Microsoft Foundry SDK, which is "
            f"not installed. Install it with:\n\n    {_SDK_INSTALL_HINT}\n"
        ) from exc


class FoundryAdapter:
    """Microsoft Agent Framework adapter — Records in, Beacon out.

    Always seeds FAST context once per ``session_id`` on the first ``stream`` /
    ``run``, always registers the two function tools (``recall_customer_context``,
    ``get_context_pack``), and always ingests at the end of the turn.

    ``agent_participant_variable`` is a full-override for Beacon's
    ``AiAgentParticipant`` label. Defaults to
    ``AiAgentType="MicrosoftAgentFramework"`` + the adapter's ``agent_id``.
    """

    capabilities = Capabilities(working=True, persistent=False)

    def __init__(
        self,
        agent_id: str,
        *,
        owner_id: str | None = None,
        auth: ContextAuth | ApiKeyAuth | None = None,
        private_key: str = "",
        consumer_key: str = "",
        subject: str = "",
        login_url: str = "",
        secrets_api_key: str = "",
        secrets_url: str = "",
        dataspace: str = "default",
        owner_type: str | None = None,
        agent_participant_variable: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        # Full-override for Beacon's AiAgentParticipant block; None = this
        # runtime's label (see _resolve_agent_participant_variable).
        self._agent_participant_variable = (
            dict(agent_participant_variable) if agent_participant_variable is not None else None
        )
        if not owner_id:
            raise ValueError(
                "FoundryAdapter requires owner_id. Pass the end-user id (or a "
                "stable service-account id for single-tenant deployments). "
                "There is no default — a shared adapter must not attribute "
                "transcripts to a placeholder owner."
            )
        if importlib.util.find_spec(_SDK_MODULE) is None:
            warnings.warn(
                f"{_SDK_MODULE} is not installed. FoundryAdapter.stream(), run(), "
                f"snapshot(), and tools will fail until you install it:\n\n"
                f"    {_SDK_INSTALL_HINT}\n\n"
                "get_context / ingest still work without it.",
                stacklevel=2,
            )
        self._owner_id = owner_id
        self._dataspace = dataspace
        self._owner_type_override = owner_type
        self._closed = False
        self._lock: asyncio.Lock | None = None
        self._sessions: dict[str, AsyncSessionManager] = {}
        self._agent_sessions: dict[str, Any] = {}
        self._seeded: set[str] = set()
        self._turn: dict[str, int] = {}
        self._context_cache: dict[str, dict[str, Any]] = {}
        # _allowed_pack_ids is the fetched list of authorized contextPackIds
        # a given agent can utilize. _allowed_context_pack_list is a formatted
        # string to be appended to system instructions with _allowed_pack_ids
        # and their descriptions.
        # The allowed context pack list instructions is defaulted to the marker block
        # and overwritten on a successful fetch with context pack list.
        self._allowed_pack_ids: set[str] = set()
        self._allowed_context_pack_list: str = f"{CONTEXT_PACK_LIST_MARKER}\n{{}}"
        # Server-side context_pack_instruction prompt, fetched on
        # a session's first run and pinned for that session.
        self._context_pack_prompt = SessionPrompt(CONTEXT_PACK_INSTRUCTION)
        # Pack-id → integer version from the last catalog refresh. The pack tool
        # fills this in for the model when the model omits context_pack_version
        # (so the MUST declared in CONTEXT_PACK_VERSION_GUIDANCE is enforced at
        # the SDK boundary, not left as prompt-only). Updated in place when a
        # ``version_expired`` response arrives, together with the rendered
        # catalog block, so the "stale catalog cliff" is bounded to one turn.
        self._pack_versions: dict[str, int] = {}
        # Per-session turn capture (transformer + middleware). Kept off
        # AgentSession.state so to_dict()/durable stores stay JSON-native.
        self._captures: dict[str, dict[str, Any]] = {}

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
                secrets_api_key, secret_server_url=secrets_url or None,
            )
        else:
            raise ValueError(
                "FoundryAdapter requires credentials. Pass auth=ContextAuth(...), "
                "inline private_key+consumer_key+subject+login_url, or "
                "secrets_api_key= to fetch them from the secrets server."
            )

        self._owner_type = (
            self._owner_type_override
            if self._owner_type_override is not None
            else getattr(self._auth, "default_owner_type", "individual")
        )

        api_key, base_url = self._auth.get_token()
        self._ingestion_client = AsyncIngestionClient(
            base_url=base_url, api_key=api_key, dataspace=dataspace, auth=self._auth,
        )
        self._retrieval_client = AsyncRetrievalClient(
            base_url=base_url, api_key=api_key, dataspace=dataspace, auth=self._auth,
        )
        self._auth.attach(self._ingestion_client)
        self._auth.attach(self._retrieval_client)

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _guard_open(self) -> None:
        if self._closed:
            raise RuntimeError("FoundryAdapter is closed")

    def _resolve_agent_participant_variable(self) -> dict[str, Any]:
        """Beacon ``AiAgentParticipant`` block for a new manager: the caller's
        full-override if given, else this runtime's label + ``agent_id``.
        Resolved per manager, not at construction, so a late-bound
        ``agent_id`` is picked up."""
        if self._agent_participant_variable is not None:
            return dict(self._agent_participant_variable)
        return {**_AGENT_PARTICIPANT_VARIABLE, "AiAgentApiName": self.agent_id}

    async def _manager(
        self, session_id: str, owner_id: str | None = None,
    ) -> AsyncSessionManager:
        self._guard_open()
        owner = owner_id or self._owner_id
        async with self._ensure_lock():
            existing = self._sessions.get(session_id)
            if existing is not None:
                bound = getattr(existing, "_owner_id", None)
                if bound and owner and bound != owner:
                    raise ValueError(
                        f"session_id {session_id!r} is bound to owner {bound!r}; "
                        f"got {owner!r}."
                    )
                return existing
            manager = AsyncSessionManager(
                self._ingestion_client,
                agent_id=self.agent_id,
                owner_id=owner,
                owner_type=self._owner_type,
                session_id=session_id,
                retrieval_client=self._retrieval_client,
                agent_participant_variable=self._resolve_agent_participant_variable(),
            )
            self._sessions[session_id] = manager
            return manager

    async def _claim_seed(self, session_id: str) -> bool:
        async with self._ensure_lock():
            if session_id in self._seeded:
                return False
            self._seeded.add(session_id)
            return True

    async def start_session(
        self, session_id: str, *, owner_id: str | None = None, title: str = "",
    ) -> dict[str, str]:
        """Optional eager Beacon lifecycle. ``stream`` does not require it."""
        manager = await self._manager(session_id, owner_id)
        owner = getattr(manager, "_owner_id", None) or owner_id or self._owner_id
        try:
            result = await manager.create_session(session_id, title=title)
        except Exception:
            logger.warning(
                "[Foundry] start_session: owner=%r session_id=%r outcome=denied",
                owner, session_id,
            )
            raise
        logger.info(
            "[Foundry] start_session: owner=%r session_id=%r outcome=success",
            owner, session_id,
        )
        return result

    async def get_context(
        self, session_id: str | None = None, *, owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """FAST profile read. Best-effort — ``None`` on failure."""
        if not session_id:
            return None
        try:
            manager = await self._manager(session_id, owner_id)
            payload = await manager.get_context()
        except Exception as exc:
            logger.warning("[Foundry] get_context failed (best-effort): %s", exc)
            return None
        if payload is not None:
            self._context_cache[session_id] = payload
        return payload

    async def recall_customer_context(
        self,
        *,
        session_id: str | None = None,
        owner_id: str | None = None,
        topic: str = "",
    ) -> dict[str, Any]:
        """Tool body: FAST profile read only. Never raises."""
        payload = await self.get_context(session_id, owner_id=owner_id)
        logger.info(
            "[Foundry] recall_customer_context: owner=%r session_id=%r "
            "topic=%r → has_context=%s",
            owner_id or self._owner_id,
            session_id,
            topic,
            bool(payload and payload.get("userContext")),
        )
        return payload or {"userContext": []}

    def make_get_context_tool(
        self, session_id: str, owner_id: str | None = None,
    ) -> Any:
        _require_sdk("make_get_context_tool")
        from agent_framework import tool

        adapter = self

        @tool(
            name="recall_customer_context",
            approval_mode="never_require",
            # LLM-visible description — drives *when* the model reaches for memory.
            description=(
                "Look up what we already know about this customer from past "
                "conversations (cross-session memory) — the customer's stored "
                "profile facts: plan or account details, issues raised before, "
                "stated preferences, or anything they'd expect you to remember. "
                "Call this when answering would benefit from memory the current "
                "conversation hasn't covered yet. You do NOT need to call it for "
                "small talk or when the current conversation already holds the "
                "answer."
            ),
        )
        async def recall_customer_context(topic: str = "") -> dict[str, Any]:
            return await adapter.recall_customer_context(
                session_id=session_id, owner_id=owner_id, topic=topic,
            )

        return recall_customer_context

    async def get_context_pack_list(self) -> dict[str, Any] | None:
        """Fetch this agent's available CONTEXT PACK list (get_context_pack_list).

        Best-effort — returns ``None`` when the runtime-context-packs read fails.
        Does not write ``_allowed_pack_ids`` / ``_allowed_context_pack_list``;
        session-start injection is the only writer of those caches.
        """
        try:
            return await self._retrieval_client.get_context_pack_list(agent_id=self.agent_id)
        except Exception as e:
            logger.warning(
                "[Foundry] get_context_pack_list failed (best-effort); "
                "returning None: %s: %s", type(e).__name__, e,
            )
            return None

    def _update_context_pack_list(self, new_context_pack_list: dict[str, Any]) -> None:
        new_context_pack_id_map: dict[str, Any] = {}
        new_pack_versions: dict[str, int] = {}
        for pack in new_context_pack_list["contextPacks"]:
            pack_id = pack["contextPackId"]
            entry = {
                "description": pack["description"],
                "dataNeeds": [
                    {
                        "name": need["name"],
                        "dataNeedDescription": need.get("description") or "",
                        "parameters": need.get("parameters") or [],
                    }
                    for need in pack.get("dataNeeds") or []
                ],
            }
            # Surface `version` to the LLM only when it is a usable int. Null
            # or non-int values from Beacon are dropped from both the rendered
            # block and the SDK-side default, so a malformed catalog entry
            # never fabricates a pin.
            version = pack.get("version")
            if isinstance(version, int) and not isinstance(version, bool):
                entry["version"] = version
                new_pack_versions[pack_id] = version
            new_context_pack_id_map[pack_id] = entry
        self._allowed_pack_ids = set(new_context_pack_id_map)
        self._allowed_context_pack_list = (
            f"{CONTEXT_PACK_LIST_MARKER}\n{json.dumps(new_context_pack_id_map)}"
        )
        self._pack_versions = new_pack_versions

    async def _maybe_inject_context_pack_list(self, *, is_open: bool = False) -> None:
        """Fetches and caches this agent's authorized CONTEXT PACK list.

        Caches the fetched CONTEXT PACK list into ``_allowed_pack_ids``
        to reuse on every turn after that session's open. Refetches this list
        again at every session-start. On API failure the shared instance
        variables for context pack lists are not updated.
        """
        if not is_open:
            return
        try:
            fetched_context_pack_list = await self._retrieval_client.get_context_pack_list(
                agent_id=self.agent_id,
            )
            self._update_context_pack_list(fetched_context_pack_list)
        except Exception as e:
            logger.warning(
                "[Foundry] context pack list fetch failed: %s: %s",
                type(e).__name__, e,
            )

    async def get_context_pack(
        self,
        context_pack_id: str,
        *,
        session_id: str,
        owner_id: str | None = None,
        params: Any = None,
        param_defaults: dict[str, Any] | None = None,
        timeout: float | None = CONTEXT_PACK_TIMEOUT_S,
        valid_pack_ids: list[str] | None = None,
        context_pack_version: int | None = None,
    ) -> dict[str, Any]:
        allowed = set(valid_pack_ids) if valid_pack_ids is not None else self._allowed_pack_ids
        if context_pack_id not in allowed:
            logger.info("[Foundry] get_context_pack: invalid pack id %r", context_pack_id)
            return {
                "status": "invalid_pack",
                "context_pack_id": context_pack_id,
                "valid_pack_ids": sorted(allowed),
                "message": (
                    f"'{context_pack_id}' is not an available context pack. "
                    f"Valid pack ids: {', '.join(sorted(allowed))}."
                ),
            }

        def not_ready(reason: str) -> dict[str, Any]:
            logger.info(
                "[Foundry] get_context_pack: %r → not_ready (%s)",
                context_pack_id, reason,
            )
            return {
                "status": "not_ready",
                "context_pack_id": context_pack_id,
                "message": (
                    "Context pack retrieval is a preview capability and is not "
                    "available in this environment yet. Fall back to the context "
                    "you already have."
                ),
                "detail": reason,
            }

        try:
            manager = await self._manager(session_id, owner_id)
        except Exception as exc:
            return not_ready(f"session failed: {type(exc).__name__}")
        owner = getattr(manager, "_owner_id", None) or owner_id or self._owner_id
        if not owner:
            return not_ready("no owner for this turn")
        resolved_params = context_pack_params(params, "", param_defaults or {})
        # Enforce CONTEXT_PACK_VERSION_GUIDANCE at the SDK boundary: when the
        # model omits ``context_pack_version`` and the cached catalog has one
        # for this pack, pin the request. The prompt still asks the model to
        # echo the version back, but the pin lands on the wire even when it
        # does not.
        if context_pack_version is None:
            context_pack_version = self._pack_versions.get(context_pack_id)
        try:
            payload = await self._retrieval_client.get_context_pack(
                session_id,
                owner,
                self.agent_id,
                context_pack_id,
                params=resolved_params,
                owner_type=self._owner_type,
                context_pack_version=context_pack_version,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning(
                "[Foundry] get_context_pack: %r request failed",
                context_pack_id, exc_info=True,
            )
            return not_ready(f"request failed: {type(exc).__name__}")

        mismatch = version_mismatch_entry(payload)
        if mismatch is not None:
            latest = mismatch["version"]
            # Patch the cached catalog so a next turn reading
            # ``_allowed_context_pack_list`` sees the fresh version and the
            # SDK-side default resolves to it instead of the stale pin. When
            # ``latest`` is ``None`` (Beacon reports no current revision),
            # drop the pin from both cache and catalog so the retry goes
            # unpinned instead of re-sending the stale version.
            if isinstance(latest, int) and not isinstance(latest, bool):
                self._pack_versions[context_pack_id] = latest
            else:
                self._pack_versions.pop(context_pack_id, None)
            self._allowed_context_pack_list = patch_pack_version_in_catalog(
                self._allowed_context_pack_list, context_pack_id, latest,
            )
            logger.info(
                "[Foundry] get_context_pack: %r → version_expired "
                "(requested=%s, latest=%s)",
                context_pack_id,
                context_pack_version,
                latest,
            )
            return {
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
            }

        logger.info("[Foundry] get_context_pack: %r → ok", context_pack_id)
        return {"status": "ok", "context_pack_id": context_pack_id, "context": payload}

    def make_get_context_pack_tool(
        self,
        session_id: str,
        owner_id: str | None = None,
        *,
        valid_pack_ids: list[str] | None = None,
        param_defaults: dict[str, Any] | None = None,
        timeout: float | None = CONTEXT_PACK_TIMEOUT_S,
    ) -> Any:
        _require_sdk("make_get_context_pack_tool")
        from agent_framework import tool

        adapter = self

        @tool(
            name="get_context_pack",
            approval_mode="never_require",
            # LLM-visible description — drives *when* the model reaches for a pack
            # and how it fills parameters.
            description=(
                "Retrieve one named context pack for the current subject. This is the "
                "same read as getContext(contextPack).\n\n"
                "The \"[Available context packs]\" block lists each pack's description "
                "and its data needs. Each data need has a description and a parameters "
                "list. If a pack or one of its data needs would help answer the user's "
                "question, call this tool with that pack's id as `context_pack_id`.\n\n"
                "Pass parameter values in `params`, a JSON object whose keys are the "
                "parameter names from those data needs. Use a value you can determine "
                "from the conversation. If you cannot determine a value, ask the user, "
                "or omit that key. Do not invent a value and do not send an empty "
                "string. A data need that declares an omitted parameter may be absent "
                "from the result; answer from the data needs that did return.\n\n"
                + CONTEXT_PACK_VERSION_GUIDANCE
            ),
        )
        # Defaults + params: Any so a model call with {} or with params as a
        # JSON object (not a string) does not die in MAF pydantic validation
        # ("Argument parsing failed") before our handler runs.
        async def get_context_pack(
            context_pack_id: str = "",
            params: Any = "",
            context_pack_version: int | None = None,
        ) -> dict[str, Any]:
            return await adapter.get_context_pack(
                context_pack_id,
                session_id=session_id,
                owner_id=owner_id,
                params=params,
                param_defaults=param_defaults,
                timeout=timeout,
                valid_pack_ids=valid_pack_ids,
                context_pack_version=context_pack_version,
            )

        return get_context_pack

    async def get_context_pack_instruction(self, session_id: str) -> str:
        """Return the context-pack instruction pinned for ``session_id``.

        :meth:`run` / :meth:`stream` (via the context provider) already add
        this to the run's instructions every turn, so most integrations
        never call it. The first call for a session fetches the
        server-published prompt
        (``GET /api/v1/agentic/prompts/system/context_pack_instruction``);
        a miss or failure pins the bundled
        :data:`agentic_context.CONTEXT_PACK_INSTRUCTION` for that session.
        """
        return await self._context_pack_prompt.get(session_id, self._retrieval_client)

    def _agent_instructions_text(self, agent: Any) -> str:
        """The agent's own instructions as one string (for duplicate checks)."""
        existing = None
        options = getattr(agent, "default_options", None)
        if isinstance(options, dict):
            existing = options.get("instructions")
        if not existing:
            existing = getattr(agent, "instructions", None)
        if isinstance(existing, (list, tuple)):
            return "\n\n".join(str(item) for item in existing if item)
        return str(existing) if existing else ""

    def context_provider(self, *, owner_id: str | None = None) -> Any:
        """
        FoundryContextProvider manages instructions/tools/middleware merging
        and calls ``after_run`.
        """
        _require_sdk("context_provider")
        from .context_provider import FoundryContextProvider

        return FoundryContextProvider(self, owner_id=owner_id)

    def _provider_on_agent(self, agent: Any) -> Any | None:
        for provider in getattr(agent, "context_providers", None) or []:
            if getattr(provider, "_adapter", None) is self:
                return provider
        return None

    def _merged_run_instructions(self, agent: Any, extras: Any) -> str | None:
        extra_parts = [str(item) for item in (extras or []) if item]
        if not extra_parts:
            return None
        existing = self._agent_instructions_text(agent)
        return "\n\n".join([existing, *extra_parts] if existing else extra_parts)

    def default_tools(
        self,
        session_id: str,
        owner_id: str | None = None,
        *,
        valid_pack_ids: list[str] | None = None,
    ) -> list[Any]:
        return [
            self.make_get_context_tool(session_id, owner_id),
            self.make_get_context_pack_tool(
                session_id, owner_id, valid_pack_ids=valid_pack_ids,
            ),
        ]

    def _format_context(self, context_data: dict[str, Any] | None) -> str:
        if not context_data:
            return ""
        entries: list[str] = []
        for item in context_data.get("userContext") or []:
            entry = item.get("entry") if isinstance(item, dict) else None
            if entry:
                entries.append(str(entry).strip())
        if not entries:
            return ""
        return "[Cross-session context]\n" + "\n".join(f"- {e}" for e in entries)

    def snapshot(self, session_id: str, *, session: Any = None) -> list[Record]:
        """Read stored conversation history in chronological order.
        """
        _require_sdk("snapshot")
        agent_session = session if session is not None else self._agent_sessions.get(session_id)
        if agent_session is None:
            raise KeyError(
                f"no AgentSession stored for {session_id!r}; pass session=... "
                "to snapshot a session rehydrated from persisted storage"
            )
        messages = session_messages(agent_session)
        owner = getattr(self._sessions.get(session_id), "_owner_id", None) or self._owner_id
        transformer = RecordTransformer(self.agent_id, owner)
        interaction_id = f"turn-{session_id}-snapshot"
        records: list[Record] = []
        for index, message in enumerate(messages):
            records.extend(
                transformer.from_message(message, session_id, interaction_id, index)
            )
        return records

    async def stream(
        self,
        agent: Any,
        messages: Any,
        *,
        session: Any,
        owner_id: str | None = None,
    ) -> AsyncIterator[Record]:
        """Call ``agent.run(..., stream=True)`` and produce Records during each model turn.
        """
        _require_sdk("stream")
        self._guard_open()
        from agent_framework import SessionContext, normalize_messages

        session_id = getattr(session, "session_id", None)
        if not session_id:
            raise ValueError("AgentSession.session_id is required")
        owner = owner_id or self._owner_id
        incoming = normalize_messages(messages)
        attached = self._provider_on_agent(agent)

        kwargs: dict[str, Any] = {"stream": True, "session": session}
        if attached is not None and owner_id is not None:
            session.state.setdefault(attached.source_id, {})["owner_override"] = owner
        if attached is None:
            provider = self.context_provider(owner_id=owner)
            context = SessionContext(
                session_id=session_id,
                service_session_id=getattr(session, "service_session_id", None),
                input_messages=incoming,
            )
            state = session.state.setdefault(provider.source_id, {})
            await provider.before_run(
                agent=agent, session=session, context=context, state=state,
            )
            capture = self._captures[session_id]
            capture["skip_after_run_ingest"] = True
            kwargs["tools"] = context.tools
            kwargs["middleware"] = context.get_middleware()
            merged = self._merged_run_instructions(agent, context.instructions)
            if merged:
                kwargs["options"] = {"instructions": merged}

        seen: set[str] = set()
        collected: list[Record] = []
        yielded_inputs = False

        def take(records: list[Record]) -> list[Record]:
            fresh: list[Record] = []
            for record in records:
                if record.event_id in seen:
                    continue
                seen.add(record.event_id)
                collected.append(record)
                fresh.append(record)
            return fresh

        def capture_bag() -> dict[str, Any]:
            return self._captures.get(session_id) or {}

        stream_error: BaseException | None = None
        try:
            if attached is None:
                for record in take(capture_bag().get("input_records") or []):
                    yield record
                    yielded_inputs = True

            response_stream = agent.run(incoming, **kwargs)
            index = len(session_messages(session)) + len(incoming)
            async for update in response_stream:
                bag = capture_bag()
                if attached is not None and not yielded_inputs:
                    for record in take(bag.get("input_records") or []):
                        yield record
                    yielded_inputs = True
                transformer = bag.get("transformer")
                interaction_id = bag.get("interaction_id") or f"turn-{session_id}-0"
                if transformer is not None:
                    for record in take(
                        transformer.from_update(
                            update, session_id, interaction_id, index, tools_only=True,
                        )
                    ):
                        yield record
                index += 1

            bag = capture_bag()
            if not yielded_inputs:
                for record in take(bag.get("input_records") or []):
                    yield record
            transformer = bag.get("transformer")
            interaction_id = bag.get("interaction_id") or f"turn-{session_id}-0"
            get_final = getattr(response_stream, "get_final_response", None)
            if transformer is not None and callable(get_final):
                try:
                    final = await get_final()
                except Exception:
                    logger.exception("[Foundry] get_final_response failed")
                    final = None
                messages = getattr(final, "messages", None) or []
                for offset, message in enumerate(messages):
                    for record in take(
                        transformer.from_message(
                            message, session_id, interaction_id, index + offset,
                        )
                    ):
                        yield record
            chat_mw = bag.get("chat_mw")
            func_mw = bag.get("func_mw")
            leftover: list[Record] = []
            if chat_mw is not None:
                leftover.extend(chat_mw.records)
            if func_mw is not None:
                leftover.extend(func_mw.records)
            for record in take(leftover):
                yield record
        except BaseException as exc:
            stream_error = exc
            raise
        finally:
            try:
                if attached is None and collected:
                    manager = await self._manager(session_id, owner)
                    await manager.ingest(collected)
                    await manager.flush()
            except Exception:
                logger.exception("[Foundry] ingest failed")
                if stream_error is None:
                    raise
            finally:
                self._captures.pop(session_id, None)

    async def run(
        self,
        agent: Any,
        messages: Any,
        *,
        session: Any,
        owner_id: str | None = None,
    ) -> list[Record]:
        """Closes 'stream' into a list."""
        _require_sdk("run")
        return [
            record
            async for record in self.stream(
                agent, messages, session=session, owner_id=owner_id,
            )
        ]

    async def close(self) -> None:
        async with self._ensure_lock():
            if self._closed:
                return
            self._closed = True
            managers = list(self._sessions.values())
            self._sessions.clear()
            self._context_pack_prompt.clear()
        for manager in managers:
            try:
                await manager.close()
            except Exception:
                logger.exception("[Foundry] manager.close failed")
        try:
            await self._ingestion_client.close()
        except Exception:
            logger.exception("[Foundry] ingestion client close failed")
        try:
            await self._retrieval_client.close()
        except Exception:
            logger.exception("[Foundry] retrieval client close failed")
