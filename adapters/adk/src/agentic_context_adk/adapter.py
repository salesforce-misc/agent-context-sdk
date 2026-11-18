# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Google ADK adapter — builds Records from ADK callbacks and ships via AsyncSessionManager.

Buffers all records for a turn and flushes the full batch at turn end
(after_model with is_tool_call=False) via an inline ``await`` on the same
turn task, so records are guaranteed delivered before the turn returns.

Multi-user safety: one adapter instance is typically shared across all users
(a module-level singleton in the host app). Per-turn owner identity is therefore
scoped to the running asyncio task via a ContextVar, and each ADK session gets its
own AsyncSessionManager (managers share the two HTTP clients but never share the
mutable owner/session state). Concurrent interleaved turns for different users
cannot stomp each other's identity. See ``_current_session_id`` and ``_manager_for``.

Usage (shared ContextAuth instance — recommended when sharing auth across adapters):
    auth = ContextAuth(private_key=..., consumer_key=..., subject=..., login_url=...)
    adapter = GoogleAdkAdapter(agent_id="my-agent", auth=auth)

Usage (inline credentials — SDK builds ContextAuth internally):
    adapter = GoogleAdkAdapter(
        agent_id="my-agent",
        private_key=Path("server.key").read_text(),
        consumer_key="3MVG...",
        subject="user@org.com",
        login_url="login.salesforce.com",
    )

    adapter.register_shutdown()  # wire SIGINT/SIGTERM in one line

    agent = LlmAgent(
        ...
        **adapter.callback_kwargs(),  # all 4 callbacks in one call
    )

    try:
        async for event in runner.run_async(...):
            pass
    finally:
        await adapter.close()
"""
from __future__ import annotations

import asyncio
import contextvars
import importlib
import importlib.util
import inspect
import json
import logging
import os
import signal
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from time import monotonic, time
from typing import Any, Callable, Literal
from uuid import uuid4

from agentic_context import (
    CONTEXT_PACK_INSTRUCTION,
    CONTEXT_PACK_TIMEOUT_S,
    CONTEXT_PACK_LIST_MARKER,
    ApiKeyAuth,
    AsyncIngestionClient,
    AsyncRetrievalClient,
    AsyncSessionManager,
    ContextAuth,
    Capabilities,
    Record,
)
from agentic_context.retrieval import (
    CONTEXT_PACK_VERSION_GUIDANCE,
    SessionPrompt,
    context_pack_params,
    patch_pack_version_in_catalog,
    version_mismatch_entry,
)

logger = logging.getLogger(__name__)

# An HTTP observer is an opt-in callback invoked for every Beacon HTTP call the
# adapter makes (ingest + retrieve). It receives one event dict per phase
# (request / response) and is used to surface live Beacon traffic in a demo UI's
# API console. Sync or async; it must never raise (exceptions are swallowed at
# DEBUG). Mirrors the ClaudeAdapter's http_observer surface so all adapters
# expose the same observability contract.
HttpObserver = Callable[[dict], Any]

# The framework SDK this adapter drives. It is NOT a dependency of
# agentic-context (you already run Google ADK); it is imported lazily inside the
# methods that touch it (the tool factories), so constructing the adapter and
# using the pure-ingest path work without it. This is the one place the install
# hint lives.
_SDK_MODULE = "google.adk"
_SDK_INSTALL_HINT = "pip install google-adk"


# Beacon ``AiAgentParticipant`` label for this runtime. Without it
# every agent lands under the SDK
# default "Custom" and analytics can't tell the frameworks apart.
_AGENT_PARTICIPANT_VARIABLE = {
    "AiAgentType": "GoogleADK",
    "AiAgentVersionApiName": "v1",
    "AiAgentTemplateApiName": "default",
}


def _require_sdk(method: str) -> None:
    """Fail with a single, actionable message if ``google.adk`` is absent.

    Call at the top of every method that imports the SDK. Turns the raw
    ``ModuleNotFoundError`` (which otherwise surfaces mid-conversation with no
    remediation) into a consistent ImportError that names the exact package to
    install — matching the top-level façade's behaviour for LangGraph. The
    original error is chained as ``__cause__`` for debuggability.
    """
    try:
        importlib.import_module(_SDK_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"GoogleAdkAdapter.{method}() requires the Google ADK, which is not "
            f"installed. Install it with:\n\n    {_SDK_INSTALL_HINT}\n"
        ) from exc


# The ADK session id of the turn running in the current asyncio task.
#
# One GoogleAdkAdapter instance is usually shared across every user (a
# module-level singleton in the host app), and ADK runs each turn as its own
# asyncio task on a single event loop, interleaving them at await points. A
# ContextVar is task-local and is inherited (as a snapshot) by the child tasks
# ADK spawns to run tools, so it lets every read of "who is this turn for" bind
# to the right session without one turn stomping another's identity on shared
# adapter state. It carries only the session id (not adapter-specific data), so
# a single module-level var is safe; per-adapter isolation comes from the
# per-instance owner maps that this id keys into.
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_context_adk_session_id", default=None
)

# The get_context_pack(context_pack_id) agent guidance and the pack-POST timeout default
# are shared, framework-neutral constants that live in the core package
# (agentic_context.retrieval) so every adapter — Claude / ADK / LangGraph — and
# raw QUICKSTART users all reference the SAME text. Imported above and re-exported
# from this package's __init__ for backward compatibility; the local alias keeps
# the method default (``timeout=_CONTEXT_PACK_TIMEOUT_S``) unchanged.
_CONTEXT_PACK_TIMEOUT_S = CONTEXT_PACK_TIMEOUT_S
_CONTEXT_PACK_LIST_MARKER = CONTEXT_PACK_LIST_MARKER


def _resolve_context_mode(mode: str) -> bool:
    """Facade ``context_mode`` → the ``inject_at_open`` injection flag.

    Returns whether this mode seeds owner-scoped profile context at
    conversation-open. The recall tool is installed in every mode.

    - ``"auto"``     → seed context once at open; the agent drives further
                       reads via the recall tool.
    - ``"manual"``   → no auto-injection at all; the recall tool is still
                       registered, the customer owns their own prompting.
    """
    if mode == "auto":
        return True
    if mode == "manual":
        return False
    raise ValueError(
        f"context_mode must be one of 'auto' or 'manual'; got {mode!r}."
    )


class GoogleAdkAdapter:
    """Google ADK adapter.

    Converts ADK callbacks to Record objects and ships them to Beacon
    via AsyncSessionManager at the end of each turn.

    Requires ContextAuth credentials (auth= or inline private_key/consumer_key/
    subject/login_url). Tests may inject a session_manager directly to skip HTTP.
    """

    capabilities = Capabilities(working=True, persistent=True)

    def __init__(
        self,
        agent_id: str,
        # Optional. The customer's identity is per-conversation, not per-adapter,
        # so the modern path is to leave this unset and supply owner_id at
        # conversation-open via adapter.hooks(on_conversation_start=...). Passing
        # it here sets the single-owner FALLBACK (bootstrap / single-user / tests
        # / direct calls made outside any turn); it is still accepted without
        # warning. When left None and no hook/auto-detect resolves an owner, the
        # adapter no-ops its Beacon writes/reads for that session rather than
        # attributing them to a placeholder. (Was `"placeholder"` pre-W2.)
        owner_id: str | None = None,
        *,
        # Pass a shared auth instance (recommended when sharing auth across
        # adapters). ContextAuth = production (direct creds); ApiKeyAuth = the demo
        # api-key path (downloads the bundle from a secret server, then behaves
        # exactly like ContextAuth). Both expose the same token/attach surface.
        auth: ContextAuth | ApiKeyAuth | None = None,
        # Credential-based auth — SDK manages token exchange and refresh
        private_key: str = "",
        consumer_key: str = "",
        subject: str = "",
        login_url: str = "",
        # API-key auth — the demo / platform-hosted path. Pass the single
        # per-agent key issued by Context Builder and the adapter builds an
        # ApiKeyAuth for you: on the first turn it downloads the credential
        # bundle (privateKey/consumerKey/subject/loginUrl/agentId) from the
        # secret server, then behaves exactly like ContextAuth. This is the
        # "customer only gives you the API key" entry point — no wrapper, no
        # token pre-fetch. secret_server_url may be omitted, in which case
        # ApiKeyAuth reads AGENTIC_CONTEXT_SECRET_SERVER_URL; there is no
        # baked-in host, so one of the two must be set (fails closed otherwise).
        agent_api_key: str = "",
        secret_server_url: str = "",
        dataspace: str | None = None,
        owner_type: str | None = None,
        # The one context-behaviour selector. One value picks a whole behaviour;
        # internally it expands into a single ``inject_at_open`` flag, which is a
        # private implementation detail — context_mode is the whole public
        # surface (runtime-switchable via set_context_mode).
        #   "auto"     — seed context once at conversation-open, then the agent
        #                decides each turn whether it needs more and calls the
        #                recall tool. Default.
        #   "manual"   — no auto-injection; the recall tool is still registered,
        #                but the customer owns their own prompting.
        context_mode: Literal["auto", "manual"] = "auto",
        # Optional callable invoked for every Beacon HTTP call (ingest +
        # retrieve). It receives two event dicts per call (a "request" then a
        # "response" phase, correlated by a shared callId) carrying method,
        # path-only url, status, latencyMs, and decoded+capped request/response
        # bodies — but NEVER headers, so the bearer token can't leak to a UI.
        # Sync or async; it must never raise (exceptions are swallowed at DEBUG).
        # When set, the adapter builds its own httpx clients pre-wired with
        # observability hooks and injects them via the core clients' `client=`
        # seam; when None (default) the adapter behaves exactly as before. This
        # is the demo-observability surface that lets a downstream shell render
        # live Beacon traffic (Get Context / Add Entries / token exchange) in an
        # API console. Mirrors ClaudeAdapter.http_observer. Unlike Claude, ADK
        # has no sync mode, so http_observer is always honoured (no warning).
        http_observer: HttpObserver | None = None,
        # Full-override for Beacon's AiAgentParticipant block.
        # None = AiAgentType="GoogleADK" + agent_id. Not combinable with an
        # injected session_manager, which carries its own.
        agent_participant_variable: dict[str, Any] | None = None,
        # Injection points for tests
        session_manager: Any = None,
        retrieval_client: AsyncRetrievalClient | None = None,
    ) -> None:
        if session_manager is not None and agent_participant_variable is not None:
            raise ValueError(
                "agent_participant_variable= has no effect with an injected "
                "session_manager=; pass it to your SessionManager instead"
            )
        self.agent_id = agent_id
        # Full-override for Beacon's AiAgentParticipant block; None = this
        # runtime's label (see _resolve_agent_participant_variable).
        self._agent_participant_variable = (
            dict(agent_participant_variable) if agent_participant_variable is not None else None
        )
        # Soft, non-fatal nudge: the Google ADK is imported lazily inside the
        # tool factories (make_start_session_tool / make_context_retrieval_tool), so
        # construction and the pure-ingest/callback path work without it. But if
        # it is missing, warn HERE — at a predictable point — rather than letting
        # the failure surface mid-turn deep inside a tool factory. find_spec does
        # not import the module, so this is cheap and side-effect-free. Non-fatal
        # by design: partial use (callbacks/ingest, no tools) is supported.
        if importlib.util.find_spec(_SDK_MODULE) is None:
            warnings.warn(
                f"{_SDK_MODULE} is not installed. GoogleAdkAdapter's "
                f"make_start_session_tool() and make_context_retrieval_tool() will fail "
                f"until you install it:\n\n    {_SDK_INSTALL_HINT}\n\n"
                "The per-turn callbacks and ingestion path work without it.",
                stacklevel=2,
            )
        # Per-turn owner identity is resolved via the _current_session_id
        # ContextVar against the per-session owner map; _default_owner is the
        # fallback for calls made outside any turn (bootstrap / single-user /
        # direct test calls). The owner_id setter below routes here.
        self._default_owner = owner_id
        self._session_owners: dict[str, str] = {}
        # _allowed_pack_ids is the fetched list of authorized contextPackIds
        # a given agent can utilize. _allowed_context_pack_list is a formatted
        # string to be appended to system instructions with _allowed_pack_ids
        # and their descriptions.
        # The allowed context pack list instructions is defaulted to the marker block
        # and overwritten on a successful fetch with context pack list.
        self._allowed_pack_ids: set[str] = set()
        self._allowed_context_pack_list: str = f"{_CONTEXT_PACK_LIST_MARKER}\n{{}}"
        # Cached pack-id -> catalog version. Populated from the fetched catalog
        # (ints only, never nulls or bools) so the get_context_pack handler can
        # fill it in for the model when the model omits context_pack_version
        # (enforcing the MUST declared in CONTEXT_PACK_VERSION_GUIDANCE at the
        # SDK boundary). Patched on status=version_expired so subsequent calls
        # pin to the new revision without a catalog refresh.
        self._pack_versions: dict[str, int] = {}
        # W2b — per-conversation config surface. `on_conversation_start` (set via
        # `hooks()`) is the customer-supplied callback invoked once at each
        # conversation-open; whatever mapping it returns is stashed per session
        # in `_session_configs`, mirroring `_session_owners`. Recognised keys:
        # owner_id (bound via set_owner), dataspace, system_prompt_override, and
        # a per-conversation context_mode (expanded into the private injection
        # flags at merge — see _merge_session_config). Empty until a hook is
        # registered, so this is a pure no-op for callers that never touch hooks().
        self._on_conversation_start: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None
        self._session_configs: dict[str, dict[str, Any]] = {}
        self._owner_type_override = owner_type
        # Opt-in HTTP observability. When set, _ensure_clients builds
        # observer-wired httpx clients and tracks them in _owned_http_clients so
        # close() can aclose() them (the core clients won't, since we inject
        # them via client= → their _owns_client is False).
        self._http_observer = http_observer

        # context_mode is the whole public surface; _resolve_construction_mode
        # expands it into the private ``_inject_at_open`` flag the rest of the
        # adapter reads (_should_inject_fast_context, wrap_before_model). See
        # that helper for the auto/manual truth table.
        self._context_mode, self._inject_at_open = (
            self._resolve_construction_mode(context_mode)
        )
        self._agent_participant_id = agent_id
        # Mirror LangGraphAdapter: explicit kwarg wins, else CDP_DATASPACE
        # env var, else "default". Treats empty strings as missing — an
        # empty `dataspace` header is a silent footgun at the wire level.
        dataspace = dataspace or os.environ.get("CDP_DATASPACE") or "default"
        self._dataspace = dataspace

        if auth is not None:
            self._auth: ContextAuth | ApiKeyAuth | None = auth
        elif agent_api_key:
            # Demo / platform-hosted path: build ApiKeyAuth from the single key.
            # secret_server_url="" lets ApiKeyAuth fall back to its env var; it
            # fails closed on first fetch if neither is set. The bundle carries
            # the agentId, but the adapter still uses the agent_id passed above
            # (explicit wins; the bundle's is a fallback resolved post-fetch via
            # auth.agent_id — see the DEVELOPER_GUIDE api-key section).
            self._auth = ApiKeyAuth(
                agent_api_key,
                secret_server_url=secret_server_url or None,
            )
        elif private_key and consumer_key and subject and login_url:
            self._auth = ContextAuth(
                private_key=private_key,
                consumer_key=consumer_key,
                subject=subject,
                login_url=login_url,
            )
        elif session_manager is None:
            raise ValueError(
                "GoogleAdkAdapter requires auth. Pass agent_api_key=\"<key>\" (demo "
                "api-key path), auth=ContextAuth(...)/ApiKeyAuth(...), or inline "
                "private_key/consumer_key/subject/login_url. "
                "For tests, inject session_manager= directly."
            )
        else:
            self._auth = None

        # owner_type defaults to the auth class's advertised value (individual
        # for service-account auth, employee for interactive user auth) when the
        # caller doesn't pass one explicitly.
        self._owner_type = (
            self._owner_type_override
            if self._owner_type_override is not None
            else getattr(self._auth, "default_owner_type", "individual")
        )

        # Per-session AsyncSessionManager instances. Managers share the two HTTP
        # clients (built once in _ensure_clients) but each owns its own
        # owner/session state, so concurrent turns for different users never
        # stomp one another. Keyed by ADK session id; LRU-capped to bound memory
        # for long-lived servers (mirrors PR #20's _SHARED_POLICY_STATE fix).
        self._managers: OrderedDict[str, AsyncSessionManager] = OrderedDict()
        self._MGR_CAP = 1024
        # Strong refs to in-flight manager.close() tasks scheduled on eviction,
        # so the event loop can't GC them mid-drain (bare create_task is only
        # weakly referenced). Discarded via done-callback when each completes.
        self._eviction_tasks: set[Any] = set()

        if session_manager is not None:
            # Single-manager / test / advanced mode: one injected manager backs
            # every session. This mode is NOT multi-user race-safe (the injected
            # manager's owner state is shared); it is intended for single-owner
            # tests and callers that manage their own manager. Production uses
            # the auth path below, which builds a distinct manager per session.
            self._injected_mgr: AsyncSessionManager | None = session_manager
            self._client: AsyncIngestionClient | None = None
            self._retrieval = retrieval_client
        else:
            # Clients will be initialised on first turn once we have a token + base_url
            self._injected_mgr = None
            self._client = None
            self._retrieval = None

        # httpx clients we built + injected (observer path). The core clients set
        # _owns_client=False when handed a client=, so they won't close these —
        # we close them ourselves in close(). Empty unless http_observer is set.
        self._owned_http_clients: list[Any] = []
        # Server-side context_pack_instruction prompt, fetched on
        # a session's first model call and pinned for that session.
        self._context_pack_prompt = SessionPrompt(CONTEXT_PACK_INSTRUCTION)
        # Serialises first-turn client construction. The adapter is a shared
        # singleton with interleaved per-user turns; without this, two turns that
        # both reach their first turn before either finishes aget_token() would
        # each build a client pair and the second would orphan the first — a
        # leaked httpx socket pair on the observer path. Created lazily inside
        # _ensure_clients (a Lock must bind to the running event loop, which may
        # not exist at construction), guarded so the lazy-create itself is
        # await-free and therefore atomic under asyncio's single-threaded scheduling.
        self._ensure_clients_lock: Any = None

        # Tracks which ADK session IDs have had their Beacon lifecycle set up
        self._sessions_initialised: set[str] = set()

        # Per-session turn buffers and interaction IDs
        self._buffers: dict[str, list[Record]] = {}
        self._interaction_ids: dict[str, str] = {}
        self._tool_start_times: dict[str, float] = {}
        # Per-session monotonic event counter → deterministic, collision-free
        # event_ids (f"{prefix}-{session_id}-{seq}"). ADK is a live streaming
        # adapter: a "re-read" of the source means re-running the agent over
        # the same conversation (crash recovery / replay), which re-fires the
        # same callback sequence. A fresh adapter restarts this counter at 0 and
        # reproduces the exact local ids (event_id is not sent, so nothing is deduped). The old
        # `uuid4().hex[:8]` ids were both non-deterministic (every replay
        # duplicated) and only 32-bit (birthday collisions in long sessions).
        # Never reset per turn — monotonic across the whole session so ids never
        # collide between turns; cleaned up with the rest of the session state.
        self._event_seq: dict[str, int] = {}
        # Per-session ADK invocation id of the turn currently in flight. ADK
        # assigns one invocation_id per turn and reuses it for every re-entrant
        # before_model call within that turn (after each tool result), so a
        # change signals a genuinely new turn — the authoritative boundary the
        # bare `session in _interaction_ids` guard couldn't detect.
        self._invocation_ids: dict[str, str] = {}
        # Strong refs to fire-and-forget flush tasks (salvaging an aborted
        # turn's buffer when the next turn starts). A bare create_task can be
        # GC'd mid-flight, truncating the flush; hold the ref until done.
        self._flush_tasks: set[asyncio.Task] = set()

    # ── Owner identity (task-scoped) ──────────────────────────────────────────

    @property
    def owner_id(self) -> str | None:
        """The owner bound to the turn running in the current asyncio task.

        Resolves via the ``_current_session_id`` ContextVar against the
        per-session owner map, so concurrent turns for different users each see
        their own owner. Falls back to ``_default_owner`` (the construction-time
        value, updated by ``set_owner`` when called outside any turn) for
        bootstrap / single-user / direct calls.

        May be ``None`` when no owner has been resolved yet — the adapter is
        constructed without an ``owner_id`` (the modern per-conversation path)
        and neither a hook, the ``session.user_id`` auto-detect, nor a direct
        ``set_owner`` has bound one for this turn. Callers that pass the owner to
        Beacon must treat ``None`` as "identity unknown → skip"; the single
        upstream gate is ``_manager_for`` (returns ``None`` for an unknown
        owner), so every downstream ingest/get_context/add_participant already
        no-ops best-effort. Never send ``None`` to Beacon as an owner.

        This is a read-only resolution — it never creates or evicts a session
        manager, so it is safe to read from the sync thread-pool contexts ADK
        uses to run tools (which have no running event loop).
        """
        sid = _current_session_id.get()
        if sid is not None:
            owner = self._session_owners.get(sid)
            if owner is not None:
                return owner
        return self._default_owner

    @owner_id.setter
    def owner_id(self, value: str | None) -> None:
        # Back-compat: __init__ and callers that assign owner_id directly land
        # on the default owner. set_owner() is the turn-aware path.
        self._default_owner = value

    @property
    def _owner_participant_id(self) -> str:
        """Beacon participant id for the current owner — always derived, never stored."""
        return self.owner_id

    # ── Per-session manager resolution ────────────────────────────────────────

    def _peek_manager(self, session_id: str | None) -> AsyncSessionManager | None:
        """Return the manager already bound to ``session_id`` without creating one.

        Read-only and loop-free — safe from any context. Returns the injected
        manager in single-manager mode, the existing per-session manager if one
        has been built, or ``None`` when nothing is bound yet.
        """
        if self._injected_mgr is not None:
            return self._injected_mgr
        if session_id is not None:
            return self._managers.get(session_id)
        return None

    def _resolve_agent_participant_variable(self) -> dict[str, Any]:
        """Beacon ``AiAgentParticipant`` block for a new manager: the caller's
        full-override if given, else this runtime's label + ``agent_id``.
        Resolved per manager, not at construction, so a late-bound
        ``agent_id`` is picked up."""
        if self._agent_participant_variable is not None:
            return dict(self._agent_participant_variable)
        return {**_AGENT_PARTICIPANT_VARIABLE, "AiAgentApiName": self.agent_id}

    def _manager_for(self, session_id: str) -> AsyncSessionManager | None:
        """Resolve (or lazily build) the AsyncSessionManager for ``session_id``.

        Only called from async turn paths (ensure-conversation, ingest,
        get_context), which always run on the event loop — so LRU eviction can
        safely schedule ``close()`` on the loop. Never call this from a sync,
        loop-free context; use ``_peek_manager`` there.
        """
        if self._injected_mgr is not None:
            # Single-manager mode: rebind the injected manager's owner to the
            # session's owner. Race-unsafe by construction (documented); fine for
            # the single-owner tests and manual-manager callers that use it.
            mgr = self._injected_mgr
            resolved = self._session_owners.get(session_id, self._default_owner)
            if resolved is not None:
                # Only rebind when the adapter actually resolved an owner; a None
                # (adapter built owner-less, no hook/set_owner yet) must not
                # clobber the owner the injected manager was constructed with.
                mgr._owner_id = resolved  # type: ignore[attr-defined]
            self._managers[session_id] = mgr
            self._managers.move_to_end(session_id)
            return mgr

        existing = self._managers.get(session_id)
        if existing is not None:
            self._managers.move_to_end(session_id)
            return existing

        if self._client is None:
            # Clients not initialised yet (pre-first-turn on the auth path).
            return None

        owner = self._session_owners.get(session_id, self._default_owner)
        if owner is None:
            # No identity resolved for this session yet: the adapter was built
            # without an owner_id (the modern per-conversation path) and neither
            # a hook, the session.user_id auto-detect, nor a direct set_owner has
            # bound one. Refuse to build a manager with a null owner — that would
            # attribute this session's Beacon writes/reads to nobody. Returning
            # None makes ingest / get_context / _ensure_conversation no-op
            # best-effort; the next turn rebuilds the manager once an owner is
            # bound. This is the SINGLE upstream gate that keeps a None owner
            # from ever reaching Beacon (see the owner_id property docstring).
            return None
        mgr = AsyncSessionManager(
            self._client,
            agent_id=self.agent_id,
            owner_id=owner,
            owner_type=self._owner_type,
            retrieval_client=self._retrieval,
            agent_participant_variable=self._resolve_agent_participant_variable(),
        )
        self._managers[session_id] = mgr
        self._evict_managers_if_needed()
        return mgr

    def _current_manager(self) -> AsyncSessionManager | None:
        """Resolve the manager for the current turn's session (no-arg get_context path).

        Used by ``get_context()`` which takes no session argument. Prefers the
        current turn's session; outside a turn, falls back to the injected
        manager or (if exactly one session manager exists) that one.
        """
        sid = _current_session_id.get()
        if sid is not None:
            return self._manager_for(sid)
        if self._injected_mgr is not None:
            return self._injected_mgr
        if len(self._managers) == 1:
            return next(iter(self._managers.values()))
        return None

    def _evict_managers_if_needed(self) -> None:
        """Bound the per-session manager map (LRU). Runs only on the event loop.

        Evicts the oldest *idle* manager (no pending deliveries) so an
        in-flight flush is never cut off. Skips managers with pending tasks by
        moving them to the front and trying the next oldest.
        """
        if self._injected_mgr is not None:
            return  # single-manager mode never grows the map meaningfully
        while len(self._managers) > self._MGR_CAP:
            evicted_sid = None
            for sid, mgr in self._managers.items():
                if not getattr(mgr, "_pending_tasks", None):
                    evicted_sid = sid
                    break
            if evicted_sid is None:
                # Every manager is busy — don't cut off in-flight deliveries.
                logger.warning(
                    "[ADK] manager cache at %d (cap %d) but all managers have "
                    "pending deliveries; not evicting this round.",
                    len(self._managers), self._MGR_CAP,
                )
                return
            mgr = self._managers.pop(evicted_sid)
            # Keep the owner binding: it's a tiny string, and dropping it would
            # silently re-default a merely-idle (e.g. paused) user's identity to
            # _default_owner ("placeholder" on the shared singleton) → their next
            # turn would ingest under the wrong owner. The manager is expensive
            # and rebuildable; the identity is cheap and must not be lost. We do
            # discard the lifecycle flag so the rebuilt manager re-runs setup.
            self._sessions_initialised.discard(evicted_sid)
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(mgr.close())
                # Hold a strong ref until done — a bare create_task can be GC'd
                # mid-drain, truncating close() and losing in-flight records.
                self._eviction_tasks.add(task)
                task.add_done_callback(self._eviction_tasks.discard)
            except RuntimeError:
                # No running loop — close() only drains already-delivered
                # per-turn records, so dropping it here is acceptable.
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def make_start_session_tool(self) -> Any:
        """Return a ready-made ADK FunctionTool that binds this adapter to the user.

        The LLM decides when to call it — pass it directly into LlmAgent's tools list::

            agent = LlmAgent(..., tools=[adapter.make_start_session_tool()])

        The tool's docstring is what the LLM sees as the tool description, so it
        knows to call it when the user provides *their own* email address, not when
        an email address appears in any other context.
        """
        _require_sdk("make_start_session_tool")
        from google.adk.tools import FunctionTool

        adapter = self

        def start_session(customer_email: str) -> dict[str, str]:
            """Bind this session to the customer's identity.

            Call this tool exactly once per conversation, as soon as the user
            provides their own email address. Do NOT call it with an email address
            that belongs to someone else (e.g. an address the user wants to
            forward something to, or a contact they are asking about).

            Args:
                customer_email: The email address the user has given as their own
                    identity for this support session.

            Returns:
                Confirmation that the session is linked to the customer.
            """
            adapter.set_owner(customer_email)
            return {"status": "success", "customerEmail": customer_email}

        return FunctionTool(start_session)

    def make_context_retrieval_tool(self) -> Any:
        """Return an ADK FunctionTool that runs an owner-scoped memory read.

        Pass it into LlmAgent's tools list — or use :meth:`default_tools` to get
        all SDK-provided tools at once::

            agent = LlmAgent(..., tools=[*adapter.default_tools(), *my_tools])

        The tool's docstring is the LLM-visible description, so it drives *when*
        the model reaches for memory. It resolves the owner from the adapter's
        current identity (bound via :meth:`set_owner` / auto-detected from the
        session), so the LLM never has to supply a customer ID.
        """
        _require_sdk("make_context_retrieval_tool")
        from google.adk.tools import FunctionTool

        adapter = self

        async def recall_customer_context(topic: str = "") -> dict[str, Any]:
            """Look up what we already know about this customer from past
            conversations (cross-session memory).

            Call this when answering would benefit from memory the current
            conversation hasn't covered yet — the customer's plan or account
            details, issues raised before, stated preferences, or anything they'd
            expect you to remember. You do NOT need to call it for small talk or
            when the current conversation already holds the answer.

            Args:
                topic: Optional focus hint (e.g. "billing history"). A soft hint,
                    not a hard filter — Beacon returns owner-scoped context.

            Returns:
                A dict with a ``userContext`` list of memory entries. An empty
                list means we have no stored memory for this customer yet.
            """
            context = await adapter.get_context()
            logger.info(
                "[ADK] recall_customer_context: owner=%r session_id=%r topic=%r → has_context=%s",
                adapter.owner_id,
                _current_session_id.get(),
                topic,
                bool(context and context.get("userContext")),
            )
            return context or {"userContext": []}

        return FunctionTool(recall_customer_context)

    def make_get_context_pack_tool(
        self,
        valid_pack_ids: list[str],
        *,
        param_defaults: dict[str, Any] | None = None,
        timeout: float | None = _CONTEXT_PACK_TIMEOUT_S,
    ) -> Any:
        """Return an ADK FunctionTool that retrieves a named CONTEXT PACK.

        This is the pack-scoped read (POST GetContext): the
        agent asks for one server-advertised ``context_pack`` of structured detail
        instead of the owner-scoped profile read the recall tool does. The
        ``before_model`` callback from :meth:`callback_kwargs` adds the
        context-pack instruction and pack catalog to the system instruction,
        so no prompt changes are needed::

            agent = LlmAgent(
                instruction=my_prompt,
                tools=[adapter.make_get_context_pack_tool(["PlanRules", "Claims"])],
                **adapter.callback_kwargs(),
            )

        It resolves this turn's conversation id / owner / owner_type / agent id
        from the adapter (like the recall tool), so the LLM never supplies an
        owner. ``owner_id`` is bound server-side and is never taken from the model.

        Args:
            valid_pack_ids: the pack ids this agent advertises. If agent
                cannot fetch self._allowed_pack_ids, an id outside this
                set is rejected with ``{"status": "invalid_pack"}`` without a call.
            param_defaults: fixed pack ``params`` merged into every call. The
                model's ``params`` override those keys. A non-empty ``query`` is
                added as ``params["query"]`` only when ``params`` did not already
                set it. Blank values are omitted.
            timeout: per-call HTTP timeout for the pack POST; defaults to
                :data:`_CONTEXT_PACK_TIMEOUT_S`, independent of the fast-read
                client timeout.

        The tool is best-effort: it NEVER raises. Beyond ``invalid_pack`` it
        returns ``{"status": "ok", "context": {...}}`` on success, or
        ``{"status": "not_ready", "detail": ...}`` on any failure (endpoint not
        deployed, auth/init failure, timeout) — matching the PREVIEW contract in
        :data:`CONTEXT_PACK_INSTRUCTION` so the agent degrades instead of failing.
        """
        _require_sdk("make_get_context_pack_tool")
        from google.adk.tools import FunctionTool

        adapter = self
        defaults = dict(param_defaults or {})

        async def get_context_pack(
            context_pack_id: str,
            params: dict[str, str] | None = None,
            query: str = "",
            context_pack_version: int | None = None,
        ) -> dict[str, Any]:
            """Retrieve one named context pack for this subject.

            This is the same read as getContext(contextPack). The available-packs
            block lists each pack's description and its data needs. If a pack or
            one of its data needs would help answer the user's question, call this
            with that pack id.

            Read the parameters on those data needs. Pass values you can determine
            in ``params``, keyed by parameter name. If you cannot determine a
            value, ask the user, or omit that key. Do not invent a value and do
            not send an empty string. A data need that declares an omitted
            parameter may be absent from the result.

            __VERSION_GUIDANCE_PLACEHOLDER__

            Args:
                context_pack_id: the pack to retrieve (one advertised id).
                params: parameter name to value, from the pack's data needs.
                    Omit a key you cannot determine.
                query: optional free-text question, sent as ``params["query"]``
                    only when ``params`` did not already set that key.
                context_pack_version: pack revision from the catalog block.
                    Pass the integer from the entry's ``version`` field when
                    present; omit when the catalog entry has no ``version``.

            Returns:
                ``{"status": "ok", "context_pack_id", "context": {...}}`` on
                success, ``{"status": "not_ready", ...}`` when the read failed or
                the endpoint is unavailable, ``{"status": "invalid_pack", ...}``
                when the id is not advertised by this agent, or
                ``{"status": "version_expired", "latest_version", "latest"}``
                when the pack has been republished since the catalog was read.
            """
            # Reject non-integer context_pack_version at the SDK boundary so a
            # bad value never masquerades as "omitted" (which would silently
            # bypass versioning). Gemini can emit integer args as floats
            # (``3.0``) and ADK does not coerce them, so accept whole-number
            # floats and normalise to int.
            if (
                isinstance(context_pack_version, float)
                and context_pack_version.is_integer()
            ):
                context_pack_version = int(context_pack_version)
            if context_pack_version is not None and (
                isinstance(context_pack_version, bool)
                or not isinstance(context_pack_version, int)
            ):
                logger.warning(
                    "[ADK] get_context_pack: %r rejected non-integer "
                    "context_pack_version=%r",
                    context_pack_id, context_pack_version,
                )
                return {
                    "status": "invalid_version",
                    "context_pack_id": context_pack_id,
                    "message": (
                        "`context_pack_version` must be an integer from the "
                        "catalog's `version` field. Re-call with the integer "
                        "value, or omit it if the catalog entry has no version."
                    ),
                }

            # Enforce CONTEXT_PACK_VERSION_GUIDANCE at the SDK boundary: when the
            # model omits ``context_pack_version`` and the cached catalog has
            # one for this pack, fill it in.
            if context_pack_version is None:
                context_pack_version = adapter._pack_versions.get(context_pack_id)

            allowed = adapter._allowed_pack_ids or set(valid_pack_ids)
            if context_pack_id not in allowed:
                logger.info("[ADK] get_context_pack: invalid pack id %r", context_pack_id)
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
                logger.info("[ADK] get_context_pack: %r → not_ready (%s)", context_pack_id, reason)
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

            # Resolve the live client + this turn's conversation/owner the same way
            # the recall tool does. Any missing piece → best-effort not_ready.
            try:
                await adapter._ensure_clients()
            except Exception as e:  # auth/token failure — degrade, never raise
                logger.warning(
                    "[ADK] get_context_pack: %r client init failed",
                    context_pack_id, exc_info=True,
                )
                return not_ready(f"client init failed: {type(e).__name__}")

            rc = adapter._retrieval
            if rc is None:
                return not_ready("no retrieval client")

            mgr = adapter._current_manager()
            conversation_id = getattr(mgr, "_session_id", None)
            owner_id = adapter.owner_id
            if not conversation_id or not owner_id:
                return not_ready("no active conversation/owner for this turn")

            # owner_id is bound server-side from the resolved owner (above) — never
            # taken from the model. Blank parameter values are omitted so a data
            # need that requires them is skipped rather than queried empty.
            sent = context_pack_params(params, query, defaults)

            logger.info(
                "[ADK] get_context_pack: %r params=%s version=%s",
                context_pack_id, sorted(sent), context_pack_version,
            )
            try:
                payload = await rc.get_context_pack(
                    conversation_id,
                    owner_id,
                    adapter.agent_id,
                    context_pack_id,
                    params=sent,
                    owner_type=adapter._owner_type,
                    context_pack_version=context_pack_version,
                    timeout=timeout,
                )
            except Exception as e:
                # Log the full exception server-side; return only its TYPE. A bare
                # httpx timeout stringifies to "", so the type keeps the not_ready
                # detail diagnosable, while repr(e) — which can carry request URLs,
                # tokens, or upstream error text — never reaches the model.
                logger.warning(
                    "[ADK] get_context_pack: %r request failed",
                    context_pack_id, exc_info=True,
                )
                return not_ready(f"request failed: {type(e).__name__}")

            mismatch = version_mismatch_entry(payload)
            if mismatch is not None:
                logger.info(
                    "[ADK] get_context_pack: %r → version_expired "
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

            logger.info("[ADK] get_context_pack: %r → ok", context_pack_id)
            return {"status": "ok", "context_pack_id": context_pack_id, "context": payload}

        # Splice the shared versioning guidance into the docstring that ADK
        # surfaces to the LLM as the tool description. Done post-def because a
        # Python docstring has to be a string literal — we can't use ``+`` or
        # an f-string inline without losing the docstring slot.
        if get_context_pack.__doc__ is not None:
            get_context_pack.__doc__ = get_context_pack.__doc__.replace(
                "__VERSION_GUIDANCE_PLACEHOLDER__", CONTEXT_PACK_VERSION_GUIDANCE,
            )
        return FunctionTool(get_context_pack)

    async def get_context_pack_instruction(self, session_id: str) -> str:
        """Return the context-pack instruction pinned for ``session_id``.

        The ``before_model`` callback from :meth:`callback_kwargs` already
        adds this to the system instruction on every model call, so most
        integrations never call it. The first call for a session fetches
        the server-published prompt
        (``GET /api/v1/agentic/prompts/system/context_pack_instruction``);
        a miss or failure pins the bundled
        :data:`agentic_context.CONTEXT_PACK_INSTRUCTION` for that session.
        Before the retrieval client exists (it is built on the first turn)
        this returns the bundled text without pinning it.
        """
        return await self._context_pack_prompt.get(session_id, self._retrieval)

    def default_tools(self) -> list[Any]:
        """SDK-provided tools to spread into ``LlmAgent(tools=[...])``.

        These are the tools the SDK auto-installs when a customer wires the
        adapter into their agent. Spread them alongside the customer's own
        tools::

            agent = LlmAgent(
                ...,
                tools=[*adapter.default_tools(), *my_tools],
                **adapter.callback_kwargs(),
            )

        Returns two tools:

        - ``recall_customer_context`` — the owner-scoped memory read over the
          customer's full profile / cross-session history.
        - ``get_context_pack`` — the pack-scoped POST read keyed by a server-
          advertised ``context_pack_id``. Call it to fetch structured detail
          (enrollments, dependents, handbook passages, etc.) that the recall
          read doesn't surface. The valid pack ids are populated at session
          start from the ``runtime-context-packs`` fetch and the tool rejects
          an un-advertised id without a round trip.

        Kept separate from :meth:`callback_kwargs` because ADK treats ``tools``
        and the ``*_callback`` kwargs as distinct ``LlmAgent`` parameters.
        """
        return [
            self.make_context_retrieval_tool(),
            self.make_get_context_pack_tool([]),
        ]

    def callback_kwargs(
        self,
        *,
        before_model: Callable | None = None,
        after_model: Callable | None = None,
        before_tool: Callable | None = None,
        after_tool: Callable | None = None,
    ) -> dict[str, Callable]:
        """Return all four ADK callback kwargs ready to unpack into LlmAgent.

        Usage::

            agent = LlmAgent(..., **adapter.callback_kwargs())

        Pass existing callbacks to compose them::

            agent = LlmAgent(..., **adapter.callback_kwargs(before_model=my_metrics_cb))
        """
        return {
            "before_model_callback": self.wrap_before_model(existing=before_model),
            "after_model_callback": self.wrap_after_model(existing=after_model),
            "before_tool_callback": self.wrap_before_tool(existing=before_tool),
            "after_tool_callback": self.wrap_after_tool(existing=after_tool),
        }

    # ── Per-conversation configuration (W2b — adapter.hooks / adapter.apply) ────

    # Config keys recognised in a hook return value / apply() call. Anything else
    # is stored verbatim (forward-compatible) but ignored by the adapter today.
    _CONFIG_KEYS = (
        "owner_id",               # bound via set_owner at conversation-open
        "dataspace",              # stored; NOT yet honored per-conversation (see note)
        "system_prompt_override", # W2c — replaces the injected system instruction
        "context_mode",            # per-conversation context_mode; expanded at merge
        # The INTERNAL expansion of a per-conversation context_mode (see
        # _merge_session_config) — the private flag _config_flag reads. Not a
        # public per-conversation surface: set context_mode= and it fans out
        # into this; a caller passing it directly is undocumented.
        "inject_at_open",
    )

    def hooks(
        self,
        *,
        on_conversation_start: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> GoogleAdkAdapter:
        """Register per-conversation lifecycle hooks. Chainable; returns ``self``.

        This is the modern, per-conversation configuration surface (2026-07-15
        sync, "Option 2"). Instead of baking a single ``owner_id`` into the
        adapter at construction — the customer's identity is different in every
        conversation — the app registers ONE adapter and supplies the
        per-conversation identity/config at conversation-open::

            adapter = GoogleAdkAdapter(agent_id="my-agent", auth=auth)  # no owner_id

            def on_start(session_id: str, ctx: dict) -> dict:
                # ctx carries {"session_id", "user_id", "current_owner"}.
                # Resolve the customer for THIS conversation however you like
                # (session lookup, request header, JWT claim, ...):
                return {
                    "owner_id": lookup_customer(ctx["user_id"]),
                    "system_prompt_override": persona_for(session_id),  # optional
                }

            adapter.hooks(on_conversation_start=on_start)

        ``on_conversation_start`` is invoked exactly once per conversation, at
        conversation-open (the first model call of an ADK session, and again if
        the owner is re-identified mid-session — an owner switch re-opens the
        Beacon conversation). It receives ``(session_id, ctx)`` and returns a
        mapping of config for that conversation (see ``_CONFIG_KEYS``); return
        ``None`` / ``{}`` to leave everything at the adapter defaults. The hook
        runs BEFORE the Beacon conversation is created, so an ``owner_id`` it
        returns scopes every write/read for the conversation from turn one.

        Best-effort: an exception in the hook is logged and swallowed — a
        misbehaving hook never breaks the turn (the conversation falls back to
        the adapter's default owner / config).

        Note on ``dataspace``: it is accepted and stored per conversation, but
        NOT yet re-routed per conversation — the Beacon HTTP clients are built
        once with the construction-time dataspace and shared across sessions, so
        honoring a per-conversation dataspace needs a shared-client change out of
        this adapter's scope. Left forward-compatible on purpose.
        """
        if on_conversation_start is not None:
            self._on_conversation_start = on_conversation_start
        return self

    def apply(
        self,
        session_id: str,
        *,
        owner_id: str | None = None,
        dataspace: str | None = None,
        system_prompt_override: str | None = None,
        **overrides: Any,
    ) -> GoogleAdkAdapter:
        """Imperatively set per-conversation config for a known session. Chainable.

        The push counterpart to :meth:`hooks` (pull). Use it when the app already
        knows the customer at the moment it creates the ADK session and would
        rather register the config directly than supply a callback::

            adapter.apply(session_id, owner_id="dana@corp.com")

        Merges into the same per-session config store the hook writes to, so a
        later turn picks it up at conversation-open identically. Safe to call
        before the first turn (pre-seed) or during a live session; when it
        changes the owner it re-arms the Beacon lifecycle so the next turn
        rebuilds the session manager under the new owner. Only the keys you pass
        are touched; unknown ``overrides`` are stored verbatim.
        """
        config: dict[str, Any] = dict(overrides)
        if owner_id is not None:
            config["owner_id"] = owner_id
        if dataspace is not None:
            config["dataspace"] = dataspace
        if system_prompt_override is not None:
            config["system_prompt_override"] = system_prompt_override
        self._merge_session_config(session_id, config)
        return self

    def _merge_session_config(self, session_id: str, config: dict[str, Any] | None) -> dict[str, Any]:
        """Merge ``config`` into the per-session store and bind owner if present.

        Shared by the declarative hook path (:meth:`hooks`) and the imperative
        push path (:meth:`apply`). Returns the merged config for this session.
        Binding the owner goes through :meth:`set_owner` when we're inside the
        session's turn (its ContextVar is set — the conversation-open path in
        ``wrap_before_model``), so per-session rebind + buffer-patch + lifecycle
        re-arm all reuse the audited turn-aware logic. Outside a turn (apply()
        pre-seed before any turn), it writes the owner map directly and re-arms
        the lifecycle so the first turn builds the manager under this owner.
        """
        if not config:
            return self._session_configs.get(session_id, {})
        # A per-conversation context_mode expands into the private bool flag
        # _config_flag reads, so the auto/manual override works through the same
        # turn-local lookup for this session. (The internal key can still be
        # supplied directly and win over the expansion via dict-merge order
        # below, but that's undocumented — context_mode= is the surface.)
        if "context_mode" in config:
            mode = config["context_mode"]
            _, iao = self._resolve_construction_mode(mode)
            config = {"inject_at_open": iao, **config}
        merged = {**self._session_configs.get(session_id, {}), **config}
        self._session_configs[session_id] = merged

        owner = config.get("owner_id")
        if owner is not None:
            if _current_session_id.get() == session_id:
                # Inside this session's turn — reuse the turn-aware path (rebinds
                # only this session, patches buffered records, re-arms open). Skip
                # when the owner is already bound so a hook re-returning the same
                # owner on a re-open turn doesn't redundantly re-patch the buffer.
                if self.owner_id != owner:
                    self.set_owner(owner)
            elif self._session_owners.get(session_id) != owner:
                # Pre-seed / cross-context push: bind the map directly and re-arm
                # the lifecycle so the next turn rebuilds the manager as this owner.
                self._session_owners[session_id] = owner
                self._sessions_initialised.discard(session_id)
                mgr = self._managers.get(session_id)
                if mgr is not None:
                    mgr._owner_id = owner  # type: ignore[attr-defined]
                    mgr._session_id = None  # type: ignore[attr-defined]
                    mgr._conversation_created = False  # type: ignore[attr-defined]
                    mgr._session_created = False  # type: ignore[attr-defined]
        return merged

    def _open_conversation_config(self, session_id: str, callback_context: Any) -> dict[str, Any]:
        """Invoke the conversation-start hook (if any) and return the session config.

        Called once per conversation-open from ``wrap_before_model``, before the
        Beacon conversation is created. Fires ``on_conversation_start`` — passing
        the session id and a small ctx dict — and merges whatever it returns into
        the per-session config (which may also have been pre-seeded via
        :meth:`apply`). Best-effort: a raising or non-dict-returning hook is
        logged and ignored so it can never break the turn.
        """
        if self._on_conversation_start is not None:
            session = getattr(callback_context, "session", None)
            ctx = {
                "session_id": session_id,
                "user_id": getattr(session, "user_id", None),
                "current_owner": self.owner_id,
            }
            try:
                result = self._on_conversation_start(session_id, ctx)
            except Exception:
                logger.exception("[ADK] on_conversation_start hook raised — ignoring (best-effort)")
                result = None
            if isinstance(result, dict):
                self._merge_session_config(session_id, result)
            elif result is not None:
                logger.warning(
                    "[ADK] on_conversation_start returned %s, expected dict/None — ignoring",
                    type(result).__name__,
                )
        return self._session_configs.get(session_id, {})

    def register_shutdown(self, timeout: float = 10.0) -> None:
        """Wire SIGINT and SIGTERM to close the adapter cleanly.

        Designed for uvicorn-based servers (``adk web``), where the event loop
        is already running when the signal fires.  Calls ``adapter.close()``
        via ``run_coroutine_threadsafe`` so the coroutine runs on the live loop
        without blocking the signal thread.

        Usage::

            adapter.register_shutdown()   # call once after constructing the adapter

        Args:
            timeout: Seconds to wait for ``close()`` to finish before giving up.
        """
        _shutdown_called = False

        def _handler(signum: int, _frame: Any) -> None:
            nonlocal _shutdown_called
            if _shutdown_called:
                return
            _shutdown_called = True
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(self.close(), loop)
                future.result(timeout=timeout)
            except Exception:
                pass
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    async def close(self) -> None:
        """Drain every session manager, then close the shared HTTP client.

        AsyncSessionManager.close() only drains that manager's pending
        deliveries — it never touches the shared client — so closing all
        managers and then the client once is correct (no double-close). In
        single-manager mode the same injected object may appear under several
        session ids; close() is idempotent, but we dedupe by identity anyway to
        keep the shutdown sequence unambiguous.
        """
        seen: set[int] = set()
        managers = list(self._managers.values())
        if self._injected_mgr is not None:
            managers.append(self._injected_mgr)
        for mgr in managers:
            if id(mgr) in seen:
                continue
            seen.add(id(mgr))
            await mgr.close()
        self._context_pack_prompt.clear()
        if self._client is not None:
            await self._client.close()
        # Close the httpx clients we injected for the observer path. The core
        # clients won't (they set _owns_client=False when handed a client=), so
        # this is where their sockets get released. Guarded so a client that was
        # never opened (no first turn) can't break shutdown. Idempotent.
        for http_client in self._owned_http_clients:
            try:
                await http_client.aclose()
            except Exception:
                logger.debug("owned http client aclose failed", exc_info=True)
        self._owned_http_clients = []

    def set_owner(self, owner_id: str) -> None:
        """Bind the current turn's session to a specific user identity (e.g. email).

        Call as soon as the user's identity is known. Scopes all Beacon writes
        and get_context calls to that user, enabling per-user multi-session memory.

        Turn-aware: when called inside a turn (the ``_current_session_id``
        ContextVar is set — the auto-detect in ``wrap_before_model`` and the
        ``start_session`` tool both run there), it rebinds *that session only*,
        so a concurrent turn for another user is unaffected. When called outside
        any turn (bootstrap / single-user), it updates the default owner.

        The owner is written to instance-level maps keyed by session id — never
        to a ContextVar — because tools run in throwaway child-task contexts
        whose ContextVar writes would not propagate back to the parent turn.

        Any records already buffered under the old participant ID are patched
        in-place so the correct owner ID is used when they are flushed.
        """
        old_participant_id = self._owner_participant_id  # read of the property, pre-change

        sid = _current_session_id.get()
        if sid is not None:
            # Rebind only this session. AsyncSessionManager unifies conversation
            # and session under _session_id, so we null _session_id (not a
            # phantom _conversation_id) to force a fresh lifecycle for the new
            # owner on the next ingest/get_context.
            self._session_owners[sid] = owner_id
            mgr = self._managers.get(sid)
            if mgr is not None:
                mgr._owner_id = owner_id  # type: ignore[attr-defined]
                mgr._session_id = None  # type: ignore[attr-defined]
                mgr._conversation_created = False  # type: ignore[attr-defined]
                mgr._session_created = False  # type: ignore[attr-defined]
            self._sessions_initialised.discard(sid)
        else:
            # Bootstrap / single-user: update the fallback owner. Managers are
            # built lazily per session and pick up the resolved owner then, so
            # nothing else is needed on the pre-init path. Clear session-init
            # state so any already-registered sessions re-run their lifecycle
            # under the new default (preserves prior clear-on-rebind behaviour).
            self._default_owner = owner_id
            self._sessions_initialised.clear()

        # Patch any buffered records that were written before the real owner was
        # known — they will have the old placeholder participant ID. Record is
        # frozen so we replace each matching entry with a corrected copy.
        #
        # CRITICAL: scope the patch to THIS turn's session when we're inside a
        # turn. The buffer map is keyed by session id and holds records for every
        # concurrent user; patching all of them would let one user's set_owner
        # rewrite another user's still-buffered records to the wrong owner — the
        # exact cross-attribution this whole change exists to prevent. Only on the
        # bootstrap / single-user path (sid is None) do we patch every buffer.
        new_participant_id = self._owner_participant_id  # read of the property, post-change
        target_sids = [sid] if sid is not None else list(self._buffers.keys())
        for session_id in target_sids:
            records = self._buffers.get(session_id)
            if not records:
                continue
            self._buffers[session_id] = [
                record.model_copy(update={"participant_id": new_participant_id})
                if record.participant_id == old_participant_id
                else record
                for record in records
            ]

        logger.debug("set_owner: bound owner %s (session=%s)", owner_id, sid)

    # ── context_mode facade ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_construction_mode(
        context_mode: Literal['auto', 'manual'],
    ) -> tuple[Literal['auto', 'manual'], bool]:
        """Expand ``context_mode`` into concrete injection state.

        Returns ``(resolved_mode, inject_at_open)``. The second is the private
        flag the injection logic reads; the first is the mode's own state.
        Raises ``ValueError`` on an unknown slug (via :func:`_resolve_context_mode`).
        """
        iao = _resolve_context_mode(context_mode)  # raises on bad slug
        return context_mode, iao

    @property
    def context_mode(self) -> Literal['auto', 'manual']:
        """The resolved ``context_mode`` — ``"auto"`` (default) or ``"manual"``.
        Runtime-switchable via :meth:`set_context_mode`."""
        return self._context_mode

    def set_context_mode(self, mode: Literal['auto', 'manual']) -> None:
        """Switch the memory mode at runtime — the developer's reversible knob.

        Re-expands the internal injection flag from the mode value, exactly as
        construction does, so the new mode takes effect on the next turn (the
        flag is re-read at the top of each model call). Operator/developer
        control only — never an end-user setting.

        Note: flipping to ``"auto"`` mid-process does not retroactively seed
        already-open sessions — their conversation-open signal is already
        consumed — so those fall to tool-only until they re-open.
        """
        previous = self._context_mode
        self._context_mode, self._inject_at_open = (
            self._resolve_construction_mode(mode)
        )
        if previous != self._context_mode:
            logger.info("[ADK] context_mode: %s → %s (runtime toggle)", previous, self._context_mode)

    def _config_flag(self, name: str, default: bool) -> bool:
        """Resolve a behaviour flag for the current turn's conversation (W2c).

        Per-conversation config (set via :meth:`hooks` / :meth:`apply`) overrides
        the instance-level default when the current turn's session carries that
        key; otherwise the instance flag (the construction-time / runtime-toggled
        value) applies. Keyed by the ``_current_session_id`` ContextVar, so the
        lookup is turn-local and concurrency-safe like every other owner read.
        Outside a turn, or with no per-session override, this is exactly the old
        ``self._<name>`` read — so the shipped default is unchanged.
        """
        sid = _current_session_id.get()
        if sid is not None:
            cfg = self._session_configs.get(sid)
            if cfg is not None and name in cfg:
                return bool(cfg[name])
        return default

    def _should_inject_fast_context(self, is_conversation_open: bool) -> bool:
        """Decide whether to inject owner-scoped profile context on this model call.

        Under ``context_mode="auto"`` the adapter seeds context once at
        conversation-open (the first model call of the session); afterwards the
        agent requests further reads itself via the recall tool. Under
        ``"manual"`` the adapter seeds nothing — the customer owns their own
        prompting. Resolved per-conversation (W2c): a hook/apply override for
        this session wins over the instance value, falling back to the instance
        flag when unset.
        """
        return is_conversation_open and self._config_flag(
            "inject_at_open", self._inject_at_open
        )

    async def add_participant(
        self,
        participant_id: str,
        participant_type: str = "agent",
        role: str = "participant",
    ) -> None:
        """Add a participant to the current Beacon conversation.

        Call before handing off to another agent, or to register a human
        participant who will be sending messages directly.

        The participant ID convention mirrors how this adapter registers itself:
        agents use the plain ``agent_id``; humans use their identifier as-is
        (e.g. an email address). Pass the final ID directly — no suffix is added.

        Args:
            participant_id: Beacon participant ID. For agents: ``agent_id``.
                For humans: the user's identifier (e.g. ``"dana@company.com"``).
            participant_type: ``"agent"`` or ``"human"`` — defaults to ``"agent"``.
            role: Participant role — defaults to ``"participant"``.
        """
        if self._client is None:
            logger.warning("add_participant: no HTTP client available (test mode)")
            return
        # Resolve the manager bound to the current turn's session (read-only —
        # add_participant has no session argument). AsyncSessionManager unifies
        # conversation and session under _session_id, so it serves as both ids.
        mgr = self._peek_manager(_current_session_id.get())
        session_id = getattr(mgr, "_session_id", None)
        if not session_id:
            logger.warning("add_participant: conversation not yet initialised — call after first turn")
            return
        try:
            await self._client.add_participants(
                conversation_id=session_id,
                session_id=session_id,
                owner_id=self.owner_id,
                agent_id=self.agent_id,
                participants=[{"id": participant_id, "type": participant_type, "role": role}],
            )
            logger.debug("add_participant: added %r (%s) to conversation %s", participant_id, participant_type, session_id)
        except Exception as e:
            logger.warning("add_participant: failed to add %r: %s", participant_id, e)

    async def get_context(self) -> dict[str, Any] | None:
        """Retrieve context for the current owner from Beacon.

        Beacon keys context on owner_id — profile facts are returned regardless
        of which conversation_id is passed. The conversation_id only scopes the
        in-session summary for the current conversation. Resolves the manager
        for the current turn's session (via the ``_current_session_id``
        ContextVar), so concurrent turns never read one another's owner state.
        """
        mgr = self._current_manager()
        if mgr is None:
            return None
        return await mgr.get_context()

    # ─── Observability (HTTP event hooks) ─────────────────────────

    def _make_event_hooks(self) -> dict[str, list]:
        """Build httpx event_hooks that forward Beacon HTTP calls to the observer.

        Emits two events per call:
        - ``{"phase": "request",  callId, method, url, requestBody, timestamp}``
        - ``{"phase": "response", callId, method, url, status, latencyMs,
          responseBody, timestamp}`` (on transport error: ``{..., error}``)

        ``callId`` ties a request to its response. ``url`` is the path only
        (query string dropped) so tokens/ids in params don't leak to the UI.
        ``requestBody``/``responseBody`` are the decoded payloads, size-capped
        (see ``_BODY_CAP``). Headers (which carry the bearer token) are NEVER
        emitted — only bodies, which for Beacon are plain JSON payloads.
        """
        observer = self._http_observer

        async def _emit(event: dict) -> None:
            try:
                result = observer(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:  # never let observability break delivery
                logger.debug("http_observer raised (ignored): %s", e)

        async def on_request(request: Any) -> None:
            request.extensions["am_call_id"] = uuid4().hex[:8]
            request.extensions["am_start"] = monotonic()
            await _emit({
                "phase": "request",
                "callId": request.extensions["am_call_id"],
                "method": request.method,
                "url": str(request.url.copy_with(query=None)),
                "requestBody": self._decode_body(getattr(request, "content", None)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        async def on_response(response: Any) -> None:
            request = response.request
            call_id = request.extensions.get("am_call_id", "")
            start = request.extensions.get("am_start")
            latency_ms = int((monotonic() - start) * 1000) if start else None
            # Read the (non-streamed) Beacon response body so we can surface it.
            # httpx caches the bytes, so the adapter's later .json()/.text still works.
            body = None
            try:
                read = response.aread()
                if inspect.isawaitable(read):
                    await read
                body = self._decode_body(response.content)
            except Exception:
                body = None
            await _emit({
                "phase": "response",
                "callId": call_id,
                "method": request.method,
                "url": str(request.url.copy_with(query=None)),
                "status": response.status_code,
                "latencyMs": latency_ms,
                "responseBody": body,
                # Server trace / correlation ids: let an operator pivot a failed
                # Beacon call into the Context Service's own logs. Response-side,
                # non-secret ids only — never auth/cookie headers.
                "traceId": self._trace_headers(getattr(response, "headers", None)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return {"request": [on_request], "response": [on_response]}

    # Response headers that carry a server-side trace / correlation / request id.
    # Surfaced so a Beacon failure (e.g. a 400 "Failed to read request") can be
    # traced in the Context Service's own logs. These are the RESPONSE side —
    # the request Authorization/bearer is never read or emitted here.
    _TRACE_HEADER_NAMES = (
        "x-b3-traceid",
        "x-trace-id",
        "x-request-id",
        "x-correlation-id",
        "x-sfdc-request-id",
        "x-sfdc-trace-id",
        "traceparent",
        "request-id",
        "x-amzn-trace-id",
        "x-envoy-request-id",
    )

    @classmethod
    def _trace_headers(cls, headers: Any) -> dict[str, str] | None:
        """Extract server trace/correlation ids from a RESPONSE's headers.

        Returns a small ``{name: value}`` dict of any header that looks like a
        trace / request / correlation id, or ``None`` if none are present. Only
        these matched, non-secret ids are returned — never the full header set,
        and ``authorization``/``cookie``/``set-cookie`` are always skipped.
        """
        if not headers:
            return None
        try:
            items = list(headers.items())
        except Exception:
            return None
        found: dict[str, str] = {}
        for name, value in items:
            low = str(name).lower()
            if low in ("authorization", "cookie", "set-cookie"):
                continue
            if (
                low in cls._TRACE_HEADER_NAMES
                or "trace" in low
                or "request-id" in low
                or "correlation" in low
            ):
                found[low] = str(value)[:200]
        return found or None

    # Max characters of any single request/response body forwarded to the
    # observer. Beacon payloads are small, but get_context responses can be
    # large — cap so we never flood the SSE channel / browser.
    _BODY_CAP = 20_000

    @staticmethod
    def _decode_body(raw: Any) -> str | None:
        """Best-effort decode of an HTTP body (bytes/str) → capped text.

        Returns None for empty bodies. Never raises — observability must not
        break delivery. Pretty-prints JSON when the body parses as JSON.
        """
        if raw is None:
            return None
        try:
            if isinstance(raw, (bytes, bytearray)):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)
            if not text:
                return None
            # Pretty-print JSON for readability in the console.
            try:
                import json as _json
                text = _json.dumps(_json.loads(text), indent=2)
            except Exception:
                pass
            if len(text) > GoogleAdkAdapter._BODY_CAP:
                text = text[: GoogleAdkAdapter._BODY_CAP] + "\n… (truncated)"
            return text
        except Exception:
            return None

    async def _ensure_clients(self) -> None:
        """Lazily initialise the shared HTTP clients on first turn.

        Only runs when credential-based auth is used. Fetches the token and
        base_url from ContextAuth (which handles all refresh logic internally),
        then builds the two clients and attaches the auth so future refreshes
        propagate automatically. Per-session managers are built lazily in
        ``_manager_for`` and share these clients. No-op after the first call.
        """
        if self._auth is None or self._client is not None:
            return
        # Lazily create the lock on the running loop (see __init__ note). This
        # create-if-None is await-free, so it's atomic under asyncio — no two
        # coroutines can each make a distinct lock.
        if self._ensure_clients_lock is None:
            self._ensure_clients_lock = asyncio.Lock()
        async with self._ensure_clients_lock:
            # Re-check under the lock: a racing first-turn may have built the
            # clients while we awaited the lock. Without this, the loser would
            # build (and, on the observer path, leak) a second client pair.
            if self._client is not None:
                return
            token, base_url = await self._auth.aget_token()

            # Observer path: build two httpx clients pre-wired with observability
            # event hooks and inject them via the core clients' documented
            # `client=` seam, so every Beacon request/response flows through the
            # observer. The core clients set _owns_client=False for injected
            # clients, so we track and close these ourselves in close(). None path
            # (default) is unchanged: the core clients build their own bare client.
            ingest_http = None
            retrieve_http = None
            if self._http_observer is not None:
                import httpx

                event_hooks = self._make_event_hooks()
                ingest_http = httpx.AsyncClient(timeout=10.0, event_hooks=event_hooks)
                retrieve_http = httpx.AsyncClient(timeout=10.0, event_hooks=event_hooks)
                self._owned_http_clients = [ingest_http, retrieve_http]

            client = AsyncIngestionClient(
                base_url=base_url,
                api_key=token,
                dataspace=self._dataspace,
                client=ingest_http,
                auth=self._auth,
            )
            self._retrieval = AsyncRetrievalClient(
                base_url=base_url,
                api_key=token,
                dataspace=self._dataspace,
                client=retrieve_http,
                auth=self._auth,
            )
            self._auth.attach(client)
            self._auth.attach(self._retrieval)
            # Assign self._client LAST: it is the sentinel other turns check
            # (both the fast guard above and the re-check under the lock). Setting
            # it only after the pair is fully built means a concurrent turn either
            # sees None (and waits on the lock) or a fully-wired client — never a
            # half-initialised one.
            self._client = client

    # ── Callback wrappers ─────────────────────────────────────────────────────

    def wrap_before_model(self, existing: Callable | None = None) -> Callable:
        async def callback(callback_context: Any, llm_request: Any) -> Any:
            # Bind this turn's session id to the running task FIRST — before any
            # await. Everything downstream (the owner auto-detect + set_owner,
            # get_context, the tools ADK spawns as child tasks) resolves owner
            # identity through this ContextVar, so it must be set before the
            # first suspension point or a concurrent turn could run set_owner
            # against the wrong (default) owner. See _current_session_id.
            session_id = self._get_session_id(callback_context)
            if session_id:
                _current_session_id.set(session_id)

            await self._ensure_clients()

            # Auto-detect owner from session.user_id (set when session created with real email)
            session = getattr(callback_context, "session", None)
            user_id = getattr(session, "user_id", None)
            logger.info(
                "[ADK] wrap_before_model: adk_session_id=%s session.user_id=%r adapter.owner_id=%r mgr._session_id=%r",
                session_id, user_id, self.owner_id,
                getattr(self._peek_manager(session_id), "_session_id", "N/A"),
            )
            if user_id and user_id not in ("user", "placeholder") and user_id != self.owner_id:
                logger.info("[ADK] set_owner: %r → %r", self.owner_id, user_id)
                self.set_owner(user_id)

            # Snapshot conversation-open state BEFORE _ensure_conversation
            # consumes it (that call adds session_id to _sessions_initialised).
            # This is the "first turn of the session" signal the context cadence
            # gate below keys on — read it here or it is always False. It also
            # gates _ensure_conversation, so the lifecycle setup and the
            # open-turn injection share a single first-turn decision.
            is_conversation_open = bool(session_id) and (
                session_id not in self._sessions_initialised
            )

            if is_conversation_open:
                # W2b — per-conversation config. Fire the on_conversation_start
                # hook (or read apply()-seeded config) BEFORE _ensure_conversation
                # so a hook-supplied owner_id scopes the Beacon conversation from
                # creation. May itself call set_owner, which re-arms open — but we
                # already snapshotted is_conversation_open above, so the cadence
                # gates below stay stable for this turn.
                if self._on_conversation_start is not None or session_id in self._session_configs:
                    self._open_conversation_config(session_id, callback_context)
                await self._ensure_conversation(session_id)

            # W2c — per-conversation system-prompt override. Applied BEFORE the
            # memory injections below so a persona swap becomes the base the
            # [Cross-session context] block layers on top of (an override
            # replaces the agent's base instruction, it does not disable
            # memory). Runs every turn — the persona persists for the whole
            # conversation — and is idempotent within a turn.
            self._apply_system_prompt_override(llm_request)

            await self._inject_context_pack_list(
                llm_request, is_conversation_open, session_id,
            )

            await self._inject_memory_context(llm_request, session_id, is_conversation_open)

            if existing is not None:
                if inspect.iscoroutinefunction(existing):
                    await existing(callback_context, llm_request)
                else:
                    existing(callback_context, llm_request)

            self._on_before_model(callback_context, llm_request)
        return callback

    def wrap_after_model(self, existing: Callable | None = None) -> Callable:
        async def callback(callback_context: Any, llm_response: Any) -> Any:
            result = existing(callback_context, llm_response) if existing else None
            await self._on_after_model_async(callback_context, llm_response)
            return result
        return callback

    def wrap_before_tool(self, existing: Callable | None = None) -> Callable:
        def callback(tool: Any, args: dict[str, Any], tool_context: Any) -> Any:
            result = existing(tool, args, tool_context) if existing else None
            self._on_before_tool(tool, args, tool_context)
            return result
        return callback

    def wrap_after_tool(self, existing: Callable | None = None) -> Callable:
        def callback(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any) -> Any:
            result = existing(tool, args, tool_context, tool_response) if existing else None
            self._on_after_tool(tool, args, tool_context, tool_response)
            return result
        return callback

    # ── Internal callback handlers ────────────────────────────────────────────

    async def _inject_memory_context(
        self, llm_request: Any, session_id: str | None, is_conversation_open: bool
    ) -> None:
        """Dispatch this turn's memory injection according to the resolved mode.

        Under ``context_mode="auto"`` the adapter seeds owner-scoped profile
        context once at conversation-open; under ``"manual"`` it seeds nothing.
        Resolved per-conversation (W2c) via ``_config_flag`` on the
        ``inject_at_open`` flag.
        """
        if self._should_inject_fast_context(is_conversation_open):
            await self._inject_fast_context(llm_request, session_id)

    def _inject_context(self, llm_request: Any, context: Any) -> None:
        """Prepend Beacon context to the system instruction in the LLM request.

        No marker guard: this relies on the fast-context cadence being
        open-only, and re-entrant before_model calls (after a tool result)
        are never conversation-open, so `_should_inject_fast_context` gates
        them out. If a future change lets fast context inject on non-open
        turns, add a marker guard here to stay idempotent.
        """
        try:
            config = getattr(llm_request, "config", None)
            if config is None:
                return
            current = getattr(config, "system_instruction", None) or ""
            # Extract just the entry text from userContext — skip contextId and metadata
            entries = []
            for item in (context.get("userContext") or []):
                entry = item.get("entry") if isinstance(item, dict) else None
                if entry:
                    entries.append(entry.strip())
            context_text = "\n\n".join(entries) if entries else str(context)
            config.system_instruction = f"[Cross-session context from Data Cloud]\n{context_text}\n\n{current}"
        except Exception as e:
            logger.warning(
                "[ADK] _inject_context failed: llm_request=%s: %s",
                type(llm_request).__name__, e,
            )

    async def _inject_context_pack_list(
        self,
        llm_request: Any,
        is_conversation_open: bool = False,
        session_id: str | None = None,
    ) -> None:
        """Fetches and injects this agent's authorized  CONTEXT PACK list into system instructions.

           The context-pack instruction (pinned per session, see
           :meth:`get_context_pack_instruction`) goes right before the list,
           unless the agent's own instruction already contains it.

           Caches the fetched CONTEXT PACK list into `_allowed_pack_ids`
           to reuse on every turn after that session's open and assemble system instructions.
           Refetches this list again at every session-start.

           This is a deterministic (not LLM-decided) call for the SDK once
           to inject the list of authorized context packs a given agent can utilize.
        """
        if is_conversation_open and self._retrieval is not None:
                try:
                    new_context_pack_list = await self._retrieval.get_context_pack_list(agent_id=self.agent_id)
                    new_context_pack_id_map = {}
                    new_pack_versions: dict[str, int] = {}
                    if new_context_pack_list is not None:
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
                            # Surface `version` to the LLM when the server
                            # reports one so it can echo it back on the call as
                            # ``context_pack_version`` and recover from
                            # ``version_expired``. Only int values are surfaced;
                            # ``null`` means the server has no revision for
                            # this pack and bool is an int subclass so both are
                            # guarded out. Leaking null/bool into the catalog
                            # block would satisfy the "entry includes a
                            # `version`" condition in
                            # CONTEXT_PACK_VERSION_GUIDANCE and make the LLM
                            # echo a value the SDK would then reject.
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
                except Exception as e:
                    logger.warning(
                        "[ADK] _inject_context_pack_list failed: %s: %s",
                        type(e).__name__, e,
                    )
        # The instruction explains the catalog; with no packs it is only cost.
        instruction = None
        if self._allowed_pack_ids:
            instruction = (
                await self.get_context_pack_instruction(session_id)
                if session_id else CONTEXT_PACK_INSTRUCTION
            )
        try:
            config = getattr(llm_request, "config", None)
            if config is None:
                return
            current = getattr(config, "system_instruction", None) or ""
            # Match the block header, not the instruction's quoted mention
            # of the marker, so a prompt that already carries the
            # instruction still gets the catalog.
            if f"{_CONTEXT_PACK_LIST_MARKER}\n" in current:
                return
            parts = [current] if current else []
            if instruction and not self._context_pack_prompt.present_in(current, instruction):
                parts.append(instruction)
            parts.append(self._allowed_context_pack_list)
            config.system_instruction = "\n\n".join(parts)
        except Exception as e:
            logger.warning(
                "[ADK] pack catalog not applied to llm_request: %s: %s",
                type(e).__name__, e,
            )

    async def _inject_fast_context(self, llm_request: Any, session_id: str | None) -> None:
        """Fetch fast (profile) context for the current owner and inject it.

        Extracted from ``wrap_before_model`` so the callback stays under the
        cognitive-complexity budget. Best-effort: any failure is swallowed so a
        memory read never blocks the turn.
        """
        try:
            context = await self.get_context()
            # Beacon registers the conversation/participant asynchronously (the
            # create/participants POSTs return 202), so the FIRST fast-context GET
            # at conversation-open can race that registration and come back empty
            # even when the owner does have profile context. Optionally retry with a
            # short backoff when we got nothing back, so a transient empty self-heals
            # before the model runs.
            #
            # DEFAULT 0 RETRIES (single call): an integrating app may wire its OWN
            # smarter empty-context fallback into get_context() (e.g. a GET-then-seed
            # fallback), which both catches the race AND covers owners that genuinely
            # lack profile context — so re-firing the SAME empty GET here would be
            # redundant and only add latency (each retry sleeps BACKOFF_S). An owner
            # with no context otherwise burns N+1 identical empty GETs. Set
            # FAST_CONTEXT_OPEN_RETRIES>0 to re-enable blind retries.
            _FAST_CONTEXT_OPEN_RETRIES = int(os.environ.get("FAST_CONTEXT_OPEN_RETRIES", "0"))
            _FAST_CONTEXT_OPEN_BACKOFF_S = float(os.environ.get("FAST_CONTEXT_OPEN_BACKOFF_S", "1.2"))
            attempts = 1
            while (
                not (context and context.get("userContext"))
                and attempts <= _FAST_CONTEXT_OPEN_RETRIES
            ):
                await asyncio.sleep(_FAST_CONTEXT_OPEN_BACKOFF_S)
                context = await self.get_context()
                attempts += 1
            logger.info(
                "[ADK] get_context: owner=%r session_id=%r → has_context=%s (attempts=%d)",
                self.owner_id,
                session_id,
                bool(context and context.get("userContext")),
                attempts,
            )
            if context:
                self._inject_context(llm_request, context)
        except Exception as e:
            logger.warning("[ADK] _inject_fast_context failed: %s", e)

    def _apply_system_prompt_override(self, llm_request: Any) -> None:
        """Replace the base system instruction with a per-conversation override (W2c).

        When the current turn's conversation config carries
        ``system_prompt_override`` (set via :meth:`hooks` / :meth:`apply`), swap
        the request's base system instruction for it. Applied before the memory
        injections in ``wrap_before_model``, so ``[Cross-session context]`` /
        ``[Context guidance]`` prepend on top of the override rather than the
        agent's original prompt — an override changes the persona, it does not
        turn memory off.

        Idempotent within a turn without polluting the prompt: after the memory
        block prepends, the override becomes a *suffix* of the instruction, so a
        re-entrant before_model (after a tool result on the same turn) detects
        the override is already present (``override in current``) and skips —
        preserving the memory block a prior before_model already layered on top,
        with no visible marker added to the persona. Best-effort: an unexpected
        request shape is left untouched. No-op when no override is configured for
        this conversation.
        """
        sid = _current_session_id.get()
        if sid is None:
            return
        cfg = self._session_configs.get(sid)
        if not cfg:
            return
        override = cfg.get("system_prompt_override")
        if not override:
            return
        try:
            config = getattr(llm_request, "config", None)
            if config is None:
                return
            current = getattr(config, "system_instruction", None) or ""
            if override in current:
                return  # already applied this turn — don't clobber memory blocks
            config.system_instruction = override
        except Exception as e:
            logger.warning(
                "[ADK] system_prompt_override not applied: llm_request=%s: %s",
                type(llm_request).__name__, e,
            )

    def _on_before_model(self, callback_context: Any, llm_request: Any) -> None:
        session_id = self._get_session_id(callback_context)
        if not session_id:
            return

        # Decide whether this is a genuinely new turn or a re-entrant
        # before_model call (ADK re-enters after each tool result within the
        # same turn). ADK stamps one invocation_id per turn and reuses it
        # across those re-entries, so it is the authoritative boundary.
        invocation_id = self._get_invocation_id(callback_context)
        current_inv = self._invocation_ids.get(session_id)

        if invocation_id is not None and invocation_id == current_inv:
            # Same invocation → genuine re-entry within this turn → skip.
            return
        if invocation_id is None and session_id in self._interaction_ids:
            # Legacy ADK without invocation_id: fall back to the original
            # "already mid-turn" heuristic. This path can't distinguish a
            # re-entry from a new turn after an aborted one — the modern
            # invocation_id path below is what fixes that.
            return

        # New turn. If prior per-turn state lingers, the previous turn aborted
        # before reaching turn-end (e.g. it raised after a tool call). Detach
        # and flush that buffer NOW — synchronously, before this turn re-seeds
        # the buffer — so its records are neither dropped nor appended-into and
        # cross-attributed to this new turn's interaction_id.
        if session_id in self._interaction_ids:
            self._salvage_aborted_turn(session_id)

        if invocation_id is not None:
            self._invocation_ids[session_id] = invocation_id

        self._interaction_ids[session_id] = str(uuid4())
        self._buffers.setdefault(session_id, [])

        user_text = self._extract_user_message(llm_request)
        if user_text:
            self._buffer(session_id, Record(
                event_id=self._next_event_id(session_id, "msg"),
                session_id=session_id,
                agent_id=self.agent_id,
                event_type="user_message",
                category="Message",
                interaction_id=self._interaction_ids[session_id],
                participant_id=self._owner_participant_id,
                content=user_text,
                variable=None,
                metadata={"source": "before_model_callback"},
                timestamp=datetime.now(tz=timezone.utc),
            ))

    def _on_after_model(self, callback_context: Any, llm_response: Any) -> None:
        session_id = self._get_session_id(callback_context)
        if not session_id:
            return

        text = self._extract_response_text(llm_response)
        is_tool_call = self._is_tool_call(llm_response)
        tokens = self._extract_tokens(llm_response)
        interaction_id = self._interaction_ids.get(session_id, str(uuid4()))

        if is_tool_call:
            # Intermediate model output before a tool call — this is reasoning
            if text:
                self._buffer(session_id, Record(
                    event_id=self._next_event_id(session_id, "step"),
                    session_id=session_id,
                    agent_id=self.agent_id,
                    event_type="agent_reasoning",
                    category="Trace",
                    interaction_id=interaction_id,
                    participant_id=self._agent_participant_id,
                    content=None,
                    variable={"OutputValueText": text},
                    metadata={"source": "after_model_callback", **tokens},
                    timestamp=datetime.now(tz=timezone.utc),
                ))
        else:
            # Final model response — buffer and flush the whole turn
            if text:
                self._buffer(session_id, Record(
                    event_id=self._next_event_id(session_id, "msg"),
                    session_id=session_id,
                    agent_id=self.agent_id,
                    event_type="agent_response",
                    category="Message",
                    interaction_id=interaction_id,
                    participant_id=self._agent_participant_id,
                    content=text,
                    variable=None,
                    metadata={"source": "after_model_callback", **tokens},
                    timestamp=datetime.now(tz=timezone.utc),
                ))
            self._on_turn_end(session_id)

    def _on_before_tool(self, tool: Any, args: dict, tool_context: Any) -> None:
        session_id = self._get_session_id_from_tool_context(tool_context)
        if not session_id:
            return

        self._tool_start_times[f"{session_id}:{tool.name}"] = time()
        self._buffer(session_id, Record(
            event_id=self._next_event_id(session_id, "step"),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="tool_call",
            category="Trace",
            interaction_id=self._interaction_ids.get(session_id, str(uuid4())),
            participant_id=self._agent_participant_id,
            content=None,
            variable={"Name": tool.name, "InputValueText": json.dumps(args, default=str)},
            metadata={"source": "before_tool_callback"},
            timestamp=datetime.now(tz=timezone.utc),
        ))

    def _on_after_tool(self, tool: Any, _args: dict[str, Any], tool_context: Any, tool_response: Any) -> None:
        session_id = self._get_session_id_from_tool_context(tool_context)
        if not session_id:
            return

        start = self._tool_start_times.pop(f"{session_id}:{tool.name}", None)
        duration_ms = int((time() - start) * 1000) if start else 0
        output = tool_response if isinstance(tool_response, str) else json.dumps(tool_response, default=str)

        self._buffer(session_id, Record(
            event_id=self._next_event_id(session_id, "step"),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="tool_result",
            category="Trace",
            interaction_id=self._interaction_ids.get(session_id, str(uuid4())),
            participant_id=self._agent_participant_id,
            content=None,
            variable={"Name": tool.name, "OutputValueText": output[:2000]},
            metadata={"source": "after_tool_callback", "duration_ms": duration_ms},
            timestamp=datetime.now(tz=timezone.utc),
        ))

    # ── Turn lifecycle ────────────────────────────────────────────────────────

    def _salvage_aborted_turn(self, session_id: str) -> None:
        """Flush a previous turn's buffer that never reached turn-end.

        The prior turn aborted (raised) after buffering records but before
        ``_on_turn_end`` fired, so its per-turn state lingered. Detach that
        state NOW — before the new turn re-seeds the buffer — so the aborted
        turn's records are neither silently dropped nor appended-into and
        cross-attributed to the new turn's ``interaction_id``.

        Best-effort: when a loop is running (the production path — this runs
        inside the async ``wrap_before_model`` callback) it schedules a tracked
        background flush of the salvaged records (which already carry the
        aborted turn's own ``interaction_id``); with no running loop (a direct
        sync/test call) it still clears the state so the new turn starts clean.
        ``_event_seq`` is deliberately NOT reset — it stays monotonic across the
        whole session so salvaged and new-turn event_ids never collide.
        """
        records = list(self._buffers.pop(session_id, []))
        self._interaction_ids.pop(session_id, None)
        self._invocation_ids.pop(session_id, None)
        if not records:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(self._async_ingest(session_id, records))
            self._flush_tasks.add(task)
            task.add_done_callback(self._flush_tasks.discard)
        else:
            logger.warning(
                "[ADK] aborted turn for session %s left %d buffered record(s) "
                "with no running loop to flush them; dropping to keep them from "
                "cross-attributing to the next turn.",
                session_id, len(records),
            )

    def _on_turn_end(self, session_id: str) -> None:
        """Sync path — only used when adapter is in sync mode."""
        records = list(self._buffers.pop(session_id, []))
        self._interaction_ids.pop(session_id, None)
        self._invocation_ids.pop(session_id, None)
        if not records:
            return
        mgr = self._manager_for(session_id)
        if mgr is not None:
            mgr.ingest(records)

    async def _on_turn_end_async(self, session_id: str) -> None:
        """Async path — awaits ingest so records are flushed before the turn returns."""
        records = list(self._buffers.pop(session_id, []))
        self._interaction_ids.pop(session_id, None)
        self._invocation_ids.pop(session_id, None)
        if not records:
            return
        await self._async_ingest(session_id, records)

    async def _on_after_model_async(self, callback_context: Any, llm_response: Any) -> None:
        """Async version of _on_after_model — awaits turn-end ingest directly."""
        session_id = self._get_session_id(callback_context)
        if not session_id:
            return

        text = self._extract_response_text(llm_response)
        is_tool_call = self._is_tool_call(llm_response)
        tokens = self._extract_tokens(llm_response)
        interaction_id = self._interaction_ids.get(session_id, str(uuid4()))

        if is_tool_call:
            if text:
                self._buffer(session_id, Record(
                    event_id=self._next_event_id(session_id, "step"),
                    session_id=session_id,
                    agent_id=self.agent_id,
                    event_type="agent_reasoning",
                    category="Trace",
                    interaction_id=interaction_id,
                    participant_id=self._agent_participant_id,
                    content=None,
                    variable={"OutputValueText": text},
                    metadata={"source": "after_model_callback", **tokens},
                    timestamp=datetime.now(tz=timezone.utc),
                ))
        else:
            if text:
                self._buffer(session_id, Record(
                    event_id=self._next_event_id(session_id, "msg"),
                    session_id=session_id,
                    agent_id=self.agent_id,
                    event_type="agent_response",
                    category="Message",
                    interaction_id=interaction_id,
                    participant_id=self._agent_participant_id,
                    content=text,
                    variable=None,
                    metadata={"source": "after_model_callback", **tokens},
                    timestamp=datetime.now(tz=timezone.utc),
                ))
            await self._on_turn_end_async(session_id)

    async def _async_ingest(self, session_id: str, records: list[Record]) -> None:
        mgr = self._manager_for(session_id)
        if mgr is None:
            return
        try:
            await mgr.ingest(records)
            # ingest() delivers on a fire-and-forget background task. An
            # environment that suspends the event loop the instant the HTTP
            # response returns (e.g. Cloud Run's request-scoped CPU) would never
            # let that task complete, so entries would never reach Beacon — the
            # conversation is created but stays empty (unlistable / 404 on
            # resume). Await the pending delivery NOW, while the turn's request
            # still holds CPU, so entries land.
            await mgr.flush()
        except Exception:
            logger.exception("Failed to ingest turn records to Beacon")

    # ── Multi-session lifecycle ───────────────────────────────────────────────

    async def _ensure_conversation(self, session_id: str) -> None:
        """Set up the Beacon conversation/session lifecycle for a new ADK session.

        Each ADK session gets a fresh conversation_id = session_id, matching the
        convention used by the LangGraph and Claude adapters. Cross-session memory
        will work once Beacon's owner-level context cache is fixed on the server side.

        Best-effort — never raises. Marks session as initialised either way so
        this only runs once per session.
        """
        self._sessions_initialised.add(session_id)
        mgr = self._manager_for(session_id)
        if mgr is None:
            return
        try:
            await mgr.create_session(session_id)
            logger.debug("_ensure_conversation: conv=%s session=%s owner=%s",
                         session_id, session_id, self.owner_id)
        except Exception as e:
            logger.warning("[ADK] _ensure_conversation failed (best-effort): %s", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _buffer(self, session_id: str, record: Record) -> None:
        self._buffers.setdefault(session_id, []).append(record)

    def _next_event_id(self, session_id: str, prefix: str) -> str:
        """Mint a deterministic, per-session-monotonic event_id.

        ``f"{prefix}-{session_id}-{seq}"`` where ``seq`` increments per event
        within the session. Stable across a re-read of the same source (a fresh
        adapter restarts at 0 and reproduces the same ids) — the contract the
        previous ``uuid4().hex[:8]`` ids violated. Local only; not sent to Beacon.
        """
        seq = self._event_seq.get(session_id, 0)
        self._event_seq[session_id] = seq + 1
        return f"{prefix}-{session_id}-{seq}"

    def _get_invocation_id(self, callback_context: Any) -> str | None:
        """Best-effort read of ADK's per-turn invocation id.

        ADK stamps one ``invocation_id`` per turn and reuses it across the
        re-entrant before_model calls of that turn, so it is the authoritative
        new-turn signal. Returns ``None`` when the attribute is absent (older
        ADK), and callers fall back to the legacy heuristic.
        """
        return getattr(callback_context, "invocation_id", None)

    def _get_session_id(self, callback_context: Any) -> str | None:
        session = getattr(callback_context, "session", None)
        if session is not None:
            return getattr(session, "id", None)
        return getattr(callback_context, "session_id", None)

    def _get_session_id_from_tool_context(self, tool_context: Any) -> str | None:
        session = getattr(tool_context, "session", None)
        if session is not None:
            return getattr(session, "id", None)
        return getattr(tool_context, "session_id", None)

    def _extract_user_message(self, llm_request: Any) -> str:
        try:
            for content in reversed(llm_request.contents or []):
                if getattr(content, "role", None) == "user":
                    parts = getattr(content, "parts", None) or []
                    texts = [p.text for p in parts if getattr(p, "text", None)]
                    if texts:
                        return " ".join(texts)
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning(
                "[ADK] _extract_user_message: %s on %s: %s",
                type(e).__name__, type(llm_request).__name__, e,
            )
            return ""
        return ""

    def _extract_response_text(self, llm_response: Any) -> str:
        try:
            if llm_response.content and llm_response.content.parts:
                return "".join(
                    p.text for p in llm_response.content.parts
                    if getattr(p, "text", None)
                )
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning(
                "[ADK] _extract_response_text: %s on %s: %s",
                type(e).__name__, type(llm_response).__name__, e,
            )
            return ""
        return ""

    def _is_tool_call(self, llm_response: Any) -> bool:
        try:
            return bool(
                llm_response.content and llm_response.content.parts and
                any(
                    hasattr(p, "function_call") and p.function_call
                    for p in llm_response.content.parts
                )
            )
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning(
                "[ADK] _is_tool_call: %s on %s: %s",
                type(e).__name__, type(llm_response).__name__, e,
            )
            return False

    def _extract_tokens(self, llm_response: Any) -> dict:
        try:
            if llm_response.usage_metadata:
                return {
                    "prompt_tokens": llm_response.usage_metadata.prompt_token_count or 0,
                    "output_tokens": llm_response.usage_metadata.candidates_token_count or 0,
                }
        except Exception as e:
            logger.warning(
                "[ADK] _extract_tokens failed on %s: %s",
                type(llm_response).__name__, e,
            )
        return {}
