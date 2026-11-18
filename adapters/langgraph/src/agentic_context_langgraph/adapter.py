# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unified LangGraph adapter — snapshot mode + streaming mode + observation hooks.

A single ``LangGraphAdapter`` class with two construction paths:

- ``graph=`` — live ``StateGraph``. Supports both ``snapshot()`` (reads
  ``graph.get_state(cfg).values["messages"]``) and ``stream()`` (drives
  ``astream_events(version="v3")`` and yields one ``Record`` per agent turn).
- ``pickle_path=`` — a serialized ``StateSnapshot`` written to disk. Supports
  ``snapshot()`` only; ``stream()`` raises ``ValueError``.

Adapter is the public façade. Application code calls ``snapshot``, ``ingest``,
``stream``, ``close``; the adapter delegates Beacon traffic to the injected
``SessionManager`` / ``AsyncSessionManager``. Adapter never POSTs directly.

Optional hooks fire at session/turn/tool boundaries. They are observation-only
— exceptions raised inside a hook are logged and swallowed; the stream loop
never aborts because of a hook.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
import pickle
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

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
    IngestionCallbacks,
    Record,
    SessionManager,
)
from agentic_context.retrieval import (
    CONTEXT_PACK_VERSION_GUIDANCE,
    SessionPrompt,
    context_pack_params,
    patch_pack_version_in_catalog,
    version_mismatch_entry,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from .events import ErrorEvent, ToolEvent, TurnEvent
from .policies import ContextPolicy, resolve_policy
from .retrieval_normalize import context_to_system_message

logger = logging.getLogger(__name__)


# Beacon ``AiAgentParticipant`` label for this runtime. Without it
# every agent lands under the SDK
# default "Custom" and analytics can't tell the frameworks apart.
_AGENT_PARTICIPANT_VARIABLE = {
    "AiAgentType": "LangGraph",
    "AiAgentVersionApiName": "v1",
    "AiAgentTemplateApiName": "default",
}


# Default system-prompt guidance injected under ``context_mode="auto"`` and
# ``"per-turn"``. Teaches the agent *when* to reach for the
# recall_customer_context tool. The tool takes no arguments and reads an
# in-process cache the adapter seeded on session start — one Beacon POST per
# session, period.
_RECALL_TOOL_SYSTEM_INSTRUCTION = """\
You have access to a memory tool called `recall_customer_context` that returns \
the customer memory already loaded for this session (profile basics, prior- \
conversation summaries, cross-session facts). It takes no arguments.

When to call it:
- When you want a reminder of what memory is available for this customer
- When grounding an answer in the customer's stored facts

The tool reads an in-process cache — it does NOT issue a new retrieval, so \
calling it is cheap and repeat calls within a session return the same payload. \
You do NOT need to call it for small talk or when the conversation already \
holds the answer.
"""


def _resolve_context_mode(mode: str) -> tuple[str, str | None]:
    """Facade → ``(context_policy slug, recall_tool_instruction)``.

    Maps each context mode to a point in the adapter's (policy, tool,
    instruction) axis space:

    - ``"auto"``     → seed fast once + tool on + built-in prompt
    - ``"per-turn"`` → seed fast EVERY turn + tool on + built-in prompt
    - ``"manual"``   → never seed + tool on + no prompt (customer owns it)

    ``per-turn`` uses ``EveryTurnPolicy`` (not ``NoFetchPolicy``) so the adapter
    injects a fast profile ``SystemMessage`` before every turn as a deterministic
    fallback. The injected block carries the stable ``context_message_id`` so the
    ``add_messages`` reducer replaces, not stacks, it across turns.

    The second element feeds :meth:`_resolve_recall_tool_instruction`, so
    ``None`` → built-in prompt and ``""`` → inject nothing, exactly as that
    method reads those two sentinels.
    """
    if mode == "auto":                       # seed once + tool + built-in prompt
        return "session_first", None
    if mode == "per-turn":                   # seed fast every turn + tool + built-in prompt
        return "every_turn", None
    if mode == "manual":                     # never seed + tool + no prompt (customer owns)
        return "no_fetch", ""
    raise ValueError(
        f"context_mode must be one of 'auto', 'per-turn', 'manual'; got {mode!r}."
    )


class LangGraphAdapter:
    """LangGraph adapter spanning snapshot reads and per-turn streaming.

    Construct with **exactly one** of ``graph`` or ``pickle_path``, plus a
    required ``session_manager`` (sync or async). The adapter binds its
    ``ingest`` / ``close`` methods to the manager's flavor at construction.

    All hooks are optional and read-only. ``persistent=False`` because store
    writes do not surface through ``astream_events`` — only working-tier
    turns flow through this adapter.
    """

    capabilities: Capabilities = Capabilities(working=True, persistent=False)

    # Process-shared policy state, keyed by (owner_id, agent_id, session_id).
    # Customers wire one short-lived adapter per HTTP request; without this
    # cache, ``_context_state`` would be empty every turn and ``last_fetched_at``
    # would always be ``None`` — i.e. ``SessionFirstPolicy`` would degrade to
    # one GET per request. Sharing across instances lets the policy honour its
    # contract regardless of adapter lifetime.
    #
    # Bounded by ``_MAX_SHARED_POLICY_STATE`` entries with strict LRU eviction
    # (an ``OrderedDict`` keyed by ``(owner_id, agent_id, session_id)``): every
    # read of an existing key bumps the entry to the most-recent end; inserts
    # evict the least-recently-used entry once the cap is exceeded. Without
    # this cap, long-running ASGI workers would accumulate one entry per
    # unique session forever, since the unified model never closes sessions.
    # No TTL — LRU only; if a customer needs eager cleanup they can call
    # :meth:`end_session`.
    #
    # Thread-safety: read-decide-write of ``last_fetched_at`` /
    # ``last_payload`` in :meth:`_maybe_fetch_and_inject_context` is guarded
    # by ``_SHARED_POLICY_LOCK`` so concurrent requests on the same key
    # serialize the decision. The lock is *never* held across ``await`` of
    # the GET; the second acquisition checks whether another worker already
    # wrote and skips the stale payload to preserve the "at most one GET per
    # session start" guarantee.
    _MAX_SHARED_POLICY_STATE: int = 1024
    _SHARED_POLICY_STATE: "OrderedDict[tuple[str, str, str], dict[str, Any]]" = OrderedDict()
    _SHARED_POLICY_LOCK: threading.Lock = threading.Lock()

    def __init__(
        self,
        agent_id: str | None = None,
        *,
        session_manager: SessionManager | AsyncSessionManager | None = None,
        # Only meaningful in manager-injection mode: lets the customer plug
        # their own already-entered AsyncRetrievalClient in so the injected
        # context-pack instruction can come from the server (otherwise the
        # adapter owns no retrieval client in that mode and always uses the
        # bundled fallback). Ignored in self-driving
        # mode, where the adapter builds and owns its own retrieval client.
        retrieval_client: AsyncRetrievalClient | None = None,
        # A compiled graph, or a builder ``(sdk_tools) -> compiled graph`` the
        # adapter calls with :meth:`default_tools` at the end of __init__.
        graph: Any | Callable[[list[BaseTool]], Any] | None = None,
        pickle_path: str | Path | None = None,
        allow_pickle: bool = False,
        # participant_id for user_message records. None (default) uses the
        # turn's resolved owner_id — Beacon only marks an entry role="owner"
        # when participantId == ownerId.
        user_participant_id: str | None = None,
        reasoning_classifier: Callable[[Any], Literal["reasoning", "reply"]] | None = None,
        owner_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dataspace: str | None = None,
        owner_type: str | None = None,
        secret_server_url: str | None = None,
        auth: ContextAuth | ApiKeyAuth | None = None,
        private_key: str = "",
        consumer_key: str = "",
        subject: str = "",
        login_url: str = "",
        # The single knob that picks a whole memory behavior. Defaults to
        # "auto", so a bare ``LangGraphAdapter(graph=…)`` is a self-driving
        # auto adapter with no further configuration.
        #   "auto"     — seed fast context once; the recall tool reads that
        #                cache on demand (built-in prompt).
        #   "per-turn" — seed fast context EVERY turn; recall tool reads the
        #                cache on demand (built-in prompt).
        #   "manual"   — never auto-seed; tool present, NO prompt injected (the
        #                customer owns their own prompting in their graph nodes).
        # Only meaningful in self-driving mode: an injected ``session_manager=``
        # owns retrieval, so the adapter seeds nothing and installs no tool
        # regardless of this value (see the resolution below). For fetch
        # cadences the three modes don't cover (time-decay, topic-shift, a custom
        # ContextPolicy), drive retrieval yourself with ``make_context_node`` and
        # the exported policy classes.
        context_mode: Literal["auto", "per-turn", "manual"] = "auto",
        context_header: str = "Previous context about this customer",
        context_message_id: str | None = "agentic-memory-context",
        context_dedupe: bool = False,
        context_rank_urgent: bool = False,
        on_session_start: Callable[[str], None] | None = None,
        on_session_end: Callable[[str], None] | None = None,
        on_turn_start: Callable[[TurnEvent], None] | None = None,
        on_tool_start: Callable[[ToolEvent], None] | None = None,
        on_tool_end: Callable[[ToolEvent], None] | None = None,
        on_error: Callable[[ErrorEvent], None] | None = None,
        ingestion_callbacks: IngestionCallbacks | None = None,
        # Full-override for Beacon's AiAgentParticipant block.
        # None = AiAgentType="LangGraph" + agent_id. Self-driving mode only —
        # an injected session_manager carries its own.
        agent_participant_variable: dict[str, Any] | None = None,
    ) -> None:
        if graph is None and pickle_path is None:
            raise ValueError(
                "LangGraphAdapter requires exactly one of graph= or pickle_path="
            )
        if graph is not None and pickle_path is not None:
            raise ValueError(
                "LangGraphAdapter requires exactly one of graph= or pickle_path=, not both"
            )

        connection_params_set = any(
            v is not None for v in (owner_id, api_key, base_url, dataspace)
        )
        if session_manager is not None and connection_params_set:
            raise ValueError(
                "Pass session_manager= OR connection params "
                "(owner_id/api_key/base_url/dataspace), not both"
            )
        if session_manager is not None and agent_participant_variable is not None:
            raise ValueError(
                "agent_participant_variable= has no effect with an injected "
                "session_manager=; pass it to your SessionManager instead"
            )
        # Full-override for Beacon's AiAgentParticipant block; None = this
        # runtime's label (see _resolve_agent_participant_variable).
        self._agent_participant_variable = (
            dict(agent_participant_variable) if agent_participant_variable is not None else None
        )
        # Resolve context_mode into the two private attrs the fetch/tool/prompt
        # paths read (context_policy slug, recall tool instruction).
        # In self-driving mode the mode fully decides behavior. In manager-
        # injection mode the customer owns retrieval: the adapter seeds nothing
        # and installs no tool, so the injected manager's own wiring drives
        # context. .context_mode still reports the resolved mode for
        # introspection either way.
        self._context_mode: Literal["auto", "per-turn", "manual"] = context_mode
        if session_manager is None:
            # Raises on an invalid slug ("must be one of 'auto', ...").
            context_policy, recall_tool_instruction = _resolve_context_mode(
                context_mode
            )
            self._install_recall_tool = True
        else:
            # Validate the slug even though the mode is inert here, so a typo is
            # caught at construction rather than silently ignored.
            _resolve_context_mode(context_mode)
            context_policy, recall_tool_instruction = None, None
            self._install_recall_tool = False
        self._recall_tool_instruction = recall_tool_instruction

        if retrieval_client is not None and session_manager is None:
            raise ValueError(
                "retrieval_client= is only meaningful in manager-injection "
                "mode (alongside session_manager=); in self-driving mode "
                "the adapter builds and owns its own retrieval client."
            )
        self._self_driving = session_manager is None
        self._owned_ingest_client: AsyncIngestionClient | None = None
        self._owned_retrieval_client: AsyncRetrievalClient | None = None
        # Caller-supplied retrieval client for manager-injection mode: used
        # ONLY by the context-pack instruction fetch, not
        # by the ingestion path (which owns session_manager=). Kept
        # separate from _owned_retrieval_client so its lifecycle (open +
        # close) stays with the caller.
        self._injected_retrieval_client: AsyncRetrievalClient | None = (
            retrieval_client
        )
        # Server-side context_pack_instruction prompt, fetched on
        # a session's first turn and pinned for that session.
        self._context_pack_prompt = SessionPrompt(CONTEXT_PACK_INSTRUCTION)
        # In self-driving mode the adapter spins up one AsyncSessionManager
        # per (session_id, owner) seen — managers are bound to one session +
        # owner each, but the customer expects to drive any number of sessions
        # (and, with per-call owner_id, any number of owners) through a single
        # adapter instance. Keying by (session_id, owner) keeps two tenants who
        # happen to reuse a session string from sharing a manager.
        self._owned_managers: dict[tuple[str, str], AsyncSessionManager] = {}
        # Resolved owner for each session_id seen by stream()/run(). Lets the
        # owner-less read paths (the recall tool) — which only receive a
        # session_id from the RunnableConfig — recover the tenant that turn
        # belongs to.
        self._session_owner: dict[str, str] = {}
        # stream() turns that haven't been ingested yet. close() and
        # end_session() flush them first, so a turn the caller abandoned
        # (break without aclose) still lands before its manager closes.
        self._active_turns: set[_TurnBatch] = set()
        self._allowed_pack_ids: set[str] = set()
        self._allowed_context_pack_list: str = f"{CONTEXT_PACK_LIST_MARKER}\n{{}}"
        # Cached pack-id -> catalog version. Populated from the fetched catalog
        # (ints only, never nulls or bools) so the get_context_pack handler can
        # fill it in for the model when the model omits context_pack_version
        # (enforcing the MUST declared in CONTEXT_PACK_VERSION_GUIDANCE at the
        # SDK boundary). Patched on status=version_expired so subsequent calls
        # pin to the new revision without a catalog refresh.
        self._pack_versions: dict[str, int] = {}

        if self._self_driving:
            if pickle_path is not None:
                raise ValueError(
                    "Self-driving mode requires graph=; pickle_path= is not "
                    "supported here. Pass a session_manager= explicitly to use "
                    "pickle_path."
                )
            # Resolve auth source first: a shared ContextAuth wins, else build
            # one from the four raw credential kwargs, else None (static-token
            # path). When auth is present, base_url + the bearer token are
            # fetched in __aenter__; callers don't pass api_key/base_url.
            if auth is not None:
                self._auth: ContextAuth | ApiKeyAuth | None = auth
            else:
                inline_creds = {
                    "private_key": private_key,
                    "consumer_key": consumer_key,
                    "subject": subject,
                    "login_url": login_url,
                }
                # Treat any truthy value as "set" — empty strings are the
                # default sentinel here, so partial provisioning surfaces
                # as a partially-truthy dict. Reject all-or-nothing so
                # callers can't accidentally fall through to the static-
                # token path when they meant to wire ContextAuth.
                provided = {k for k, v in inline_creds.items() if v}
                if provided and len(provided) < len(inline_creds):
                    raise ValueError(
                        "Inline ContextAuth credentials are all-or-nothing: "
                        "pass all of private_key, consumer_key, subject, "
                        "login_url, or none of them."
                    )
                if provided:
                    self._auth = ContextAuth(
                        private_key=private_key,
                        consumer_key=consumer_key,
                        subject=subject,
                        login_url=login_url,
                    )
                elif api_key and not base_url and not os.environ.get("CDP_BASE_URL"):
                    # Demo path: only an api_key, no base_url and no
                    # raw creds → resolve every credential from the secret
                    # server. ApiKeyAuth fetches the bundle lazily in
                    # __aenter__ and behaves like ContextAuth thereafter (token
                    # exchange, refresh, 401 retry). base_url and agent_id both
                    # come from that bundle, so callers pass neither here.
                    self._auth = ApiKeyAuth(
                        api_key, secret_server_url=secret_server_url
                    )
                    # Consume api_key so the static-token branch below doesn't
                    # also treat it as a raw CDP bearer token.
                    api_key = None
                else:
                    self._auth = None

            # Env-var fallback for the static-token path only (kept for
            # 12-factor deployments that already wired CDP_TOKEN / CDP_BASE_URL
            # before ContextAuth landed). Explicit kwargs always win.
            api_key = api_key or os.environ.get("CDP_TOKEN")
            base_url = base_url or os.environ.get("CDP_BASE_URL")
            dataspace = dataspace or os.environ.get("CDP_DATASPACE")
            # agent_id can be deferred to __aenter__ only on the ApiKeyAuth
            # (demo) path, where it may arrive from the secret bundle's
            # ``agentId`` field. Every other path requires it up front.
            self._agent_id_deferred = (
                isinstance(self._auth, ApiKeyAuth) and agent_id is None
            )
            # Treat empty strings as missing — `Bearer ` headers and ""
            # dataspaces are silent footguns at the wire level.
            #
            # owner_id is NO LONGER required at construction: a
            # multi-tenant server can construct one adapter and pass a
            # per-request owner_id to stream()/run() instead. It must still be
            # resolvable *somewhere* by the time a turn runs — that check lives
            # in _resolve_owner(). What stays required here is the connection
            # itself (agent_id + api_key/base_url unless auth handles them).
            required: list[tuple[str, Any]] = []
            if not self._agent_id_deferred:
                required.insert(0, ("agent_id", agent_id))
            if self._auth is None:
                required += [("api_key", api_key), ("base_url", base_url)]
            missing = [name for name, val in required if not val]
            if missing:
                # agent_id has no env-var fallback and is required in every
                # path; call it out by itself so the error is unambiguous.
                if "agent_id" in missing:
                    raise ValueError(
                        "agent_id is required; pass agent_id= to LangGraphAdapter(...)"
                    )
                raise ValueError(
                    "Self-driving LangGraphAdapter requires: "
                    f"{', '.join(missing)}. Either supply these connection "
                    "parameters or pass session_manager= directly."
                )
            resolved_dataspace = dataspace if dataspace else "default"
            self._resolved_dataspace = resolved_dataspace
            if self._auth is None:
                assert api_key
                assert base_url
                self._owned_ingest_client = AsyncIngestionClient(
                    base_url=base_url, api_key=api_key, dataspace=resolved_dataspace
                )
                self._owned_retrieval_client = AsyncRetrievalClient(
                    base_url=base_url, api_key=api_key, dataspace=resolved_dataspace
                )
            # When self._auth is set, clients are built lazily in __aenter__
            # once aget_token() has resolved (token, base_url).
            self._mgr: SessionManager | AsyncSessionManager | None = None
            self._is_async = True
            # Normalize to "" so the owner-less read paths (_owner_for_session)
            # never surface None; _resolve_owner treats "" as "not supplied".
            self._owner_id = owner_id or ""
        else:
            assert session_manager is not None
            if agent_id is None:
                raise ValueError(
                    "agent_id is required; pass agent_id= to LangGraphAdapter(...)"
                )
            self._mgr = session_manager
            self._is_async = isinstance(session_manager, AsyncSessionManager)
            mgr_owner = getattr(session_manager, "owner_id", None)
            self._owner_id = mgr_owner if isinstance(mgr_owner, str) else ""
            # Manager-injection paths don't own clients; auth, if passed, is
            # already wired into the caller's IngestionClient. Stored only so
            # the attribute always exists on the instance.
            self._auth = auth
            self._agent_id_deferred = False

        # agent_id is required in every path except the ApiKeyAuth demo path,
        # where it can arrive from the secret bundle in __aenter__. When
        # deferred it holds "" until then; _agent_participant_id is re-derived
        # once the real id is known.
        if not self._agent_id_deferred:
            assert agent_id is not None  # narrowed: required otherwise
        self.agent_id: str = agent_id or ""
        self._graph: Any | None = graph
        self._pickle_path: Path | None = (
            Path(pickle_path) if pickle_path is not None else None
        )
        self._allow_pickle = allow_pickle
        self._user_participant_id = user_participant_id
        self._agent_participant_id = self.agent_id
        self._reasoning_classifier = reasoning_classifier
        self._on_session_start = on_session_start
        self._on_session_end = on_session_end
        self._on_turn_start = on_turn_start
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._on_error = on_error
        # Forwarded to each per-session AsyncSessionManager built in
        # self-driving mode (see _mgr_for). In manager-injection mode the
        # customer owns the manager and wires callbacks directly; this kwarg
        # is stored but never consumed in that path.
        self._ingestion_callbacks = ingestion_callbacks
        # owner_type defaults to the auth class's advertised value (individual
        # for service-account auth, employee for interactive user auth) when the
        # caller doesn't pass one explicitly.
        self._owner_type = (
            owner_type
            if owner_type is not None
            else getattr(self._auth, "default_owner_type", "individual")
        )

        # Smart-fetch state (only consulted in self-driving mode).
        self._policy: ContextPolicy = resolve_policy(context_policy)
        self._context_header = context_header
        self._context_message_id = context_message_id
        self._context_dedupe = context_dedupe
        self._context_rank_urgent = context_rank_urgent
        # Per-instance fallback used only when this adapter is NOT self-driving
        # (the legacy ``session_manager=`` path doesn't have an owner_id we
        # could safely use as a process-wide key).
        self._context_state: dict[str, dict[str, Any]] = {}
        self._aenter_called: bool = False

        # Per-session turn counters and the in-flight interaction_id for
        # the currently-running stream() call on a given session_id. Keeping
        # these per-session lets one adapter drive multiple sessions
        # concurrently without misrouting tool records to the wrong turn.
        self._turn_idx: dict[str, int] = {}
        self._current_interaction_id: dict[str, str] = {}
        self._last_session_id: str | None = None

        # Bind ingest/close to the manager's flavor at __init__. The two
        # flavors have different signatures (sync vs awaitable), so this is
        # intentionally untyped on the public attribute.
        if self._is_async:
            self.ingest = self._async_ingest
            self.close = self._async_close
        else:
            self.ingest = self._sync_ingest  # type: ignore[assignment]
            self.close = self._sync_close  # type: ignore[assignment]

        # graph= may be a builder: a callable taking the SDK tools and returning
        # a compiled graph. This lets the customer bind default_tools() into
        # their model before compiling, which a pre-built graph can't do. Runs
        # last so the tools' closures see a fully initialized adapter. A
        # compiled graph (or a snapshot-only wrapper) has astream_events or
        # get_state; a plain function has neither.
        if (
            callable(graph)
            and not hasattr(graph, "astream_events")
            and not hasattr(graph, "get_state")
        ):
            built = graph(self.default_tools())
            if inspect.iscoroutine(built):
                built.close()  # never awaited; close it to avoid the RuntimeWarning
                raise TypeError(
                    "graph= builder must be a plain (sync) function; it returned "
                    "a coroutine. Remove `async` from the builder."
                )
            if not (hasattr(built, "astream_events") and hasattr(built, "get_state")):
                raise TypeError(
                    "graph= builder must return a compiled graph (call "
                    f".compile() on your StateGraph); got {type(built).__name__}"
                )
            self._graph = built

    @property
    def is_self_driving(self) -> bool:
        """``True`` when the adapter owns its clients (path 3 / all-in-one)."""
        return self._self_driving

    @property
    def context_policy(self) -> ContextPolicy:
        """The resolved :class:`ContextPolicy` driving auto-fetch decisions."""
        return self._policy

    @property
    def context_mode(self) -> Literal["auto", "per-turn", "manual"]:
        """The resolved ``context_mode`` — ``"auto"`` (the default), ``"per-turn"``,
        or ``"manual"``.

        In manager-injection mode this still reports the constructed value, but
        the mode is inert: the injected ``session_manager`` owns retrieval, so
        the adapter seeds no context and installs no tool. The effective
        fetch state is always introspectable via :attr:`context_policy`.
        """
        return self._context_mode

    # ─── owner resolution (dynamic per-session owner_id) ──────

    def _resolve_owner(self, owner_id: str | None) -> str:
        """Resolve the effective owner for a turn: per-call → constructor → raise.

        Per-call owner_id lets a multi-tenant server construct one adapter and
        pass a per-request ``owner_id`` to :meth:`stream` / :meth:`run`.
        Precedence:

        1. the ``owner_id`` passed to this turn (wins — it's the most specific),
        2. else the constructor ``owner_id`` (the single-tenant default),
        3. else ``ValueError`` — neither was supplied, so we have no tenant to
           scope Beacon traffic to and must not silently post to a blank owner.

        Empty strings count as "not supplied" at both levels (a ``""`` owner is
        a wire-level footgun, same rationale as the connection-param checks).
        """
        resolved = owner_id or self._owner_id
        if not resolved:
            raise ValueError(
                "owner_id is required but was not provided: pass owner_id= to "
                "stream()/run() for this turn, or set it once on "
                "LangGraphAdapter(owner_id=...)."
            )
        return resolved

    def _owner_for_session(self, session_id: str | None) -> str:
        """Best-effort owner for an owner-less read path (recall tool).

        The recall tool receives only a ``session_id`` (from the run's
        ``RunnableConfig``), never an owner. Recover the tenant from the map
        ``stream()`` populated for that session; fall back to the constructor
        default when the session hasn't been seen (e.g. a direct recall call
        outside a stream). Returns ``""`` when nothing resolves — callers
        treat that as "no owner", degrading to an empty read rather than
        raising, matching the best-effort contract of those paths.
        """
        if session_id and session_id in self._session_owner:
            return self._session_owner[session_id]
        return self._owner_id

    def _user_participant_for(self, session_id: str, owner: str | None = None) -> str:
        """participant_id for a user_message: explicit kwarg → turn owner → session owner → "user".

        ``stream()`` passes its turn's resolved ``owner`` so a concurrent turn
        for the same session under another owner (which overwrites the shared
        ``_session_owner`` map while this turn awaits its context fetch) can't
        leak its id into this turn's record. The "user" literal is the last
        resort only when no owner resolves (manager-injection with a manager
        that doesn't expose ``owner_id``).
        """
        if self._user_participant_id:
            return self._user_participant_id
        return owner or self._owner_for_session(session_id) or "user"

    # ─── on-demand get_context tool ──────────────────────────────────

    def make_context_retrieval_tool(self) -> StructuredTool:
        """Return a LangChain tool that lets the LLM pull cross-session memory on demand.

        This is the on-demand counterpart to the adapter's per-turn context
        injection (the policy-driven ``GET`` + ``SystemMessage`` prepend that
        still runs before every turn in self-driving mode — see
        :attr:`context_policy`). The tool does not change or replace that path;
        it adds a lever the *agent* can pull mid-turn when it decides it needs
        the customer's history. Splat it into your agent's tool list — or use
        :meth:`default_tools` to get all SDK-provided tools at once. Bind them
        in a ``graph=`` builder, which the adapter calls with ``default_tools()``::

            def build_graph(sdk_tools):
                tools = [*sdk_tools, *my_tools]
                return make_my_graph(model.bind_tools(tools), ToolNode(tools))

            adapter = LangGraphAdapter(graph=build_graph, ...)

        The tool's docstring is the LLM-visible description, so it drives *when*
        the model reaches for memory. The ``session_id`` is resolved from the
        per-invocation ``RunnableConfig`` (LangGraph threads the run's
        ``configurable.thread_id`` — which equals the SDK ``session_id`` — into
        every tool call), so the LLM never has to supply a conversation id and
        concurrent multi-session streams never read one another's identity.

        Implementation note: the tool is a **zero-argument cache read** under
        the self-seeding modes. The adapter's policy-driven fetch has already
        seeded this session's memory into ``_SHARED_POLICY_STATE`` (one GET
        per session under ``SessionFirstPolicy``, refreshed per turn under
        ``EveryTurnPolicy``), and the tool just returns that payload. There
        is **no GET** on the tool path for ``auto`` / ``per-turn``, so a
        reflexive model that calls it every turn costs zero extra Beacon
        traffic.

        Under ``context_mode="manual"`` the policy is ``NoFetchPolicy`` and
        the cache is never seeded, so on a cache miss the tool falls through
        to a single ``get_context`` POST and returns that payload — otherwise
        manual mode would leave the tool permanently empty. This fallback
        runs only when ``context_mode == "manual"``; it leaves ``auto`` /
        ``per-turn`` unchanged.

        Best-effort by contract: returns ``{"userContext": []}`` without raising
        whenever there is no cached payload (no seed in ``auto`` /
        ``per-turn``, the manual-mode fallback declined, manager-injection
        mode) or the ``session_id`` can't be resolved.
        """
        adapter = self

        async def recall_customer_context(
            *, config: RunnableConfig,
        ) -> dict[str, Any]:
            """Recall this customer's memory that was loaded for this session.

            Returns the customer memory the adapter already fetched for this
            session — profile basics, prior-conversation summaries, and any
            cross-session facts loaded on session start. Takes no arguments.
            Safe to call reflexively: it reads an in-process cache and never
            issues a new retrieval, so repeat calls within a session cost
            nothing.

            Use this when you want a reminder of what memory is loaded. If you
            need data that is NOT in this cache (specific enrollments, policy
            details, handbook passages, etc.), use ``get_context_pack`` with the
            matching ``context_pack_id`` from the available-packs list instead
            — that tool executes server-side queries and returns new data.

            Returns:
                A dict with a ``userContext`` list of memory entries. An empty
                list means no memory has been loaded for this session yet.
            """
            # Resolve (session_id, owner) from THIS invocation's config: the
            # run's thread_id == session_id, and stream() stamped the owner
            # alongside it. Reading the owner from the per-invocation config
            # — not the process-shared _session_owner map — is what makes
            # concurrent same-session/different-owner turns read the right
            # tenant's cache entry.
            configurable = (config or {}).get("configurable") or {}
            sid = configurable.get("thread_id") or adapter._last_session_id
            owner = configurable.get("agentic_owner_id")
            if owner is None:
                owner = adapter._owner_for_session(sid)

            payload: dict[str, Any] | None = None
            if sid:
                state = adapter._policy_state_get(sid, owner)
                if state is not None:
                    cached = state.get("last_payload")
                    if isinstance(cached, dict):
                        payload = cached

            has_context = bool(payload and payload.get("userContext"))

            # Manual-mode fallback. Under context_mode="manual" the policy is
            # NoFetchPolicy, so the cache is never seeded and this tool would
            # otherwise return an empty list forever. Fall through to a single
            # on-demand get_context POST so the tool remains the retrieval
            # lever the manual-mode contract advertises.
            if not has_context and sid and adapter._context_mode == "manual":
                fetched = await adapter._fetch_recall_context(
                    session_id=sid, owner=owner
                )
                if isinstance(fetched, dict) and fetched.get("userContext"):
                    payload = fetched
                    has_context = True

            logger.info(
                "[LangGraph] recall_customer_context: owner=%r session_id=%r "
                "→ has_context=%s",
                owner, sid, has_context,
            )
            if has_context and payload is not None:
                # Deep-copy so a caller or framework that mutates the tool
                # result cannot write through into the shared policy cache.
                return copy.deepcopy(payload)
            return {"userContext": []}

        return StructuredTool.from_function(coroutine=recall_customer_context)

    def make_get_context_pack_tool(
        self,
        valid_pack_ids: list[str],
        *,
        param_defaults: dict[str, Any] | None = None,
        timeout: float | None = CONTEXT_PACK_TIMEOUT_S,
    ) -> StructuredTool:
        """Return a LangChain tool that retrieves a named CONTEXT PACK.

        This is the pack-scoped read (POST GetContext): the
        agent asks for one server-advertised ``context_pack`` of structured detail
        instead of the owner-scoped profile read :meth:`make_context_retrieval_tool` does. It is
        the LangGraph counterpart of the ADK/Claude ``make_get_context_pack_tool``
        — the same capability in every adapter, so any agent built from QUICKSTART
        can offer packs. :meth:`default_tools` already includes it (accepting
        any server-advertised pack id), so a ``graph=`` builder receives it with
        the memory tool; :meth:`stream` adds the context-pack instruction and
        pack catalog as a ``SystemMessage`` every turn. Don't also add a second
        instance alongside ``default_tools()`` — tool names must be unique::

            def build_graph(sdk_tools):  # sdk_tools includes get_context_pack
                return make_my_graph(model.bind_tools([*sdk_tools, *my_tools]))

            adapter = LangGraphAdapter(graph=build_graph, ...)

        The ``session_id`` (== the run's ``configurable.thread_id``) and owner are
        resolved from the per-invocation ``RunnableConfig`` LangGraph threads into
        every tool call — the same way :meth:`make_context_retrieval_tool` does — so the
        LLM never supplies an owner and concurrent multi-session streams never read
        one another's identity. ``owner_id`` is bound server-side, never from the
        model.

        Args:
            valid_pack_ids: the pack ids this agent advertises. If agent
                cannot fetch self._allowed_pack_ids, an id outside this
                set is rejected with ``{"status": "invalid_pack"}`` without a call.
            param_defaults: fixed pack ``params`` merged into every call. The
                model's ``params`` override those keys. A non-empty ``query`` is
                added as ``params["query"]`` only when ``params`` did not already
                set it. Blank values are omitted.
            timeout: per-call HTTP timeout for the pack POST; defaults
                to :data:`agentic_context.CONTEXT_PACK_TIMEOUT_S`.

        Best-effort by contract: it NEVER raises. Beyond ``invalid_pack`` it
        returns ``{"status": "ok", "context": {...}}`` on success, or
        ``{"status": "not_ready", "detail": ...}`` on any failure (no owned
        retrieval client — manager-injection mode or pre-``__aenter__`` — endpoint
        not deployed, or timeout) — matching the PREVIEW contract in
        :data:`agentic_context.CONTEXT_PACK_INSTRUCTION`.
        """
        adapter = self
        defaults = dict(param_defaults or {})

        async def get_context_pack(
            context_pack_id: str,
            params: dict[str, str] | None = None,
            query: str = "",
            context_pack_version: int | None = None,
            *,
            config: RunnableConfig,
        ) -> dict[str, Any]:
            """Retrieve one named context pack for this subject.

            This is the same read as getContext(contextPack). The available-packs
            block (the ``agentic-context-pack-list`` system message) lists each
            pack's description and its data needs. If a pack or one of its data
            needs would help answer the user's question, call this with that
            pack id.

            Use this when the memory already loaded for the session (which you
            can inspect cheaply via ``recall_customer_context``) does not
            contain the detail you need — enrollments, dependents, handbook
            passages, claims, etc. This tool makes a POST call per invocation,
            so prefer the recall tool when the answer is already in cache.

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
            # bypass versioning).
            if context_pack_version is not None and (
                isinstance(context_pack_version, bool)
                or not isinstance(context_pack_version, int)
            ):
                logger.warning(
                    "[LangGraph] get_context_pack: %r rejected non-integer "
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

            # Enforce CONTEXT_PACK_VERSION_GUIDANCE at the SDK boundary: when
            # the model omits ``context_pack_version`` and the cached catalog
            # has one for this pack, fill it in.
            if context_pack_version is None:
                context_pack_version = adapter._pack_versions.get(context_pack_id)

            allowed = adapter._allowed_pack_ids or set(valid_pack_ids)

            if context_pack_id not in allowed:
                logger.info("[LangGraph] get_context_pack: invalid pack id %r", context_pack_id)
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
                logger.info("[LangGraph] get_context_pack: %r → not_ready (%s)", context_pack_id, reason)
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

            # Only the self-driving path owns a retrieval client; under
            # manager-injection (or before __aenter__) there's nothing to call.
            client = adapter._owned_retrieval_client
            if client is None:
                return not_ready("no owned retrieval client (manager-injection mode)")

            # Resolve the session_id AND owner from THIS invocation's config (the
            # run's thread_id == session_id, and the owner stream() stamped
            # alongside it) — the same resolution make_context_retrieval_tool uses.
            configurable = (config or {}).get("configurable") or {}
            sid = configurable.get("thread_id") or adapter._last_session_id
            if not sid:
                return not_ready("no active conversation for this turn")
            owner = configurable.get("agentic_owner_id")
            resolved_owner = owner if owner is not None else adapter._owner_for_session(sid)
            if not resolved_owner:
                # Parity with Claude/ADK: no owner resolved for this turn → skip
                # the round-trip (the server would only reject a blank owner).
                return not_ready("no owner for this turn")

            # owner_id is bound server-side from the resolved owner — never taken
            # from the model. Blank parameter values are omitted so a data need
            # that requires them is skipped rather than queried empty.
            sent = context_pack_params(params, query, defaults)

            logger.info(
                "[LangGraph] get_context_pack: %r params=%s version=%s",
                context_pack_id, sorted(sent), context_pack_version,
            )
            try:
                payload = await client.get_context_pack(
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
                # Log the full exception server-side; return only its TYPE. A bare
                # httpx timeout stringifies to "", so the type keeps the not_ready
                # detail diagnosable, while repr(e) — which can carry request URLs,
                # tokens, or upstream error text — never reaches the model.
                logger.warning(
                    "[LangGraph] get_context_pack: %r request failed",
                    context_pack_id, exc_info=True,
                )
                return not_ready(f"request failed: {type(e).__name__}")

            mismatch = version_mismatch_entry(payload)
            if mismatch is not None:
                logger.info(
                    "[LangGraph] get_context_pack: %r → version_expired "
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

            logger.info("[LangGraph] get_context_pack: %r → ok", context_pack_id)
            return {"status": "ok", "context_pack_id": context_pack_id, "context": payload}

        # Splice the shared versioning guidance into the docstring that drives
        # the LLM-visible tool description. Done post-def because a Python
        # docstring has to be a string literal.
        if get_context_pack.__doc__ is not None:
            get_context_pack.__doc__ = get_context_pack.__doc__.replace(
                "__VERSION_GUIDANCE_PLACEHOLDER__", CONTEXT_PACK_VERSION_GUIDANCE,
            )
        return StructuredTool.from_function(coroutine=get_context_pack)

    async def get_context_pack_instruction(self, session_id: str) -> str:
        """Return the context-pack instruction pinned for ``session_id``.

        :meth:`stream` / :meth:`run` already add this (with the pack
        catalog) as a ``SystemMessage`` every turn, so most integrations
        never call it. The first call for a session fetches the
        server-published prompt
        (``GET /api/v1/agentic/prompts/system/context_pack_instruction``);
        a miss or failure pins the bundled
        :data:`agentic_context.CONTEXT_PACK_INSTRUCTION` for that session.
        With no retrieval client (before ``async with`` under ``auth=``, or
        manager-injection without ``retrieval_client=``) it returns the
        bundled text without pinning.
        """
        # Prefer the caller-supplied client (manager-injection mode) over
        # the adapter-owned one. Explicit ``is not None``: a truthy test
        # would skip an injected client whose ``__bool__`` returns False.
        client = (
            self._injected_retrieval_client
            if self._injected_retrieval_client is not None
            else self._owned_retrieval_client
        )
        return await self._context_pack_prompt.get(session_id, client)

    def default_tools(self) -> list[BaseTool]:
        """SDK-provided tools to splat into your agent's tool list.

        These are the tools the SDK auto-installs when a customer wires the
        adapter into their agent. A ``graph=`` builder receives this list and
        spreads it alongside the customer's own tools::

            def build_graph(sdk_tools):
                return make_my_graph(model.bind_tools([*sdk_tools, *my_tools]))

        Returns two tools in every self-driving ``context_mode`` (``auto`` /
        ``per-turn`` / ``manual`` all install both):

        - ``recall_customer_context`` — zero-arg cache read of the memory the
          adapter already seeded for this session. Cheap, idempotent, no HTTP.
          Call it for a reminder of what memory is loaded.
        - ``get_context_pack`` — pack-scoped POST read keyed by a server-
          advertised ``context_pack_id``. Call it to fetch structured detail
          not already in the cache (enrollments, dependents, handbook passages,
          etc.). The ids come from the ``agentic-context-pack-list``
          SystemMessage; invalid ids are rejected without a round trip.

        Returns ``[]`` under manager-injection (the injected ``session_manager``
        owns retrieval, so there's no owned client to back either tool). The
        individual factories (``make_context_retrieval_tool`` /
        ``make_get_context_pack_tool``) remain public for callers who want a
        tool regardless of mode.
        """
        if not self._install_recall_tool:
            return []
        return [
            self.make_context_retrieval_tool(),
            self.make_get_context_pack_tool([]),
        ]

    def _resolve_recall_tool_instruction(self) -> str | None:
        """Return the mode's system instruction, or ``None`` to inject nothing.

        The instruction slot is set by ``context_mode`` in :func:`_resolve_context_mode`:

        - tool off (manager-injection)     → ``None`` (never inject)
        - ``auto``     (instruction ``None``) → the built-in default prompt
        - ``per-turn`` (instruction ``None``) → the built-in default prompt
        - ``manual``   (instruction ``""``)   → ``None`` (tool registered, no
          prompting — the customer owns it)
        """
        if not self._install_recall_tool:
            return None
        if self._recall_tool_instruction is None:
            return _RECALL_TOOL_SYSTEM_INSTRUCTION
        if self._recall_tool_instruction == "":
            return None
        return self._recall_tool_instruction

    def _maybe_inject_recall_tool_instruction(self, inputs: Any) -> Any:
        """Prepend the recall-tool guidance SystemMessage to ``inputs`` when set.

        No-op unless ``inputs`` is a dict and
        :meth:`_resolve_recall_tool_instruction` returns a non-empty string.
        The message carries a stable id so the ``add_messages`` reducer
        replaces rather than duplicates it across the turns of one thread.
        """
        instruction = self._resolve_recall_tool_instruction()
        if not instruction or not isinstance(inputs, dict):
            return inputs
        msg = SystemMessage(
            content=instruction, id="agentic-memory-recall-tool-instruction"
        )
        return {**inputs, "messages": [msg, *inputs.get("messages", [])]}

    async def get_context_pack_list(self) -> dict[str, Any] | None:
        """Fetch this agent's available CONTEXT PACK list (get_context_pack_list).

        Best-effort — returns ``None`` when there is no owned retrieval client
        (manager-injection mode) or the read fails.
        """
        client = self._owned_retrieval_client
        if client is None:
            return None
        try:
            return await client.get_context_pack_list(agent_id=self.agent_id)
        except Exception as e:
            logger.warning(
                "[LangGraph] get_context_pack_list failed (best-effort); "
                "returning None: %s: %s", type(e).__name__, e,
            )
            return None

    async def _maybe_inject_context_pack_list(
        self, inputs: Any, *, is_open: bool = False, session_id: str | None = None
    ) -> Any:
        """Fetches and injects this agent's authorized CONTEXT PACK list into system instructions.

        Caches the fetched CONTEXT PACK list into `_allowed_pack_ids`
        to reuse on every turn after that session's open and assemble system instructions (SystemMessage).
        Refetches this list again at every session-start.

        This is a deterministic (not LLM-decided) call for the SDK once
        to inject the list of authorized context packs a given agent can utilize.
        """
        if not isinstance(inputs, dict):
            return inputs
        if is_open:
            new_context_pack_list = await self.get_context_pack_list()
            if new_context_pack_list is not None:
                new_context_pack_id_map = {}
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
                    # Surface `version` to the LLM when the server reports one
                    # so it can echo it back on the call as
                    # ``context_pack_version`` and recover from
                    # ``version_expired``. Only int values are surfaced;
                    # ``null`` means the server has no revision for this pack
                    # and bool is an int subclass so both are guarded out.
                    # Leaking null/bool into the catalog block would satisfy
                    # the "entry includes a `version`" condition in
                    # CONTEXT_PACK_VERSION_GUIDANCE and make the LLM echo a
                    # value the SDK would then reject.
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
        # The instruction rides in the same SystemMessage as the catalog it
        # explains (and is left out when there are no packs to explain); the
        # stable id lets add_messages replace it each turn.
        instruction = None
        if self._allowed_pack_ids:
            instruction = (
                await self.get_context_pack_instruction(session_id)
                if session_id is not None else CONTEXT_PACK_INSTRUCTION
            )
        content = "\n\n".join(
            part for part in (instruction, self._allowed_context_pack_list) if part
        )
        return {
            **inputs,
            "messages": [
                SystemMessage(content=content, id="agentic-context-pack-list"),
                *inputs.get("messages", []),
            ],
        }

    async def _fetch_recall_context(
        self, *, session_id: str | None = None, owner: str | None = None
    ) -> dict[str, Any] | None:
        """Issue a single on-demand ``get_context`` POST and return the payload.

        Used by the recall tool's manual-mode fallback so the tool still has
        a payload to return when :class:`NoFetchPolicy` leaves the cache
        unseeded. Threads only the four identifier fields the server accepts
        (``conversation_id`` + ``owner_id`` + ``agent_id`` + ``owner_type``).

        Best-effort — returns ``None`` if there is no owned retrieval client,
        the session can't be resolved, or the read fails.
        """
        client = self._owned_retrieval_client
        sid = session_id or self._last_session_id
        if client is None or not sid:
            return None
        if owner is None:
            owner = self._owner_for_session(sid)
        if not owner:
            # No resolvable tenant — decline rather than a blank-scoped read.
            return None
        try:
            return await client.get_context(
                conversation_id=sid,
                owner_id=owner,
                agent_id=self.agent_id,
                owner_type=self._owner_type,
            )
        except Exception:
            logger.exception(
                "[LangGraph] _fetch_recall_context failed (best-effort); "
                "returning empty"
            )
            return None

    async def __aenter__(self) -> LangGraphAdapter:
        """Open owned clients. Only valid in self-driving mode.

        When the adapter was constructed with a ``ContextAuth`` (either passed
        directly or built from raw credentials), this is where the first
        token exchange happens: ``aget_token()`` yields ``(token, base_url)``
        which seeds two freshly-built clients. The clients are then
        ``auth.attach()``-ed so future refreshes propagate without each
        request having to re-read the token.
        """
        if not self._self_driving:
            raise RuntimeError(
                "`async with adapter:` is only valid in self-driving mode "
                "(no session_manager passed). Manage your manager and clients "
                "yourself when injecting a SessionManager."
            )
        if self._auth is not None and self._owned_ingest_client is None:
            token, base_url = await self._auth.aget_token()
            # ApiKeyAuth resolves agent_id from the secret bundle during the
            # token fetch above. Adopt it as our default when the caller didn't
            # pass one. Only overwrite a deferred (empty) id — an
            # explicit constructor agent_id always wins.
            if self._agent_id_deferred:
                resolved_agent_id = getattr(self._auth, "agent_id", "") or ""
                if not resolved_agent_id:
                    raise ValueError(
                        "agent_id is required but the secret server returned no "
                        "agentId for this api_key; pass agent_id= to "
                        "LangGraphAdapter(...)."
                    )
                self.agent_id = resolved_agent_id
                self._agent_participant_id = self.agent_id
                self._agent_id_deferred = False
            self._owned_ingest_client = AsyncIngestionClient(
                base_url=base_url,
                api_key=token,
                dataspace=self._resolved_dataspace,
                auth=self._auth,
            )
            self._owned_retrieval_client = AsyncRetrievalClient(
                base_url=base_url,
                api_key=token,
                dataspace=self._resolved_dataspace,
                auth=self._auth,
            )
            self._auth.attach(self._owned_ingest_client)
            self._auth.attach(self._owned_retrieval_client)
        assert self._owned_ingest_client is not None
        assert self._owned_retrieval_client is not None
        await self._owned_ingest_client.__aenter__()
        await self._owned_retrieval_client.__aenter__()
        self._aenter_called = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Drain pending work, close manager, then close owned clients.

        No-op when ``__aenter__`` never succeeded so a failed enter does not
        cascade into a double-close.
        """
        if not self._aenter_called:
            return
        try:
            await self._async_close()
        finally:
            assert self._owned_retrieval_client is not None
            assert self._owned_ingest_client is not None
            try:
                await self._owned_retrieval_client.__aexit__(exc_type, exc_val, exc_tb)
            finally:
                await self._owned_ingest_client.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def mode(self) -> Literal["stream", "snapshot"]:
        """``"stream"`` when constructed with ``graph=``; ``"snapshot"`` otherwise.

        Streaming-mode adapters can also serve ``snapshot()`` reads from the
        live graph; the mode label reflects construction, not capability.
        """
        return "stream" if self._graph is not None else "snapshot"

    # ─── self-driving context fetch (path 3 only) ────────────────────

    def _policy_state_for(
        self, session_id: str, owner: str | None = None
    ) -> dict[str, Any]:
        """Return the policy state dict for ``(owner, session_id)``.

        Self-driving mode looks up a process-shared LRU keyed by
        ``(owner_id, agent_id, session_id)`` so per-request adapters share
        ``last_fetched_at`` and ``turn_idx`` across their short lifetimes.
        On hit the entry is bumped to the most-recent end; on miss a fresh
        seed dict is inserted and, if the map exceeds
        ``_MAX_SHARED_POLICY_STATE``, the least-recently-used entry is
        evicted. Other modes use the per-instance dict (one map per
        adapter; nothing to evict).

        ``owner`` defaults to the constructor ``owner_id`` when omitted; with
        per-call owner_id ``stream()`` passes the resolved per-turn
        tenant so two owners reusing a session string keep separate state.
        """
        owner = owner if owner is not None else self._owner_id
        if self._self_driving and owner:
            key = (owner, self.agent_id, session_id)
            cls = type(self)
            with cls._SHARED_POLICY_LOCK:
                state = cls._SHARED_POLICY_STATE.get(key)
                if state is not None:
                    cls._SHARED_POLICY_STATE.move_to_end(key)
                    return state
                state = {
                    "last_payload": None,
                    "last_fetched_at": None,
                    "last_user_text": None,
                    "turn_idx": 0,
                }
                cls._SHARED_POLICY_STATE[key] = state
                # Evict the LRU entry(s) until under cap. ``> cap`` (not
                # ``>=``) so the cap is the maximum allowed size, not one
                # less. Loop because callers can lower ``_MAX_SHARED_POLICY_STATE``
                # at runtime; a single ``popitem`` wouldn't catch up.
                while len(cls._SHARED_POLICY_STATE) > cls._MAX_SHARED_POLICY_STATE:
                    cls._SHARED_POLICY_STATE.popitem(last=False)
                return state
        return self._context_state.setdefault(
            session_id,
            {
                "last_payload": None,
                "last_fetched_at": None,
                "last_user_text": None,
                "turn_idx": 0,
            },
        )

    def _policy_state_get(
        self, session_id: str, owner: str | None = None
    ) -> dict[str, Any] | None:
        """Read-only counterpart to :meth:`_policy_state_for`.

        Does not bump LRU position — used by post-turn bookkeeping that
        shouldn't affect eviction order. Concurrent ``_policy_state_for``
        callers may still mutate the returned dict's fields; that is fine
        because :meth:`_commit_policy_turn` only writes monotonic counters.
        """
        owner = owner if owner is not None else self._owner_id
        if self._self_driving and owner:
            key = (owner, self.agent_id, session_id)
            with type(self)._SHARED_POLICY_LOCK:
                return type(self)._SHARED_POLICY_STATE.get(key)
        return self._context_state.get(session_id)

    def _policy_state_write(
        self, session_id: str, updates: dict[str, Any], owner: str | None = None
    ) -> None:
        """Atomically re-resolve and update the policy-state entry for ``session_id``.

        Why this exists: callers snapshot state under the lock, drop the
        lock to ``await`` a slow GET, then need to write the result back.
        Holding only the original ``dict`` reference across the await is
        unsafe — between the read and the write, a flood of
        :meth:`_policy_state_for` calls from other sessions may evict that
        entry from the LRU. Mutating the orphaned dict would land the
        write in a dict no longer reachable from
        ``_SHARED_POLICY_STATE``, silently dropping it. This helper
        re-acquires the lock, looks up the current entry by key (or
        re-seeds + inserts with LRU bookkeeping if evicted), bumps LRU
        position, and applies ``updates`` to the dict that is actually
        stored in the map. Callers must compute any race-loss decisions
        before calling — this helper unconditionally applies ``updates``.
        """
        owner = owner if owner is not None else self._owner_id
        if self._self_driving and owner:
            key = (owner, self.agent_id, session_id)
            cls = type(self)
            with cls._SHARED_POLICY_LOCK:
                state = cls._SHARED_POLICY_STATE.get(key)
                if state is None:
                    # Was evicted during the await window. Re-seed with the
                    # base shape and apply LRU eviction so the cap holds.
                    state = {
                        "last_payload": None,
                        "last_fetched_at": None,
                        "last_user_text": None,
                        "turn_idx": 0,
                    }
                    cls._SHARED_POLICY_STATE[key] = state
                    while len(cls._SHARED_POLICY_STATE) > cls._MAX_SHARED_POLICY_STATE:
                        cls._SHARED_POLICY_STATE.popitem(last=False)
                else:
                    cls._SHARED_POLICY_STATE.move_to_end(key)
                state.update(updates)
            return
        # Manager-injection mode: no LRU, just per-instance dict. Still
        # re-resolve under no contention — keeps the call site uniform.
        state = self._context_state.get(session_id)
        if state is None:
            state = {
                "last_payload": None,
                "last_fetched_at": None,
                "last_user_text": None,
                "turn_idx": 0,
            }
            self._context_state[session_id] = state
        state.update(updates)

    def _policy_state_pop(self, session_id: str, owner: str | None = None) -> None:
        """Drop policy state for ``(owner, session_id)`` from whichever store owns it."""
        owner = owner if owner is not None else self._owner_id
        if self._self_driving and owner:
            key = (owner, self.agent_id, session_id)
            with type(self)._SHARED_POLICY_LOCK:
                type(self)._SHARED_POLICY_STATE.pop(key, None)
            return
        self._context_state.pop(session_id, None)

    async def _maybe_fetch_and_inject_context(
        self, inputs: Any, session_id: str, owner: str | None = None
    ) -> Any:
        """Consult the policy and inject a SystemMessage when it says to fetch.

        Returns the (possibly-rewritten) ``inputs`` dict. When the policy
        declines, when the GET returns empty, or when the GET raises, the
        original inputs are returned unchanged so the graph still runs.

        Thread-safety: the lock serializes read-decide-write with
        concurrent commits / fetches in other workers; it is held only
        across the snapshot and write-back, never across the GET. On
        write-back we re-resolve the entry by key under the lock
        (:meth:`_policy_state_write`) so an LRU eviction during the await
        can't strand the mutation in an orphaned dict. We also re-check
        whether another worker raced ahead and wrote ``last_fetched_at``
        already; if so we skip the write-back so a slower fetch can't
        trample a fresher payload. The "at most one GET per session
        start" invariant is therefore preserved: at most one
        ``should_fetch=True`` decision wins; others see the
        freshly-stamped state and decline.
        """
        if not isinstance(inputs, dict):
            return inputs
        if self._owned_retrieval_client is None:
            return inputs

        owner = owner if owner is not None else self._owner_id

        # Snapshot policy state under the lock: the seeding path in
        # _policy_state_for already holds the lock; this read mirrors that
        # so we never observe a torn entry.
        cls = type(self)
        if self._self_driving and owner:
            key = (owner, self.agent_id, session_id)
            with cls._SHARED_POLICY_LOCK:
                state = cls._SHARED_POLICY_STATE.get(key)
                if state is None:
                    state = {
                        "last_payload": None,
                        "last_fetched_at": None,
                        "last_user_text": None,
                        "turn_idx": 0,
                    }
                    cls._SHARED_POLICY_STATE[key] = state
                    while len(cls._SHARED_POLICY_STATE) > cls._MAX_SHARED_POLICY_STATE:
                        cls._SHARED_POLICY_STATE.popitem(last=False)
                else:
                    cls._SHARED_POLICY_STATE.move_to_end(key)
                snapshot_turn_idx = int(state["turn_idx"])
                snapshot_last_user_text = state["last_user_text"]
                snapshot_last_payload = state["last_payload"]
                snapshot_last_fetched_at = state["last_fetched_at"]
        else:
            state = self._policy_state_for(session_id, owner)
            snapshot_turn_idx = int(state["turn_idx"])
            snapshot_last_user_text = state["last_user_text"]
            snapshot_last_payload = state["last_payload"]
            snapshot_last_fetched_at = state["last_fetched_at"]

        should_fetch = False
        try:
            should_fetch = self._policy.should_fetch(
                session_id=session_id,
                turn_idx=snapshot_turn_idx,
                last_user_text=snapshot_last_user_text,
                last_payload=snapshot_last_payload,
                last_fetched_at=snapshot_last_fetched_at,
            )
        except Exception:
            logger.exception(
                "context_policy.should_fetch raised; treating as fetch=False"
            )

        if should_fetch:
            try:
                payload = await self._owned_retrieval_client.get_context(
                    conversation_id=session_id,
                    owner_id=owner,
                    agent_id=self.agent_id,
                    owner_type=self._owner_type,
                )
            except Exception:
                logger.exception(
                    "GET /agentic/context failed; running graph without "
                    "injected context for session %s",
                    session_id,
                )
                payload = None

            if payload:
                msg = context_to_system_message(
                    payload,
                    header=self._context_header,
                    message_id=self._context_message_id,
                    dedupe=self._context_dedupe,
                    rank_urgent=self._context_rank_urgent,
                )
                if msg is not None:
                    inputs = {
                        **inputs,
                        "messages": [msg, *inputs.get("messages", [])],
                    }
            # Commit fetch-state via the helper so the write lands in the
            # entry currently stored in the map — even if our original
            # entry was evicted while we were awaiting the GET. The
            # helper re-resolves under the lock and re-seeds on eviction.
            # Race-loss check: if another worker has already stamped
            # ``last_fetched_at`` (we lost the race while awaiting), skip
            # the write-back rather than clobbering a potentially-fresher
            # payload with our older one. Compute the decision against
            # the *current* entry under the lock so we don't read a stale
            # snapshot. Only stamp fetch-state when a fetch actually
            # happened, so a cancelled cached miss doesn't starve
            # subsequent turns of a real fetch.
            if self._self_driving and owner:
                key = (owner, self.agent_id, session_id)
                with cls._SHARED_POLICY_LOCK:
                    current = cls._SHARED_POLICY_STATE.get(key)
                    if current is None:
                        # Evicted during the await. Re-seed and apply our
                        # write — we are the only known fetcher for this
                        # key (the evicted entry's writer would have
                        # re-seeded too), so trample is acceptable here.
                        current = {
                            "last_payload": payload,
                            "last_fetched_at": datetime.now(timezone.utc),
                            "last_user_text": None,
                            "turn_idx": 0,
                        }
                        cls._SHARED_POLICY_STATE[key] = current
                        while len(cls._SHARED_POLICY_STATE) > cls._MAX_SHARED_POLICY_STATE:
                            cls._SHARED_POLICY_STATE.popitem(last=False)
                    else:
                        cls._SHARED_POLICY_STATE.move_to_end(key)
                        if (
                            current["last_fetched_at"] is None
                            or current["last_fetched_at"] == snapshot_last_fetched_at
                        ):
                            current["last_payload"] = payload
                            current["last_fetched_at"] = datetime.now(timezone.utc)
            else:
                # Manager-injection mode: per-instance dict, no LRU pressure,
                # but apply the same race-loss semantics for parity.
                current = self._context_state.get(session_id)
                if current is None:
                    current = {
                        "last_payload": payload,
                        "last_fetched_at": datetime.now(timezone.utc),
                        "last_user_text": None,
                        "turn_idx": 0,
                    }
                    self._context_state[session_id] = current
                elif (
                    current["last_fetched_at"] is None
                    or current["last_fetched_at"] == snapshot_last_fetched_at
                ):
                    current["last_payload"] = payload
                    current["last_fetched_at"] = datetime.now(timezone.utc)
        return inputs

    def _commit_policy_turn(
        self, session_id: str, user_text: str | None, owner: str | None = None
    ) -> None:
        """Advance per-session policy counters after a user record yields.

        Split out from ``_maybe_fetch_and_inject_context`` so cancellation
        between context fetch and the user record yield doesn't strand
        ``turn_idx`` / ``last_user_text`` in a half-advanced state.

        The whole read-compute-write must be atomic under the lock —
        going through ``_policy_state_get`` then re-acquiring the lock to
        write would leak a stale ``turn_idx`` if a concurrent
        ``_maybe_fetch_and_inject_context`` ran on the same key in
        between. Resolve the entry by key inside the same lock acquire
        as the write so concurrent stream() calls on the same session
        can't lose a turn or trample each other's bookkeeping.
        """
        owner = owner if owner is not None else self._owner_id
        if self._self_driving and owner:
            key = (owner, self.agent_id, session_id)
            cls = type(self)
            with cls._SHARED_POLICY_LOCK:
                state = cls._SHARED_POLICY_STATE.get(key)
                if state is None:
                    # Evicted between the user-record yield and this
                    # commit (or no fetch ever ran for this session).
                    # Re-seed so the increment lands in the live map.
                    state = {
                        "last_payload": None,
                        "last_fetched_at": None,
                        "last_user_text": user_text,
                        "turn_idx": 1,
                    }
                    cls._SHARED_POLICY_STATE[key] = state
                    while len(cls._SHARED_POLICY_STATE) > cls._MAX_SHARED_POLICY_STATE:
                        cls._SHARED_POLICY_STATE.popitem(last=False)
                else:
                    cls._SHARED_POLICY_STATE.move_to_end(key)
                    state["last_user_text"] = user_text
                    state["turn_idx"] = int(state["turn_idx"]) + 1
            return
        # Manager-injection mode: per-instance dict; no concurrent writers
        # from other workers, so a plain mutate is safe.
        state = self._context_state.get(session_id)
        if state is None:
            self._context_state[session_id] = {
                "last_payload": None,
                "last_fetched_at": None,
                "last_user_text": user_text,
                "turn_idx": 1,
            }
            return
        state["last_user_text"] = user_text
        state["turn_idx"] = int(state["turn_idx"]) + 1

    # ─── snapshot ────────────────────────────────────────────────────

    def snapshot(self, session_id: str) -> list[Record]:
        """Return all records for ``session_id`` in chronological order.

        - ``graph=`` mode: reads ``graph.get_state(cfg).values["messages"]``.
        - ``pickle_path=`` mode: deserializes the on-disk ``StateSnapshot``.

        Branches on message type:
          - ``HumanMessage``                              → ``user_message``
          - ``AIMessage`` with content + tool_calls       → ``agent_reasoning`` + per-call ``tool_call``
          - ``AIMessage`` with tool_calls only            → ``tool_call`` per entry
          - ``AIMessage`` with content only               → ``agent_response``
          - ``ToolMessage``                               → ``tool_result``
          - other ``BaseMessage`` (e.g. ``SystemMessage``) → ``agent_reasoning`` (Trace fallback)

        All snapshot records share one ``interaction_id``
        (``f"turn-{session_id}-snapshot"``) so Beacon groups them.
        """
        self._last_session_id = session_id

        if self._graph is not None:
            cfg = _translate_session_id(session_id)
            state = self._graph.get_state(cfg)
        else:
            assert self._pickle_path is not None  # narrowed by constructor invariant
            if not self._allow_pickle:
                raise ValueError(
                    "Loading a pickle requires allow_pickle=True; pickle is unsafe — see SDK security docs"
                )
            with self._pickle_path.open("rb") as f:
                state = pickle.load(f)

        values = getattr(state, "values", None) or {}
        messages = values.get("messages", []) or []

        interaction_id = f"turn-{session_id}-snapshot"
        records: list[Record] = []

        for idx, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                records.append(
                    self._record_from_human_message(
                        msg,
                        session_id=session_id,
                        idx=idx,
                        snapshot_state=state,
                        interaction_id=interaction_id,
                    )
                )
            elif isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                content = getattr(msg, "content", None)
                content_str = _translate_content(content)
                if tool_calls:
                    if content_str and self._classify_as_reasoning(msg):
                        records.append(
                            self._record_from_reasoning(
                                content_str,
                                session_id=session_id,
                                idx=idx,
                                interaction_id=interaction_id,
                                snapshot_state=state,
                            )
                        )
                    elif content_str and not self._classify_as_reasoning(msg):
                        records.append(
                            self._record_from_agent_response(
                                content_str,
                                session_id=session_id,
                                idx=idx,
                                interaction_id=interaction_id,
                                snapshot_state=state,
                                message=msg,
                            )
                        )
                    for tc_idx, tc in enumerate(tool_calls):
                        records.append(
                            self._record_from_tool_call(
                                tc,
                                session_id=session_id,
                                idx=idx,
                                tc_idx=tc_idx,
                                snapshot_state=state,
                                interaction_id=interaction_id,
                            )
                        )
                else:
                    if content_str:
                        records.append(
                            self._record_from_agent_response(
                                content_str,
                                session_id=session_id,
                                idx=idx,
                                interaction_id=interaction_id,
                                snapshot_state=state,
                                message=msg,
                            )
                        )
            elif isinstance(msg, ToolMessage):
                records.append(
                    self._record_from_tool_message(
                        msg,
                        session_id=session_id,
                        idx=idx,
                        snapshot_state=state,
                        interaction_id=interaction_id,
                    )
                )
            else:
                records.append(
                    self._record_from_message(
                        msg,
                        session_id=session_id,
                        idx=idx,
                        snapshot_state=state,
                        interaction_id=interaction_id,
                    )
                )

        return records

    # ─── run / stream ────────────────────────────────────────────────

    async def run(
        self, inputs: Any, *, session_id: str, owner_id: str | None = None
    ) -> list[Record]:
        """Drive one turn end-to-end and return the emitted records.

        Convenience wrapper over :meth:`stream` for callers that don't need
        per-record observability — drains the async generator internally and
        collects every ``Record`` it yielded. Records are still ingested per
        turn through the same path as :meth:`stream`; the only difference is
        that the caller waits for the whole turn instead of iterating.

        Use :meth:`stream` directly when you need to observe records as they
        emit (e.g. server-sent events, audit tap, early-termination logic).

        ``owner_id`` overrides the constructor owner for this turn —
        see :meth:`stream` for the multi-tenant resolution rules.
        """
        records: list[Record] = []
        async for record in self.stream(inputs, session_id, owner_id=owner_id):
            records.append(record)
        return records

    async def stream(
        self, inputs: Any, session_id: str, *, owner_id: str | None = None
    ) -> AsyncIterator[Record]:
        """Drive the graph and yield a ``Record`` per turn-relevant event.

        Steps:
          1. Per-turn ``interaction_id``.
          2. Extract ``user_message`` from ``inputs``.
          3. Drive ``astream_events(version="v3")`` and iterate its
             ``stream.messages`` projection (one item per aggregated agent
             utterance). For each, await ``.output`` and classify via
             ``_has_pending_tool_calls`` into ``agent_reasoning`` (tool calls
             pending) or ``agent_response`` (final reply).
          4. Tool events flow through ``_ToolCaptureHandler`` registered on
             ``cfg["callbacks"]``; its ``tool_call`` / ``tool_result``
             records are yielded in chronological order with the rest, when
             the next model call starts streaming or the turn ends.
          5. Ingest every record of the turn in one batch — one ``/entries``
             POST per turn — when the turn ends, raises, or is closed early.
             A turn abandoned without ``aclose()`` is flushed by
             ``close()`` / ``end_session()``.
        """
        if self._graph is None:
            raise ValueError(
                "LangGraphAdapter.stream() requires graph=; got pickle_path= construction"
            )
        if self._self_driving and not self._aenter_called:
            raise RuntimeError(
                "Self-driving LangGraphAdapter must be used as `async with "
                "adapter:` so its owned clients are opened before stream()."
            )

        # Convenience: a bare string becomes one human turn.
        if isinstance(inputs, str):
            inputs = {"messages": [("human", inputs)]}

        # Resolve the owner for this turn (per-call arg > constructor
        # default > ValueError) and record it for the session so the owner-less
        # read paths (recall tool, context seed) and _mgr_for can recover
        # it from just a session_id. Only self-driving mode owns the manager and
        # posts on the adapter's behalf; the manager-injected paths carry their
        # own owner on the injected manager, so we don't require one here.
        owner: str | None = None
        if self._self_driving:
            owner = self._resolve_owner(owner_id)
            self._session_owner[session_id] = owner
        elif owner_id is not None:
            raise ValueError(
                "owner_id= is only valid in self-driving mode; the "
                "manager-injected paths carry the owner on the injected manager."
            )

        # 1. Per-turn interaction_id — keyed by session_id so
        #    concurrent streams on different sessions don't clobber each
        #    other's counters / interaction ids.
        turn_idx = self._turn_idx.get(session_id, 0)
        interaction_id = f"turn-{session_id}-{turn_idx}"
        self._current_interaction_id[session_id] = interaction_id
        self._turn_idx[session_id] = turn_idx + 1
        self._last_session_id = session_id

        # Self-driving mode: consult the policy and inject context BEFORE the
        # graph runs. Best-effort — a Beacon outage degrades to "no context"
        # rather than aborting the user-facing turn.
        if self._self_driving:
            inputs = await self._maybe_fetch_and_inject_context(
                inputs, session_id, owner
            )

        inputs = await self._maybe_inject_context_pack_list(
            inputs, is_open=(turn_idx == 0), session_id=session_id
        )

        # Recall-tool mode: prepend the tool-guidance SystemMessage so it
        # sits ahead of the context block. A stable message id lets the
        # add_messages reducer replace (not duplicate) it across turns when a
        # checkpointer is in play. Skipped under manager-injection (tool not
        # installed) or when the resolved instruction is "" (manual mode —
        # customer owns their own prompting).
        inputs = self._maybe_inject_recall_tool_instruction(inputs)

        _safe_call(self._on_session_start, session_id)
        _safe_call(
            self._on_turn_start,
            TurnEvent(
                session_id=session_id,
                inputs=inputs,
                turn_started_at=datetime.now(timezone.utc),
            ),
        )

        # 2. Translate cfg and register the tool-capture handler. It is always
        #    registered: tool records are captured whether or not a tool hook
        #    is set. It captures the per-turn interaction_id so its records
        #    route to the correct turn even when concurrent stream() calls
        #    advance _current_interaction_id.
        # Thread the turn-resolved owner into the run config so the recall
        # tool recovers this run's tenant from its own invocation config
        # (race-free) rather than the process-shared _session_owner map.
        cfg = _translate_session_id(session_id, owner)
        handler = _ToolCaptureHandler(self, session_id, interaction_id)
        cfg["callbacks"] = [handler]

        # Every record of the turn is collected here and ingested in one
        # batch by the ``finally`` below — or by close()/end_session() if
        # the caller abandons the generator before it finalizes.
        turn = _TurnBatch(session_id, owner, handler)
        turn_records = turn.records
        self._active_turns.add(turn)
        completed = False
        try:
            # 3. Extract user_message from inputs.
            idx = 0
            user_record = self._record_from_user_input(
                inputs,
                session_id=session_id,
                idx=idx,
                interaction_id=interaction_id,
                owner=owner,
            )
            if user_record is not None:
                turn_records.append(user_record)
                yield user_record
                idx += 1
                if self._self_driving:
                    self._commit_policy_turn(session_id, user_record.content, owner)

            # 4. Drive astream_events. v3 returns an AsyncGraphRunStream whose
            #    .messages projection yields one AsyncChatModelStream per LLM
            #    round-trip; we await its .output to classify reasoning vs
            #    reply. Tool records captured since the previous message are
            #    yielded first so the turn stays in chronological order.
            try:
                stream = await self._graph.astream_events(inputs, cfg, version="v3")
                async for message in stream.messages:
                    # Drain before awaiting the message so tool records are
                    # yielded when the next model call starts, not after it
                    # finishes.
                    for tool_record in handler.drain():
                        turn_records.append(tool_record)
                        yield tool_record
                    text = await message.text
                    output_msg = await message.output
                    if not text:
                        continue
                    if _has_pending_tool_calls(output_msg):
                        record = self._record_from_reasoning(
                            text,
                            session_id=session_id,
                            idx=idx,
                            interaction_id=interaction_id,
                        )
                    else:
                        record = self._record_from_v3_message_obj(
                            message,
                            output_msg,
                            text,
                            session_id=session_id,
                            idx=idx,
                            interaction_id=interaction_id,
                        )
                    turn_records.append(record)
                    yield record
                    idx += 1
                for tool_record in handler.drain():
                    turn_records.append(tool_record)
                    yield tool_record
            except Exception as exc:
                _safe_call(
                    self._on_error,
                    ErrorEvent(
                        session_id=session_id,
                        phase="stream",
                        exception=exc,
                        occurred_at=datetime.now(timezone.utc),
                    ),
                )
                raise
            completed = True
        finally:
            # 5. One ingest per turn. Runs on success, on a graph error, and
            #    when the caller stops iterating early, so a partial turn
            #    still lands. If close()/end_session() already flushed this
            #    turn, its manager is closed; report what came after instead.
            if turn.flushed:
                self._report_records_after_flush(turn)
            elif completed:
                await self._flush_turn(turn)
            else:
                # Don't let an ingest failure mask the error or early exit
                # that cut the turn short; report it through on_error.
                await self._flush_turn_reporting(turn)

    async def _flush_turn(self, turn: _TurnBatch) -> None:
        """Ingest ``turn`` once: the first caller wins, later calls no-op.

        Tools still running get an error ``tool_result`` so no ``tool_call``
        is left without one (a cancelled tool never reaches on_tool_error).
        """
        if turn.flushed:
            return
        turn.flushed = True
        self._active_turns.discard(turn)
        turn.records.extend(turn.handler.drain())
        turn.records.extend(turn.handler.finish_pending())
        turn.sent = len(turn.records)
        await self._ingest_async(turn.records, turn.owner)

    def _report_records_after_flush(self, turn: _TurnBatch) -> None:
        """Warn about records a turn produced after close()/end_session() sent it."""
        dropped = len(turn.records) + len(turn.handler.drain()) - turn.sent
        if dropped <= 0:
            return
        exc = RuntimeError(
            f"{dropped} record(s) produced after close()/end_session() "
            f"flushed the turn were not ingested"
        )
        logger.warning("session %s: %s", turn.session_id, exc)
        _safe_call(
            self._on_error,
            ErrorEvent(
                session_id=turn.session_id,
                phase="ingest",
                exception=exc,
                occurred_at=datetime.now(timezone.utc),
            ),
        )

    async def _flush_turn_reporting(self, turn: _TurnBatch) -> None:
        """:meth:`_flush_turn` that logs and fires ``on_error`` instead of raising."""
        try:
            await self._flush_turn(turn)
        except Exception as exc:
            logger.exception(
                "ingest of partial turn failed for session %s", turn.session_id
            )
            _safe_call(
                self._on_error,
                ErrorEvent(
                    session_id=turn.session_id,
                    phase="ingest",
                    exception=exc,
                    occurred_at=datetime.now(timezone.utc),
                ),
            )

    async def _flush_active_turns(self, session_id: str | None = None) -> None:
        """Flush unfinished turns (all, or one session's) before managers close."""
        for turn in list(self._active_turns):
            if session_id is None or turn.session_id == session_id:
                await self._flush_turn_reporting(turn)

    # ─── ingest / close — bound to manager flavor at __init__ ────────

    def _mgr_for(
        self, session_id: str, owner: str | None = None
    ) -> AsyncSessionManager:
        """Self-driving mode only: lazily build one manager per ``(session, owner)``.

        The same ``session_id`` can be driven for different owners
        across the adapter's lifetime, so the manager pool is keyed by the
        ``(session_id, owner)`` pair.

        ``owner`` MUST be threaded in from the turn that resolved it — the
        stream loop passes its captured owner here. Re-deriving it from
        :attr:`_session_owner` would be a cross-tenant hazard: that map holds
        one slot per ``session_id``, so a
        concurrent turn for the *same* ``session_id`` under a *different* owner
        can overwrite the slot during the ``await`` between resolving the
        owner and the ingest, routing this turn's records to the wrong tenant.
        The ``owner is None`` fallback is only for the public ``ingest()``
        entry point, which has no turn context to carry.
        """
        assert self._self_driving
        assert self._owned_ingest_client is not None
        if owner is None:
            owner = self._owner_for_session(session_id)
        key = (session_id, owner)
        existing = self._owned_managers.get(key)
        if existing is not None:
            return existing
        mgr = AsyncSessionManager(
            self._owned_ingest_client,
            agent_id=self.agent_id,
            owner_id=owner,
            session_id=session_id,
            owner_type=self._owner_type,
            retrieval_client=self._owned_retrieval_client,
            callbacks=self._ingestion_callbacks,
            agent_participant_variable=self._resolve_agent_participant_variable(),
        )
        self._owned_managers[key] = mgr
        return mgr

    def _resolve_agent_participant_variable(self) -> dict[str, Any]:
        """Beacon ``AiAgentParticipant`` block for a new manager: the caller's
        full-override if given, else this runtime's label + ``agent_id``.
        Resolved per manager, not at construction, so a late-bound
        ``agent_id`` is picked up."""
        if self._agent_participant_variable is not None:
            return dict(self._agent_participant_variable)
        return {**_AGENT_PARTICIPANT_VARIABLE, "AiAgentApiName": self.agent_id}

    def _sync_ingest(self, records: list[Record]) -> None:
        assert self._mgr is not None
        self._mgr.ingest(records)

    async def _async_ingest(
        self, records: list[Record], owner: str | None = None
    ) -> None:
        if not records:
            return
        if self._self_driving:
            mgr = self._mgr_for(records[0].session_id, owner)
            await mgr.ingest(records)
            return
        assert isinstance(self._mgr, AsyncSessionManager)
        await self._mgr.ingest(records)

    async def _ingest_async(
        self, records: list[Record], owner: str | None = None
    ) -> None:
        """Internal async ingest used by stream(). Bridges across manager flavors.

        ``owner`` is the turn-resolved tenant; it is threaded through
        to :meth:`_mgr_for` so concurrent same-session/different-owner turns
        never route through each other's manager. ``None`` falls back to the
        per-session lookup (used by the public ``ingest()`` surface, which has
        no turn context).
        """
        if not records:
            return
        if self._self_driving:
            mgr = self._mgr_for(records[0].session_id, owner)
            await mgr.ingest(records)
            return
        if isinstance(self._mgr, AsyncSessionManager):
            await self._mgr.ingest(records)
        else:
            assert self._mgr is not None
            self._mgr.ingest(records)

    def _sync_close(self) -> None:
        try:
            assert self._mgr is not None
            self._mgr.close()
        finally:
            # Always fire on_session_end — observability must not skip on error.
            _safe_call(self._on_session_end, self._last_session_id)

    async def _async_close(self) -> None:
        self._context_pack_prompt.clear()
        try:
            await self._flush_active_turns()
            if self._self_driving:
                # Close every per-session manager the adapter created.
                managers = list(self._owned_managers.values())
                self._owned_managers.clear()
                for mgr in managers:
                    try:
                        await mgr.close()
                    except Exception:
                        logger.exception(
                            "owned manager close failed; continuing"
                        )
            else:
                assert isinstance(self._mgr, AsyncSessionManager)
                await self._mgr.close()
        finally:
            # Always fire on_session_end — observability must not skip on error.
            _safe_call(self._on_session_end, self._last_session_id)

    async def end_session(self, session_id: str) -> None:
        """Close and forget the per-session manager for ``session_id``.

        Long-running adapters (web servers, schedulers) that drive many
        sessions through one adapter instance should call this when a
        session ends so the manager pool doesn't grow unbounded. Drops the
        per-session ``AsyncSessionManager``, the cached fetch state, and
        the turn counters so a later ``stream(..., session_id)`` starts
        fresh. No-op for sessions the adapter never saw.

        Self-driving mode only — the legacy paths don't own the manager.
        """
        if not self._self_driving:
            raise RuntimeError(
                "end_session() is only valid in self-driving mode. The "
                "manager-injected paths don't own the manager."
            )
        # Land any unfinished turn for this session through its manager
        # before the manager is dropped.
        await self._flush_active_turns(session_id)
        # The manager pool is keyed by (session_id, owner); drop every
        # manager for this session_id regardless of which owner(s) drove it.
        keys = [k for k in self._owned_managers if k[0] == session_id]
        managers = [self._owned_managers.pop(k) for k in keys]
        # Drop policy state for the owner(s) that actually drove this session,
        # plus the constructor default for the manager-injection-parity path.
        owners = {k[1] for k in keys}
        owners.add(self._session_owner.get(session_id, self._owner_id))
        for owner in owners:
            self._policy_state_pop(session_id, owner)
        self._session_owner.pop(session_id, None)
        self._turn_idx.pop(session_id, None)
        self._current_interaction_id.pop(session_id, None)
        self._context_pack_prompt.forget(session_id)
        for mgr in managers:
            try:
                await mgr.close()
            except Exception:
                logger.exception(
                    "owned manager close failed for session %s; continuing",
                    session_id,
                )
        _safe_call(self._on_session_end, session_id)

    # ─── record builders (private methods on the class) ─────────────

    def _record_from_user_input(
        self,
        inputs: Any,
        *,
        session_id: str,
        idx: int,
        interaction_id: str,
        owner: str | None = None,
    ) -> Record | None:
        """Extract a ``user_message`` from ``inputs["messages"][-1]``.

        Recognizes ``("human", text)`` tuples and ``HumanMessage`` instances.
        Returns ``None`` if no recognizable user input — e.g., the graph was
        kicked off internally by a scheduler.
        """
        if not isinstance(inputs, dict):
            return None
        messages = inputs.get("messages")
        if not messages:
            return None
        last = messages[-1]
        text: str | None = None
        framework_id: Any = None
        if isinstance(last, HumanMessage):
            text = _translate_content(last.content)
            framework_id = getattr(last, "id", None)
        elif isinstance(last, tuple) and len(last) == 2 and last[0] == "human":
            text = _translate_content(last[1])
        elif isinstance(last, dict):
            # LangChain accepts {"role": "user"|"human", "content": ...}
            # dicts everywhere else, so recognize that shape here too.
            role = last.get("role") or last.get("type")
            if role in ("user", "human"):
                text = _translate_content(last.get("content"))
                framework_id = last.get("id")
        if text is None:
            return None
        # Honor msg.id when present so that re-snapshotting the same
        # HumanMessage (after streaming it once) produces the same local
        # event_id (logs, callbacks). event_id is not sent, so ingesting both
        # stream and snapshot of a session still writes duplicates.
        return Record(
            event_id=_translate_event_id(framework_id, session_id, idx),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="user_message",
            participant_id=self._user_participant_for(session_id, owner),
            content=text,
            category="Message",
            interaction_id=interaction_id,
            timestamp=datetime.now(timezone.utc),
        )

    def _record_from_human_message(
        self,
        msg: Any,
        *,
        session_id: str,
        idx: int,
        snapshot_state: Any,
        interaction_id: str,
    ) -> Record:
        return Record(
            event_id=_translate_event_id(getattr(msg, "id", None), session_id, idx),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="user_message",
            participant_id=self._user_participant_for(session_id),
            content=_translate_content(getattr(msg, "content", "")),
            category="Message",
            interaction_id=interaction_id,
            timestamp=_resolve_created_at(msg, snapshot_state),
        )

    def _record_from_message(
        self,
        msg: Any,
        *,
        session_id: str,
        idx: int,
        snapshot_state: Any,
        interaction_id: str,
    ) -> Record:
        """Fallback for unknown ``BaseMessage`` types (e.g., ``SystemMessage``).

        Treated as ``agent_reasoning`` Trace records so they reach Beacon as
        non-Message context rather than masquerading as agent text.
        """
        return Record(
            event_id=_translate_event_id(getattr(msg, "id", None), session_id, idx),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="agent_reasoning",
            participant_id=self._agent_participant_id,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={"OutputValueText": _translate_content(getattr(msg, "content", ""))},
            timestamp=_resolve_created_at(msg, snapshot_state),
        )

    def _record_from_tool_call(
        self,
        tc: dict[str, Any],
        *,
        session_id: str,
        idx: int,
        tc_idx: int = 0,
        snapshot_state: Any,
        interaction_id: str,
    ) -> Record:
        tc_id = tc.get("id")
        tool_name = tc.get("name", "unknown")
        args = tc.get("args", {})
        # Always generate a deterministic event_id from (session, idx, tc_idx, name).
        # LangGraph populates tc["id"] lazily — re-snapshotting the same
        # AIMessage can flip from "no id" to "id present", so using tc["id"]
        # as the event_id would diverge across reads. Keep the framework's
        # tc["id"] in metadata for cross-record correlation only.
        event_id = f"step-tool-call-{session_id}-{idx}-{tc_idx}-{tool_name}"
        return Record(
            event_id=event_id,
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="tool_call",
            participant_id=self._agent_participant_id,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={
                "Name": tool_name,
                "InputValueText": json.dumps(_coerce_metadata_value(args)),
            },
            metadata={"tool_call_id": tc_id} if tc_id else {},
            timestamp=_resolve_created_at(None, snapshot_state),
        )

    def _record_from_tool_message(
        self,
        msg: Any,
        *,
        session_id: str,
        idx: int,
        snapshot_state: Any,
        interaction_id: str,
    ) -> Record:
        tool_name = getattr(msg, "name", None) or "unknown"
        tool_call_id = getattr(msg, "tool_call_id", None)
        return Record(
            event_id=str(getattr(msg, "id", None) or f"step-tool-result-{session_id}-{idx}"),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="tool_result",
            participant_id=self._agent_participant_id,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={
                "Name": tool_name,
                "OutputValueText": _tool_output_text(getattr(msg, "content", "")),
            },
            metadata={"tool_call_id": tool_call_id} if tool_call_id else {},
            timestamp=_resolve_created_at(msg, snapshot_state),
        )

    def _record_from_reasoning(
        self,
        text: str,
        *,
        session_id: str,
        idx: int,
        interaction_id: str,
        timestamp: datetime | None = None,
        snapshot_state: Any | None = None,
    ) -> Record:
        ts = timestamp
        if ts is None:
            ts = (
                _resolve_created_at(None, snapshot_state)
                if snapshot_state is not None
                else datetime.now(timezone.utc)
            )
        return Record(
            event_id=f"step-reasoning-{session_id}-{idx}",
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="agent_reasoning",
            participant_id=self._agent_participant_id,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={"OutputValueText": _translate_content(text)},
            timestamp=ts,
        )

    def _record_from_agent_response(
        self,
        text: str,
        *,
        session_id: str,
        idx: int,
        interaction_id: str,
        timestamp: datetime | None = None,
        snapshot_state: Any | None = None,
        message: Any | None = None,
        output_msg: Any | None = None,
    ) -> Record:
        framework_id = (
            getattr(message, "id", None)
            or getattr(output_msg, "id", None)
            or None
        )
        ts = timestamp
        if ts is None:
            ts = (
                _resolve_created_at(message, snapshot_state)
                if snapshot_state is not None
                else datetime.now(timezone.utc)
            )
        return Record(
            event_id=str(
                framework_id or f"step-agent-response-{session_id}-{idx}"
            ),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="agent_response",
            participant_id=self._agent_participant_id,
            content=str(text),
            category="Message",
            interaction_id=interaction_id,
            timestamp=ts,
        )

    def _record_from_v3_message_obj(
        self,
        message: Any,
        output_msg: Any,
        text: str,
        *,
        session_id: str,
        idx: int,
        interaction_id: str,
    ) -> Record:
        """Build an ``agent_response`` record from a v3 ``stream.messages`` item.

        ``message`` is the ``AsyncChatModelStream`` (carries ``message_id`` /
        ``node``); ``output_msg`` is the awaited ``AIMessage`` projection
        (carries ``id`` / ``usage_metadata``); ``text`` is the awaited text.
        """
        framework_id = (
            getattr(output_msg, "id", None) or getattr(message, "message_id", None)
        )
        metadata: dict[str, Any] = {}
        node = getattr(message, "node", None)
        if node is not None:
            metadata["node"] = node
        metadata["event"] = "utterance_complete"
        usage = getattr(output_msg, "usage_metadata", None)
        if usage is not None:
            metadata["usage"] = _coerce_metadata_value(usage)

        return Record(
            event_id=str(framework_id or f"step-agent-response-{session_id}-{idx}"),
            session_id=session_id,
            agent_id=self.agent_id,
            event_type="agent_response",
            participant_id=self._agent_participant_id,
            content=_translate_content(text),
            category="Message",
            interaction_id=interaction_id,
            metadata=metadata,
            timestamp=_resolve_created_at(output_msg, None),
        )

    def _classify_as_reasoning(self, msg: Any) -> bool:
        """Decide whether AIMessage content is reasoning or just incidental text.

        Default heuristic: presence of tool_calls + non-empty content → reasoning.
        Override via the ``reasoning_classifier`` constructor kwarg.
        """
        if self._reasoning_classifier is not None:
            return self._reasoning_classifier(msg) == "reasoning"
        return True


# ─── module-level translation helpers (private, pure) ──────────────


def _safe_call(hook: Callable[..., Any] | None, *args: Any) -> None:
    """Invoke ``hook(*args)`` if non-None; log and swallow any exception."""
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:
        logger.exception("agentic_context hook raised; continuing")


def _translate_session_id(
    session_id: str, owner: str | None = None
) -> dict[str, Any]:
    """SDK ``session_id`` (str) → LangGraph cfg shape.

    LangGraph identifies a checkpointer thread by ``thread_id`` inside
    ``config["configurable"]``. Under the unified model the SDK's
    ``session_id`` IS that ``thread_id`` — same identifier, different
    vocabulary across the two layers.

    When ``owner`` is supplied it is stamped under the private
    ``agentic_owner_id`` key. LangGraph propagates ``configurable`` into every
    child runnable — including tool calls — so the recall tool recovers *this
    run's* tenant from its own invocation config, race-free, instead of the
    process-shared ``_session_owner`` map (whose single slot per ``session_id``
    a concurrent same-session/different-owner turn could overwrite mid-``await``
    — a cross-tenant read hazard). This mirrors the defence already applied to
    the write path via ``_mgr_for``.
    """
    configurable: dict[str, Any] = {"thread_id": session_id}
    if owner is not None:
        configurable["agentic_owner_id"] = owner
    return {"configurable": configurable}


def _translate_event_id(framework_id: Any, session_id: str, idx: int) -> str:
    """Stable ``event_id``, with deterministic fallback when framework didn't supply one."""
    if framework_id is not None:
        return str(framework_id)
    return f"msg-{session_id}-{idx}"


def _translate_role(framework_role: str | None) -> str | None:
    """LangGraph BaseMessage.type → standardized participant_id (identity for LangGraph)."""
    return framework_role


def _translate_content(framework_content: Any) -> str:
    """Framework-native content → flat string. Handles None and list-of-blocks."""
    if framework_content is None:
        return ""
    if isinstance(framework_content, str):
        return framework_content
    if isinstance(framework_content, list):
        # v3 sometimes carries a list of content blocks; pull text out, skip non-text.
        parts: list[str] = []
        for block in framework_content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and "text" in block:
                    parts.append(str(block["text"]))
                # tool_use / tool_call blocks intentionally skipped — they don't carry agent text
        return "".join(parts)
    return str(framework_content)


def _resolve_created_at(msg: Any, snapshot_state: Any) -> datetime:
    """Best-effort timestamp resolution. Always returns tz-aware UTC."""
    msg_ts = getattr(msg, "timestamp", None) if msg is not None else None
    if isinstance(msg_ts, datetime):
        return msg_ts if msg_ts.tzinfo else msg_ts.replace(tzinfo=timezone.utc)

    snap_ts = getattr(snapshot_state, "created_at", None) if snapshot_state is not None else None
    if isinstance(snap_ts, datetime):
        return snap_ts if snap_ts.tzinfo else snap_ts.replace(tzinfo=timezone.utc)
    if isinstance(snap_ts, str):
        try:
            parsed = datetime.fromisoformat(snap_ts.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime.now(timezone.utc)


def _coerce_metadata_value(value: Any) -> Any:
    """JSON-safe coercion. None/bool/int/float/str pass; dict/list recurse; everything else → str."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_metadata_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_metadata_value(v) for v in value]
    return str(value)


def _extract_text(output: Any) -> str:
    """Pull text from a v3 ``on_chat_model_end`` event payload's output object."""
    if output is None:
        return ""
    content = getattr(output, "content", None)
    if content is None and isinstance(output, dict):
        content = output.get("content")
    return _translate_content(content)


def _extract_user_text(messages: list[Any]) -> str | None:
    """Pull the latest human-message text from a LangGraph ``messages`` list.

    Mirrors the recognition logic of ``_record_from_user_input`` so the
    smart-fetch policy sees the same text that the adapter ingests as a
    ``user_message``.
    """
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, HumanMessage):
        return _translate_content(last.content)
    if isinstance(last, tuple) and len(last) == 2 and last[0] == "human":
        return _translate_content(last[1])
    if isinstance(last, dict):
        role = last.get("role") or last.get("type")
        if role in ("user", "human"):
            return _translate_content(last.get("content"))
    return None


def _has_pending_tool_calls(output: Any) -> bool:
    """Check whether a v3 LLM output payload carries pending tool calls."""
    if output is None:
        return False
    tool_calls = getattr(output, "tool_calls", None)
    if tool_calls is None and isinstance(output, dict):
        tool_calls = output.get("tool_calls")
    return bool(tool_calls)


# ─── _ToolCaptureHandler — bridges LangChain callbacks to the manager ─


# Cap on a tool_result's OutputValueText, matching the ADK adapter.
_TOOL_OUTPUT_MAX_CHARS = 2000


def _tool_output_text(output: Any) -> str:
    """Tool output → bounded text. Never raises (it runs inside a callback).

    Unwraps a ``ToolMessage`` to its content and a ``Command`` to the
    ``ToolMessage`` content in its ``update["messages"]``. Dicts and lists
    are JSON-encoded; anything else is ``str()``-ed.
    """
    try:
        if isinstance(output, Command):
            update = output.update
            messages = update.get("messages") if isinstance(update, dict) else None
            tool_texts = [
                _translate_content(m.content)
                for m in messages or []
                if isinstance(m, ToolMessage)
            ]
            output = "\n".join(tool_texts) if tool_texts else update
        content = getattr(output, "content", output)
        if isinstance(content, str):
            text = content
        elif isinstance(content, (dict, list, tuple)):
            try:
                text = json.dumps(content, default=str, ensure_ascii=False)
            except (TypeError, ValueError):  # non-str keys, cycles
                text = str(content)
        else:
            text = str(content)
    except Exception:
        text = f"<unrenderable {type(output).__name__}>"
    return text[:_TOOL_OUTPUT_MAX_CHARS]


class _TurnBatch:
    """One stream() turn's records, ingested once by whoever flushes first."""

    __slots__ = ("session_id", "owner", "handler", "records", "flushed", "sent")

    def __init__(
        self, session_id: str, owner: str | None, handler: _ToolCaptureHandler
    ) -> None:
        self.session_id = session_id
        self.owner = owner
        self.handler = handler
        self.records: list[Record] = []
        self.flushed = False
        # How many records the flush sent; anything appended later is reported.
        self.sent = 0


class _ToolCaptureHandler(BaseCallbackHandler):
    """Captures tool lifecycle events from LangChain's callback system.

    Builds ``tool_call`` and ``tool_result`` Records and buffers them for
    the owning stream() call, which yields them in order and ingests them
    with the rest of the turn in one batch. Also fans out user observation
    events (``on_tool_start`` / ``on_tool_end`` / ``on_error``).

    Each handler is bound to one stream() call and captures that turn's
    ``interaction_id`` directly — reading ``adapter._current_interaction_id``
    here would race when two stream() calls on the same session or different
    sessions overlap.
    """

    def __init__(
        self,
        adapter: LangGraphAdapter,
        session_id: str,
        interaction_id: str,
    ) -> None:
        self._adapter = adapter
        self._session_id = session_id
        self._interaction_id = interaction_id
        self._starts: dict[UUID, tuple[str, dict[str, Any] | None, datetime]] = {}
        # LangGraph runs sync nodes (and their tool callbacks) on worker
        # threads, so the buffer is appended off-loop and drained on-loop.
        self._lock = threading.Lock()
        self._records: list[Record] = []

    # Only tool events are needed. LangChain dispatches every event a sync
    # handler doesn't ignore through a thread pool, so opt out of the rest.
    # ``ignore_agent`` stays False: it gates the on_tool_* events.
    @property
    def ignore_llm(self) -> bool:
        return True

    @property
    def ignore_chat_model(self) -> bool:
        return True

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    @property
    def ignore_retry(self) -> bool:
        return True

    @property
    def ignore_custom_event(self) -> bool:
        return True

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = (serialized or {}).get("name") or "unknown"
        args: dict[str, Any] = inputs if inputs is not None else {"input": input_str}
        started_at = datetime.now(timezone.utc)
        self._starts[run_id] = (tool_name, args, started_at)

        record = Record(
            event_id=f"step-tool-call-{run_id}",
            session_id=self._session_id,
            agent_id=self._adapter.agent_id,
            event_type="tool_call",
            participant_id=self._adapter._agent_participant_id,
            content=None,
            category="Trace",
            interaction_id=self._interaction_id,
            variable={
                "Name": tool_name,
                "InputValueText": json.dumps(_coerce_metadata_value(args)),
            },
            metadata={"tool_call_id": str(run_id)},
            timestamp=started_at,
        )
        self._buffer(record)

        _safe_call(
            self._adapter._on_tool_start,
            ToolEvent(
                session_id=self._session_id,
                tool_name=tool_name,
                tool_call_id=str(run_id),
                args=args,
                output=None,
                started_at=started_at,
                ended_at=None,
            ),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._starts.pop(run_id, None)
        if start is not None:
            tool_name, args, started_at = start
        else:
            tool_name, args, started_at = "unknown", None, datetime.now(timezone.utc)
        ended_at = datetime.now(timezone.utc)

        record = self._result_record(
            run_id, tool_name, _tool_output_text(output), ended_at
        )
        self._buffer(record)

        _safe_call(
            self._adapter._on_tool_end,
            ToolEvent(
                session_id=self._session_id,
                tool_name=tool_name,
                tool_call_id=str(run_id),
                args=args,
                output=output,
                started_at=started_at,
                ended_at=ended_at,
            ),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(error, asyncio.CancelledError):
            # Cancellation is not a tool failure. Newer langchain-core reports
            # it here (older versions skipped on_tool_error); either way leave
            # the start pending so finish_pending() closes it as "did not finish".
            return
        start = self._starts.pop(run_id, None)
        if isinstance(error, GraphBubbleUp):
            # interrupt() / a handoff Command: control flow, not a failure.
            # The tool_call stays open; on resume the tool runs again.
            return
        tool_name = start[0] if start is not None else "unknown"
        # Close the tool_call with an error result, as the other adapters do.
        # Only the exception type is recorded: messages often carry URLs,
        # tokens or DSNs, and Beacon is outside the process. The full
        # exception still goes to on_error.
        self._buffer(
            self._result_record(
                run_id,
                tool_name,
                f"Tool raised {type(error).__name__}",
                datetime.now(timezone.utc),
                error=True,
            )
        )
        _safe_call(
            self._adapter._on_error,
            ErrorEvent(
                session_id=self._session_id,
                phase="tool",
                exception=error,
                occurred_at=datetime.now(timezone.utc),
            ),
        )

    def _result_record(
        self,
        run_id: UUID,
        tool_name: str,
        output_text: str,
        timestamp: datetime,
        *,
        error: bool = False,
    ) -> Record:
        metadata: dict[str, Any] = {"tool_call_id": str(run_id)}
        if error:
            metadata["error"] = True
        return Record(
            event_id=f"step-tool-result-{run_id}",
            session_id=self._session_id,
            agent_id=self._adapter.agent_id,
            event_type="tool_result",
            participant_id=self._adapter._agent_participant_id,
            content=None,
            category="Trace",
            interaction_id=self._interaction_id,
            variable={"Name": tool_name, "OutputValueText": output_text},
            metadata=metadata,
            timestamp=timestamp,
        )

    def finish_pending(self) -> list[Record]:
        """Error results for tools that started but never ended (e.g. cancelled)."""
        with self._lock:
            pending, self._starts = self._starts, {}
        now = datetime.now(timezone.utc)
        return [
            self._result_record(
                run_id, name, "Tool did not finish before the turn ended", now,
                error=True,
            )
            for run_id, (name, _args, _started) in pending.items()
        ]

    def _buffer(self, record: Record) -> None:
        with self._lock:
            self._records.append(record)

    def drain(self) -> list[Record]:
        """Return and clear the tool records captured since the last drain."""
        with self._lock:
            records, self._records = self._records, []
        return records


__all__ = [
    "LangGraphAdapter",
    "_RECALL_TOOL_SYSTEM_INSTRUCTION",
]
