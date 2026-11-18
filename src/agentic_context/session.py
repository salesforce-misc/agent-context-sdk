# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Session Manager — sync and async lifecycle middleware.

Sits between adapter and IngestionClient, orchestrating the Beacon
conversation/session lifecycle. The adapter owns timing (when to ingest);
the manager owns lifecycle (create/close); the client owns HTTP.

Flow is strictly one-directional: adapter -> SessionManager -> IngestionClient.
The manager NEVER calls back into the adapter.

Design rule: One session_id maps to one Beacon conversation + session (same ID
for both). Sessions do not need to be explicitly closed. Returning users to the
same session_id get their past data without creating a new session.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from . import _retry
from .callbacks import IngestionCallbacks
from .client import IngestionClient, IngestionError
from .client_async import AsyncIngestionClient
from .models import Record, to_ace_timestamp
from .retrieval import AsyncRetrievalClient, RetrievalClient

logger = logging.getLogger(__name__)

_ERR_SESSION_CLOSED = "Session already closed"

# ─── Payload Chunking ─────────────────────────────────────────────────────────
# Beacon's Streaming Ingest API has a 200KB per-request payload limit.
# We use 180KB as a safe ceiling to account for JSON serialization overhead
# (wrapper fields like sessionId, conversationOwnerId, etc.).
_MAX_PAYLOAD_BYTES = 180 * 1024  # 180KB — safe margin below 200KB API limit

# ─── Async Delivery Backpressure ──────────────────────────────────────────────
# AsyncSessionManager.ingest() is fire-and-forget: every call schedules an
# asyncio.create_task and returns in <1 ms (the non-blocking guarantee). If Beacon is healthy each task finishes in ~50 ms and the
# in-flight set stays tiny. If Beacon degrades (slow POSTs + 30 s retry
# backoff), a chatty agent can accumulate thousands of pending tasks — each
# retaining its records batch, retry state, and an HTTP connection — until
# the host process OOMs and every in-flight record is lost at once.
#
# Cap the in-flight set at _MAX_PENDING_DELIVERIES. When saturated, we DROP
# THE NEW BATCH: log an error, fire on_error with a RuntimeError, and
# return normally. This keeps the customer's agent alive and the memory
# bounded during a Beacon outage. We deliberately reject the two other
# options:
#   - Backpressure (await a slot): breaks the <1 ms ingest() contract.
#     If Beacon is down for 5 min, the agent's event loop freezes for 5 min.
#   - Drop-oldest (cancel the oldest task): cancels tasks mid-retry, still
#     loses data, and hides the failure signal from on_error.
#
# 1000 is picked so a 20-record/turn agent at 17 turns/s tolerates ~60 s of
# Beacon slowdown before dropping, and memory stays at ~10 MB (1000 tasks
# at ~10 KB each) — cheap. Not customer-tunable, matching the fixed retry
# policy in _retry.py: knobs invite mis-tuning, and the failure mode is
# one we own.
_MAX_PENDING_DELIVERIES = 1000


def _build_agent_participant(
    agent_id: str,
    start_time: str,
    variable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Agent-participant dict used by add_participants everywhere in the SDK.

    Shape mirrors the ACE Postman reference: ``endTime: null``, ``orgId``,
    plus a ``variable`` block feeding Beacon's AiAgentParticipant DMO. The
    SDK is framework-agnostic, so the default ``AiAgentType`` is ``"Custom"``.
    Adapters that know a better label (``"LangGraph"``, ``"ClaudeAgentSDK"``,
    ``"GoogleADK"``, ``"Agentforce"`` …) pass ``variable=`` through the
    SessionManager's ``agent_participant_variable=`` kwarg — a full-override:
    the supplied dict replaces the default block entirely.
    """
    return {
        "id": agent_id,
        "type": "agent",
        "role": "participant",
        "startTime": start_time,
        "endTime": None,
        "orgId": "default",
        "variable": variable if variable is not None else {
            "AiAgentType": "Custom",
            "AiAgentApiName": agent_id,
            "AiAgentVersionApiName": "v1",
            "AiAgentTemplateApiName": "default",
        },
    }


def _chunk_entries(
    entries: list[dict[str, Any]],
    max_bytes: int = _MAX_PAYLOAD_BYTES,
) -> list[list[dict[str, Any]]]:
    """Split entries into chunks that each fit under the payload size limit.

    Uses greedy accumulation: adds entries to the current chunk until adding
    the next entry would exceed max_bytes, then starts a new chunk.

    The overhead of the wrapper payload (sessionId, conversationOwnerId, etc.)
    is ~150 bytes — already accounted for in the 180KB vs 200KB margin.

    Returns a list of entry lists. If all entries fit in one request (the
    common case for typical agent turns), returns a single-element list.
    """
    if not entries:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_size = 0

    for entry in entries:
        entry_size = len(json.dumps(entry, separators=(",", ":")))
        if current_chunk and current_size + entry_size > max_bytes:
            chunks.append(current_chunk)
            current_chunk = [entry]
            current_size = entry_size
        else:
            current_chunk.append(entry)
            current_size += entry_size

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) > 1:
        logger.info(
            "Payload exceeded %dKB limit; split %d entries into %d chunks",
            max_bytes // 1024, len(entries), len(chunks),
        )

    return chunks


class SessionManager:
    """Sync middleware. Orchestrates Beacon API lifecycle for a single session.

    Pairs with sync IngestionClient.

    Design rule: session_id maps to both Beacon's conversationId and sessionId
    (same value for both). Sessions do not need to be explicitly closed.
    Returning users to the same session_id get their past data (via get_context)
    without creating a new session.

    Handles:
    - Lazy creation of conversation + session on first ingest()
    - Transforming Records to Beacon entry format
    - Always POSTs on ingest() — no buffering, no mode flag
    - Internal retry with exponential backoff for transient errors
    - Clean shutdown via close()

    NO in-memory buffering. No circular dependency with adapter.

    Cadence (per-turn vs end-of-session) is decided entirely by the adapter
    via WHEN it calls ingest(). The manager is cadence-agnostic.
    """

    def __init__(
        self,
        client: IngestionClient,
        *,
        agent_id: str,
        owner_id: str,
        owner_type: str = "individual",
        session_id: str | None = None,
        callbacks: IngestionCallbacks | None = None,
        retrieval_client: RetrievalClient | None = None,
        channel: dict[str, Any] | None = None,
        variable: dict[str, Any] | None = None,
        agent_participant_variable: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._retrieval = retrieval_client
        self._agent_id = agent_id
        self._owner_id = owner_id
        self._owner_type = owner_type
        self._session_id = session_id
        self._callbacks = callbacks
        self._channel = channel
        self._variable = variable
        self._agent_participant_variable = agent_participant_variable

        # State (minimal — no record buffer)
        self._conversation_created = False
        self._session_created = False
        self._closed = False

    @property
    def session_id(self) -> str | None:
        """The resolved session ID."""
        return self._session_id

    @property
    def owner_id(self) -> str:
        """The conversation owner this manager posts for."""
        return self._owner_id

    def ingest(self, records: list[Record]) -> None:
        """Ingest a batch of Records through the Beacon lifecycle.

        Creates lifecycle objects (conversation + session) if needed on first
        call, then POSTs entries with automatic retry for transient failures.
        """
        if self._closed:
            raise RuntimeError(_ERR_SESSION_CLOSED)
        if not records:
            return

        # Capture session_id from first record if not set
        if not self._session_id:
            self._session_id = records[0].session_id

        self._ensure_lifecycle()
        entries = self._to_beacon_entries(records, self._owner_id)
        chunks = _chunk_entries(entries)

        for chunk in chunks:
            for attempt in range(_retry._MAX_RETRIES + 1):
                try:
                    self._client.add_entries(
                        self._session_id, self._session_id, chunk,
                        owner_id=self._owner_id, agent_id=self._agent_id,
                    )
                    break
                except Exception as e:
                    if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                        delay = _retry.backoff(attempt)
                        self._safe_callback("on_retry", records, e, attempt, delay)
                        time.sleep(delay)
                    else:
                        self._safe_callback("on_error", records, e, attempt + 1)
                        raise

        self._safe_callback("on_success", records, 202)

    def close(self) -> None:
        """Mark the manager as closed. Local cleanup only — no close_session to Beacon.

        Sessions do not need to be explicitly closed in Beacon.
        The adapter is responsible for calling ingest() with all remaining data
        BEFORE calling close(). The manager never calls back into the adapter.
        """
        if self._closed:
            return
        self._closed = True

    # ─── Retrieval Methods ──────────────────────────────────────────

    def get_context(self) -> dict[str, Any] | None:
        """Retrieve context for the current session.

        Call before each LLM turn and inject into the system prompt.
        Returns None if no retrieval client is configured or on any error.
        Best-effort — never raises.
        """
        if not self._retrieval or not self._session_id:
            return None
        try:
            return self._retrieval.get_context(
                self._session_id, self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "SessionManager.get_context failed (best-effort, returning None): %s", e,
            )
            return None

    def get_session(self) -> dict[str, Any] | None:
        """Retrieve full session data (messages, metadata).

        Returns None if no retrieval client or on error.
        """
        if not self._retrieval or not self._session_id:
            return None
        try:
            return self._retrieval.get_conversation(
                self._session_id, self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "SessionManager.get_session failed (best-effort, returning None): %s", e,
            )
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions for this owner.

        Returns empty list if no retrieval client or on error.
        """
        if not self._retrieval:
            return []
        try:
            return self._retrieval.list_conversations(
                self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "SessionManager.list_sessions failed (best-effort, returning []): %s", e,
            )
            return []

    # ─── Eager Lifecycle ──────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        *,
        title: str = "",
    ) -> dict[str, str]:
        """Eagerly create conversation + session + participants in Beacon.

        Uses session_id as both Beacon's conversationId and sessionId (unified).

        Use when you need lifecycle objects to exist BEFORE any message is
        sent — e.g., a UI agent where the user clicks "+ New Conversation"
        and the frontend needs IDs immediately.

        After this call, ``ingest()`` will skip lazy lifecycle creation and
        go straight to ``add_entries``.

        If this session already exists in Beacon (409), the call succeeds
        silently — returning users to the same session_id just works.

        Args:
            session_id: Unique ID for the session (maps to both Beacon
                conversationId and sessionId).
            title: Optional human-readable title (passed to Beacon).

        Returns:
            Dict with ``sessionId``, ``ownerId``, ``agentId``
            confirming what was created.

        Raises:
            IngestionError: If any lifecycle call fails (non-409).
            RuntimeError: If the manager is already closed.
        """
        if self._closed:
            raise RuntimeError(_ERR_SESSION_CLOSED)

        self._session_id = session_id

        # Create conversation (409 = already exists = OK)
        try:
            self._client.create_conversation(
                self._session_id, self._owner_id, self._agent_id,
                name=title or None,
                owner_type=self._owner_type,
                channel=self._channel,
                variable=self._variable,
            )
        except IngestionError as e:
            if e.status_code != 409:
                raise
        self._conversation_created = True

        # Create session with same ID (409 = OK — returning user)
        try:
            self._client.create_session(
                self._session_id, self._session_id,
                self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except IngestionError as e:
            if e.status_code != 409:
                raise

        # Add agent as participant (two-step: session created with owner,
        # then agent added separately — Sylvester's fix for get_context scoping)
        now_iso = to_ace_timestamp()
        try:
            self._client.add_participants(
                self._session_id, self._session_id,
                self._owner_id, self._agent_id,
                participants=[_build_agent_participant(self._agent_id, now_iso, self._agent_participant_variable)],
            )
        except IngestionError as e:
            if e.status_code != 409:
                raise
        self._session_created = True

        return {
            "sessionId": session_id,
            "ownerId": self._owner_id,
            "agentId": self._agent_id,
        }

    # ─── Lifecycle (internal) ────────────────────────────────────

    def _ensure_lifecycle(self) -> None:
        """Lazy-create conversation + session on first actual POST.

        Uses session_id as both Beacon's conversationId and sessionId (unified).

        Treat HTTP 409 (Conflict / already-exists) as success so a partial
        failure on a previous attempt — or a returning user — does not wedge
        subsequent ingest calls into a permanent retry-conflict loop.
        """
        if not self._conversation_created:
            try:
                self._client.create_conversation(
                    self._session_id, self._owner_id, self._agent_id,
                    owner_type=self._owner_type,
                    channel=self._channel,
                    variable=self._variable,
                )
            except IngestionError as e:
                if e.status_code != 409:
                    raise
            self._conversation_created = True

        if not self._session_created:
            try:
                self._client.create_session(
                    self._session_id, self._session_id,
                    self._owner_id, self._agent_id,
                    owner_type=self._owner_type,
                )
            except IngestionError as e:
                if e.status_code != 409:
                    raise
            now_iso = to_ace_timestamp()
            try:
                self._client.add_participants(
                    self._session_id, self._session_id,
                    self._owner_id, self._agent_id,
                    participants=[_build_agent_participant(self._agent_id, now_iso, self._agent_participant_variable)],
                )
            except IngestionError as e:
                if e.status_code != 409:
                    raise
            self._session_created = True

    def _safe_callback(self, method: str, *args: Any) -> None:
        """Call callback safely — never let callback errors disrupt delivery."""
        if self._callbacks:
            try:
                getattr(self._callbacks, method)(*args)
            except Exception:
                pass

    @staticmethod
    def _to_beacon_entries(records: list[Record], owner_id: str) -> list[dict[str, Any]]:
        """Transform Records to Beacon entry format.

        Direct field mapping — Record fields already use the Beacon wire
        vocabulary. Trace ``variable`` keys are copied as-is, with a
        compatibility copy of legacy ``InputValue``/``OutputValue`` onto
        ``InputValueText``/``OutputValueText`` when the new keys are absent.
        ``role`` on each entry is derived from ``participant_id`` — if it
        matches the conversation owner, role is ``"owner"``; otherwise
        ``"participant"``.
        """
        entries: list[dict[str, Any]] = []
        for r in records:
            participant_id = r.participant_id or r.agent_id
            # Omit id. Beacon dedupes on a client-supplied id, and the
            # adapter's fallback event_id repeats msg-{session}-0 on every
            # turn, so sending it would collapse later messages. Beacon assigns
            # the entry id; with none sent, nothing is deduped.
            entry: dict[str, Any] = {
                "participantId": participant_id,
                "type": r.event_type,
                # role per Beacon's wire shape: "owner" if this entry came
                # from the conversation owner, "participant" otherwise.
                "role": "owner" if participant_id == owner_id else "participant",
                "timestamp": to_ace_timestamp(r.timestamp),
                "category": r.category,
            }
            # interactionId is optional per API — only include if set
            if r.interaction_id:
                entry["interactionId"] = r.interaction_id
            # Add content for Message entries (user_message, agent_response)
            if r.content is not None:
                entry["content"] = r.content
            if r.content_type is not None:
                entry["contentType"] = r.content_type
            if r.attachments:
                entry["attachments"] = r.attachments
            # Add variable for Trace entries (tool_call, tool_result, agent_reasoning)
            if r.variable:
                entry["variable"] = SessionManager._normalize_trace_variable(r.variable)
            entries.append(entry)
        return entries

    # Beacon ingest projects entity-XML keys.
    # Copy legacy aliases onto the new names so callers who still build
    # Record.variable with InputValue/OutputValue are not silently dropped.
    _TRACE_KEY_ALIASES = (
        ("InputValue", "InputValueText"),
        ("OutputValue", "OutputValueText"),
    )

    @staticmethod
    def _normalize_trace_variable(variable: dict[str, Any]) -> dict[str, Any]:
        """Copy legacy trace keys onto the current Beacon names.

        Beacon's ingest converter reads ``InputValueText``/``OutputValueText``;
        callers still emitting the legacy ``InputValue``/``OutputValue`` would be
        silently dropped. Copy old→new when the new key is absent (new keys win).
        """
        out = dict(variable)
        for old, new in SessionManager._TRACE_KEY_ALIASES:
            if new not in out and old in out:
                out[new] = out[old]
        return out


class AsyncSessionManager:
    """Async, non-blocking middleware. Orchestrates Beacon API lifecycle.

    Pairs with AsyncIngestionClient.

    Design rule: session_id maps to both Beacon's conversationId and sessionId
    (same value for both). Sessions do not need to be explicitly closed.
    Returning users to the same session_id get their past data (via get_context)
    without creating a new session.

    Key difference from SessionManager: ingest() is NON-BLOCKING.
    It fires a background task and returns immediately. The adapter/agent
    is never stalled by Beacon latency.

    Delivery (with retries) happens in background asyncio tasks.
    close() flushes all pending tasks.

    Flow is strictly one-directional: adapter -> AsyncSessionManager -> AsyncIngestionClient
    """

    def __init__(
        self,
        client: AsyncIngestionClient,
        *,
        agent_id: str,
        owner_id: str,
        owner_type: str = "individual",
        session_id: str | None = None,
        callbacks: IngestionCallbacks | None = None,
        retrieval_client: AsyncRetrievalClient | None = None,
        channel: dict[str, Any] | None = None,
        variable: dict[str, Any] | None = None,
        agent_participant_variable: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._retrieval = retrieval_client
        self._agent_id = agent_id
        self._owner_id = owner_id
        self._owner_type = owner_type
        self._session_id = session_id
        self._callbacks = callbacks
        self._channel = channel
        self._variable = variable
        self._agent_participant_variable = agent_participant_variable

        self._conversation_created = False
        self._session_created = False
        self._closed = False
        # Two locks with different hold-times. ``_state_lock`` guards the
        # fast, IO-free critical sections (closed-flag check, session_id
        # resolution) that ingest()/close() run; it is never held across an
        # await on the network, so ingest() returns without ever waiting on
        # an in-flight POST — the non-blocking guarantee. ``_lifecycle_lock``
        # serializes the lifecycle HTTP (create_conversation/session/
        # participants) in the background delivery task, which DOES span
        # network awaits. Keeping them separate is what stops a turn-2
        # ingest() from stalling behind turn-1's lifecycle round-trips.
        self._state_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()

    @property
    def session_id(self) -> str | None:
        """The resolved session ID."""
        return self._session_id

    @property
    def owner_id(self) -> str:
        """The conversation owner this manager posts for."""
        return self._owner_id

    async def ingest(self, records: list[Record]) -> None:
        """Non-blocking. Fires a background task and returns immediately.

        The caller (adapter/agent framework) is never stalled by Beacon latency.
        Delivery happens in the background with automatic retries.

        Backpressure: if ``_MAX_PENDING_DELIVERIES`` background deliveries are
        already in flight (Beacon is degraded and tasks aren't draining), this
        batch is DROPPED — an error is logged, ``on_error`` fires with a
        ``RuntimeError``, and the call returns normally. This preserves the
        non-blocking guarantee and bounds memory during an outage rather than
        letting the process OOM and lose every pending batch at once. See the
        ``_MAX_PENDING_DELIVERIES`` block comment above for the rationale.

        Raises ``RuntimeError`` if the session is already closed (or being
        closed concurrently) so callers can't lose records into the void.
        Raises ``ValueError`` if a record's ``session_id`` disagrees with the
        manager's resolved session — managers are bound to one session.
        """
        if self._closed:
            raise RuntimeError(_ERR_SESSION_CLOSED)
        if not records:
            return

        # Resolve identity under the fast state lock (IO-free — never held
        # across a network await), so this returns immediately even while a
        # prior turn's lifecycle POSTs are in flight under _lifecycle_lock.
        # Two concurrent ingests can't race here, and a record from a
        # different session is rejected loudly instead of silently mis-routed.
        async with self._state_lock:
            if self._closed:
                raise RuntimeError(_ERR_SESSION_CLOSED)
            first_session = records[0].session_id
            if not self._session_id:
                self._session_id = first_session
            elif first_session and first_session != self._session_id:
                raise ValueError(
                    f"Record session_id={first_session!r} does not match "
                    f"this manager's session_id={self._session_id!r}; "
                    "managers are bound to a single session."
                )

        # Bounded fan-out. Checked AFTER _state_lock so a saturated queue
        # still gets a session_id/closed check first (loud rejection over
        # silent drop where possible). No lock needed for the len() read
        # itself: set-size races only shift the boundary by a handful of
        # tasks either way — well within the safety margin baked into 1000.
        if len(self._pending_tasks) >= _MAX_PENDING_DELIVERIES:
            error = RuntimeError(
                f"AsyncSessionManager saturated: {_MAX_PENDING_DELIVERIES} "
                "background deliveries already pending. Dropping this batch "
                "to protect the host process from OOM. Beacon is likely "
                "degraded — check ingestion latency and retry-rate metrics."
            )
            # error, not warning: this is a records-lost event. Default-path
            # callers (callbacks=None) get no other signal — a warning-level
            # log is easy to filter out and hides real data loss.
            logger.error(
                "ingest(): dropping %d record(s) — %d pending deliveries "
                "(cap=%d). Beacon appears degraded.",
                len(records), len(self._pending_tasks), _MAX_PENDING_DELIVERIES,
            )
            self._safe_callback("on_error", records, error, 0)
            return

        task = asyncio.create_task(self._deliver(records))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def close(self) -> None:
        """Flush all pending background deliveries. Local cleanup only.

        Sessions do not need to be explicitly closed in Beacon.
        Must be called on shutdown to ensure no in-flight data is lost.
        Idempotent and safe under concurrent invocation: only the first caller
        drains; the rest return immediately.
        """
        # Take ownership atomically so concurrent close() calls and any
        # ingest() that interleaves see _closed=True from this point on.
        # Uses the fast state lock (same one ingest() checks _closed under),
        # not _lifecycle_lock — close() must not block behind an in-flight
        # lifecycle POST just to flip a flag.
        async with self._state_lock:
            if self._closed:
                return
            self._closed = True

        # Snapshot + drain. Loop in case in-flight tasks scheduled new work
        # before we set _closed (impossible after the lock above, but cheap).
        pending = list(self._pending_tasks)
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=30.0)
            if still_pending:
                logger.warning(
                    "close(): %d delivery task(s) exceeded 30s drain budget; "
                    "cancelling. Records may not have reached Beacon.",
                    len(still_pending),
                )
                for t in still_pending:
                    t.cancel()
                await asyncio.gather(*still_pending, return_exceptions=True)

    async def flush(self, timeout: float = 20.0) -> None:
        """Await in-flight background deliveries WITHOUT closing the manager.

        ``ingest()`` is non-blocking — it schedules delivery on a background
        task and returns. In an environment that suspends the event loop the
        instant a request's response is sent (e.g. Cloud Run's request-scoped
        CPU allocation), those tasks may never run to completion, so nothing
        reaches Beacon. Call ``flush()`` at end-of-turn — while the request
        still holds CPU — to await the pending deliveries so they actually
        land, while leaving the manager OPEN for the next turn (unlike
        :meth:`close`, which marks it closed). Best-effort: stragglers past
        ``timeout`` are left running (not cancelled) and drained later by the
        next ``flush()`` or by ``close()``.
        """
        pending = list(self._pending_tasks)
        if not pending:
            return
        await asyncio.wait(pending, timeout=timeout)

    # ─── Retrieval Methods ──────────────────────────────────────────

    async def get_context(self) -> dict[str, Any] | None:
        """Retrieve context for the current session (async).

        Call before each LLM turn and inject into the system prompt.
        Returns None if no retrieval client is configured or on any error.
        Best-effort — never raises.
        """
        if not self._retrieval or not self._session_id:
            return None
        try:
            return await self._retrieval.get_context(
                self._session_id, self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "AsyncSessionManager.get_context failed (best-effort, returning None): %s", e,
            )
            return None

    async def get_session(self) -> dict[str, Any] | None:
        """Retrieve full session data (messages, metadata) — async.

        Returns None if no retrieval client or on error.
        """
        if not self._retrieval or not self._session_id:
            return None
        try:
            return await self._retrieval.get_conversation(
                self._session_id, self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "AsyncSessionManager.get_session failed (best-effort, returning None): %s", e,
            )
            return None

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions for this owner (async).

        Returns empty list if no retrieval client or on error.
        """
        if not self._retrieval:
            return []
        try:
            return await self._retrieval.list_conversations(
                self._owner_id, self._agent_id,
                owner_type=self._owner_type,
            )
        except Exception as e:
            logger.warning(
                "AsyncSessionManager.list_sessions failed (best-effort, returning []): %s", e,
            )
            return []

    # ─── Eager Lifecycle ──────────────────────────────────────────

    async def create_session(
        self,
        session_id: str,
        *,
        title: str = "",
    ) -> dict[str, str]:
        """Eagerly create conversation + session + participants in Beacon (async).

        Uses session_id as both Beacon's conversationId and sessionId (unified).

        Use when you need lifecycle objects to exist BEFORE any message is
        sent — e.g., a UI agent where the user clicks "+ New Conversation"
        and the frontend needs IDs immediately.

        After this call, ``ingest()`` will skip lazy lifecycle creation and
        go straight to ``add_entries``.

        If this session already exists in Beacon (409), the call succeeds
        silently — returning users to the same session_id just works.

        Args:
            session_id: Unique ID for the session (maps to both Beacon
                conversationId and sessionId).
            title: Optional human-readable title (passed to Beacon).

        Returns:
            Dict with ``sessionId``, ``ownerId``, ``agentId``
            confirming what was created.

        Raises:
            IngestionError: If any lifecycle call fails (non-409).
            RuntimeError: If the manager is already closed.
        """
        if self._closed:
            raise RuntimeError(_ERR_SESSION_CLOSED)

        async with self._lifecycle_lock:
            self._session_id = session_id

            # Create conversation (409 = already exists = OK)
            try:
                await self._client.create_conversation(
                    self._session_id, self._owner_id, self._agent_id,
                    name=title or None,
                    owner_type=self._owner_type,
                    channel=self._channel,
                    variable=self._variable,
                )
            except IngestionError as e:
                if e.status_code != 409:
                    raise
            self._conversation_created = True

            # Create session with same ID (409 = OK — returning user)
            try:
                await self._client.create_session(
                    self._session_id, self._session_id,
                    self._owner_id, self._agent_id,
                    owner_type=self._owner_type,
                )
            except IngestionError as e:
                if e.status_code != 409:
                    raise

            # Add agent as participant
            now_iso = to_ace_timestamp()
            try:
                await self._client.add_participants(
                    self._session_id, self._session_id,
                    self._owner_id, self._agent_id,
                    participants=[_build_agent_participant(self._agent_id, now_iso, self._agent_participant_variable)],
                )
            except IngestionError as e:
                if e.status_code != 409:
                    raise
            self._session_created = True

        return {
            "sessionId": session_id,
            "ownerId": self._owner_id,
            "agentId": self._agent_id,
        }

    # ─── Internal ────────────────────────────────────────────────

    async def _deliver(self, records: list[Record]) -> None:
        """Background delivery with retry. Runs as a detached asyncio task."""
        entries = SessionManager._to_beacon_entries(records, self._owner_id)
        chunks = _chunk_entries(entries)

        for chunk in chunks:
            for attempt in range(_retry._MAX_RETRIES + 1):
                try:
                    # Lifecycle creation is part of delivery, so it must share
                    # the retry + on_error treatment of add_entries. It used to
                    # be awaited ONCE before this loop, outside the try — a
                    # non-409 lifecycle failure (503, 500, ConnectionError…)
                    # then raised out of this detached task, whose done-callback
                    # only does `.discard`: never retried, on_error never fired,
                    # records silently lost. Now it runs inside the attempt so a
                    # transient lifecycle failure is retried and a terminal one
                    # is reported. It is flag-guarded and idempotent (409 ==
                    # success), so re-running it on a retry — or on later chunks
                    # — is a cheap no-op once the session exists, and a partial
                    # failure resumes from whichever step didn't complete.
                    await self._ensure_lifecycle()
                    await self._client.add_entries(
                        self._session_id, self._session_id, chunk,
                        owner_id=self._owner_id, agent_id=self._agent_id,
                    )
                    break
                except asyncio.CancelledError as e:
                    self._safe_callback("on_error", records, e, attempt + 1)
                    raise
                except Exception as e:
                    if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                        delay = _retry.backoff(attempt)
                        self._safe_callback("on_retry", records, e, attempt, delay)
                        try:
                            await asyncio.sleep(delay)
                        except asyncio.CancelledError as ce:
                            self._safe_callback("on_error", records, ce, attempt + 1)
                            raise
                    else:
                        # error, not warning: this is a records-lost event.
                        # Default-path callers (callbacks=None) get no other
                        # signal — without this line the background task
                        # dies and the record vanishes with zero trace.
                        logger.error(
                            "_deliver: dropping %d record(s) after %d attempt(s) — %s",
                            len(records), attempt + 1, e,
                        )
                        self._safe_callback("on_error", records, e, attempt + 1)
                        return

        self._safe_callback("on_success", records, 202)

    async def _ensure_lifecycle(self) -> None:
        """Lazy-create conversation + session on first POST (async, lock-protected).

        Uses session_id as both Beacon's conversationId and sessionId (unified).

        Treat HTTP 409 (Conflict / already-exists) as success so a partial
        failure on a previous attempt — or a returning user — does not
        permanently wedge subsequent ingest calls.
        """
        async with self._lifecycle_lock:
            if not self._conversation_created:
                try:
                    await self._client.create_conversation(
                        self._session_id, self._owner_id, self._agent_id,
                        owner_type=self._owner_type,
                        channel=self._channel,
                        variable=self._variable,
                    )
                except IngestionError as e:
                    if e.status_code != 409:
                        raise
                self._conversation_created = True

            if not self._session_created:
                try:
                    await self._client.create_session(
                        self._session_id, self._session_id,
                        self._owner_id, self._agent_id,
                        owner_type=self._owner_type,
                    )
                except IngestionError as e:
                    if e.status_code != 409:
                        raise
                now_iso = to_ace_timestamp()
                try:
                    await self._client.add_participants(
                        self._session_id, self._session_id,
                        self._owner_id, self._agent_id,
                        participants=[_build_agent_participant(self._agent_id, now_iso, self._agent_participant_variable)],
                    )
                except IngestionError as e:
                    if e.status_code != 409:
                        raise
                self._session_created = True

    def _safe_callback(self, method: str, *args: Any) -> None:
        """Call callback safely — never let callback errors disrupt delivery."""
        if self._callbacks:
            try:
                getattr(self._callbacks, method)(*args)
            except Exception:
                pass
