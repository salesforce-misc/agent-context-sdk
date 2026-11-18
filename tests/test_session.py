# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for SessionManager (sync) and AsyncSessionManager.

Design rule: session_id maps to one Beacon conversation + session (same ID).
Sessions do not need to be explicitly closed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

from agentic_context import Record, SessionManager, AsyncSessionManager


# ─── Helpers ────────────────────────────────────────────────────────

def _make_record(
    event_id: str = "evt-1",
    session_id: str = "ses-1",
    event_type: str = "user_message",
    category: str = "Message",
    content: str | None = "hello",
    interaction_id: str | None = "turn-001",
    variable: dict[str, Any] | None = None,
) -> Record:
    return Record(
        event_id=event_id,
        session_id=session_id,
        agent_id="agent-1",
        event_type=event_type,
        participant_id="user-001",
        content=content,
        category=category,
        interaction_id=interaction_id,
        variable=variable,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.create_conversation.return_value = {}
    client.create_session.return_value = {}
    client.add_participants.return_value = {}
    client.add_entries.return_value = {}
    client.close_session.return_value = {}
    return client


def _async_mock_client() -> MagicMock:
    client = MagicMock()
    client.create_conversation = AsyncMock(return_value={})
    client.create_session = AsyncMock(return_value={})
    client.add_participants = AsyncMock(return_value={})
    client.add_entries = AsyncMock(return_value={})
    client.close_session = AsyncMock(return_value={})
    return client


class _Mode:
    """Runs one test body against SessionManager (sync) or AsyncSessionManager.

    ``count`` reads await_count in async mode so a coroutine that is called but
    never awaited still fails the assertion.
    """

    def __init__(self, is_async: bool) -> None:
        self.is_async = is_async
        self.client = _async_mock_client() if is_async else _mock_client()

    def manager(self, **kwargs: Any):
        cls = AsyncSessionManager if self.is_async else SessionManager
        kwargs.setdefault("agent_id", "agent-1")
        kwargs.setdefault("owner_id", "owner-1")
        return cls(self.client, **kwargs)

    async def run(self, value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value

    async def ingest_and_flush(self, mgr, *batches: list[Record]) -> None:
        """Ingest each batch; in async mode close() to drain background delivery."""
        for batch in batches:
            await self.run(mgr.ingest(batch))
        if self.is_async:
            await mgr.close()

    def count(self, mock: MagicMock) -> int:
        return mock.await_count if self.is_async else mock.call_count

    def fail_with(self, name: str, error: Exception) -> None:
        if self.is_async:
            setattr(self.client, name, AsyncMock(side_effect=error))
        else:
            getattr(self.client, name).side_effect = error


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def mode(request) -> _Mode:
    return _Mode(request.param)


def test_owner_id_property_is_read_only(mode: _Mode) -> None:
    """Both managers expose the owner they post for; adapters read it to tag
    user records (Beacon's role="owner" needs participantId == ownerId)."""
    mgr = mode.manager(owner_id="owner-xyz")
    assert mgr.owner_id == "owner-xyz"
    with pytest.raises(AttributeError):
        mgr.owner_id = "other"


# ─── Lazy lifecycle (sync + async) ──────────────────────────────

class TestLazyLifecycle:
    """ingest() creates the Beacon lifecycle on first use. In async mode
    ingest() is fire-and-forget, so close() flushes before asserting."""

    async def test_ingest_creates_lifecycle_on_first_call(self, mode: _Mode) -> None:
        mgr = mode.manager()

        await mode.ingest_and_flush(mgr, [_make_record()])

        assert mode.count(mode.client.create_conversation) == 1
        assert mode.count(mode.client.create_session) == 1
        assert mode.count(mode.client.add_entries) == 1

    async def test_later_ingests_only_post_entries(self, mode: _Mode) -> None:
        """Every ingest() delivers (in background for async); lifecycle is created once."""
        mgr = mode.manager()

        await mode.ingest_and_flush(
            mgr,
            [_make_record(event_id="evt-1")],
            [_make_record(event_id="evt-2")],
            [_make_record(event_id="evt-3")],
        )

        assert mode.count(mode.client.create_conversation) == 1
        assert mode.count(mode.client.create_session) == 1
        assert mode.count(mode.client.add_entries) == 3

    async def test_ingest_empty_list_is_noop(self, mode: _Mode) -> None:
        mgr = mode.manager()

        await mode.ingest_and_flush(mgr, [])

        mode.client.create_conversation.assert_not_called()
        mode.client.add_entries.assert_not_called()

    async def test_lifecycle_409_treated_as_success(self, mode: _Mode) -> None:
        """409 from create_conversation on the lazy ingest path is success."""
        from agentic_context.client import IngestionError

        mode.fail_with("create_conversation", IngestionError(409, "already exists"))
        mgr = mode.manager(session_id="ses-1")

        await mode.ingest_and_flush(mgr, [_make_record()])  # should NOT raise

        assert mgr._conversation_created is True
        assert mode.count(mode.client.add_entries) == 1

    def test_sync_lifecycle_non_409_raises(self) -> None:
        """Sync: non-409 errors from create_conversation propagate out of ingest().
        (Async retries lifecycle failures and fires on_error — see
        TestAsyncSessionManager.test_terminal_lifecycle_failure_fires_on_error.)"""
        from agentic_context.client import IngestionError

        client = _mock_client()
        client.create_conversation.side_effect = IngestionError(500, "server error")
        mgr = SessionManager(client, agent_id="agent-1", owner_id="owner-1", session_id="ses-1")

        with pytest.raises(IngestionError) as exc_info:
            mgr.ingest([_make_record()])
        assert exc_info.value.status_code == 500

    @pytest.mark.parametrize(
        ("ctor_session_id", "record_session_id", "expected"),
        [("ses-explicit", "ses-1", "ses-explicit"), (None, "ses-from-record", "ses-from-record")],
        ids=["from-constructor", "from-first-record"],
    )
    async def test_session_id_source(
        self, mode: _Mode, ctor_session_id, record_session_id, expected,
    ) -> None:
        """session_id comes from the constructor, else from the first record's
        session_id, and is used as both conv and session id in Beacon."""
        if mode.is_async:
            # async rejects a record from a different session (tested separately)
            record_session_id = expected
        mgr = mode.manager(session_id=ctor_session_id)

        await mode.ingest_and_flush(mgr, [_make_record(session_id=record_session_id)])

        assert mgr.session_id == expected
        call_args = mode.client.add_entries.call_args
        assert call_args[0][0] == expected  # conversation_id (Beacon)
        assert call_args[0][1] == expected  # session_id (Beacon, same)


# ─── close() (sync + async) ─────────────────────────────────────

class TestClose:
    async def test_ingest_after_close_raises(self, mode: _Mode) -> None:
        mgr = mode.manager()
        await mode.run(mgr.ingest([_make_record()]))
        await mode.run(mgr.close())

        with pytest.raises(RuntimeError, match="Session already closed"):
            await mode.run(mgr.ingest([_make_record(event_id="evt-2")]))

    async def test_close_is_local_only_and_idempotent(self, mode: _Mode) -> None:
        """close() is local cleanup only — no close_session to Beacon — and a
        second call is a noop."""
        mgr = mode.manager()
        await mode.run(mgr.ingest([_make_record()]))

        await mode.run(mgr.close())
        await mode.run(mgr.close())

        mode.client.close_session.assert_not_called()

    async def test_close_without_ingest_does_not_call_close_session(self, mode: _Mode) -> None:
        mgr = mode.manager()

        await mode.run(mgr.close())

        mode.client.close_session.assert_not_called()


class TestToBeaconEntries:
    """Test the static _to_beacon_entries transformation."""

    def test_message_entry_format(self) -> None:
        records = [_make_record(), _make_record(event_id="evt-2", interaction_id=None)]
        entries = SessionManager._to_beacon_entries(records, "user-001")

        assert len(entries) == 2
        e = entries[0]
        assert "id" not in e
        assert e["participantId"] == "user-001"
        assert e["type"] == "user_message"
        assert e["category"] == "Message"
        assert e["content"] == "hello"
        assert e["interactionId"] == "turn-001"
        # Wire timestamp is millisecond-precision UTC with a Z suffix
        # (to_ace_timestamp) so Data Cloud can coerce the dateTime field.
        assert e["timestamp"] == "2026-01-01T12:00:00.000Z"
        # No variable for Message entries
        assert "variable" not in e
        # interactionId is omitted when the record has none
        assert "interactionId" not in entries[1]

    def test_trace_entry_format(self) -> None:
        records = [_make_record(
            event_id="step-1",
            event_type="tool_call",
            category="Trace",
            content=None,
            variable={"Name": "OrderLookup", "InputValueText": '{"orderId":"123"}'},
        )]
        entries = SessionManager._to_beacon_entries(records, "user-001")

        e = entries[0]
        assert e["type"] == "tool_call"
        assert e["category"] == "Trace"
        assert "content" not in e  # content is None, should not be in entry
        assert e["variable"]["Name"] == "OrderLookup"
        assert e["variable"]["InputValueText"] == '{"orderId":"123"}'

    @pytest.mark.parametrize(
        ("event_type", "variable", "expected"),
        [
            (
                "tool_call",
                {"Name": "OrderLookup", "InputValue": '{"orderId":"123"}'},
                {"Name": "OrderLookup", "InputValue": '{"orderId":"123"}', "InputValueText": '{"orderId":"123"}'},
            ),
            (
                "tool_result",
                {"Name": "OrderLookup", "OutputValue": "old", "OutputValueText": "new"},
                {"OutputValueText": "new"},
            ),
        ],
        ids=["legacy-key-copied", "current-key-not-overwritten"],
    )
    def test_trace_legacy_variable_keys(self, event_type, variable, expected) -> None:
        """Legacy InputValue/OutputValue are copied onto the current
        InputValueText/OutputValueText names the ingest converter reads, the
        legacy key is preserved alongside the new one, and an existing current
        key is never overwritten by the legacy one."""
        records = [_make_record(
            event_id="step-1", event_type=event_type, category="Trace",
            content=None, variable=variable,
        )]
        entries = SessionManager._to_beacon_entries(records, "user-001")

        actual = entries[0]["variable"]
        assert {k: actual.get(k) for k in expected} == expected

    def test_participant_id_falls_back_to_agent_id(self) -> None:
        record = Record(
            event_id="evt-fallback",
            session_id="ses-1",
            agent_id="agent-1",
            event_type="agent_response",
            participant_id=None,
            content="response",
            category="Message",
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        entries = SessionManager._to_beacon_entries([record], "owner-1")
        assert entries[0]["participantId"] == "agent-1"


# ─── AsyncSessionManager-only behaviour ─────────────────────────

class TestAsyncSessionManager:
    """Async-only paths: session-mismatch guard, close() races, background
    retry/cancellation. Non-blocking timing is proven in test_nonblocking_proof.py."""

    @pytest.mark.asyncio
    async def test_ingest_with_mismatched_session_id_raises(self) -> None:
        """A record from a different session must be rejected."""
        client = _async_mock_client()
        mgr = AsyncSessionManager(
            client, agent_id="agent-1", owner_id="owner-1",
            session_id="ses-A",
        )
        with pytest.raises(ValueError, match="does not match"):
            await mgr.ingest([_make_record(session_id="ses-B")])

    @pytest.mark.asyncio
    async def test_concurrent_close_calls_are_safe(self) -> None:
        """Two close() calls racing must not double-close or error."""
        client = _async_mock_client()
        mgr = AsyncSessionManager(client, agent_id="agent-1", owner_id="owner-1")
        await mgr.ingest([_make_record(event_id="evt-1")])

        await asyncio.gather(mgr.close(), mgr.close())
        # No close_session calls (we don't close sessions anymore)
        client.close_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancellation_during_retry_fires_on_error(self) -> None:
        """A delivery cancelled mid-retry-sleep must fire on_error before propagating."""
        from agentic_context import IngestionError

        client = _async_mock_client()

        async def always_503(*args, **kwargs):
            raise IngestionError(503, "down")

        client.add_entries = AsyncMock(side_effect=always_503)

        events: dict[str, list] = {"error": []}

        class _CB:
            def on_success(self, records, status_code):
                pass

            def on_error(self, records, error, attempts):
                events["error"].append(type(error).__name__)

            def on_retry(self, records, error, attempt, delay):
                pass

        mgr = AsyncSessionManager(
            client, agent_id="agent-1", owner_id="owner-1", callbacks=_CB()
        )
        await mgr.ingest([_make_record(event_id="evt-1")])

        # The first add_entries fails immediately; the manager schedules a
        # retry sleep. Cancel the in-flight task before the sleep elapses.
        await asyncio.sleep(0)  # yield once so _deliver starts
        for t in list(mgr._pending_tasks):
            t.cancel()
        # Drain (gather will absorb the CancelledError thanks to return_exceptions)
        await asyncio.gather(*mgr._pending_tasks, return_exceptions=True)

        # on_error must have fired even though we cancelled.
        assert events["error"], "on_error did not fire on cancellation"

    @pytest.mark.asyncio
    async def test_transient_lifecycle_failure_is_retried(self, monkeypatch) -> None:
        """A retryable lifecycle failure (e.g. 503 on create_conversation) is
        retried inside the delivery loop, not raised out of the detached task.

        Regression: _ensure_lifecycle used to be awaited OUTSIDE the retry
        loop, so a transient lifecycle error was never retried and the records
        were silently lost.
        """
        from agentic_context import _retry, IngestionError

        monkeypatch.setattr(_retry, "backoff", lambda attempt: 0.0)

        client = _async_mock_client()
        # First create_conversation attempt fails with a retryable 503; the
        # retry succeeds.
        client.create_conversation = AsyncMock(
            side_effect=[IngestionError(503, "down"), {}]
        )

        events: dict[str, list] = {"success": [], "retry": [], "error": []}

        class _CB:
            def on_success(self, records, status_code):
                events["success"].append(status_code)

            def on_retry(self, records, error, attempt, delay):
                events["retry"].append(attempt)

            def on_error(self, records, error, attempts):
                events["error"].append(attempts)

        mgr = AsyncSessionManager(
            client, agent_id="agent-1", owner_id="owner-1", callbacks=_CB()
        )
        await mgr.ingest([_make_record()])
        await mgr.close()

        assert events["retry"], "on_retry did not fire for a transient lifecycle failure"
        assert events["success"], "delivery did not eventually succeed after retry"
        assert not events["error"], "on_error fired despite a successful retry"
        assert client.create_conversation.await_count == 2
        client.add_entries.assert_awaited()

    @pytest.mark.asyncio
    async def test_terminal_lifecycle_failure_fires_on_error(self, monkeypatch) -> None:
        """A non-retryable lifecycle failure (e.g. 500 that keeps failing) must
        fire on_error and NOT silently vanish from the detached task.
        """
        from agentic_context import _retry, IngestionError

        monkeypatch.setattr(_retry, "backoff", lambda attempt: 0.0)

        client = _async_mock_client()
        client.create_conversation = AsyncMock(side_effect=IngestionError(503, "down"))

        events: dict[str, list] = {"success": [], "error": []}

        class _CB:
            def on_success(self, records, status_code):
                events["success"].append(status_code)

            def on_retry(self, records, error, attempt, delay):
                pass

            def on_error(self, records, error, attempts):
                events["error"].append(attempts)

        mgr = AsyncSessionManager(
            client, agent_id="agent-1", owner_id="owner-1", callbacks=_CB()
        )
        await mgr.ingest([_make_record()])
        await mgr.close()

        assert events["error"], "on_error did not fire when lifecycle failed terminally"
        assert not events["success"], "on_success fired despite lifecycle failure"
        # add_entries must never run if the conversation was never created.
        client.add_entries.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_nonblocking_during_slow_lifecycle(self) -> None:
        """The non-blocking guarantee holds even while a prior turn's lifecycle
        POSTs are in flight — a second ingest() must not block on the
        lifecycle lock held across those HTTP calls.

        Regression: ingest() used to acquire _lifecycle_lock, which _deliver
        holds across three sequential lifecycle HTTP calls, so a turn-2
        ingest() stalled behind full Beacon lifecycle latency.
        """
        import time

        started = asyncio.Event()

        async def slow_create_conversation(*args, **kwargs):
            started.set()
            await asyncio.sleep(1.0)  # simulate slow lifecycle leg
            return {}

        client = _async_mock_client()
        client.create_conversation = AsyncMock(side_effect=slow_create_conversation)
        mgr = AsyncSessionManager(client, agent_id="agent-1", owner_id="owner-1")

        # Turn 1 kicks off the slow lifecycle in the background.
        await mgr.ingest([_make_record(event_id="evt-1")])
        await started.wait()  # ensure the lifecycle POST is mid-flight

        # Turn 2 must still return immediately despite the in-flight lifecycle.
        start = time.perf_counter()
        await mgr.ingest([_make_record(event_id="evt-2")])
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, (
            f"second ingest() took {elapsed_ms}ms — blocked on the lifecycle lock"
        )

        await mgr.close()


# ─── Eager lifecycle: create_session() (sync + async) ───────────

class TestEagerLifecycle:
    async def test_create_session_creates_full_lifecycle(self, mode: _Mode) -> None:
        """create_session() eagerly creates conversation + session + participants,
        uses session_id as both conv and session id in Beacon, and sets state."""
        mgr = mode.manager()

        result = await mode.run(mgr.create_session("ses-123"))

        assert mode.count(mode.client.create_conversation) == 1
        assert mode.count(mode.client.create_session) == 1
        assert mode.count(mode.client.add_participants) == 1
        assert result["sessionId"] == "ses-123"
        assert result["ownerId"] == "owner-1"
        assert result["agentId"] == "agent-1"
        call_args = mode.client.create_session.call_args[0]
        assert call_args[0] == "ses-123"  # conversation_id (Beacon)
        assert call_args[1] == "ses-123"  # session_id (Beacon, same)
        assert mgr.session_id == "ses-123"
        assert mgr._conversation_created is True
        assert mgr._session_created is True

    async def test_ingest_after_create_session_skips_lifecycle(self, mode: _Mode) -> None:
        """ingest() should skip lazy lifecycle if create_session() was called first."""
        mgr = mode.manager()

        await mode.run(mgr.create_session("ses-eager"))
        await mode.ingest_and_flush(mgr, [_make_record(session_id="ses-eager")])

        # Lifecycle calls happen only once (in create_session)
        assert mode.count(mode.client.create_conversation) == 1
        assert mode.count(mode.client.create_session) == 1
        assert mode.count(mode.client.add_participants) == 1
        # But entries are posted
        assert mode.count(mode.client.add_entries) == 1

    async def test_create_session_409_treated_as_success(self, mode: _Mode) -> None:
        """409 (returning user to the same session_id) from any lifecycle call is success."""
        from agentic_context.client import IngestionError

        for name in ("create_conversation", "create_session", "add_participants"):
            mode.fail_with(name, IngestionError(409, "exists"))
        mgr = mode.manager()

        result = await mode.run(mgr.create_session("ses-dup"))

        # Should not raise, and state should be set
        assert result["sessionId"] == "ses-dup"
        assert mgr._conversation_created is True
        assert mgr._session_created is True

    async def test_create_session_non_409_raises(self, mode: _Mode) -> None:
        """Non-409 errors from lifecycle calls should propagate."""
        from agentic_context.client import IngestionError

        mode.fail_with("create_conversation", IngestionError(500, "server error"))
        mgr = mode.manager()

        with pytest.raises(IngestionError) as exc_info:
            await mode.run(mgr.create_session("ses-fail"))
        assert exc_info.value.status_code == 500

    async def test_create_session_after_close_raises(self, mode: _Mode) -> None:
        """create_session() on a closed manager raises RuntimeError."""
        mgr = mode.manager()
        await mode.run(mgr.close())

        with pytest.raises(RuntimeError, match="already closed"):
            await mode.run(mgr.create_session("ses-x"))


# ─── owner_type parity: reads must use the manager's owner_type ────────


def test_sync_reads_forward_owner_type() -> None:
    retrieval = MagicMock()
    mgr = SessionManager(
        _mock_client(), agent_id="bot", owner_id="005x", session_id="ses-1",
        owner_type="employee", retrieval_client=retrieval,
    )

    mgr.get_context()
    mgr.get_session()
    mgr.list_sessions()

    for call in (retrieval.get_context, retrieval.get_conversation, retrieval.list_conversations):
        assert call.call_args.kwargs["owner_type"] == "employee"


async def test_async_reads_forward_owner_type() -> None:
    retrieval = MagicMock()
    retrieval.get_context = AsyncMock(return_value={})
    retrieval.get_conversation = AsyncMock(return_value={})
    retrieval.list_conversations = AsyncMock(return_value=[])
    mgr = AsyncSessionManager(
        _async_mock_client(), agent_id="bot", owner_id="005x", session_id="ses-1",
        owner_type="employee", retrieval_client=retrieval,
    )

    await mgr.get_context()
    await mgr.get_session()
    await mgr.list_sessions()

    for call in (retrieval.get_context, retrieval.get_conversation, retrieval.list_conversations):
        assert call.call_args.kwargs["owner_type"] == "employee"
