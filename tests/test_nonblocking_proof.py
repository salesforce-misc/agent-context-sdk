# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Non-blocking guarantee tests for the real AsyncSessionManager.

Drives the SHIPPED ``agentic_context.AsyncSessionManager`` against an
in-process slow / flaky fake of ``AsyncIngestionClient``.

Verifies:
  1. ``ingest()`` returns in <50 ms even when add_entries takes 1 s.
  2. Multiple ``ingest()`` calls don't serialize — they fan out.
  3. Retry-on-503 fires and ``on_retry``/``on_success`` callbacks invoked.
  4. ``close()`` drains all pending background deliveries before returning.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic_context import (
    AsyncIngestionClient,
    AsyncSessionManager,
    IngestionError,
    Record,
)
from agentic_context.session import _MAX_PENDING_DELIVERIES


# ─── Slow fake client ─────────────────────────────────────────────


class SlowFakeClient:
    """Quacks like AsyncIngestionClient. Each call sleeps ``latency`` seconds.

    ``fail_first_n`` controls how many initial ``add_entries`` calls return
    503 before succeeding (drives the retry path).
    """

    def __init__(self, *, fail_first_n: int = 0, latency: float = 0.05) -> None:
        self.fail_first_n = fail_first_n
        self.latency = latency
        self.add_entries_calls = 0
        self.successful_batches: list[list[dict[str, Any]]] = []
        self.created_conv = False
        self.created_sess = False
        self.closed_sess = False

    async def create_conversation(self, *args, **kwargs):
        await asyncio.sleep(self.latency)
        self.created_conv = True
        return {}

    async def create_session(self, *args, **kwargs):
        await asyncio.sleep(self.latency)
        self.created_sess = True
        return {}

    async def add_participants(self, *args, **kwargs):
        await asyncio.sleep(self.latency)
        return {}

    async def add_entries(self, conv_id, sess_id, entries, *, owner_id, agent_id):
        await asyncio.sleep(self.latency)
        self.add_entries_calls += 1
        if self.add_entries_calls <= self.fail_first_n:
            raise IngestionError(503, "slow fake: transient failure")
        self.successful_batches.append(entries)
        return {"accepted": len(entries)}

    async def close_session(self, *args, **kwargs):
        await asyncio.sleep(self.latency)
        self.closed_sess = True
        return {}


def _record(session_id: str, idx: int) -> Record:
    return Record(
        event_id=f"evt-{session_id}-{idx}",
        session_id=session_id,
        agent_id="bot",
        event_type="user_message",
        participant_id="user",
        content=f"hello {idx}",
        category="Message",
        interaction_id=f"turn-{session_id}-{idx}",
        timestamp=datetime.now(timezone.utc),
    )


# ─── Tests ────────────────────────────────────────────────────────


async def test_ingest_returns_in_under_50ms_even_when_beacon_is_slow():
    """ingest() must not stall the caller while Beacon is slow."""
    client = SlowFakeClient(latency=1.0)  # 1s per HTTP call
    mgr = AsyncSessionManager(client, agent_id="bot", owner_id="u", session_id="ses-fast")

    t0 = time.perf_counter()
    await mgr.ingest([_record("ses-fast", 0)])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.050, f"ingest blocked for {elapsed*1000:.1f} ms"

    await mgr.close()


async def test_close_drains_pending_deliveries():
    """close() must wait for in-flight background tasks before returning."""
    client = SlowFakeClient(latency=0.2)
    mgr = AsyncSessionManager(client, agent_id="bot", owner_id="u", session_id="ses-drain")

    for i in range(3):
        await mgr.ingest([_record("ses-drain", i)])

    await mgr.close()
    assert client.add_entries_calls == 3, "close() returned before all ingests completed"
    # close() is local-only — no close_session call to Beacon (unified model)


async def test_concurrent_ingests_run_in_parallel():
    """Three slow ingests should overlap, not serialize."""
    client = SlowFakeClient(latency=0.3)
    mgr = AsyncSessionManager(client, agent_id="bot", owner_id="u", session_id="ses-par")

    t0 = time.perf_counter()
    await asyncio.gather(*(mgr.ingest([_record("ses-par", i)]) for i in range(3)))
    elapsed = time.perf_counter() - t0
    # All three ingest() calls return effectively instantly.
    assert elapsed < 0.050, f"three concurrent ingests took {elapsed*1000:.1f} ms"

    await mgr.close()
    # Three deliveries plus one create_conv + one create_sess + one close_sess.
    # If they had serialized at the lifecycle layer, total wall time would
    # exceed 3 * 0.3 = 0.9s. asyncio.gather + lifecycle lock should overlap.
    assert client.add_entries_calls == 3


async def test_retry_path_fires_callbacks():
    """503 -> 503 -> success: on_retry fires twice, on_success once."""
    client = SlowFakeClient(fail_first_n=2, latency=0.01)

    events: dict[str, list[Any]] = {"retry": [], "error": [], "success": []}

    class _CB:
        def on_success(self, records, status_code):
            events["success"].append(status_code)

        def on_error(self, records, error, attempts):
            events["error"].append(attempts)

        def on_retry(self, records, error, attempt, delay):
            events["retry"].append((attempt, type(error).__name__))

    mgr = AsyncSessionManager(
        client,
        agent_id="bot",
        owner_id="u",
        session_id="ses-retry",
        callbacks=_CB(),
    )
    await mgr.ingest([_record("ses-retry", 0)])
    await mgr.close()

    assert client.add_entries_calls == 3, "expected 2 failures + 1 success"
    assert len(events["retry"]) == 2
    assert events["success"] == [202]
    assert events["error"] == []


# ── Bounded fan-out regressions ─────────────────────────────────────


async def test_ingest_drops_batch_when_pending_tasks_saturated():
    """When _pending_tasks is at the cap, a new ingest() must drop the batch
    (not block, not raise) and fire on_error with a RuntimeError.

    Protects against unbounded task fan-out under a Beacon slowdown.
    """
    # Client that hangs forever so every delivery task stays pending.
    hang = asyncio.Event()

    class HangingClient:
        async def create_conversation(self, *a, **k): return {}
        async def create_session(self, *a, **k): return {}
        async def add_participants(self, *a, **k): return {}
        async def close_session(self, *a, **k): return {}
        async def add_entries(self, *a, **k):
            await hang.wait()
            return {"accepted": 0}

    events: dict[str, list[Any]] = {"error": [], "success": []}

    class _CB:
        def on_success(self, records, status_code):
            events["success"].append(status_code)
        def on_error(self, records, error, attempts):
            events["error"].append((type(error).__name__, attempts, len(records)))
        def on_retry(self, records, error, attempt, delay):
            pass

    mgr = AsyncSessionManager(
        HangingClient(),
        agent_id="bot",
        owner_id="u",
        session_id="ses-sat",
        callbacks=_CB(),
    )
    try:
        # Fill the pending set to the cap.
        for i in range(_MAX_PENDING_DELIVERIES):
            await mgr.ingest([_record("ses-sat", i)])
        # Yield so create_task instances actually schedule.
        await asyncio.sleep(0)
        assert len(mgr._pending_tasks) == _MAX_PENDING_DELIVERIES

        # One more ingest — must be dropped.
        t0 = time.perf_counter()
        await mgr.ingest([_record("ses-sat", _MAX_PENDING_DELIVERIES)])
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.050, f"saturated ingest blocked for {elapsed*1000:.1f} ms"
        assert len(events["error"]) == 1, "on_error should fire exactly once"
        err_type, attempts, n_records = events["error"][0]
        assert err_type == "RuntimeError"
        assert attempts == 0, "dropped batch was never attempted"
        assert n_records == 1
        # Set size unchanged — dropped batch never entered the set.
        assert len(mgr._pending_tasks) == _MAX_PENDING_DELIVERIES
    finally:
        hang.set()
        # Cancel outstanding hanging tasks so close() doesn't wait 30s.
        for t in list(mgr._pending_tasks):
            t.cancel()
        await mgr.close()


async def test_non_retryable_error_stops_after_one_attempt():
    """400 must fire on_error exactly once and not retry."""
    client = AsyncMock(spec=AsyncIngestionClient)
    call_count = {"n": 0}

    async def fail400(*args, **kwargs):
        call_count["n"] += 1
        raise IngestionError(400, "bad request")

    client.add_entries = fail400
    client.create_conversation = AsyncMock(return_value={})
    client.create_session = AsyncMock(return_value={})
    client.close_session = AsyncMock(return_value={})

    events: dict[str, list[Any]] = {"retry": [], "error": []}

    class _CB:
        def on_success(self, records, status_code):  # noqa: D401
            pass

        def on_error(self, records, error, attempts):
            events["error"].append(attempts)

        def on_retry(self, records, error, attempt, delay):
            events["retry"].append(attempt)

    mgr = AsyncSessionManager(
        client, agent_id="bot", owner_id="u", session_id="ses-bad", callbacks=_CB()
    )
    await mgr.ingest([_record("ses-bad", 0)])
    await mgr.close()

    assert call_count["n"] == 1
    assert events["retry"] == []
    assert events["error"] == [1]
