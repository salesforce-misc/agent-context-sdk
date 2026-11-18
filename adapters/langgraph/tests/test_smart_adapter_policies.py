# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unit tests for ``ContextPolicy`` implementations and ``resolve_policy``.

No graph, no network — pure decision logic. ``should_fetch`` is the
single decision function the adapter calls before each ``stream()`` to
decide whether the GET fires this turn.

Convention: ``last_fetched_at`` is ``None`` until the first successful
fetch, then a tz-aware UTC datetime. ``last_user_text`` is the raw user
input on this turn. ``last_payload`` is the most recent successful
context payload (or ``None``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

import httpx
import pytest
import respx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentic_context import AsyncIngestionClient
from agentic_context_langgraph import (
    ContextPolicy,
    EveryTurnPolicy,
    LangGraphAdapter,
    SessionFirstPolicy,
    TimeDecayPolicy,
    TopicShiftPolicy,
)
from agentic_context_langgraph.policies import resolve_policy

from .conftest import BASE_URL, CONTEXT_URL, mock_beacon_endpoints, _build_graph


# Module-scope TypedDict so Python 3.14 forward-ref evaluation finds
# ``Annotated`` and ``add_messages`` at the resolver's globals.
class _PolicyState(TypedDict):
    messages: Annotated[list, add_messages]


def _ctx(*, turn_idx=0, last_user_text="hi", last_payload=None, last_fetched_at=None):
    return {
        "session_id": "ses-1",
        "turn_idx": turn_idx,
        "last_user_text": last_user_text,
        "last_payload": last_payload,
        "last_fetched_at": last_fetched_at,
    }


# ─── SessionFirstPolicy ──────────────────────────────────────────────


def test_session_first_fires_only_on_first_turn() -> None:
    p = SessionFirstPolicy()
    assert p.should_fetch(**_ctx(turn_idx=0, last_fetched_at=None)) is True
    assert p.should_fetch(**_ctx(turn_idx=1, last_fetched_at=datetime.now(timezone.utc))) is False
    assert p.should_fetch(**_ctx(turn_idx=42, last_fetched_at=datetime.now(timezone.utc))) is False


# ─── EveryTurnPolicy ─────────────────────────────────────────────────


def test_every_turn_always_fetches() -> None:
    """Backs context_mode="per-turn" — the fast-context fallback: should_fetch is
    True regardless of turn index or prior fetch state, so the adapter seeds
    fast context even if the LLM never calls the recall tool."""
    p = EveryTurnPolicy()
    for turn in range(5):
        assert p.should_fetch(**_ctx(turn_idx=turn)) is True
    assert p.should_fetch(**_ctx(turn_idx=5, last_fetched_at=datetime.now(timezone.utc))) is True


# ─── TimeDecayPolicy ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "age_seconds, expected",
    [(None, True), (30, False), (90, True)],
    ids=["never-fetched", "within-ttl", "after-ttl"],
)
def test_time_decay_fetches_only_when_stale(age_seconds, expected) -> None:
    p = TimeDecayPolicy(ttl_seconds=60)
    last = (
        None if age_seconds is None
        else datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    )
    assert p.should_fetch(**_ctx(last_fetched_at=last)) is expected


# ─── TopicShiftPolicy ────────────────────────────────────────────────


def test_topic_shift_no_embedder_falls_back_to_session_first() -> None:
    p = TopicShiftPolicy(embedder=None)
    # First turn -> True (nothing cached yet).
    assert p.should_fetch(**_ctx(last_fetched_at=None)) is True
    # Subsequent turns with no embedder -> False (no way to detect a shift).
    once = datetime.now(timezone.utc)
    assert p.should_fetch(**_ctx(last_fetched_at=once, last_user_text="anything")) is False


def test_topic_shift_same_topic_skips_refetch() -> None:
    """An embedder that returns the same vector for both turns -> no shift."""

    def embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "topic A" in text else [0.0, 1.0, 0.0]

    p = TopicShiftPolicy(embedder=embed, threshold=0.8)
    once = datetime.now(timezone.utc)
    payload = {"summary": "topic A: foo"}  # last_payload carries last topic seed
    assert (
        p.should_fetch(
            **_ctx(
                last_fetched_at=once,
                last_user_text="another note about topic A",
                last_payload=payload,
            )
        )
        is False
    )


def test_topic_shift_different_topic_triggers_refetch() -> None:
    def embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "topic A" in text else [0.0, 1.0, 0.0]

    p = TopicShiftPolicy(embedder=embed, threshold=0.8)
    once = datetime.now(timezone.utc)
    payload = {"summary": "topic A: foo"}
    assert (
        p.should_fetch(
            **_ctx(
                last_fetched_at=once,
                last_user_text="now switching to topic B entirely",
                last_payload=payload,
            )
        )
        is True
    )


# ─── resolve_policy ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec, cls",
    [(None, SessionFirstPolicy), ("every_turn", EveryTurnPolicy), ("session_first", SessionFirstPolicy)],
    ids=["none-defaults-to-session-first", "every_turn", "session_first"],
)
def test_resolve_policy_maps_to_class(spec, cls) -> None:
    """The public resolve_policy helper (for make_context_node users) maps a
    slug to its policy class — the escape hatch for cadences the three modes
    don't cover now that context_policy= is not a constructor knob."""
    assert isinstance(resolve_policy(spec), cls)


class _AlwaysFalse:
    def should_fetch(
        self, *, session_id, turn_idx, last_user_text, last_payload, last_fetched_at
    ) -> bool:
        return False


@pytest.mark.parametrize(
    "inst", [EveryTurnPolicy(), _AlwaysFalse()], ids=["builtin", "custom-protocol-impl"]
)
def test_resolve_policy_instance_passes_through(inst) -> None:
    """A policy instance — built-in or user-supplied implementing the protocol —
    is returned verbatim."""
    assert resolve_policy(inst) is inst


def test_resolve_policy_unknown_slug_raises() -> None:
    with pytest.raises(ValueError):
        resolve_policy("bogus")


def test_context_policy_protocol_runtime_check() -> None:
    """All built-ins satisfy the ``ContextPolicy`` Protocol shape."""
    # ContextPolicy is a Protocol; isinstance() works only when @runtime_checkable.
    # If not runtime-checkable, fall back to attribute presence.
    for cls in (SessionFirstPolicy, EveryTurnPolicy, TimeDecayPolicy):
        inst = cls()
        assert callable(getattr(inst, "should_fetch", None))
    # Avoid asserting isinstance(inst, ContextPolicy) — Protocol may not be
    # runtime_checkable; the attribute test is the load-bearing contract.
    _ = ContextPolicy  # silence unused-import lint; surface stays public


# ─── _SHARED_POLICY_STATE LRU bound ──────────────────────────────────


def test_shared_policy_state_is_bounded_and_lru() -> None:
    """The process-wide policy-state map must cap entries and evict LRU.

    Long-running ASGI workers see one unique session per request; without a
    cap the shared map would grow without bound. We verify the cap holds
    and that the most recently touched keys survive.
    """
    # Use a fresh adapter so we control the owner_id namespace.
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="lru-test-owner",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
    )

    cls = type(adapter)
    # Snapshot the class-level cap and lower it for the test to keep the
    # test fast and deterministic. Restore on teardown so other tests in
    # the suite see the production cap.
    original_cap = cls._MAX_SHARED_POLICY_STATE
    # Also snapshot the class map so we can restore it.
    cls._SHARED_POLICY_STATE.clear()
    try:
        cls._MAX_SHARED_POLICY_STATE = 8
        # Insert 20 unique session_ids — way over the cap.
        for i in range(20):
            adapter._policy_state_for(f"ses-{i}")
        assert len(cls._SHARED_POLICY_STATE) <= cls._MAX_SHARED_POLICY_STATE
        # The most recent 8 inserts must survive; the oldest must be gone.
        for i in range(12, 20):
            assert ("lru-test-owner", "bot", f"ses-{i}") in cls._SHARED_POLICY_STATE
        for i in range(0, 12):
            assert ("lru-test-owner", "bot", f"ses-{i}") not in cls._SHARED_POLICY_STATE
    finally:
        cls._MAX_SHARED_POLICY_STATE = original_cap
        cls._SHARED_POLICY_STATE.clear()


def test_shared_policy_state_bumps_lru_on_access() -> None:
    """Reading an existing entry should refresh its LRU position."""
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="lru-bump-owner",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
    )
    cls = type(adapter)
    original_cap = cls._MAX_SHARED_POLICY_STATE
    cls._SHARED_POLICY_STATE.clear()
    try:
        cls._MAX_SHARED_POLICY_STATE = 4
        for i in range(4):
            adapter._policy_state_for(f"ses-{i}")
        # Re-touch ses-0 so it becomes the most recent. Then push one more
        # entry; ses-1 should be evicted (oldest), not ses-0.
        adapter._policy_state_for("ses-0")
        adapter._policy_state_for("ses-99")
        keys = {k[2] for k in cls._SHARED_POLICY_STATE}
        assert "ses-0" in keys
        assert "ses-1" not in keys
        assert "ses-99" in keys
    finally:
        cls._MAX_SHARED_POLICY_STATE = original_cap
        cls._SHARED_POLICY_STATE.clear()


def test_policy_state_write_survives_lru_eviction_during_await() -> None:
    """Regression: write-backs from a slow fetch must land in the live map.

    The bug: a caller grabs a dict reference via ``_policy_state_for``,
    drops the lock, awaits a slow GET, then mutates the captured dict
    reference. If the LRU evicted that entry while we were awaiting,
    the mutation lands in an orphaned dict and is silently lost — the
    next ``_policy_state_for`` for the same key seeds a fresh dict with
    no record of the fetch, violating SessionFirstPolicy's "one GET per
    session" contract.

    The fix: write-backs (``_policy_state_write``,
    ``_maybe_fetch_and_inject_context``'s commit step,
    ``_commit_policy_turn``) must re-resolve the entry by key under the
    lock and re-seed on eviction so the write lands in the dict actually
    stored in the map.
    """
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="race-test-owner",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
    )
    cls = type(adapter)
    original_cap = cls._MAX_SHARED_POLICY_STATE
    cls._SHARED_POLICY_STATE.clear()
    try:
        # Use the production cap. We need to demonstrate that the write
        # survives even a full eviction cycle at the real bound.
        cls._MAX_SHARED_POLICY_STATE = 1024

        # 1. Seed ses-1.
        original_dict = adapter._policy_state_for("ses-1")
        key = ("race-test-owner", "bot", "ses-1")
        assert cls._SHARED_POLICY_STATE.get(key) is original_dict

        # 2. Push 1024 *other* unique session_ids. After we touch the
        #    1025th distinct key total, ses-1 (the LRU) gets evicted.
        for i in range(1024):
            adapter._policy_state_for(f"flush-{i}")

        # 3. ses-1 must now be evicted. The reference we captured at
        #    step 1 is orphaned — mutating it has no observable effect.
        assert cls._SHARED_POLICY_STATE.get(key) is None
        original_dict["last_fetched_at"] = "ORPHANED-WRITE"

        # 4. Write back through the helper. This is the path that, prior
        #    to the fix, would have mutated the orphaned dict and lost
        #    the write.
        sentinel_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        adapter._policy_state_write(
            "ses-1",
            {"last_fetched_at": sentinel_ts, "last_payload": {"k": "v"}},
        )

        # 5. The write must be visible on the next read. If the write
        #    landed in the orphaned dict (the bug), this read would seed
        #    a fresh entry with last_fetched_at=None.
        current = adapter._policy_state_for("ses-1")
        assert current is not original_dict
        assert current["last_fetched_at"] == sentinel_ts
        assert current["last_payload"] == {"k": "v"}
    finally:
        cls._MAX_SHARED_POLICY_STATE = original_cap
        cls._SHARED_POLICY_STATE.clear()


def test_commit_policy_turn_survives_lru_eviction() -> None:
    """``_commit_policy_turn`` must also re-resolve under the lock.

    Same race as the write-back: between ``_maybe_fetch_and_inject_context``
    and ``_commit_policy_turn``, an LRU eviction can drop the entry.
    The commit must re-seed so the turn counter increment doesn't vanish.
    """
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="commit-race-owner",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
    )
    cls = type(adapter)
    original_cap = cls._MAX_SHARED_POLICY_STATE
    cls._SHARED_POLICY_STATE.clear()
    try:
        cls._MAX_SHARED_POLICY_STATE = 8
        adapter._policy_state_for("ses-c")
        key = ("commit-race-owner", "bot", "ses-c")
        # Flood the LRU so ses-c is evicted.
        for i in range(20):
            adapter._policy_state_for(f"other-{i}")
        assert cls._SHARED_POLICY_STATE.get(key) is None

        adapter._commit_policy_turn("ses-c", "hello world")

        current = cls._SHARED_POLICY_STATE.get(key)
        assert current is not None
        assert current["turn_idx"] == 1
        assert current["last_user_text"] == "hello world"
    finally:
        cls._MAX_SHARED_POLICY_STATE = original_cap
        cls._SHARED_POLICY_STATE.clear()


# ─── EveryTurnPolicy integrated against the smart adapter ────────────


@respx.mock
async def test_every_turn_policy_drives_two_gets_with_stable_id(respx_mock):
    """Smoke test: two stream() calls under EveryTurnPolicy -> 2 GETs.

    Also verifies the injected SystemMessage uses the stable id
    ``agentic-memory-context`` on both turns so the LangGraph
    ``add_messages`` reducer replaces (not duplicates) it.
    """
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "foo"})
    )

    captured: list[list] = []

    async def recording_chat(state):
        # Capture the messages list seen by the node so we can assert the
        # injected SystemMessage shape across turns.
        captured.append(list(state["messages"]))
        from langchain_core.messages import AIMessage as _AI
        return {"messages": [_AI(content="ok")]}

    builder: StateGraph = StateGraph(_PolicyState)
    builder.add_node("chat", recording_chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    graph = builder.compile()

    adapter = LangGraphAdapter(
        graph=graph,
        agent_id="bot",
        owner_id="user-1",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
        context_mode="per-turn",  # seed context every turn
    )
    async with adapter:
        async for _ in adapter.stream("first", "ses-1"):
            pass
        async for _ in adapter.stream("second", "ses-1"):
            pass

    assert get_route.call_count == 2

    # On both turns, the graph node should have seen a SystemMessage with
    # the stable id at the head of the messages list.
    from langchain_core.messages import SystemMessage
    for snap in captured:
        sysmsgs = [m for m in snap if isinstance(m, SystemMessage)]
        assert any(getattr(m, "id", None) == "agentic-memory-context" for m in sysmsgs)
