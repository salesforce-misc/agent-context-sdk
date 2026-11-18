# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Snapshot-mode tests for ``LangGraphAdapter`` against master's core SDK.

Snapshot-mode dispatch coverage: HumanMessage → user_message,
AIMessage with content + tool_calls → agent_reasoning + tool_call, AIMessage
content-only → agent_response, AIMessage tool_calls-only → tool_call only,
ToolMessage → tool_result. All snapshot records share a single interaction_id.
"""
from __future__ import annotations

import json
import pickle  # nosec — round-trips test-owned StateSnapshot in tmp_path only
import re
from typing import Annotated, TypedDict
from unittest.mock import MagicMock

import pytest
import responses
from agentic_context import (
    AsyncSessionManager,
    IngestionClient,
    ContextSource,
    SessionManager,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from agentic_context_langgraph import LangGraphAdapter

from .conftest import BASE_URL, _build_graph_with_tool


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _stub_graph_with_messages(messages: list[BaseMessage]):
    """Build a tiny checkpointed graph and seed it with the given messages."""
    def node(state: _State, config):
        return {}

    g = StateGraph(_State)
    g.add_node("node", node)
    g.set_entry_point("node")
    g.set_finish_point("node")
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "ses-1"}}
    compiled.invoke({"messages": messages}, cfg)
    return compiled


def _new_async_mgr_mock():
    """A MagicMock that quacks like AsyncSessionManager for isinstance checks."""
    return MagicMock(spec=AsyncSessionManager)


def _new_sync_mgr_mock():
    return MagicMock(spec=SessionManager)


# ─── Per-message-type dispatch ────────────────────────────────────


@pytest.mark.parametrize(
    "message, event_type, expected, metadata",
    [
        (
            HumanMessage(content="hi", id="hu-1"),
            "user_message",
            {"category": "Message", "content": "hi", "participant_id": "user"},
            {},
        ),
        (
            AIMessage(content="hi back", id="ai-1"),
            "agent_response",
            {"category": "Message", "content": "hi back", "participant_id": "bot"},
            {},
        ),
        (
            ToolMessage(content="result data", tool_call_id="tc-1", name="lookup"),
            "tool_result",
            {
                "category": "Trace",
                "content": None,
                "variable": {"Name": "lookup", "OutputValueText": "result data"},
            },
            {"tool_call_id": "tc-1"},
        ),
    ],
    ids=["human-message", "ai-content-only", "tool-message"],
)
def test_snapshot_single_message_emits_one_record(message, event_type, expected, metadata):
    g = _stub_graph_with_messages([message])
    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
    )
    records = adapter.snapshot("ses-1")
    matching = [r for r in records if r.event_type == event_type]
    assert len(matching) == 1
    r = matching[0]
    for attr, value in expected.items():
        assert getattr(r, attr) == value, attr
    for key, value in metadata.items():
        assert r.metadata.get(key) == value, key


def test_snapshot_ai_with_content_and_tool_calls_emits_reasoning_then_tool_call():
    g = _stub_graph_with_messages([
        AIMessage(
            content="I'll look this up",
            tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "tc-1"}],
            id="ai-1",
        )
    ])
    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
    )
    records = adapter.snapshot("ses-1")
    types = [r.event_type for r in records]
    assert "agent_reasoning" in types
    assert "tool_call" in types
    reasoning = next(r for r in records if r.event_type == "agent_reasoning")
    tool_call = next(r for r in records if r.event_type == "tool_call")
    assert reasoning.category == "Trace"
    assert reasoning.content is None
    assert reasoning.variable == {"OutputValueText": "I'll look this up"}
    assert tool_call.category == "Trace"
    assert tool_call.variable["Name"] == "lookup"
    assert json.loads(tool_call.variable["InputValueText"]) == {"q": "x"}
    assert tool_call.metadata.get("tool_call_id") == "tc-1"
    # Order: reasoning before tool_call in the same AIMessage
    assert types.index("agent_reasoning") < types.index("tool_call")


def test_snapshot_ai_empty_content_with_tool_calls_no_reasoning():
    g = _stub_graph_with_messages([
        AIMessage(
            content="",
            tool_calls=[{"name": "lookup", "args": {}, "id": "tc-1"}],
            id="ai-1",
        )
    ])
    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
    )
    records = adapter.snapshot("ses-1")
    types = [r.event_type for r in records]
    assert "agent_reasoning" not in types
    assert "tool_call" in types


def test_snapshot_full_turn_five_records():
    """Tool graph yields user + reasoning + tool_call + tool_result + response."""
    g = _build_graph_with_tool(tool_output="ok")
    cfg = {"configurable": {"thread_id": "ses-1"}}
    g.invoke({"messages": [HumanMessage(content="go", id="hu-1")]}, cfg)

    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
    )
    records = adapter.snapshot("ses-1")
    types = [r.event_type for r in records]
    for required in ["user_message", "agent_reasoning", "tool_call", "tool_result", "agent_response"]:
        assert required in types, f"missing {required} in {types}"

    interaction_ids = {r.interaction_id for r in records}
    assert interaction_ids == {"turn-ses-1-snapshot"}


def test_snapshot_participant_id_multi_agent_contract():
    """All agent-side records carry ``agent_id``; user records carry the owner.

    The mocked manager exposes no ``owner_id``, so user records fall back to ``"user"``.

    Regression guard for the multi-agent wiring: customers deploy one
    ``LangGraphAdapter`` per sub-agent so Beacon can distinguish them by
    ``participantId``. Mirrors the ADK adapter's convention. The user-side
    invariant is locked in the same test so a future refactor can't silently
    cross the streams.
    """
    g = _build_graph_with_tool(tool_output="ok")
    cfg = {"configurable": {"thread_id": "ses-1"}}
    g.invoke({"messages": [HumanMessage(content="go", id="hu-1")]}, cfg)

    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
    )
    records = adapter.snapshot("ses-1")

    # Sanity: the snapshot produced every agent-side record type plus the user message.
    types = {r.event_type for r in records}
    for required in (
        "user_message",
        "agent_reasoning",
        "tool_call",
        "tool_result",
        "agent_response",
    ):
        assert required in types, f"missing {required} in {types}"

    for r in records:
        if r.event_type == "user_message":
            assert r.participant_id == "user", (
                f"user_message participant_id={r.participant_id!r}, expected 'user'"
            )
        else:
            assert r.participant_id == "bot", (
                f"{r.event_type} participant_id={r.participant_id!r}, "
                "expected 'bot' (derived from agent_id='bot')"
            )


def test_snapshot_user_record_uses_injected_manager_owner():
    """With a real injected SessionManager, user records carry the manager's
    ``owner_id`` so Beacon marks them role="owner" in that owner's conversation."""
    g = _stub_graph_with_messages([HumanMessage(content="hi", id="hu-1")])
    mgr = SessionManager(MagicMock(), agent_id="bot", owner_id="mgr-owner", session_id="ses-1")
    adapter = LangGraphAdapter(agent_id="bot", session_manager=mgr, graph=g)

    (user,) = [r for r in adapter.snapshot("ses-1") if r.event_type == "user_message"]
    assert user.participant_id == "mgr-owner"


def test_snapshot_pickle_path_matches_graph_path(tmp_path):
    g = _build_graph_with_tool(tool_output="ok")
    cfg = {"configurable": {"thread_id": "ses-1"}}
    g.invoke({"messages": [HumanMessage(content="go", id="hu-1")]}, cfg)

    state = g.get_state(cfg)
    pkl = tmp_path / "ses-1.pkl"
    with pkl.open("wb") as f:
        pickle.dump(state, f)

    graph_adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
    )
    pickle_adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        pickle_path=pkl,
        allow_pickle=True,
    )
    graph_records = [(r.event_type, r.content) for r in graph_adapter.snapshot("ses-1")]
    pickle_records = [(r.event_type, r.content) for r in pickle_adapter.snapshot("ses-1")]
    assert graph_records == pickle_records
    assert ("user_message", "go") in pickle_records


def test_snapshot_pickle_without_allow_pickle_raises(tmp_path):
    """allow_pickle is enforced lazily: construction succeeds, but snapshot()
    refuses to load the pickle until the risk is acknowledged."""
    g = _stub_graph_with_messages([HumanMessage(content="hi", id="hu-1")])
    pkl = tmp_path / "ses-1.pkl"
    with pkl.open("wb") as f:
        pickle.dump(g.get_state({"configurable": {"thread_id": "ses-1"}}), f)
    adapter = LangGraphAdapter(
        agent_id="bot", session_manager=_new_sync_mgr_mock(), pickle_path=pkl
    )
    with pytest.raises(ValueError, match="allow_pickle=True"):
        adapter.snapshot("ses-1")


def test_snapshot_constructor_xor_enforced(tmp_path):
    pkl = tmp_path / "anything.pkl"
    g = _stub_graph_with_messages([HumanMessage(content="hi", id="hu-1")])
    mgr = _new_sync_mgr_mock()
    with pytest.raises(ValueError):
        LangGraphAdapter(agent_id="bot", session_manager=mgr, graph=g, pickle_path=pkl)
    with pytest.raises(ValueError):
        LangGraphAdapter(agent_id="bot", session_manager=mgr)


async def test_snapshot_pickle_mode_rejects_stream_call(tmp_path):
    g = _stub_graph_with_messages([HumanMessage(content="hi", id="hu-1")])
    cfg = {"configurable": {"thread_id": "ses-1"}}
    state = g.get_state(cfg)
    pkl = tmp_path / "ses-1.pkl"
    with pkl.open("wb") as f:
        pickle.dump(state, f)
    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_async_mgr_mock(),
        pickle_path=pkl,
    )
    with pytest.raises(ValueError):
        async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass


def test_snapshot_mode_property(tmp_path):
    g = _stub_graph_with_messages([HumanMessage(content="hi", id="hu-1")])
    cfg = {"configurable": {"thread_id": "ses-1"}}
    state = g.get_state(cfg)
    pkl = tmp_path / "ses-1.pkl"
    with pkl.open("wb") as f:
        pickle.dump(state, f)
    stream_adapter = LangGraphAdapter(
        agent_id="bot", session_manager=_new_sync_mgr_mock(), graph=g
    )
    snap_adapter = LangGraphAdapter(
        agent_id="bot", session_manager=_new_sync_mgr_mock(), pickle_path=pkl
    )
    assert stream_adapter.mode == "stream"
    assert snap_adapter.mode == "snapshot"


def test_snapshot_reasoning_classifier_override_makes_text_a_reply():
    g = _stub_graph_with_messages([
        AIMessage(
            content="here you go",
            tool_calls=[{"name": "lookup", "args": {}, "id": "tc-1"}],
            id="ai-1",
        )
    ])
    adapter = LangGraphAdapter(
        agent_id="bot",
        session_manager=_new_sync_mgr_mock(),
        graph=g,
        reasoning_classifier=lambda msg: "reply",
    )
    records = adapter.snapshot("ses-1")
    types = [r.event_type for r in records]
    assert "agent_response" in types
    assert "agent_reasoning" not in types
    assert "tool_call" in types


def test_snapshot_satisfies_contextsource_protocol():
    g = _stub_graph_with_messages([HumanMessage(content="hi", id="hu-1")])
    adapter = LangGraphAdapter(
        agent_id="bot", session_manager=_new_sync_mgr_mock(), graph=g
    )
    assert isinstance(adapter, ContextSource)


# ─── End-to-end ───────────────────────────────────────────────────


@responses.activate
def test_snapshot_end_to_end_posts_lifecycle_endpoints():
    """Sync IngestionClient + SessionManager: snapshot → ingest → close hits the lifecycle endpoints (no close_session under unified model)."""
    conv = responses.add(responses.POST, re.compile(rf"{BASE_URL}/api/v1/agentic/conversations$"), json={})
    sess = responses.add(
        responses.POST,
        re.compile(rf"{BASE_URL}/api/v1/agentic/conversations/[^/]+/sessions$"),
        json={},
    )
    parts = responses.add(
        responses.POST,
        re.compile(rf"{BASE_URL}/api/v1/agentic/conversations/[^/]+/sessions/[^/]+/participants$"),
        json={},
    )
    entries = responses.add(
        responses.POST,
        re.compile(rf"{BASE_URL}/api/v1/agentic/conversations/[^/]+/entries$"),
        json={"accepted": 1, "rejected": 0, "errors": []},
    )
    close = responses.add(
        responses.POST,
        re.compile(rf"{BASE_URL}/api/v1/agentic/conversations/[^/]+/sessions/[^/]+/close$"),
        json={},
    )

    g = _build_graph_with_tool(tool_output="ok")
    cfg = {"configurable": {"thread_id": "ses-1"}}
    g.invoke({"messages": [HumanMessage(content="go", id="hu-1")]}, cfg)

    with IngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = SessionManager(
            client, agent_id="bot", owner_id="user-1", session_id="ses-1"
        )
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=g
        )
        records = adapter.snapshot("ses-1")
        adapter.ingest(records)
        adapter.close()

    assert conv.call_count == 1
    assert sess.call_count == 1
    assert parts.call_count == 1
    assert entries.call_count == 1
    # Unified model: close() is local-only, no close_session POST to Beacon.
    assert close.call_count == 0

    entries_call = next(
        c for c in responses.calls if c.request.url.endswith("/entries")
    )
    body = json.loads(entries_call.request.body)
    assert "entries" in body
    posted_entries = body["entries"]
    assert all("category" in e for e in posted_entries)
    assert all("type" in e for e in posted_entries)
    interaction_ids = {e.get("interactionId") for e in posted_entries}
    assert interaction_ids == {"turn-ses-1-snapshot"}


# ─── Regression: event_id stability across re-snapshots ──────────


def test_tool_call_event_id_does_not_use_framework_tc_id():
    """tool_call event_id must be derived from (session, idx, name), not tc['id'].

    LangGraph's tool_call dicts can carry differing 'id' values across
    re-snapshots (server-generated, retry-generated, etc.). Using tc['id']
    as the event_id would give the same logical call different ids across
    reads. The adapter pins event_id to (session, idx, tool_name)
    so re-snapshots round-trip cleanly.
    """
    g = _stub_graph_with_messages([
        AIMessage(
            content="checking",
            tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "call_xyz"}],
            id="ai-1",
        )
    ])
    a = LangGraphAdapter(agent_id="bot", session_manager=_new_sync_mgr_mock(), graph=g)
    rec = next(r for r in a.snapshot("ses-1") if r.event_type == "tool_call")
    # Even though tc['id']="call_xyz", the event_id must NOT be that value.
    assert rec.event_id != "call_xyz"
    # And the framework id is preserved in metadata for correlation.
    assert rec.metadata.get("tool_call_id") == "call_xyz"


@pytest.mark.parametrize(
    "content, expected",
    [
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
         '[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]'),
        ("x" * 5000, "x" * 2000),
    ],
    ids=["list-content-json", "capped-at-2000"],
)
def test_snapshot_tool_result_output_text_matches_streaming(content, expected):
    """Snapshot tool_result text uses the same rendering + cap as streaming."""
    g = _stub_graph_with_messages(
        [ToolMessage(content=content, tool_call_id="tc-1", name="lookup")]
    )
    adapter = LangGraphAdapter(
        agent_id="bot", session_manager=_new_sync_mgr_mock(), graph=g
    )
    (record,) = [r for r in adapter.snapshot("ses-1") if r.event_type == "tool_result"]
    assert record.variable["OutputValueText"] == expected
