# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""End-to-end lifecycle tests for the self-driving (path-3) adapter.

The contract under test:

- ``async with adapter:`` opens the owned ``AsyncIngestionClient`` /
  ``AsyncSessionManager`` / ``AsyncRetrievalClient``; exit closes them.
- ``stream()`` POSTs records via the four-endpoint Beacon lifecycle.
- ``stream()`` calls GET /agentic/context BEFORE driving the graph
  (subject to the ``context_policy``); injects the result as a
  ``SystemMessage(id="agentic-memory-context")`` prepended to
  ``inputs["messages"]``.
- ``inputs`` accepts a plain string (wrapped to ``[("human", inputs)]``).
- A 5xx from /agentic/context is logged but does not break the stream.
- Back-compat: a manager-injected adapter (``session_manager=``) is *not*
  self-driving — no auto-fetch, no injected context SystemMessage.
"""
from __future__ import annotations

import warnings
from typing import Annotated, Any, TypedDict
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentic_context import AsyncIngestionClient, AsyncSessionManager, Record
from agentic_context_langgraph import LangGraphAdapter

from .conftest import BASE_URL, CONTEXT_URL, mock_beacon_endpoints


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_recording_graph(captured: list[list[Any]]):
    """Graph whose node appends a snapshot of state['messages'] to ``captured``."""
    model = FakeListChatModel(responses=["hi there"])

    async def chat(state: _State) -> dict:
        captured.append(list(state["messages"]))
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile()


def _conn_kwargs(**overrides):
    base = {
        "agent_id": "bot",
        "owner_id": "user-1",
        "api_key": "tok",
        "base_url": BASE_URL,
        "dataspace": "default",
    }
    base.update(overrides)
    return base


@respx.mock
async def test_aenter_aexit_round_trip(respx_mock):
    """All four lifecycle routes plus GET /context fire; no warnings."""
    routes = mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "ok"})
    )

    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        async with adapter:
            async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
                pass

    assert routes["create_conv"].call_count == 1
    assert routes["create_sess"].call_count == 1
    assert routes["add_entries"].call_count >= 1
    # Unified model: close() is local-only, no close_session POST to Beacon.
    assert routes["close_sess"].call_count == 0
    assert get_route.call_count == 1
    # No surprises from the adapter's own resource management.
    assert [w for w in caught if "unclosed" in str(w.message).lower()] == []


@respx.mock
async def test_stream_emits_records_in_self_driving_mode(respx_mock):
    """FakeListChatModel(['hi there']) -> [user_message, agent_response]."""
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )
    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        records: list[Record] = []
        async for r in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            records.append(r)

    types = [r.event_type for r in records]
    assert "user_message" in types
    assert "agent_response" in types
    user = next(r for r in records if r.event_type == "user_message")
    assert user.content == "hi"
    response = next(r for r in records if r.event_type == "agent_response")
    assert response.content == "hi there"


@respx.mock
async def test_run_returns_collected_records(respx_mock):
    """``run()`` is the drain-and-collect convenience wrapper over stream()."""
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )
    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        records = await adapter.run("hi", session_id="ses-1")

    assert isinstance(records, list)
    types = [r.event_type for r in records]
    assert "user_message" in types
    assert "agent_response" in types


@respx.mock
async def test_stream_string_input_wraps_to_human_message(respx_mock):
    """``stream("hi", sid)`` is equivalent to passing the dict form."""
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )
    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        records: list[Record] = []
        async for r in adapter.stream("hi", "ses-1"):
            records.append(r)

    user = next(r for r in records if r.event_type == "user_message")
    assert user.content == "hi"


@respx.mock
async def test_stream_calls_get_context_on_first_call(respx_mock):
    """GET fires once before the graph runs; SystemMessage is prepended."""
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "summary": "Returning customer.",
                "userContext": [
                    {"text": "Owns Pro plan", "type": "preference", "confidence": 0.9}
                ],
            },
        )
    )

    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        post_call_count_before = 0  # placeholder; respx exposes call counts directly.
        async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass

    assert get_route.call_count == 1
    # By the time the chat node ran, a SystemMessage with the stable id
    # had been prepended to messages.
    assert captured, "graph node never executed"
    sysmsgs = [m for m in captured[0] if isinstance(m, SystemMessage)]
    ids = [getattr(m, "id", None) for m in sysmsgs]
    assert "agentic-memory-context" in ids
    only = next(m for m in sysmsgs if getattr(m, "id", None) == "agentic-memory-context")
    assert "Owns Pro plan" in only.content


@respx.mock
async def test_stream_skips_get_context_on_second_call_with_session_first_policy(respx_mock):
    """``session_first`` (default) -> exactly one GET across two turns."""
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "ok"})
    )
    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        async for _ in adapter.stream({"messages": [("human", "first")]}, "ses-1"):
            pass
        async for _ in adapter.stream({"messages": [("human", "second")]}, "ses-1"):
            pass

    assert get_route.call_count == 1


@respx.mock
async def test_session_first_shared_across_per_request_adapters(respx_mock):
    """Per-request adapters with the same (owner, agent, session) must share
    policy state — otherwise ``SessionFirstPolicy`` degrades to one GET per
    request, defeating the entire point of the policy."""
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "ok"})
    )
    captured_first: list[list[Any]] = []
    captured_second: list[list[Any]] = []

    # Two short-lived adapters mimic two HTTP requests on the same conversation.
    first = LangGraphAdapter(
        graph=_build_recording_graph(captured_first), **_conn_kwargs()
    )
    async with first:
        async for _ in first.stream({"messages": [("human", "turn 1")]}, "ses-1"):
            pass

    second = LangGraphAdapter(
        graph=_build_recording_graph(captured_second), **_conn_kwargs()
    )
    async with second:
        async for _ in second.stream({"messages": [("human", "turn 2")]}, "ses-1"):
            pass

    assert get_route.call_count == 1


@respx.mock
async def test_session_first_state_isolated_per_owner(respx_mock):
    """Same session_id under two different owner_ids must NOT share state —
    otherwise a tenant could starve another's first-turn fetch."""
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "ok"})
    )

    captured_a: list[list[Any]] = []
    adapter_a = LangGraphAdapter(
        graph=_build_recording_graph(captured_a),
        **_conn_kwargs(owner_id="tenant-a"),
    )
    async with adapter_a:
        async for _ in adapter_a.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass

    captured_b: list[list[Any]] = []
    adapter_b = LangGraphAdapter(
        graph=_build_recording_graph(captured_b),
        **_conn_kwargs(owner_id="tenant-b"),
    )
    async with adapter_b:
        async for _ in adapter_b.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass

    assert get_route.call_count == 2


@respx.mock
async def test_concurrent_sessions_have_independent_turn_counters(respx_mock):
    """Two interleaved sessions on one adapter must produce
    ``turn-{sid}-0`` for each session's first turn — not ``turn-A-0`` then
    ``turn-B-1`` from a shared instance counter."""
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )
    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        a_records: list[Record] = []
        async for r in adapter.stream("hi-A", "ses-A"):
            a_records.append(r)
        b_records: list[Record] = []
        async for r in adapter.stream("hi-B", "ses-B"):
            b_records.append(r)
        a2_records: list[Record] = []
        async for r in adapter.stream("hi-A2", "ses-A"):
            a2_records.append(r)

    # Each session's first turn is "turn-{sid}-0".
    assert a_records[0].interaction_id == "turn-ses-A-0"
    assert b_records[0].interaction_id == "turn-ses-B-0"
    # Second turn of ses-A advances ses-A's counter, not the global one.
    assert a2_records[0].interaction_id == "turn-ses-A-1"


async def test_stream_outside_async_context_raises():
    """Self-driving adapter must be used as ``async with`` — owned clients
    aren't open until __aenter__ runs, and silently calling httpx on a
    closed client surfaces as a confusing 401-shaped failure deep in the
    stream loop. Reject up front."""
    adapter = LangGraphAdapter(
        graph=_build_recording_graph([]), **_conn_kwargs()
    )
    with pytest.raises(RuntimeError, match="async with"):
        async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass


@respx.mock
async def test_end_session_drops_per_session_state(respx_mock):
    """``end_session(sid)`` evicts the per-session manager + counters so
    long-running adapters don't leak."""
    routes = mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )
    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1"):
            pass
        # Manager pool is keyed by (session_id, owner).
        assert ("ses-1", "user-1") in adapter._owned_managers
        await adapter.end_session("ses-1")
        assert ("ses-1", "user-1") not in adapter._owned_managers
        assert "ses-1" not in adapter._session_owner
        assert "ses-1" not in adapter._turn_idx
        assert "ses-1" not in adapter._context_state
        # Unified model: close() / end_session() are local-only, no
        # close_session POST to Beacon.
        assert routes["close_sess"].call_count == 0


@respx.mock
async def test_owned_manager_carries_framework_agent_label(respx_mock):
    """Beacon analytics bucket this adapter's agent as LangGraph, not Custom."""
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(return_value=httpx.Response(200, json={}))
    captured: list[list[Any]] = []
    async with LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs()) as adapter:
        async for _ in adapter.stream("hi", "ses-1"):
            pass
        mgr = adapter._owned_managers[("ses-1", "user-1")]
        assert mgr._agent_participant_variable["AiAgentType"] == "LangGraph"
        assert mgr._agent_participant_variable["AiAgentApiName"] == adapter.agent_id


@respx.mock
async def test_agent_participant_variable_full_override_wins(respx_mock):
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(return_value=httpx.Response(200, json={}))
    custom = {"AiAgentType": "MyRuntime", "AiAgentApiName": "override"}
    captured: list[list[Any]] = []
    async with LangGraphAdapter(
        graph=_build_recording_graph(captured), agent_participant_variable=custom, **_conn_kwargs(),
    ) as adapter:
        async for _ in adapter.stream("hi", "ses-1"):
            pass
        assert adapter._owned_managers[("ses-1", "user-1")]._agent_participant_variable == custom


async def test_agent_participant_variable_rejected_with_injected_manager():
    with pytest.raises(ValueError, match="agent_participant_variable"):
        LangGraphAdapter(
            agent_id="bot", graph=_build_recording_graph([]),
            session_manager=MagicMock(), agent_participant_variable={},
        )


@respx.mock
async def test_get_context_outage_does_not_break_stream(respx_mock):
    """A 500 from /context is logged + swallowed; the stream still POSTs records."""
    routes = mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )

    captured: list[list[Any]] = []
    adapter = LangGraphAdapter(graph=_build_recording_graph(captured), **_conn_kwargs())
    async with adapter:
        records: list[Record] = []
        async for r in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            records.append(r)

    # The stream still produced records and POSTed them.
    assert any(r.event_type == "user_message" for r in records)
    assert routes["add_entries"].call_count >= 1
    # And no SystemMessage was injected on outage — the node sees only the user turn.
    assert captured, "graph node never executed"
    sysmsgs_with_id = [
        m
        for m in captured[0]
        if isinstance(m, SystemMessage) and getattr(m, "id", None) == "agentic-memory-context"
    ]
    assert sysmsgs_with_id == []


@respx.mock
async def test_manager_injected_adapter_does_not_auto_fetch_or_inject_context(respx_mock):
    """Back-compat: the smart-adapter wiring must not regress customers who use
    ``session_manager=`` injection. The legacy adapter must not call
    GET /agentic/context (that's opt-in) and doesn't prepend a SystemMessage
    with the smart-adapter id."""
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "should not be seen"})
    )

    captured: list[list[Any]] = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = AsyncSessionManager(
            client, agent_id="bot", owner_id="user-1", session_id="ses-1"
        )
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_recording_graph(captured)
        )
        async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass

    assert get_route.call_count == 0
    assert captured, "graph node never executed"
    smart_id_msgs = [
        m
        for m in captured[0]
        if isinstance(m, SystemMessage) and getattr(m, "id", None) == "agentic-memory-context"
    ]
    assert smart_id_msgs == []
