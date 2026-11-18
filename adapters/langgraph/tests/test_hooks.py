# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Hook lifecycle tests for ``LangGraphAdapter`` against the new core API.

All six hooks fire at expected moments. ``on_session_end`` fires inside
``close()`` (the removed ``session_ended`` method is gone).
A hook that raises does not abort the stream — ``_safe_call`` swallows.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

import pytest
import respx
from agentic_context import AsyncIngestionClient, AsyncSessionManager
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentic_context_langgraph import (
    ErrorEvent,
    LangGraphAdapter,
    ToolEvent,
    TurnEvent,
)

from .conftest import (
    BASE_URL,
    _build_graph,
    _build_graph_with_tool,
    mock_beacon_endpoints,
)



class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _make_async_mgr(client, *, session_id: str = "ses-1"):
    return AsyncSessionManager(
        client, agent_id="bot", owner_id="user-1", session_id=session_id
    )


@respx.mock
async def test_session_and_turn_start_hooks_fire_once_per_stream_call(respx_mock):
    """on_session_start and on_turn_start are per-``stream()`` call (not
    once-per-session): two calls on one session fire each hook twice."""
    mock_beacon_endpoints(respx_mock)
    sessions_seen: list[str] = []
    starts: list[TurnEvent] = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client, session_id="ses-x"),
            graph=_build_graph(responses=["t1", "t2"]),
            on_session_start=sessions_seen.append,
            on_turn_start=starts.append,
        )
        async for _ in adapter.stream({"messages": [("human", "first")]}, "ses-x"):
            pass
        assert sessions_seen == ["ses-x"]
        async for _ in adapter.stream({"messages": [("human", "second")]}, "ses-x"):
            pass
    assert sessions_seen == ["ses-x", "ses-x"]
    assert len(starts) == 2
    assert all(s.session_id == "ses-x" for s in starts)
    assert starts[0].turn_started_at <= starts[1].turn_started_at


@respx.mock
async def test_on_tool_start_and_end_fire_with_event_payloads(respx_mock):
    mock_beacon_endpoints(respx_mock)
    starts: list[ToolEvent] = []
    ends: list[ToolEvent] = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client, session_id="ses-tool"),
            graph=_build_graph_with_tool(tool_output="ok"),
            on_tool_start=starts.append,
            on_tool_end=ends.append,
        )
        async for _ in adapter.stream({"messages": [("human", "go")]}, "ses-tool"):
            pass

    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].tool_name == "lookup"
    assert starts[0].args == {"query": "anything"}
    assert starts[0].output is None
    assert starts[0].ended_at is None
    assert ends[0].tool_call_id == starts[0].tool_call_id
    assert ends[0].started_at == starts[0].started_at
    assert "ok" in str(ends[0].output)
    assert ends[0].ended_at is not None


@respx.mock
async def test_on_error_phase_stream_when_node_raises(respx_mock):
    mock_beacon_endpoints(respx_mock)

    async def boom_node(state: _State) -> dict:
        raise RuntimeError("graph boom")

    builder: StateGraph = StateGraph(_State)
    builder.add_node("boom", boom_node)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    graph = builder.compile()

    errors: list[ErrorEvent] = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client, session_id="ses-err"),
            graph=graph,
            on_error=errors.append,
        )
        with pytest.raises(Exception):  # noqa: B017
            async for _ in adapter.stream({"messages": [("human", "go")]}, "ses-err"):
                pass

    assert any(e.phase == "stream" for e in errors)


@respx.mock
async def test_hook_that_raises_does_not_abort_stream(respx_mock):
    mock_beacon_endpoints(respx_mock)

    def bad_hook(_session_id: str) -> None:
        raise RuntimeError("hook boom")

    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client),
            graph=_build_graph(),
            on_session_start=bad_hook,
        )
        records = []
        async for r in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            records.append(r)
    # Stream completed normally despite the bad hook.
    assert any(r.event_type == "agent_response" for r in records)


@respx.mock
async def test_close_fires_on_session_end_hook(respx_mock):
    mock_beacon_endpoints(respx_mock)
    ended: list[str] = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client),
            graph=_build_graph(),
            on_session_end=ended.append,
        )
        async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass
        await adapter.close()
    assert ended == ["ses-1"]
