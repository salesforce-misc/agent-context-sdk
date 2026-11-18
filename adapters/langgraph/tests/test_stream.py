# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Streaming-mode tests for ``LangGraphAdapter`` (v3 ``stream.messages``).

Streaming-mode coverage: per-turn interaction_id, streaming reasoning capture
via ``_has_pending_tool_calls`` classification over the ``stream.messages``
projection, tool capture via ``_ToolCaptureHandler``, one ingest batch per
turn, and partial-turn ingest on errors, early exit, close() and
end_session().
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx
from agentic_context import (
    AsyncIngestionClient,
    AsyncSessionManager,
    Record,
)
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from agentic_context_langgraph import LangGraphAdapter
from agentic_context_langgraph.adapter import (
    _TOOL_OUTPUT_MAX_CHARS,
    _ToolCaptureHandler,
    _tool_output_text,
)

# One ReAct tool turn, in order.
TOOL_TURN = ["user_message", "agent_reasoning", "tool_call", "tool_result", "agent_response"]

from .conftest import (
    BASE_URL,
    CONTEXT_URL,
    FAKE_RESPONSE,
    REACT_REASONING,
    _build_graph,
    _build_graph_with_tool,
    _build_react_graph,
    mock_beacon_endpoints,
)



def _make_async_mgr(client: AsyncIngestionClient) -> AsyncSessionManager:
    return AsyncSessionManager(
        client, agent_id="bot", owner_id="user-1", session_id="ses-1"
    )


@respx.mock
async def test_stream_simple_graph_two_records_per_turn(respx_mock):
    mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_graph()
        )
        records: list[Record] = []
        async for r in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            records.append(r)

    types = [r.event_type for r in records]
    assert "user_message" in types
    assert "agent_response" in types
    assert {r.interaction_id for r in records} == {"turn-ses-1-0"}
    user = next(r for r in records if r.event_type == "user_message")
    assert user.participant_id == "user-1"  # defaults to the session owner
    assert user.content == "hi"
    response = next(r for r in records if r.event_type == "agent_response")
    assert response.participant_id == "bot"
    assert response.content == FAKE_RESPONSE


def _posted_batches(routes: dict) -> list[list[str]]:
    """Entry types per ``/entries`` POST, in call order."""
    return [
        [e["type"] for e in json.loads(call.request.content).get("entries", [])]
        for call in routes["add_entries"].calls
    ]


@respx.mock
async def test_stream_captures_sync_node_tool_records_without_hooks(respx_mock):
    """Tool records from a sync node (worker thread) are captured with no hooks set."""
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_graph_with_tool()
        )
        records: list[Record] = []
        async for r in adapter.stream({"messages": [("human", "go")]}, "ses-1"):
            records.append(r)
        await adapter.close()

    yielded_types = [r.event_type for r in records]
    assert "user_message" in yielded_types
    assert "tool_call" in yielded_types
    assert "tool_result" in yielded_types
    assert all(r.interaction_id == "turn-ses-1-0" for r in records)
    assert _posted_batches(routes) == [yielded_types]


@respx.mock
async def test_stream_posts_one_ordered_batch_per_turn(respx_mock):
    """Each tool turn ingests as one chronological POST — N turns, N POSTs."""
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_react_graph()
        )
        turns: list[list[Record]] = []
        for text in ("where?", "and now?"):
            turn: list[Record] = []
            async for r in adapter.stream({"messages": [("human", text)]}, "ses-1"):
                turn.append(r)
            turns.append(turn)
        await adapter.close()

    assert [[r.event_type for r in t] for t in turns] == [TOOL_TURN, TOOL_TURN]
    assert _posted_batches(routes) == [TOOL_TURN, TOOL_TURN]
    assert [{r.interaction_id for r in t} for t in turns] == [
        {"turn-ses-1-0"},
        {"turn-ses-1-1"},
    ]
    first = turns[0]
    reasoning = next(r for r in first if r.event_type == "agent_reasoning")
    assert reasoning.variable == {"OutputValueText": REACT_REASONING}
    tool_call = next(r for r in first if r.event_type == "tool_call")
    assert tool_call.variable["Name"] == "lookup"
    assert json.loads(tool_call.variable["InputValueText"]) == {"query": "A-1"}
    tool_result = next(r for r in first if r.event_type == "tool_result")
    assert tool_result.variable["OutputValueText"] == "tool result: A-1"
    assert tool_result.metadata["tool_call_id"] == tool_call.metadata["tool_call_id"]


@respx.mock
async def test_stream_tool_error_still_emits_tool_result(respx_mock):
    """A tool that raises closes its tool_call with an error tool_result."""
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=mgr,
            graph=_build_react_graph(tool_raises=True),
        )
        turn = [r async for r in adapter.stream({"messages": [("human", "x")]}, "ses-1")]
        await adapter.close()

    assert [r.event_type for r in turn] == TOOL_TURN
    assert _posted_batches(routes) == [TOOL_TURN]
    result = next(r for r in turn if r.event_type == "tool_result")
    # Only the exception type leaves the process — not the message (here a
    # URL with a token).
    assert result.variable == {
        "Name": "lookup",
        "OutputValueText": "Tool raised RuntimeError",
    }
    assert result.metadata["error"] is True


@pytest.mark.parametrize(
    "output, expected",
    [
        (ToolMessage(content="r:A", tool_call_id="t2", name="lookup"), "r:A"),
        (
            Command(update={"messages": [ToolMessage(content="ok", tool_call_id="t")]}),
            "ok",
        ),
        ({"city": "Zürich"}, '{"city": "Zürich"}'),
        ([1, "a"], '[1, "a"]'),
        ({(1, 2): 3}, "{(1, 2): 3}"),
        (__import__("datetime").date(2026, 1, 1), "2026-01-01"),
    ],
    ids=["tool-message", "command", "dict", "list", "non-str-keys", "date"],
)
def test_tool_output_text(output, expected):
    assert _tool_output_text(output) == expected


def test_tool_output_text_is_bounded_and_never_raises():
    assert len(_tool_output_text("x" * 5000)) == _TOOL_OUTPUT_MAX_CHARS

    class Unprintable:
        def __str__(self):
            raise ValueError("no")

    assert _tool_output_text(Unprintable()) == "<unrenderable Unprintable>"


@respx.mock
async def test_tool_capture_handler_gets_no_non_tool_events(respx_mock, monkeypatch):
    """LangChain skips chain / LLM events for the handler; tool events still arrive."""
    mock_beacon_endpoints(respx_mock)
    seen: list[str] = []
    for name in ("on_chain_start", "on_chain_end", "on_llm_start", "on_chat_model_start",
                 "on_llm_end", "on_llm_new_token"):
        monkeypatch.setattr(
            _ToolCaptureHandler, name, lambda self, *a, _n=name, **k: seen.append(_n),
            raising=False,
        )
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=_make_async_mgr(client), graph=_build_react_graph()
        )
        turn = [r async for r in adapter.stream({"messages": [("human", "x")]}, "ses-1")]
        await adapter.close()

    assert seen == []
    assert [r.event_type for r in turn] == TOOL_TURN


@respx.mock
async def test_stream_tool_interrupt_is_not_a_tool_error(respx_mock):
    """interrupt() in a tool is HITL control flow: no error result, no on_error."""
    routes = mock_beacon_endpoints(respx_mock)
    errors: list = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client),
            graph=_build_react_graph(tool_interrupts=True),
            on_error=errors.append,
        )
        async for _ in adapter.stream({"messages": [("human", "x")]}, "ses-1"):
            pass
        async for _ in adapter.stream(Command(resume="yes"), "ses-1"):
            pass
        await adapter.close()

    assert errors == []
    batches = _posted_batches(routes)
    assert batches == [
        ["user_message", "agent_reasoning", "tool_call"],
        ["tool_call", "tool_result", "agent_response"],
    ]
    body = json.loads(routes["add_entries"].calls[1].request.content)
    result = next(e for e in body["entries"] if e["type"] == "tool_result")
    assert "A-1 (yes)" in json.dumps(result)


@respx.mock
async def test_stream_cancelled_tool_gets_error_result(respx_mock):
    """A tool cancelled mid-run gets a "did not finish" result, not a tool error.

    Newer langchain-core reports the CancelledError to on_tool_error (older
    versions skipped it); either way it must not count as the tool raising.
    """
    routes = mock_beacon_endpoints(respx_mock)
    errors: list = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client),
            graph=_build_react_graph(tool_delay=5),
            on_error=errors.append,
        )

        async def consume():
            async for _ in adapter.stream({"messages": [("human", "x")]}, "ses-1"):
                pass

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(consume(), 0.3)
        await adapter.close()

    assert _posted_batches(routes) == [
        ["user_message", "agent_reasoning", "tool_call", "tool_result"]
    ]
    body = json.loads(routes["add_entries"].calls[0].request.content)
    result = body["entries"][-1]
    assert "did not finish" in json.dumps(result)
    assert [e for e in errors if e.phase == "tool"] == []


@respx.mock
async def test_stream_graph_error_still_ingests_partial_turn(respx_mock):
    """A turn that raises after its tool ran still posts what it captured, once."""
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=mgr,
            graph=_build_react_graph(fail_after_tool=True),
        )
        with pytest.raises(RuntimeError, match="model boom"):
            async for _ in adapter.stream({"messages": [("human", "where?")]}, "ses-1"):
                pass
        await adapter.close()

    assert _posted_batches(routes) == [
        ["user_message", "agent_reasoning", "tool_call", "tool_result"]
    ]


@respx.mock
async def test_stream_partial_ingest_failure_does_not_mask_graph_error(respx_mock):
    """A failed partial-turn ingest goes to on_error; the graph error still raises."""
    mock_beacon_endpoints(respx_mock)
    errors: list = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=mgr,
            graph=_build_react_graph(fail_after_tool=True),
            on_error=errors.append,
        )
        await mgr.close()  # the turn's ingest will raise "Session already closed"
        with pytest.raises(RuntimeError, match="model boom"):
            async for _ in adapter.stream({"messages": [("human", "where?")]}, "ses-1"):
                pass

    assert [e.phase for e in errors] == ["stream", "ingest"]
    assert "closed" in str(errors[1].exception)


@respx.mock
async def test_stream_completed_turn_ingest_failure_raises(respx_mock):
    """With no other error in flight, a failed turn ingest propagates."""
    mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(agent_id="bot", session_manager=mgr, graph=_build_graph())
        await mgr.close()
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
                pass


@respx.mock
async def test_stream_records_after_close_are_reported(respx_mock):
    """close() mid-turn sends what exists; later records go to on_error, not silence."""
    routes = mock_beacon_endpoints(respx_mock)
    errors: list = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=_make_async_mgr(client),
            graph=_build_react_graph(),
            on_error=errors.append,
        )
        stream = adapter.stream({"messages": [("human", "x")]}, "ses-1")
        await stream.__anext__()
        await adapter.close()
        rest = [r.event_type async for r in stream]

    assert rest == TOOL_TURN[1:]
    assert _posted_batches(routes) == [["user_message"]]
    assert [e.phase for e in errors] == ["ingest"]
    assert "4 record(s)" in str(errors[0].exception)


@respx.mock
async def test_stream_early_exit_ingests_partial_turn(respx_mock):
    """Closing the stream early still posts the records yielded so far."""
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_graph()
        )
        stream = adapter.stream({"messages": [("human", "hi")]}, "ses-1")
        first = await stream.__anext__()
        await stream.aclose()
        await adapter.close()

    assert first.event_type == "user_message"
    assert _posted_batches(routes) == [["user_message"]]


@pytest.mark.parametrize("how", ["break", "raise"])
@respx.mock
async def test_stream_abandoned_turn_is_flushed_by_close(respx_mock, how):
    """break / an exception in the loop body, then close(): the partial turn lands."""
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_react_graph()
        )
        stream = adapter.stream({"messages": [("human", "where?")]}, "ses-1")
        try:
            async for _ in stream:
                if how == "break":
                    break
                raise KeyError("consumer bug")
        except KeyError:
            pass
        await adapter.close()
        # The generator finalizes later; its finally must not ingest again.
        await stream.aclose()

    assert _posted_batches(routes) == [["user_message"]]


def _self_driving(graph) -> LangGraphAdapter:
    return LangGraphAdapter(
        agent_id="bot",
        owner_id="user-1",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
        graph=graph,
    )


@respx.mock
async def test_self_driving_stream_batches_per_turn_and_flushes_on_exit(respx_mock):
    """Self-driving: one POST per finished turn; an abandoned turn lands on __aexit__."""
    routes = mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(return_value=httpx.Response(200, json={}))
    async with _self_driving(_build_react_graph()) as adapter:
        async for _ in adapter.stream("where?", "ses-1"):
            pass
        stream = adapter.stream("again?", "ses-1")
        async for _ in stream:
            break
    await stream.aclose()

    assert _posted_batches(routes) == [TOOL_TURN, ["user_message"]]


@respx.mock
async def test_end_session_flushes_abandoned_turn_without_reviving_manager(respx_mock):
    routes = mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(return_value=httpx.Response(200, json={}))
    async with _self_driving(_build_react_graph()) as adapter:
        stream = adapter.stream("where?", "ses-1")
        async for _ in stream:
            break
        await adapter.end_session("ses-1")
        await stream.aclose()
        assert adapter._owned_managers == {}

    assert _posted_batches(routes) == [["user_message"]]


@respx.mock
async def test_end_session_only_flushes_its_own_session(respx_mock):
    routes = mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(return_value=httpx.Response(200, json={}))
    async with _self_driving(_build_react_graph()) as adapter:
        streams = {}
        for sid in ("ses-a", "ses-b"):
            streams[sid] = adapter.stream(f"hi {sid}", sid)
            await streams[sid].__anext__()
        await adapter.end_session("ses-a")
        assert len(routes["add_entries"].calls) == 1
        assert "/conversations/ses-a/" in str(routes["add_entries"].calls[0].request.url)
    # ses-b's abandoned turn lands on __aexit__.
    for stream in streams.values():
        await stream.aclose()

    urls = [str(c.request.url) for c in routes["add_entries"].calls]
    assert len(urls) == 2
    assert "/conversations/ses-b/" in urls[1]


@respx.mock
async def test_stream_multi_turn_distinct_interaction_ids(respx_mock):
    mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=mgr,
            graph=_build_graph(responses=["a", "b"]),
        )
        first: list[Record] = []
        async for r in adapter.stream({"messages": [("human", "1st")]}, "ses-1"):
            first.append(r)
        second: list[Record] = []
        async for r in adapter.stream({"messages": [("human", "2nd")]}, "ses-1"):
            second.append(r)

    assert {r.interaction_id for r in first} == {"turn-ses-1-0"}
    assert {r.interaction_id for r in second} == {"turn-ses-1-1"}


@respx.mock
async def test_stream_tool_error_fires_on_error_phase_tool(respx_mock):
    mock_beacon_endpoints(respx_mock)
    errors: list = []
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot",
            session_manager=mgr,
            graph=_build_graph_with_tool(raise_in_tool=True),
            on_error=errors.append,
        )
        async for _ in adapter.stream({"messages": [("human", "go")]}, "ses-1"):
            pass

    tool_phase = [e for e in errors if e.phase == "tool"]
    assert len(tool_phase) == 1


@respx.mock
async def test_stream_end_to_end_posts_lifecycle_in_order(respx_mock):
    routes = mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_graph()
        )
        async for _ in adapter.stream({"messages": [("human", "hi")]}, "ses-1"):
            pass
        await adapter.close()

    assert routes["create_conv"].call_count == 1
    assert routes["create_sess"].call_count == 1
    assert routes["add_participants"].call_count == 1
    assert routes["add_entries"].call_count >= 1
    # Unified model: close() is local-only, no close_session POST to Beacon.
    assert routes["close_sess"].call_count == 0


@pytest.mark.parametrize(
    "message",
    [("human", "hello there"), HumanMessage(content="hello there", id="hu-1")],
    ids=["tuple", "HumanMessage"],
)
@respx.mock
async def test_stream_user_input_extraction(respx_mock, message):
    mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_graph()
        )
        records: list[Record] = []
        async for r in adapter.stream({"messages": [message]}, "ses-1"):
            records.append(r)
    user = next(r for r in records if r.event_type == "user_message")
    assert user.content == "hello there"


@respx.mock
async def test_stream_no_user_message_when_inputs_have_none(respx_mock):
    mock_beacon_endpoints(respx_mock)
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as client:
        mgr = _make_async_mgr(client)
        adapter = LangGraphAdapter(
            agent_id="bot", session_manager=mgr, graph=_build_graph()
        )
        records: list[Record] = []
        async for r in adapter.stream(
            {"messages": [SystemMessage(content="you are a bot")]},
            "ses-1",
        ):
            records.append(r)
    types = [r.event_type for r in records]
    assert "user_message" not in types
