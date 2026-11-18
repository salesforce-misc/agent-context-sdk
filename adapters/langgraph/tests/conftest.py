# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Shared fixtures and graph builders for LangGraph adapter tests.

Targets master's new core SDK: ``Record`` / ``SessionManager`` /
``AsyncSessionManager`` / four-endpoint Beacon lifecycle.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

import httpx
import pytest
from agentic_context import (
    AsyncIngestionClient,
    AsyncSessionManager,
    IngestionClient,
    SessionManager,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

BASE_URL = "https://beacon.example.com"
CONTEXT_URL = f"{BASE_URL}/api/v1/agentic/context"
FAKE_RESPONSE = "hello from fake"
REACT_REASONING = "Let me look that up."


@pytest.fixture(autouse=True)
def _reset_shared_policy_state():
    """Clear the process-shared policy cache between tests.

    ``LangGraphAdapter`` shares ``last_fetched_at`` / ``turn_idx`` across
    instances keyed by ``(owner_id, agent_id, session_id)`` so that
    per-request adapters don't degrade ``SessionFirstPolicy`` to a
    fetch-every-turn loop. The tests reuse identifiers, so the cache must
    be cleared between cases.
    """
    from agentic_context_langgraph.adapter import LangGraphAdapter
    LangGraphAdapter._SHARED_POLICY_STATE.clear()
    yield
    LangGraphAdapter._SHARED_POLICY_STATE.clear()


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph(responses: list[str] | None = None, *, checkpointer: bool = False) -> Any:
    """Tiny single-node graph driving a ``FakeListChatModel``."""
    model = FakeListChatModel(responses=responses or [FAKE_RESPONSE])

    async def chat_node(state: _State) -> dict:
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=InMemorySaver()) if checkpointer else builder.compile()


def _build_capturing_graph(captured: list, responses: list[str] | None = None) -> Any:
    """Single-node graph that records the messages it sees before replying.

    ``captured`` is appended to (one entry per node invocation) so tests can
    assert what the adapter injected ahead of the user turn (e.g. the
    recall-tool instruction + context SystemMessages).
    """
    model = FakeListChatModel(responses=responses or [FAKE_RESPONSE])

    async def chat_node(state: _State) -> dict:
        captured.append(list(state["messages"]))
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile()


def _build_graph_with_tool(
    tool_output: str = "tool result",
    *,
    raise_in_tool: bool = False,
    checkpointer: bool = True,
) -> Any:
    """Graph with a ``@tool``-decorated function so LangChain tool callbacks fire.

    The node emits an AIMessage with non-empty content + tool_calls (so snapshot
    mode produces an agent_reasoning record), invokes the @tool to fire callbacks,
    then emits a ToolMessage and a final AIMessage reply.
    """
    from langchain_core.tools import tool

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        if raise_in_tool:
            raise RuntimeError("tool boom")
        return f"{tool_output}: {query}"

    def call_tool_node(state: _State) -> dict:
        try:
            result = lookup.invoke({"query": "anything"})
        except Exception:
            result = "ERROR"
        return {
            "messages": [
                AIMessage(
                    content="I will look this up",
                    tool_calls=[
                        {"name": "lookup", "args": {"query": "anything"}, "id": "tc-1"}
                    ],
                ),
                ToolMessage(content=result, tool_call_id="tc-1", name="lookup"),
                AIMessage(content=FAKE_RESPONSE),
            ]
        }

    builder: StateGraph = StateGraph(_State)
    builder.add_node("call_tool", call_tool_node)
    builder.add_edge(START, "call_tool")
    builder.add_edge("call_tool", END)
    return builder.compile(checkpointer=InMemorySaver()) if checkpointer else builder.compile()


def _build_react_graph(
    *,
    fail_after_tool: bool = False,
    tool_raises: bool = False,
    tool_interrupts: bool = False,
    tool_delay: float = 0.0,
) -> Any:
    """ReAct-shaped graph: model -> ToolNode -> model, with a real tool round-trip.

    The first model call returns ``REACT_REASONING`` text plus a ``lookup``
    tool call (so streaming emits ``agent_reasoning``); the second returns
    ``FAKE_RESPONSE``. Options:

    - ``fail_after_tool``: the second model call raises, so a turn fails
      after its tool records exist.
    - ``tool_raises``: the tool raises and ``ToolNode`` hands the error back
      to the model, so the turn still finishes.
    - ``tool_interrupts``: the tool calls ``interrupt()`` (HITL); the graph
      gets a checkpointer so ``Command(resume=...)`` can continue it.
    - ``tool_delay``: the (async) tool sleeps this long before returning.
    """
    import asyncio

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.types import interrupt

    @tool
    async def lookup(query: str) -> str:
        """Look something up."""
        if tool_delay:
            await asyncio.sleep(tool_delay)
        if tool_raises:
            raise RuntimeError("tool boom https://x.example/?token=secret")
        if tool_interrupts:
            approved = interrupt({"q": query})
            return f"tool result: {query} ({approved})"
        return f"tool result: {query}"

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=REACT_REASONING,
                tool_calls=[{"name": "lookup", "args": {"query": "A-1"}, "id": "tc-1"}],
            ),
            AIMessage(content=FAKE_RESPONSE),
        ]
    )
    calls = 0

    async def agent(state: _State) -> dict:
        nonlocal calls
        calls += 1
        if fail_after_tool and calls > 1:
            raise RuntimeError("model boom")
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("agent", agent)
    tool_node = ToolNode([lookup], handle_tool_errors=True) if tool_raises else ToolNode([lookup])
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver() if tool_interrupts else None)


# ─── Beacon endpoint mocks ────────────────────────────────────────


_CONV_PATH = re.compile(rf"^{re.escape(BASE_URL)}/api/v1/agentic/conversations$")
_SESSIONS_PATH = re.compile(
    rf"^{re.escape(BASE_URL)}/api/v1/agentic/conversations/[^/]+/sessions$"
)
_ENTRIES_PATH = re.compile(
    rf"^{re.escape(BASE_URL)}/api/v1/agentic/conversations/[^/]+/entries$"
)
_CLOSE_SESS_PATH = re.compile(
    rf"^{re.escape(BASE_URL)}/api/v1/agentic/conversations/[^/]+/sessions/[^/]+/close$"
)
_PARTICIPANTS_PATH = re.compile(
    rf"^{re.escape(BASE_URL)}/api/v1/agentic/conversations/[^/]+/sessions/[^/]+/participants$"
)


def mock_beacon_endpoints(respx_mock: Any) -> dict[str, Any]:
    """Mount respx routes for the lifecycle endpoints plus a close_session guard route.

    Under the unified ``conversation_id == session_id`` model the adapter
    no longer POSTs ``close_session``; the ``close_sess`` route is mounted
    purely so tests can assert it stayed at zero calls.

    Returns a dict so tests can assert call counts and inspect bodies.
    """
    return {
        "create_conv": respx_mock.post(_CONV_PATH).mock(
            return_value=httpx.Response(200, json={})
        ),
        "create_sess": respx_mock.post(_SESSIONS_PATH).mock(
            return_value=httpx.Response(200, json={})
        ),
        "add_participants": respx_mock.post(_PARTICIPANTS_PATH).mock(
            return_value=httpx.Response(200, json={})
        ),
        "add_entries": respx_mock.post(_ENTRIES_PATH).mock(
            return_value=httpx.Response(200, json={"accepted": 1, "rejected": 0, "errors": []})
        ),
        "close_sess": respx_mock.post(_CLOSE_SESS_PATH).mock(
            return_value=httpx.Response(200, json={})
        ),
    }


@pytest.fixture
def respx_routes(respx_mock):
    """Auto-mounted respx routes for the lifecycle endpoints + close_session guard."""
    return mock_beacon_endpoints(respx_mock)


@pytest.fixture
async def async_client():
    async with AsyncIngestionClient(base_url=BASE_URL, api_key="t") as c:
        yield c


@pytest.fixture
def sync_client():
    with IngestionClient(base_url=BASE_URL, api_key="t") as c:
        yield c


@pytest.fixture
async def async_mgr(async_client):
    mgr = AsyncSessionManager(
        async_client,
        agent_id="bot",
        owner_id="user-1",
        session_id="ses-1",
    )
    yield mgr


@pytest.fixture
def sync_mgr(sync_client):
    return SessionManager(
        sync_client,
        agent_id="bot",
        owner_id="user-1",
        session_id="ses-1",
    )
