# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Smoke: LangGraph adapter (``agentic_context_langgraph``).

Wiring under test is the smart (self-driving) path: a real ReAct-style graph
(agent node + ToolNode + tools_condition + InMemorySaver) passed in builder
form so the adapter's default tools are bound, driven by
``await adapter.run(text, session_id=sid)`` inside ``async with``.
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from harness import (
    AGENT_ID, OWNER_ID, SYSTEM_PROMPT, TOOL_NAME, TURNS,
    Capture, FakeAuth, LiveConfig, LiveContract, MockBeacon, RecordingPassThrough,
    SmokeContract, lookup_order, new_session_id, read_back, script_next,
)

pytestmark = pytest.mark.langgraph

pytest.importorskip("langgraph")
pytest.importorskip("agentic_context_langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from agentic_context_langgraph import LangGraphAdapter  # noqa: E402

lookup_order_tool = tool(lookup_order)


def _progress(messages: list[BaseMessage]) -> tuple[int, bool]:
    """(user messages seen, whether the last one already has a tool result)."""
    users, answered = 0, False
    for m in messages:
        if isinstance(m, HumanMessage):
            users, answered = users + 1, False
        elif isinstance(m, ToolMessage):
            answered = True
    return users, answered


class ScriptedChatModel(BaseChatModel):
    """Offline model: calls lookup_order on turn 1, then replies from the script."""

    @property
    def _llm_type(self) -> str:
        return "smoke-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        action, payload = script_next(*_progress(messages))
        if action == "tool":
            msg = AIMessage(content="", tool_calls=[{
                "name": TOOL_NAME, "args": payload,
                "id": f"call_{uuid4().hex[:12]}", "type": "tool_call",
            }])
        else:
            msg = AIMessage(content=payload)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def build_graph(model: BaseChatModel):
    """Builder ``(sdk_tools) -> compiled graph`` handed to the adapter."""

    def build(sdk_tools: list[Any]):
        tools = [*sdk_tools, lookup_order_tool]
        bound = model.bind_tools(tools)

        async def agent(state: MessagesState) -> dict[str, Any]:
            reply = await bound.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
            return {"messages": [reply]}

        g = StateGraph(MessagesState)
        g.add_node("agent", agent)
        g.add_node("tools", ToolNode(tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=InMemorySaver())

    return build


async def drive(model: BaseChatModel, capture: Capture, auth: Any) -> None:
    async with LangGraphAdapter(
        capture.agent_id, graph=build_graph(model), owner_id=capture.owner_id, auth=auth,
    ) as adapter:
        for turn in TURNS:
            records = await adapter.run(turn.user, session_id=capture.session_id)
            replies = [r.content for r in records if r.event_type == "agent_response"]
            capture.replies.append(replies[-1] if replies else "")


# ─── Tier 1: offline ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def offline_capture() -> Capture:
    capture = Capture("langgraph", new_session_id("langgraph"), OWNER_ID, AGENT_ID)
    with MockBeacon(capture):
        asyncio.run(drive(ScriptedChatModel(), capture, FakeAuth()))
    return capture


KNOWN_GAPS: dict[str, str] = {}


@pytest.mark.offline
class TestLangGraphOffline(SmokeContract):
    agent_type = "LangGraph"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, offline_capture: Capture) -> Capture:
        return offline_capture


# ─── Tier 2: live ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_capture(live_config: LiveConfig) -> Capture:
    cfg = live_config
    capture = Capture("langgraph", new_session_id("langgraph"), cfg.owner_id, cfg.agent_id)

    async def run() -> None:
        with RecordingPassThrough(capture):
            await drive(ScriptedChatModel(), capture, cfg.auth)
        capture.readback = await read_back(cfg, capture.session_id)

    asyncio.run(run())
    return capture


@pytest.mark.live
class TestLangGraphLive(LiveContract):
    agent_type = "LangGraph"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, live_capture: Capture) -> Capture:
        return live_capture
