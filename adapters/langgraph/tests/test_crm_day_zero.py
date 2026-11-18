# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Day-Zero CRM Context.

Brand-new conversation, agent has never spoken to user. CRM data
("Sarah Chen", Enterprise Gold, open P1 case) is preloaded by the
backend Context Service and exposed via ``GET /agentic/context``.

What's under test at the SDK layer:

1. ``make_context_node`` works on the very first turn (no conversation
   history) — context becomes available *before* the user types.
2. The hierarchical CRM payload (``identity``, ``open_case``,
   ``churn_risk``, ``account_manager``) renders into a single
   prompt-ready ``SystemMessage`` with all fields preserved.
3. With ``rank_urgent=True``, the open P1 case and churn risk surface
   ahead of generic preference / location facts so the LLM sees them
   first.
"""
from __future__ import annotations

import httpx
import pytest
import respx
from agentic_context import AsyncRetrievalClient
from langchain_core.messages import SystemMessage

from agentic_context_langgraph import make_context_node
from agentic_context_langgraph.retrieval_normalize import (
    context_to_system_message,
    format_context_block,
)

from tests.fixtures.crm_payloads import (
    SARAH_CHEN_DEEP,
    SARAH_CHEN_FAST,
    SARAH_CHEN_QUIET,
    URGENT_FACT_TYPES,
)

from .conftest import BASE_URL, CONTEXT_URL
API_KEY = "test-token"
AGENT_ID = "support-bot"
CONV_ID = "conv-day-zero"
OWNER_ID = "sarah.chen@enterprise.example.com"


def _config() -> dict:
    return {"configurable": {"thread_id": CONV_ID}}


# ─── Fast-mode payload ───────────────────────────────────────────────


@respx.mock
async def test_first_turn_node_surfaces_name_and_p1_case() -> None:
    """Turn 0: state has no messages yet, but context node still fires.

    The agent's first reply must be able to greet by name and
    proactively reference the open P1 case — both come from the
    SystemMessage produced here.
    """
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=SARAH_CHEN_FAST)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        node = make_context_node(rc, agent_id=AGENT_ID)
        # Empty state — no prior messages, no user input yet. Day zero.
        delta = await node({"customer_id": OWNER_ID}, _config())

    assert "messages" in delta and len(delta["messages"]) == 1
    msg = delta["messages"][0]
    assert isinstance(msg, SystemMessage)
    assert "Sarah Chen" in msg.content
    assert "P1 case #4521" in msg.content
    assert "Enterprise Gold" in msg.content


@respx.mock
async def test_deep_payload_renders_every_signal() -> None:
    """All nine flagship CRM signals from the deep payload must land."""
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=SARAH_CHEN_DEEP)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        node = make_context_node(rc, agent_id=AGENT_ID)
        delta = await node({"customer_id": OWNER_ID}, _config())

    content = delta["messages"][0].content
    for must_appear in (
        "Sarah Chen",
        "Singapore",
        "Preferred contact: email",
        "Data 360 Enterprise License",
        "P1 case #4521",
        "Churn risk: Medium",
        "NPS: 7",
        "James Rodriguez",
        "Enterprise Gold",
    ):
        assert must_appear in content, f"missing {must_appear!r} in:\n{content}"


# ─── Urgency ranking ─────────────────────────────────────────────────


def test_format_without_ranking_preserves_payload_order() -> None:
    """Default behaviour: identity first (since payload lists it first)."""
    block = format_context_block(SARAH_CHEN_DEEP)
    lines = [line for line in block.splitlines() if line.startswith("- ")]
    # First fact rendered = first fact in payload.
    assert lines[0].startswith("- Customer name: Sarah Chen")


def test_format_rank_urgent_pulls_p1_and_churn_to_top() -> None:
    """rank_urgent=True surfaces open_case + churn_risk ahead of identity."""
    block = format_context_block(SARAH_CHEN_DEEP, rank_urgent=True)
    lines = [line for line in block.splitlines() if line.startswith("- ")]

    # First two lines must be the urgent ones (in their original mutual
    # order — open_case appeared before churn_risk in the payload).
    assert "P1 case #4521" in lines[0], lines
    assert "Churn risk: Medium" in lines[1], lines
    # Identity / location got pushed below the urgent block.
    identity_idx = next(i for i, ln in enumerate(lines) if "Sarah Chen" in ln)
    p1_idx = next(i for i, ln in enumerate(lines) if "P1 case #4521" in ln)
    assert p1_idx < identity_idx


def test_rank_urgent_is_noop_when_no_urgent_facts() -> None:
    """A profile with no open cases / churn risk renders unchanged."""
    a = format_context_block(SARAH_CHEN_QUIET)
    b = format_context_block(SARAH_CHEN_QUIET, rank_urgent=True)
    assert a == b


def test_rank_urgent_custom_types_extends_default() -> None:
    """Caller-supplied urgent_types overrides DEFAULT_URGENT_TYPES."""
    payload = {
        "userContext": [
            {"text": "calm fact", "type": "preference", "confidence": 0.5},
            {"text": "tagged urgent", "type": "compliance_flag", "confidence": 0.9},
        ]
    }
    out = format_context_block(
        payload,
        rank_urgent=True,
        urgent_types=("compliance_flag",),
    )
    lines = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert "tagged urgent" in lines[0]
    assert "calm fact" in lines[1]


def test_default_urgent_types_covers_scenario_2_signals() -> None:
    """Sanity: open_case + churn_risk are in the shipped default set."""
    from agentic_context_langgraph.retrieval_normalize import DEFAULT_URGENT_TYPES

    assert "open_case" in DEFAULT_URGENT_TYPES
    assert "churn_risk" in DEFAULT_URGENT_TYPES
    # Mirrors the fixture's URGENT_FACT_TYPES contract.
    for t in ("open_case", "churn_risk"):
        assert t in URGENT_FACT_TYPES


# ─── Node-level wiring with rank_urgent ──────────────────────────────


@respx.mock
async def test_node_with_rank_urgent_emits_urgent_first_in_system_message() -> None:
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=SARAH_CHEN_DEEP)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        node = make_context_node(rc, agent_id=AGENT_ID, rank_urgent=True)
        delta = await node({"customer_id": OWNER_ID}, _config())

    msg = delta["messages"][0]
    assert isinstance(msg, SystemMessage)
    p1_idx = msg.content.find("P1 case #4521")
    name_idx = msg.content.find("Customer name: Sarah Chen")
    assert p1_idx >= 0 and name_idx >= 0
    assert p1_idx < name_idx


# ─── Path-3 self-driving variant ─────────────────────────────────────


from typing import Annotated, Any, TypedDict  # noqa: E402

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from agentic_context_langgraph import LangGraphAdapter  # noqa: E402

from tests.conftest import mock_beacon_endpoints  # noqa: E402


@respx.mock
async def test_crm_day_zero_with_self_driving_adapter(respx_mock) -> None:
    """Day-zero greet-by-name flows through the path-3 constructor.

    Same SARAH_CHEN_FAST fixture; the smart adapter auto-fetches and
    prepends the SystemMessage so the graph node sees Sarah's name + open
    P1 case + Enterprise tier on its first execution.
    """
    mock_beacon_endpoints(respx_mock)
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=SARAH_CHEN_FAST)
    )

    captured: list[list[Any]] = []

    class _S(TypedDict):
        messages: Annotated[list, add_messages]

    model = FakeListChatModel(responses=["Hi Sarah, I see your P1 case is still open."])

    async def chat(state: _S) -> dict:
        captured.append(list(state["messages"]))
        out = await model.ainvoke(state["messages"])
        return {"messages": [out]}

    builder: StateGraph = StateGraph(_S)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    graph = builder.compile()

    adapter = LangGraphAdapter(
        graph=graph,
        agent_id=AGENT_ID,
        owner_id=OWNER_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        dataspace="default",
    )
    # Day-zero: no prior user input, but the smart adapter still fetches
    # and injects context before driving the node. Pass a one-word user
    # turn so the wrapping kicks in (the contract for day-zero greet is
    # the agent speaks first, but the adapter pipeline is "GET, then
    # drive on user input"; the greet-by-name property still surfaces
    # because the SystemMessage carries Sarah's name).
    async with adapter:
        async for _ in adapter.stream("hi", CONV_ID):
            pass

    assert captured, "graph node never executed"
    sysmsgs = [m for m in captured[0] if isinstance(m, SystemMessage)]
    only = next(m for m in sysmsgs if getattr(m, "id", None) == "agentic-memory-context")
    assert "Sarah Chen" in only.content
    assert "P1 case #4521" in only.content
    assert "Enterprise Gold" in only.content
