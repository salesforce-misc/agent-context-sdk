# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Multi-session conversation rehydration.

Validates ``conversation_to_messages`` against a realistic multi-session
payload (S1 = laptop shopping, S2 = day-after follow-up). Today the only
multi-session test in the suite uses toy ``"s1-u" / "s2-u"`` strings —
this fills the gap with a payload that resembles what Beacon returns when
``GET /agentic/conversations/{id}?scope=all`` stitches a user's day-N
session with their day-N+1 session.

Also asserts that re-injecting a freshly-retrieved context block on every
S2 turn replaces (not appends) the prior SystemMessage via the stable
``agentic-memory-context`` id.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from agentic_context_langgraph.retrieval_normalize import (
    context_to_system_message,
    conversation_to_messages,
)


# Realistic two-session conversation: S1 = day 1 shopping, S2 = day 2
# follow-up where user resolves the S1 port concern.
TWO_DAY_PAYLOAD = {
    "sessions": [
        {
            "entries": [
                {"type": "user_message", "content": "I'm looking for a laptop for video editing. Budget ~$2000."},
                {"type": "agent_response", "content": "Got it — what about screen size and weight?"},
                {"type": "user_message", "content": "16-inch, 32GB RAM, and weight matters since I travel."},
                {"type": "agent_response", "content": "MacBook Pro 16 fits — though check the port situation for external drives."},
                {"type": "user_message", "content": "Yeah, I'm worried about the ports. Let me think on it."},
                {"type": "agent_response", "content": "No rush — I'll be here when you're back."},
            ]
        },
        # New session = next day. Same owner_id, different session_id.
        {
            "entries": [
                {"type": "user_message", "content": "Hey, I'm back."},
                {"type": "agent_response", "content": "Welcome back — picking up the laptop hunt?"},
                {"type": "user_message", "content": "Yeah. I'm okay with using a hub for the ports."},
                {"type": "agent_response", "content": "Then the MacBook Pro 16 with 32GB lands in your $2000 budget cleanly."},
            ]
        },
    ]
}


def test_two_day_conversation_rehydrates_in_chronological_order() -> None:
    """Rehydrating a two-session conversation keeps every turn, in order:

    - session boundary is preserved: 'Hey I'm back' lands after S1's wrap-up;
    - all five S1 constraints (budget, RAM, screen, weight, port) survive;
    - the S2 'I'm okay with a hub' resolution and the synthesis reply both land.
    """
    msgs = conversation_to_messages(TWO_DAY_PAYLOAD)
    # 6 user/agent pairs total: 3 from S1 (3U+3A=6) + 2 from S2 (2U+2A=4) = 10.
    assert len(msgs) == 10

    # First message is the very first user turn.
    assert isinstance(msgs[0], HumanMessage)
    assert "video editing" in msgs[0].content
    # Last message is the final agent reply in S2.
    assert isinstance(msgs[-1], AIMessage)
    assert "MacBook Pro 16" in msgs[-1].content

    contents = [m.content for m in msgs]
    s1_wrap_idx = next(i for i, c in enumerate(contents) if "I'll be here" in c)
    s2_open_idx = next(i for i, c in enumerate(contents) if c == "Hey, I'm back.")
    assert s2_open_idx == s1_wrap_idx + 1

    blob = " ".join(contents)
    assert "$2000" in blob
    assert "32GB RAM" in blob
    assert "16-inch" in blob
    assert "weight matters" in blob.lower()
    assert "port situation" in blob
    assert "okay with using a hub" in blob
    # Synthesis cites budget + RAM + model.
    assert "MacBook Pro 16 with 32GB" in blob


def test_re_injected_context_replaces_via_stable_id() -> None:
    """S2 turns 1 and 2 each fetch fresh context; only the latest survives.

    Without a stable id, every turn would accumulate a SystemMessage —
    poisoning the prompt with stale context blocks.
    """
    s2_turn1_payload = {
        "summary": "User left mid-shopping; concerned about ports.",
        "userContext": [
            {"text": "Budget ~$2000", "type": "preference", "confidence": 0.95},
            {"text": "Open concern: ports", "type": "open_concern", "confidence": 0.88},
        ],
    }
    s2_turn2_payload = {
        "summary": "Port concern resolved (user accepts hub).",
        "userContext": [
            {"text": "Budget ~$2000", "type": "preference", "confidence": 0.95},
            {"text": "Resolved: hub acceptable for ports", "type": "resolved", "confidence": 0.91},
        ],
    }

    m1 = context_to_system_message(s2_turn1_payload)
    m2 = context_to_system_message(s2_turn2_payload)
    assert isinstance(m1, SystemMessage) and isinstance(m2, SystemMessage)
    assert m1.id == m2.id == "agentic-memory-context"

    merged = add_messages([m1], [m2])
    assert len(merged) == 1
    only = merged[0]
    assert "Resolved: hub acceptable" in only.content
    # The stale "Open concern: ports" line from turn 1 is gone.
    assert "Open concern: ports" not in only.content


# ─── Path-3 self-driving variant ─────────────────────────────────────


import httpx  # noqa: E402
import respx  # noqa: E402
from typing import Annotated, Any, TypedDict  # noqa: E402

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from agentic_context_langgraph import LangGraphAdapter  # noqa: E402

from .conftest import BASE_URL, CONTEXT_URL, mock_beacon_endpoints  # noqa: E402


@respx.mock
async def test_multi_session_carryover_with_self_driving_adapter(respx_mock):
    """S2 turn-1 sees the S1-derived facts via the smart adapter's auto-fetch.

    Same contract as the unit tests above, exercised through the path-3
    constructor: no manual ``AsyncRetrievalClient`` / ``make_context_node`` /
    ``SessionManager`` plumbing. The test asserts that the SystemMessage
    the graph node sees on S2 turn 1 carries the carry-over signals
    (port concern + budget) extracted by the upstream summarizer.
    """
    mock_beacon_endpoints(respx_mock)
    s2_payload = {
        "summary": "User left mid-shopping; concerned about ports.",
        "userContext": [
            {"text": "Budget ~$2000", "type": "preference", "confidence": 0.95},
            {"text": "Open concern: ports", "type": "open_concern", "confidence": 0.88},
        ],
    }
    respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=s2_payload)
    )

    captured: list[list[Any]] = []

    class _S(TypedDict):
        messages: Annotated[list, add_messages]

    model = FakeListChatModel(responses=["welcome back"])

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
        agent_id="bot",
        owner_id="user-1",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
    )
    async with adapter:
        async for _ in adapter.stream("hey, I'm back", "ses-2"):
            pass

    assert captured, "graph node never executed"
    sysmsgs = [m for m in captured[0] if isinstance(m, SystemMessage)]
    only = next(m for m in sysmsgs if getattr(m, "id", None) == "agentic-memory-context")
    assert "Budget ~$2000" in only.content
    assert "Open concern: ports" in only.content
