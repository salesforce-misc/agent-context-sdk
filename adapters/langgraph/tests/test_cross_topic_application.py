# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Cross-Topic Memory Application.

Three sequential conversations under one ``owner_id``: travel planning,
restaurant pick, gift ideas. Each is a separate ``conversation_id`` and a
separate ``GET /context`` call.

The capability under test:

1. The SDK does not impose conversation scoping on top of Beacon's
   owner-keyed retrieval. Whatever Beacon returns for owner_id=X is what
   the formatter renders.
2. With the same generalized fact set returned for all three retrievals,
   the SystemMessage at S2 (restaurant) and S3 (gifts) carries facts
   originally extracted from S1 (travel) — vegetarian, bad knee, has
   wife — without re-prompting.
3. ``dedupe=True`` collapses duplicate facts that show up when a fact is
   recalled from multiple memory sources.
"""
from __future__ import annotations

import json
import re

import httpx
import respx
from agentic_context import AsyncRetrievalClient

from agentic_context_langgraph import make_context_node
from agentic_context_langgraph.retrieval_normalize import (
    context_to_system_message,
    format_context_block,
)

from tests.fixtures.cross_topic_facts import (
    DUPLICATED_FACTS,
    GIFT_CONTEXT,
    RESTAURANT_CONTEXT,
    TRAVEL_CONTEXT,
)

from .conftest import BASE_URL, CONTEXT_URL
API_KEY = "test-token"
AGENT_ID = "concierge-bot"
OWNER_ID = "user-cross-topic"

S1_CONV = "conv-travel"
S2_CONV = "conv-restaurant"
S3_CONV = "conv-gifts"


# ─── Cross-topic carryover via three node invocations ───────────────


@respx.mock
async def test_three_topics_one_owner_share_facts() -> None:
    """All three topics' SystemMessages contain the same generalized facts."""
    # Beacon returns the same fact set per call; only the ``summary``
    # differs because that's per-conversation.
    payloads = iter([TRAVEL_CONTEXT, RESTAURANT_CONTEXT, GIFT_CONTEXT])
    route = respx.post(CONTEXT_URL).mock(
        side_effect=lambda req: httpx.Response(200, json=next(payloads))
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        node = make_context_node(rc, agent_id=AGENT_ID)
        s1 = await node({"customer_id": OWNER_ID}, {"configurable": {"thread_id": S1_CONV}})
        s2 = await node({"customer_id": OWNER_ID}, {"configurable": {"thread_id": S2_CONV}})
        s3 = await node({"customer_id": OWNER_ID}, {"configurable": {"thread_id": S3_CONV}})

    assert len(route.calls) == 3
    # Each call carried a different conversation_id but the same owner_id.
    seen_convs = set()
    for call in route.calls:
        body = json.loads(call.request.content)
        seen_convs.add(body["conversation_id"])
        assert body["owner_id"] == OWNER_ID
    assert seen_convs == {S1_CONV, S2_CONV, S3_CONV}

    # Cross-cutting facts present in every topic's SystemMessage.
    for delta in (s1, s2, s3):
        content = delta["messages"][0].content
        assert "Vegetarian" in content
        assert "Bad knee" in content
        assert "Travels with wife" in content
        assert "Photographer" in content


@respx.mock
async def test_per_conversation_summary_differs_facts_match() -> None:
    """The summary slot is conversation-specific; the fact list is not."""
    payloads = iter([TRAVEL_CONTEXT, RESTAURANT_CONTEXT, GIFT_CONTEXT])
    respx.post(CONTEXT_URL).mock(
        side_effect=lambda req: httpx.Response(200, json=next(payloads))
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        node = make_context_node(rc, agent_id=AGENT_ID)
        s1 = await node({"customer_id": OWNER_ID}, {"configurable": {"thread_id": S1_CONV}})
        s2 = await node({"customer_id": OWNER_ID}, {"configurable": {"thread_id": S2_CONV}})
        s3 = await node({"customer_id": OWNER_ID}, {"configurable": {"thread_id": S3_CONV}})

    s1c = s1["messages"][0].content
    s2c = s2["messages"][0].content
    s3c = s3["messages"][0].content

    # Each summary mentions the topic-specific intent.
    assert "March trip to Japan" in s1c
    assert "date-night dinner" in s2c
    assert "birthday gift" in s3c

    # Topic-specific summaries do NOT leak into the fact section of
    # other topics. (Ensures the formatter isn't accidentally
    # concatenating across calls.)
    assert "March trip to Japan" not in s2c
    assert "birthday gift" not in s2c
    assert "date-night dinner" not in s3c


# ─── Stable id ensures S1's SystemMessage is overwritten in S2 ──────


def test_cross_topic_system_messages_share_stable_id() -> None:
    """All three topics produce the same id so add_messages dedupes.

    Without this, the LLM would see three accumulated SystemMessages
    (one per topic) instead of just the latest.
    """
    m1 = context_to_system_message(TRAVEL_CONTEXT)
    m2 = context_to_system_message(RESTAURANT_CONTEXT)
    m3 = context_to_system_message(GIFT_CONTEXT)
    assert m1 is not None and m2 is not None and m3 is not None
    assert m1.id == m2.id == m3.id == "agentic-memory-context"


# ─── Fact-level dedup ───────────────────────────────────────────────


def test_dedupe_off_renders_duplicates_verbatim() -> None:
    block = format_context_block(DUPLICATED_FACTS)
    veg_count = len(re.findall(r"Vegetarian", block, re.IGNORECASE))
    # Three vegetarian entries in the fixture (two exact, one lowercase).
    assert veg_count == 3, block


def test_dedupe_on_keeps_first_occurrence_of_case_insensitive_duplicates() -> None:
    """Dedup collapses case-insensitive duplicates to the first occurrence and
    keeps the first row's confidence (0.92), not later ones."""
    block = format_context_block(DUPLICATED_FACTS, dedupe=True)
    veg_count = len(re.findall(r"Vegetarian", block, re.IGNORECASE))
    # Dedup keeps first occurrence only.
    assert veg_count == 1, block
    # Other unique facts still present.
    assert "Bad knee" in block
    assert "Travels with wife" in block
    # First Vegetarian row had confidence=0.92.
    assert "confidence=0.92" in block
    # The 0.85 / 0.7 duplicates should be dropped entirely.
    assert "confidence=0.85" not in block
    assert "confidence=0.70" not in block


# ─── Node wiring ────────────────────────────────────────────────────


@respx.mock
async def test_node_with_dedupe_strips_duplicate_facts_per_turn() -> None:
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=DUPLICATED_FACTS)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        node = make_context_node(rc, agent_id=AGENT_ID, dedupe=True)
        delta = await node(
            {"customer_id": OWNER_ID},
            {"configurable": {"thread_id": S2_CONV}},
        )

    content = delta["messages"][0].content
    veg_count = len(re.findall(r"Vegetarian", content, re.IGNORECASE))
    assert veg_count == 1


# ─── Path-3 self-driving variant ─────────────────────────────────────


from typing import Annotated, Any, TypedDict  # noqa: E402

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import SystemMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from agentic_context_langgraph import LangGraphAdapter  # noqa: E402

from tests.conftest import mock_beacon_endpoints  # noqa: E402


@respx.mock
async def test_cross_topic_application_with_self_driving_adapter(respx_mock) -> None:
    """Three smart-adapter sessions (one per topic) all see the same generalized facts.

    Each ``stream()`` runs under a distinct ``session_id`` (= LangGraph
    ``thread_id``; under the unified model Beacon maps this onto both
    ``conversationId`` and ``sessionId`` on the wire). The three GETs Beacon
    receives carry the three different session_ids. The fact set is identical
    across all three.
    """
    mock_beacon_endpoints(respx_mock)
    payloads = iter([TRAVEL_CONTEXT, RESTAURANT_CONTEXT, GIFT_CONTEXT])
    respx_mock.post(CONTEXT_URL).mock(
        side_effect=lambda req: httpx.Response(200, json=next(payloads))
    )

    captured: list[list[Any]] = []

    class _S(TypedDict):
        messages: Annotated[list, add_messages]

    # Three responses so each topic's turn produces a clean reply.
    model = FakeListChatModel(responses=["s1-ack", "s2-ack", "s3-ack"])

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
    async with adapter:
        async for _ in adapter.stream("planning travel", S1_CONV):
            pass
        async for _ in adapter.stream("picking a restaurant", S2_CONV):
            pass
        async for _ in adapter.stream("gift ideas", S3_CONV):
            pass

    # Three node executions, each saw a SystemMessage with the same
    # generalized fact set.
    assert len(captured) == 3
    for snapshot in captured:
        sysmsgs = [m for m in snapshot if isinstance(m, SystemMessage)]
        only = next(m for m in sysmsgs if getattr(m, "id", None) == "agentic-memory-context")
        assert "Vegetarian" in only.content
        assert "Bad knee" in only.content
        assert "Travels with wife" in only.content
        assert "Photographer" in only.content
