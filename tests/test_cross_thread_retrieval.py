# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Scenario 1 — Multi-Session Memory.

Verifies the SDK contract: ingest under ``conversation_id="conv-S1"`` for
``owner_id="user-X"``, then call ``get_context`` with a *different*
``conversation_id="conv-S2"`` (same ``owner_id``) and confirm that facts
from S1 surface in the S2 retrieval. The Beacon-side aggregation is
mocked — what's under test here is that the SDK correctly keys by owner
across conversations and doesn't accidentally scope by conversation.
"""
from __future__ import annotations

import json
import re

import httpx
import pytest
import respx

from agentic_context import AsyncRetrievalClient

from .conftest import CONTEXT_URL

BASE_URL = "https://beacon.example.com"
API_KEY = "test-token"
AGENT_ID = "laptop-bot"
OWNER_ID = "user-X"

S1_CONV = "conv-S1-laptop-shopping"
S2_CONV = "conv-S2-followup-next-day"


# Facts the user established in S1 — laptop shopping. The S2 retrieval
# must surface these without the user re-stating.
S1_EXTRACTED_FACTS = {
    "summary": (
        "User shopping for a video-editing laptop. Budget ~$2000. Wants "
        "16-inch screen, 32GB RAM, weight matters (travels). Concerned "
        "about MacBook Pro port situation for external drives."
    ),
    "userContext": [
        {"text": "Budget: ~$2000", "type": "preference", "confidence": 0.95},
        {"text": "Use case: video editing", "type": "use_case", "confidence": 0.99},
        {"text": "Min 32GB RAM required", "type": "spec_requirement", "confidence": 0.97},
        {"text": "Prefers 16-inch screen", "type": "preference", "confidence": 0.93},
        {"text": "Weight matters — travels frequently", "type": "preference", "confidence": 0.91},
        {
            "text": "Open concern: external-drive port situation on MacBook Pro",
            "type": "open_concern",
            "confidence": 0.88,
        },
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_get_context_for_s2_carries_s1_facts() -> None:
    """Same owner_id, different conversation_id → S1 facts must surface in S2."""
    route = respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=S1_EXTRACTED_FACTS)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        s2_payload = await rc.get_context(
            conversation_id=S2_CONV, owner_id=OWNER_ID, agent_id=AGENT_ID
        )

    assert route.called
    body = json.loads(route.calls[0].request.content)
    # Different conversation_id from S1 — proves we're not scoping by S1's id.
    assert body["conversation_id"] == S2_CONV
    assert body["owner_id"] == OWNER_ID

    # And the response carries forward S1's facts.
    summary = s2_payload.get("summary", "")
    assert "video-editing" in summary or "video editing" in summary
    assert "$2000" in summary

    fact_texts = [f["text"] for f in s2_payload.get("userContext", [])]
    assert any("32GB RAM" in t for t in fact_texts), fact_texts
    assert any("16-inch" in t for t in fact_texts), fact_texts
    assert any("port situation" in t for t in fact_texts), fact_texts


@pytest.mark.asyncio
@respx.mock
async def test_two_s2_conversations_share_owner_facts() -> None:
    """Two distinct S2 conversations for the same owner both see S1 facts.

    Beacon-side aggregation is owner-keyed, not conversation-keyed. This
    test asserts the SDK does not impose conversation scoping on top.
    """
    route = respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=S1_EXTRACTED_FACTS)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        a = await rc.get_context(
            conversation_id="conv-S2-A", owner_id=OWNER_ID, agent_id=AGENT_ID
        )
        b = await rc.get_context(
            conversation_id="conv-S2-B", owner_id=OWNER_ID, agent_id=AGENT_ID
        )

    assert len(route.calls) == 2
    # Both calls hit the same owner with different conversation ids.
    body_a = json.loads(route.calls[0].request.content)
    body_b = json.loads(route.calls[1].request.content)
    assert body_a["conversation_id"] != body_b["conversation_id"]
    assert body_a["owner_id"] == body_b["owner_id"] == OWNER_ID

    # Both retrievals return identical facts — the owner profile.
    assert a == b == S1_EXTRACTED_FACTS


@pytest.mark.asyncio
@respx.mock
async def test_s2_payload_text_renders_all_s1_constraints() -> None:
    """S2 turn 1 retrieval must surface every S1 constraint as text.

    Asserts at the SDK layer (no LangChain dependency) — every fact text
    and the summary land in the response payload, ready for whatever
    formatter a caller wants to plug in.
    """
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=S1_EXTRACTED_FACTS)
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        payload = await rc.get_context(
            conversation_id=S2_CONV, owner_id=OWNER_ID, agent_id=AGENT_ID
        )

    # Concatenate everything a formatter would render and assert each
    # S1 constraint is recoverable.
    fact_blob = " ".join(
        f["text"] for f in payload.get("userContext", []) if "text" in f
    )
    assert "Budget: ~$2000" in fact_blob
    assert "32GB RAM" in fact_blob
    assert "16-inch" in fact_blob
    assert "Weight matters" in fact_blob
    assert "port situation" in fact_blob
    # Plus the summary line.
    assert re.search(r"User shopping for a video-editing laptop", payload["summary"])
