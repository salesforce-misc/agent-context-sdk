# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Integration tests for make_context_node — uses respx to mock the POST."""
from __future__ import annotations

import json

import httpx
import respx
from agentic_context import AsyncRetrievalClient
from langchain_core.messages import SystemMessage

from agentic_context_langgraph import make_context_node

from .conftest import BASE_URL, CONTEXT_URL

API_KEY = "test-token"
AGENT_ID = "cs-bot"
CONV_ID = "conv-1"
OWNER_ID = "cust-42"


def _config(thread_id: str | None = CONV_ID) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@respx.mock
async def test_node_default_messages_injection() -> None:
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "summary": "Prefers email.",
                "userContext": [
                    {"text": "Owns Pro plan", "type": "account_info", "confidence": 0.9}
                ],
            },
        )
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        node = make_context_node(client, agent_id=AGENT_ID)
        delta = await node({"customer_id": OWNER_ID}, _config())

    assert "messages" in delta
    msgs = delta["messages"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert isinstance(msg, SystemMessage)
    assert "Owns Pro plan" in msg.content
    assert "In-session summary:\nPrefers email." in msg.content


@respx.mock
async def test_node_into_custom_field() -> None:
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "Just a summary."})
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        node = make_context_node(
            client, agent_id=AGENT_ID, into="customer_context"
        )
        delta = await node({"customer_id": OWNER_ID}, _config())

    assert "messages" not in delta
    assert "customer_context" in delta
    assert "Just a summary." in delta["customer_context"]


@respx.mock
async def test_node_request_carries_correct_params_and_headers() -> None:
    route = respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "x"})
    )

    async with AsyncRetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace="ds-1"
    ) as client:
        node = make_context_node(client, agent_id=AGENT_ID)
        await node({"customer_id": OWNER_ID}, _config())

    assert route.called
    req = route.calls[0].request
    body = json.loads(req.content)
    assert body["conversation_id"] == CONV_ID
    assert body["owner_id"] == OWNER_ID
    assert body["agent_id"] == AGENT_ID
    assert body["owner_type"] == "individual"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert req.headers["ctx-dataspace-ds_name"] == "ds-1"


@respx.mock
async def test_node_empty_payload_yields_empty_delta() -> None:
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        node = make_context_node(client, agent_id=AGENT_ID)
        delta = await node({"customer_id": OWNER_ID}, _config())

    assert delta == {}


async def test_node_missing_owner_or_thread_id_noops() -> None:
    """No HTTP call should fire if session_id (thread_id) or owner_id is missing."""
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        node = make_context_node(client, agent_id=AGENT_ID)
        # Missing thread_id in config
        assert await node({"customer_id": OWNER_ID}, _config(thread_id=None)) == {}
        # Missing customer_id in state
        assert await node({}, _config()) == {}


@respx.mock
async def test_node_swallows_errors(caplog) -> None:
    """A 500 from Beacon must not break the user-facing run."""
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        node = make_context_node(client, agent_id=AGENT_ID)
        delta = await node({"customer_id": OWNER_ID}, _config())

    assert delta == {}
    # Verify that the error-swallow path logged something at WARNING+ level
    assert any(
        rec.levelname in ("WARNING", "ERROR", "CRITICAL") for rec in caplog.records
    ), "Expected a WARNING/ERROR log entry for 500 response"


@respx.mock
async def test_node_reads_thread_id_from_state_when_configured() -> None:
    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "y"})
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        node = make_context_node(
            client, agent_id=AGENT_ID, session_id_from_config=False
        )
        delta = await node(
            {"customer_id": OWNER_ID, "thread_id": CONV_ID},
            None,
        )

    assert "messages" in delta


@respx.mock
async def test_node_accepts_deprecated_conversation_id_kwargs_with_warning() -> None:
    """Pre-0.2.0 kwargs (conversation_id_*) still work, emit DeprecationWarning,
    and forward to the new names."""
    import warnings

    respx.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={"summary": "y"})
    )
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            node = make_context_node(
                client,
                agent_id=AGENT_ID,
                conversation_id_from_config=False,
                conversation_id_key="thread_id",
            )
        kinds = [w.category for w in captured]
        assert DeprecationWarning in kinds
        # Both deprecated kwargs were passed; expect two warnings.
        assert sum(1 for k in kinds if k is DeprecationWarning) == 2
        delta = await node(
            {"customer_id": OWNER_ID, "thread_id": CONV_ID},
            None,
        )
    assert "messages" in delta


async def test_node_rejects_both_old_and_new_kwarg_for_same_knob() -> None:
    """Passing both the deprecated and the new name for the same knob raises."""
    import pytest

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with pytest.raises(TypeError, match="session_id_from_config"):
            make_context_node(
                client,
                agent_id=AGENT_ID,
                session_id_from_config=True,
                conversation_id_from_config=True,
            )
        with pytest.raises(TypeError, match="session_id_key"):
            make_context_node(
                client,
                agent_id=AGENT_ID,
                session_id_key="thread_id",
                conversation_id_key="thread_id",
            )
