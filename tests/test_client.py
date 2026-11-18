# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for IngestionClient and AsyncIngestionClient (lifecycle endpoints).

Behaviour shared by both clients runs once per client through the ``http``
fixture in conftest.py.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from agentic_context import (
    AsyncIngestionClient,
    AsyncSessionManager,
    IngestionClient,
    IngestionError,
    Record,
    SessionManager,
)

from .conftest import _Http

BASE_URL = "https://beacon.test.example.com"
API_KEY = "test-secret"
OWNER_ID = "user-001"
AGENT_ID = "agent-support-bot"
CONV_ID = "conv-test-001"
SESS_ID = "sess-test-001"

# Sync/async client classes for the shared ``http`` fixture (conftest.py).
HTTP_CLIENTS = (IngestionClient, AsyncIngestionClient)

CONVERSATIONS_URL = BASE_URL + "/api/v1/agentic/conversations"


@pytest.mark.parametrize(
    ("client_kwargs", "expected_dataspace"),
    [({}, "default"), ({"dataspace": "my-ds"}, "my-ds")],
    ids=["default-dataspace", "custom-dataspace"],
)
async def test_create_conversation(
    http: _Http, client_kwargs: dict, expected_dataspace: str
) -> None:
    http.add("POST", CONVERSATIONS_URL, 201, json={"conversationId": CONV_ID})

    async with http.client(**client_kwargs) as client:
        resp = await http.run(client.create_conversation(CONV_ID, OWNER_ID, AGENT_ID))

    assert resp["conversationId"] == CONV_ID
    sent = http.calls[0]
    assert sent.headers["Authorization"] == f"Bearer {API_KEY}"
    assert sent.headers["dataspace"] == expected_dataspace
    assert sent.headers["owner_id"] == OWNER_ID
    assert sent.headers["agent_id"] == AGENT_ID
    body = sent.json()
    assert body["conversationId"] == CONV_ID
    assert body["conversationOwnerId"] == OWNER_ID
    assert body["ownerType"] == "individual"


async def test_create_session(http: _Http) -> None:
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/sessions", 201, json={"sessionId": SESS_ID})

    async with http.client() as client:
        resp = await http.run(client.create_session(CONV_ID, SESS_ID, OWNER_ID, AGENT_ID))

    assert resp["sessionId"] == SESS_ID
    body = http.calls[0].json()
    assert body["sessionId"] == SESS_ID
    assert body["conversationOwnerId"] == OWNER_ID
    # Only the owner goes in create_session; agent is added separately via add_participants
    participants = body["participants"]
    assert len(participants) == 1
    assert participants[0]["type"] == "individual"
    assert participants[0]["role"] == "owner"
    assert participants[0]["id"] == OWNER_ID  # plain owner id: governance scopes by it


async def test_add_participants(http: _Http) -> None:
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/sessions/{SESS_ID}/participants", json={})

    agent_participant = {
        "id": AGENT_ID,
        "type": "agent",
        "role": "participant",
        "startTime": "2026-06-17T00:00:00+00:00",
    }
    async with http.client() as client:
        await http.run(
            client.add_participants(CONV_ID, SESS_ID, OWNER_ID, AGENT_ID, [agent_participant])
        )

    body = http.calls[0].json()
    assert body["participants"][0]["id"] == AGENT_ID
    assert body["participants"][0]["type"] == "agent"
    assert body["participants"][0]["role"] == "participant"


async def test_add_entries(http: _Http) -> None:
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/entries", json={"accepted": 2})

    entries = [
        {
            "id": "msg-001",
            "interactionId": "turn-001",
            "participantId": "user-001",
            "content": "hello",
            "type": "user_message",
            "timestamp": "2026-01-01T12:00:00+00:00",
            "category": "Message",
        },
        {
            "id": "msg-002",
            "interactionId": "turn-001",
            "participantId": "agent-support-bot",
            "content": "Hi! How can I help?",
            "type": "agent_response",
            "timestamp": "2026-01-01T12:00:01+00:00",
            "category": "Message",
        },
    ]

    async with http.client() as client:
        resp = await http.run(client.add_entries(CONV_ID, SESS_ID, entries, OWNER_ID, AGENT_ID))

    assert resp["accepted"] == 2
    body = http.calls[0].json()
    assert body["sessionId"] == SESS_ID
    assert body["conversationOwnerId"] == OWNER_ID
    assert len(body["entries"]) == 2
    assert body["entries"][0]["type"] == "user_message"
    assert body["entries"][1]["type"] == "agent_response"


async def test_close_session_and_conversation(http: _Http) -> None:
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/sessions/{SESS_ID}/close", json={})
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/close", json={})

    async with http.client() as client:
        assert await http.run(client.close_session(CONV_ID, SESS_ID, OWNER_ID, AGENT_ID)) == {}
        assert await http.run(client.close_conversation(CONV_ID, OWNER_ID, AGENT_ID)) == {}


async def test_raises_on_non_2xx(http: _Http) -> None:
    http.add("POST", CONVERSATIONS_URL, 400, json={"error": "bad request"})

    async with http.client() as client:
        with pytest.raises(IngestionError) as exc:
            await http.run(client.create_conversation(CONV_ID, OWNER_ID, AGENT_ID))

    assert exc.value.status_code == 400


async def test_non_json_2xx_body_is_success(http: _Http) -> None:
    """A 2xx with a non-JSON body means the write landed. Raising would make
    the retry layer re-send it and write duplicate entries."""
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/entries", 201, body=b"Created")

    async with http.client() as client:
        resp = await http.run(client.add_entries(CONV_ID, SESS_ID, [], OWNER_ID, AGENT_ID))

    assert resp == {}
    assert len(http.calls) == 1


@pytest.mark.parametrize("body", [b"null", b'"OK"', b"[1]"])
async def test_non_object_json_2xx_body_returns_empty_dict(http: _Http, body: bytes) -> None:
    http.add("POST", CONVERSATIONS_URL, 201, body=body)

    async with http.client() as client:
        resp = await http.run(client.create_conversation(CONV_ID, OWNER_ID, AGENT_ID))

    assert resp == {}


async def test_non_json_2xx_through_manager_posts_entries_once(http: _Http) -> None:
    http.add("POST", CONVERSATIONS_URL, 201, json={})
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/sessions", 201, json={})
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/sessions/{CONV_ID}/participants", 201, json={})
    http.add("POST", f"{CONVERSATIONS_URL}/{CONV_ID}/entries", 201, body=b"OK")
    record = Record(
        event_id="evt-1", session_id=CONV_ID, agent_id=AGENT_ID,
        event_type="user_message", content="hi",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    callbacks = MagicMock()

    async with http.client() as client:
        cls = AsyncSessionManager if http.is_async else SessionManager
        mgr = cls(
            client, agent_id=AGENT_ID, owner_id=OWNER_ID, session_id=CONV_ID,
            callbacks=callbacks,
        )
        await http.run(mgr.ingest([record]))
        await http.run(mgr.close())

    assert sum(c.url.endswith("/entries") for c in http.calls) == 1
    # The async client raised a plain ValueError (not retried) and reported
    # the delivered batch as failed; both paths must report success.
    callbacks.on_success.assert_called_once()
    callbacks.on_error.assert_not_called()
    callbacks.on_retry.assert_not_called()


async def test_url_path_segments_are_encoded(http: _Http) -> None:
    """A conversation_id containing '/' or '..' must NOT be allowed to traverse the URL."""
    nasty_conv = "../admin"
    encoded = "..%2Fadmin"
    http.add("POST", f"{CONVERSATIONS_URL}/{encoded}/entries", json={"accepted": 0})

    async with http.client() as client:
        await http.run(client.add_entries(nasty_conv, SESS_ID, [], OWNER_ID, AGENT_ID))

    assert len(http.calls) == 1
    assert encoded in http.calls[0].url


async def test_base_url_trailing_slash_normalized(http: _Http) -> None:
    async with http.client(base_url=BASE_URL + "/") as client:
        assert client.base_url == BASE_URL


@pytest.mark.parametrize("client_cls", HTTP_CLIENTS, ids=["sync", "async"])
@pytest.mark.parametrize(
    "base_url",
    ["beacon.example.com", "ftp://beacon.example.com"],
    ids=["no-scheme", "unsupported-scheme"],
)
def test_base_url_without_supported_scheme_raises(client_cls: type, base_url: str) -> None:
    """A bare hostname (no scheme) must raise — never silently send bearer to the wrong place."""
    with pytest.raises(ValueError, match="must start with 'https://'"):
        client_cls(base_url=base_url, api_key=API_KEY)


# ── 401 retry (async client only) ────────────────────────────────────────────

def _async_return(value):
    f = asyncio.get_event_loop().create_future()
    f.set_result(value)
    return f


@respx.mock
async def test_post_retries_once_on_401_with_auth() -> None:
    NEW_TOKEN = "refreshed-token"
    route = respx.post(CONVERSATIONS_URL).mock(
        side_effect=[
            httpx.Response(401, text="Unauthorized"),
            httpx.Response(201, json={"conversationId": CONV_ID}),
        ]
    )

    auth = MagicMock()
    auth.invalidate = MagicMock()
    auth.aget_token = MagicMock(return_value=_async_return((NEW_TOKEN, BASE_URL)))

    async with AsyncIngestionClient(base_url=BASE_URL, api_key=API_KEY, auth=auth) as client:
        resp = await client.create_conversation(CONV_ID, OWNER_ID, AGENT_ID)

    assert resp["conversationId"] == CONV_ID
    assert auth.invalidate.call_count == 1
    assert client.api_key == NEW_TOKEN
    assert route.call_count == 2


@respx.mock
async def test_post_raises_on_401_without_auth() -> None:
    respx.post(CONVERSATIONS_URL).mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )

    async with AsyncIngestionClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with pytest.raises(IngestionError) as exc:
            await client.create_conversation(CONV_ID, OWNER_ID, AGENT_ID)

    assert exc.value.status_code == 401
