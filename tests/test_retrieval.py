# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for RetrievalClient and AsyncRetrievalClient.

Behaviour shared by both clients runs once per client through the ``http``
fixture in conftest.py.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import responses as responses_lib
import respx
from requests import exceptions as requests_exceptions

from agentic_context import AsyncRetrievalClient, RetrievalClient
from agentic_context.client import IngestionError

from .conftest import CONTEXT_URL, _Http

BASE_URL  = "https://beacon.example.com"
API_KEY   = "test-token"
DATASPACE = "default"
CONV_ID   = "conv-abc-123"
OWNER_ID  = "user-001"
AGENT_ID  = "my-agent"
PACK_LIST_URL = BASE_URL + "/api/v1/agentic/runtime-context-packs"


# Sync/async client classes for the shared ``http`` fixture (conftest.py).
HTTP_CLIENTS = (RetrievalClient, AsyncRetrievalClient)


# ─── get_context / get_conversation / list_conversations ────────────────────

async def test_get_context_correct_headers_and_body(http: _Http) -> None:
    http.add("POST", CONTEXT_URL, json={"summary": "user prefers morning meetings"})

    async with http.client(dataspace=DATASPACE) as rc:
        result = await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert result == {"summary": "user prefers morning meetings"}
    req = http.calls[0]
    # Must use ctx-dataspace-ds_name header, NOT x-sfdc-dataspace
    assert req.headers["ctx-dataspace-ds_name"] == DATASPACE
    # agent_id is sent as a header on every /context call (parity with
    # ingestion), in addition to the field in the body.
    assert req.headers["agent_id"] == AGENT_ID
    assert "ctx-agent-id" not in req.headers
    assert "x-sfdc-dataspace" not in req.headers
    body = req.json()
    assert body["conversation_id"] == CONV_ID
    assert body["owner_id"] == OWNER_ID
    assert body["agent_id"] == AGENT_ID
    assert body["owner_type"] == "individual"
    # Body is exactly the four base keys — no extra fields leak onto the wire
    # (the merged ``GetContextRequest`` schema is ``additionalProperties:
    # false``).
    assert set(body) == {"conversation_id", "owner_id", "agent_id", "owner_type"}


async def test_get_context_returns_empty_dict_on_empty_body(http: _Http) -> None:
    http.add("POST", CONTEXT_URL, body=b"")

    async with http.client() as rc:
        result = await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert result == {}


async def test_non_json_2xx_body_raises(http: _Http) -> None:
    """A 2xx HTML page (proxy, wrong base_url) must not read as "no context".
    It raises once, without a retry."""
    http.add("POST", CONTEXT_URL, body=b"<html>login</html>", content_type="text/html")

    async with http.client() as rc:
        with pytest.raises(IngestionError, match="non-JSON response body") as exc:
            await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert exc.value.status_code == 200
    assert "text/html" in str(exc.value)
    assert len(http.calls) == 1


@pytest.mark.parametrize("body", [b"null", b'"OK"', b"7"])
async def test_non_container_json_2xx_body_returns_empty_dict(http: _Http, body: bytes) -> None:
    http.add("POST", CONTEXT_URL, body=body)

    async with http.client() as rc:
        result = await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert result == {}


async def test_get_conversation_correct_headers_and_params(http: _Http) -> None:
    url = BASE_URL + f"/api/v1/agentic/conversations/{CONV_ID}"
    http.add("GET", url, json={"entries": [{"id": "e1"}]})
    http.add("GET", url, json={})

    async with http.client(dataspace=DATASPACE) as rc:
        result = await http.run(rc.get_conversation(CONV_ID, OWNER_ID, AGENT_ID))
        await http.run(rc.get_conversation(CONV_ID, OWNER_ID, AGENT_ID, scope="TRACES"))

    assert result == {"entries": [{"id": "e1"}]}
    req = http.calls[0]
    # Must use x-sfdc-dataspace header, NOT ctx-dataspace-ds_name
    assert req.headers["x-sfdc-dataspace"] == DATASPACE
    assert req.headers["agent_id"] == AGENT_ID
    assert "ctx-dataspace-ds_name" not in req.headers
    assert "scope=all" in req.url
    assert f"owner_id={OWNER_ID}" in req.url
    # Custom scope is forwarded verbatim.
    assert "scope=TRACES" in http.calls[1].url


async def test_list_conversations_correct_headers_and_params(http: _Http) -> None:
    http.add("GET", BASE_URL + "/api/v1/agentic/conversations",
             json={"conversations": [{"id": CONV_ID}]})

    async with http.client(dataspace=DATASPACE) as rc:
        result = await http.run(rc.list_conversations(OWNER_ID, AGENT_ID))

    # list_conversations returns a flat list (normalized from response wrapper)
    assert result == [{"id": CONV_ID}]
    req = http.calls[0]
    assert req.headers["x-sfdc-dataspace"] == DATASPACE
    assert req.headers["agent_id"] == AGENT_ID
    assert f"owner_id={OWNER_ID}" in req.url
    assert "owner_type=individual" in req.url


# ─── base_url validation ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("cls", "url"),
    [(RetrievalClient, "ftp://bad.example.com"), (AsyncRetrievalClient, "bad-url")],
    ids=["sync-unsupported-scheme", "async-no-scheme"],
)
def test_client_rejects_invalid_base_url(cls, url) -> None:
    with pytest.raises(ValueError, match="must start with"):
        cls(base_url=url, api_key=API_KEY)


def test_sync_client_strips_trailing_slash() -> None:
    with RetrievalClient(base_url=BASE_URL + "/", api_key=API_KEY) as rc:
        assert rc.base_url == BASE_URL


# ─── URL encoding (conversation_id with special chars) ──────────────────────

@responses_lib.activate
def test_get_conversation_url_encodes_conversation_id() -> None:
    """Verify conversation_id with slashes/spaces is URL-encoded in the path."""
    special_id = "conv/with spaces&special"
    # The response mock needs to match the encoded URL
    from urllib.parse import quote
    encoded_id = quote(special_id, safe='')
    responses_lib.add(
        responses_lib.GET,
        BASE_URL + f"/api/v1/agentic/conversations/{encoded_id}",
        json={"entries": []},
        status=200,
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        result = rc.get_conversation(special_id, OWNER_ID, AGENT_ID)
    assert result == {"entries": []}


# ─── list_conversations normalization ───────────────────────────────────────

@responses_lib.activate
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # API returns a bare list — should pass through unchanged.
        ([{"id": "conv-1"}, {"id": "conv-2"}], [{"id": "conv-1"}, {"id": "conv-2"}]),
        # API wraps in {"data": [...]} — should unwrap.
        ({"data": [{"id": "conv-1"}]}, [{"id": "conv-1"}]),
        # API returns {} — should return empty list.
        ({}, []),
    ],
    ids=["bare-list", "data-wrapper", "empty-dict"],
)
def test_list_conversations_normalizes_response(payload, expected) -> None:
    responses_lib.add(
        responses_lib.GET,
        BASE_URL + "/api/v1/agentic/conversations",
        json=payload,
        status=200,
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        result = rc.list_conversations(OWNER_ID, AGENT_ID)
    assert result == expected


# ─── SessionManager retrieval integration ───────────────────────────────────

@responses_lib.activate
def test_session_manager_get_context_delegates_to_retrieval_client() -> None:
    """SessionManager.get_context() delegates to RetrievalClient."""
    from agentic_context import SessionManager, IngestionClient

    responses_lib.add(
        responses_lib.POST,
        CONTEXT_URL,
        json={"summary": "user likes coffee"},
        status=200,
    )

    ingestion = MagicMock(spec=IngestionClient)
    retrieval = RetrievalClient(base_url=BASE_URL, api_key=API_KEY)

    mgr = SessionManager(
        ingestion,
        agent_id=AGENT_ID,
        owner_id=OWNER_ID,
        session_id=CONV_ID,
        retrieval_client=retrieval,
    )
    result = mgr.get_context()
    assert result == {"summary": "user likes coffee"}
    retrieval.close()


def test_session_manager_get_context_returns_none_without_retrieval() -> None:
    """SessionManager.get_context() returns None if no retrieval client."""
    from agentic_context import SessionManager, IngestionClient

    ingestion = MagicMock(spec=IngestionClient)
    mgr = SessionManager(
        ingestion,
        agent_id=AGENT_ID,
        owner_id=OWNER_ID,
        session_id=CONV_ID,
    )
    assert mgr.get_context() is None


def test_session_manager_get_context_returns_none_on_error() -> None:
    """SessionManager.get_context() swallows errors and returns None."""
    from agentic_context import SessionManager, IngestionClient

    ingestion = MagicMock(spec=IngestionClient)
    retrieval = MagicMock(spec=RetrievalClient)
    retrieval.get_context.side_effect = IngestionError(500, "boom")

    mgr = SessionManager(
        ingestion,
        agent_id=AGENT_ID,
        owner_id=OWNER_ID,
        session_id=CONV_ID,
        retrieval_client=retrieval,
    )
    assert mgr.get_context() is None


# ─── read-path retry on transient 5xx ────────────────────────────────────────
#
# The read path (POST /agentic/context — the ADK/Claude adapters' before-model
# Fast Context read) must retry transient failures with the same backoff policy
# the ingestion + auth paths use. Before this, a single transient 500 from the
# Context Service surfaced as "no context" — a blank, non-personalised greeting.
# These tests pin: transient 5xx recovers, persistent 5xx raises after the retry
# budget is exhausted, and a non-retryable 4xx raises immediately (no retries).

@pytest.fixture
def no_backoff_sleep(monkeypatch):
    """Neutralise backoff delay so retry tests don't sleep for real seconds."""
    from agentic_context import _retry
    monkeypatch.setattr(_retry, "backoff", lambda attempt: 0.0)


async def test_get_context_retries_transient_5xx_then_succeeds(http: _Http, no_backoff_sleep) -> None:
    """Two transient 5xx (500, then 502) then a 200 → the 200 body is returned; 3 calls made."""
    http.add("POST", CONTEXT_URL, 500, json={"error": "transient"})
    http.add("POST", CONTEXT_URL, 502, json={"error": "transient"})
    http.add("POST", CONTEXT_URL, json={"summary": "recovered after retry"})

    async with http.client() as rc:
        result = await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert result == {"summary": "recovered after retry"}
    assert len(http.calls) == 3


async def test_get_context_persistent_5xx_raises_after_retries(http: _Http, no_backoff_sleep) -> None:
    """A never-recovering 503 raises IngestionError after 1 + _MAX_RETRIES calls."""
    from agentic_context import _retry

    http.add("POST", CONTEXT_URL, 503, json={"error": "down"}, times=_retry._MAX_RETRIES + 1)

    async with http.client() as rc:
        with pytest.raises(IngestionError) as exc_info:
            await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert exc_info.value.status_code == 503
    assert len(http.calls) == _retry._MAX_RETRIES + 1


@pytest.mark.parametrize("status", [401, 404], ids=["401", "404"])
async def test_get_context_non_retryable_4xx_raises_immediately(
    http: _Http, no_backoff_sleep, status: int,
) -> None:
    """A 4xx is not retried — exactly one call, then IngestionError carrying the status."""
    http.add("POST", CONTEXT_URL, status, json={"error": "nope"})

    async with http.client() as rc:
        with pytest.raises(IngestionError) as exc_info:
            await http.run(rc.get_context(CONV_ID, OWNER_ID, AGENT_ID))

    assert exc_info.value.status_code == status
    assert len(http.calls) == 1


# ─── AsyncRetrievalClient: memory endpoints ──────────────────────────────────

@pytest.mark.asyncio
@respx.mock
async def test_async_list_memory_maps_owner_to_profile_params() -> None:
    """owner_id/owner_type map to the endpoint's profile_id/profile_type params."""
    route = respx.get(BASE_URL + "/api/v1/agentic/memory").mock(
        return_value=httpx.Response(200, json={"memories": [{"memoryId": "m1", "content": "likes tea"}]})
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        memories = await rc.list_memory(OWNER_ID, AGENT_ID, owner_type="employee", category="PROFILE_FACT")

    assert memories == [{"memoryId": "m1", "content": "likes tea"}]
    assert route.call_count == 1
    url = str(respx.calls[0].request.url)
    assert f"profile_id={OWNER_ID}" in url
    assert "profile_type=employee" in url
    assert "category=PROFILE_FACT" in url
    assert respx.calls[0].request.headers["agent_id"] == AGENT_ID


@pytest.mark.asyncio
@respx.mock
async def test_async_list_memory_paginates() -> None:
    """Every page is followed via nextPageToken and the entries are flattened."""
    respx.get(BASE_URL + "/api/v1/agentic/memory").mock(side_effect=[
        httpx.Response(200, json={"memories": [{"memoryId": "m1"}], "nextPageToken": "p2"}),
        httpx.Response(200, json={"memories": [{"memoryId": "m2"}]}),
    ])

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        memories = await rc.list_memory(OWNER_ID, AGENT_ID)

    assert [m["memoryId"] for m in memories] == ["m1", "m2"]
    assert "page_token=p2" in str(respx.calls[1].request.url)


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_memory_sends_ids_body_and_params() -> None:
    route = respx.delete(BASE_URL + "/api/v1/agentic/memory").mock(
        return_value=httpx.Response(200, json={"statusCode": "deleted", "memories": [{"memoryId": "m1"}]})
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        result = await rc.delete_memory(["m1", "m2"], OWNER_ID, AGENT_ID, owner_type="employee")

    assert result["statusCode"] == "deleted"
    assert route.call_count == 1
    req = respx.calls[0].request
    assert json.loads(req.content) == {"memories": [{"memoryId": "m1"}, {"memoryId": "m2"}]}
    assert f"profile_id={OWNER_ID}" in str(req.url)
    assert "profile_type=employee" in str(req.url)
    assert req.headers["agent_id"] == AGENT_ID


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_memory_empty_is_noop() -> None:
    """An empty id list is a no-op — the endpoint requires minItems 1."""
    route = respx.delete(BASE_URL + "/api/v1/agentic/memory").mock(
        return_value=httpx.Response(200, json={})
    )
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY) as rc:
        result = await rc.delete_memory([], OWNER_ID, AGENT_ID)

    assert result == {"statusCode": "deleted", "memories": []}
    assert route.call_count == 0


# ─── get_context_pack — POST GetContext (sync + async) ───────────────────────

async def test_get_context_pack_posts_body_and_context_headers(http: _Http) -> None:
    """Pack read POSTs the full body to /context with the ctx-dataspace header."""
    http.add("POST", CONTEXT_URL, json={"contextPack": "PlanRules", "detail": {"deductible": 500}})

    async with http.client(dataspace=DATASPACE) as rc:
        result = await http.run(rc.get_context_pack(
            CONV_ID, OWNER_ID, AGENT_ID, "PlanRules",
            params={"policy_type": "MEDICAL"}, owner_type="employee",
        ))

    assert result == {"contextPack": "PlanRules", "detail": {"deductible": 500}}
    assert len(http.calls) == 1
    req = http.calls[0]
    # Same header convention as GET /context (ctx-dataspace-ds_name), not x-sfdc-*.
    assert req.headers["ctx-dataspace-ds_name"] == DATASPACE
    assert req.headers["agent_id"] == AGENT_ID
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert req.json() == {
        "conversation_id": CONV_ID,
        "owner_id": OWNER_ID,
        "owner_type": "employee",
        "agent_id": AGENT_ID,
        "context_pack": "PlanRules",
        # owner_id is auto-filled into params so pack SQL that uses :owner_id
        # resolves even when the LLM omits it (the design rule: owner_id is
        # bound server-side and never taken from the model).
        "params": {"policy_type": "MEDICAL", "owner_id": OWNER_ID},
    }


async def test_get_context_pack_defaults_params_to_owner_id_only(http: _Http) -> None:
    """Omitting params still sends owner_id in params so pack SQL resolves."""
    http.add("POST", CONTEXT_URL, json={})

    async with http.client(dataspace=DATASPACE) as rc:
        await http.run(rc.get_context_pack(CONV_ID, OWNER_ID, AGENT_ID, "PlanRules"))

    body = http.calls[0].json()
    assert body["params"] == {"owner_id": OWNER_ID}
    assert body["owner_type"] == "individual"  # default


async def test_get_context_pack_trusted_owner_id_overrides_params(http: _Http) -> None:
    """owner_id in params is a trust boundary: the trusted argument always wins.

    Pack queries bind ``:owner_id``; if a model-populated params dict could
    overwrite the resolved identity, an LLM could request another user's data.
    Both clients must bind the owner from the trusted argument unconditionally.
    """
    http.add("POST", CONTEXT_URL, json={})

    async with http.client(dataspace=DATASPACE) as rc:
        await http.run(rc.get_context_pack(
            CONV_ID, OWNER_ID, AGENT_ID, "PlanRules",
            params={"owner_id": "attacker"},
        ))

    assert len(http.calls) == 1
    assert http.calls[0].json()["params"] == {"owner_id": OWNER_ID}


@pytest.mark.parametrize(
    ("method", "url", "status"),
    [("POST", CONTEXT_URL, 405), ("GET", PACK_LIST_URL, 404)],
    ids=["pack-405", "pack-list-404"],
)
async def test_context_pack_endpoints_unavailable_raise(http: _Http, method, url, status) -> None:
    """PREVIEW: where the pack POST op or the list endpoint is not deployed
    (404/405), the method raises IngestionError carrying the status."""
    http.add(method, url, status, json={})

    async with http.client(dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            if method == "POST":
                await http.run(rc.get_context_pack(CONV_ID, OWNER_ID, AGENT_ID, "PlanRules"))
            else:
                await http.run(rc.get_context_pack_list(AGENT_ID))
    assert exc_info.value.status_code == status


# ─── get_context_pack — context_pack_version (sync + async) ─────────────────

@pytest.mark.parametrize("version", [3, None], ids=["set", "none"])
async def test_get_context_pack_context_pack_version(http: _Http, version) -> None:
    """When context_pack_version is supplied, it rides at the top level of the body
    (not inside params) so Beacon can validate before running the query.
    Omitted version → no key in the body (not ``null``); an absent key tells
    Beacon to accept whatever revision is live."""
    http.add("POST", CONTEXT_URL, json={})

    async with http.client(dataspace=DATASPACE) as rc:
        await http.run(rc.get_context_pack(
            CONV_ID, OWNER_ID, AGENT_ID, "PlanRules", context_pack_version=version,
        ))

    assert len(http.calls) == 1
    body = http.calls[0].json()
    if version is None:
        assert "context_pack_version" not in body
    else:
        assert body["context_pack_version"] == version
    # version must NOT leak into params — that is the data-needs dict.
    assert "context_pack_version" not in body["params"]


# ─── get_context_pack_list (catalog read) ───────────────────────────────────

async def test_get_context_pack_list_sends_params_and_context_headers(http: _Http) -> None:
    http.add("GET", PACK_LIST_URL,
             json={"contextPacks": [{"contextPackId": "PlanRules", "description": "x"}], "hasMore": False})

    async with http.client(dataspace=DATASPACE) as rc:
        result = await http.run(rc.get_context_pack_list(AGENT_ID))

    assert result == {"contextPacks": [{"contextPackId": "PlanRules", "description": "x"}]}
    assert len(http.calls) == 1
    req = http.calls[0]
    assert req.headers["ctx-dataspace-ds_name"] == DATASPACE
    assert req.headers["agent_id"] == AGENT_ID
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert "ctx-agent-id" not in req.headers
    assert req.params["agentApiName"] == AGENT_ID
    assert req.params["pageSize"] == "10"
    assert req.params["page"] == "1"


@responses_lib.activate
def test_get_context_pack_list_overrides_page_and_page_size() -> None:
    responses_lib.add(
        responses_lib.GET, BASE_URL + "/api/v1/agentic/runtime-context-packs",
        json={"contextPacks": [], "hasMore": False}, status=200,
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        rc.get_context_pack_list(AGENT_ID, page_size=5, page=2)

    req = responses_lib.calls[0].request
    assert req.params["pageSize"] == "5"
    assert req.params["page"] == "2"


async def test_get_context_pack_list_paginates_on_has_more(http: _Http) -> None:
    """Every page is followed via hasMore and the list is flattened."""
    http.add("GET", PACK_LIST_URL, json={"contextPacks": [{"contextPackId": "PlanRules"}], "hasMore": True})
    http.add("GET", PACK_LIST_URL, json={"contextPacks": [{"contextPackId": "Claims"}], "hasMore": False})

    async with http.client(dataspace=DATASPACE) as rc:
        result = await http.run(rc.get_context_pack_list(AGENT_ID))

    assert [p["contextPackId"] for p in result["contextPacks"]] == ["PlanRules", "Claims"]
    assert http.calls[0].params["page"] == "1"
    assert http.calls[1].params["page"] == "2"


# ─── get_system_prompt ─────────────────────────────────────────

SYSTEM_PROMPT_URL = BASE_URL + "/api/v1/agentic/prompts/system/context_pack_instruction"


@responses_lib.activate
def test_get_system_prompt_returns_text_on_200() -> None:
    """200 text/plain → the body is returned verbatim as a str."""
    server_text = "SERVER CONTEXT PACK PROMPT -- call get_context_pack when ..."
    responses_lib.add(
        responses_lib.GET,
        SYSTEM_PROMPT_URL,
        body=server_text,
        status=200,
        content_type="text/plain",
    )

    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        result = rc.get_system_prompt("context_pack_instruction")

    assert result == server_text
    req = responses_lib.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    # Tenant-neutral endpoint: no dataspace / agent_id header is sent.
    assert "ctx-dataspace-ds_name" not in req.headers
    assert "x-sfdc-dataspace" not in req.headers
    assert "agent_id" not in req.headers


@responses_lib.activate
def test_get_system_prompt_returns_none_on_404() -> None:
    """404 is the "fallback" signal — returns None, does NOT raise."""
    responses_lib.add(
        responses_lib.GET,
        SYSTEM_PROMPT_URL,
        body="unknown prompt type",
        status=404,
        content_type="text/plain",
    )

    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        result = rc.get_system_prompt("context_pack_instruction")

    assert result is None
    assert len(responses_lib.calls) == 1  # 404 is not retried


@responses_lib.activate
def test_get_system_prompt_url_encodes_prompt_type() -> None:
    """Prompt type with unsafe characters is percent-encoded in the URL path."""
    responses_lib.add(
        responses_lib.GET,
        BASE_URL + "/api/v1/agentic/prompts/system/weird%2Fid",
        body="ok",
        status=200,
        content_type="text/plain",
    )

    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        result = rc.get_system_prompt("weird/id")

    assert result == "ok"


@responses_lib.activate
def test_get_system_prompt_raises_on_5xx(no_backoff_sleep) -> None:
    """Non-404 non-2xx raises IngestionError (after retries for 5xx)."""
    from agentic_context import _retry
    for _ in range(_retry._MAX_RETRIES + 1):
        responses_lib.add(
            responses_lib.GET,
            SYSTEM_PROMPT_URL,
            body="down",
            status=503,
            content_type="text/plain",
        )

    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            rc.get_system_prompt("context_pack_instruction")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_returns_text_on_200() -> None:
    server_text = "SERVER CONTEXT PACK PROMPT -- call get_context_pack when ..."
    route = respx.get(SYSTEM_PROMPT_URL).mock(
        return_value=httpx.Response(
            200, text=server_text, headers={"content-type": "text/plain"},
        )
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        result = await rc.get_system_prompt("context_pack_instruction")

    assert result == server_text
    assert route.call_count == 1
    req = respx.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert "ctx-dataspace-ds_name" not in req.headers
    assert "agent_id" not in req.headers


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_returns_none_on_404() -> None:
    respx.get(SYSTEM_PROMPT_URL).mock(
        return_value=httpx.Response(404, text="unknown prompt type")
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        result = await rc.get_system_prompt("context_pack_instruction")

    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_raises_on_5xx(no_backoff_sleep) -> None:
    respx.get(SYSTEM_PROMPT_URL).mock(
        return_value=httpx.Response(503, text="down")
    )

    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            await rc.get_system_prompt("context_pack_instruction")

    assert exc_info.value.status_code == 503


# ─── Hardened branches (code-review fixes) ────────────────────


@responses_lib.activate
def test_get_system_prompt_whitespace_body_raises_transient() -> None:
    """A 200 body of only whitespace is a transient failure (not a cacheable
    miss) — the next request might land on a healthy backend. Raise so
    the caller sees an error rather than a definitive miss."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="   \n\t  ", status=200, content_type="text/plain",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 200


@responses_lib.activate
def test_get_system_prompt_html_200_raises_transient() -> None:
    """Gateways sometimes return 200 + text/html (e.g. a login page). Raise —
    the next request may hit the real backend, so this is NOT cached."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="<html><body>Please log in</body></html>",
        status=200, content_type="text/html",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 200


@responses_lib.activate
@pytest.mark.parametrize("status", [404, 410])
def test_get_system_prompt_cacheable_miss_statuses(status: int) -> None:
    """404/410 are definitive ``no prompt here`` — return None (cacheable)
    so the caller pins the fallback. Narrower than the earlier draft:
    400/405/415 can come from a gateway or a rolling deploy and are
    transient, so they belong on the raise/short-negative-cache path."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="nope", status=status, content_type="text/plain",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") is None
    assert len(responses_lib.calls) == 1  # not retried


@responses_lib.activate
@pytest.mark.parametrize("status", [400, 405, 415])
def test_get_system_prompt_gateway_4xx_is_transient(no_backoff_sleep, status: int) -> None:
    """400/405/415 can originate at a gateway/WAF or mid-rollout and will
    self-heal, so they are not cacheable misses. They are also not retryable,
    so the call raises after one attempt."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="boom", status=status, content_type="text/plain",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == status
    assert len(responses_lib.calls) == 1  # non-retryable: single attempt


@responses_lib.activate
def test_get_system_prompt_retries_transport_error(no_backoff_sleep) -> None:
    """Transport errors (connection reset, DNS blip) hit the retry policy."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body=requests_exceptions.ConnectionError("boom"),
    )
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="OK", status=200, content_type="text/plain",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == "OK"
    assert len(responses_lib.calls) == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_retries_transport_error(no_backoff_sleep) -> None:
    route = respx.get(SYSTEM_PROMPT_URL).mock(
        side_effect=[
            httpx.ConnectError("blip"),
            httpx.Response(200, text="OK", headers={"content-type": "text/plain"}),
        ]
    )
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") == "OK"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_second_401_raises(no_backoff_sleep) -> None:
    """A 401 that persists after one refresh is a transient failure (token
    race / policy propagation), NOT a cacheable miss — raise rather than return None."""
    from agentic_context import ContextAuth

    auth = MagicMock(spec=ContextAuth)
    auth.aget_token = AsyncMock(return_value=(API_KEY, BASE_URL))
    auth.invalidate = MagicMock()

    respx.get(SYSTEM_PROMPT_URL).mock(
        side_effect=[httpx.Response(401), httpx.Response(401)]
    )
    async with AsyncRetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE, auth=auth,
    ) as rc:
        with pytest.raises(IngestionError) as exc_info:
            await rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 401
    auth.invalidate.assert_called_once()  # exactly one refresh, no loop


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_403_raises() -> None:
    """A 403 is a transient scope race (policy propagation), not a cacheable
    miss — raise rather than return None."""
    respx.get(SYSTEM_PROMPT_URL).mock(return_value=httpx.Response(403))
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            await rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 403


# ─── Sync 401-refresh branch (Finding 11) ───────────────────────


@responses_lib.activate
def test_get_system_prompt_sync_401_then_refresh_then_200() -> None:
    """Sync client: 401 triggers one in-call token refresh, retries succeed."""
    from agentic_context import ContextAuth

    auth = MagicMock(spec=ContextAuth)
    auth.get_token = MagicMock(return_value=("fresh-token", BASE_URL))
    auth.invalidate = MagicMock()

    responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=401)
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="HELLO", status=200, content_type="text/plain",
    )
    with RetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE, auth=auth,
    ) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == "HELLO"
    auth.invalidate.assert_called_once()
    auth.get_token.assert_called_once()


@responses_lib.activate
def test_get_system_prompt_sync_second_401_raises() -> None:
    """Sync client: a 401 that persists after refresh is transient — raise."""
    from agentic_context import ContextAuth

    auth = MagicMock(spec=ContextAuth)
    auth.get_token = MagicMock(return_value=(API_KEY, BASE_URL))
    auth.invalidate = MagicMock()

    responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=401)
    responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=401)
    with RetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE, auth=auth,
    ) as rc:
        with pytest.raises(IngestionError) as exc_info:
            rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 401
    auth.invalidate.assert_called_once()


@responses_lib.activate
def test_get_system_prompt_sync_401_refresh_does_not_consume_retry(no_backoff_sleep) -> None:
    """A 401 that gets a refresh must NOT consume a retry attempt. Three
    transient 503s fill attempts 0/1/2, a 401 triggers refresh (free attempt),
    and the final 200 on attempt 3 still succeeds."""
    from agentic_context import ContextAuth

    auth = MagicMock(spec=ContextAuth)
    auth.get_token = MagicMock(return_value=("fresh", BASE_URL))
    auth.invalidate = MagicMock()

    # 503, 503, 503, 401 (refresh, free), 200 → 5 physical requests.
    for _ in range(3):
        responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=503)
    responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=401)
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="FINALLY", status=200, content_type="text/plain",
    )

    with RetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE, auth=auth,
    ) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == "FINALLY"
    assert len(responses_lib.calls) == 5
    auth.invalidate.assert_called_once()


@responses_lib.activate
def test_get_system_prompt_rebuilds_auth_header_per_attempt(no_backoff_sleep) -> None:
    """Each attempt must rebuild Authorization from ``self.api_key`` so an
    out-of-band token refresh is picked up on retry — not a stale header
    captured once and reused."""
    responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=503)
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="OK", status=200, content_type="text/plain",
    )

    with RetrievalClient(
        base_url=BASE_URL, api_key="stale-token", dataspace=DATASPACE,
    ) as rc:
        orig_get = rc._session.get
        def _rotate_mid_flight(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            resp = orig_get(*args, **kwargs)
            if resp.status_code >= 500 and rc.api_key == "stale-token":
                # Simulate an out-of-band refresher updating the token
                # between the first (503) and second (200) attempts.
                rc.api_key = "fresh-token"
            return resp
        rc._session.get = _rotate_mid_flight  # type: ignore[method-assign]
        assert rc.get_system_prompt("context_pack_instruction") == "OK"

    assert len(responses_lib.calls) == 2
    first = responses_lib.calls[0].request
    second = responses_lib.calls[1].request
    assert first.headers["Authorization"] == "Bearer stale-token"
    # The retry MUST see the rotated key — if the implementation captured the
    # header dict once at loop start, this would still read "stale-token".
    assert second.headers["Authorization"] == "Bearer fresh-token"


@responses_lib.activate
def test_get_system_prompt_accepts_text_markdown() -> None:
    """A 200 with any non-text/html content type (e.g. text/markdown,
    text/plain, application/octet-stream, no content-type) is accepted
    verbatim — the HTML guard only rejects gateway login pages."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="# header\nbody", status=200, content_type="text/markdown",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == "# header\nbody"


@responses_lib.activate
def test_get_system_prompt_json_content_type_passes_through_verbatim() -> None:
    """The server returns ``text/plain`` (decisioning-pipeline
    ``SystemPromptProvider``); the SDK does not decode JSON. A body a gateway
    labels ``application/json`` is used as-is."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body='{"text": "x"}', status=200, content_type="application/json",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == '{"text": "x"}'


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_json_content_type_passes_through_verbatim() -> None:
    respx.get(SYSTEM_PROMPT_URL).mock(return_value=httpx.Response(
        200, text='{"text": "x"}', headers={"content-type": "application/json"},
    ))
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") == '{"text": "x"}'


# ─── SessionPrompt helper ───────────────────────────────────────────────
#
# Per-session pin used by every adapter for the server context-pack prompt.


def _prompt_client(*results: object) -> MagicMock:
    client = MagicMock()
    client.get_system_prompt = MagicMock(side_effect=list(results))
    return client


@pytest.mark.asyncio
async def test_session_prompt_pins_server_text_per_session() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client("SERVER")
    assert await sp.get("s1", client) == "SERVER"
    assert await sp.get("s1", client) == "SERVER"
    client.get_system_prompt.assert_called_once_with("context_pack_instruction")


@pytest.mark.asyncio
async def test_session_prompt_fetches_once_per_new_session() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client("V1", "V2")
    assert await sp.get("s1", client) == "V1"
    assert await sp.get("s2", client) == "V2"
    # s1 stays on the text it started with.
    assert await sp.get("s1", client) == "V1"
    assert client.get_system_prompt.call_count == 2


@pytest.mark.asyncio
async def test_session_prompt_miss_pins_fallback() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client(None, "SERVER")
    assert await sp.get("s1", client) == "FALLBACK"
    assert await sp.get("s1", client) == "FALLBACK"
    client.get_system_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_session_prompt_error_pins_fallback_and_next_session_retries(
    caplog, monkeypatch
) -> None:
    from agentic_context.retrieval import SessionPrompt

    monkeypatch.setattr(SessionPrompt, "FAILURE_COOLDOWN_S", 0.0)
    sp = SessionPrompt("FALLBACK")
    client = _prompt_client(RuntimeError("boom"), "SERVER")
    with caplog.at_level("WARNING", logger="agentic_context.retrieval"):
        assert await sp.get("session-abcdefgh-1234", client) == "FALLBACK"
    assert "boom" in caplog.text
    assert "session-abcdefgh-1234" not in caplog.text  # id is truncated
    # No retry within the session.
    assert await sp.get("session-abcdefgh-1234", client) == "FALLBACK"
    assert client.get_system_prompt.call_count == 1
    # A new session tries again.
    assert await sp.get("s2", client) == "SERVER"


@pytest.mark.asyncio
async def test_session_prompt_failure_cooldown_skips_fetch() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client(RuntimeError("boom"), "SERVER")
    assert await sp.get("s1", client) == "FALLBACK"
    # Within the cooldown a new session pins the fallback without fetching.
    assert await sp.get("s2", client) == "FALLBACK"
    assert sp.get_sync("s3", client) == "FALLBACK"
    assert client.get_system_prompt.call_count == 1
    # After the cooldown the next new session fetches again.
    assert sp._failed_at is not None
    sp._failed_at -= SessionPrompt.FAILURE_COOLDOWN_S
    assert await sp.get("s4", client) == "SERVER"
    assert client.get_system_prompt.call_count == 2


@pytest.mark.asyncio
async def test_session_prompt_without_client_returns_fallback_unpinned() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    assert await sp.get("s1", None) == "FALLBACK"
    assert sp.get_sync("s1", None) == "FALLBACK"
    # A later turn with a client still fetches.
    assert await sp.get("s1", _prompt_client("SERVER")) == "SERVER"


@pytest.mark.asyncio
async def test_session_prompt_runs_sync_client_off_the_loop() -> None:
    import threading

    from agentic_context.retrieval import SessionPrompt

    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fetch(_: str) -> str:
        seen.append(threading.get_ident())
        return "SERVER"

    client = MagicMock()
    client.get_system_prompt = fetch
    assert await SessionPrompt("FALLBACK").get("s1", client) == "SERVER"
    assert seen and seen[0] != loop_thread


@pytest.mark.asyncio
async def test_session_prompt_awaits_async_client() -> None:
    from agentic_context.retrieval import SessionPrompt

    client = MagicMock()
    client.get_system_prompt = AsyncMock(return_value="SERVER")
    assert await SessionPrompt("FALLBACK").get("s1", client) == "SERVER"
    client.get_system_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_prompt_awaits_awaitable_from_thread_path() -> None:
    """A sync-looking callable that returns a coroutine must be awaited,
    never pinned as the prompt string."""
    from agentic_context.retrieval import SessionPrompt

    async def _inner(_: str) -> str:
        return "SERVER"

    def sync_looking(prompt_type: str):  # noqa: ANN202
        return _inner(prompt_type)

    client = MagicMock()
    client.get_system_prompt = sync_looking
    sp = SessionPrompt("FALLBACK")
    assert await sp.get("s1", client) == "SERVER"
    assert await sp.get("s1", client) == "SERVER"


def test_session_prompt_get_sync_pins() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client("SERVER")
    assert sp.get_sync("s1", client) == "SERVER"
    assert sp.get_sync("s1", client) == "SERVER"
    client.get_system_prompt.assert_called_once()


def test_session_prompt_get_sync_error_pins_fallback() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client(RuntimeError("boom"), "SERVER")
    assert sp.get_sync("s1", client) == "FALLBACK"
    assert sp.get_sync("s1", client) == "FALLBACK"
    client.get_system_prompt.assert_called_once()


def test_session_prompt_get_sync_rejects_async_client() -> None:
    from agentic_context.retrieval import SessionPrompt

    async def fetch(_: str) -> str:
        return "SERVER"

    client = MagicMock()
    client.get_system_prompt = fetch
    sp = SessionPrompt("FALLBACK")
    with pytest.raises(RuntimeError, match="async client"):
        sp.get_sync("s1", client)
    # Nothing pinned: a correct async call still fetches.
    assert sp._pinned("s1") is None


def test_session_prompt_concurrent_first_turn_keeps_first_pin() -> None:
    """If another first turn of the same session pins while this fetch is in
    flight, the session keeps that answer instead of switching mid-session."""
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")

    def fetch_after_concurrent_pin(_: str) -> None:
        assert sp.get_sync("s1", _prompt_client("SERVER")) == "SERVER"
        raise RuntimeError("late failure")

    client = MagicMock()
    client.get_system_prompt = fetch_after_concurrent_pin
    assert sp.get_sync("s1", client) == "SERVER"
    assert sp._pinned("s1") == "SERVER"


def test_session_prompt_evicts_oldest_session() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK", max_sessions=2)
    client = _prompt_client("A", "B", "C", "A2")
    assert sp.get_sync("a", client) == "A"
    assert sp.get_sync("b", client) == "B"
    assert sp.get_sync("c", client) == "C"
    assert sp._pinned("a") is None
    assert sp._pinned("b") == "B"
    # An evicted session simply re-fetches.
    assert sp.get_sync("a", client) == "A2"


def test_session_prompt_evicts_least_recently_used() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK", max_sessions=2)
    client = _prompt_client("A", "B", "C")
    sp.get_sync("a", client)
    sp.get_sync("b", client)
    sp.get_sync("a", client)  # touch: "b" is now the oldest
    sp.get_sync("c", client)
    assert sp._pinned("a") == "A"
    assert sp._pinned("b") is None


def test_session_prompt_forget_refetches() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client("V1", "V2")
    assert sp.get_sync("s1", client) == "V1"
    sp.forget("s1")
    sp.forget("never-seen")  # no-op
    assert sp.get_sync("s1", client) == "V2"


def test_session_prompt_clear_forgets_every_session() -> None:
    from agentic_context.retrieval import SessionPrompt

    sp = SessionPrompt("FALLBACK")
    client = _prompt_client("A", "B")
    sp.get_sync("a", client)
    sp.get_sync("b", client)
    sp.clear()
    assert sp._pinned("a") is None
    assert sp._pinned("b") is None


@pytest.mark.parametrize(
    ("prompt", "text", "expected"),
    [
        (None, "SERVER", False),
        ("", "SERVER", False),
        ("base\n\nSERVER", "SERVER", True),
        ("base\n\nFALLBACK", "SERVER", True),
        ("base only", "SERVER", False),
    ],
)
def test_session_prompt_present_in(prompt: str | None, text: str, expected: bool) -> None:
    from agentic_context.retrieval import SessionPrompt

    assert SessionPrompt("FALLBACK").present_in(prompt, text) is expected


# ─── behaviours pinned by earlier review rounds ──────────────────────────


@responses_lib.activate
def test_get_system_prompt_204_is_transient_not_cacheable(no_backoff_sleep) -> None:
    """204 can come from a gateway mid-deploy; it raises rather than
    reporting a definitive miss like 404/410 do."""
    responses_lib.add(responses_lib.GET, SYSTEM_PROMPT_URL, status=204)
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError):
            rc.get_system_prompt("context_pack_instruction")


@responses_lib.activate
def test_get_context_honours_retry_after(monkeypatch) -> None:
    import agentic_context.retrieval as retrieval_mod

    slept: list[float] = []
    monkeypatch.setattr(retrieval_mod.time, "sleep", slept.append)
    responses_lib.add(
        responses_lib.POST, CONTEXT_URL, status=429, headers={"Retry-After": "7"},
    )
    responses_lib.add(responses_lib.POST, CONTEXT_URL, json={"ok": 1}, status=200)
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_context(CONV_ID, OWNER_ID, AGENT_ID) == {"ok": 1}
    assert slept == [7.0]


@responses_lib.activate
def test_get_conversation_honours_retry_after(monkeypatch) -> None:
    import agentic_context.retrieval as retrieval_mod

    slept: list[float] = []
    monkeypatch.setattr(retrieval_mod.time, "sleep", slept.append)
    url = BASE_URL + f"/api/v1/agentic/conversations/{CONV_ID}"
    responses_lib.add(responses_lib.GET, url, status=503, headers={"Retry-After": "3"})
    responses_lib.add(responses_lib.GET, url, json={"entries": []}, status=200)
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_conversation(CONV_ID, OWNER_ID, AGENT_ID) == {"entries": []}
    assert slept == [3.0]


@pytest.mark.asyncio
@respx.mock
async def test_async_get_context_honours_retry_after(monkeypatch) -> None:
    import agentic_context.retrieval as retrieval_mod

    sleep = AsyncMock()
    monkeypatch.setattr(retrieval_mod.asyncio, "sleep", sleep)
    respx.post(CONTEXT_URL).mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "5"}),
        httpx.Response(200, json={"ok": 1}),
    ])
    async with AsyncRetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE,
    ) as rc:
        assert await rc.get_context(CONV_ID, OWNER_ID, AGENT_ID) == {"ok": 1}
    sleep.assert_awaited_once_with(5.0)


# ─── Final-review gaps: UTF-8 parity, httpx transport errors, async twins ──

_NON_ASCII_PROMPT = "Use get_context_pack — don’t guess."


@responses_lib.activate
def test_get_system_prompt_charsetless_text_decodes_utf8() -> None:
    """requests defaults charset-less ``text/*`` to ISO-8859-1; the sync client
    must decode as UTF-8 to match httpx (mojibake would reach the model)."""
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body=_NON_ASCII_PROMPT.encode("utf-8"), status=200,
        headers={"content-type": "text/plain"},
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == _NON_ASCII_PROMPT


@responses_lib.activate
def test_get_system_prompt_explicit_charset_is_respected() -> None:
    body = "café"
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body=body.encode("latin-1"), status=200,
        headers={"content-type": "text/plain; charset=iso-8859-1"},
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == body


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_charsetless_text_decodes_utf8() -> None:
    respx.get(SYSTEM_PROMPT_URL).mock(return_value=httpx.Response(
        200, content=_NON_ASCII_PROMPT.encode("utf-8"),
        headers={"content-type": "text/plain"},
    ))
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") == _NON_ASCII_PROMPT


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("exc", [
    httpx.ReadError("reset"),
    httpx.RemoteProtocolError("server disconnected"),
])
async def test_async_get_system_prompt_retries_non_oserror_transport_error(
    no_backoff_sleep, exc: Exception,
) -> None:
    """httpx ReadError/RemoteProtocolError are not OSErrors, so the shared
    retry predicate misses them; the idempotent prompt GET retries anyway."""
    route = respx.get(SYSTEM_PROMPT_URL).mock(side_effect=[
        exc,
        httpx.Response(200, text="OK", headers={"content-type": "text/plain"}),
    ])
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") == "OK"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_410_is_miss() -> None:
    route = respx.get(SYSTEM_PROMPT_URL).mock(return_value=httpx.Response(410, text="gone"))
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") is None
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_html_200_raises_transient() -> None:
    respx.get(SYSTEM_PROMPT_URL).mock(return_value=httpx.Response(
        200, text="<html><body>Please log in</body></html>",
        headers={"content-type": "text/html"},
    ))
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            await rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 200


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("body", ["", "   \n\t  "])
async def test_async_get_system_prompt_empty_body_raises_transient(body: str) -> None:
    respx.get(SYSTEM_PROMPT_URL).mock(return_value=httpx.Response(
        200, text=body, headers={"content-type": "text/plain"},
    ))
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        with pytest.raises(IngestionError) as exc_info:
            await rc.get_system_prompt("context_pack_instruction")
    assert exc_info.value.status_code == 200


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_401_then_refresh_then_200() -> None:
    from agentic_context import ContextAuth

    auth = MagicMock(spec=ContextAuth)
    auth.aget_token = AsyncMock(return_value=("fresh-token", BASE_URL))
    auth.invalidate = MagicMock()

    route = respx.get(SYSTEM_PROMPT_URL).mock(side_effect=[
        httpx.Response(401),
        httpx.Response(200, text="HELLO", headers={"content-type": "text/plain"}),
    ])
    async with AsyncRetrievalClient(
        base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE, auth=auth,
    ) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") == "HELLO"
    auth.invalidate.assert_called_once()
    auth.aget_token.assert_awaited_once()
    assert route.calls[1].request.headers["Authorization"] == "Bearer fresh-token"


@responses_lib.activate
def test_get_system_prompt_honours_retry_after(monkeypatch) -> None:
    import agentic_context.retrieval as retrieval_mod

    slept: list[float] = []
    monkeypatch.setattr(retrieval_mod.time, "sleep", slept.append)
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL, status=503, headers={"Retry-After": "4"},
    )
    responses_lib.add(
        responses_lib.GET, SYSTEM_PROMPT_URL,
        body="OK", status=200, content_type="text/plain",
    )
    with RetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert rc.get_system_prompt("context_pack_instruction") == "OK"
    assert slept == [4.0]


@pytest.mark.asyncio
@respx.mock
async def test_async_get_system_prompt_honours_retry_after(monkeypatch) -> None:
    import agentic_context.retrieval as retrieval_mod

    sleep = AsyncMock()
    monkeypatch.setattr(retrieval_mod.asyncio, "sleep", sleep)
    respx.get(SYSTEM_PROMPT_URL).mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "6"}),
        httpx.Response(200, text="OK", headers={"content-type": "text/plain"}),
    ])
    async with AsyncRetrievalClient(base_url=BASE_URL, api_key=API_KEY, dataspace=DATASPACE) as rc:
        assert await rc.get_system_prompt("context_pack_instruction") == "OK"
    sleep.assert_awaited_once_with(6.0)


def test_patch_pack_version_keeps_description_and_data_needs():
    import json

    from agentic_context.retrieval import (
        CONTEXT_PACK_LIST_MARKER,
        patch_pack_version_in_catalog,
    )

    block = f"{CONTEXT_PACK_LIST_MARKER}\n" + json.dumps(
        {"P": {"description": "d", "dataNeeds": [1], "version": 1}}
    )
    patched = patch_pack_version_in_catalog(block, "P", 2)
    assert json.loads(patched.partition("\n")[2])["P"] == {
        "description": "d", "dataNeeds": [1], "version": 2,
    }
    dropped = patch_pack_version_in_catalog(block, "P", None)
    assert json.loads(dropped.partition("\n")[2])["P"] == {
        "description": "d", "dataNeeds": [1],
    }


def _pack_entry(entry, source_type="contextPack"):
    return {"userContext": [{"sourceType": source_type, "entry": entry}]}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_pack_entry("id:P\nversion:3\ndescription:d"), {"entry": "id:P\nversion:3\ndescription:d", "version": 3}),
        (_pack_entry("id:P\nversion: 4 "), {"entry": "id:P\nversion: 4 ", "version": 4}),
        # first version line wins
        (_pack_entry("version:2\nversion:9"), {"entry": "version:2\nversion:9", "version": 2}),
        # no version line, unparseable value, near-miss key: entry still surfaces
        (_pack_entry("id:P\ndescription:no version line"), {"entry": "id:P\ndescription:no version line", "version": None}),
        (_pack_entry("version:latest"), {"entry": "version:latest", "version": None}),
        (_pack_entry("versionX:5"), {"entry": "versionX:5", "version": None}),
        (_pack_entry(None), {"entry": None, "version": None}),
        # not a mismatch envelope
        ({"contextPacks": []}, None),
        (_pack_entry("id:user\nversion:1", source_type="profile"), None),
        ({"userContext": [{"sourceType": "contextPack", "entry": "version:1"}] * 2}, None),
        ({"userContext": []}, None),
        ({"userContext": "version:1"}, None),
        ({"userContext": ["version:1"]}, None),
        ("version:1", None),
        (None, None),
    ],
    ids=[
        "version", "version-whitespace", "first-version-wins", "no-version-line",
        "non-int-version", "near-miss-key", "non-str-entry", "normal-pack-payload",
        "other-source-type", "many-entries", "empty-list", "userContext-not-list",
        "item-not-dict", "payload-not-dict", "payload-none",
    ],
)
def test_version_mismatch_entry(payload, expected):
    from agentic_context.retrieval import version_mismatch_entry

    assert version_mismatch_entry(payload) == expected
