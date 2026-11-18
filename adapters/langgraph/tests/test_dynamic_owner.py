# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Dynamic per-session ``owner_id``.

A multi-tenant server constructs one adapter and passes a per-request
``owner_id`` to ``stream()`` / ``run()``. Resolution order per turn:

1. the ``owner_id=`` passed to that turn (most specific — wins),
2. else the constructor ``owner_id`` (the single-tenant default),
3. else ``ValueError`` — nothing to scope Beacon traffic to.

The owner also flows into: the per-turn POST /context ``owner_id`` param, the
per-``(session_id, owner)`` manager pool, and the owner-less read paths (the
recall tool / context seed) that only receive a ``session_id`` from the
``RunnableConfig``.

The autouse ``_reset_shared_policy_state`` fixture (conftest) clears the
process-shared policy cache between cases.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, Any, TypedDict

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentic_context_langgraph import LangGraphAdapter

from .conftest import BASE_URL, CONTEXT_URL, mock_beacon_endpoints


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph():
    model = FakeListChatModel(responses=["hi there"])

    async def chat(state: _State) -> dict:
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile()


def _conn_kwargs(**overrides):
    base = {
        "agent_id": "bot",
        "api_key": "tok",
        "base_url": BASE_URL,
        "dataspace": "default",
    }
    base.update(overrides)
    return base


def _mount(respx_mock) -> Any:
    """Mount the lifecycle routes + a POST /context route; return the route."""
    mock_beacon_endpoints(respx_mock)
    return respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json={})
    )


# ─── resolution order ────────────────────────────────────────────────


@respx.mock
async def test_per_call_owner_overrides_constructor(respx_mock):
    """A per-turn ``owner_id=`` wins over the constructor default and scopes
    the POST /context to that tenant."""
    get_route = _mount(respx_mock)
    adapter = LangGraphAdapter(
        graph=_build_graph(), **_conn_kwargs(owner_id="ctor-owner")
    )
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1", owner_id="turn-owner"):
            pass

    assert json.loads(get_route.calls.last.request.content)["owner_id"] == "turn-owner"


@respx.mock
async def test_constructor_owner_used_when_no_per_call(respx_mock):
    """With no per-turn owner, the constructor default scopes the traffic."""
    get_route = _mount(respx_mock)
    adapter = LangGraphAdapter(
        graph=_build_graph(), **_conn_kwargs(owner_id="ctor-owner")
    )
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1"):
            pass

    assert json.loads(get_route.calls.last.request.content)["owner_id"] == "ctor-owner"


@respx.mock
async def test_missing_owner_everywhere_raises(respx_mock):
    """No constructor owner AND no per-turn owner → ValueError at turn time,
    not a silent post to a blank owner."""
    _mount(respx_mock)
    adapter = LangGraphAdapter(graph=_build_graph(), **_conn_kwargs())
    async with adapter:
        with pytest.raises(ValueError, match="owner_id is required"):
            async for _ in adapter.stream("hi", "ses-1"):
                pass


@respx.mock
async def test_empty_string_per_call_owner_falls_back_to_constructor(respx_mock):
    """An empty-string per-turn owner counts as "not supplied" and falls back
    to the constructor default (empty owner is a wire-level footgun)."""
    get_route = _mount(respx_mock)
    adapter = LangGraphAdapter(
        graph=_build_graph(), **_conn_kwargs(owner_id="ctor-owner")
    )
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1", owner_id=""):
            pass

    assert json.loads(get_route.calls.last.request.content)["owner_id"] == "ctor-owner"


@respx.mock
async def test_run_forwards_owner_id(respx_mock):
    """``run()`` forwards ``owner_id=`` to ``stream()`` (it's a thin wrapper)."""
    get_route = _mount(respx_mock)
    adapter = LangGraphAdapter(
        graph=_build_graph(), **_conn_kwargs(owner_id="ctor-owner")
    )
    async with adapter:
        await adapter.run("hi", session_id="ses-1", owner_id="turn-owner")

    assert json.loads(get_route.calls.last.request.content)["owner_id"] == "turn-owner"


# ─── per-(session, owner) isolation ──────────────────────────────────


@respx.mock
async def test_managers_keyed_by_session_and_owner(respx_mock):
    """The same ``session_id`` driven for two owners gets two managers, keyed
    by ``(session_id, owner)`` — two tenants never share a manager."""
    _mount(respx_mock)
    adapter = LangGraphAdapter(graph=_build_graph(), **_conn_kwargs())
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1", owner_id="tenant-a"):
            pass
        async for _ in adapter.stream("hi", "ses-1", owner_id="tenant-b"):
            pass

        assert ("ses-1", "tenant-a") in adapter._owned_managers
        assert ("ses-1", "tenant-b") in adapter._owned_managers
        assert len(adapter._owned_managers) == 2


@respx.mock
async def test_end_session_drops_all_owners_for_session(respx_mock):
    """``end_session(sid)`` evicts every manager for that session_id regardless
    of which owner(s) drove it, plus the session→owner mapping."""
    _mount(respx_mock)
    adapter = LangGraphAdapter(graph=_build_graph(), **_conn_kwargs())
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1", owner_id="tenant-a"):
            pass
        async for _ in adapter.stream("hi", "ses-1", owner_id="tenant-b"):
            pass

        await adapter.end_session("ses-1")

        assert not any(k[0] == "ses-1" for k in adapter._owned_managers)
        assert "ses-1" not in adapter._session_owner
        assert "ses-1" not in adapter._turn_idx


@respx.mock
async def test_per_owner_session_first_state_isolated(respx_mock):
    """SessionFirstPolicy fetches once per (owner, session). Two owners on the
    same session_id through ONE adapter each get their own first-turn fetch."""
    get_route = _mount(respx_mock)
    adapter = LangGraphAdapter(graph=_build_graph(), **_conn_kwargs())
    async with adapter:
        # tenant-a: two turns → one fetch (session_first).
        async for _ in adapter.stream("hi", "ses-1", owner_id="tenant-a"):
            pass
        async for _ in adapter.stream("hi again", "ses-1", owner_id="tenant-a"):
            pass
        # tenant-b on the SAME session_id → its own first fetch.
        async for _ in adapter.stream("hi", "ses-1", owner_id="tenant-b"):
            pass

    assert get_route.call_count == 2
    owners = [json.loads(c.request.content)["owner_id"] for c in get_route.calls]
    assert set(owners) == {"tenant-a", "tenant-b"}


@respx.mock
async def test_concurrent_same_session_different_owner_never_cross_posts(respx_mock):
    """Two turns for the SAME ``session_id`` under DIFFERENT owners, streamed
    concurrently, must each POST their entries under their OWN ``owner_id``.

    This is the regression guard for the tenant-misroute fix: ``_session_owner``
    holds a single slot per ``session_id``, so while both turns are parked on
    the (awaited) POST /context the slot gets overwritten by whichever turn
    resolved its owner last. If the ingest path re-derived the owner from that
    shared slot instead of carrying the turn-resolved owner, both turns would
    route through the same manager and one tenant's message would be POSTed
    under the other's ``conversationOwnerId``.

    A two-party barrier on the context GET forces the interleave
    deterministically: neither turn proceeds to its ingest until BOTH have
    resolved their owner and clobbered the shared slot.
    """
    mock_beacon_endpoints(respx_mock)

    barrier = asyncio.Barrier(2)

    async def _gated_context(request: httpx.Request) -> httpx.Response:
        # Park here until both concurrent turns have arrived — by then the
        # shared _session_owner["ses-1"] slot has been written twice and holds
        # only the last writer's owner.
        await barrier.wait()
        return httpx.Response(200, json={})

    respx_mock.post(CONTEXT_URL).mock(
        side_effect=_gated_context
    )
    entries_route = respx_mock.post(
        re.compile(
            rf"^{re.escape(BASE_URL)}/api/v1/agentic/conversations/[^/]+/entries$"
        )
    ).mock(return_value=httpx.Response(200, json={"accepted": 1, "rejected": 0, "errors": []}))

    adapter = LangGraphAdapter(graph=_build_graph(), **_conn_kwargs())

    async def _drive(owner: str) -> None:
        async for _ in adapter.stream(f"hi from {owner}", "ses-1", owner_id=owner):
            pass

    async with adapter:
        await asyncio.gather(_drive("tenant-a"), _drive("tenant-b"))

        # Two managers, one per tenant — the shared slot never collapsed them.
        # (Assert inside the context: __aexit__/close() drains and clears the
        # manager pool.)
        assert ("ses-1", "tenant-a") in adapter._owned_managers
        assert ("ses-1", "tenant-b") in adapter._owned_managers
        assert len(adapter._owned_managers) == 2

    # Every add_entries body must carry the owner matching its own message —
    # no entry POSTed under the wrong tenant's conversationOwnerId.
    seen_owners = set()
    for call in entries_route.calls:
        body = json.loads(call.request.content)
        owner = body["conversationOwnerId"]
        seen_owners.add(owner)
        for entry in body["entries"]:
            content = entry.get("content")
            if content and content.startswith("hi from "):
                assert content == f"hi from {owner}", (
                    f"user message {content!r} POSTed under wrong owner {owner!r}"
                )
                # The user entry's participant is its own turn's owner, not the
                # last writer of the shared slot — else Beacon marks it
                # role="participant" in the owner's own conversation.
                assert entry["participantId"] == owner
                assert entry["role"] == "owner"
        # The owner_id request header must agree with the body too.
        assert call.request.headers["owner_id"] == owner
    assert seen_owners == {"tenant-a", "tenant-b"}


# ─── owner-less read paths recover the tenant ────────────────────────


@respx.mock
async def test_recall_tool_uses_session_owner(respx_mock):
    """After a per-turn owner drives ``ses-1``, the recall tool (which only sees
    a session_id via RunnableConfig) recovers that tenant and reads *that
    tenant's* cached payload — not the constructor default or any other owner.

    The tool itself is cache-only (no GET), so this validates owner recovery
    through the one route it still has: the ``_SHARED_POLICY_STATE`` key
    ``(owner, agent_id, session_id)``. The stream's own GET must still have
    carried ``owner_id=turn-owner`` so the cache lands under the right key."""
    turn_payload = {"userContext": [{"text": "turn-owner-fact"}]}
    mock_beacon_endpoints(respx_mock)
    get_route = respx_mock.post(CONTEXT_URL).mock(
        return_value=httpx.Response(200, json=turn_payload)
    )
    adapter = LangGraphAdapter(graph=_build_graph(), **_conn_kwargs(owner_id="ctor-owner"))
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1", owner_id="turn-owner"):
            pass

        # The stream's GET was scoped to turn-owner (precondition for the
        # cache entry below to be keyed under turn-owner).
        assert json.loads(get_route.calls.last.request.content)["owner_id"] == "turn-owner"
        gets_before_tool = get_route.call_count

        tool = adapter.make_context_retrieval_tool()
        result = await tool.ainvoke(
            {},
            config={"configurable": {"thread_id": "ses-1"}},
        )

    # The tool recovered turn-owner's cache entry (not ctor-owner's).
    assert result == turn_payload
    # And it did so WITHOUT firing another GET — the recall tool is cache-only.
    assert get_route.call_count == gets_before_tool


@pytest.mark.parametrize(
    "recorded, sid, expected",
    [
        # A session the adapter never streamed resolves to the constructor owner.
        ({}, "never-seen", "ctor-owner"),
        ({}, None, "ctor-owner"),
        # Once ``stream()`` records a session's owner, the read paths recover it.
        ({"ses-1": "tenant-x"}, "ses-1", "tenant-x"),
    ],
    ids=["never-seen-falls-back", "none-falls-back", "recorded-owner"],
)
def test_owner_for_session_resolution(recorded, sid, expected):
    adapter = LangGraphAdapter(
        graph=_build_graph(), **_conn_kwargs(owner_id="ctor-owner")
    )
    adapter._session_owner.update(recorded)
    assert adapter._owner_for_session(sid) == expected


# ─── manager-injection mode rejects per-call owner ───────────────────


async def test_manager_injection_rejects_per_call_owner():
    """The manager carries its own owner in path-2; a per-turn owner_id= is a
    wiring error there and must raise rather than be silently ignored."""
    from unittest.mock import AsyncMock

    from agentic_context import AsyncSessionManager

    mgr = AsyncMock(spec=AsyncSessionManager)
    mgr.owner_id = "mgr-owner"
    adapter = LangGraphAdapter(
        agent_id="bot", session_manager=mgr, graph=_build_graph()
    )
    with pytest.raises(ValueError, match="only valid in self-driving mode"):
        async for _ in adapter.stream("hi", "ses-1", owner_id="turn-owner"):
            pass
