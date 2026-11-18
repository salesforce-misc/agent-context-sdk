# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""On-demand ``recall_customer_context`` tool for the LangGraph adapter.

The tool is a **zero-argument cache read**: it returns the payload the
adapter's policy-driven fetch already seeded into ``_SHARED_POLICY_STATE``.
It does NOT issue a ``GET /context`` of its own, so a reflexive model calling
the tool every turn costs zero extra Beacon traffic.

Session identity is resolved from the per-invocation ``RunnableConfig``
(``configurable.thread_id`` == ``session_id``), so the tool is invoked here via
``tool.ainvoke({}, config={"configurable": {"thread_id": sid}})`` — the same
path LangGraph's ToolNode/bind_tools use — with ``_last_session_id`` as a
best-effort fallback for direct invocation.

The autouse ``_reset_shared_policy_state`` fixture (conftest) clears the
process-shared policy cache between cases.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from agentic_context_langgraph import LangGraphAdapter

from .conftest import BASE_URL, _build_graph, mock_beacon_endpoints

_ADAPTER_LOGGER = "agentic_context_langgraph.adapter"


def _make_adapter(**overrides: Any) -> LangGraphAdapter:
    """Self-driving adapter for tool tests."""
    kwargs: dict[str, Any] = {
        "graph": _build_graph(),
        "agent_id": "bot",
        "owner_id": "u@o.com",
        "api_key": "tok",
        "base_url": BASE_URL,
        "dataspace": "default",
    }
    kwargs.update(overrides)
    return LangGraphAdapter(**kwargs)


def _seed_cache(
    adapter: LangGraphAdapter,
    *,
    owner: str,
    sid: str,
    payload: dict[str, Any] | None,
) -> None:
    """Seed the process-shared policy state as if the policy fetch had run."""
    cls = type(adapter)
    key = (owner, adapter.agent_id, sid)
    with cls._SHARED_POLICY_LOCK:
        cls._SHARED_POLICY_STATE[key] = {
            "last_payload": payload,
            "last_fetched_at": datetime.now(timezone.utc),
            "last_user_text": None,
            "turn_idx": 1,
        }


async def _invoke(tool: StructuredTool, *, sid: str | None = "ses-1") -> Any:
    """Invoke the tool the way ToolNode/bind_tools does — zero args, config carries thread_id."""
    config = {"configurable": {"thread_id": sid}} if sid is not None else {}
    return await tool.ainvoke({}, config=config)


# ─── tool shape + docstring ──────────────────────────────────────────


@pytest.mark.parametrize("mode", ["auto", "per-turn", "manual"], ids=["auto", "per-turn", "manual"])
def test_make_context_retrieval_tool_returns_structured_tool(mode: str) -> None:
    """make_context_retrieval_tool() is unconditional; only default_tools() is gated
    (and it's ungated in every self-driving mode)."""
    adapter = _make_adapter(context_mode=mode)
    tool = adapter.make_context_retrieval_tool()
    assert isinstance(tool, StructuredTool)
    assert isinstance(tool, BaseTool)
    assert tool.coroutine is not None
    assert tool.coroutine.__name__ == "recall_customer_context"


def test_tool_llm_surface_is_zero_arg_and_describes_cache_only_behavior() -> None:
    """Zero-argument tool — ``config`` is auto-injected and hidden; no user args.
    The LLM-visible docstring makes clear this is a cache read, not a retrieval."""
    adapter = _make_adapter()
    tool = adapter.make_context_retrieval_tool()
    assert tool.args == {}
    doc = (tool.coroutine.__doc__ or "").lower()
    assert "customer" in doc
    assert "memory" in doc
    # Cache-only semantics must be visible to the LLM so it understands it is
    # safe to call reflexively and doesn't expect fresh retrieval each turn.
    assert "cache" in doc or "no new retrieval" in doc or "already" in doc


# ─── cache-only recall ───────────────────────────────────────────────


async def test_returns_cached_payload_from_shared_state() -> None:
    """The tool reads whatever the policy-driven fetch wrote to _SHARED_POLICY_STATE."""
    adapter = _make_adapter()
    payload = {"userContext": [{"entry": "plan=Pro"}]}
    _seed_cache(adapter, owner="u@o.com", sid="ses-1", payload=payload)
    tool = adapter.make_context_retrieval_tool()

    assert await _invoke(tool, sid="ses-1") == payload


_NOT_SEEDED = object()


@pytest.mark.parametrize(
    "payload",
    [_NOT_SEEDED, {"summary": "x", "userContext": []}, None],
    ids=["cache-miss", "cached-empty-user-context", "cached-none-payload"],
)
async def test_empty_cache_states_return_empty_list(payload: Any) -> None:
    """Each empty state returns the clean ``{"userContext": []}`` shape, never raises:
    no cached payload for this (owner, agent, session); a cached payload with an
    empty userContext; a cache entry whose last_payload is None (fetch returned
    nothing)."""
    adapter = _make_adapter()
    if payload is not _NOT_SEEDED:
        _seed_cache(adapter, owner="u@o.com", sid="ses-1", payload=payload)
    tool = adapter.make_context_retrieval_tool()
    assert await _invoke(tool, sid="ses-1") == {"userContext": []}


async def test_manual_mode_cache_miss_falls_through_to_get_context() -> None:
    """Under context_mode='manual' the cache is never seeded, so the tool
    MUST issue a single on-demand get_context POST on cache miss or it would
    be permanently empty.
    """
    adapter = _make_adapter(context_mode="manual")
    client = AsyncMock()
    fetched = {"userContext": [{"entry": "from-server"}]}
    client.get_context = AsyncMock(return_value=fetched)
    adapter._owned_retrieval_client = client
    adapter._session_owner["ses-1"] = "u@o.com"
    tool = adapter.make_context_retrieval_tool()

    result = await _invoke(tool, sid="ses-1")

    client.get_context.assert_awaited_once()
    _, kwargs = client.get_context.call_args
    assert kwargs["conversation_id"] == "ses-1"
    assert kwargs["owner_id"] == "u@o.com"
    assert kwargs["agent_id"] == adapter.agent_id
    # Server schema is additionalProperties: false — the fallback must not send
    # any mode/query/latency/budget fields.
    for forbidden in ("mode", "query", "latency_sensitivity", "token_budget"):
        assert forbidden not in kwargs
    assert result == fetched


async def test_auto_mode_cache_miss_does_not_fall_through() -> None:
    """Under self-seeding modes (auto/per-turn) a cache miss stays an empty
    read — we must not introduce extra POST traffic on reflexive tool calls.
    """
    adapter = _make_adapter(context_mode="auto")
    client = AsyncMock()
    client.get_context = AsyncMock(return_value={"userContext": [{"entry": "x"}]})
    adapter._owned_retrieval_client = client
    adapter._session_owner["ses-1"] = "u@o.com"
    tool = adapter.make_context_retrieval_tool()

    result = await _invoke(tool, sid="ses-1")

    client.get_context.assert_not_awaited()
    assert result == {"userContext": []}


async def test_manual_mode_prefers_cache_when_seeded() -> None:
    """If something seeded the cache (manual retrieval wiring), the tool reads
    that payload instead of issuing another on-demand POST.
    """
    adapter = _make_adapter(context_mode="manual")
    client = AsyncMock()
    client.get_context = AsyncMock(return_value={"userContext": [{"entry": "server"}]})
    adapter._owned_retrieval_client = client
    _seed_cache(
        adapter,
        owner="u@o.com",
        sid="ses-1",
        payload={"userContext": [{"entry": "seeded"}]},
    )
    tool = adapter.make_context_retrieval_tool()

    result = await _invoke(tool, sid="ses-1")

    client.get_context.assert_not_awaited()
    assert result == {"userContext": [{"entry": "seeded"}]}


async def test_manual_mode_fallback_failure_degrades_to_empty() -> None:
    """A failing fallback POST in manual mode must not raise — it degrades to empty."""
    adapter = _make_adapter(context_mode="manual")
    client = AsyncMock()
    client.get_context = AsyncMock(side_effect=httpx.ConnectError("boom"))
    adapter._owned_retrieval_client = client
    adapter._session_owner["ses-1"] = "u@o.com"
    tool = adapter.make_context_retrieval_tool()

    assert await _invoke(tool, sid="ses-1") == {"userContext": []}


async def test_tool_return_is_isolated_from_shared_cache() -> None:
    """A caller that mutates the tool result must not corrupt the shared cache.
    A pure cache read: back-to-back invocations are byte-identical."""
    adapter = _make_adapter()
    original = {"userContext": [{"entry": "plan=Pro"}]}
    _seed_cache(adapter, owner="u@o.com", sid="ses-1", payload=original)
    tool = adapter.make_context_retrieval_tool()

    first = await _invoke(tool, sid="ses-1")
    assert first == original
    first["userContext"].append({"entry": "injected"})
    first["userContext"][0]["entry"] = "mutated"

    second = await _invoke(tool, sid="ses-1")
    assert second == {"userContext": [{"entry": "plan=Pro"}]}


# ─── session_id resolution ───────────────────────────────────────────


async def test_sid_falls_back_to_last_session_id() -> None:
    """No thread_id in config → fall back to the adapter's _last_session_id."""
    adapter = _make_adapter()
    adapter._last_session_id = "ses-fallback"
    _seed_cache(
        adapter,
        owner="u@o.com",
        sid="ses-fallback",
        payload={"userContext": [{"entry": "x"}]},
    )
    tool = adapter.make_context_retrieval_tool()

    result = await _invoke(tool, sid=None)  # no thread_id in config

    assert result == {"userContext": [{"entry": "x"}]}


async def test_no_sid_anywhere_returns_empty() -> None:
    """No config thread_id and no _last_session_id → empty, no raise."""
    adapter = _make_adapter()
    adapter._last_session_id = None
    tool = adapter.make_context_retrieval_tool()

    assert await _invoke(tool, sid=None) == {"userContext": []}


# ─── audit logging ───────────────────────────────────────────────────


@pytest.mark.parametrize("seeded", [True, False], ids=["cache-hit", "cache-miss"])
async def test_audit_log_records_owner_session_and_has_context(caplog, seeded: bool) -> None:
    """Exactly one audit line per cache read, attributing owner + session and
    logging a has_context bool only — never the retrieved memory text. A cache
    miss still logs exactly one line, with has_context=False."""
    secret = "PROFILE_FACT[1]\n  card_last_four=4242 plan=Enterprise"
    adapter = _make_adapter()
    if seeded:
        _seed_cache(
            adapter,
            owner="u@o.com",
            sid="ses-1",
            payload={"userContext": [{"entry": secret}]},
        )
    tool = adapter.make_context_retrieval_tool()

    with caplog.at_level(logging.INFO, logger=_ADAPTER_LOGGER):
        await _invoke(tool, sid="ses-1")

    audit = [r for r in caplog.records if "recall_customer_context:" in r.getMessage()]
    assert len(audit) == 1, "expected exactly one audit log for the cache read"
    msg = audit[0].getMessage()
    assert "u@o.com" in msg  # owner attribution
    assert "ses-1" in msg  # session attribution
    assert f"has_context={seeded}" in msg
    assert "4242" not in msg
    assert "PROFILE_FACT" not in msg


# ─── default_tools ───────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["auto", "per-turn", "manual"], ids=["auto", "per-turn", "manual"])
def test_default_tools_has_tool_in_every_self_driving_mode(mode: str) -> None:
    """In every self-driving mode, default_tools() exposes both the recall tool
    (cache read) and the pack-call tool (mid-turn POST for data not in cache)."""
    adapter = _make_adapter(context_mode=mode)
    tools = adapter.default_tools()
    assert isinstance(tools, list) and len(tools) == 2
    assert all(isinstance(t, StructuredTool) for t in tools)
    names = {t.coroutine.__name__ for t in tools}
    assert names == {"recall_customer_context", "get_context_pack"}


# ─── context_mode → instruction ───────────────────────────────────────


@pytest.mark.parametrize("mode", ["auto", "per-turn"], ids=["auto", "per-turn"])
def test_builtin_instruction_injected(mode: str) -> None:
    from agentic_context_langgraph.adapter import _RECALL_TOOL_SYSTEM_INSTRUCTION

    adapter = _make_adapter(context_mode=mode)
    out = adapter._maybe_inject_recall_tool_instruction({"messages": [("human", "hi")]})
    first = out["messages"][0]
    assert isinstance(first, SystemMessage)
    assert first.content == _RECALL_TOOL_SYSTEM_INSTRUCTION
    assert first.id == "agentic-memory-recall-tool-instruction"
    # Existing messages are preserved after the injected instruction.
    assert out["messages"][1] == ("human", "hi")


def test_manual_mode_disables_injection() -> None:
    """manual: tool registered, but the adapter injects no prompt (customer owns)."""
    adapter = _make_adapter(context_mode="manual")
    inputs = {"messages": [("human", "hi")]}
    out = adapter._maybe_inject_recall_tool_instruction(inputs)
    # No SystemMessage prepended — inputs pass through untouched.
    assert out["messages"] == [("human", "hi")]
    # ...but the tools are still installed (manual mode still gets both levers).
    assert len(adapter.default_tools()) == 2


@respx.mock
async def test_recall_tool_instruction_prepended_ahead_of_context_in_stream(
    respx_mock,
) -> None:
    """End-to-end: in a real stream() the recall-tool instruction sits ahead of
    the injected context SystemMessage (both before the user turn)."""
    from .conftest import _build_capturing_graph

    captured: list[list[Any]] = []
    graph = _build_capturing_graph(captured)

    mock_beacon_endpoints(respx_mock)
    respx_mock.post(BASE_URL + "/api/v1/agentic/context").mock(
        return_value=httpx.Response(
            200, json={"userContext": [{"text": "Owns Pro plan"}]}
        )
    )
    adapter = LangGraphAdapter(
        graph=graph,
        agent_id="bot",
        owner_id="user-1",
        api_key="tok",
        base_url=BASE_URL,
        dataspace="default",
        context_mode="auto",
    )
    async with adapter:
        async for _ in adapter.stream("hi", "ses-1"):
            pass

    system_msgs = [
        m for m in captured[0]
        if isinstance(m, SystemMessage) and m.id != "agentic-context-pack-list"
    ]
    # Two system messages: the recall-tool instruction (first) then the context block.
    assert len(system_msgs) == 2
    assert system_msgs[0].id == "agentic-memory-recall-tool-instruction"
    assert system_msgs[1].id == "agentic-memory-context"
