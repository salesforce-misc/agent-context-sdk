# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Cross-tenant isolation for the on-demand recall tool (cache-only).

Regression for the read-path hazard: the recall tool used to call
``get_context`` with the owner resolved from the process-shared
``_session_owner`` map (one slot per ``session_id``). Two concurrent turns on
the *same* ``session_id`` but different owners could overwrite that slot
mid-``await``, so one tenant's recall could read another tenant's memory.

Under the current cache-only tool there is no live GET at all — the tool
reads whatever the adapter's policy-driven fetch stored in
``_SHARED_POLICY_STATE[(owner, agent_id, session_id)]``. The *same hazard*
still applies at the cache lookup: if the tool resolved owner from the
shared map, interleaved turns would read the wrong tenant's cached payload.

The fix is the same shape as before — the tool threads the turn's owner
through the per-invocation ``RunnableConfig`` (``configurable.agentic_owner_id``,
alongside ``thread_id``) so each tool call recovers *its own* run's tenant,
race-free. These tests pin that behaviour against the cache-only read:

- the config owner wins over the ``_session_owner`` map (the whole point);
- two same-session/different-owner reads each hit their own cache entry even
  when interleaved;
- the map is used only as a fallback when no owner rode in on the config.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

from agentic_context_langgraph import LangGraphAdapter

from .conftest import BASE_URL, _build_graph


def _make_adapter(**overrides: Any) -> LangGraphAdapter:
    kwargs: dict[str, Any] = {
        "graph": _build_graph(),
        "agent_id": "bot",
        "owner_id": "default@o.com",
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
    payload: dict[str, Any],
) -> None:
    """Seed the process-shared policy state for (owner, adapter.agent_id, sid)."""
    cls = type(adapter)
    key = (owner, adapter.agent_id, sid)
    with cls._SHARED_POLICY_LOCK:
        cls._SHARED_POLICY_STATE[key] = {
            "last_payload": payload,
            "last_fetched_at": datetime.now(timezone.utc),
            "last_user_text": None,
            "turn_idx": 1,
        }


async def _invoke(tool, *, sid: str, owner: str | None) -> Any:
    """Invoke the tool the way LangGraph's ToolNode does — zero args, owner rides on config."""
    configurable: dict[str, Any] = {"thread_id": sid}
    if owner is not None:
        configurable["agentic_owner_id"] = owner
    return await tool.ainvoke({}, config={"configurable": configurable})


# ─── no GET regardless of owner (sanity) ─────────────────────────────


async def test_tool_never_calls_get_context_even_with_retrieval_wired() -> None:
    """Baseline for the whole file: the cache-only tool must not touch the
    retrieval client, so any isolation story below is isolation of CACHE READS,
    not of GET /context calls."""
    client = AsyncMock()
    client.get_context = AsyncMock(return_value={"userContext": [{"entry": "x"}]})
    adapter = _make_adapter()
    adapter._owned_retrieval_client = client

    _seed_cache(
        adapter, owner="alice@o.com", sid="ses-shared",
        payload={"userContext": [{"entry": "alice"}]},
    )
    result = await _invoke(
        adapter.make_context_retrieval_tool(), sid="ses-shared", owner="alice@o.com"
    )

    client.get_context.assert_not_awaited()
    assert result == {"userContext": [{"entry": "alice"}]}


# ─── config owner wins over stale map (the main hazard) ──────────────


async def test_config_owner_overrides_stale_session_owner_map() -> None:
    """The owner threaded through the invocation config wins over the
    process-shared ``_session_owner`` map — even when the map holds a
    *different* owner for the same session (the cross-tenant hazard)."""
    adapter = _make_adapter()
    # Simulate another turn/instance having stamped a DIFFERENT owner for this
    # very session into the shared map.
    adapter._session_owner["ses-shared"] = "victim@o.com"

    # Seed separate cache entries under each owner so a wrong-owner cache
    # lookup would surface the *other* tenant's payload.
    _seed_cache(
        adapter, owner="victim@o.com", sid="ses-shared",
        payload={"userContext": [{"entry": "victim-secret"}]},
    )
    _seed_cache(
        adapter, owner="caller@o.com", sid="ses-shared",
        payload={"userContext": [{"entry": "caller-fact"}]},
    )

    result = await _invoke(
        adapter.make_context_retrieval_tool(), sid="ses-shared", owner="caller@o.com"
    )

    assert result == {"userContext": [{"entry": "caller-fact"}]}
    # And crucially — the victim payload MUST NOT leak.
    assert "victim-secret" not in str(result)


# ─── interleaved reads stay isolated ─────────────────────────────────


async def test_concurrent_same_session_different_owner_reads_isolated() -> None:
    """Two interleaved reads on the same session but different owners each hit
    their own cache entry. A single-slot map would let one clobber the other."""
    adapter = _make_adapter()
    _seed_cache(
        adapter, owner="alice@o.com", sid="ses-shared",
        payload={"userContext": [{"entry": "alice-fact"}]},
    )
    _seed_cache(
        adapter, owner="bob@o.com", sid="ses-shared",
        payload={"userContext": [{"entry": "bob-fact"}]},
    )
    tool = adapter.make_context_retrieval_tool()

    async def _read(owner: str) -> dict[str, Any]:
        # Yield control so the two coroutines genuinely interleave — this is
        # exactly the window in which a shared single-slot map gets clobbered.
        await asyncio.sleep(0)
        return await _invoke(tool, sid="ses-shared", owner=owner)

    r_alice, r_bob = await asyncio.gather(_read("alice@o.com"), _read("bob@o.com"))

    assert r_alice == {"userContext": [{"entry": "alice-fact"}]}
    assert r_bob == {"userContext": [{"entry": "bob-fact"}]}


# ─── fallback to the map when no owner rode in ───────────────────────


async def test_session_owner_map_used_only_as_fallback() -> None:
    """With no owner on the config (direct/non-graph invocation), the tool
    falls back to ``_owner_for_session(sid)`` — which reads the per-session
    map. Preserves the pre-existing behaviour for callers that don't thread
    an owner."""
    adapter = _make_adapter()
    adapter._session_owner["ses-1"] = "mapped@o.com"
    _seed_cache(
        adapter, owner="mapped@o.com", sid="ses-1",
        payload={"userContext": [{"entry": "mapped-fact"}]},
    )
    # Also seed a decoy entry under the constructor default owner so the only
    # way to return "mapped-fact" is to resolve through the session map.
    _seed_cache(
        adapter, owner="default@o.com", sid="ses-1",
        payload={"userContext": [{"entry": "default-fact"}]},
    )

    result = await _invoke(adapter.make_context_retrieval_tool(), sid="ses-1", owner=None)

    assert result == {"userContext": [{"entry": "mapped-fact"}]}
