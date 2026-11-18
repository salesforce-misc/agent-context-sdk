# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Pack-list injection via ``OpenAIAdapter.instructions_provider``.

Parity with the Claude adapter's ``UserPromptSubmit`` hook: at session-open
the adapter fetches ``get_context_pack_list(agent_id=…)`` from Beacon,
caches the formatted ``[Available context packs]\\n<json>`` block, and
every subsequent turn's instructions callable appends it to the
customer's ``base`` string. The LLM sees which pack ids exist; the pack
tool's ``_allowed_pack_ids`` is populated for server-side validation.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from agentic_context import CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_LIST_MARKER
from agentic_context_openai import OpenAIAdapter

from .test_adapter import CREDS, mock_async_clients, mock_sync_clients  # noqa: F401


# ─── Shared pack-catalog response ─────────────────────────────────

PACK_CATALOG = {
    "contextPacks": [
        {
            "contextPackId": "PlanRules",
            "description": "Plan coverage rules",
            "dataNeeds": [
                {"name": "plan", "description": "plan doc", "parameters": []},
            ],
        },
        {
            "contextPackId": "Claims",
            "description": "Recent claims",
            "dataNeeds": [],
        },
    ],
}


# ─── First-turn fetch + caching ────────────────────────────────────

@pytest.mark.asyncio
async def test_fetches_on_first_turn_and_appends_block(mock_async_clients) -> None:
    """First turn fetches the catalog, appends the block to ``base`` and
    publishes ``_allowed_pack_ids``; later turns reuse the cached block."""
    adapter = OpenAIAdapter(agent_id="agent-A", owner_id="u@test", **CREDS)
    mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(
        return_value=PACK_CATALOG,
    )
    provider = adapter.instructions_provider(
        base="You are a helpful assistant.",
        session_id="conv-1",
    )
    assert adapter._allowed_pack_ids == set()  # empty until first fire

    out = await provider(run_context=None, agent=None)

    rc = mock_async_clients["retrieval"]
    rc.get_context_pack_list.assert_awaited_once_with(agent_id="agent-A")
    assert out.startswith("You are a helpful assistant.\n\n")
    assert CONTEXT_PACK_LIST_MARKER in out
    # The JSON tail round-trips.
    tail = out.split(CONTEXT_PACK_LIST_MARKER + "\n", 1)[1]
    parsed = json.loads(tail)
    assert set(parsed) == {"PlanRules", "Claims"}
    assert adapter._allowed_pack_ids == {"PlanRules", "Claims"}

    out2 = await provider(run_context=None, agent=None)
    out3 = await provider(run_context=None, agent=None)
    assert out == out2 == out3
    # Fetch fires exactly once — ``_catalog_fetched_sessions`` gates subsequent turns.
    rc.get_context_pack_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_distinct_sessions_each_trigger_their_own_fetch(mock_async_clients) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(
        return_value=PACK_CATALOG,
    )
    p1 = adapter.instructions_provider(base="B", session_id="conv-1")
    p2 = adapter.instructions_provider(base="B", session_id="conv-2")

    await p1(run_context=None, agent=None)
    await p2(run_context=None, agent=None)

    assert mock_async_clients["retrieval"].get_context_pack_list.await_count == 2


# ─── Degraded paths ───────────────────────────────────────────────

@pytest.mark.parametrize(
    ("fetch", "pinned", "expected_block", "expected_ids"),
    [
        # Fetch failed, but the integrator wired ``make_get_context_pack_tool``
        # with pinned ids first. The catalog block degrades to the pinned view
        # (empty description/dataNeeds) and the pack tool accepts those same
        # ids — the block and the tool agree.
        (
            AsyncMock(side_effect=RuntimeError("405 Method Not Allowed")),
            ["PlanRules", "Claims"],
            {"PlanRules": {"description": "", "dataNeeds": []},
             "Claims": {"description": "", "dataNeeds": []}},
            {"PlanRules", "Claims"},
        ),
        # Server advertises no packs AND no pack tool was constructed: the
        # block is ``{}`` (the LLM knows the catalog was consulted; the pack
        # tool, if somehow called, rejects every id).
        (AsyncMock(return_value={"contextPacks": []}), None, {}, set()),
        # Server advertises no packs but the integrator pinned ids: the block
        # shows the pinned view and the pack tool accepts the same ids.
        (
            AsyncMock(return_value={"contextPacks": []}),
            ["PlanRules"],
            {"PlanRules": {"description": "", "dataNeeds": []}},
            {"PlanRules"},
        ),
    ],
    ids=["fetch-failure-with-pinned", "empty-catalog-without-pinned", "empty-catalog-with-pinned"],
)
@pytest.mark.asyncio
async def test_degraded_catalog_falls_back_to_pinned(
    mock_async_clients, fetch, pinned, expected_block, expected_ids,
) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    mock_async_clients["retrieval"].get_context_pack_list = fetch
    if pinned is not None:
        adapter.make_get_context_pack_tool(pinned, session_id="conv-1")
    provider = adapter.instructions_provider(base="BASE", session_id="conv-1")

    out = await provider(run_context=None, agent=None)

    assert out.startswith("BASE\n\n")
    assert CONTEXT_PACK_LIST_MARKER in out
    tail = out.split(CONTEXT_PACK_LIST_MARKER + "\n", 1)[1]
    assert json.loads(tail) == expected_block
    if not expected_block:
        assert out == f"BASE\n\n{CONTEXT_PACK_LIST_MARKER}\n{{}}"
    assert adapter._allowed_pack_ids == expected_ids


@pytest.mark.asyncio
async def test_fetch_failure_marks_session_opened_no_retry(mock_async_clients) -> None:
    """Fetch failed, no pack tool was constructed, so no pinned ids to fall
    back to. The LLM still sees the marker (so it knows the catalog was
    consulted), with an empty map — the pack tool rejects every id.

    Mirrors the Claude adapter: a failed fetch is NOT retried on later
    turns. The catalog gate (``_catalog_fetched_sessions``) is independent
    of ``push()``'s fast-context seed gate, so either lifecycle path can
    run first without suppressing the other."""
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    fetch = AsyncMock(side_effect=RuntimeError("transient"))
    mock_async_clients["retrieval"].get_context_pack_list = fetch
    provider = adapter.instructions_provider(base="BASE", session_id="conv-1")

    out1 = await provider(run_context=None, agent=None)
    out2 = await provider(run_context=None, agent=None)
    out3 = await provider(run_context=None, agent=None)

    # Fetch fires exactly once; subsequent turns return the same (pinned-
    # or empty-) catalog block.
    assert fetch.await_count == 1
    assert out1 == out2 == out3
    assert out1 == f"BASE\n\n{CONTEXT_PACK_LIST_MARKER}\n{{}}"
    assert adapter._allowed_pack_ids == set()
    assert "conv-1" in adapter._catalog_fetched_sessions


@pytest.mark.asyncio
async def test_push_before_provider_does_not_suppress_catalog_fetch(
    mock_async_clients,
) -> None:
    """Regression: the fast-context seed gate (``_fast_context_seeded``)
    and the catalog gate (``_catalog_fetched_sessions``) must be
    independent. A prior revision shared one set, so ``push()``'s auto-seed
    would silently suppress the pack-catalog fetch on the subsequent
    ``instructions_provider`` call."""
    adapter = OpenAIAdapter(agent_id="agent-A", owner_id="u@test", **CREDS)
    rc = mock_async_clients["retrieval"]
    rc.get_context = AsyncMock(return_value={"items": []})
    rc.get_context_pack_list = AsyncMock(return_value=PACK_CATALOG)

    await adapter.push("conv-1", messages=[{"role": "user", "content": "hi"}])
    provider = adapter.instructions_provider(base="BASE", session_id="conv-1")
    out = await provider(run_context=None, agent=None)

    rc.get_context_pack_list.assert_awaited_once_with(agent_id="agent-A")
    assert CONTEXT_PACK_LIST_MARKER in out
    assert adapter._allowed_pack_ids == {"PlanRules", "Claims"}


@pytest.mark.asyncio
async def test_provider_before_push_does_not_suppress_fast_context_seed(
    mock_async_clients,
) -> None:
    """Regression: the reverse ordering — provider first, push second — must
    also preserve the push-side fast-context seed. The prior shared-set
    revision suppressed the seed when the provider ran first."""
    adapter = OpenAIAdapter(agent_id="agent-A", owner_id="u@test", **CREDS)
    rc = mock_async_clients["retrieval"]
    rc.get_context_pack_list = AsyncMock(return_value=PACK_CATALOG)
    seed_payload = {"items": [{"foo": "bar"}]}
    rc.get_context = AsyncMock(return_value=seed_payload)

    provider = adapter.instructions_provider(base="BASE", session_id="conv-1")
    await provider(run_context=None, agent=None)
    await adapter.push("conv-1", messages=[{"role": "user", "content": "hi"}])

    assert adapter._seeded_contexts.get("conv-1") == seed_payload
    assert "conv-1" in adapter._fast_context_seeded


def test_sync_mode_raises(mock_sync_clients) -> None:
    adapter = OpenAIAdapter(
        agent_id="a", owner_id="u@test", use_async=False, **CREDS,
    )
    with pytest.raises(RuntimeError, match="use_async=True"):
        adapter.instructions_provider(base="B", session_id="conv-1")


# ─── Session↔owner binding ────────────────────────────────────────

@pytest.mark.asyncio
async def test_owner_mismatch_raises_at_provider_construction(mock_async_clients) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="first@test", **CREDS)
    # First call pins session↔first@test.
    adapter.instructions_provider(base="B", session_id="conv-1")
    # Second call with a different owner for the same session must raise.
    with pytest.raises(ValueError):
        adapter.instructions_provider(
            base="B", session_id="conv-1", owner_id="second@test",
        )


# ─── Pack version surfacing ──────────────────────────

@pytest.mark.asyncio
async def test_catalog_block_surfaces_pack_versions(mock_async_clients) -> None:
    """When Beacon advertises a ``version`` on a pack, it rides into the
    ``[Available context packs]`` JSON block so the LLM can pass it as
    ``context_pack_version`` on a get_context_pack call. Packs without a
    version stay version-less in the block (not every env reports it yet)."""
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(
        return_value={
            "contextPacks": [
                {
                    "contextPackId": "PlanRules",
                    "description": "Plan rules",
                    "version": 4,
                    "dataNeeds": [],
                },
                {
                    "contextPackId": "Claims",
                    "description": "Claims",
                    "dataNeeds": [],
                },
            ],
        },
    )
    provider = adapter.instructions_provider(base="B", session_id="conv-1")

    out = await provider(run_context=None, agent=None)

    tail = out.split(CONTEXT_PACK_LIST_MARKER + "\n", 1)[1]
    parsed = json.loads(tail)
    assert parsed["PlanRules"]["version"] == 4
    # Packs the server did not version are emitted without a ``version`` key —
    # their tool calls omit ``context_pack_version`` and Beacon accepts the
    # current revision.
    assert "version" not in parsed["Claims"]


# ─── Context-pack instruction injection ──────────────────────────────


@pytest.mark.asyncio
async def test_injects_server_instruction_between_base_and_catalog(mock_async_clients) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    retrieval = mock_async_clients["retrieval"]
    retrieval.get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
    provider = adapter.instructions_provider(base="BASE", session_id="conv-1")

    out = await provider(run_context=None, agent=None)
    assert out.startswith(f"BASE\n\nSERVER INSTR\n\n{CONTEXT_PACK_LIST_MARKER}\n{{\"PlanRules\"")
    # Pinned: a later turn doesn't refetch.
    assert await provider(run_context=None, agent=None) == out
    retrieval.get_system_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_instruction_without_base(mock_async_clients) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
    provider = adapter.instructions_provider(base="", session_id="conv-1")

    out = await provider(run_context=None, agent=None)
    assert out.startswith(CONTEXT_PACK_INSTRUCTION)


@pytest.mark.asyncio
@pytest.mark.parametrize("already_there", ["server", "bundled"])
async def test_skips_instruction_already_in_base(mock_async_clients, already_there) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u@test", **CREDS)
    mock_async_clients["retrieval"].get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
    text = "SERVER INSTR" if already_there == "server" else CONTEXT_PACK_INSTRUCTION
    base = f"{text}\n\nMY PROMPT"
    provider = adapter.instructions_provider(base=base, session_id="conv-1")

    out = await provider(run_context=None, agent=None)
    assert out.startswith(f"{base}\n\n{CONTEXT_PACK_LIST_MARKER}\n")
