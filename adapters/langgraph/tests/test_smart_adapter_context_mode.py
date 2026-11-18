# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""``context_mode`` — the single knob that drives the self-driving adapter.

``context_mode`` is one clean kwarg (``auto`` / ``per-turn`` / ``manual``) that
resolves — in ``__init__`` only — into the two private attrs the fetch / tool
/ prompt paths read (``_policy`` / ``_install_recall_tool`` /
``_recall_tool_instruction``). It defaults to ``"auto"``: a bare
``LangGraphAdapter(graph=…)`` is a self-driving auto adapter with no further
configuration. There are no granular constructor knobs and no XOR guard — the
mode is the whole surface.

Coverage:

- **Mapping** — one case per mode pinning the resolved policy, tool presence,
  recall-tool instruction, and ``.context_mode``.
- **Default** — a bare adapter resolves to ``auto`` (the default).
- **per-turn fallback** — ``per-turn`` uses ``EveryTurnPolicy`` so the adapter
  seeds fast context every turn even when the agent never calls the tool
  (``should_fetch`` behaviour pinned in ``test_smart_adapter_policies``).
- **Seam** — ``NoFetchPolicy`` never fetches (backs manual).
- **Manager-injection** — an injected ``session_manager`` makes the mode inert:
  tool off, no seed (the injected manager owns retrieval).
- **Validation** — an invalid slug raises with the canonical message.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agentic_context import AsyncSessionManager

from agentic_context_langgraph import (
    EveryTurnPolicy,
    LangGraphAdapter,
    NoFetchPolicy,
    SessionFirstPolicy,
)
from agentic_context_langgraph.adapter import _RECALL_TOOL_SYSTEM_INSTRUCTION

from .conftest import _build_graph

BASE_URL = "https://beacon.example.com"


def _make(**overrides):
    """Self-driving adapter with default connection params; kwargs override."""
    kwargs = {
        "graph": _build_graph(),
        "agent_id": "bot",
        "owner_id": "user-1",
        "api_key": "tok",
        "base_url": BASE_URL,
        "dataspace": "default",
    }
    kwargs.update(overrides)
    return LangGraphAdapter(**kwargs)


# ─── Mapping — one case per mode ─────────────────────────────────────────


@pytest.mark.parametrize(
    "mode_kwargs, expected_mode, policy_cls, instruction",
    [
        ({"context_mode": "auto"}, "auto", SessionFirstPolicy, _RECALL_TOOL_SYSTEM_INSTRUCTION),
        # per-turn seeds fast context EVERY turn (the fallback), not never-fetch.
        ({"context_mode": "per-turn"}, "per-turn", EveryTurnPolicy, _RECALL_TOOL_SYSTEM_INSTRUCTION),
        # Manual: tool present, but the adapter injects nothing (customer owns prompt).
        ({"context_mode": "manual"}, "manual", NoFetchPolicy, None),
        # The default: a bare adapter (no context_mode passed) resolves to
        # ``auto`` — identical to an explicit ``context_mode="auto"``.
        ({}, "auto", SessionFirstPolicy, _RECALL_TOOL_SYSTEM_INSTRUCTION),
    ],
    ids=["auto", "per-turn", "manual", "bare-defaults-to-auto"],
)
def test_mode_maps_to_policy_tool_and_instruction(
    mode_kwargs, expected_mode, policy_cls, instruction
) -> None:
    adapter = _make(**mode_kwargs)
    assert adapter.context_mode == expected_mode
    assert adapter._install_recall_tool is True
    assert type(adapter.context_policy) is policy_cls
    assert len(adapter.default_tools()) == 2
    assert adapter._resolve_recall_tool_instruction() == instruction


# ─── Behavior at the seam ────────────────────────────────────────────────


def test_no_fetch_policy_never_fetches() -> None:
    """NoFetchPolicy backs manual: ``should_fetch`` is False for any args, so
    the adapter seeds no context SystemMessage."""
    policy = NoFetchPolicy()
    assert (
        policy.should_fetch(
            session_id="ses-1",
            turn_idx=0,
            last_user_text=None,
            last_payload=None,
            last_fetched_at=None,
        )
        is False
    )


# ─── Manager-injection makes the mode inert ──────────────────────────────


@pytest.mark.parametrize(
    "mode_kwargs, expected_mode",
    [({}, "auto"), ({"context_mode": "per-turn"}, "per-turn")],
    ids=["default-auto", "explicit-per-turn"],
)
def test_injected_session_manager_makes_mode_inert(mode_kwargs, expected_mode) -> None:
    """A manager-injected adapter: the mode still reports (default auto, or the
    explicit value — validated, no raise for a valid slug), but the injected
    manager owns retrieval, so the recall tool is not installed and no
    instruction is injected."""
    mgr = MagicMock(spec=AsyncSessionManager)
    adapter = LangGraphAdapter(
        agent_id="bot", session_manager=mgr, graph=_build_graph(), **mode_kwargs
    )
    assert adapter.context_mode == expected_mode
    assert adapter._install_recall_tool is False
    assert adapter.default_tools() == []
    assert adapter._resolve_recall_tool_instruction() is None


# ─── Validation ──────────────────────────────────────────────────────────


def _make_manager_injected(**overrides):
    return LangGraphAdapter(
        agent_id="bot",
        session_manager=MagicMock(spec=AsyncSessionManager),
        graph=_build_graph(),
        **overrides,
    )


@pytest.mark.parametrize(
    "factory, bad_slug",
    [(_make, s) for s in ("on", "off", "agentic", "per_turn", "Auto", "manual ")]
    # The slug is validated even when the mode is otherwise inert.
    + [(_make_manager_injected, "bogus")],
    ids=["on", "off", "agentic", "per_turn", "Auto", "manual-trailing-space",
         "manager-injection-bogus"],
)
def test_invalid_context_mode_slug_raises(factory, bad_slug: str) -> None:
    with pytest.raises(ValueError, match="must be one of 'auto', 'per-turn', 'manual'"):
        factory(context_mode=bad_slug)
