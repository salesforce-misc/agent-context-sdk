# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""The shared context-pack guidance/timeout live in core and are re-exported.

``CONTEXT_PACK_INSTRUCTION`` and ``CONTEXT_PACK_TIMEOUT_S`` are the single source of
truth every adapter's ``make_get_context_pack_tool`` (Claude / ADK / LangGraph)
references — so the agent guidance never drifts between adapters and a raw
QUICKSTART user (no adapter) can wire the pack POST with the same guidance. The
per-adapter re-export identity is asserted in each adapter's own pack-tool test
(those need the framework SDK installed); here we pin the core contract, which
needs nothing but core.
"""
from __future__ import annotations

import agentic_context as ac
from agentic_context.retrieval import (
    CONTEXT_PACK_INSTRUCTION as R_INSTRUCTION,
    CONTEXT_PACK_TIMEOUT_S as R_TIMEOUT,
    context_pack_params,
)


def test_constants_exported_from_facade_and_submodule_are_identical():
    assert ac.CONTEXT_PACK_INSTRUCTION is R_INSTRUCTION
    assert ac.CONTEXT_PACK_TIMEOUT_S == R_TIMEOUT
    assert "CONTEXT_PACK_INSTRUCTION" in ac.__all__
    assert "CONTEXT_PACK_TIMEOUT_S" in ac.__all__


def test_timeout_is_generous_but_under_serverless_ceiling():
    # A pack POST fans out across slow sources, so it must outlast the fast-read
    # default (10s) yet stay well under a typical serverless request timeout.
    assert R_TIMEOUT == 60.0


def test_instruction_is_domain_neutral_and_states_the_pack_contract():
    text = ac.CONTEXT_PACK_INSTRUCTION
    assert "get_context_pack" in text
    assert "invalid_pack" in text
    assert "preview" not in text.lower()
    assert "params" in text
    assert "data need" in text
    assert "ask the user" in text
    # Reusable across agents: no domain vocabulary leaks into the shared text.
    lowered = text.lower()
    for domain_word in ("insurance", "dental", "claim", "enrollment"):
        assert domain_word not in lowered


def test_context_pack_params_drops_blanks_and_keeps_query():
    assert context_pack_params(
        {"person_id": "005xx", "policy_id": "  "},
        "copay",
        {"policy_type": "MEDICAL"},
    ) == {"policy_type": "MEDICAL", "person_id": "005xx", "query": "copay"}
    assert context_pack_params('{"query": "from-params"}', "from-query", {}) == {
        "query": "from-params",
    }
    assert context_pack_params("not-json", "still-sent", {}) == {"query": "still-sent"}
    assert context_pack_params(None, "   ", {"unused": ""}) == {}


def test_context_pack_params_strips_model_supplied_owner_id():
    """``owner_id`` is the authenticated session owner — never a model-chosen
    value. The parsing boundary must drop it whether it arrives as a dict key
    or inside a JSON string, so a model cannot even attempt to pin the pack
    read to someone else's owner."""
    assert context_pack_params(
        {"owner_id": "attacker", "person_id": "005xx"},
        "",
        {},
    ) == {"person_id": "005xx"}
    assert context_pack_params(
        '{"owner_id": "attacker", "person_id": "005xx"}',
        "",
        {},
    ) == {"person_id": "005xx"}
    # Defaults still flow through — the strip only touches the model payload.
    assert context_pack_params(
        {"owner_id": "attacker"},
        "",
        {"owner_id": "server-chosen"},
    ) == {"owner_id": "server-chosen"}
