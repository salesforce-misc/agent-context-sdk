# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unit tests for AgentContextClient (Design.md §8)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_context import (
    CONTEXT_PACK_INSTRUCTION,
    CONTEXT_PACK_LIST_MARKER,
    AgentContextClient,
    ContextAuth,
    UserContextAuth,
    allocate_session_id,
)
from agentic_context.ace_sdk_tool_specs import TOOL_SPECS
from agentic_context.retrieval import CONTEXT_PACK_TIMEOUT_S



_DEFAULT_PACKS = [
    {
        "contextPackId": "ActiveHealthPlanCoverageV2",
        "description": "Active enrollments",
        "dataNeeds": [
            {
                "name": "enrollments",
                "description": "need enrollments",
                "parameters": [{"name": "query", "description": "free text"}],
            }
        ],
    },
    {
        "contextPackId": "EmployeeInsurancePreferencesV2",
        "description": "Preferences",
        "dataNeeds": [],
    },
]


@pytest.fixture(autouse=True)
def _stop_wire_patches():
    """``_wire`` starts patches; stop them so mocks don't leak into other tests."""
    yield
    patch.stopall()


def _auth_mock(
    *,
    user_id: str | None = "005OWNER",
    default_owner_id: str | None = None,
    aid: str | None = "TestAgent",
    owner_type: str = "employee",
    spec: type | None = None,
) -> MagicMock:
    auth = MagicMock(spec=spec) if spec else MagicMock()
    auth.get_token.return_value = ("tok", "https://tse.example.com")
    auth.user_id = user_id
    auth.default_owner_id = default_owner_id
    auth.aid = aid
    auth.default_owner_type = owner_type
    auth.attach = MagicMock()
    return auth


def _wire(
    ace: AgentContextClient,
    auth: MagicMock | None = None,
    agent_id: str | None = None,
    *,
    packs: list[dict] | None = None,
    get_pack_side_effect=None,
    get_pack_return=None,
    context: dict | None = None,
):
    """Authenticate with mocked Beacon transport (patches stopped after each test)."""
    agent_id = agent_id or (auth.aid if auth is not None and auth.aid else "TestAgent")
    auth = auth or _auth_mock(aid=agent_id)
    ic_p = patch("agentic_context.ace_sdk_client.IngestionClient")
    rc_p = patch("agentic_context.ace_sdk_client.RetrievalClient")
    sm_p = patch("agentic_context.ace_sdk_client.SessionManager")
    IC = ic_p.start()
    RC = rc_p.start()
    SM = sm_p.start()
    retrieval = RC.return_value
    retrieval.get_context.return_value = (
        {"contextId": "c1", "userContext": {"n": 1}} if context is None else context
    )
    retrieval.get_context_pack_list.return_value = {
        "contextPacks": _DEFAULT_PACKS if packs is None else packs,
    }
    if get_pack_side_effect is not None:
        retrieval.get_context_pack.side_effect = get_pack_side_effect
    else:
        retrieval.get_context_pack.return_value = get_pack_return or {"rows": [1]}
    mgr = SM.return_value
    mgr.create_session.return_value = {
        "sessionId": "ses-x",
        "ownerId": "005OWNER",
        "agentId": agent_id,
    }
    ace.authenticate(auth=auth, agent_id=agent_id)
    ace._test_patchers = (ic_p, rc_p, sm_p)  # type: ignore[attr-defined]
    return retrieval, mgr, IC, RC, SM


def _seed_and_open(ace: AgentContextClient, cid: str = "ses-1", *, title: str = "") -> None:
    ace.get_initial_context(cid)
    ace.open_conversation(cid, title=title)


def test_allocate_session_id_format() -> None:
    sid = allocate_session_id()
    assert sid.startswith("ses-")
    assert len(sid) > 4


def test_methods_require_authenticate() -> None:
    ace = AgentContextClient()
    with pytest.raises(RuntimeError, match="authenticate"):
        ace.get_initial_context("ses-x")
    with pytest.raises(RuntimeError, match="authenticate"):
        ace.open_conversation("ses-x")


def test_ingest_requires_open() -> None:
    ace = AgentContextClient()
    ace._auth = MagicMock()
    ace._client = MagicMock()
    ace._retrieval = MagicMock()
    ace._owner_id = "005ABC"
    ace._agent_id = "TestAgent"
    ace._owner_type = "employee"
    with pytest.raises(RuntimeError, match="open_conversation"):
        ace.ingest("ses-x", role="user", content="hi")


def test_ordered_happy_path() -> None:
    ace = AgentContextClient()
    auth = MagicMock()
    auth.get_token.return_value = ("tok", "https://tse.example.com")
    auth.user_id = "005OWNER"
    auth.default_owner_type = "employee"
    auth.aid = "TestAgent"
    auth.attach = MagicMock()

    with (
        patch("agentic_context.ace_sdk_client.IngestionClient") as IC,
        patch("agentic_context.ace_sdk_client.RetrievalClient") as RC,
        patch("agentic_context.ace_sdk_client.SessionManager") as SM,
    ):
        retrieval = RC.return_value
        retrieval.get_context.return_value = {"facts": []}
        retrieval.get_context_pack_list.return_value = {
            "contextPacks": [{"contextPackId": "ActiveHealthPlanCoverageV2"}],
        }
        retrieval.get_context_pack.return_value = {"enrollments": []}
        mgr = SM.return_value
        mgr.create_session.return_value = {
            "sessionId": "ses-demo",
            "ownerId": "005OWNER",
            "agentId": "TestAgent",
        }

        step1 = ace.authenticate(auth=auth, agent_id="TestAgent")
        assert step1["ok"] is True
        assert step1["owner_id"] == "005OWNER"
        auth.attach.assert_any_call(IC.return_value)

        step2 = ace.allocate_session_id()
        cid = step2["conversation_id"]
        assert cid == step2["session_id"]

        step3 = ace.get_initial_context(cid)
        retrieval.get_context.assert_called_once()
        retrieval.get_context_pack_list.assert_called_once_with("TestAgent")
        assert step3["pack_ids"] == ["ActiveHealthPlanCoverageV2"]

        step4 = ace.open_conversation(cid, title="t")
        assert step4["ok"] is True
        mgr.create_session.assert_called_once_with(cid, title="t")

        assert "system_prompt_suffix" in step3
        assert "[Available context packs]" in step3["pack_catalog_text"]

        step5a = ace.ingest(cid, role="user", content="hi")
        assert step5a["event_type"] == "user_message"
        mgr.ingest.assert_called_once()
        records = mgr.ingest.call_args[0][0]
        assert records[0].event_type == "user_message"
        assert records[0].content == "hi"

        step5b = ace.get_context_pack(
            cid, "ActiveHealthPlanCoverageV2", params={"query": "coverage"},
        )
        retrieval.get_context_pack.assert_called_once()
        assert step5b["status"] == "ok"
        assert step5b["context_pack_id"] == "ActiveHealthPlanCoverageV2"

        ace.ingest(cid, role="assistant", content="hello")
        assert mgr.ingest.call_count == 2
        user_rec = mgr.ingest.call_args_list[0][0][0][0]
        asst_rec = mgr.ingest.call_args_list[1][0][0][0]
        assert user_rec.interaction_id == asst_rec.interaction_id


def test_tool_specs_are_framework_neutral() -> None:
    ace = AgentContextClient()
    specs = ace.tool_specs()
    names = [s["name"] for s in specs]
    # tool_specs() is model tools only
    assert names == [
        "ace_get_context",
        "ace_get_context_pack",
    ]
    for s in specs:
        assert set(s) == {"name", "description", "parameters"}
        assert s["parameters"]["type"] == "object"
        # Vendor-specific keys must not appear in core descriptors
        assert "input_schema" not in s
        assert "args_schema" not in s
        assert "function" not in s


def test_tool_specs_are_model_catalog_only() -> None:
    ace = AgentContextClient()
    names = {s["name"] for s in ace.tool_specs()}
    assert names == {
        "ace_get_context",
        "ace_get_context_pack",
    }
    # Host lifecycle methods / lifecycle stay off the model catalog
    assert "ace_start_conversation" not in names
    assert "ace_ingest_customer_message" not in names
    assert "ace_ingest_agent_message" not in names
    assert "ace_authenticate" not in names
    assert "ace_ingest" not in names


def test_start_conversation_composite() -> None:
    ace = AgentContextClient()
    auth = MagicMock()
    auth.get_token.return_value = ("tok", "https://tse.example.com")
    auth.user_id = "005OWNER"
    auth.default_owner_type = "employee"
    auth.aid = "TestAgent"
    auth.attach = MagicMock()

    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient") as RC,
        patch("agentic_context.ace_sdk_client.SessionManager") as SM,
    ):
        retrieval = RC.return_value
        retrieval.get_context.return_value = {"facts": []}
        retrieval.get_context_pack_list.return_value = {
            "contextPacks": [{"contextPackId": "ActiveHealthPlanCoverageV2"}],
        }
        SM.return_value.create_session.return_value = {
            "sessionId": "cid-1",
            "ownerId": "005OWNER",
            "agentId": "TestAgent",
        }
        # Host lifecycle method owns auth when not yet authenticated
        out = ace.start_conversation(
            "cid-1", title="hi", auth=auth, agent_id="TestAgent"
        )
        assert out["ok"] is True
        assert out["conversation_id"] == "cid-1"
        assert out["bound"] is True
        assert out["context"] == {"facts": []}
        assert "system_prompt_suffix" in out
        assert "ActiveHealthPlanCoverageV2" in out["pack_ids"]
        assert ace._active_conversation == "cid-1"


def test_call_tool_dispatches_model_tools() -> None:
    ace = AgentContextClient()
    auth = MagicMock()
    auth.get_token.return_value = ("tok", "https://tse.example.com")
    auth.user_id = "005OWNER"
    auth.default_owner_type = "employee"
    auth.aid = "TestAgent"
    auth.attach = MagicMock()
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient") as RC,
        patch("agentic_context.ace_sdk_client.SessionManager") as SM,
    ):
        RC.return_value.get_context.return_value = {"facts": []}
        RC.return_value.get_context_pack_list.return_value = {"contextPacks": []}
        SM.return_value.create_session.return_value = {
            "sessionId": "ses-1",
            "ownerId": "005OWNER",
            "agentId": "TestAgent",
        }
        ace.authenticate(auth=auth, agent_id="TestAgent")
        ace.get_initial_context("ses-1")
        ace.open_conversation("ses-1")
        out = ace.call_tool("ace_get_context", {})
        assert out["status"] == "ok"
        assert out["context"] == {"facts": []}


def test_call_tool_rejects_host_and_unknown() -> None:
    ace = AgentContextClient()
    with pytest.raises(ValueError, match="unknown model tool"):
        ace.call_tool("not_a_tool", {})
    with pytest.raises(ValueError, match="unknown model tool"):
        ace.call_tool("ace_ingest_customer_message", {"content": "hi"})
    with pytest.raises(ValueError, match="unknown model tool"):
        ace.call_tool("ace_reset", {})


def test_open_without_seed_rejected() -> None:
    ace = AgentContextClient()
    auth = MagicMock()
    auth.get_token.return_value = ("tok", "https://tse.example.com")
    auth.user_id = "005"
    auth.default_owner_type = "employee"
    auth.aid = "A"
    auth.attach = MagicMock()
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
        patch("agentic_context.ace_sdk_client.SessionManager"),
    ):
        ace.authenticate(auth=auth, agent_id="A")
        with pytest.raises(RuntimeError, match="get_initial_context"):
            ace.open_conversation("ses-x")


# ─── Adapter-parity / edge cases ─────────────────────────────────


# ─── Prompt / catalog parity (adapters inject these automatically) ─


def test_seed_returns_adapter_style_prompt_helpers() -> None:
    ace = AgentContextClient()
    _wire(ace)
    initial = ace.get_initial_context("ses-1")
    assert initial["prompt_guidance"] == CONTEXT_PACK_INSTRUCTION
    assert initial["pack_catalog_text"].startswith(CONTEXT_PACK_LIST_MARKER)
    assert "ActiveHealthPlanCoverageV2" in initial["pack_catalog"]
    assert initial["pack_catalog"]["ActiveHealthPlanCoverageV2"]["dataNeeds"][0]["name"] == (
        "enrollments"
    )
    assert CONTEXT_PACK_INSTRUCTION in initial["system_prompt_suffix"]
    assert CONTEXT_PACK_LIST_MARKER in initial["system_prompt_suffix"]
    assert "ace_get_context" in initial["system_prompt_suffix"]
    assert "ace_get_context_pack" in initial["context_guidance"]
    opened = ace.open_conversation("ses-1")
    assert opened["pack_catalog_text"] == initial["pack_catalog_text"]
    assert opened["bound"] is True


def test_bound_mid_turn_omits_conversation_id() -> None:
    """After open, model need not supply conversation_id / owner."""
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    seed_gets = retrieval.get_context.call_count
    out = ace.call_tool(
        "ace_get_context_pack",
        {"context_pack_id": "ActiveHealthPlanCoverageV2"},
    )
    assert out["status"] == "ok"
    assert out["conversation_id"] == "ses-1"
    retrieval.get_context_pack.assert_called_once()
    assert retrieval.get_context_pack.call_args[0][1] == "005OWNER"

    ctx = ace.call_tool("ace_get_context", {})
    assert ctx["status"] == "ok"
    assert ctx["context"]["userContext"] == {"n": 1}
    # Always hits Beacon (no SDK cache)
    assert retrieval.get_context.call_count == seed_gets + 1
    # Model-supplied owner_id / query must be stripped
    again = ace.call_tool(
        "ace_get_context",
        {"query": "why refund denied", "owner_id": "EVIL"},
    )
    assert again["status"] == "ok"
    assert retrieval.get_context.call_count == seed_gets + 2


def test_get_context_always_hits_beacon() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    retrieval.get_context.return_value = {"userContext": [{"entry": "x"}]}
    _seed_and_open(ace)
    seed_gets = retrieval.get_context.call_count
    out = ace.get_context()
    assert out["status"] == "ok"
    assert out["context"]["userContext"] == [{"entry": "x"}]
    assert retrieval.get_context.call_count == seed_gets + 1
    ace.get_context()
    assert retrieval.get_context.call_count == seed_gets + 2


# ─── Pack tool parity ─────────────────────────────────────────────


def test_invalid_pack_without_http_call() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    out = ace.get_context_pack("ses-1", "NotARealPack")
    assert out["status"] == "invalid_pack"
    assert set(out["valid_pack_ids"]) == {
        "ActiveHealthPlanCoverageV2",
        "EmployeeInsurancePreferencesV2",
    }
    retrieval.get_context_pack.assert_not_called()


def test_pack_id_alias_and_context_pack_id() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace)
    via_alias = ace.call_tool(
        "ace_get_context_pack",
        {"conversation_id": "ses-1", "pack_id": "ActiveHealthPlanCoverageV2"},
    )
    assert via_alias["status"] == "ok"
    via_name = ace.call_tool(
        "ace_get_context_pack",
        {
            "conversation_id": "ses-1",
            "context_pack_id": "EmployeeInsurancePreferencesV2",
        },
    )
    assert via_name["status"] == "ok"


def test_blank_params_stripped_like_adapter() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    ace.get_context_pack(
        "ses-1",
        "ActiveHealthPlanCoverageV2",
        params={"query": "  ", "tier": "Gold", "empty": ""},
    )
    sent = retrieval.get_context_pack.call_args.kwargs["params"]
    assert sent == {"tier": "Gold"}
    assert retrieval.get_context_pack.call_args.kwargs["timeout"] == CONTEXT_PACK_TIMEOUT_S


def test_params_as_json_string_from_llm() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    ace.get_context_pack(
        "ses-1",
        "ActiveHealthPlanCoverageV2",
        params='{"query":"coverage"}',
    )
    assert retrieval.get_context_pack.call_args.kwargs["params"] == {"query": "coverage"}


def test_not_ready_on_beacon_failure_no_raise() -> None:
    ace = AgentContextClient()
    _wire(ace, get_pack_side_effect=RuntimeError("boom"))
    _seed_and_open(ace)
    out = ace.get_context_pack("ses-1", "ActiveHealthPlanCoverageV2")
    assert out["status"] == "not_ready"
    assert "detail" in out
    assert out["context_pack_id"] == "ActiveHealthPlanCoverageV2"


def test_get_context_not_ready_on_beacon_failure_no_raise() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    retrieval.get_context.side_effect = RuntimeError("boom")
    out = ace.get_context()
    assert out["status"] == "not_ready"
    assert "detail" in out
    assert out["conversation_id"] == "ses-1"


def test_pack_list_404_does_not_abort_seed() -> None:
    """Preview GET runtime-context-packs may 404 — start must still open."""
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    retrieval.get_context_pack_list.side_effect = RuntimeError("404 preview")
    initial = ace.get_initial_context("ses-1")
    assert initial["pack_ids"] == []
    assert ace._allowed_packs["ses-1"] is None  # skip invalid_pack enforcement
    opened = ace.open_conversation("ses-1")
    assert opened["ok"] is True
    # Pack fetch still allowed (no catalog gate) when list was unavailable
    out = ace.get_context_pack("ses-1", "ActiveHealthPlanCoverageV2")
    assert out["status"] == "ok"


def test_empty_catalog_rejects_all_packs() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace, packs=[])
    _seed_and_open(ace)
    out = ace.get_context_pack("ses-1", "Anything")
    assert out["status"] == "invalid_pack"
    assert out["valid_pack_ids"] == []
    retrieval.get_context_pack.assert_not_called()


def test_include_pack_list_false_skips_invalid_check() -> None:
    """Host opt-out of catalog: allow any pack id (advanced / tests)."""
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    ace.get_initial_context("ses-1", include_pack_list=False)
    ace.open_conversation("ses-1")
    out = ace.get_context_pack("ses-1", "CustomPack")
    assert out["status"] == "ok"
    retrieval.get_context_pack.assert_called_once()


# ─── Turn grouping ────────────────────────────────────────────────


def test_call_tool_normalizes_camel_case_gateway_keys() -> None:
    ace = AgentContextClient()
    retrieval, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    out = ace.call_tool(
        "ace_get_context_pack",
        {
            "conversationId": "ses-1",
            "contextPackId": "ActiveHealthPlanCoverageV2",
        },
    )
    assert out["status"] == "ok"
    retrieval.get_context_pack.assert_called_once()
    assert retrieval.get_context_pack.call_args[0][0] == "ses-1"
    assert retrieval.get_context_pack.call_args[0][3] == "ActiveHealthPlanCoverageV2"

    ingested = ace.ingest_customer_message(
        "hi", interaction_id="turn-camel",
    )
    assert ingested["interaction_id"] == "turn-camel"
    assert mgr.ingest.called


def test_ingest_accepts_interaction_id_and_metadata() -> None:
    ace = AgentContextClient()
    _, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    out = ace.ingest_customer_message(
        "hi",
        interaction_id="turn-host",
        metadata={"channel": "demo"},
    )
    assert out["interaction_id"] == "turn-host"
    rec = mgr.ingest.call_args[0][0][0]
    assert rec.metadata.get("channel") == "demo"


def test_ingest_accepts_event_id() -> None:
    ace = AgentContextClient()
    _, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    first = ace.ingest_customer_message("hi", event_id="user_message-retry1")
    assert first["event_id"] == "user_message-retry1"
    again = ace.ingest_customer_message("hi", event_id="user_message-retry1")
    assert again["event_id"] == "user_message-retry1"
    assert mgr.ingest.call_count == 2
    assert mgr.ingest.call_args_list[0][0][0][0].event_id == "user_message-retry1"
    assert mgr.ingest.call_args_list[1][0][0][0].event_id == "user_message-retry1"


def test_user_assistant_share_interaction_id() -> None:
    ace = AgentContextClient()
    _, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace)
    u = ace.ingest("ses-1", role="user", content="hi")
    t = ace.ingest("ses-1", role="tool", content="trace")
    a = ace.ingest("ses-1", role="assistant", content="hello")
    assert u["interaction_id"] == t["interaction_id"] == a["interaction_id"]
    # next user starts a new turn
    u2 = ace.ingest("ses-1", role="user", content="again")
    a2 = ace.ingest("ses-1", role="assistant", content="ok")
    assert u2["interaction_id"] != u["interaction_id"]
    assert u2["interaction_id"] == a2["interaction_id"]
    assert mgr.ingest.call_count == 5


def test_multi_agent_messages_share_interaction_id() -> None:
    """0..N agent messages stay on the open turn until the next user message."""
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace)
    u = ace.ingest_customer_message("hi", "ses-1")
    a1 = ace.ingest_agent_message("part1", "ses-1")
    a2 = ace.ingest_agent_message("part2", "ses-1")
    assert u["interaction_id"] == a1["interaction_id"] == a2["interaction_id"]
    u2 = ace.ingest_customer_message("again", "ses-1")
    assert u2["interaction_id"] != u["interaction_id"]


def test_explicit_interaction_id_overrides_auto() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace)
    u = ace.ingest("ses-1", role="user", content="hi", interaction_id="turn-fixed")
    a = ace.ingest("ses-1", role="assistant", content="yo")
    assert u["interaction_id"] == "turn-fixed"
    # assistant without explicit id still reuses pending turn-fixed
    assert a["interaction_id"] == "turn-fixed"
    a2 = ace.ingest("ses-1", role="assistant", content="continued")
    assert a2["interaction_id"] == "turn-fixed"


# ─── Host misuse / multi-session UX ───────────────────────────────


def test_seed_wrong_id_then_open_other_rejected() -> None:
    ace = AgentContextClient()
    _wire(ace)
    ace.get_initial_context("ses-a")
    with pytest.raises(RuntimeError, match="get_initial_context"):
        ace.open_conversation("ses-b")


def test_pack_on_wrong_conversation_id() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace, "ses-1")
    with pytest.raises(RuntimeError, match="not open"):
        ace.get_context_pack("ses-other", "ActiveHealthPlanCoverageV2")


def test_two_conversations_independent_pack_allowlists() -> None:
    ace = AgentContextClient()
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient") as RC,
        patch("agentic_context.ace_sdk_client.SessionManager") as SM,
    ):
        RC.return_value.get_context.return_value = {}
        RC.return_value.get_context_pack.return_value = {}
        SM.return_value.create_session.return_value = {
            "sessionId": "x", "ownerId": "o", "agentId": "Agent",
        }
        ace.authenticate(auth=_auth_mock(aid="Agent"), agent_id="Agent")

        RC.return_value.get_context_pack_list.return_value = {
            "contextPacks": [{"contextPackId": "OnlyA", "description": "", "dataNeeds": []}],
        }
        ace.get_initial_context("ses-a")
        ace.open_conversation("ses-a")

        RC.return_value.get_context_pack_list.return_value = {
            "contextPacks": [{"contextPackId": "OnlyB", "description": "", "dataNeeds": []}],
        }
        ace.get_initial_context("ses-b")
        ace.open_conversation("ses-b")

        assert ace.get_context_pack("ses-a", "OnlyB")["status"] == "invalid_pack"
        assert ace.get_context_pack("ses-a", "OnlyA")["status"] == "ok"
        assert ace.get_context_pack("ses-b", "OnlyA")["status"] == "invalid_pack"
        assert ace.get_context_pack("ses-b", "OnlyB")["status"] == "ok"


def test_reseed_before_open_refreshes_catalog() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    ace.get_initial_context("ses-1")
    retrieval.get_context_pack_list.return_value = {
        "contextPacks": [{"contextPackId": "NewPack", "description": "n", "dataNeeds": []}],
    }
    again = ace.get_initial_context("ses-1")  # allowed before open
    assert again["pack_ids"] == ["NewPack"]
    ace.open_conversation("ses-1")
    assert ace.get_context_pack("ses-1", "ActiveHealthPlanCoverageV2")["status"] == "invalid_pack"
    assert ace.get_context_pack("ses-1", "NewPack")["status"] == "ok"


def test_allocate_id_not_required_to_match_used_id() -> None:
    """Host may ignore allocate and pick their own id — still valid if seeded."""
    ace = AgentContextClient()
    _wire(ace)
    allocated = ace.allocate_session_id()["conversation_id"]
    custom = "ses-custom-host"
    assert allocated != custom
    ace.get_initial_context(custom)
    ace.open_conversation(custom)
    assert ace.ingest(custom, role="user", content="hi")["ok"] is True


def test_call_tool_pack_missing_id_validated() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace)
    with pytest.raises(ValueError, match="context_pack_id"):
        ace.call_tool("ace_get_context_pack", {"conversation_id": "ses-1"})


# ─── Auth, lifecycle, and contract scenarios ──────────────────────


# ─── Auth validations ─────────────────────────────────────────────


def test_authenticate_requires_credentials() -> None:
    ace = AgentContextClient()
    with pytest.raises(ValueError, match="authenticate requires"):
        ace.authenticate()


def test_authenticate_requires_owner() -> None:
    ace = AgentContextClient()
    auth = _auth_mock(user_id=None, default_owner_id=None)
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        with pytest.raises(ValueError, match="owner_id"):
            ace.authenticate(auth=auth, agent_id="TestAgent")


def test_authenticate_requires_agent_id() -> None:
    ace = AgentContextClient()
    auth = _auth_mock(aid=None)
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        with pytest.raises(ValueError, match="agent_id"):
            ace.authenticate(auth=auth)


def test_authenticate_resolves_private_aid() -> None:
    """Real UserContextAuth stores ``_aid`` without a public ``.aid`` property."""
    ace = AgentContextClient()

    class _Auth:
        user_id = "005OWNER"
        default_owner_type = "employee"

        def __init__(self) -> None:
            self._aid = "FromPrivateAid"

        def get_token(self):
            return ("tok", "https://tse.example.com/")

        def attach(self, _client) -> None:
            return None

    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        out = ace.authenticate(auth=_Auth())  # type: ignore[arg-type]
    assert out["agent_id"] == "FromPrivateAid"


def test_reset_closes_http_clients() -> None:
    ace = AgentContextClient()
    client, retrieval = MagicMock(), MagicMock()
    _wire(ace)
    ace._client = client
    ace._retrieval = retrieval
    _seed_and_open(ace, "ses-1")
    reset = ace.reset()
    assert reset["ok"] is True
    assert "ses-1" in reset["closed_conversations"]
    client.close.assert_called_once()
    retrieval.close.assert_called_once()
    assert ace._client is None
    assert ace._retrieval is None


def test_start_conversation_idempotent_returns_cached_seed() -> None:
    ace = AgentContextClient()
    _wire(ace)
    first = ace.start_conversation(conversation_id="ses-1", title="t")
    assert first["already_open"] is False
    assert first["context"] is not None
    assert "pack_ids" in first
    second = ace.start_conversation(conversation_id="ses-1")
    assert second["already_open"] is True
    assert second["context"] == first["context"]
    assert second["pack_ids"] == first["pack_ids"]
    assert second["pack_catalog"] == first["pack_catalog"]


def test_tool_skill_dir_resolves_packaged_skill() -> None:
    """Wheel-safe path: skill lives under the installed agentic_context package."""
    from agentic_context import ace_sdk_client as dt

    path = AgentContextClient.tool_skill_dir()
    assert (path / "SKILL.md").is_file()
    assert path.name == "ace-mid-turn-tools"
    packaged = Path(dt.__file__).resolve().parent / "skills" / "ace-mid-turn-tools"
    assert packaged.is_dir()
    assert (packaged / "SKILL.md").is_file()
    # Prefer packaged location (first candidate) when present
    assert path == packaged


def test_authenticate_keeps_agent_id() -> None:
    """authenticate() must keep the explicit agent_id."""
    ace = AgentContextClient()
    fake = _auth_mock()
    with (
        patch("agentic_context.ace_sdk_client.UserContextAuth", return_value=fake),
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        out = ace.authenticate(
            agent_id="FacadeAgent",
            consumer_key="ck",
            login_url="login.example.com",
            refresh_token="rt",
        )
    assert out["ok"] is True
    assert out["agent_id"] == "FacadeAgent"


def test_authenticate_falls_back_to_default_owner_id() -> None:
    ace = AgentContextClient()
    auth = _auth_mock(user_id=None, default_owner_id="005FALLBACK")
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        out = ace.authenticate(auth=auth, agent_id="A1")
    assert out["owner_id"] == "005FALLBACK"
    assert out["dataspace"] == "default"
    assert out["base_url"] == "https://tse.example.com"


def test_authenticate_refresh_token_path() -> None:
    ace = AgentContextClient()
    fake = _auth_mock()
    with (
        patch("agentic_context.ace_sdk_client.UserContextAuth", return_value=fake) as UCA,
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        out = ace.authenticate(
            agent_id="A1",
            consumer_key="ck",
            login_url="login.example.com",
            refresh_token="rt",
        )
    UCA.assert_called_once()
    assert UCA.call_args.kwargs["refresh_token"] == "rt"
    assert out["ok"] is True


def test_authenticate_login_path_when_no_refresh() -> None:
    ace = AgentContextClient()
    fake = _auth_mock()
    with (
        patch("agentic_context.ace_sdk_client.UserContextAuth") as UCA,
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient"),
    ):
        UCA.login.return_value = fake
        out = ace.authenticate(
            agent_id="A1",
            consumer_key="ck",
            login_url="login.example.com",
        )
    UCA.login.assert_called_once()
    assert out["ok"] is True


def test_reauthenticate_requires_reset_when_seeded() -> None:
    ace = AgentContextClient()
    _wire(ace)
    ace.get_initial_context("ses-keep")
    with pytest.raises(RuntimeError, match="reset"):
        ace.authenticate(auth=_auth_mock(), agent_id="TestAgent")
    ace.reset()
    _wire(ace)  # ok after reset
    assert ace._sessions == {}
    assert ace._initial_context_done == set()


def test_reauthenticate_requires_reset_when_open() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace, "ses-keep")
    with pytest.raises(RuntimeError, match="reset"):
        ace.authenticate(auth=_auth_mock(), agent_id="TestAgent")


def test_idle_reauthenticate_allowed() -> None:
    """Re-auth with no seeded/open conversations replaces clients."""
    ace = AgentContextClient()
    _wire(ace)
    out = ace.authenticate(auth=_auth_mock(), agent_id="TestAgent")
    assert out["ok"] is True


# ─── Order / readiness (SDK parity) ───────────────────────────────


def test_open_requires_initial_context() -> None:
    ace = AgentContextClient()
    _wire(ace)
    with pytest.raises(RuntimeError, match="get_initial_context"):
        ace.open_conversation("ses-1")


def test_pack_requires_open() -> None:
    ace = AgentContextClient()
    _wire(ace)
    ace.get_initial_context("ses-new")
    with pytest.raises(RuntimeError, match="open_conversation before get_context_pack"):
        ace.get_context_pack("ses-new", "ActiveHealthPlanCoverageV2")
    with pytest.raises(RuntimeError, match="open_conversation"):
        ace.ingest("ses-new", role="user", content="hi")


def test_no_initial_context_after_open() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace, "ses-1")
    with pytest.raises(RuntimeError, match="get_context / get_context_pack"):
        ace.get_initial_context("ses-1")
    # pack still works mid-turn
    ace.get_context_pack("ses-1", "ActiveHealthPlanCoverageV2")
    assert retrieval.get_context_pack.called


def test_double_open_rejected() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace, "ses-1")
    with pytest.raises(RuntimeError, match="already open"):
        ace.open_conversation("ses-1")


def test_get_initial_context_skips_pack_list_when_disabled() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    out = ace.get_initial_context("ses-1", include_pack_list=False)
    retrieval.get_context_pack_list.assert_not_called()
    assert out["context_packs"] == []
    assert out["pack_ids"] == []


def test_get_initial_context_pack_id_fallbacks() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    retrieval.get_context_pack_list.return_value = {
        "contextPacks": [
            {"contextPackId": "A"},
            {"id": "B"},
            {"name": "C"},
            {"other": "D"},
            "skip-non-dict",
        ],
    }
    out = ace.get_initial_context("ses-1")
    assert out["pack_ids"] == ["A", "B", "C"]


def test_multi_conversation_isolation() -> None:
    ace = AgentContextClient()
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient") as RC,
        patch("agentic_context.ace_sdk_client.SessionManager") as SM,
    ):
        RC.return_value.get_context_pack_list.return_value = {"contextPacks": []}
        RC.return_value.get_context.return_value = {}
        m1, m2 = MagicMock(), MagicMock()
        SM.side_effect = [m1, m2]
        m1.create_session.return_value = {"sessionId": "ses-1", "ownerId": "o", "agentId": "a"}
        m2.create_session.return_value = {"sessionId": "ses-2", "ownerId": "o", "agentId": "a"}
        ace.authenticate(auth=_auth_mock(), agent_id="a")
        _seed_and_open(ace, "ses-1")
        _seed_and_open(ace, "ses-2")
        ace.ingest("ses-1", role="user", content="one")
        ace.ingest("ses-2", role="user", content="two")
        assert m1.ingest.call_count == 1
        assert m2.ingest.call_count == 1
        assert m1.ingest.call_args[0][0][0].content == "one"
        assert m2.ingest.call_args[0][0][0].content == "two"


# ─── Ingest role / record shape ───────────────────────────────────


@pytest.mark.parametrize(
    "role,expected_et,expected_category,participant",
    [
        ("user", "user_message", "Message", "005OWNER"),
        ("Human", "user_message", "Message", "005OWNER"),
        ("assistant", "agent_response", "Message", "TestAgent"),
        ("agent", "agent_response", "Message", "TestAgent"),
        ("AI", "agent_response", "Message", "TestAgent"),
        ("tool", "tool_result", "Trace", "TestAgent"),
    ],
)
def test_ingest_role_mapping(role, expected_et, expected_category, participant) -> None:
    ace = AgentContextClient()
    _, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace, "ses-1")
    out = ace.ingest("ses-1", role=role, content="payload")
    assert out["event_type"] == expected_et
    rec = mgr.ingest.call_args[0][0][0]
    assert rec.event_type == expected_et
    assert rec.category == expected_category
    assert rec.participant_id == participant
    assert rec.session_id == "ses-1"
    assert rec.agent_id == "TestAgent"
    if expected_category == "Message":
        assert rec.content == "payload"
        assert rec.variable is None
    else:
        assert rec.content is None
        assert rec.variable == {"OutputValueText": "payload"}


def test_ingest_explicit_event_type_overrides_role() -> None:
    ace = AgentContextClient()
    _, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace, "ses-1")
    ace.ingest(
        "ses-1",
        role="user",
        content="x",
        event_type="agent_reasoning",
        interaction_id="turn-fixed",
    )
    rec = mgr.ingest.call_args[0][0][0]
    assert rec.event_type == "agent_reasoning"
    assert rec.category == "Trace"
    assert rec.content is None
    assert rec.variable == {"OutputValueText": "x"}
    assert rec.interaction_id == "turn-fixed"


def test_ingest_tool_result_uses_trace_variable() -> None:
    ace = AgentContextClient()
    _, mgr, _, _, _ = _wire(ace)
    _seed_and_open(ace, "ses-1")
    ace.ingest("ses-1", role="tool", content='{"ok":true}')
    rec = mgr.ingest.call_args[0][0][0]
    assert rec.category == "Trace"
    assert rec.content is None
    assert rec.variable == {"OutputValueText": '{"ok":true}'}


# ─── call_tool / schema contract ──────────────────────────────────


def test_tool_specs_required_align_with_call_tool() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace, "ses-1")

    for spec in ace.tool_specs():
        name = spec["name"]
        if name == "ace_get_context":
            args: dict = {"conversation_id": "ses-1"}
        elif name == "ace_get_context_pack":
            args = {"conversation_id": "ses-1", "context_pack_id": "ActiveHealthPlanCoverageV2"}
        else:
            raise AssertionError(f"unexpected model tool: {name}")
        out = ace.call_tool(name, args)
        assert isinstance(out, dict)


def test_call_tool_missing_required_raises() -> None:
    ace = AgentContextClient()
    _wire(ace)
    _seed_and_open(ace, "ses-1")
    with pytest.raises(ValueError, match="missing required"):
        ace.call_tool("ace_get_context_pack", {"conversation_id": "ses-1"})


def test_call_tool_strips_unknown_keys() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace, "ses-1")
    out = ace.call_tool(
        "ace_get_context",
        {
            "conversation_id": "ses-1",
            "extra_llm_field": "oops",
            "owner_id": "EVIL",
        },
    )
    assert out["status"] == "ok"
    assert retrieval.get_context.called
    # owner_id stripped — trusted bound owner used
    assert retrieval.get_context.call_args[0][1] == "005OWNER"


def test_call_tool_none_arguments() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace)
    _seed_and_open(ace, "ses-1")
    out = ace.call_tool("ace_get_context", None)
    assert out["status"] == "ok"
    assert retrieval.get_context.called


def test_allocate_requires_authenticate() -> None:
    ace = AgentContextClient()
    with pytest.raises(RuntimeError, match="authenticate"):
        ace.allocate_session_id()


def test_tool_specs_are_deep_copy_isolated() -> None:
    ace = AgentContextClient()
    specs = ace.tool_specs()
    specs[0]["parameters"]["properties"]["conversation_id"]["description"] = "MUTATED"
    fresh = ace.tool_specs()
    assert (
        fresh[0]["parameters"]["properties"]["conversation_id"]["description"]
        != "MUTATED"
    )


def test_module_tool_specs_match_instance() -> None:
    ace = AgentContextClient()
    assert [s["name"] for s in ace.tool_specs()] == [s["name"] for s in TOOL_SPECS]


def test_get_context_uses_owner_type() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace, auth=_auth_mock(owner_type="employee"))
    ace.get_initial_context("ses-1")
    assert retrieval.get_context.call_args.kwargs["owner_type"] == "employee"
    ace.open_conversation("ses-1")
    out = ace.get_context_pack("ses-1", "ActiveHealthPlanCoverageV2", params={"q": 1})
    assert out["status"] == "ok"
    pack_call = retrieval.get_context_pack.call_args
    assert pack_call.kwargs["owner_type"] == "employee"
    assert pack_call.kwargs["params"] == {"q": "1"}


def test_get_conversation_passes_owner_type() -> None:
    ace = AgentContextClient()
    retrieval, _, _, _, _ = _wire(ace, auth=_auth_mock(owner_type="employee"))
    _seed_and_open(ace, "ses-1")
    retrieval.get_conversation.return_value = {"id": "ses-1"}
    out = ace.get_conversation("ses-1")
    assert out == {"id": "ses-1"}
    retrieval.get_conversation.assert_called_once_with(
        "ses-1", "005OWNER", "TestAgent", owner_type="employee",
    )


def test_start_conversation_rejects_owner_mismatch() -> None:
    ace = AgentContextClient()
    _wire(ace)
    with pytest.raises(ValueError, match="owner_id"):
        ace.start_conversation(
            conversation_id="ses-other",
            owner_id="005OTHER",
        )


def test_start_conversation_rejects_auth_for_different_user() -> None:
    ace = AgentContextClient()
    _wire(ace)
    with pytest.raises(ValueError, match="auth user '005OTHER' does not match"):
        ace.start_conversation(
            conversation_id="ses-other", auth=_auth_mock(user_id="005OTHER"),
        )


def test_start_conversation_rejects_auth_for_different_agent() -> None:
    ace = AgentContextClient()
    _wire(ace)
    with pytest.raises(ValueError, match="auth agent 'OtherAgent' does not match"):
        ace.start_conversation(
            conversation_id="ses-other", auth=_auth_mock(aid="OtherAgent"),
        )


def test_start_conversation_accepts_fresh_auth_for_same_user() -> None:
    ace = AgentContextClient()
    _wire(ace)
    out = ace.start_conversation(conversation_id="ses-1", auth=_auth_mock())
    assert out["conversation_id"] == "ses-1"


def test_start_conversation_rejects_unresolved_user_without_fetching() -> None:
    """A fresh UserContextAuth has no user_id until its first fetch. Probing it
    would cost a refresh + exchange and could rotate the refresh token, so the
    unverifiable auth is rejected (fail closed) instead of trusted."""
    ace = AgentContextClient()
    _wire(ace)
    ace.start_conversation(conversation_id="ses-1")
    fresh = _auth_mock(user_id="")
    with pytest.raises(ValueError, match="cannot verify the auth user"):
        ace.start_conversation(conversation_id="ses-1", auth=fresh)
    fresh.get_token.assert_not_called()


def test_start_conversation_rejects_real_fresh_user_auth_for_another_user() -> None:
    """Reviewer case: a brand-new UserContextAuth (another user's refresh
    token, same connected app) has user_id "" and must not be trusted."""
    ace = AgentContextClient()
    bound = _auth_mock(spec=UserContextAuth)
    bound._login_url, bound._consumer_key = "login.salesforce.com", "ck"
    _wire(ace, bound)
    fresh = UserContextAuth(
        refresh_token="user-b-refresh", consumer_key="ck",
        login_url="login.salesforce.com", aid="TestAgent",
    )
    with pytest.raises(ValueError, match="cannot verify the auth user"):
        ace.start_conversation(conversation_id="ses-1", auth=fresh)


def test_start_conversation_rejects_different_auth_kind() -> None:
    ace = AgentContextClient()
    _wire(ace, _auth_mock(spec=UserContextAuth))
    other = _auth_mock(user_id=None, default_owner_id="003A", spec=ContextAuth)
    with pytest.raises(ValueError, match="auth type ContextAuth does not match"):
        ace.start_conversation(conversation_id="ses-1", auth=other)


def test_start_conversation_rejects_context_auth_with_unset_vs_set_xid() -> None:
    """Unset on one side is a mismatch, not a skipped comparison."""
    ace = AgentContextClient()
    bound = _auth_mock(user_id=None, default_owner_id="003A", spec=ContextAuth)
    bound._xid = None
    _wire(ace, bound)
    other = _auth_mock(user_id=None, default_owner_id="003A", spec=ContextAuth)
    other._xid = "003B"
    with pytest.raises(ValueError, match="auth xid '003B' does not match"):
        ace.start_conversation(conversation_id="ses-1", auth=other)


def test_start_conversation_accepts_equivalent_context_auth() -> None:
    ace = AgentContextClient()
    bound = _auth_mock(user_id=None, default_owner_id="003A", spec=ContextAuth)
    bound._subject, bound._xid = "svc@example.com", "003A"
    _wire(ace, bound)
    other = _auth_mock(user_id=None, default_owner_id="003A", spec=ContextAuth)
    other._subject, other._xid = "svc@example.com", "003A"
    out = ace.start_conversation(conversation_id="ses-1", auth=other)
    assert out["conversation_id"] == "ses-1"


def test_start_conversation_compares_agent_with_resolved_agent_id() -> None:
    """The bound agent is the explicit agent_id, not the bound auth's aid."""
    ace = AgentContextClient()
    bound = _auth_mock(aid=None)
    _wire(ace, bound, agent_id="TestAgent")
    out = ace.start_conversation(
        conversation_id="ses-1", auth=_auth_mock(aid="TestAgent"),
    )
    assert out["conversation_id"] == "ses-1"


@pytest.mark.parametrize("attr", ["_xid", "_subject"])
def test_start_conversation_rejects_context_auth_for_different_owner(attr: str) -> None:
    ace = AgentContextClient()
    bound = _auth_mock(user_id=None, default_owner_id="003A")
    setattr(bound, attr, "owner-a")
    _wire(ace, bound)
    other = _auth_mock(user_id=None, default_owner_id="003A")
    setattr(other, attr, "owner-b")
    with pytest.raises(ValueError, match="'owner-b' does not match"):
        ace.start_conversation(conversation_id="ses-1", auth=other)


def test_start_conversation_accepts_same_auth() -> None:
    ace = AgentContextClient()
    auth = _auth_mock()
    _wire(ace, auth)
    out = ace.start_conversation(conversation_id="ses-1", auth=auth)
    assert out["conversation_id"] == "ses-1"


def test_deprecated_alias_warns_from_submodule() -> None:
    from agentic_context import ace_sdk_client

    with pytest.warns(DeprecationWarning, match="AceSdkClient is deprecated"):
        alias = ace_sdk_client.AceSdkClient
    assert alias is AgentContextClient


def test_deprecated_alias_warns() -> None:
    import agentic_context

    with pytest.warns(DeprecationWarning, match="AceSdkClient is deprecated; use AgentContextClient"):
        alias = agentic_context.AceSdkClient
    assert alias is agentic_context.AgentContextClient


@pytest.mark.parametrize("name", ["BeaconAuth", "UserBeaconAuth"])
def test_removed_beacon_auth_aliases(name: str) -> None:
    import agentic_context
    import agentic_context.auth

    assert not hasattr(agentic_context, name)
    assert not hasattr(agentic_context.auth, name)


def test_context_guidance_treats_memory_as_context() -> None:
    from agentic_context.ace_sdk_client import _CONTEXT_GUIDANCE

    assert "not as instructions" in _CONTEXT_GUIDANCE


# ─── Full skill-order scenario (host methods + model call_tool) ───


def test_skill_order_host_methods_and_model_tools() -> None:
    ace = AgentContextClient()
    auth = _auth_mock()
    with (
        patch("agentic_context.ace_sdk_client.IngestionClient"),
        patch("agentic_context.ace_sdk_client.RetrievalClient") as RC,
        patch("agentic_context.ace_sdk_client.SessionManager") as SM,
    ):
        RC.return_value.get_context.return_value = {"contextId": "c1"}
        RC.return_value.get_context_pack_list.return_value = {
            "contextPacks": [{"contextPackId": "ActiveHealthPlanCoverageV2"}],
        }
        RC.return_value.get_context_pack.return_value = {"enrollments": []}
        SM.return_value.create_session.return_value = {
            "sessionId": "ignored",
            "ownerId": "005OWNER",
            "agentId": "TestAgent",
        }

        ace.authenticate(auth=auth, agent_id="TestAgent")
        ids = ace.allocate_session_id(prefix="ses")
        cid = ids["conversation_id"]
        initial = ace.get_initial_context(cid)
        assert "ActiveHealthPlanCoverageV2" in initial["pack_ids"]
        opened = ace.open_conversation(cid, title="t")
        assert opened["ok"] is True
        ace.ingest(cid, role="user", content="hi")
        ace.call_tool(
            "ace_get_context_pack",
            {
                "conversation_id": cid,
                "context_pack_id": "ActiveHealthPlanCoverageV2",
                "params": {"query": "x"},
            },
        )
        ace.ingest(cid, role="assistant", content="hello")
        assert SM.return_value.ingest.call_count == 2
        # mid-turn seed blocked
        with pytest.raises(RuntimeError, match="get_context / get_context_pack"):
            ace.get_initial_context(cid)
