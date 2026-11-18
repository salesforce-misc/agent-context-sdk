# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for the pack-scoped get_context_pack tool on ClaudeAdapter.

Parity with the ADK/LangGraph ``make_get_context_pack_tool`` coverage, adapted to
the Claude Agent SDK's MCP-tool shape (@tool). The tool wraps
:meth:`agentic_context.RetrievalClient.get_context_pack` (POST GetContext) and is
best-effort: it NEVER raises, returning ``invalid_pack`` for an unadvertised id,
``not_ready`` on any failure (endpoint down / sync mode / timeout), and ``ok``
with the pack payload on success. ``owner_id`` is bound server-side from the
resolved session owner — never taken from the model. The LLM-facing schema is
``context_pack_id`` and a JSON ``params`` object of data-need parameters.
"""
from __future__ import annotations

import importlib
import json as _json
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Exercise the REAL claude_agent_sdk @tool (mirrors test_on_demand_tool.py): evict
# any MagicMock stub other modules installed, then import the real package. Skip
# the whole module if it isn't installed.
_stub = sys.modules.get("claude_agent_sdk")
if _stub is not None and isinstance(_stub, MagicMock):
    del sys.modules["claude_agent_sdk"]
try:
    import claude_agent_sdk  # noqa: F401
    importlib.reload(claude_agent_sdk)
    _HAS_REAL_SDK = not isinstance(sys.modules["claude_agent_sdk"], MagicMock)
except ImportError:
    _HAS_REAL_SDK = False

pytestmark = pytest.mark.skipif(
    not _HAS_REAL_SDK,
    reason="claude_agent_sdk not installed — pack tool tests need the real @tool",
)

from agentic_context import CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_TIMEOUT_S
from agentic_context_claude import ClaudeAdapter
from agentic_context_claude import CONTEXT_PACK_INSTRUCTION as CL_INSTRUCTION

pytest.register_assert_rewrite("context_pack_contract")
from context_pack_contract import PACK, ContextPackToolContract  # noqa: E402

CREDS = {
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
    "consumer_key": "3MVG.test",
    "subject": "user@org.com",
    "login_url": "login.salesforce.com",
}
FAKE_TOKEN = "test-key"
FAKE_BASE_URL = "https://beacon.test"

@pytest.fixture
def mock_async_clients():
    with patch("agentic_context_claude.adapter.AsyncIngestionClient") as MockIngestion, \
         patch("agentic_context_claude.adapter.AsyncRetrievalClient") as MockRetrieval, \
         patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        auth_inst.default_owner_type = "individual"
        MockAuth.return_value = auth_inst

        ingestion = MagicMock()
        ingestion.close = AsyncMock(return_value=None)
        MockIngestion.return_value = ingestion

        retrieval = MagicMock()
        # Prime a fast profile read so hooks()/manager resolution stays happy, and
        # the pack POST used by these tests.
        retrieval.get_context = AsyncMock(return_value={"userContext": []})
        retrieval.get_context_pack = AsyncMock(return_value=PACK)
        retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
        retrieval.close = AsyncMock(return_value=None)
        MockRetrieval.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval}


def _make_adapter(**overrides):
    kwargs = dict(agent_id="agent-test", owner_id="user-001", **CREDS, use_async=True)
    kwargs.update(overrides)
    return ClaudeAdapter(**kwargs)


async def _invoke(tool_obj, args: dict) -> dict:
    handler = getattr(tool_obj, "handler", None) or getattr(tool_obj, "_handler", None)
    assert handler is not None, f"no handler on tool {tool_obj!r}"
    return await handler(args)


def _payload(result: dict) -> dict:
    return _json.loads(result["content"][0]["text"])


class _Harness:
    conversation_id = "conv-1"
    owner_id = "user-001"
    agent_id = "agent-test"
    instruction = CL_INSTRUCTION

    def __init__(self, clients) -> None:
        self.retrieval = clients["retrieval"]
        self.adapter = _make_adapter(context_mode="manual")
        hooks = self.adapter.hooks(self.conversation_id, self.owner_id)
        self._on_prompt = hooks["UserPromptSubmit"][0].hooks[0]

    def set_pack_response(self, return_value=None, *, raises=None) -> None:
        self.retrieval.get_context_pack = AsyncMock(return_value=return_value, side_effect=raises)

    def make_tool(self, pack_ids):
        self.adapter._allowed_pack_ids = set(pack_ids)
        return self.adapter.make_get_context_pack_tool(self.conversation_id, self.owner_id)

    async def invoke(self, tool, **args) -> dict:
        return _payload(await _invoke(tool, args))

    async def load_catalog(self, packs) -> None:
        self.retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": packs})
        await self._on_prompt({}, None, None)


@pytest.fixture
def pack_harness(mock_async_clients) -> _Harness:
    return _Harness(mock_async_clients)


class TestContextPackContract(ContextPackToolContract):
    pass


# ─── Claude-specific ──────────────────────────────────────────────────


class TestToolSurface:
    def test_returns_sdk_tool_named_get_context_pack(self, mock_async_clients):
        adapter = _make_adapter()
        t = adapter.make_get_context_pack_tool("conv-1", "user-001")
        assert getattr(t, "name", None) == "get_context_pack"

    def test_default_timeout_is_core_constant(self, mock_async_clients):
        import inspect
        sig = inspect.signature(_make_adapter().make_get_context_pack_tool)
        assert sig.parameters["timeout"].default == CONTEXT_PACK_TIMEOUT_S


class TestPackReads:
    @pytest.mark.asyncio
    async def test_param_defaults_merge_and_drop_query(self, pack_harness):
        h = pack_harness
        h.adapter._allowed_pack_ids = {"PlanRules"}
        tool = h.adapter.make_get_context_pack_tool(
            "conv-1", "user-001",
            param_defaults={"policy_type": "MEDICAL"},
        )

        await h.invoke(tool, context_pack_id="PlanRules",
                       params='{"person_id": "005xx"}', query="copay")

        # A leftover query argument is not a getContext parameter.
        _, kwargs = h.retrieval.get_context_pack.call_args
        assert kwargs["params"] == {"policy_type": "MEDICAL", "person_id": "005xx"}

        # Model values override param_defaults; blank values are dropped.
        await h.invoke(tool, context_pack_id="PlanRules",
                       params='{"person_id": "005xx", "policy_id": "  ", "policy_type": "DENTAL"}')
        _, kwargs = h.retrieval.get_context_pack.call_args
        assert kwargs["params"] == {"person_id": "005xx", "policy_type": "DENTAL"}

    @pytest.mark.asyncio
    async def test_empty_catalog_rejects_every_pack(self, pack_harness):
        h = pack_harness
        tool = h.adapter.make_get_context_pack_tool("conv-1", "user-001")
        await h.load_catalog([])

        result = await h.invoke(tool, context_pack_id="PlanRules")

        assert h.adapter._allowed_pack_ids == set()
        assert h.adapter._allowed_context_pack_list == "[Available context packs]\n{}"
        # With no constructor fallback, an empty catalog means EVERY pack id is
        # invalid — the model learns to stop calling the tool until the catalog
        # populates.
        assert result["status"] == "invalid_pack"
        assert result["valid_pack_ids"] == []
        h.retrieval.get_context_pack.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_version_empty_string_treated_as_none(self, pack_harness):
        """Claude's MCP tool schema types ``context_pack_version`` as int but
        the model sometimes emits ``""`` — the handler coerces that to
        ``None`` so Beacon isn't passed garbage."""
        h = pack_harness
        tool = h.make_tool(["PlanRules"])

        await h.invoke(tool, context_pack_id="PlanRules", context_pack_version="")

        _, kwargs = h.retrieval.get_context_pack.call_args
        assert kwargs["context_pack_version"] is None


# ─── Server-side context_pack_instruction (pinned per session + fallback) ──


class TestServerContextPackInstruction:
    @pytest.mark.asyncio
    async def test_returns_server_text_pinned_per_session(self, mock_async_clients):
        rc = mock_async_clients["retrieval"]
        rc.get_system_prompt = AsyncMock(side_effect=["SERVER V1", "SERVER V2"])
        adapter = _make_adapter()

        assert await adapter.get_context_pack_instruction("conv-1") == "SERVER V1"
        assert await adapter.get_context_pack_instruction("conv-1") == "SERVER V1"
        assert await adapter.get_context_pack_instruction("conv-2") == "SERVER V2"
        assert rc.get_system_prompt.await_count == 2
        rc.get_system_prompt.assert_awaited_with("context_pack_instruction")

    @pytest.mark.asyncio
    async def test_404_falls_back_to_bundled_constant(self, mock_async_clients):
        rc = mock_async_clients["retrieval"]
        rc.get_system_prompt = AsyncMock(return_value=None)
        adapter = _make_adapter()

        assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION

    @pytest.mark.asyncio
    async def test_error_pins_fallback_for_the_session(self, mock_async_clients):
        rc = mock_async_clients["retrieval"]
        rc.get_system_prompt = AsyncMock(side_effect=[RuntimeError("boom"), "SERVER PROMPT"])
        adapter = _make_adapter()

        assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
        # No retry mid-session — the session keeps one instruction.
        assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
        # Within the failure cooldown a new session also skips the fetch.
        assert await adapter.get_context_pack_instruction("conv-2") == CONTEXT_PACK_INSTRUCTION
        assert rc.get_system_prompt.await_count == 1

    @pytest.mark.asyncio
    async def test_hook_injects_instruction_before_catalog_every_turn(
        self, mock_async_clients,
    ):
        rc = mock_async_clients["retrieval"]
        rc.get_system_prompt = AsyncMock(return_value="SERVER PROMPT")
        rc.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
        adapter = _make_adapter()
        callback = adapter.hooks("conv-1", "user-001")["UserPromptSubmit"][0].hooks[0]

        # No conversation id in the hook input, so both blocks go every turn.
        for _ in range(2):
            ctx = (await callback({}, None, None))["hookSpecificOutput"]["additionalContext"]
            assert ctx.index("SERVER PROMPT") < ctx.index("[Available context packs]")
        rc.get_system_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hook_opt_out_skips_instruction(self, mock_async_clients):
        rc = mock_async_clients["retrieval"]
        rc.get_system_prompt = AsyncMock(return_value="SERVER PROMPT")
        rc.get_context_pack_list = AsyncMock(
            return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]}
        )
        adapter = _make_adapter()
        callback = adapter.hooks(
            "conv-1", "user-001", inject_context_pack_instruction=False,
        )["UserPromptSubmit"][0].hooks[0]

        ctx = (await callback({}, None, None))["hookSpecificOutput"]["additionalContext"]
        assert "SERVER PROMPT" not in ctx
        rc.get_system_prompt.assert_not_awaited()


# ─── Sync hook path ──────────────────────────────────────────────────


@pytest.fixture
def mock_sync_clients():
    with patch("agentic_context_claude.adapter.IngestionClient") as MockIngestion, \
         patch("agentic_context_claude.adapter.RetrievalClient") as MockRetrieval, \
         patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        auth_inst.default_owner_type = "individual"
        MockAuth.return_value = auth_inst

        MockIngestion.return_value = MagicMock()
        retrieval = MagicMock()
        retrieval.get_context_pack_list = MagicMock(
            return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]}
        )
        MockRetrieval.return_value = retrieval

        yield {"retrieval": retrieval}


def test_sync_hook_injects_server_instruction(mock_sync_clients):
    retrieval = mock_sync_clients["retrieval"]
    retrieval.get_system_prompt = MagicMock(return_value="SERVER PROMPT")
    adapter = _make_adapter(use_async=False, context_mode="manual")
    callback = adapter.hooks("conv-1", "user-001")["UserPromptSubmit"][0].hooks[0]

    for _ in range(2):
        ctx = callback({}, None, None)["hookSpecificOutput"]["additionalContext"]
        assert ctx.index("SERVER PROMPT") < ctx.index("[Available context packs]")
    retrieval.get_system_prompt.assert_called_once_with("context_pack_instruction")


def test_sync_hook_404_falls_back(mock_sync_clients):
    mock_sync_clients["retrieval"].get_system_prompt = MagicMock(return_value=None)
    adapter = _make_adapter(use_async=False, context_mode="manual")
    callback = adapter.hooks("conv-1", "user-001")["UserPromptSubmit"][0].hooks[0]

    ctx = callback({}, None, None)["hookSpecificOutput"]["additionalContext"]
    assert CONTEXT_PACK_INSTRUCTION in ctx
