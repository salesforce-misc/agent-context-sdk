# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for the on-demand memory tool on ClaudeAdapter.

Mirrors the ADK adapter's on-demand coverage, adapted to the Claude Agent SDK's
MCP-tool shape (@tool + create_sdk_mcp_server):

    - make_context_retrieval_tool() returns an SdkMcpTool wrapping recall_customer_context
    - default_tools() / mcp_server() are the batteries-included install path
    - the tool runs a single owner-scoped get_context read every call
    - the tool is the default retrieval seam in both modes (auto seeds
      context at open; manual is tool-only)
    - the recall read is audit-logged (has_context bool, never payload)
"""
from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# This module exercises the REAL claude_agent_sdk @tool / create_sdk_mcp_server
# (unlike test_hooks.py, which only needs HookMatcher and stubs the SDK). Other
# test modules may have installed a MagicMock stub into sys.modules first; evict
# it and import the real package. Skip the whole module if it isn't installed.
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
    reason="claude_agent_sdk not installed — on-demand tool tests need the real @tool",
)

from agentic_context_claude import ClaudeAdapter

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
    """Mock async clients + faked ContextAuth exchange (mirrors test_hooks.py)."""
    with patch("agentic_context_claude.adapter.AsyncIngestionClient") as MockIngestion, \
         patch("agentic_context_claude.adapter.AsyncRetrievalClient") as MockRetrieval, \
         patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        MockAuth.return_value = auth_inst

        ingestion = MagicMock()
        ingestion.add_entries = AsyncMock(return_value={})
        ingestion.close = AsyncMock(return_value=None)
        MockIngestion.return_value = ingestion

        retrieval = MagicMock()
        retrieval.get_context = AsyncMock(return_value={
            "contextId": "ctx-123",
            "userContext": [
                {"entry": "User prefers dark mode", "sourceType": "Profile"},
                {"entry": "Previously asked about order ORD-001", "sourceType": "Conversation"},
            ],
        })
        retrieval.list_conversations = AsyncMock(return_value=[])
        retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
        retrieval.close = AsyncMock(return_value=None)
        MockRetrieval.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval}


def _make_adapter():
    return ClaudeAdapter(
        agent_id="agent-test",
        owner_id="user-001",
        **CREDS,
        use_async=True,
    )


async def _invoke(tool_obj, args: dict) -> dict:
    """Call the SdkMcpTool's underlying handler and return its dict result."""
    # claude_agent_sdk's @tool exposes the coroutine as `.handler`.
    handler = getattr(tool_obj, "handler", None) or getattr(tool_obj, "_handler", None)
    assert handler is not None, f"no handler on tool {tool_obj!r}"
    return await handler(args)


def _text(result: dict) -> str:
    return result["content"][0]["text"]


async def _prompt_submit(options, conversation_id: str = "sdk") -> str:
    """Run the wired UserPromptSubmit hook once; return its additionalContext."""
    callback = options.hooks["UserPromptSubmit"][-1].hooks[0]
    result = await callback(
        {"session_id": conversation_id, "transcript_path": "/tmp/t.jsonl", "cwd": "/"},
        None,
        {"signal": None},
    )
    return result.get("hookSpecificOutput", {}).get("additionalContext", "")


# ─── The tool exists and is installable ───────────────────────────────────────


class TestToolSurface:
    def test_make_context_retrieval_tool_returns_sdk_tool(self, mock_async_clients):
        adapter = _make_adapter()
        t = adapter.make_context_retrieval_tool("conv-1", "user-001")
        assert getattr(t, "name", None) == "recall_customer_context"

    def test_default_tools_contains_recall_and_pack(self, mock_async_clients):
        """default_tools() exposes both the recall tool (owner-scoped profile
        read) and the pack-call tool (named-pack POST for data not in the
        recall)."""
        adapter = _make_adapter()
        tools = adapter.default_tools("conv-1", "user-001")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"recall_customer_context", "get_context_pack"}

    def test_mcp_server_builds(self, mock_async_clients):
        adapter = _make_adapter()
        server = adapter.mcp_server("conv-1", "user-001")
        # create_sdk_mcp_server returns an McpSdkServerConfig-shaped object.
        assert server is not None


# ─── System-prompt guidance (opt-in, additive) ────────────────────────────────


class TestSystemPromptGuidance:
    def test_default_guidance_names_tool_and_is_append_safe(self, mock_async_clients):
        adapter = _make_adapter()
        text = adapter.context_system_prompt_guidance()
        # Unlike the tool builders, guidance is pure text — no claude_agent_sdk
        # import, so it works even where the SDK isn't installed.
        assert isinstance(text, str)
        # Names the real tool the agent will see.
        assert "recall_customer_context" in text
        # Leading newline(s) so `MY_PROMPT + guidance` never runs words together.
        assert text.startswith("\n")
        combined = "You are a support agent." + text
        assert combined.startswith("You are a support agent.\n")

    def test_guidance_honors_custom_tool_name(self, mock_async_clients):
        adapter = _make_adapter()
        text = adapter.context_system_prompt_guidance(tool_name="lookup_history")
        assert "lookup_history" in text
        assert "recall_customer_context" not in text


# ─── One-call wiring: apply_to() (Option B over the primitives) ───────────────


class TestApplyTo:
    def _options(self, **kw):
        from claude_agent_sdk import ClaudeAgentOptions
        return ClaudeAgentOptions(**kw)

    def test_wires_server_tool_and_hooks(self, mock_async_clients):
        adapter = _make_adapter()
        opts = self._options(system_prompt="You are a support agent.")
        out = adapter.apply_to(opts, "conv-apply", "user-001")

        # context server registered under default key "context"
        assert "context" in out.mcp_servers
        # Both SDK-provided tools whitelisted on the wire. Recall must be
        # explicitly allowed so unattended agents don't hit the SDK's
        # permission layer when the model calls it.
        assert "mcp__context__get_context_pack" in out.allowed_tools
        assert "mcp__context__recall_customer_context" in out.allowed_tools
        # hooks wired (Stop + UserPromptSubmit)
        assert "Stop" in out.hooks and "UserPromptSubmit" in out.hooks
        # The instruction rides in the hook now; system_prompt is untouched.
        assert out.system_prompt == "You are a support agent."

    @pytest.mark.asyncio
    async def test_hook_injects_instruction_before_catalog(self, mock_async_clients):
        from agentic_context import CONTEXT_PACK_INSTRUCTION

        mock_async_clients["retrieval"].get_system_prompt = AsyncMock(return_value="SERVER INSTR")
        mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
        adapter = _make_adapter()
        out = adapter.apply_to(self._options(system_prompt="P"), "conv-apply-i", "user-001")
        ctx = await _prompt_submit(out)
        assert "SERVER INSTR" in ctx
        assert CONTEXT_PACK_INSTRUCTION not in ctx
        assert ctx.index("SERVER INSTR") < ctx.index("[Available context packs]")
        # Pinned for the session: a new Claude conversation reuses it without
        # refetching.
        assert "SERVER INSTR" in await _prompt_submit(out, "sdk-2")
        mock_async_clients["retrieval"].get_system_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hook_skips_instruction_already_in_system_prompt(self, mock_async_clients):
        from agentic_context import CONTEXT_PACK_INSTRUCTION

        mock_async_clients["retrieval"].get_system_prompt = AsyncMock(return_value="SERVER INSTR")
        mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
        adapter = _make_adapter()
        opts = self._options(system_prompt=f"P\n\n{CONTEXT_PACK_INSTRUCTION}")
        out = adapter.apply_to(opts, "conv-apply-dup", "user-001")
        ctx = await _prompt_submit(out)
        assert "PlanRules" in ctx
        assert "SERVER INSTR" not in ctx
        assert CONTEXT_PACK_INSTRUCTION not in ctx

    @pytest.mark.asyncio
    async def test_hook_skips_server_instruction_in_preset_append(self, mock_async_clients):
        mock_async_clients["retrieval"].get_system_prompt = AsyncMock(return_value="SERVER INSTR")
        mock_async_clients["retrieval"].get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
        adapter = _make_adapter()
        opts = self._options(system_prompt={
            "type": "preset", "preset": "claude_code", "append": "P\n\nSERVER INSTR",
        })
        out = adapter.apply_to(opts, "conv-apply-dup2", "user-001")
        ctx = await _prompt_submit(out)
        assert "PlanRules" in ctx
        assert "SERVER INSTR" not in ctx

    def test_does_not_mutate_input_options(self, mock_async_clients):
        adapter = _make_adapter()
        opts = self._options(system_prompt="P", allowed_tools=["some_other_tool"])
        out = adapter.apply_to(opts, "conv-apply2", "user-001")
        assert out is not opts
        # original untouched
        assert opts.allowed_tools == ["some_other_tool"]
        assert opts.system_prompt == "P"
        assert opts.mcp_servers == {}

    def test_preserves_existing_servers_and_tools(self, mock_async_clients):
        adapter = _make_adapter()
        sentinel_server = object()
        opts = self._options(
            mcp_servers={"weather": sentinel_server},
            allowed_tools=["mcp__weather__forecast"],
        )
        out = adapter.apply_to(opts, "conv-apply3", "user-001")
        # existing server preserved, memory added alongside
        assert out.mcp_servers["weather"] is sentinel_server
        assert "context" in out.mcp_servers
        # existing tool preserved (appended, not replaced); both SDK tools whitelisted
        assert "mcp__weather__forecast" in out.allowed_tools
        assert "mcp__context__get_context_pack" in out.allowed_tools
        assert "mcp__context__recall_customer_context" in out.allowed_tools

    def test_custom_server_name(self, mock_async_clients):
        adapter = _make_adapter()
        opts = self._options(mcp_servers={"context": object()})
        # A different server_name sidesteps the clobber and renames the wire tool.
        out = adapter.apply_to(opts, "conv-apply5", "user-001", server_name="packs")
        assert "packs" in out.mcp_servers
        # Custom server_name reaches both tool wire names.
        assert "mcp__packs__get_context_pack" in out.allowed_tools
        assert "mcp__packs__recall_customer_context" in out.allowed_tools

    def test_idempotent_double_apply(self, mock_async_clients):
        adapter = _make_adapter()
        opts = self._options(system_prompt="P")
        once = adapter.apply_to(opts, "conv-apply6", "user-001")
        # Second apply must not duplicate either whitelisted tool wire name.
        twice = adapter.apply_to(
            once, "conv-apply6", "user-001", add_system_prompt_guidance=False,
        )
        assert twice.allowed_tools.count("mcp__context__get_context_pack") == 1
        assert twice.allowed_tools.count("mcp__context__recall_customer_context") == 1
        # ...nor register our hooks twice (that would ingest every turn twice).
        for event in ("UserPromptSubmit", "Stop", "PreCompact"):
            assert len(twice.hooks[event]) == 1

    def test_double_apply_keeps_caller_hooks(self, mock_async_clients):
        from claude_agent_sdk import HookMatcher
        adapter = _make_adapter()
        mine = HookMatcher(matcher=".*", hooks=[lambda *a: {}])
        opts = self._options(hooks={"Stop": [mine]})
        once = adapter.apply_to(opts, "conv-apply8", "user-001")
        twice = adapter.apply_to(once, "conv-apply8", "user-001")
        assert twice.hooks["Stop"][0] is mine
        assert len(twice.hooks["Stop"]) == 2
        assert len(twice.hooks["UserPromptSubmit"]) == 1

    def test_second_adapter_hooks_are_kept(self, mock_async_clients):
        """Only this adapter's own matchers are replaced — another adapter's
        hooks on the same options are the caller's business."""
        first, second = _make_adapter(), _make_adapter()
        opts = first.apply_to(self._options(), "conv-apply9", "user-001")
        out = second.apply_to(
            opts, "conv-apply9", "user-001", add_system_prompt_guidance=False,
        )
        assert len(out.hooks["Stop"]) == 2

    def test_merges_existing_hooks(self, mock_async_clients):
        from claude_agent_sdk import HookMatcher
        adapter = _make_adapter()
        mine = HookMatcher(matcher=".*", hooks=[lambda *a: {}])
        opts = self._options(hooks={"Stop": [mine], "PreToolUse": [mine]})
        out = adapter.apply_to(opts, "conv-apply7", "user-001")
        # my Stop matcher survives, ours appended after it
        assert out.hooks["Stop"][0] is mine
        assert len(out.hooks["Stop"]) == 2
        # my unrelated event is preserved untouched
        assert out.hooks["PreToolUse"] == [mine]
        # ours-only event added
        assert "UserPromptSubmit" in out.hooks

    @pytest.mark.parametrize(
        "system_prompt",
        [
            None,
            {"type": "preset", "preset": "claude_code"},
            {"type": "preset", "preset": "claude_code", "append": "EXISTING"},
            {"type": "file", "path": "/tmp/sp.txt"},
        ],
    )
    def test_system_prompt_is_left_unchanged(self, mock_async_clients, system_prompt):
        adapter = _make_adapter()
        out = adapter.apply_to(
            self._options(system_prompt=system_prompt), "conv-apply8", "user-001",
        )
        assert out.system_prompt == system_prompt

    @pytest.mark.asyncio
    async def test_opt_out_of_guidance(self, mock_async_clients):
        adapter = _make_adapter()
        opts = self._options(system_prompt="P")
        out = adapter.apply_to(
            opts, "conv-apply12", "user-001", add_system_prompt_guidance=False,
        )
        assert out.system_prompt == "P"
        assert "CONTEXT PACKS" not in await _prompt_submit(out)

    @pytest.mark.parametrize(
        ("make_options", "exc", "match"),
        [
            (lambda o: o(mcp_servers={"context": object()}), ValueError,
             "overwrite an existing MCP server"),
            (lambda o: o(mcp_servers="/path/to/mcp.json"), TypeError, "filesystem config"),
            (lambda o: {"not": "options"}, TypeError, "ClaudeAgentOptions dataclass"),
        ],
        ids=["server-key-clobber", "filesystem-mcp-config", "non-dataclass-options"],
    )
    def test_unusable_options_raise(self, mock_async_clients, make_options, exc, match):
        adapter = _make_adapter()
        opts = make_options(self._options)
        with pytest.raises(exc, match=match):
            adapter.apply_to(opts, "conv-apply-err", "user-001")


# ─── Recall tool behaviour (single owner-scoped read) ─────────────────────────
# The recall tool does a single get_context read every call; the OpenAPI
# GetContext schema only takes identifier fields (conversation_id, owner_id,
# agent_id, owner_type), so no mode/query/latency/budget params go over the wire.


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_user_context(self, mock_async_clients):
        adapter = _make_adapter()
        # Prime a session so _last_session_id resolves.
        adapter.hooks("conv-recall", "user-001")
        tool_obj = adapter.make_context_retrieval_tool("conv-recall", "user-001")

        result = await _invoke(tool_obj, {"topic": "billing"})
        assert "User prefers dark mode" in _text(result)
        calls = mock_async_clients["retrieval"].get_context.call_args_list
        assert len(calls) == 1
        # Only identifier fields go to Beacon — the OpenAPI schema is
        # additionalProperties=false on GetContextRequest.
        kw = calls[0].kwargs
        for forbidden in ("mode", "query", "latency_sensitivity", "token_budget"):
            assert forbidden not in kw

    @pytest.mark.asyncio
    async def test_empty_context_returns_empty_userContext(self, mock_async_clients):
        mock_async_clients["retrieval"].get_context.return_value = None
        adapter = _make_adapter()
        adapter.hooks("conv-empty", "user-001")
        tool_obj = adapter.make_context_retrieval_tool("conv-empty", "user-001")
        result = await _invoke(tool_obj, {})
        assert '"userContext": []' in _text(result)


# ─── Audit logging (parity with the injection path, no PII) ───────────────────


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_recall_logs_has_context(self, mock_async_clients, caplog):
        adapter = _make_adapter()
        adapter.hooks("conv-log", "user-001")
        tool_obj = adapter.make_context_retrieval_tool("conv-log", "user-001")
        with caplog.at_level(logging.INFO, logger="agentic_context_claude.adapter"):
            await _invoke(tool_obj, {"topic": "billing"})
        audit = [r for r in caplog.records if "recall_customer_context" in r.getMessage()]
        assert len(audit) == 1
        msg = audit[0].getMessage()
        assert "has_context=True" in msg

    @pytest.mark.asyncio
    async def test_recall_log_omits_payload(self, mock_async_clients, caplog):
        adapter = _make_adapter()
        adapter.hooks("conv-log2", "user-001")
        tool_obj = adapter.make_context_retrieval_tool("conv-log2", "user-001")
        with caplog.at_level(logging.INFO, logger="agentic_context_claude.adapter"):
            await _invoke(tool_obj, {"topic": "account"})
        audit = [r for r in caplog.records if "recall_customer_context" in r.getMessage()]
        assert len(audit) == 1
        msg = audit[0].getMessage()
        # Never log the retrieved payload.
        assert "User prefers dark mode" not in msg
