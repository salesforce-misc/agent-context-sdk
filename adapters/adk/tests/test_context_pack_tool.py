# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Pack-scoped get_context_pack tool for the Google ADK adapter.

Parity with the Claude/LangGraph ``make_get_context_pack_tool`` coverage. The tool
wraps :meth:`agentic_context.AsyncRetrievalClient.get_context_pack` (POST
GetContext) and is best-effort: it NEVER raises, returning ``invalid_pack`` for an
unadvertised id, ``not_ready`` on any failure (client init / no client / call
error), and ``ok`` with the pack payload on success. ``conversation_id`` / owner /
owner_type / agent_id are resolved from the adapter (turn context), never from the
model — the LLM-facing args are just ``context_pack_id`` + ``query``.
"""
from __future__ import annotations

import warnings as _warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_context import CONTEXT_PACK_INSTRUCTION
from agentic_context_adk import CONTEXT_PACK_INSTRUCTION as ADK_INSTRUCTION
from agentic_context_adk import GoogleAdkAdapter

pytest.register_assert_rewrite("context_pack_contract")
from context_pack_contract import PACK, ContextPackToolContract  # noqa: E402

_ADK_IMPORT_WARNING = ("ignore", ".*_UnionGenericAlias.*", DeprecationWarning)
_ADK_BASE_AGENT_CONFIG_WARNING = ("ignore", ".*BaseAgentConfig.*", DeprecationWarning)


def _adapter_with_mock_pack(pack_return=PACK, *,
                            retrieval: object | None = ...) -> GoogleAdkAdapter:
    """Adapter wired so the pack tool resolves a live client + turn context.

    Pins ``context_mode="manual"`` (no auto-seed) and mocks the pack-tool seams:
    ``_ensure_clients`` (no-op), ``_retrieval`` (the pack client), and
    ``_current_manager`` (a stub carrying ``_session_id``). Owner resolves to the
    construction-time ``_default_owner`` because no turn ContextVar is set.
    """
    mgr = AsyncMock()
    adapter = GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com", session_manager=mgr,
        context_mode="manual",
    )
    adapter._ensure_clients = AsyncMock()
    if retrieval is ...:
        rc = AsyncMock()
        rc.get_context_pack = AsyncMock(return_value=pack_return)
        adapter._retrieval = rc
    else:
        adapter._retrieval = retrieval  # e.g. None to exercise the no-client path
    adapter._current_manager = MagicMock(
        return_value=SimpleNamespace(_session_id="conv-1"),
    )
    return adapter


def _make_pack_tool(adapter, valid_pack_ids, **kw):
    with _warnings.catch_warnings():
        _warnings.filterwarnings(*_ADK_IMPORT_WARNING)
        _warnings.filterwarnings(*_ADK_BASE_AGENT_CONFIG_WARNING)
        return adapter.make_get_context_pack_tool(valid_pack_ids, **kw)


class _Harness:
    conversation_id = "conv-1"
    owner_id = "user@test.com"
    agent_id = "test-agent"
    instruction = ADK_INSTRUCTION

    def __init__(self) -> None:
        self.adapter = _adapter_with_mock_pack(PACK)
        self.retrieval = self.adapter._retrieval

    def set_pack_response(self, return_value=None, *, raises=None) -> None:
        self.retrieval.get_context_pack = AsyncMock(return_value=return_value, side_effect=raises)

    def make_tool(self, pack_ids):
        return _make_pack_tool(self.adapter, pack_ids)

    async def invoke(self, tool, **args) -> dict:
        return await tool.func(**args)

    async def load_catalog(self, packs) -> None:
        self.retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": packs})
        await self.adapter._inject_context_pack_list(
            SimpleNamespace(config=SimpleNamespace(system_instruction="BASE")),
            is_conversation_open=True,
        )


@pytest.fixture
def pack_harness() -> _Harness:
    return _Harness()


class TestContextPackContract(ContextPackToolContract):
    pass


# ─── ADK-specific ─────────────────────────────────────────────────────


def test_returns_function_tool():
    adapter = _adapter_with_mock_pack()
    tool = _make_pack_tool(adapter, ["PlanRules"])
    assert hasattr(tool, "func") and callable(tool.func)
    assert tool.__class__.__name__ == "FunctionTool"
    assert tool.func.__name__ == "get_context_pack"


@pytest.mark.asyncio
async def test_no_client_degrades_to_not_ready():
    adapter = _adapter_with_mock_pack(retrieval=None)
    tool = _make_pack_tool(adapter, ["PlanRules"])

    result = await tool.func(context_pack_id="PlanRules")

    assert result["status"] == "not_ready"
    assert "retrieval client" in result["detail"]


@pytest.mark.asyncio
async def test_param_defaults_merge_with_query():
    adapter = _adapter_with_mock_pack(PACK)
    tool = _make_pack_tool(adapter, ["PlanRules"], param_defaults={"policy_type": "MEDICAL"})

    await tool.func(context_pack_id="PlanRules", query="copay")

    _, kwargs = adapter._retrieval.get_context_pack.call_args
    assert kwargs["params"] == {"policy_type": "MEDICAL", "query": "copay"}


@pytest.mark.asyncio
async def test_empty_catalog_keeps_constructor_fallback():
    adapter = _adapter_with_mock_pack(PACK)
    adapter._retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    tool = _make_pack_tool(adapter, ["PlanRules"])

    req = SimpleNamespace(config=SimpleNamespace(system_instruction="BASE"))
    await adapter._inject_context_pack_list(req, is_conversation_open=True)
    adapter._retrieval.get_context_pack_list.reset_mock()
    await adapter._inject_context_pack_list(req, is_conversation_open=False)
    adapter._retrieval.get_context_pack_list.assert_not_awaited()
    result = await tool.func(context_pack_id="PlanRules")

    assert adapter._allowed_pack_ids == set()
    assert adapter._allowed_context_pack_list == "[Available context packs]\n{}"
    assert result["status"] == "ok"


# ─── Server-side context_pack_instruction (pinned per session + fallback) ──


@pytest.mark.asyncio
async def test_get_context_pack_instruction_pins_server_text_per_session():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_system_prompt = AsyncMock(side_effect=["SERVER V1", "SERVER V2"])

    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("conv-2") == "SERVER V2"
    assert adapter._retrieval.get_system_prompt.await_count == 2
    adapter._retrieval.get_system_prompt.assert_awaited_with("context_pack_instruction")


@pytest.mark.asyncio
async def test_get_context_pack_instruction_404_falls_back_to_bundled():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_system_prompt = AsyncMock(return_value=None)

    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION


@pytest.mark.asyncio
async def test_get_context_pack_instruction_error_pins_fallback_for_session():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_system_prompt = AsyncMock(
        side_effect=[RuntimeError("boom"), "SERVER PROMPT"],
    )

    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
    # Within the failure cooldown a new session also skips the fetch.
    assert await adapter.get_context_pack_instruction("conv-2") == CONTEXT_PACK_INSTRUCTION
    assert adapter._retrieval.get_system_prompt.await_count == 1


@pytest.mark.asyncio
async def test_get_context_pack_instruction_no_client_falls_back_unpinned():
    # Manager-injection or pre-_ensure_clients scenarios leave _retrieval=None;
    # the helper degrades to the bundled text without pinning it, so a later
    # turn with a client still fetches.
    adapter = _adapter_with_mock_pack(retrieval=None)
    adapter._ensure_clients = AsyncMock()  # should NOT be awaited

    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
    adapter._ensure_clients.assert_not_awaited()

    rc = AsyncMock()
    rc.get_system_prompt = AsyncMock(return_value="SERVER PROMPT")
    adapter._retrieval = rc
    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER PROMPT"


# ─── Instruction injection in the before-model callback ──────────────


def _llm_request(system_instruction):
    return SimpleNamespace(config=SimpleNamespace(system_instruction=system_instruction))


@pytest.mark.asyncio
async def test_inject_puts_instruction_between_base_and_catalog():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    adapter._retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})

    req = _llm_request("BASE")
    await adapter._inject_context_pack_list(req, is_conversation_open=True, session_id="conv-1")

    assert req.config.system_instruction.startswith(
        "BASE\n\nSERVER INSTR\n\n[Available context packs]\n{\"PlanRules\""
    )


@pytest.mark.asyncio
async def test_inject_skips_instruction_when_catalog_is_empty():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})

    req = _llm_request("BASE")
    await adapter._inject_context_pack_list(req, is_conversation_open=True, session_id="conv-1")

    assert req.config.system_instruction == "BASE\n\n[Available context packs]\n{}"
    adapter._retrieval.get_system_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_inject_without_session_uses_bundled_instruction():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})

    req = _llm_request(None)
    await adapter._inject_context_pack_list(req, is_conversation_open=True)

    assert req.config.system_instruction.startswith(CONTEXT_PACK_INSTRUCTION)
    adapter._retrieval.get_system_prompt.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_there", ["server", "bundled"])
async def test_inject_skips_instruction_already_in_agent_prompt(already_there):
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    adapter._retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
    text = "SERVER INSTR" if already_there == "server" else CONTEXT_PACK_INSTRUCTION
    base = f"{text}\n\nMY PROMPT"

    req = _llm_request(base)
    await adapter._inject_context_pack_list(req, is_conversation_open=True, session_id="conv-1")

    assert req.config.system_instruction.startswith(f"{base}\n\n[Available context packs]\n")


@pytest.mark.asyncio
async def test_inject_is_idempotent_within_a_request():
    adapter = _adapter_with_mock_pack()
    adapter._retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})

    req = _llm_request("BASE")
    await adapter._inject_context_pack_list(req, is_conversation_open=True, session_id="conv-1")
    once = req.config.system_instruction
    await adapter._inject_context_pack_list(req, session_id="conv-1")

    assert req.config.system_instruction == once


@pytest.mark.asyncio
async def test_whole_number_float_version_is_accepted():
    """Gemini may emit integer args as floats (3.0); ADK doesn't coerce."""
    adapter = _adapter_with_mock_pack(PACK)
    tool = _make_pack_tool(adapter, ["PlanRules"])

    result = await tool.func(context_pack_id="PlanRules", context_pack_version=3.0)

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_fractional_float_version_is_rejected():
    adapter = _adapter_with_mock_pack(PACK)
    tool = _make_pack_tool(adapter, ["PlanRules"])

    result = await tool.func(context_pack_id="PlanRules", context_pack_version=3.5)

    assert result["status"] == "invalid_version"
