# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Pack-scoped get_context_pack tool for the OpenAI adapter.

Parity with the Claude/ADK/LangGraph ``make_get_context_pack_tool`` coverage.
The tool wraps :meth:`agentic_context.AsyncRetrievalClient.get_context_pack`
(POST GetContext) and is best-effort: it NEVER raises, returning
``invalid_pack`` for an unadvertised id, ``not_ready`` on any failure (sync
mode / no session / request error), and ``ok`` with the pack payload on
success. ``session_id`` / owner / owner_type / agent_id are resolved from the
adapter (turn context), never from the model — the LLM-facing args are just
``context_pack_id`` + ``params``.
"""
from __future__ import annotations

import importlib.util
import json
from unittest.mock import AsyncMock

import pytest

from agentic_context import CONTEXT_PACK_INSTRUCTION
from agentic_context_openai import CONTEXT_PACK_INSTRUCTION as OPENAI_INSTRUCTION
from agentic_context_openai import OpenAIAdapter

from .test_adapter import CREDS, mock_async_clients, mock_sync_clients  # noqa: F401

pytest.register_assert_rewrite("context_pack_contract")
from context_pack_contract import PACK, ContextPackToolContract  # noqa: E402

_AGENTS_SDK_AVAILABLE = importlib.util.find_spec("agents") is not None

pytestmark = pytest.mark.skipif(
    not _AGENTS_SDK_AVAILABLE,
    reason="openai-agents not installed",
)

# ─── Invocation helper ────────────────────────────────────────────────

async def _invoke(tool, **kwargs) -> dict:
    """Invoke an Agents SDK FunctionTool and parse its JSON string result."""
    from agents.tool_context import ToolContext
    ctx = ToolContext(
        context=None, tool_name=tool.name,
        tool_call_id="call_test", tool_arguments={},
    )
    raw = await tool.on_invoke_tool(ctx, json.dumps(kwargs))
    return json.loads(raw)


def _adapter_with_session(mock_async_clients, pack_return=PACK):
    """Build an adapter with one open session so the pack tool can resolve ids."""
    adapter = OpenAIAdapter(
        agent_id="test-agent", owner_id="user@test.com",
        owner_type="individual", **CREDS,
    )
    # Create a live manager for session "conv-1" so _last_session_id resolves.
    manager = adapter._get_or_create_manager("conv-1")
    # MagicMock-returned manager — anchor _owner_id so getattr resolves the str.
    manager._owner_id = "user@test.com"
    mock_async_clients["retrieval"].get_context_pack = AsyncMock(return_value=pack_return)
    return adapter


class _Harness:
    conversation_id = "conv-1"
    owner_id = "user@test.com"
    agent_id = "test-agent"
    instruction = OPENAI_INSTRUCTION

    def __init__(self, clients) -> None:
        self.retrieval = clients["retrieval"]
        self.adapter = _adapter_with_session(clients, PACK)

    def set_pack_response(self, return_value=None, *, raises=None) -> None:
        self.retrieval.get_context_pack = AsyncMock(return_value=return_value, side_effect=raises)

    def make_tool(self, pack_ids):
        return self.adapter.make_get_context_pack_tool(pack_ids, session_id=self.conversation_id)

    async def invoke(self, tool, **args) -> dict:
        return await _invoke(tool, params="", **args)

    async def load_catalog(self, packs) -> None:
        self.retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": packs})
        provider = self.adapter.instructions_provider("BASE", self.conversation_id)
        await provider(run_context=None, agent=None)


@pytest.fixture
def pack_harness(mock_async_clients) -> _Harness:
    return _Harness(mock_async_clients)


class TestContextPackContract(ContextPackToolContract):
    pass


# ─── OpenAI-specific ──────────────────────────────────────────────────

def test_returns_function_tool(mock_async_clients) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    tool = adapter.make_get_context_pack_tool(["PlanRules"])

    assert tool.__class__.__name__ == "FunctionTool"
    assert tool.name == "get_context_pack"


@pytest.mark.asyncio
async def test_sync_mode_degrades_to_not_ready(mock_sync_clients) -> None:
    adapter = OpenAIAdapter(agent_id="a", owner_id="u", use_async=False, **CREDS)
    adapter._get_or_create_manager("conv-1")
    tool = adapter.make_get_context_pack_tool(["PlanRules"])

    result = await _invoke(tool, context_pack_id="PlanRules", params="")

    assert result["status"] == "not_ready"
    assert "async mode" in result["detail"]


@pytest.mark.asyncio
async def test_param_defaults_merge(mock_async_clients) -> None:
    adapter = _adapter_with_session(mock_async_clients, PACK)
    tool = adapter.make_get_context_pack_tool(
        ["PlanRules"], session_id="conv-1",
        param_defaults={"policy_type": "MEDICAL"},
    )

    await _invoke(tool, context_pack_id="PlanRules",
                  params=json.dumps({"zip": "94105"}))

    _, kwargs = mock_async_clients["retrieval"].get_context_pack.call_args
    # defaults + model-supplied params merge, defaults do NOT override model values.
    assert kwargs["params"] == {"policy_type": "MEDICAL", "zip": "94105"}

    await _invoke(tool, context_pack_id="PlanRules",
                  params=json.dumps({"policy_type": "DENTAL"}))
    _, kwargs = mock_async_clients["retrieval"].get_context_pack.call_args
    assert kwargs["params"] == {"policy_type": "DENTAL"}


# ─── default_tools ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "kwargs", [{"valid_pack_ids": ["PlanRules"]}, {}], ids=["pinned-packs", "no-pack-list"],
)
def test_default_tools_returns_both(mock_async_clients, kwargs) -> None:
    """Default (no valid_pack_ids) still returns both tools — the pack tool
    is just not useful until the server catalog populates ``_allowed_pack_ids``."""
    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    tools = adapter.default_tools(**kwargs)

    assert len(tools) == 2
    assert {t.name for t in tools} == {"recall_customer_context", "get_context_pack"}


# ─── Eager-seed regression ────────────────────────────────────────────

def test_eager_seed_populates_allowed_pack_ids(mock_async_clients) -> None:
    """Load-bearing for the whole pack-tool test cluster: constructing the
    pack tool with pinned ids must publish them to ``_allowed_pack_ids``
    immediately, so a tools-only integrator (no ``instructions_provider``
    wired) still has a working pack tool. The ``TestContextPackContract``
    tests depend on this behavior — without the eager seed, they would get
    ``invalid_pack`` for every id."""
    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    adapter.make_get_context_pack_tool(["PlanRules", "Claims"])

    assert adapter._allowed_pack_ids == {"PlanRules", "Claims"}


def test_second_pack_tool_with_different_pinned_raises(mock_async_clients) -> None:
    """One adapter supports one pinned set only — a second call with a
    different list would race on the shared ``_allowed_pack_ids`` state."""
    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    adapter.make_get_context_pack_tool(["PlanRules"])

    # Same list is fine (idempotent).
    adapter.make_get_context_pack_tool(["PlanRules"])
    # Different list raises.
    with pytest.raises(RuntimeError, match="already called"):
        adapter.make_get_context_pack_tool(["Claims"])


# ─── Server-side context_pack_instruction (pinned per session + fallback) ──

@pytest.mark.asyncio
async def test_get_context_pack_instruction_pins_server_text_per_session(
    mock_async_clients,
) -> None:
    from unittest.mock import AsyncMock

    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    retrieval = mock_async_clients["retrieval"]
    retrieval.get_system_prompt = AsyncMock(side_effect=["SERVER V1", "SERVER V2"])

    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("conv-2") == "SERVER V2"
    assert retrieval.get_system_prompt.await_count == 2
    retrieval.get_system_prompt.assert_awaited_with("context_pack_instruction")


@pytest.mark.asyncio
async def test_get_context_pack_instruction_404_falls_back_to_bundled(
    mock_async_clients,
) -> None:
    from unittest.mock import AsyncMock

    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    mock_async_clients["retrieval"].get_system_prompt = AsyncMock(return_value=None)

    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION


@pytest.mark.asyncio
async def test_get_context_pack_instruction_error_pins_fallback_for_session(
    mock_async_clients,
) -> None:
    from unittest.mock import AsyncMock

    adapter = OpenAIAdapter(agent_id="a", owner_id="u", **CREDS)
    retrieval = mock_async_clients["retrieval"]
    retrieval.get_system_prompt = AsyncMock(
        side_effect=[RuntimeError("boom"), "SERVER PROMPT"],
    )

    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
    assert await adapter.get_context_pack_instruction("conv-1") == CONTEXT_PACK_INSTRUCTION
    # Within the failure cooldown a new session also skips the fetch.
    assert await adapter.get_context_pack_instruction("conv-2") == CONTEXT_PACK_INSTRUCTION
    assert retrieval.get_system_prompt.await_count == 1


@pytest.mark.asyncio
async def test_get_context_pack_instruction_sync_client_runs_off_loop(
    mock_sync_clients,
) -> None:
    from unittest.mock import MagicMock

    adapter = OpenAIAdapter(agent_id="a", owner_id="u", use_async=False, **CREDS)
    retrieval = mock_sync_clients["retrieval"]
    retrieval.get_system_prompt = MagicMock(return_value="SERVER PROMPT")

    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER PROMPT"
    assert await adapter.get_context_pack_instruction("conv-1") == "SERVER PROMPT"
    retrieval.get_system_prompt.assert_called_once_with("context_pack_instruction")
