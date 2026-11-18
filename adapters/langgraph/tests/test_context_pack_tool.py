# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Pack-scoped get_context_pack tool for the LangGraph adapter.

Parity with the ADK/Claude ``make_get_context_pack_tool`` coverage. The tool wraps
:meth:`agentic_context.AsyncRetrievalClient.get_context_pack` (POST GetContext) and
is best-effort: it NEVER raises, returning ``invalid_pack`` for an unadvertised id,
``not_ready`` when there is no owned retrieval client or the call fails, and ``ok``
with the pack payload on success. Session identity + owner are resolved from the
per-invocation ``RunnableConfig`` (``configurable.thread_id`` == ``session_id``),
so the tool is invoked here via ``tool.ainvoke({...}, config=...)`` — the same path
LangGraph's ToolNode/bind_tools use. ``owner_id`` is bound server-side, never taken
from the model (the LLM-facing schema is just ``context_pack_id`` + ``query``).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from agentic_context import CONTEXT_PACK_INSTRUCTION
from agentic_context_langgraph import CONTEXT_PACK_INSTRUCTION as LG_INSTRUCTION
from agentic_context_langgraph import LangGraphAdapter

from .conftest import BASE_URL, _build_graph

pytest.register_assert_rewrite("context_pack_contract")
from context_pack_contract import PACK, ContextPackToolContract  # noqa: E402


def _make_adapter(retrieval_client: Any = None, **overrides: Any) -> LangGraphAdapter:
    kwargs: dict[str, Any] = {
        "graph": _build_graph(),
        "agent_id": "bot",
        "owner_id": "u@o.com",
        "api_key": "tok",
        "base_url": BASE_URL,
        "dataspace": "default",
    }
    kwargs.update(overrides)
    adapter = LangGraphAdapter(**kwargs)
    adapter._owned_retrieval_client = retrieval_client
    return adapter


def _mock_retrieval(return_value: Any = PACK) -> AsyncMock:
    client = AsyncMock()
    client.get_context_pack = AsyncMock(return_value=return_value)
    return client


async def _invoke(tool: StructuredTool, *, sid: str | None = "ses-1", **args: Any) -> Any:
    config = {"configurable": {"thread_id": sid}} if sid is not None else {}
    return await tool.ainvoke(args, config=config)


class _Harness:
    conversation_id = "ses-1"
    owner_id = "u@o.com"
    agent_id = "bot"
    instruction = LG_INSTRUCTION

    def __init__(self) -> None:
        self.retrieval = _mock_retrieval(PACK)
        self.adapter = _make_adapter(self.retrieval)

    def set_pack_response(self, return_value: Any = None, *, raises: Exception | None = None) -> None:
        self.retrieval.get_context_pack = AsyncMock(return_value=return_value, side_effect=raises)

    def make_tool(self, pack_ids: list[str]) -> StructuredTool:
        return self.adapter.make_get_context_pack_tool(pack_ids)

    async def invoke(self, tool: StructuredTool, **args: Any) -> dict:
        return await _invoke(tool, sid=self.conversation_id, **args)

    async def load_catalog(self, packs: list[dict]) -> None:
        self.retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": packs})
        await self.adapter._maybe_inject_context_pack_list({"messages": []}, is_open=True)


@pytest.fixture
def pack_harness() -> _Harness:
    return _Harness()


class TestContextPackContract(ContextPackToolContract):
    pass


# ─── LangGraph-specific ───────────────────────────────────────────────


def test_returns_structured_tool_hiding_config() -> None:
    adapter = _make_adapter(_mock_retrieval())
    tool = adapter.make_get_context_pack_tool(["PlanRules"])
    assert isinstance(tool, StructuredTool)
    assert isinstance(tool, BaseTool)
    assert tool.coroutine is not None
    assert tool.coroutine.__name__ == "get_context_pack"
    # context_pack_id, params, query, and context_pack_version are LLM-visible;
    # config is hidden.
    assert list(tool.args.keys()) == [
        "context_pack_id", "params", "query", "context_pack_version",
    ]


async def test_custom_owner_type_propagates() -> None:
    client = _mock_retrieval(PACK)
    adapter = _make_adapter(client, owner_type="account")
    tool = adapter.make_get_context_pack_tool(["PlanRules"])

    await _invoke(tool, sid="ses-1", context_pack_id="PlanRules")

    _, kwargs = client.get_context_pack.call_args
    assert kwargs["owner_type"] == "account"


async def test_no_owned_client_degrades_to_not_ready() -> None:
    # Manager-injection mode (or pre-__aenter__): no owned retrieval client.
    adapter = _make_adapter(retrieval_client=None)
    tool = adapter.make_get_context_pack_tool(["PlanRules"])

    result = await _invoke(tool, sid="ses-1", context_pack_id="PlanRules")

    assert result["status"] == "not_ready"
    assert "retrieval client" in result["detail"]


async def test_param_defaults_merge_with_query() -> None:
    client = _mock_retrieval(PACK)
    adapter = _make_adapter(client)
    tool = adapter.make_get_context_pack_tool(
        ["PlanRules"], param_defaults={"policy_type": "MEDICAL"},
    )

    await _invoke(tool, sid="ses-1", context_pack_id="PlanRules", query="copay")

    _, kwargs = client.get_context_pack.call_args
    assert kwargs["params"] == {"policy_type": "MEDICAL", "query": "copay"}


# ─── context-pack catalog (discovery) field projection ────────────────

async def test_empty_catalog_keeps_constructor_fallback() -> None:
    client = _mock_retrieval(PACK)
    client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter = _make_adapter(client)
    tool = adapter.make_get_context_pack_tool(["PlanRules"])

    await adapter._maybe_inject_context_pack_list({"messages": []}, is_open=True)
    client.get_context_pack_list.reset_mock()
    await adapter._maybe_inject_context_pack_list({"messages": []}, is_open=False)
    client.get_context_pack_list.assert_not_awaited()
    result = await _invoke(tool, sid="ses-1", context_pack_id="PlanRules")

    assert adapter._allowed_pack_ids == set()
    assert adapter._allowed_context_pack_list == "[Available context packs]\n{}"
    assert result["status"] == "ok"


# ─── Server-side context_pack_instruction (pinned per session + fallback) ──


async def test_get_context_pack_instruction_pins_server_text_per_session() -> None:
    client = _mock_retrieval()
    client.get_system_prompt = AsyncMock(side_effect=["SERVER V1", "SERVER V2"])
    adapter = _make_adapter(client)

    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("ses-2") == "SERVER V2"
    assert client.get_system_prompt.await_count == 2
    client.get_system_prompt.assert_awaited_with("context_pack_instruction")


async def test_get_context_pack_instruction_404_falls_back_to_bundled() -> None:
    client = _mock_retrieval()
    client.get_system_prompt = AsyncMock(return_value=None)
    adapter = _make_adapter(client)

    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION


async def test_get_context_pack_instruction_error_pins_fallback_for_session() -> None:
    client = _mock_retrieval()
    client.get_system_prompt = AsyncMock(side_effect=[RuntimeError("boom"), "SERVER PROMPT"])
    adapter = _make_adapter(client)

    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION
    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION
    # Within the failure cooldown a new session also skips the fetch.
    assert await adapter.get_context_pack_instruction("ses-2") == CONTEXT_PACK_INSTRUCTION
    assert client.get_system_prompt.await_count == 1


async def test_get_context_pack_instruction_manager_injection_falls_back() -> None:
    # Manager-injection mode with no `retrieval_client=` owns no retrieval
    # client — helper must still return the bundled constant instead of
    # raising.
    adapter = _make_adapter(retrieval_client=None)

    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION


async def test_manager_injection_uses_injected_retrieval_client() -> None:
    # Manager-injection mode with an injected `retrieval_client=` fetches the
    # server prompt through that client.
    from unittest.mock import MagicMock
    rc = _mock_retrieval()
    rc.get_system_prompt = AsyncMock(return_value="SERVER PROMPT")
    session_manager = MagicMock()
    session_manager.owner_id = "u@o.com"
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        session_manager=session_manager,
        retrieval_client=rc,
    )

    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER PROMPT"
    rc.get_system_prompt.assert_awaited_once_with("context_pack_instruction")


def test_retrieval_client_requires_session_manager() -> None:
    # Passing retrieval_client= in self-driving mode is a configuration
    # error — the adapter builds its own, so an injected one would be
    # silently ignored.
    with pytest.raises(ValueError, match="manager-injection"):
        LangGraphAdapter(
            graph=_build_graph(),
            agent_id="bot",
            owner_id="u@o.com",
            api_key="tok",
            base_url=BASE_URL,
            dataspace="default",
            retrieval_client=_mock_retrieval(),
        )


async def test_injected_system_message_carries_instruction_then_catalog() -> None:
    from langchain_core.messages import SystemMessage

    client = _mock_retrieval()
    client.get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    client.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
    adapter = _make_adapter(client)

    out = await adapter._maybe_inject_context_pack_list(
        {"messages": []}, is_open=True, session_id="ses-1",
    )
    msg = out["messages"][0]
    assert isinstance(msg, SystemMessage)
    assert msg.id == "agentic-context-pack-list"
    assert msg.content.startswith("SERVER INSTR\n\n[Available context packs]\n{\"PlanRules\"")

    # Later turns reuse the pinned text.
    await adapter._maybe_inject_context_pack_list({"messages": []}, session_id="ses-1")
    client.get_system_prompt.assert_awaited_once()


async def test_injection_without_session_uses_bundled_instruction() -> None:
    client = _mock_retrieval()
    client.get_context_pack_list = AsyncMock(return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]})
    adapter = _make_adapter(client)

    out = await adapter._maybe_inject_context_pack_list({"messages": []}, is_open=True)
    assert out["messages"][0].content.startswith(CONTEXT_PACK_INSTRUCTION)
    client.get_system_prompt.assert_not_awaited()


async def test_end_session_forgets_pinned_instruction() -> None:
    client = _mock_retrieval()
    client.get_system_prompt = AsyncMock(side_effect=["SERVER V1", "SERVER V2"])
    adapter = _make_adapter(client)

    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER V1"
    await adapter.end_session("ses-1")
    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER V2"


def test_owner_type_reaches_ingestion_manager() -> None:
    # Reads (get_context / get_context_pack) and writes (create_conversation /
    # create_session ownerType) must agree on owner_type, or the conversation is
    # ingested as one owner type and read back as another.
    adapter = _make_adapter(_mock_retrieval(), owner_type="employee")
    adapter._owned_ingest_client = AsyncMock()

    mgr = adapter._mgr_for("ses-1", "u@o.com")

    assert mgr._owner_type == "employee"
