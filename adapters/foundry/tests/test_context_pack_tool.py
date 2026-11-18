"""Pack-scoped get_context_pack tool for the Foundry adapter.

The shared behaviour runs from ``context_pack_contract``; this file supplies the
Foundry harness plus the Foundry-only checks. Catalog refresh cadence and
failure handling live in ``test_adapter.py``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic_context_foundry import CONTEXT_PACK_INSTRUCTION as FOUNDRY_INSTRUCTION
from agentic_context_foundry.adapter import FoundryAdapter

pytest.register_assert_rewrite("context_pack_contract")
from context_pack_contract import PACK, ContextPackToolContract  # noqa: E402


class _Auth:
    def get_token(self) -> tuple[str, str]:
        return "token", "http://beacon.test"

    def attach(self, _client: object) -> None:
        return None


class _Harness:
    conversation_id = "ses-1"
    owner_id = "user@x.com"
    agent_id = "bot"
    instruction = FOUNDRY_INSTRUCTION

    def __init__(self) -> None:
        self.adapter = FoundryAdapter(agent_id=self.agent_id, owner_id=self.owner_id, auth=_Auth())
        self.adapter._manager = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(_owner_id=self.owner_id),
        )
        self.retrieval = self.adapter._retrieval_client
        self.retrieval.get_context_pack = AsyncMock(return_value=PACK)
        self.retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})

    def set_pack_response(self, return_value: Any = None, *, raises: Exception | None = None) -> None:
        self.retrieval.get_context_pack = AsyncMock(return_value=return_value, side_effect=raises)

    def make_tool(self, pack_ids: list[str]) -> Any:
        self.adapter._allowed_pack_ids = set(pack_ids)
        return self.adapter.make_get_context_pack_tool(self.conversation_id, self.owner_id)

    async def invoke(self, tool: Any, **args: Any) -> dict:
        return await tool.invoke(arguments=args, skip_parsing=True)

    async def load_catalog(self, packs: list[dict]) -> None:
        self.retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": packs})
        await self.adapter._maybe_inject_context_pack_list(is_open=True)


@pytest.fixture
async def pack_harness():
    harness = _Harness()
    yield harness
    await harness.adapter.close()


class TestContextPackContract(ContextPackToolContract):
    pass


# ─── Foundry-specific ─────────────────────────────────────────────────


async def test_pack_tool_forwards_object_params(pack_harness: _Harness) -> None:
    """``params`` may arrive as a JSON object (not a string); it reaches the
    retrieval client unchanged."""
    h = pack_harness
    tool = h.make_tool(["PlanRules"])

    await h.invoke(tool, context_pack_id="PlanRules", params={"a": "1"})

    assert h.retrieval.get_context_pack.await_args.kwargs["params"] == {"a": "1"}
