# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Shared contract for every adapter's ``get_context_pack`` tool.

Each adapter's test suite subclasses :class:`ContextPackToolContract` and
supplies a ``pack_harness`` fixture implementing :class:`PackHarness`. The
inherited tests then run against that adapter's real tool object, so the
behaviours every adapter must share (owner bound server-side, allowlist
gating, best-effort ``not_ready``, version passthrough, ``version_expired``
mapping, catalog versions) are written once.

Adapter-specific behaviour (float coercion, empty-string versions, param
merging rules, not_ready variants, catalog refresh cadence) stays in the
adapter's own test files.

This directory is test-only: it is on each suite's ``pythonpath`` and is
never packaged.
"""
from __future__ import annotations

from typing import Any, Protocol

import pytest

from agentic_context import CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_TIMEOUT_S

PACK = {"plan": {"name": "Gold PPO", "deductible": 500}}

_PARSED_ENTRY = (
    "id:PlanRules\n"
    "description:Plan rules\n"
    "version:9\n"
    "dataNeeds:[{name:plan,description:plan doc}]"
)
_UNPARSEABLE_ENTRY = "id:PlanRules\ndescription:no version line here"


class PackHarness(Protocol):
    """What an adapter's ``pack_harness`` fixture must provide.

    ``retrieval`` is the retrieval-client double the adapter calls; its
    ``get_context_pack`` returns :data:`PACK` until ``set_pack_response``
    changes it. ``conversation_id`` / ``owner_id`` / ``agent_id`` are the
    values the adapter must resolve on its own, never from the model.
    """

    adapter: Any
    retrieval: Any
    conversation_id: str
    owner_id: str
    agent_id: str
    instruction: str

    def set_pack_response(
        self, return_value: Any = None, *, raises: Exception | None = None,
    ) -> None: ...

    def make_tool(self, pack_ids: list[str]) -> Any: ...

    async def invoke(self, tool: Any, **args: Any) -> dict[str, Any]:
        """Call the tool the way its framework would; return the result dict."""
        ...

    async def load_catalog(self, packs: list[dict[str, Any]]) -> None:
        """Run the adapter's session-open catalog fetch with ``packs``."""
        ...


class ContextPackToolContract:
    """Inherit as ``class TestContextPackContract(ContextPackToolContract)``."""

    def test_instruction_is_shared_core_object(self, pack_harness: PackHarness) -> None:
        assert pack_harness.instruction is CONTEXT_PACK_INSTRUCTION
        # Core carries the versioning guidance so every adapter's LLM sees it.
        assert "context_pack_version" in CONTEXT_PACK_INSTRUCTION
        assert "version_expired" in CONTEXT_PACK_INSTRUCTION

    async def test_ok_binds_owner_server_side_and_returns_payload(
        self, pack_harness: PackHarness,
    ) -> None:
        h = pack_harness
        tool = h.make_tool(["PlanRules", "Claims"])

        result = await h.invoke(tool, context_pack_id="PlanRules")

        assert result["status"] == "ok"
        assert result["context"] == PACK
        h.retrieval.get_context_pack.assert_awaited_once()
        args, kwargs = h.retrieval.get_context_pack.call_args
        assert args[:4] == (h.conversation_id, h.owner_id, h.agent_id, "PlanRules")
        assert kwargs["owner_type"] == "individual"
        assert kwargs["timeout"] == CONTEXT_PACK_TIMEOUT_S

    async def test_invalid_pack_never_calls_beacon(self, pack_harness: PackHarness) -> None:
        h = pack_harness
        tool = h.make_tool(["PlanRules"])

        ok = await h.invoke(tool, context_pack_id="PlanRules")
        nope = await h.invoke(tool, context_pack_id="Nope")

        assert ok["status"] == "ok"
        assert nope["status"] == "invalid_pack"
        assert nope["valid_pack_ids"] == ["PlanRules"]
        h.retrieval.get_context_pack.assert_awaited_once()

    async def test_beacon_failure_degrades_to_not_ready(self, pack_harness: PackHarness) -> None:
        h = pack_harness
        h.set_pack_response(raises=RuntimeError("405 Method Not Allowed"))
        tool = h.make_tool(["PlanRules"])

        result = await h.invoke(tool, context_pack_id="PlanRules")

        assert result["status"] == "not_ready"
        # detail keeps the exception TYPE so a bare timeout stays diagnosable.
        assert "RuntimeError" in result["detail"]

    @pytest.mark.parametrize(("version", "expected"), [(5, 5), (None, None)],
                             ids=["passed-through", "omitted-sends-none"])
    async def test_context_pack_version_reaches_client(
        self, pack_harness: PackHarness, version: int | None, expected: int | None,
    ) -> None:
        """A model-supplied version lands on the client call; an omitted one
        is sent as ``None`` so retrieval.py drops the key from the body."""
        h = pack_harness
        tool = h.make_tool(["PlanRules"])
        args = {"context_pack_version": version} if version is not None else {}

        await h.invoke(tool, context_pack_id="PlanRules", **args)

        _, kwargs = h.retrieval.get_context_pack.call_args
        assert kwargs["context_pack_version"] == expected

    @pytest.mark.parametrize(("entry", "latest_version"),
                             [(_PARSED_ENTRY, 9), (_UNPARSEABLE_ENTRY, None)],
                             ids=["parsed-version", "unparseable-toon"])
    async def test_version_mismatch_maps_to_version_expired(
        self, pack_harness: PackHarness, entry: str, latest_version: int | None,
    ) -> None:
        """One ``contextPack`` userContext entry means the requested version is
        stale. Surface ``version_expired`` with the TOON-parsed latest version;
        when it can't be parsed, report ``None`` and drop the cached pin so the
        retry goes unpinned instead of re-sending the stale version."""
        h = pack_harness
        h.set_pack_response({"userContext": [{"sourceType": "contextPack", "entry": entry}]})
        h.adapter._pack_versions["PlanRules"] = 3
        tool = h.make_tool(["PlanRules"])

        result = await h.invoke(tool, context_pack_id="PlanRules", context_pack_version=3)

        assert result["status"] == "version_expired"
        assert result["context_pack_id"] == "PlanRules"
        assert result["requested_version"] == 3
        assert result["latest_version"] == latest_version
        assert result["latest"] == entry
        if latest_version is None:
            assert "PlanRules" not in h.adapter._pack_versions
        else:
            assert h.adapter._pack_versions["PlanRules"] == latest_version

    async def test_fetched_catalog_is_the_tool_allowlist(self, pack_harness: PackHarness) -> None:
        h = pack_harness
        tool = h.make_tool(["PlanRules"])

        await h.load_catalog([{"contextPackId": "Claims", "description": ""}])
        h.retrieval.get_context_pack_list.reset_mock()

        claimed = await h.invoke(tool, context_pack_id="Claims")
        stale = await h.invoke(tool, context_pack_id="PlanRules")

        assert claimed["status"] == "ok"
        assert stale["status"] == "invalid_pack"
        assert stale["valid_pack_ids"] == ["Claims"]
        # Tool calls read the cached catalog; they never refetch it.
        h.retrieval.get_context_pack_list.assert_not_awaited()

    async def test_catalog_block_surfaces_pack_versions(self, pack_harness: PackHarness) -> None:
        """The catalog JSON block carries ``version`` so the model can echo it
        back on the tool call."""
        h = pack_harness

        await h.load_catalog([{
            "contextPackId": "PlanRules", "description": "plan rules",
            "version": 7, "dataNeeds": [],
        }])

        assert '"version": 7' in h.adapter._allowed_context_pack_list
