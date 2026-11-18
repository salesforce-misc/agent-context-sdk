from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from agent_framework import AgentSession, Content, Message, SessionContext

from agentic_context_foundry.adapter import FoundryAdapter
from agentic_context_foundry.context_provider import collect_turn_records
from agentic_context_foundry.transformer import RecordTransformer


class _Rec:
    def __init__(self, event_id: str) -> None:
        self.event_id = event_id


class _Auth:
    def get_token(self) -> tuple[str, str]:
        return "token", "http://beacon.test"

    def attach(self, _client: object) -> None:
        return None


def _adapter(**kwargs: Any) -> FoundryAdapter:
    adapter = FoundryAdapter(agent_id="bot", owner_id="user@x.com", auth=_Auth(), **kwargs)
    # beacon.test doesn't resolve; keep before_run's instruction fetch offline.
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)
    return adapter


def _context(session_id: str, messages: list[Message]) -> SessionContext:
    return SessionContext(session_id=session_id, input_messages=messages)


def _pack_payload(pack_id: str) -> dict:
    return {"contextPacks": [{"contextPackId": pack_id, "description": "d", "dataNeeds": []}]}


@pytest.mark.asyncio
async def test_pack_tool_validates_against_this_turns_snapshot_not_live_cache() -> None:
    """A concurrent session's catalog refresh mid-turn must not invalidate a
    pack this turn's own instructions already showed as available."""
    adapter = _adapter()
    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(
        return_value=_pack_payload("PlanRules")
    )
    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-race")
    context = _context("ses-race", [])

    await provider.before_run(
        agent=None, session=session, context=context,
        state=session.state.setdefault(provider.source_id, {}),
    )
    pack_tool = next(t for t in context.tools if t.name == "get_context_pack")

    # Simulate a different, concurrent session refreshing the shared cache
    # mid-turn, after this turn's instructions/tools were already built.
    adapter._allowed_pack_ids = {"SomeOtherPack"}

    adapter._retrieval_client.get_context_pack = AsyncMock(return_value={})
    result = await pack_tool.func("PlanRules")
    assert result["status"] != "invalid_pack"


@pytest.mark.asyncio
async def test_before_run_rejects_owner_mismatch_before_capture() -> None:
    adapter = _adapter()
    await adapter._manager("ses-bound", "alice")
    provider = adapter.context_provider(owner_id="bob")
    session = AgentSession(session_id="ses-bound")
    context = _context("ses-bound", [Message("user", [Content.from_text("hi")])])
    state = session.state.setdefault(provider.source_id, {})

    with pytest.raises(ValueError, match="bound to owner"):
        await provider.before_run(
            agent=None, session=session, context=context, state=state,
        )

    assert "ses-bound" not in adapter._captures
    assert adapter._turn.get("ses-bound", 0) == 0
    await adapter.close()


@pytest.mark.asyncio
async def test_before_run_injects_instructions_tools_and_middleware() -> None:
    adapter = _adapter()
    adapter.get_context = AsyncMock(
        return_value={"userContext": [{"entry": "prefers email"}]}
    )
    adapter._retrieval_client.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": []}
    )
    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-1")
    context = _context("ses-1", [Message("user", [Content.from_text("hi")])])
    state = session.state.setdefault(provider.source_id, {})

    await provider.before_run(agent=None, session=session, context=context, state=state)

    assert any("prefers email" in instr for instr in context.instructions)
    assert [t.name for t in context.tools] == [
        "recall_customer_context",
        "get_context_pack",
    ]
    assert len(context.get_middleware()) == 2
    assert adapter._captures["ses-1"]["input_records"][0].content == "hi"
    assert adapter._agent_sessions["ses-1"] is session


@pytest.mark.asyncio
async def test_before_run_skips_instructions_on_repeat_turn() -> None:
    adapter = _adapter()
    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": []}
    )
    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-2")

    first_ctx = _context("ses-2", [])
    await provider.before_run(
        agent=None, session=session, context=first_ctx,
        state=session.state.setdefault(provider.source_id, {}),
    )
    adapter.get_context.assert_awaited_once()

    second_ctx = _context("ses-2", [])
    await provider.before_run(
        agent=None, session=session, context=second_ctx,
        state=session.state.setdefault(provider.source_id, {}),
    )
    adapter.get_context.assert_awaited_once()  # not called again on turn 2


@pytest.mark.asyncio
async def test_before_run_injects_instruction_before_catalog() -> None:
    adapter = _adapter()
    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    adapter._retrieval_client.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]}
    )
    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-instr")

    for _ in range(2):
        context = _context("ses-instr", [])
        await provider.before_run(
            agent=None, session=session, context=context,
            state=session.state.setdefault(provider.source_id, {}),
        )
        instructions = list(context.instructions)
        assert "SERVER INSTR" in instructions
        catalog = next(i for i, text in enumerate(instructions)
                       if text.startswith("[Available context packs]"))
        assert instructions.index("SERVER INSTR") < catalog
    # Pinned for the session — fetched once across both turns.
    adapter._retrieval_client.get_system_prompt.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_there", ["server", "bundled"])
async def test_before_run_skips_instruction_already_in_agent(already_there: str) -> None:
    from types import SimpleNamespace

    from agentic_context import CONTEXT_PACK_INSTRUCTION

    adapter = _adapter()
    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value="SERVER INSTR")
    adapter._retrieval_client.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]}
    )
    text = "SERVER INSTR" if already_there == "server" else CONTEXT_PACK_INSTRUCTION
    agent = SimpleNamespace(default_options={"instructions": f"{text}\n\nMY PROMPT"})
    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-dup")
    context = _context("ses-dup", [])

    await provider.before_run(
        agent=agent, session=session, context=context,
        state=session.state.setdefault(provider.source_id, {}),
    )
    assert "SERVER INSTR" not in context.instructions
    assert CONTEXT_PACK_INSTRUCTION not in context.instructions


def test_collect_turn_records_keeps_one_final_assistant_reply() -> None:
    tx = RecordTransformer("bot", "user@x.com")
    sid, iid = "s", "turn-s-0"
    response_msg = Message("assistant", [Content.from_text("hi back")])
    from_messages = tx.from_message(response_msg, sid, iid, 1)
    assert from_messages[0].event_id == "msg-s-1"
    chat_mw = type("_MW", (), {"records": [
        tx.agent_reasoning("think", sid, iid),
        tx.agent_response("hi back", sid, iid),
    ]})()
    capture = {
        "transformer": tx,
        "session_id": sid,
        "interaction_id": iid,
        "history_len": 0,
        "input_message_count": 1,
        "input_records": [],
        "chat_mw": chat_mw,
        "func_mw": None,
    }
    context = _context(sid, [])
    context._response = type("_Resp", (), {"messages": [response_msg]})()

    records = collect_turn_records(capture, context)
    types = [r.event_type for r in records]
    assert types.count("agent_response") == 1
    assert types == ["agent_response", "agent_reasoning"]
    assert records[0].event_id == "msg-s-1"
    assert records[0].content == "hi back"


@pytest.mark.asyncio
async def test_before_run_uses_owner_override_from_session_state() -> None:
    adapter = FoundryAdapter(agent_id="bot", owner_id="alice", auth=_Auth())
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)
    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": []}
    )
    provider = adapter.context_provider()
    session = AgentSession(session_id="ses-owner")
    context = _context("ses-owner", [Message("user", [Content.from_text("hi")])])
    state = session.state.setdefault(provider.source_id, {})
    state["owner_override"] = "bob"

    await provider.before_run(agent=None, session=session, context=context, state=state)

    assert "owner_override" not in state
    assert adapter._captures["ses-owner"]["owner"] == "bob"
    assert adapter._captures["ses-owner"]["transformer"].owner_id == "bob"
    adapter.get_context.assert_awaited_once()
    assert adapter.get_context.await_args.kwargs["owner_id"] == "bob"
    await adapter.close()


@pytest.mark.asyncio
async def test_after_run_ingests_captured_records_and_flushes() -> None:
    adapter = _adapter()
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    adapter._manager.return_value = manager

    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-3")
    chat_mw = type("_MW", (), {"records": [_Rec("r1")]})()
    func_mw = type("_MW", (), {"records": [_Rec("r2")]})()
    adapter._captures["ses-3"] = {
        "session_id": "ses-3",
        "owner": "user@x.com",
        "transformer": RecordTransformer("bot", "user@x.com"),
        "interaction_id": "turn-ses-3-0",
        "history_len": 0,
        "input_message_count": 0,
        "input_records": [],
        "chat_mw": chat_mw,
        "func_mw": func_mw,
        "skip_after_run_ingest": False,
    }

    await provider.after_run(
        agent=None, session=session, context=_context("ses-3", []), state={},
    )

    manager.ingest.assert_awaited_once()
    ingested = manager.ingest.await_args.args[0]
    assert [r.event_id for r in ingested] == ["r1", "r2"]
    manager.flush.assert_awaited_once()
    assert "ses-3" in adapter._captures


@pytest.mark.parametrize(
    "capture",
    [
        None,
        {
            "session_id": "ses-5",
            "owner": "user@x.com",
            "input_records": [_Rec("r1")],
            "skip_after_run_ingest": True,
        },
    ],
    ids=["no-capture", "skip-flag-set"],
)
@pytest.mark.asyncio
async def test_after_run_skips_ingest(capture: dict | None) -> None:
    adapter = _adapter()
    adapter._manager = AsyncMock()  # type: ignore[method-assign]

    provider = adapter.context_provider(owner_id="user@x.com")
    session = AgentSession(session_id="ses-5")
    if capture is not None:
        adapter._captures["ses-5"] = capture

    await provider.after_run(
        agent=None, session=session, context=_context("ses-5", []), state={},
    )

    adapter._manager.assert_not_called()
    # A skipped capture is left for its owner (stream()) to pop.
    assert ("ses-5" in adapter._captures) is (capture is not None)
