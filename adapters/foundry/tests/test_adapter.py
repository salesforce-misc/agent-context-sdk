from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agent_framework import AgentSession, Content, Message, SessionContext

from agentic_context_foundry.adapter import FoundryAdapter
from agentic_context_foundry.transformer import RecordTransformer, stream_tool_args_ready


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


def test_transformer_maps_five_event_types() -> None:
    tx = RecordTransformer("bot", "user@x.com")
    sid, iid = "ses-1", "turn-ses-1-0"
    user = tx.from_message(Message("user", [Content.from_text("hi")]), sid, iid, 0)
    assert user[0].event_type == "user_message"
    assert user[0].content == "hi"
    assert user[0].participant_id == "user@x.com"

    assistant = tx.from_message(
        Message("assistant", [Content.from_text("hello")]), sid, iid, 1,
    )
    assert assistant[0].event_type == "agent_response"

    reasoning = tx.from_message(
        Message("assistant", [Content.from_text_reasoning(text="think")]), sid, iid, 2,
    )
    assert reasoning[0].event_type == "agent_reasoning"
    assert reasoning[0].variable["OutputValueText"] == "think"

    call = tx.from_message(
        Message("assistant", [
            Content.from_function_call(call_id="c1", name="recall_customer_context", arguments={"query": "refund"}),
        ]),
        sid, iid, 3,
    )
    assert call[0].event_type == "tool_call"
    assert call[0].event_id == "c1"
    assert call[0].variable["Name"] == "recall_customer_context"

    result = tx.from_message(
        Message("tool", [Content.from_function_result(call_id="c1", result="ok")]),
        sid, iid, 4,
    )
    assert result[0].event_type == "tool_result"
    assert result[0].event_id == "c1-result"
    assert result[0].variable["Name"] == "recall_customer_context"
    assert result[0].variable["OutputValueText"] == "ok"

    # Content.from_function_call's real signature allows arguments as a raw
    # JSON string (not just a Mapping) — must not be silently dropped.
    json_call = tx.from_message(
        Message("assistant", [
            Content.from_function_call(
                call_id="c2", name="get_context_pack", arguments='{"context_pack_id": "PlanRules"}',
            ),
        ]),
        sid, iid, 5,
    )
    assert json_call[0].variable["InputValueText"] == '{"context_pack_id": "PlanRules"}'


def test_stream_update_skips_partial_tool_args() -> None:
    tx = RecordTransformer("bot", "user@x.com")
    sid, iid = "ses-1", "turn-ses-1-0"
    assert stream_tool_args_ready("{") is False
    assert stream_tool_args_ready('{"q": "refund"}') is True

    partial = tx.from_update(
        Message("assistant", [
            Content.from_function_call(call_id="c1", name="recall_customer_context", arguments="{"),
        ]),
        sid, iid, 0, tools_only=True,
    )
    assert partial == []

    complete = tx.from_update(
        Message("assistant", [
            Content.from_function_call(
                call_id="c1", name="recall_customer_context", arguments='{"q": "refund"}',
            ),
        ]),
        sid, iid, 1, tools_only=True,
    )
    assert complete[0].event_type == "tool_call"
    assert complete[0].event_id == "c1"
    assert complete[0].variable["InputValueText"] == '{"q": "refund"}'


@pytest.mark.parametrize(
    "state",
    [
        {"messages": [
            Message("user", [Content.from_text("one")]),
            Message("assistant", [Content.from_text("two")]),
        ]},
        # InMemoryHistoryProvider's namespace wins over a stale top-level
        # ``messages`` list.
        {
            "messages": [Message("user", [Content.from_text("stale")])],
            "in_memory": {"messages": [
                Message("user", [Content.from_text("one")]),
                Message("assistant", [Content.from_text("two")]),
            ]},
        },
    ],
    ids=["session-messages", "in-memory-history-provider-namespace"],
)
def test_snapshot_reads_session_messages_in_order(state: dict) -> None:
    adapter = _adapter()
    session = AgentSession(session_id="ses-1")
    session.state.update(state)
    adapter._agent_sessions["ses-1"] = session
    records = adapter.snapshot("ses-1")
    assert [r.event_type for r in records] == ["user_message", "agent_response"]
    assert [r.content for r in records] == ["one", "two"]
    assert records[0].interaction_id == "turn-ses-1-snapshot"


def test_snapshot_unknown_session_raises() -> None:
    with pytest.raises(KeyError):
        _adapter().snapshot("missing")


def test_snapshot_accepts_rehydrated_session_without_stream() -> None:
    adapter = _adapter()
    persisted = AgentSession(session_id="ses-cold")
    persisted.state["messages"] = [Message("user", [Content.from_text("one")])]
    rehydrated = AgentSession.from_dict(persisted.to_dict())

    records = adapter.snapshot("ses-cold", session=rehydrated)
    assert [r.content for r in records] == ["one"]
    assert "ses-cold" not in adapter._agent_sessions


@pytest.mark.asyncio
async def test_owner_type_propagates_to_session_manager() -> None:
    adapter = _adapter(owner_type="employee")
    manager = await adapter._manager("ses-1", "a@x.com")
    assert manager._owner_type == "employee"
    await adapter.close()


@pytest.mark.asyncio
async def test_session_manager_carries_framework_agent_label() -> None:
    adapter = _adapter()
    manager = await adapter._manager("ses-1", "a@x.com")
    assert manager._agent_participant_variable["AiAgentType"] == "MicrosoftAgentFramework"
    assert manager._agent_participant_variable["AiAgentApiName"] == "bot"
    await adapter.close()


@pytest.mark.asyncio
async def test_agent_participant_variable_full_override_wins() -> None:
    custom = {"AiAgentType": "MyRuntime", "AiAgentApiName": "override"}
    adapter = _adapter(agent_participant_variable=custom)
    manager = await adapter._manager("ses-1", "a@x.com")
    assert manager._agent_participant_variable == custom
    await adapter.close()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, "employee"), ({"owner_type": "individual"}, "individual")],
    ids=["defaults-from-auth-class", "override-wins-over-auth-default"],
)
def test_owner_type_resolution(kwargs: dict, expected: str) -> None:
    class _AuthWithDefault(_Auth):
        default_owner_type = "employee"

    adapter = FoundryAdapter(
        agent_id="bot", owner_id="user@x.com", auth=_AuthWithDefault(), **kwargs,
    )
    assert adapter._owner_type == expected


@pytest.mark.asyncio
async def test_owner_isolation() -> None:
    adapter = _adapter()
    await adapter._manager("ses-1", "a@x.com")
    with pytest.raises(ValueError, match="bound to owner"):
        await adapter._manager("ses-1", "b@x.com")
    await adapter.close()


@pytest.mark.asyncio
async def test_empty_format_context_is_blank() -> None:
    adapter = _adapter()
    assert adapter._format_context(None) == ""
    assert adapter._format_context({"userContext": []}) == ""
    assert adapter._format_context({"userContext": [{"entry": "prefers email"}]})
    await adapter.close()


@pytest.mark.asyncio
async def test_recall_customer_context_is_fast_only() -> None:
    adapter = _adapter()
    assert not hasattr(adapter, "get_deep_context")
    adapter.get_context = AsyncMock(return_value={"userContext": [{"entry": "profile"}]})
    fast = await adapter.recall_customer_context(session_id="ses-1")
    assert fast["userContext"][0]["entry"] == "profile"
    adapter.get_context.assert_awaited_once()
    await adapter.close()


def _pack_payload(pack_id: str = "PlanRules") -> dict:
    return {
        "contextPacks": [
            {
                "contextPackId": pack_id,
                "description": "plan rules",
                "dataNeeds": [],
            }
        ]
    }


@pytest.mark.asyncio
async def test_pack_list_and_invalid_pack() -> None:
    adapter = _adapter()
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value=_pack_payload())
    listed = await adapter.get_context_pack_list()
    assert listed["contextPacks"][0]["contextPackId"] == "PlanRules"
    adapter._allowed_pack_ids = {"PlanRules"}
    invalid = await adapter.get_context_pack("Nope", session_id="ses-1")
    assert invalid["status"] == "invalid_pack"
    await adapter.close()


@pytest.mark.asyncio
async def test_pack_catalog_refreshes_only_on_session_start() -> None:
    adapter = _adapter()
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value=_pack_payload())
    await adapter._maybe_inject_context_pack_list(is_open=True)
    assert adapter._allowed_pack_ids == {"PlanRules"}
    adapter._retrieval_client.get_context_pack_list.assert_awaited_once_with(agent_id="bot")

    adapter._retrieval_client.get_context_pack_list.reset_mock()
    await adapter._maybe_inject_context_pack_list(is_open=False)
    adapter._retrieval_client.get_context_pack_list.assert_not_awaited()
    assert adapter._allowed_pack_ids == {"PlanRules"}
    await adapter.close()


@pytest.mark.asyncio
async def test_pack_catalog_unchanged_on_api_failure() -> None:
    adapter = _adapter()
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value=_pack_payload())
    await adapter._maybe_inject_context_pack_list(is_open=True)
    assert adapter._allowed_pack_ids == {"PlanRules"}
    previous_list = adapter._allowed_context_pack_list

    adapter._retrieval_client.get_context_pack_list = AsyncMock(side_effect=RuntimeError("down"))
    await adapter._maybe_inject_context_pack_list(is_open=True)
    assert adapter._allowed_pack_ids == {"PlanRules"}
    assert adapter._allowed_context_pack_list == previous_list
    await adapter.close()


@pytest.mark.asyncio
async def test_default_tools_names() -> None:
    adapter = _adapter()
    tools = adapter.default_tools("ses-1", "user@x.com")
    assert [t.name for t in tools] == [
        "recall_customer_context",
        "get_context_pack",
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_records_user_message_and_ingests() -> None:
    adapter = _adapter()
    session = AgentSession(session_id="ses-stream")

    class _Stream:
        def __aiter__(self) -> _Stream:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def get_final_response(self) -> Any:
            return type("_Final", (), {"messages": [
                Message("assistant", [Content.from_text("hi back")]),
            ]})()

    class _Agent:
        def run(self, *_a: Any, **_k: Any) -> _Stream:
            return _Stream()

    adapter.get_context = AsyncMock(return_value=None)
    adapter.get_context_pack_list = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    adapter._manager.return_value = manager

    records = [r async for r in adapter.stream(_Agent(), "hello", session=session)]
    assert [r.event_type for r in records] == ["user_message", "agent_response"]
    assert records[0].content == "hello"
    assert records[1].content == "hi back"
    manager.ingest.assert_awaited_once()
    manager.flush.assert_awaited_once()
    # One coalesced assistant reply — not one Record per streamed token.
    assert len(manager.ingest.await_args.args[0]) == 2
    assert "ses-stream" not in adapter._captures
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_ingests_collected_records_if_model_fails() -> None:
    adapter = _adapter()
    session = AgentSession(session_id="ses-fail")

    class _Stream:
        def __aiter__(self) -> _Stream:
            return self

        async def __anext__(self) -> Any:
            raise RuntimeError("model down")

    class _Agent:
        def run(self, *_a: Any, **_k: Any) -> _Stream:
            return _Stream()

    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    adapter._manager.return_value = manager

    yielded: list[Any] = []
    with pytest.raises(RuntimeError, match="model down"):
        async for record in adapter.stream(_Agent(), "hello", session=session):
            yielded.append(record)

    assert [r.event_type for r in yielded] == ["user_message"]
    manager.ingest.assert_awaited_once()
    assert manager.ingest.await_args.args[0][0].event_type == "user_message"
    manager.flush.assert_awaited_once()
    assert "ses-fail" not in adapter._captures
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_pops_capture_even_when_ingest_itself_fails() -> None:
    adapter = _adapter()
    session = AgentSession(session_id="ses-ok")

    class _FinalResponse:
        messages: list[Any] = []

    class _Stream:
        def __aiter__(self) -> _Stream:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def get_final_response(self) -> Any:
            return _FinalResponse()

    class _Agent:
        def run(self, *_a: Any, **_k: Any) -> _Stream:
            return _Stream()

    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    manager.ingest = AsyncMock(side_effect=RuntimeError("beacon down"))
    adapter._manager.return_value = manager

    with pytest.raises(RuntimeError, match="beacon down"):
        async for _ in adapter.stream(_Agent(), "hello", session=session):
            pass

    assert "ses-ok" not in adapter._captures
    await adapter.close()


@pytest.mark.asyncio
async def test_start_session_logs_outcome(caplog: pytest.LogCaptureFixture) -> None:
    adapter = _adapter()
    manager = AsyncMock()
    manager._owner_id = "user@x.com"
    manager.create_session = AsyncMock(return_value={"session_id": "ses-1"})
    adapter._manager = AsyncMock(return_value=manager)  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO):
        await adapter.start_session("ses-1")
    assert "start_session" in caplog.text
    assert "outcome=success" in caplog.text
    assert "user@x.com" in caplog.text

    manager.create_session = AsyncMock(side_effect=RuntimeError("denied"))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="denied"):
        await adapter.start_session("ses-1")
    assert "outcome=denied" in caplog.text
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_attached_provider_returns_assistant_after_after_run() -> None:
    """MAF runs after_run on stream exhaustion, before stream() maps the final reply."""
    adapter = _adapter()
    session = AgentSession(session_id="ses-attached")
    provider = adapter.context_provider(owner_id="user@x.com")
    incoming = [Message("user", [Content.from_text("hello")])]
    context = SessionContext(session_id="ses-attached", input_messages=incoming)
    state = session.state.setdefault(provider.source_id, {})

    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    adapter._manager.return_value = manager

    await provider.before_run(agent=None, session=session, context=context, state=state)

    class _Stream:
        def __aiter__(self) -> _Stream:
            return self

        async def __anext__(self) -> Any:
            context._response = type("_Resp", (), {"messages": [
                Message("assistant", [Content.from_text("hi back")]),
            ]})()
            await provider.after_run(
                agent=None, session=session, context=context, state=state,
            )
            raise StopAsyncIteration

        async def get_final_response(self) -> Any:
            return context.response

    class _Agent:
        context_providers = [provider]

        def run(self, *_a: Any, **_k: Any) -> _Stream:
            return _Stream()

    records = [r async for r in adapter.stream(_Agent(), incoming, session=session)]
    assert [r.event_type for r in records] == ["user_message", "agent_response"]
    assert records[1].content == "hi back"
    manager.ingest.assert_awaited_once()
    assert "ses-attached" not in adapter._captures
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_ingests_complete_tool_args_not_partial_delta() -> None:
    adapter = _adapter()
    session = AgentSession(session_id="ses-tools")

    class _Stream:
        def __init__(self) -> None:
            self._step = 0

        def __aiter__(self) -> _Stream:
            return self

        async def __anext__(self) -> Any:
            self._step += 1
            if self._step == 1:
                return Message("assistant", [
                    Content.from_function_call(
                        call_id="c1", name="recall_customer_context", arguments="{",
                    ),
                ])
            if self._step == 2:
                return Message("assistant", [
                    Content.from_function_call(
                        call_id="c1",
                        name="recall_customer_context",
                        arguments='{"q": "refund"}',
                    ),
                ])
            raise StopAsyncIteration

        async def get_final_response(self) -> Any:
            return type("_Final", (), {"messages": [
                Message("assistant", [
                    Content.from_function_call(
                        call_id="c1",
                        name="recall_customer_context",
                        arguments='{"q": "refund"}',
                    ),
                ]),
            ]})()

    class _Agent:
        def run(self, *_a: Any, **_k: Any) -> _Stream:
            return _Stream()

    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    adapter._manager.return_value = manager

    records = [r async for r in adapter.stream(_Agent(), "hello", session=session)]
    tool_calls = [r for r in records if r.event_type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].variable["InputValueText"] == '{"q": "refund"}'
    ingested = manager.ingest.await_args.args[0]
    ingested_calls = [r for r in ingested if r.event_type == "tool_call"]
    assert ingested_calls[0].variable["InputValueText"] == '{"q": "refund"}'
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_attached_provider_honors_owner_id() -> None:
    adapter = FoundryAdapter(agent_id="bot", owner_id="alice", auth=_Auth())
    session = AgentSession(session_id="ses-owner-run")
    provider = adapter.context_provider()
    incoming = [Message("user", [Content.from_text("hello")])]
    context = SessionContext(session_id="ses-owner-run", input_messages=incoming)

    adapter.get_context = AsyncMock(return_value=None)
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)
    adapter._retrieval_client.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
    adapter._manager = AsyncMock()  # type: ignore[method-assign]
    manager = AsyncMock()
    adapter._manager.return_value = manager

    class _Stream:
        def __aiter__(self) -> _Stream:
            return self

        async def __anext__(self) -> Any:
            state = session.state.setdefault(provider.source_id, {})
            await provider.before_run(
                agent=None, session=session, context=context, state=state,
            )
            assert adapter._captures["ses-owner-run"]["owner"] == "bob"
            context._response = type("_Resp", (), {"messages": [
                Message("assistant", [Content.from_text("hi back")]),
            ]})()
            await provider.after_run(
                agent=None, session=session, context=context, state=state,
            )
            raise StopAsyncIteration

        async def get_final_response(self) -> Any:
            return context.response

    class _Agent:
        context_providers = [provider]

        def run(self, *_a: Any, **_k: Any) -> _Stream:
            return _Stream()

    records = [
        r async for r in adapter.stream(
            _Agent(), incoming, session=session, owner_id="bob",
        )
    ]
    assert [r.event_type for r in records] == ["user_message", "agent_response"]
    assert records[0].participant_id == "bob"
    adapter.get_context.assert_awaited()
    assert adapter.get_context.await_args.kwargs["owner_id"] == "bob"
    await adapter.close()


# ─── Server-side context_pack_instruction (pinned per session + fallback) ──


@pytest.mark.asyncio
async def test_get_context_pack_instruction_pins_server_text_per_session() -> None:
    adapter = _adapter()
    rc = adapter._retrieval_client
    rc.get_system_prompt = AsyncMock(side_effect=["SERVER V1", "SERVER V2"])

    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("ses-1") == "SERVER V1"
    assert await adapter.get_context_pack_instruction("ses-2") == "SERVER V2"
    assert rc.get_system_prompt.await_count == 2
    rc.get_system_prompt.assert_awaited_with("context_pack_instruction")
    await adapter.close()


@pytest.mark.asyncio
async def test_get_context_pack_instruction_404_falls_back_to_bundled() -> None:
    from agentic_context import CONTEXT_PACK_INSTRUCTION

    adapter = _adapter()
    adapter._retrieval_client.get_system_prompt = AsyncMock(return_value=None)

    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION
    await adapter.close()


@pytest.mark.asyncio
async def test_get_context_pack_instruction_error_pins_fallback_for_session() -> None:
    from agentic_context import CONTEXT_PACK_INSTRUCTION

    adapter = _adapter()
    adapter._retrieval_client.get_system_prompt = AsyncMock(
        side_effect=[RuntimeError("boom"), "SERVER PROMPT"],
    )

    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION
    assert await adapter.get_context_pack_instruction("ses-1") == CONTEXT_PACK_INSTRUCTION
    # Within the failure cooldown a new session also skips the fetch.
    assert await adapter.get_context_pack_instruction("ses-2") == CONTEXT_PACK_INSTRUCTION
    assert adapter._retrieval_client.get_system_prompt.await_count == 1
    await adapter.close()
