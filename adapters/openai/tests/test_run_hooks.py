# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for OpenAIAdapter.run_hooks() — the streaming path.

Verifies that Agents SDK lifecycle events translate into Records in the right
order via the existing RecordTransformer, that one run is one turn (one
interaction_id, one /entries POST), that dedup works across repeated
on_llm_end calls, that composition with a customer's RunHooks preserves their
callbacks, and that sync mode rejects the call up front.
"""
from __future__ import annotations

import importlib.util
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_AGENTS_SDK_AVAILABLE = importlib.util.find_spec("agents") is not None

pytestmark = pytest.mark.skipif(
    not _AGENTS_SDK_AVAILABLE,
    reason="openai-agents not installed; run_hooks is SDK-gated.",
)

from agentic_context_openai import OpenAIAdapter

CREDS = {
    "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----",
    "consumer_key": "3MVG.test",
    "subject": "user@org.com",
    "login_url": "login.salesforce.com",
}


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mock_async_clients():
    """Patch the async clients so no network/JWT runs."""
    with patch("agentic_context_openai.adapter.AsyncIngestionClient") as MockIng, \
         patch("agentic_context_openai.adapter.AsyncRetrievalClient") as MockRet, \
         patch("agentic_context_openai.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = ("test-token", "https://beacon.test")
        MockAuth.return_value = auth_inst
        MockIng.return_value = MagicMock(close=AsyncMock())
        MockRet.return_value = MagicMock(close=AsyncMock())
        yield


@pytest.fixture
def adapter(mock_async_clients):
    """Adapter with a mocked ingest() so hooks tests inspect records easily."""
    a = OpenAIAdapter(agent_id="agent-support", owner_id="user-001", **CREDS)
    a.ingest = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return a


def _make_response(output: list[dict]) -> MagicMock:
    """Fake ModelResponse — only .output is needed by on_llm_end."""
    r = MagicMock()
    r.output = output
    return r


def _tool_context(call_id: str) -> MagicMock:
    """Fake ToolContext — on_tool_end reads tool_call_id off it."""
    ctx = MagicMock()
    ctx.tool_call_id = call_id
    return ctx


async def _flushed(adapter, hooks) -> list:
    """Flush the turn and return the one batch it posted."""
    await hooks.flush()
    adapter.ingest.assert_awaited_once()
    return adapter.ingest.call_args.args[1]


def _circular_list() -> list:
    """Circular reference → json raises ValueError even with default=str."""
    circular: list = []
    circular.append(circular)
    return circular


# ─── on_llm_end ─────────────────────────────────────────────────

class TestOnLlmEnd:
    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            (
                {"type": "message", "role": "assistant", "id": "msg_1",
                 "content": [{"type": "output_text", "text": "Hello!"}]},
                {"event_type": "agent_response", "content": "Hello!"},
            ),
            (
                {"type": "function_call", "name": "get_weather",
                 "call_id": "fc_1", "arguments": json.dumps({"city": "SF"})},
                {"event_type": "tool_call", "event_id": "fc_1",
                 "variable": {"Name": "get_weather", "InputValueText": json.dumps({"city": "SF"})}},
            ),
            (
                {"type": "reasoning",
                 "summary": [{"type": "summary_text", "text": "Thinking..."}]},
                {"event_type": "agent_reasoning", "variable": {"OutputValueText": "Thinking..."}},
            ),
        ],
        ids=["message-emits-agent-response", "function-call-emits-tool-call",
             "reasoning-summary-emits-agent-reasoning"],
    )
    @pytest.mark.asyncio
    async def test_output_item_emits_record(self, adapter, item, expected) -> None:
        hooks = adapter.run_hooks("ses-1")

        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([item]))
        adapter.ingest.assert_not_awaited()  # buffered until the turn ends

        await hooks.flush()
        adapter.ingest.assert_awaited_once()
        # positional args: (session_id, records); owner_id kw.
        args, kwargs = adapter.ingest.call_args
        assert args[0] == "ses-1"
        assert kwargs["owner_id"] == "user-001"
        records = args[1]
        assert len(records) == 1
        for field, value in expected.items():
            assert getattr(records[0], field) == value, field

    @pytest.mark.asyncio
    async def test_mixed_output_emits_in_order(self, adapter) -> None:
        """Multi-item ModelResponse — reasoning, message, function_call — all
        emit in chronological (list) order."""
        hooks = adapter.run_hooks("ses-1")
        response = _make_response([
            {"type": "reasoning",
             "summary": [{"type": "summary_text", "text": "Need weather tool"}]},
            {"type": "message", "role": "assistant", "id": "msg_1",
             "content": [{"type": "output_text", "text": "Looking up..."}]},
            {"type": "function_call", "name": "get_weather",
             "call_id": "fc_1", "arguments": "{}"},
        ])

        await hooks.on_llm_end(MagicMock(), MagicMock(), response)

        records = await _flushed(adapter, hooks)
        assert [r.event_type for r in records] == [
            "agent_reasoning", "agent_response", "tool_call",
        ]

    @pytest.mark.asyncio
    async def test_dedup_across_calls_by_item_id(self, adapter) -> None:
        """Repeated on_llm_end with overlapping item ids ingests each once."""
        hooks = adapter.run_hooks("ses-1")
        shared = {"type": "function_call", "name": "f", "call_id": "fc_1",
                  "arguments": "{}"}
        new = {"type": "function_call", "name": "g", "call_id": "fc_2",
               "arguments": "{}"}

        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([shared]))
        await hooks.on_llm_end(MagicMock(), MagicMock(),
                                _make_response([shared, new]))

        # fc_1 from the first call; the second call's fc_1 is dedup'd.
        records = await _flushed(adapter, hooks)
        assert [r.event_id for r in records] == ["fc_1", "fc_2"]

    @pytest.mark.asyncio
    async def test_empty_output_skips_ingest(self, adapter) -> None:
        hooks = adapter.run_hooks("ses-1")
        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([]))
        await hooks.flush()
        adapter.ingest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pydantic_item_is_serialized_via_model_dump(self, adapter) -> None:
        """TResponseOutputItem is pydantic in the SDK — _normalize_item must
        call .model_dump() to feed the transformer dicts."""
        hooks = adapter.run_hooks("ses-1")
        fake_item = MagicMock()
        fake_item.model_dump.return_value = {
            "type": "message", "role": "assistant", "id": "m1",
            "content": [{"type": "output_text", "text": "hi"}],
        }
        # strip dict interface so _normalize_item hits the model_dump branch
        fake_item.__class__ = type("FakePydantic", (), {})
        del fake_item.__dict__  # type: ignore[attr-defined]

        await hooks.on_llm_end(MagicMock(), MagicMock(),
                                _make_response([fake_item]))

        fake_item.model_dump.assert_called_once()
        records = await _flushed(adapter, hooks)
        assert records[0].content == "hi"


# ─── on_tool_end ────────────────────────────────────────────────

class TestOnToolEnd:
    @pytest.mark.asyncio
    async def test_string_result_emits_tool_result(self, adapter) -> None:
        hooks = adapter.run_hooks("ses-1")
        # Prime _tool_names via a prior function_call so the Trace carries Name.
        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([
            {"type": "function_call", "name": "get_weather",
             "call_id": "fc_1", "arguments": "{}"},
        ]))

        await hooks.on_tool_end(_tool_context("fc_1"), MagicMock(),
                                 MagicMock(), "72F sunny")

        records = await _flushed(adapter, hooks)
        assert [r.event_type for r in records] == ["tool_call", "tool_result"]
        assert records[1].event_id == "fc_1-out"
        assert records[1].variable == {
            "Name": "get_weather",
            "OutputValueText": "72F sunny",
        }

    @pytest.mark.parametrize(
        ("make_result", "expected"),
        [
            # Non-string tool results JSON-encode — Record.content contract rule.
            (lambda: {"status": "ok", "code": 200}, json.dumps({"status": "ok", "code": 200})),
            # Objects json.dumps can't encode (even with default=str) fall
            # through to str(result).
            (_circular_list, "[[...]]"),
        ],
        ids=["structured-jsonified", "unserializable-falls-back-to-str"],
    )
    @pytest.mark.asyncio
    async def test_non_string_result_becomes_string(self, adapter, make_result, expected) -> None:
        """The Record.OutputValueText contract is always a str."""
        result = make_result()
        hooks = adapter.run_hooks("ses-1")
        await hooks.on_tool_end(_tool_context("fc_x"), MagicMock(),
                                 MagicMock(), result)

        records = await _flushed(adapter, hooks)
        out = records[0].variable["OutputValueText"]
        assert isinstance(out, str)
        assert out == expected


# ─── One run = one turn ─────────────────────────────────────────

class TestOneTurnPerRun:
    """One /entries POST per turn, one interaction_id per turn."""

    @pytest.mark.asyncio
    async def test_tool_turn_is_one_post_with_one_interaction_id(self, adapter) -> None:
        hooks = adapter.run_hooks("ses-1", user_input="weather in SF?")

        await hooks.on_agent_start(MagicMock(), MagicMock())
        await hooks.on_llm_start(MagicMock(), MagicMock(), "sys", [])
        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([
            {"type": "function_call", "name": "get_weather",
             "call_id": "fc_1", "arguments": "{}"},
        ]))
        await hooks.on_tool_end(_tool_context("fc_1"), MagicMock(),
                                 MagicMock(), "72F")
        await hooks.on_llm_start(MagicMock(), MagicMock(), "sys", [])
        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([
            {"type": "message", "role": "assistant", "id": "msg_1",
             "content": [{"type": "output_text", "text": "72F and sunny"}]},
        ]))
        adapter.ingest.assert_not_awaited()

        await hooks.on_agent_end(MagicMock(), MagicMock(), "72F and sunny")

        adapter.ingest.assert_awaited_once()
        records = adapter.ingest.call_args.args[1]
        assert [r.event_type for r in records] == [
            "user_message", "tool_call", "tool_result", "agent_response",
        ]
        assert records[0].content == "weather in SF?"
        assert records[0].participant_id == "user-001"
        assert {r.interaction_id for r in records} == {hooks.interaction_id}
        assert hooks.interaction_id.startswith("turn-ses-1-")

    @pytest.mark.asyncio
    async def test_user_input_accepts_items(self, adapter) -> None:
        hooks = adapter.run_hooks(
            "ses-1", user_input=[{"role": "user", "content": "hi"}],
        )
        (rec,) = await _flushed(adapter, hooks)
        assert rec.event_type == "user_message"
        assert rec.content == "hi"

    @pytest.mark.asyncio
    async def test_flush_after_agent_end_is_a_noop(self, adapter) -> None:
        hooks = adapter.run_hooks("ses-1", user_input="hi")
        await hooks.on_agent_end(MagicMock(), MagicMock(), "done")
        await hooks.flush()
        adapter.ingest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_each_run_gets_its_own_interaction_id(self, adapter) -> None:
        h1 = adapter.run_hooks("ses-1", user_input="a")
        h2 = adapter.run_hooks("ses-1", user_input="b")
        assert h1.interaction_id != h2.interaction_id

    @pytest.mark.asyncio
    async def test_close_flushes_a_run_that_never_finished(self, adapter) -> None:
        """No final output (exception, max turns, interruption) → close() posts it."""
        hooks = adapter.run_hooks("ses-1", user_input="hi")
        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([
            {"type": "function_call", "name": "f", "call_id": "fc_1", "arguments": "{}"},
        ]))

        await adapter.close()

        adapter.ingest.assert_awaited_once()
        records = adapter.ingest.call_args.args[1]
        assert [r.event_type for r in records] == ["user_message", "tool_call"]

    @pytest.mark.asyncio
    async def test_close_warns_about_an_unflushed_run(self, adapter, caplog) -> None:
        hooks = adapter.run_hooks("ses-1", user_input="hi")
        with caplog.at_level("WARNING"):
            await adapter.close()
        assert hooks.interaction_id in caplog.text
        assert "never flushed" in caplog.text

    @pytest.mark.asyncio
    async def test_id_less_items_get_distinct_event_ids_within_a_turn(self, adapter) -> None:
        """Chat Completions items carry __fake_id__; two model calls in one
        turn share the interaction_id, so the fallback idx must keep running."""
        hooks = adapter.run_hooks("ses-1", user_input="hi")
        for text in ("checking", "done"):
            await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([
                {"type": "message", "role": "assistant", "id": "__fake_id__",
                 "content": [{"type": "output_text", "text": text}]},
            ]))
        records = await _flushed(adapter, hooks)
        assert [r.event_type for r in records] == [
            "user_message", "agent_response", "agent_response",
        ]
        assert len({r.event_id for r in records}) == 3

    @pytest.mark.asyncio
    async def test_first_run_seeds_fast_context_once(self, adapter) -> None:
        adapter.get_context = AsyncMock(return_value={"userContext": []})  # type: ignore[method-assign]
        hooks = adapter.run_hooks("ses-1", user_input="hi")
        await hooks.on_agent_start(MagicMock(), MagicMock())
        await hooks.on_agent_start(MagicMock(), MagicMock())  # e.g. a handoff
        await adapter.run_hooks("ses-1").on_agent_start(MagicMock(), MagicMock())
        adapter.get_context.assert_awaited_once()


# ─── Composition with customer hooks ────────────────────────────

class TestComposition:
    @pytest.mark.asyncio
    async def test_next_hooks_called_for_every_lifecycle_method(self, adapter) -> None:
        user = MagicMock()
        user.on_agent_start = AsyncMock()
        user.on_agent_end = AsyncMock()
        user.on_handoff = AsyncMock()
        user.on_llm_start = AsyncMock()
        user.on_llm_end = AsyncMock()
        user.on_tool_start = AsyncMock()
        user.on_tool_end = AsyncMock()

        hooks = adapter.run_hooks("ses-1", next_hooks=user)

        await hooks.on_agent_start(MagicMock(), MagicMock())
        await hooks.on_agent_end(MagicMock(), MagicMock(), "done")
        await hooks.on_handoff(MagicMock(), MagicMock(), MagicMock())
        await hooks.on_llm_start(MagicMock(), MagicMock(), "sys", [])
        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response([]))
        await hooks.on_tool_start(MagicMock(), MagicMock(), MagicMock())
        await hooks.on_tool_end(_tool_context("c"), MagicMock(),
                                 MagicMock(), "r")

        user.on_agent_start.assert_awaited_once()
        user.on_agent_end.assert_awaited_once()
        user.on_handoff.assert_awaited_once()
        user.on_llm_start.assert_awaited_once()
        user.on_llm_end.assert_awaited_once()
        user.on_tool_start.assert_awaited_once()
        user.on_tool_end.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingest_failure_still_calls_next_hooks(self, adapter) -> None:
        """Ingest exceptions are swallowed so the customer's hook always fires."""
        adapter.ingest.side_effect = RuntimeError("beacon down")
        user = MagicMock()
        user.on_tool_end = AsyncMock()
        user.on_agent_end = AsyncMock()

        hooks = adapter.run_hooks("ses-1", next_hooks=user)
        await hooks.on_tool_end(_tool_context("c"), MagicMock(),
                                 MagicMock(), "r")
        # The turn's POST happens here and fails — must not raise:
        await hooks.on_agent_end(MagicMock(), MagicMock(), "done")

        adapter.ingest.assert_awaited_once()
        user.on_tool_end.assert_awaited_once()
        user.on_agent_end.assert_awaited_once()


# ─── Factory + sync-mode guard ──────────────────────────────────

class TestRunHooksFactory:
    def test_sync_mode_raises(self, mock_async_clients) -> None:
        with patch("agentic_context_openai.adapter.IngestionClient"), \
             patch("agentic_context_openai.adapter.RetrievalClient"):
            a = OpenAIAdapter(
                agent_id="a", owner_id="o", use_async=False, **CREDS,
            )
        with pytest.raises(RuntimeError, match="async-only"):
            a.run_hooks("ses-1")

    @pytest.mark.parametrize(
        ("kwargs", "owner"),
        [({}, "user-001"), ({"owner_id": "alice@corp.com"}, "alice@corp.com")],
        ids=["defaults-to-adapter-owner", "override-wins"],
    )
    def test_owner_id_resolution(self, adapter, kwargs, owner) -> None:
        hooks = adapter.run_hooks("ses-1", **kwargs)
        assert hooks._owner_id == owner

    def test_fresh_dedup_state_per_instance(self, adapter) -> None:
        """Two calls return independent hooks (so dedup state is per-run, not
        shared across runs)."""
        h1 = adapter.run_hooks("ses-1")
        h2 = adapter.run_hooks("ses-1")
        assert h1 is not h2
        assert h1._seen_item_ids is not h2._seen_item_ids


# ─── Agents SDK placeholder-id handling ─────────────────────────

class TestFakeIdDedup:
    """id="__fake_id__" is the Chat Completions converter's sentinel for
    'no stable id'. The dedup set must not collapse later items under it."""

    @pytest.mark.parametrize(
        ("items", "field", "values"),
        [
            (
                [{"type": "message", "role": "assistant", "id": "__fake_id__",
                  "content": [{"type": "output_text", "text": "first"}]},
                 {"type": "message", "role": "assistant", "id": "__fake_id__",
                  "content": [{"type": "output_text", "text": "second"}]}],
                lambda r: r.content, ["first", "second"],
            ),
            (
                [{"type": "function_call", "name": "a", "id": "__fake_id__", "arguments": "{}"},
                 {"type": "function_call", "name": "b", "id": "__fake_id__", "arguments": "{}"}],
                lambda r: r.variable["Name"], ["a", "b"],
            ),
        ],
        ids=["messages", "function-calls"],
    )
    @pytest.mark.asyncio
    async def test_fake_id_items_are_not_deduped(self, adapter, items, field, values) -> None:
        hooks = adapter.run_hooks("ses-1")

        await hooks.on_llm_end(MagicMock(), MagicMock(), _make_response(items))

        records = await _flushed(adapter, hooks)
        assert [field(r) for r in records] == values
        assert "__fake_id__" not in hooks._seen_item_ids
        # Distinct event_ids via the transformer's slot fallback — the
        # critical property is uniqueness, not the exact format.
        assert len({r.event_id for r in records}) == 2


# ─── Empty tool_call_id (computer / local-shell callbacks) ──────

class TestEmptyToolCallId:
    """RunContextWrapper for computer / local-shell tool callbacks has no
    tool_call_id. Repeated invocations must still produce distinct event_ids."""

    @pytest.mark.parametrize(
        "make_ctx",
        [
            # Simulate a RunContextWrapper that lacks tool_call_id entirely.
            lambda: MagicMock(spec=[]),
            # tool_call_id='__fake_id__' is the SDK placeholder and must not
            # propagate into the Record's event_id.
            lambda: _tool_context("__fake_id__"),
        ],
        ids=["missing-tool-call-id", "fake-id-tool-call-id"],
    )
    @pytest.mark.asyncio
    async def test_repeated_empty_call_id_invocations_have_distinct_event_ids(
        self, adapter, make_ctx,
    ) -> None:
        hooks = adapter.run_hooks("ses-1")
        ctx = make_ctx()

        await hooks.on_tool_end(ctx, MagicMock(), MagicMock(), "shell out 1")
        await hooks.on_tool_end(ctx, MagicMock(), MagicMock(), "shell out 2")

        first, second = await _flushed(adapter, hooks)
        assert first.event_id != second.event_id
        for rec in (first, second):
            assert rec.event_id.startswith("tool-")
            assert "__fake_id__" not in rec.event_id
        # Each result also carries the distinct output.
        assert first.variable["OutputValueText"] == "shell out 1"
        assert second.variable["OutputValueText"] == "shell out 2"

    @pytest.mark.asyncio
    async def test_real_tool_call_id_still_used_verbatim(self, adapter) -> None:
        hooks = adapter.run_hooks("ses-1")
        await hooks.on_tool_end(_tool_context("fc_real"), MagicMock(),
                                 MagicMock(), "r")
        (rec,) = await _flushed(adapter, hooks)
        assert rec.event_id == "fc_real-out"


# ─── Version gate for openai-agents ─────────────────────────────

class TestVersionGate:
    """BeaconRunHooks requires on_llm_start / on_llm_end on RunHooks (added
    after openai-agents 0.2). The factory must raise a clear ImportError on
    older SDKs rather than silently drop every assistant turn."""

    def test_missing_on_llm_end_raises(self, adapter) -> None:
        import types

        class StubRunHooks:
            async def on_agent_start(self, *a, **kw): ...
            async def on_agent_end(self, *a, **kw): ...
            async def on_handoff(self, *a, **kw): ...
            async def on_tool_start(self, *a, **kw): ...
            async def on_tool_end(self, *a, **kw): ...
            # Deliberately missing: on_llm_start, on_llm_end.

        fake_lifecycle = types.ModuleType("agents.lifecycle")
        fake_lifecycle.RunHooks = StubRunHooks  # type: ignore[attr-defined]
        # _build_run_hooks_class() imports via `from agents.lifecycle import
        # RunHooks` every call (no cache), so swapping sys.modules is enough.
        with patch.dict(
            "sys.modules",
            {"agents.lifecycle": fake_lifecycle},
        ):
            with pytest.raises(ImportError, match=r"openai-agents >= 0\.3"):
                adapter.run_hooks("ses-1")
