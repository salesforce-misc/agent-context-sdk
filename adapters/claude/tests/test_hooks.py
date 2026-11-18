# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for ClaudeAdapter.hooks() method.

Verifies that adapter.hooks(session_id, owner_id) returns a proper
Claude Agent SDK hooks dict with UserPromptSubmit and Stop callbacks.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock claude_agent_sdk if not installed in this venv
if "claude_agent_sdk" not in sys.modules:
    @dataclass
    class _MockHookMatcher:
        matcher: str | None = None
        hooks: list = field(default_factory=list)
        timeout: float | None = None

    @dataclass
    class _MockClaudeAgentOptions:
        model: str = ""
        hooks: dict | None = None
        system_prompt: str = ""

    mock_sdk = MagicMock()
    mock_sdk.HookMatcher = _MockHookMatcher
    mock_sdk.ClaudeAgentOptions = _MockClaudeAgentOptions
    sys.modules["claude_agent_sdk"] = mock_sdk

from agentic_context import CONTEXT_PACK_INSTRUCTION
from agentic_context_claude import ClaudeAdapter

FIXTURES = Path(__file__).parent / "fixtures"

# Inline credentials used across adapter constructions. The static
# base_url/api_key path is gone — the adapter always builds a ContextAuth, so
# the fixtures patch it and fake the JWT → CDP exchange (no network).
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
    """Mock async clients + faked ContextAuth exchange."""
    with patch("agentic_context_claude.adapter.AsyncIngestionClient") as MockIngestion, \
         patch("agentic_context_claude.adapter.AsyncRetrievalClient") as MockRetrieval, \
         patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        MockAuth.return_value = auth_inst
        ingestion = MagicMock()
        ingestion.create_conversation = AsyncMock(return_value={})
        ingestion.create_session = AsyncMock(return_value={})
        ingestion.add_participants = AsyncMock(return_value={})
        ingestion.add_entries = AsyncMock(return_value={})
        ingestion.close_session = AsyncMock(return_value={})
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
        retrieval.get_conversation = AsyncMock(return_value={"entries": []})
        retrieval.list_conversations = AsyncMock(return_value=[])
        retrieval.get_context_pack_list = AsyncMock(return_value={"contextPacks": []})
        retrieval.close = AsyncMock(return_value=None)
        MockRetrieval.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval}


@pytest.fixture
def mock_sync_clients():
    """Mock sync clients + faked ContextAuth exchange."""
    with patch("agentic_context_claude.adapter.IngestionClient") as MockIngestion, \
         patch("agentic_context_claude.adapter.RetrievalClient") as MockRetrieval, \
         patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        MockAuth.return_value = auth_inst
        ingestion = MagicMock()
        ingestion.create_conversation.return_value = {}
        ingestion.create_session.return_value = {}
        ingestion.add_participants.return_value = {}
        ingestion.add_entries.return_value = {}
        ingestion.close_session.return_value = {}
        ingestion.close.return_value = None
        MockIngestion.return_value = ingestion

        retrieval = MagicMock()
        retrieval.get_context.return_value = {
            "contextId": "ctx-sync",
            "userContext": [
                {"entry": "User is vegetarian", "sourceType": "Profile"},
            ],
        }
        retrieval.get_context_pack_list.return_value = {"contextPacks": []}
        retrieval.close.return_value = None
        MockRetrieval.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval}

@pytest.fixture(params=["async", "sync"])
def clients(request):
    """Run a test against both the async and sync hook paths.

    The sync and async callbacks are separate code paths in the adapter, so each
    parametrized test executes once per mode. ``fact`` is the profile entry the
    mode's fixture returns from ``get_context``.
    """
    is_async = request.param == "async"
    mocks = request.getfixturevalue("mock_async_clients" if is_async else "mock_sync_clients")
    fact = "User prefers dark mode" if is_async else "User is vegetarian"
    return {**mocks, "use_async": is_async, "fact": fact}


async def _run(result):
    """Await a hook result if the callback was async; pass sync results through."""
    return await result if inspect.isawaitable(result) else result


def _stub(clients, name, **kwargs):
    """Replace ``retrieval.<name>`` with a mock matching the client's mode."""
    mock_cls = AsyncMock if clients["use_async"] else MagicMock
    setattr(clients["retrieval"], name, mock_cls(**kwargs))


def _fetches(mock) -> int:
    """Awaits for an async client mock, calls for a sync one."""
    return mock.await_count if isinstance(mock, AsyncMock) else mock.call_count


def _end_turn(adapter, conversation_id="sdk-ses"):
    """Stand-in for the turn reaching Stop: blocks attached on it count as
    delivered, so the next turn doesn't re-attach them."""
    adapter._confirm_attached_blocks({"session_id": conversation_id})


def _adapter(clients, **overrides):
    kwargs = dict(
        agent_id="agent-test", owner_id="user-001", **CREDS, use_async=clients["use_async"],
    )
    kwargs.update(overrides)
    return ClaudeAdapter(**kwargs)


PACK_LIST_ONLY = {
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "[Available context packs]\n{}",
    }
}


# ─── Structure Tests ──────────────────────────────────────────────────────────


class TestHooksStructure:
    """Verify hooks() returns the correct dict shape."""

    def test_returns_three_hook_matcher_lists(self, mock_async_clients):
        HookMatcher = sys.modules["claude_agent_sdk"].HookMatcher

        adapter = ClaudeAdapter(
            agent_id="agent-test",
            owner_id="user-001",
            **CREDS,
            use_async=True,
        )

        hooks = adapter.hooks("conv-hooks", "user-001", inject_context_pack_instruction=False)

        assert set(hooks) == {"UserPromptSubmit", "Stop", "PreCompact"}
        for key in ("UserPromptSubmit", "Stop", "PreCompact"):
            assert isinstance(hooks[key], list)
            assert len(hooks[key]) == 1
            assert isinstance(hooks[key][0], HookMatcher)
            assert hooks[key][0].matcher == ".*"
            assert len(hooks[key][0].hooks) == 1
            assert callable(hooks[key][0].hooks[0])

    def test_hook_callbacks_match_adapter_mode(self, clients):
        adapter = _adapter(clients)

        hooks = adapter.hooks("conv-hooks", "user-001", inject_context_pack_instruction=False)

        for key in ("UserPromptSubmit", "Stop"):
            callback = hooks[key][0].hooks[0]
            assert inspect.iscoroutinefunction(callback) is clients["use_async"], (
                f"{key} hook should be {'async' if clients['use_async'] else 'sync'}"
            )


# ─── UserPromptSubmit Hook ────────────────────────────────────────────────────


class TestUserPromptSubmit:
    """Test the UserPromptSubmit hook injects context."""

    @pytest.mark.asyncio
    async def test_injects_context_as_additional_context(self, clients):
        _stub(clients, "get_context", return_value={
            "contextId": "ctx-123",
            "userContext": [
                {"entry": "User prefers dark mode", "sourceType": "Profile"},
                {"entry": "Previously asked about order ORD-001", "sourceType": "Conversation"},
            ],
        })
        adapter = _adapter(clients, context_mode="auto")  # seeds fast context at conversation-open

        hooks = adapter.hooks("conv-ctx", "user-001", inject_context_pack_instruction=False)
        callback = hooks["UserPromptSubmit"][0].hooks[0]

        # Simulate Claude Agent SDK calling the hook
        result = await _run(callback(
            {"session_id": "sdk-ses", "transcript_path": "/tmp/t.jsonl", "cwd": "/"},
            None,  # tool_use_id
            {"signal": None},  # context
        ))

        # Should return hookSpecificOutput with additionalContext
        assert "hookSpecificOutput" in result
        output = result["hookSpecificOutput"]
        assert output["hookEventName"] == "UserPromptSubmit"
        assert "additionalContext" in output

        ctx_str = output["additionalContext"]
        assert "[Context from prior conversations]" in ctx_str
        assert "User prefers dark mode" in ctx_str
        assert "Previously asked about order ORD-001" in ctx_str
        assert "[Profile]" in ctx_str
        assert "[Conversation]" in ctx_str

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_context",
        [
            AsyncMock(return_value=None),
            AsyncMock(return_value={"contextId": "c1", "userContext": []}),
            AsyncMock(side_effect=RuntimeError("Beacon 500")),
        ],
        ids=["no-context", "empty-userContext", "beacon-error"],
    )
    async def test_no_usable_context_injects_only_pack_catalog(
        self, mock_async_clients, get_context,
    ):
        """A None payload, an empty ``userContext`` list, or a Beacon error all
        degrade to the bare pack-catalog block — the hook never raises."""
        mock_async_clients["retrieval"].get_context = get_context

        adapter = ClaudeAdapter(
            agent_id="agent-test",
            owner_id="user-001",
            **CREDS,
            use_async=True,
        )

        hooks = adapter.hooks("conv-no-ctx", "user-001", inject_context_pack_instruction=False)
        callback = hooks["UserPromptSubmit"][0].hooks[0]

        result = await callback(
            {"session_id": "sdk-ses", "transcript_path": "/tmp/t.jsonl", "cwd": "/"},
            None,
            {"signal": None},
        )

        assert result == PACK_LIST_ONLY

    @pytest.mark.asyncio
    async def test_custom_context_formatter(self, mock_async_clients):
        adapter = ClaudeAdapter(
            agent_id="agent-test",
            owner_id="user-001",
            **CREDS,
            use_async=True,
        )

        adapter.set_context_mode("auto")  # seeds fast context at open

        def custom_formatter(payload):
            entries = payload.get("userContext", [])
            return f"CUSTOM: {len(entries)} facts loaded"

        hooks = adapter.hooks(
            "conv-fmt", "user-001", context_formatter=custom_formatter,
            inject_context_pack_instruction=False,
        )
        callback = hooks["UserPromptSubmit"][0].hooks[0]

        result = await callback(
            {"session_id": "sdk-ses", "transcript_path": "/tmp/t.jsonl", "cwd": "/"},
            None,
            {"signal": None},
        )

        assert result["hookSpecificOutput"]["additionalContext"] == (
            "CUSTOM: 2 facts loaded\n\n[Available context packs]\n{}"
        )


class TestContextPackListInjection:
    @pytest.mark.asyncio
    async def test_injects_catalog_at_conversation_open(self, clients):
        _stub(clients, "get_context_pack_list", return_value={
            "contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]
        })
        adapter = _adapter(clients, context_mode="manual")
        hooks = adapter.hooks("conv-pack", "user-001", inject_context_pack_instruction=False)
        callback = hooks["UserPromptSubmit"][0].hooks[0]

        result = await _run(_prompt(callback))

        pack_list = clients["retrieval"].get_context_pack_list
        if clients["use_async"]:
            pack_list.assert_awaited_once_with(agent_id="agent-test")
        else:
            pack_list.assert_called_once_with(agent_id="agent-test")
        ctx_str = result["hookSpecificOutput"]["additionalContext"]
        assert "[Available context packs]" in ctx_str
        assert "PlanRules" in ctx_str
        assert adapter._allowed_pack_ids == {"PlanRules"}

    @pytest.mark.asyncio
    async def test_catalog_fetched_once_not_on_later_turns(self, clients):
        _stub(clients, "get_context_pack_list", return_value={
            "contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]
        })
        adapter = _adapter(clients, context_mode="manual")
        hooks = adapter.hooks("conv-pack-2", "user-001", inject_context_pack_instruction=False)
        callback = hooks["UserPromptSubmit"][0].hooks[0]
        pack_list = clients["retrieval"].get_context_pack_list

        await _run(_prompt(callback))
        assert _fetches(pack_list) == 1

        _end_turn(adapter)
        result2 = await _run(_prompt(callback))
        assert _fetches(pack_list) == 1
        assert result2 == {}  # already in this conversation's history

    @pytest.mark.asyncio
    async def test_catalog_refetched_on_new_session(self, clients):
        _stub(clients, "get_context_pack_list", return_value={
            "contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]
        })
        adapter = _adapter(clients, context_mode="manual")
        cb_a = adapter.hooks("conv-pack-a", "user-001", inject_context_pack_instruction=False)["UserPromptSubmit"][0].hooks[0]
        cb_b = adapter.hooks("conv-pack-b", "user-001", inject_context_pack_instruction=False)["UserPromptSubmit"][0].hooks[0]
        pack_list = clients["retrieval"].get_context_pack_list

        await _run(_prompt(cb_a))
        assert _fetches(pack_list) == 1
        await _run(_prompt(cb_b))
        assert _fetches(pack_list) == 2

    @pytest.mark.asyncio
    async def test_catalog_fetch_failure_degrades_silently(self, clients):
        _stub(clients, "get_context_pack_list", side_effect=RuntimeError("down"))
        adapter = _adapter(clients, context_mode="manual")
        hooks = adapter.hooks("conv-pack-3", "user-001", inject_context_pack_instruction=False)
        callback = hooks["UserPromptSubmit"][0].hooks[0]
        pack_list = clients["retrieval"].get_context_pack_list

        result = await _run(_prompt(callback))  # must not raise

        assert result == PACK_LIST_ONLY
        pack_list.reset_mock()
        _end_turn(adapter)
        assert await _run(_prompt(callback)) == {}  # already in this conversation's history
        assert _fetches(pack_list) == 0
        assert adapter._allowed_context_pack_list == "[Available context packs]\n{}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "catalog", [None, {"contextPacks": []}], ids=["none-catalog", "empty-catalog"],
    )
    async def test_empty_catalog_does_not_refetch(self, clients, catalog):
        _stub(clients, "get_context_pack_list", return_value=catalog)
        adapter = _adapter(clients, context_mode="manual")
        hooks = adapter.hooks("conv-pack-none", "user-001", inject_context_pack_instruction=False)
        callback = hooks["UserPromptSubmit"][0].hooks[0]
        pack_list = clients["retrieval"].get_context_pack_list

        assert await _run(_prompt(callback)) == PACK_LIST_ONLY
        assert _fetches(pack_list) == 1
        _end_turn(adapter)
        assert await _run(_prompt(callback)) == {}  # already in this conversation's history
        assert _fetches(pack_list) == 1
        assert adapter._allowed_context_pack_list == "[Available context packs]\n{}"


# ─── Stop Hook ────────────────────────────────────────────────────────────────


class TestStopHook:
    """Test the Stop hook pushes transcript to Beacon."""

    @pytest.mark.asyncio
    async def test_calls_push_with_input_data(self, clients):
        adapter = _adapter(clients)

        hooks = adapter.hooks("conv-stop", "user-001", inject_context_pack_instruction=False)
        callback = hooks["Stop"][0].hooks[0]

        input_data = {
            "session_id": "conv-stop",
            "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
            "cwd": "/",
            "last_assistant_message": "Here's your order status!",
        }

        result = await _run(callback(input_data, None, {"signal": None}))

        assert result == {}
        # async push() is fire-and-forget (background task) — yield to let it run
        await asyncio.sleep(0)
        # Verify entries were ingested (lifecycle + add_entries)
        clients["ingestion"].create_conversation.assert_called_once()
        clients["ingestion"].add_entries.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_op_without_transcript_path(self, mock_async_clients):
        adapter = ClaudeAdapter(
            agent_id="agent-test",
            owner_id="user-001",
            **CREDS,
            use_async=True,
        )

        hooks = adapter.hooks("conv-stop-noop", "user-001", inject_context_pack_instruction=False)
        callback = hooks["Stop"][0].hooks[0]

        input_data = {
            "session_id": "sdk-ses-1",
            "cwd": "/",
        }

        result = await callback(input_data, None, {"signal": None})

        assert result == {}
        mock_async_clients["ingestion"].add_entries.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_message_attributed_to_per_session_owner(self, mock_async_clients):
        # A shared, multi-user adapter must stamp each user's transcript with
        # THAT user's owner_id — never the process-wide constructor default.
        adapter = ClaudeAdapter(
            agent_id="agent-test",
            # constructor owner is only a fallback for single-user usage
            owner_id="fallback-owner",
            **CREDS,
            use_async=True,
        )
        # Wire the session for a specific end-user (as a multi-user app does).
        hooks = adapter.hooks("conv-attr", owner_id="alice@corp.com", inject_context_pack_instruction=False)
        callback = hooks["Stop"][0].hooks[0]

        await callback(
            {
                "session_id": "conv-attr",
                "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
                "cwd": "/",
                "last_assistant_message": "Done!",
            },
            None,
            {"signal": None},
        )
        await asyncio.sleep(0)  # let the fire-and-forget push run

        mock_async_clients["ingestion"].add_entries.assert_called_once()
        call = mock_async_clients["ingestion"].add_entries.call_args
        # The session is scoped to this end-user, so delivery carries their owner.
        assert call.kwargs["owner_id"] == "alice@corp.com"
        # And each user_message entry is stamped with that owner (wire dicts).
        entries = call.args[2]
        user_msgs = [e for e in entries if e["type"] == "user_message"]
        assert user_msgs, "fixture should contain a user message"
        assert all(e["participantId"] == "alice@corp.com" for e in user_msgs)
        # constructor default ("default") must NOT leak onto the entries
        assert all(e["participantId"] != "default" for e in user_msgs)

    @pytest.mark.asyncio
    async def test_handles_push_error_gracefully(self, mock_async_clients):
        # Simulate ingestion failure
        mock_async_clients["ingestion"].add_entries = AsyncMock(
            side_effect=RuntimeError("Network error")
        )

        adapter = ClaudeAdapter(
            agent_id="agent-test",
            owner_id="user-001",
            **CREDS,
            use_async=True,
        )

        hooks = adapter.hooks("conv-stop-err", "user-001", inject_context_pack_instruction=False)
        callback = hooks["Stop"][0].hooks[0]

        input_data = {
            "session_id": "sdk-ses-1",
            "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
            "cwd": "/",
            "last_assistant_message": "Hello!",
        }

        # Should NOT raise
        result = await callback(input_data, None, {"signal": None})
        assert result == {}


# ─── Context-mode cadence (auto / manual) ──────────────────────────────────────


def _prompt(callback):
    """Invoke a UserPromptSubmit callback with a canned SDK payload."""
    return callback(
        {"session_id": "sdk-ses", "transcript_path": "/tmp/t.jsonl", "cwd": "/"},
        None,
        {"signal": None},
    )


class TestMemoryModeCadence:
    """The two-mode UserPromptSubmit cadence facade.

    - auto   → seed FAST context once at conversation-open, then agent-driven.
    - manual → no adapter injection at all (tool-only).

    Same knob name/values as the ADK adapter.
    """

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [({}, "auto"), ({"context_mode": "MANUAL"}, "manual")],
        ids=["default-is-auto", "spelling-normalized"],
    )
    def test_resolved_mode(self, mock_async_clients, overrides, expected):
        adapter = ClaudeAdapter(agent_id="a", owner_id="u", **CREDS, use_async=True, **overrides)
        assert adapter.context_mode == expected

    def test_invalid_mode_raises(self, mock_async_clients):
        with pytest.raises(ValueError):
            ClaudeAdapter(
                agent_id="a", owner_id="u", **CREDS, use_async=True,
                context_mode="sometimes",
            )

    def test_set_context_mode_rejects_invalid(self, mock_async_clients):
        """The runtime toggle validates too, and leaves the mode unchanged on
        a rejected switch."""
        adapter = ClaudeAdapter(agent_id="a", owner_id="u", **CREDS, use_async=True)
        with pytest.raises(ValueError, match="'auto', 'manual'"):
            adapter.set_context_mode("sometimes")
        # Mode is unchanged after the rejected switch.
        assert adapter.context_mode == "auto"

    @pytest.mark.asyncio
    async def test_manual_never_injects(self, mock_async_clients):
        """manual — no fast context across turns (tool-only)."""
        adapter = ClaudeAdapter(
            agent_id="a", owner_id="u", **CREDS, use_async=True,
            context_mode="manual",
        )
        cb = adapter.hooks("conv-q1", "u", inject_context_pack_instruction=False)["UserPromptSubmit"][0].hooks[0]
        assert await _prompt(cb) == PACK_LIST_ONLY
        _end_turn(adapter)
        assert await _prompt(cb) == {}  # already in this conversation's history
        mock_async_clients["retrieval"].get_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_seeds_fast_only_at_open(self, clients):
        """auto → fast context seeded once at open, then not again."""
        adapter = _adapter(clients, agent_id="a", owner_id="u", context_mode="auto")
        cb = adapter.hooks("conv-q3", "u", inject_context_pack_instruction=False)["UserPromptSubmit"][0].hooks[0]

        first = await _run(_prompt(cb))
        ctx1 = first["hookSpecificOutput"]["additionalContext"]
        assert clients["fact"] in ctx1
        # auto injects FAST context only — never an instruction block.
        assert "[Context guidance]" not in ctx1
        # Subsequent turns: no fast context (agent drives via the tool), and
        # the catalog is already in this conversation's history.
        _end_turn(adapter)
        assert await _run(_prompt(cb)) == {}
        _end_turn(adapter)
        assert await _run(_prompt(cb)) == {}
        assert clients["retrieval"].get_context.call_count == 1

    @pytest.mark.asyncio
    async def test_runtime_mode_switch_takes_effect_next_turn(self, mock_async_clients):
        """The mode is re-read each turn by the hook, so a runtime toggle takes
        effect on the next turn. auto seeds at conversation-open; flipping to
        manual then suppresses adapter injection."""
        adapter = ClaudeAdapter(
            agent_id="a", owner_id="u", **CREDS, use_async=True,
            context_mode="auto",
        )
        cb = adapter.hooks("conv-q7", "u", inject_context_pack_instruction=False)["UserPromptSubmit"][0].hooks[0]
        first = await _prompt(cb)  # conversation-open seed
        assert "User prefers dark mode" in first["hookSpecificOutput"]["additionalContext"]

        adapter.set_context_mode("manual")
        assert adapter.context_mode == "manual"
        _end_turn(adapter)
        assert await _prompt(cb) == {}  # switched to tool-only, re-read next turn

        adapter.set_context_mode("auto")
        assert adapter.context_mode == "auto"
        # Session already opened, so auto does not re-seed — the toggle is
        # honored (no error), and the open-once cadence still holds.
        _end_turn(adapter)
        assert await _prompt(cb) == {}


# ─── Attach-once: instruction + catalog per Claude conversation ───────────────


_PACKS = {"contextPacks": [{"contextPackId": "PlanRules", "description": "d"}]}


def _turn(hooks, conversation_id="sdk-ses", *, stop=True):
    """Run one sync turn (UserPromptSubmit, then Stop); return the attached text."""
    payload = {"cwd": "/"}
    if conversation_id is not None:
        payload["session_id"] = conversation_id
    result = hooks["UserPromptSubmit"][0].hooks[0](
        {**payload, "transcript_path": "/tmp/t.jsonl"}, None, {"signal": None},
    )
    if stop:
        # No transcript_path, so Stop skips the Beacon push.
        hooks["Stop"][0].hooks[0](payload, None, {"signal": None})
    return result.get("hookSpecificOutput", {}).get("additionalContext", "")


async def _aturn(hooks, conversation_id="sdk-ses", *, stop=True):
    payload = {"cwd": "/"}
    if conversation_id is not None:
        payload["session_id"] = conversation_id
    result = await hooks["UserPromptSubmit"][0].hooks[0](
        {**payload, "transcript_path": "/tmp/t.jsonl"}, None, {"signal": None},
    )
    if stop:
        await hooks["Stop"][0].hooks[0](payload, None, {"signal": None})
    return result.get("hookSpecificOutput", {}).get("additionalContext", "")


class TestAttachOnce:
    """additionalContext stays in Claude's history, so the instruction and
    catalog are attached once per Claude conversation, not every turn."""

    def _adapter(self, clients, packs=_PACKS, **kw):
        clients["retrieval"].get_system_prompt = AsyncMock(return_value="SERVER INSTR")
        clients["retrieval"].get_context_pack_list = AsyncMock(return_value=packs)
        return ClaudeAdapter(
            agent_id="a", owner_id="u", **CREDS, use_async=True,
            context_mode="manual", **kw,
        )

    @pytest.mark.asyncio
    async def test_attached_on_first_turn_only(self, mock_async_clients):
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a1", "u")

        first = await _aturn(hooks)
        assert first.index("SERVER INSTR") < first.index("[Available context packs]")
        assert await _aturn(hooks) == ""
        assert await _aturn(hooks) == ""

    @pytest.mark.asyncio
    async def test_reattached_when_turn_never_reached_stop(self, mock_async_clients):
        """If the hook output may not have reached the model (no Stop for that
        turn), the next turn sends the blocks again."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a1b", "u")

        await _aturn(hooks, stop=False)
        again = await _aturn(hooks)
        assert "SERVER INSTR" in again
        assert "PlanRules" in again
        assert await _aturn(hooks) == ""

    @pytest.mark.asyncio
    async def test_new_conversation_gets_both_again(self, mock_async_clients):
        """A reset (new Claude conversation id) starts with an empty history."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a2", "u")

        await _aturn(hooks, "sdk-1")
        again = await _aturn(hooks, "sdk-2")
        assert "SERVER INSTR" in again
        assert "PlanRules" in again
        # Same Beacon session — the instruction is still fetched only once.
        mock_async_clients["retrieval"].get_system_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pre_compact_reattaches(self, mock_async_clients):
        """Compaction can summarize the earlier attachment away."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a3", "u")
        pre_compact = hooks["PreCompact"][0].hooks[0]

        await _aturn(hooks)
        assert await pre_compact(
            {"session_id": "sdk-ses", "trigger": "auto"}, None, {"signal": None},
        ) == {}
        after = await _aturn(hooks)
        assert "SERVER INSTR" in after
        assert "PlanRules" in after
        assert await _aturn(hooks) == ""

    @pytest.mark.asyncio
    async def test_pre_compact_only_affects_its_conversation(self, mock_async_clients):
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a4", "u")
        pre_compact = hooks["PreCompact"][0].hooks[0]

        await _aturn(hooks, "sdk-1")
        await _aturn(hooks, "sdk-2")
        await pre_compact({"session_id": "sdk-1"}, None, {"signal": None})
        assert await _aturn(hooks, "sdk-2") == ""
        assert "SERVER INSTR" in await _aturn(hooks, "sdk-1")

    @pytest.mark.asyncio
    async def test_changed_catalog_reattached_alone(self, mock_async_clients):
        """A version_expired patch changes the catalog text; the model needs
        the new version, but the unchanged instruction isn't repeated."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a5", "u")

        await _aturn(hooks)
        adapter._allowed_context_pack_list += " "  # stand-in for a version patch
        patched = await _aturn(hooks)
        assert patched == adapter._allowed_context_pack_list
        assert "SERVER INSTR" not in patched
        assert await _aturn(hooks) == ""

    @pytest.mark.asyncio
    async def test_no_conversation_id_attaches_every_turn(self, mock_async_clients):
        """Without an id there's nothing to key on — keep the safe old behaviour."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a6", "u")

        for _ in range(2):
            text = await _aturn(hooks, conversation_id=None)
            assert "SERVER INSTR" in text
            assert "PlanRules" in text
        assert adapter._attached_blocks == {}

    @pytest.mark.asyncio
    async def test_tracking_is_bounded(self, mock_async_clients, monkeypatch):
        monkeypatch.setattr(ClaudeAdapter, "_MAX_TRACKED_CONVERSATIONS", 2)
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a7", "u")

        await _aturn(hooks, "sdk-1")
        await _aturn(hooks, "sdk-2")
        await _aturn(hooks, "sdk-1")  # touch: sdk-2 is now the oldest
        await _aturn(hooks, "sdk-3")
        assert list(adapter._attached_blocks) == ["sdk-1", "sdk-3"]
        assert await _aturn(hooks, "sdk-1") == ""
        # Evicted conversations just get one extra copy.
        assert "SERVER INSTR" in await _aturn(hooks, "sdk-2")

    @pytest.mark.asyncio
    async def test_memory_context_cadence_unchanged(self, mock_async_clients):
        """auto still seeds memory context once per Beacon session; a new
        Claude conversation re-attaches only the instruction and catalog."""
        adapter = self._adapter(mock_async_clients)
        adapter.set_context_mode("auto")
        hooks = adapter.hooks("conv-a8", "u")

        first = await _aturn(hooks, "sdk-1")
        assert "User prefers dark mode" in first
        second = await _aturn(hooks, "sdk-2")
        assert "User prefers dark mode" not in second
        assert "SERVER INSTR" in second

    @pytest.mark.asyncio
    @pytest.mark.parametrize("packs", [{"contextPacks": []}, None])
    async def test_no_packs_no_instruction(self, mock_async_clients, packs):
        """The instruction explains the catalog; with no packs it's only cost."""
        adapter = self._adapter(mock_async_clients, packs=packs)
        hooks = adapter.hooks("conv-a11", "u")

        assert await _aturn(hooks) == "[Available context packs]\n{}"
        mock_async_clients["retrieval"].get_system_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("system_prompt", [
        "MY PROMPT\n\nSERVER INSTR",
        f"MY PROMPT\n\n{CONTEXT_PACK_INSTRUCTION}",
        {"type": "preset", "preset": "claude_code", "append": "SERVER INSTR"},
        {"type": "custom", "prompt": "SERVER INSTR"},
    ])
    async def test_instruction_already_in_system_prompt(
        self, mock_async_clients, system_prompt,
    ):
        """hooks(system_prompt=...) skips the instruction when the prompt
        already carries the server copy or the bundled text."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a12", "u", system_prompt=system_prompt)

        text = await _aturn(hooks)
        assert "PlanRules" in text
        assert "SERVER INSTR" not in text
        assert CONTEXT_PACK_INSTRUCTION not in text

    @pytest.mark.asyncio
    async def test_close_forgets_session_state(self, mock_async_clients):
        """A session id reused after close() fetches its prompt again."""
        adapter = self._adapter(mock_async_clients)
        hooks = adapter.hooks("conv-a13", "u")
        await _aturn(hooks)

        await adapter.close()
        assert adapter._attached_blocks == {}
        assert adapter._context_pack_prompt._pinned("conv-a13") is None

    def test_sync_attached_on_first_turn_only(self, mock_sync_clients):
        mock_sync_clients["retrieval"].get_system_prompt.return_value = "SERVER INSTR"
        mock_sync_clients["retrieval"].get_context_pack_list.return_value = _PACKS
        adapter = ClaudeAdapter(
            agent_id="a", owner_id="u", **CREDS, use_async=False,
            context_mode="manual",
        )
        hooks = adapter.hooks("conv-a9", "u")
        pre_compact = hooks["PreCompact"][0].hooks[0]

        assert "SERVER INSTR" in _turn(hooks)
        assert _turn(hooks) == ""
        assert pre_compact({"session_id": "sdk-ses"}, None, {"signal": None}) == {}
        assert "SERVER INSTR" in _turn(hooks, stop=False)
        assert "SERVER INSTR" in _turn(hooks)  # previous turn never reached Stop
        assert _turn(hooks) == ""

    def test_hooks_existing_hooks_replaces_own_matchers(self, mock_async_clients):
        adapter = self._adapter(mock_async_clients)
        mine = sys.modules["claude_agent_sdk"].HookMatcher(matcher=".*", hooks=[lambda *a: {}])
        once = adapter.hooks("conv-a10", "u", existing_hooks={"Stop": [mine]})
        twice = adapter.hooks("conv-a10", "u", existing_hooks=once)
        assert twice["Stop"][0] is mine
        assert len(twice["Stop"]) == 2
        assert len(twice["UserPromptSubmit"]) == 1
        assert len(twice["PreCompact"]) == 1


# ─── Integration: hooks() with ClaudeAgentOptions ─────────────────────────────


class TestHooksIntegration:
    """Verify hooks dict is compatible with ClaudeAgentOptions."""

    def test_passable_to_claude_agent_options(self, mock_async_clients):
        ClaudeAgentOptions = sys.modules["claude_agent_sdk"].ClaudeAgentOptions

        adapter = ClaudeAdapter(
            agent_id="agent-test",
            owner_id="user-001",
            **CREDS,
            use_async=True,
        )

        hooks = adapter.hooks("conv-int", "user-001", inject_context_pack_instruction=False)

        # This should not raise
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-20250514",
            hooks=hooks,
        )

        assert options.hooks is not None
        assert "UserPromptSubmit" in options.hooks
        assert "Stop" in options.hooks
