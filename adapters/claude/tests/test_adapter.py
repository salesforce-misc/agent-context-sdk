# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for ClaudeAdapter — multi-session, hooks-first design.

The adapter manages N sessions internally. Lifecycle is lazy (triggered
on first ingestion). No separate adapter-manager needed.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Mock claude_agent_sdk if not installed
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

from agentic_context_claude import ClaudeAdapter

FIXTURES = Path(__file__).parent / "fixtures"

# Inline credentials used across constructor tests. The static base_url/api_key
# path is gone — the adapter now always builds a ContextAuth. The client-mock
# fixtures below also patch ContextAuth so the JWT → CDP exchange is faked
# (get_token() returns a deterministic token + base_url) and no network runs.
CREDS = {
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
    "consumer_key": "3MVG.test",
    "subject": "user@org.com",
    "login_url": "login.salesforce.com",
}
FAKE_TOKEN = "test-key"
FAKE_BASE_URL = "https://beacon.test"


@pytest.fixture
def mock_clients():
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
        retrieval.get_context.return_value = {"summary": "test context"}
        retrieval.get_conversation.return_value = {"entries": []}
        retrieval.list_conversations.return_value = [{"id": "conv-1"}]
        retrieval.close.return_value = None
        MockRetrieval.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval}


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
        retrieval.get_context = AsyncMock(return_value={"summary": "async context"})
        retrieval.get_conversation = AsyncMock(return_value={"entries": []})
        retrieval.list_conversations = AsyncMock(return_value=[{"id": "conv-1"}])
        retrieval.close = AsyncMock(return_value=None)
        MockRetrieval.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval}


# ─── Constructor ─────────────────────────────────────────────────────────────

class TestConstructor:
    def test_minimal_constructor(self, mock_async_clients) -> None:
        """Requires agent_id + ContextAuth credentials (here: inline creds)."""
        adapter = ClaudeAdapter(
            agent_id="my-agent",
            owner_id="user-001",
            **CREDS,
        )
        assert adapter.agent_id == "my-agent"
        assert adapter._use_async is True  # default
        assert adapter._sessions == {}  # no sessions until accessed

    def test_sync_mode(self, mock_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="my-agent",
            owner_id="user-001",
            **CREDS,
            use_async=False,
        )
        assert adapter._use_async is False

    @pytest.mark.parametrize(
        "owner", [{}, {"owner_id": ""}], ids=["missing", "empty-string"],
    )
    def test_owner_id_is_required(self, mock_async_clients, owner) -> None:
        """owner_id has no default — a shared adapter must not attribute
        transcripts to a placeholder owner. An empty string is rejected too."""
        with pytest.raises(ValueError, match="requires owner_id"):
            ClaudeAdapter(agent_id="my-agent", **owner, **CREDS)


# ─── Credential-based auth (ContextAuth) ──────────────────────────────────────

class TestAuth:
    """Credential / ContextAuth wiring — mirrors the GoogleAdkAdapter surface."""

    def test_shared_auth_instance_used(self, mock_async_clients) -> None:
        """A shared ContextAuth is used as-is: token+base_url come from it, it
        is threaded into both core clients (so their 401-retry path can
        re-fetch), and both clients are attach()-ed for refresh propagation."""
        auth = MagicMock()
        auth.get_token.return_value = ("cdp-token", "https://beacon.from-auth")

        with patch("agentic_context_claude.adapter.AsyncIngestionClient") as MockIng, \
             patch("agentic_context_claude.adapter.AsyncRetrievalClient") as MockRet:
            MockIng.return_value = MagicMock()
            MockRet.return_value = MagicMock()
            adapter = ClaudeAdapter(agent_id="a", owner_id="o", auth=auth)
            assert MockIng.call_args.kwargs["auth"] is auth
            assert MockRet.call_args.kwargs["auth"] is auth

        assert adapter._auth is auth
        auth.get_token.assert_called_once()
        # base_url discovered from the exchange, not passed in
        assert adapter._base_url == "https://beacon.from-auth"
        # both shared clients registered for refresh propagation
        assert auth.attach.call_count == 2

    def test_inline_credentials_build_context_auth(self, mock_async_clients) -> None:
        """private_key + consumer_key + subject + login_url → SDK builds ContextAuth."""
        with patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
            inst = MagicMock()
            inst.get_token.return_value = ("cdp-token", "https://beacon.exchanged")
            MockAuth.return_value = inst

            adapter = ClaudeAdapter(
                agent_id="a",
                owner_id="o",
                private_key="-----BEGIN PRIVATE KEY-----\n...",
                consumer_key="3MVG...",
                subject="user@org.com",
                login_url="login.salesforce.com",
            )

            MockAuth.assert_called_once()
            assert adapter._auth is inst
            assert adapter._base_url == "https://beacon.exchanged"

    def test_shared_auth_takes_precedence_over_inline_credentials(self, mock_async_clients) -> None:
        """A shared auth= instance wins over inline credentials — the SDK does
        not build a second ContextAuth when one is supplied."""
        auth = MagicMock()
        auth.get_token.return_value = ("cdp-token", "https://beacon.from-auth")
        with patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
            adapter = ClaudeAdapter(
                agent_id="a", owner_id="o", auth=auth, **CREDS,
            )
            MockAuth.assert_not_called()  # inline creds ignored when auth= given
        assert adapter._auth is auth
        assert adapter._base_url == "https://beacon.from-auth"

    @pytest.mark.parametrize(
        "creds",
        [
            {},
            {"private_key": "-----BEGIN PRIVATE KEY-----\n...", "consumer_key": "3MVG..."},
        ],
        ids=["no-credentials", "partial-inline-credentials"],
    )
    def test_missing_or_partial_credentials_raise(self, creds) -> None:
        """Neither a shared auth=, complete inline credentials, nor
        secrets_api_key → clear ValueError listing all three options. Incomplete
        inline creds (missing subject/login_url) raise too, never a silent
        fallback to an unauthenticated client."""
        with pytest.raises(ValueError, match="requires credentials"):
            ClaudeAdapter(agent_id="a", owner_id="o", **creds)

    @pytest.mark.parametrize(
        ("url_kwargs", "expected_url"),
        [({}, None), ({"secrets_url": "https://secrets.internal"}, "https://secrets.internal")],
        ids=["env-fallback-url", "custom-url"],
    )
    def test_secrets_api_key_builds_auth_via_apikeyauth(
        self, mock_async_clients, url_kwargs, expected_url,
    ) -> None:
        """secrets_api_key= → adapter builds an ApiKeyAuth (which downloads the
        creds from the secrets server on first token fetch). With no
        secrets_url, secret_server_url=None defers to ApiKeyAuth's env fallback
        (AGENTIC_CONTEXT_SECRET_SERVER_URL); a custom URL is passed through."""
        with patch("agentic_context_claude.adapter.ApiKeyAuth") as MockApiKeyAuth, \
             patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
            inst = MagicMock()
            inst.get_token.return_value = ("cdp-token", "https://beacon.viakey")
            MockApiKeyAuth.return_value = inst

            adapter = ClaudeAdapter(
                agent_id="a", owner_id="o", secrets_api_key="the-key", **url_kwargs,
            )

            MockApiKeyAuth.assert_called_once()
            assert MockApiKeyAuth.call_args.args[0] == "the-key"
            assert MockApiKeyAuth.call_args.kwargs["secret_server_url"] == expected_url
            # No ContextAuth(...) built directly on the api-key path.
            MockAuth.assert_not_called()
            assert adapter._auth is inst
            assert adapter._base_url == "https://beacon.viakey"

    def test_inline_credentials_take_precedence_over_secrets_api_key(
        self, mock_async_clients
    ) -> None:
        """Explicit inline creds win over secrets_api_key — the broker is not
        contacted when full credentials are already supplied."""
        with patch("agentic_context_claude.adapter.ApiKeyAuth") as MockApiKeyAuth, \
             patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
            inst = MagicMock()
            inst.get_token.return_value = ("cdp-token", "https://beacon.inline")
            MockAuth.return_value = inst

            ClaudeAdapter(agent_id="a", owner_id="o", secrets_api_key="broker-key", **CREDS)

            MockAuth.assert_called_once()          # inline creds → direct build
            MockApiKeyAuth.assert_not_called()     # broker never called


# ─── Push (Async, Per-Turn) ──────────────────────────────────────────────────

class TestPushAsync:
    @pytest.mark.asyncio
    async def test_push_triggers_lazy_lifecycle_and_ingests(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.push("ses-001", {
            "session_id": "ses-001",
            "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
            "last_assistant_message": "Your order has been delivered!",
        })

        # Let background task complete
        await asyncio.sleep(0.05)

        ingestion = mock_async_clients["ingestion"]
        ingestion.create_conversation.assert_called_once()
        ingestion.create_session.assert_called_once()
        ingestion.add_entries.assert_called_once()

        # Verify entries payload
        entries = ingestion.add_entries.call_args[0][2]
        types = [e["type"] for e in entries]
        assert "user_message" in types
        assert "agent_reasoning" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "agent_response" in types

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_data",
        [{"session_id": "ses-001"}, {"transcript_path": "/some/path"}],
        ids=["no-transcript-path", "no-session-id"],
    )
    async def test_push_no_op_on_incomplete_input(self, mock_async_clients, input_data) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.push("ses-001", input_data)

        mock_async_clients["ingestion"].add_entries.assert_not_called()


# ─── Push (Sync) ─────────────────────────────────────────────────────────────

class TestPushSync:
    def test_push_ingests_records(self, mock_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
            use_async=False,
        )

        # Create a session manager then push directly (sync)
        manager = adapter._get_or_create_manager("ses-001")
        records = adapter._extract_turn(
            str(FIXTURES / "sample_transcript.jsonl"),
            "ses-001",
            "Done!",
        )
        manager.ingest(records)

        ingestion = mock_clients["ingestion"]
        ingestion.create_conversation.assert_called_once()
        ingestion.add_entries.assert_called_once()


    def test_session_manager_carries_framework_agent_label(self, mock_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support", owner_id="user-001", **CREDS, use_async=False,
        )
        manager = adapter._get_or_create_manager("ses-001")
        assert manager._agent_participant_variable["AiAgentType"] == "ClaudeAgentSDK"
        assert manager._agent_participant_variable["AiAgentApiName"] == "agent-support"

    def test_agent_participant_variable_full_override_wins(self, mock_clients) -> None:
        custom = {"AiAgentType": "MyRuntime", "AiAgentApiName": "override"}
        adapter = ClaudeAdapter(
            agent_id="agent-support", owner_id="user-001", **CREDS, use_async=False,
            agent_participant_variable=custom,
        )
        manager = adapter._get_or_create_manager("ses-001")
        assert manager._agent_participant_variable == custom


# ─── Turn grouping ───────────────────────────────────────────────────────────

class TestTurnGrouping:
    """All entries from one turn share an interaction_id."""

    def test_turn_records_share_one_interaction_id(self, mock_clients, tmp_path) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support", owner_id="user-001", **CREDS, use_async=False,
        )
        transcript = tmp_path / "t.jsonl"
        transcript.write_text((FIXTURES / "sample_transcript.jsonl").read_text())

        turn1 = adapter._extract_turn(str(transcript), "ses-001", "Done!")
        assert len(turn1) > 2
        assert {r.interaction_id for r in turn1} == {"turn-ses-001-usr-001"}

        with transcript.open("a") as f:
            f.write(json.dumps({
                "type": "user", "uuid": "usr-900", "timestamp": "2026-06-08T20:05:00Z",
                "message": {"role": "user", "content": "One more thing"},
            }) + "\n")
        turn2 = adapter._extract_turn(str(transcript), "ses-001", "Sure.")
        assert [r.event_type for r in turn2] == ["user_message", "agent_response"]
        assert {r.interaction_id for r in turn2} == {"turn-ses-001-usr-900"}


# ─── Beacon Wire Format ──────────────────────────────────────────────────────

class TestBeaconWireFormat:
    @pytest.mark.asyncio
    async def test_entry_has_required_fields(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.push("ses-001", {
            "session_id": "ses-001",
            "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
            "last_assistant_message": "Your order is delivered!",
        })

        await asyncio.sleep(0.05)

        entries = mock_async_clients["ingestion"].add_entries.call_args[0][2]

        for entry in entries:
            # The SDK omits ``id`` so Beacon assigns it — the adapter's
            # fallback event_id repeats per turn and Beacon dedupes on id.
            assert "id" not in entry
            assert "participantId" in entry
            assert "type" in entry
            assert "timestamp" in entry
            assert "category" in entry

            if entry["type"] in ("user_message", "agent_response"):
                assert entry["category"] == "Message"
                assert "content" in entry
            elif entry["type"] in ("tool_call", "tool_result", "agent_reasoning"):
                assert entry["category"] == "Trace"
                assert "variable" in entry

    @pytest.mark.asyncio
    async def test_tool_name_on_both_call_and_result(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.push("ses-001", {
            "session_id": "ses-001",
            "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
            "last_assistant_message": "Done!",
        })

        await asyncio.sleep(0.05)

        entries = mock_async_clients["ingestion"].add_entries.call_args[0][2]
        tool_calls = [e for e in entries if e["type"] == "tool_call"]
        tool_results = [e for e in entries if e["type"] == "tool_result"]

        assert len(tool_calls) >= 1
        assert len(tool_results) >= 1
        assert tool_calls[0]["variable"]["Name"] == "lookup_order"
        assert tool_results[0]["variable"]["Name"] == "lookup_order"


# ─── Retrieval ───────────────────────────────────────────────────────────────

class TestRetrieval:
    @pytest.mark.asyncio
    async def test_get_context(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        result = await adapter.get_context("ses-123")
        assert result == {"summary": "async context"}
        mock_async_clients["retrieval"].get_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_cold_call_scopes_to_owner(self, mock_async_clients) -> None:
        """A cold get_context (unseen session_id) must scope to the passed
        owner_id, not fall back to the adapter default."""
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="default-owner",
            **CREDS,
        )

        await adapter.get_context("cold-sid", owner_id="real-user@example.com")
        # SessionManager.get_context passes (session_id, owner_id, agent_id).
        args = mock_async_clients["retrieval"].get_context.call_args[0]
        assert args[0] == "cold-sid"
        assert args[1] == "real-user@example.com"

    @pytest.mark.asyncio
    async def test_get_session(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        result = await adapter.get_session("ses-123")
        assert result == {"entries": []}

    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        result = await adapter.list_sessions()
        assert result == [{"id": "conv-1"}]
        # Defaults to the adapter's own owner.
        mock_async_clients["retrieval"].list_conversations.assert_called_with(
            "user-001", "agent-support", owner_type=adapter._owner_type,
        )

    @pytest.mark.asyncio
    async def test_list_sessions_scopes_to_passed_owner(self, mock_async_clients) -> None:
        """list_sessions(owner_id) must query Beacon for THAT owner, not the
        adapter default — otherwise a process-wide adapter leaks other users'
        conversations into the caller's list."""
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="default-owner",
            **CREDS,
        )

        await adapter.list_sessions(owner_id="someone-else@example.com")
        mock_async_clients["retrieval"].list_conversations.assert_called_with(
            "someone-else@example.com", "agent-support", owner_type=adapter._owner_type,
        )

    @pytest.mark.asyncio
    async def test_get_context_returns_none_without_session(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        # No session accessed yet → returns None
        result = await adapter.get_context()
        assert result is None


# ─── Eager Lifecycle ─────────────────────────────────────────────────────────

class TestEagerLifecycle:
    @pytest.mark.asyncio
    async def test_start_session_creates_lifecycle(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        result = await adapter.start_session("ses-new", title="Test")

        assert result["sessionId"] == "ses-new"
        mock_async_clients["ingestion"].create_conversation.assert_called_once()
        mock_async_clients["ingestion"].create_session.assert_called_once()
        mock_async_clients["ingestion"].add_participants.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_session_then_push_skips_lifecycle(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.start_session("ses-eager")

        # Reset to track subsequent calls
        mock_async_clients["ingestion"].reset_mock()
        mock_async_clients["ingestion"].add_entries = AsyncMock(return_value={})

        await adapter.push("ses-eager", {
            "session_id": "ses-eager",
            "transcript_path": str(FIXTURES / "sample_transcript.jsonl"),
            "last_assistant_message": "Hi!",
        })

        await asyncio.sleep(0.05)

        # Lifecycle NOT called again
        mock_async_clients["ingestion"].create_conversation.assert_not_called()
        mock_async_clients["ingestion"].create_session.assert_not_called()
        # But entries are posted
        mock_async_clients["ingestion"].add_entries.assert_called_once()

    @pytest.mark.asyncio
    async def test_returning_user_409_is_success(self, mock_async_clients) -> None:
        """Calling start_session with same ID is fine (409 = returning user)."""
        from agentic_context.client import IngestionError

        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        # First call works
        await adapter.start_session("ses-return")

        # Simulate returning user — 409 on lifecycle
        mock_async_clients["ingestion"].create_conversation = AsyncMock(
            side_effect=IngestionError(409, "exists")
        )
        mock_async_clients["ingestion"].create_session = AsyncMock(
            side_effect=IngestionError(409, "exists")
        )
        mock_async_clients["ingestion"].add_participants = AsyncMock(
            side_effect=IngestionError(409, "exists")
        )

        # Reset internal session state to simulate fresh adapter
        manager = adapter._sessions["ses-return"]
        manager._conversation_created = False
        manager._session_created = False

        # Should not raise (409 = success)
        result = await adapter.start_session("ses-return")
        assert result["sessionId"] == "ses-return"


# ─── Multi-Session ───────────────────────────────────────────────────────────

class TestMultiSession:
    @pytest.mark.asyncio
    async def test_manages_multiple_sessions(self, mock_async_clients) -> None:
        """One adapter instance manages N sessions."""
        adapter = ClaudeAdapter(
            agent_id="my-agent",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.start_session("ses-1")
        await adapter.start_session("ses-2")

        assert "ses-1" in adapter._sessions
        assert "ses-2" in adapter._sessions
        assert len(adapter._sessions) == 2

    @pytest.mark.asyncio
    async def test_hooks_creates_session_manager_lazily(self, mock_async_clients) -> None:
        """hooks() creates a session manager without calling start_session."""
        adapter = ClaudeAdapter(
            agent_id="my-agent",
            owner_id="user-001",
            **CREDS,
        )

        assert len(adapter._sessions) == 0

        hooks = adapter.hooks("ses-lazy", "user-001")

        assert "ses-lazy" in adapter._sessions
        assert "UserPromptSubmit" in hooks
        assert "Stop" in hooks

    @pytest.mark.asyncio
    async def test_hooks_merges_existing_hooks(self, mock_async_clients) -> None:
        """existing_hooks are preserved and our memory hooks appended alongside,
        so an agent that already uses hooks stays drop-in."""
        from claude_agent_sdk import HookMatcher

        adapter = ClaudeAdapter(
            agent_id="my-agent",
            owner_id="user-001",
            **CREDS,
        )

        my_stop = HookMatcher(matcher=".*", hooks=["my_stop_cb"])
        my_pretool = HookMatcher(matcher="Bash", hooks=["my_pretool_cb"])
        existing = {"PreToolUse": [my_pretool], "Stop": [my_stop]}

        merged = adapter.hooks("ses-1", "user-001", existing_hooks=existing)

        # Caller's own event survives untouched.
        assert merged["PreToolUse"] == [my_pretool]
        # Shared event keeps the caller's matcher FIRST, ours appended.
        assert merged["Stop"][0] is my_stop
        assert len(merged["Stop"]) == 2
        # Our memory-only event is added.
        assert "UserPromptSubmit" in merged
        assert len(merged["UserPromptSubmit"]) == 1
        # The caller's dict is not mutated.
        assert len(existing["Stop"]) == 1
        assert "UserPromptSubmit" not in existing

    @pytest.mark.asyncio
    async def test_session_id_property_returns_last(self, mock_async_clients) -> None:
        adapter = ClaudeAdapter(
            agent_id="my-agent",
            owner_id="user-001",
            **CREDS,
        )

        assert adapter.session_id is None
        adapter.hooks("ses-a", "user-001")
        adapter.hooks("ses-b", "user-001")
        assert adapter.session_id == "ses-b"


# ─── Close ───────────────────────────────────────────────────────────────────

class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up_all_sessions(self, mock_async_clients) -> None:
        """close() is local cleanup only — no close_session to Beacon."""
        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        await adapter.start_session("ses-1")
        await adapter.start_session("ses-2")

        await adapter.close()

        assert adapter._sessions == {}
        mock_async_clients["ingestion"].close.assert_awaited_once()
        mock_async_clients["retrieval"].close.assert_awaited_once()
        mock_async_clients["ingestion"].close_session.assert_not_awaited()


# ─── Ingest Direct ───────────────────────────────────────────────────────────

class TestIngestDirect:
    @pytest.mark.asyncio
    async def test_ingest_records_directly(self, mock_async_clients) -> None:
        from agentic_context import Record
        from datetime import datetime, timezone

        adapter = ClaudeAdapter(
            agent_id="agent-support",
            owner_id="user-001",
            **CREDS,
        )

        records = [Record(
            event_id="test-1",
            session_id="ses-direct",
            agent_id="agent-support",
            event_type="user_message",
            participant_id="user-001",
            content="hello",
            category="Message",
            timestamp=datetime.now(timezone.utc),
        )]

        await adapter.ingest("ses-direct", records)
        await asyncio.sleep(0.05)

        mock_async_clients["ingestion"].create_conversation.assert_called_once()
        mock_async_clients["ingestion"].add_entries.assert_called_once()
