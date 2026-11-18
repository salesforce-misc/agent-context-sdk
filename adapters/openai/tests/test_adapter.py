# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for OpenAIAdapter — mirrors ClaudeAdapter's test shape.

The adapter manages N sessions internally. Lifecycle is lazy (triggered on
first ingestion). No separate adapter-manager needed.
"""
from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_AGENTS_SDK_AVAILABLE = importlib.util.find_spec("agents") is not None

# The pure-ingest path (snapshot/push/ingest) does NOT need openai-agents, so
# we don't mock the SDK into sys.modules — the adapter imports it lazily inside
# make_context_retrieval_tool(). Tests that touch the tool skip when absent.
from agentic_context import ContextSource
from agentic_context_openai import OpenAIAdapter

CREDS = {
    "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----",
    "consumer_key": "3MVG.test",
    "subject": "user@org.com",
    "login_url": "login.salesforce.com",
}
FAKE_TOKEN = "test-token"
FAKE_BASE_URL = "https://beacon.test"


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mock_async_clients():
    """Patch the async clients + ContextAuth so no network/JWT runs."""
    with patch("agentic_context_openai.adapter.AsyncIngestionClient") as MockIng, \
         patch("agentic_context_openai.adapter.AsyncRetrievalClient") as MockRet, \
         patch("agentic_context_openai.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        MockAuth.return_value = auth_inst

        ingestion = MagicMock()
        ingestion.create_conversation = AsyncMock(return_value={})
        ingestion.create_session = AsyncMock(return_value={})
        ingestion.add_participants = AsyncMock(return_value={})
        ingestion.add_entries = AsyncMock(return_value={})
        ingestion.close = AsyncMock(return_value=None)
        MockIng.return_value = ingestion

        retrieval = MagicMock()
        retrieval.get_context = AsyncMock(return_value={"userContext": [{"entry": "x"}]})
        retrieval.get_conversation = AsyncMock(return_value={"entries": []})
        retrieval.list_conversations = AsyncMock(return_value=[{"id": "conv-1"}])
        retrieval.close = AsyncMock(return_value=None)
        MockRet.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval, "auth": auth_inst}


@pytest.fixture
def mock_sync_clients():
    with patch("agentic_context_openai.adapter.IngestionClient") as MockIng, \
         patch("agentic_context_openai.adapter.RetrievalClient") as MockRet, \
         patch("agentic_context_openai.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = (FAKE_TOKEN, FAKE_BASE_URL)
        MockAuth.return_value = auth_inst

        ingestion = MagicMock()
        ingestion.create_conversation.return_value = {}
        ingestion.create_session.return_value = {}
        ingestion.add_participants.return_value = {}
        ingestion.add_entries.return_value = {}
        ingestion.close.return_value = None
        MockIng.return_value = ingestion

        retrieval = MagicMock()
        retrieval.get_context.return_value = {"userContext": []}
        retrieval.list_conversations.return_value = []
        retrieval.close.return_value = None
        MockRet.return_value = retrieval

        yield {"ingestion": ingestion, "retrieval": retrieval, "auth": auth_inst}


# ─── Constructor ─────────────────────────────────────────────────

class TestConstructor:
    def test_minimal_constructor(self, mock_async_clients) -> None:
        adapter = OpenAIAdapter(agent_id="my-agent", owner_id="user-001", **CREDS)
        assert adapter.agent_id == "my-agent"
        assert adapter._use_async is True
        assert adapter._sessions == {}

    def test_sync_mode(self, mock_sync_clients) -> None:
        adapter = OpenAIAdapter(
            agent_id="my-agent", owner_id="user-001", use_async=False, **CREDS,
        )
        assert adapter._use_async is False

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (CREDS, "requires owner_id"),
            ({"owner_id": "", **CREDS}, "requires owner_id"),
            ({"owner_id": "user-001"}, "requires credentials"),
        ],
        ids=["owner-id-missing", "owner-id-empty", "credentials-missing"],
    )
    def test_constructor_validation(self, mock_async_clients, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            OpenAIAdapter(agent_id="my-agent", **kwargs)

    def test_shared_auth_takes_precedence(self, mock_async_clients) -> None:
        auth = MagicMock()
        auth.get_token.return_value = ("cdp-token", "https://beacon.shared")
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", auth=auth)
        assert adapter._auth is auth
        assert adapter._base_url == "https://beacon.shared"
        assert auth.attach.call_count == 2  # both clients attach for refresh

    def test_inline_credentials_build_context_auth(self, mock_async_clients) -> None:
        with patch("agentic_context_openai.adapter.ContextAuth") as MockAuth:
            inst = MagicMock()
            inst.get_token.return_value = ("cdp-token", "https://beacon.exchanged")
            MockAuth.return_value = inst

            adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
            MockAuth.assert_called_once()
            assert adapter._auth is inst
            assert adapter._base_url == "https://beacon.exchanged"

    def test_api_key_builds_api_key_auth(self, mock_async_clients) -> None:
        with patch("agentic_context_openai.adapter.ApiKeyAuth") as MockApiKey:
            inst = MagicMock()
            inst.get_token.return_value = ("cdp-token", "https://beacon.secrets")
            inst.default_owner_type = "individual"
            MockApiKey.return_value = inst
            adapter = OpenAIAdapter(
                agent_id="a", owner_id="o",
                secrets_api_key="sk-xyz", secrets_url="https://secrets.test",
            )
            MockApiKey.assert_called_once_with(
                "sk-xyz", secret_server_url="https://secrets.test",
            )
            assert adapter._auth is inst


# ─── Agent participant variable ──────────────────────────────────

class TestAgentParticipantVariable:
    """Beacon analytics bucket (the ``agent_participant_variable`` override)."""

    def test_default_is_openai_label(self, mock_async_clients) -> None:
        adapter = OpenAIAdapter(agent_id="support-bot", owner_id="o", **CREDS)
        assert adapter._agent_participant_variable["AiAgentType"] == "OpenAI"
        assert adapter._agent_participant_variable["AiAgentApiName"] == "support-bot"

    def test_full_override_wins(self, mock_async_clients) -> None:
        custom = {"AiAgentType": "CustomOpenAI", "AiAgentApiName": "override"}
        adapter = OpenAIAdapter(
            agent_id="support-bot", owner_id="o",
            agent_participant_variable=custom, **CREDS,
        )
        assert adapter._agent_participant_variable == custom


# ─── ContextSource conformance ───────────────────────────────────

class TestContextSourceConformance:
    """``isinstance(adapter, ContextSource)`` must hold."""

    def test_isinstance_context_source(self, mock_async_clients) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        assert isinstance(adapter, ContextSource)
        assert adapter.capabilities.working is True
        assert adapter.capabilities.persistent is False


# ─── Snapshot ─────────────────────────────────────────────────────

class TestSnapshot:
    """snapshot() is the ContextSource contract — accepts both wire formats."""

    @pytest.mark.parametrize(
        ("kwargs", "event_types", "participant_ids"),
        [
            (
                {"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]},
                ["user_message", "agent_response"], ["default-owner", "a"],
            ),
            (
                {"items": [
                    {"type": "function_call", "name": "lookup",
                     "call_id": "fc_1", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "fc_1", "output": "done"},
                ]},
                ["tool_call", "tool_result"], None,
            ),
            # Both inputs can be supplied — appended in order and sniffed per-entry.
            (
                {"messages": [{"role": "user", "content": "hi"}],
                 "items": [{"type": "reasoning", "content": "thinking…"}]},
                ["user_message", "agent_reasoning"], None,
            ),
            ({}, [], None),
            # Per-call owner wins over the constructor default — multi-user path.
            (
                {"messages": [{"role": "user", "content": "hi"}], "owner_id": "alice@corp.com"},
                ["user_message"], ["alice@corp.com"],
            ),
        ],
        ids=["chat-completions-messages", "responses-items", "mixed-formats", "empty",
             "owner-id-override"],
    )
    def test_snapshot(self, mock_async_clients, kwargs, event_types, participant_ids) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="default-owner", **CREDS)
        records = adapter.snapshot("ses-1", **kwargs)
        assert [r.event_type for r in records] == event_types
        if participant_ids is not None:
            assert [r.participant_id for r in records] == participant_ids


# ─── Push (async) ────────────────────────────────────────────────

class TestPushAsync:
    """push() is the primary entry point — lazy lifecycle, non-blocking ingest."""

    async def test_push_messages_calls_manager_ingest(
        self, mock_async_clients,
    ) -> None:
        """A second push on the same session_id reuses the same manager —
        lifecycle (create_conversation/create_session) is lazy ONCE."""
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr.ingest = AsyncMock(return_value=None)
            mgr._owner_id = "o"
            MockMgr.return_value = mgr

            await adapter.push("ses-1", messages=[
                {"role": "user", "content": "hi"},
            ])
            mgr.ingest.assert_awaited_once()
            batch = mgr.ingest.await_args.args[0]
            assert len(batch) == 1
            assert batch[0].event_type == "user_message"

            await adapter.push("ses-1", messages=[{"role": "user", "content": "b"}])
            assert MockMgr.call_count == 1
            assert mgr.ingest.await_count == 2

    async def test_push_empty_is_noop(self, mock_async_clients) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            await adapter.push("ses-1")
            MockMgr.assert_not_called()


# ─── Ingest (direct) ─────────────────────────────────────────────

class TestIngestDirect:
    """ingest() lets callers pre-build Records and bypass the transformer."""

    async def test_ingest_passes_through(self, mock_async_clients) -> None:
        from datetime import datetime, timezone
        from agentic_context import Record

        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        rec = Record(
            event_id="custom-1", session_id="ses-1", agent_id="a",
            event_type="user_message", participant_id="o",
            content="direct", timestamp=datetime.now(timezone.utc),
        )
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr.ingest = AsyncMock(return_value=None)
            MockMgr.return_value = mgr
            await adapter.ingest("ses-1", [rec])
            mgr.ingest.assert_awaited_once_with([rec])


# ─── Retrieval ───────────────────────────────────────────────────

class TestRetrieval:
    async def test_get_context_routes_through_manager(
        self, mock_async_clients,
    ) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr._owner_id = "o"
            mgr.ingest = AsyncMock(return_value=None)
            mgr.get_context = AsyncMock(return_value={"userContext": [{"entry": "x"}]})
            MockMgr.return_value = mgr
            await adapter.push("ses-1", messages=[{"role": "user", "content": "hi"}])
            ctx = await adapter.get_context("ses-1")
            assert ctx == {"userContext": [{"entry": "x"}]}

    async def test_get_context_returns_none_when_no_session(
        self, mock_async_clients,
    ) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        assert await adapter.get_context() is None

    async def test_list_sessions_bypasses_manager_cache(
        self, mock_async_clients,
    ) -> None:
        """list_sessions must call retrieval directly with the resolved owner —
        routing through an arbitrary cached SessionManager would leak another
        user's sessions in a multi-user process."""
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        result = await adapter.list_sessions(owner_id="alice")
        mock_async_clients["retrieval"].list_conversations.assert_awaited_once()
        args = mock_async_clients["retrieval"].list_conversations.await_args.args
        assert args[0] == "alice"
        assert result == [{"id": "conv-1"}]


# ─── Context mode ────────────────────────────────────────────────

class TestContextMode:
    def test_set_context_mode_runtime_toggle(self, mock_async_clients) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        adapter.set_context_mode("manual")
        assert adapter.context_mode == "manual"
        adapter.set_context_mode("auto")
        assert adapter.context_mode == "auto"

    def test_invalid_mode_raises(self, mock_async_clients) -> None:
        with pytest.raises(ValueError, match="context_mode must be one of"):
            OpenAIAdapter(agent_id="a", owner_id="o", context_mode="bogus", **CREDS)

    @pytest.mark.parametrize(
        ("mode_kwargs", "mode", "inject_at_open"),
        [({}, "auto", True), ({"context_mode": "manual"}, "manual", False)],
        ids=["default-auto-injects-only-at-open", "manual-never-injects"],
    )
    def test_fast_context_injection_by_mode(
        self, mock_async_clients, mode_kwargs, mode, inject_at_open,
    ) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **mode_kwargs, **CREDS)
        assert adapter.context_mode == mode
        assert adapter._should_inject_fast_context("s", is_open=True) is inject_at_open
        assert adapter._should_inject_fast_context("s", is_open=False) is False


# ─── Close ────────────────────────────────────────────────────────

class TestClose:
    async def test_close_drains_managers_and_clients(
        self, mock_async_clients,
    ) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr._owner_id = "o"
            mgr.ingest = AsyncMock(return_value=None)
            mgr.close = AsyncMock(return_value=None)
            MockMgr.return_value = mgr
            await adapter.push("ses-1", messages=[{"role": "user", "content": "hi"}])
            await adapter.close()
            mgr.close.assert_awaited_once()
        mock_async_clients["ingestion"].close.assert_awaited_once()
        mock_async_clients["retrieval"].close.assert_awaited_once()
        assert adapter._sessions == {}
        await adapter.close()  # idempotent: second call must not raise


# ─── On-demand recall tool (Agents SDK) ───────────────────────────

@pytest.mark.skipif(
    not _AGENTS_SDK_AVAILABLE, reason="openai-agents not installed",
)
class TestRecallTool:
    """Exercised only when ``openai-agents`` is present on the test host."""

    def test_make_context_retrieval_tool_returns_function_tool(
        self, mock_async_clients,
    ) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        tool = adapter.make_context_retrieval_tool("ses-1", "o")
        # Agents SDK function_tool wraps the callable in a FunctionTool object;
        # name should flow through name_override. Duck-type the result rather
        # than importing the Agents SDK class in the test.
        assert getattr(tool, "name", None) == "recall_customer_context" or callable(tool)


# Constructor's SDK-missing warning + the lazy ImportError from the recall
# tool are covered in test_sdk_missing.py (matches the Claude/ADK layout).


# ─── PR-review regression tests ──────────────────────────────────

class TestFastContextSeedingAtOpen:
    """Finding #1: push() in auto mode must seed fast-context on the first turn
    of a session and skip the seed on later turns (same cadence as Claude's
    UserPromptSubmit hook). A prior revision had _should_inject_fast_context
    defined but never called — making ``auto`` behave identically to ``manual``.
    """

    async def test_push_seeds_fast_context_once_per_session(
        self, mock_async_clients,
    ) -> None:
        """A successful auto-seed must also cache the payload so the recall
        tool serves it without a second Beacon round-trip."""
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        payload = {"userContext": [{"k": "v"}]}
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr._owner_id = "o"
            mgr.ingest = AsyncMock(return_value=None)
            mgr.get_context = AsyncMock(return_value={"userContext": []})
            MockMgr.return_value = mgr
            with patch.object(
                adapter, "get_context", AsyncMock(return_value=payload),
            ) as seed:
                await adapter.push(
                    "ses-open", messages=[{"role": "user", "content": "hi"}],
                )
                await adapter.push(
                    "ses-open", messages=[{"role": "user", "content": "again"}],
                )
                assert seed.await_count == 1
        assert adapter._seeded_contexts["ses-open"] == payload

    async def test_push_in_manual_mode_does_not_seed(
        self, mock_async_clients,
    ) -> None:
        adapter = OpenAIAdapter(
            agent_id="a", owner_id="o", context_mode="manual", **CREDS,
        )
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr._owner_id = "o"
            mgr.ingest = AsyncMock(return_value=None)
            MockMgr.return_value = mgr
            with patch.object(
                adapter, "get_context", AsyncMock(return_value={"userContext": []}),
            ) as seed:
                await adapter.push(
                    "ses-m", messages=[{"role": "user", "content": "hi"}],
                )
                seed.assert_not_awaited()

    async def test_failed_seed_does_not_mark_session_opened(
        self, mock_async_clients,
    ) -> None:
        """When the auto-seed raises, the session must stay un-opened so the
        next push retries the seed. A prior revision marked opened regardless
        of success, swallowing every retry."""
        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr._owner_id = "o"
            mgr.ingest = AsyncMock(return_value=None)
            MockMgr.return_value = mgr
            with patch.object(
                adapter, "get_context",
                AsyncMock(side_effect=RuntimeError("boom")),
            ) as seed:
                await adapter.push(
                    "ses-fail", messages=[{"role": "user", "content": "hi"}],
                )
                await adapter.push(
                    "ses-fail", messages=[{"role": "user", "content": "again"}],
                )
                # Both pushes must retry the seed — the first failure doesn't
                # count as "opened".
                assert seed.await_count == 2
        assert "ses-fail" not in adapter._fast_context_seeded
        assert "ses-fail" not in adapter._seeded_contexts


async def _invoke_openai_tool(tool, **kwargs) -> str:
    """Invoke an Agents SDK FunctionTool wrapper and return its raw string result."""
    import json as _json
    from agents.tool_context import ToolContext

    ctx = ToolContext(
        context=None, tool_name=tool.name,
        tool_call_id="call_test", tool_arguments={},
    )
    return await tool.on_invoke_tool(ctx, _json.dumps(kwargs))


class TestRecallToolSessionBinding:
    """Finding #4: tools built without a session binding must return a
    not_ready envelope instead of leaking context from the adapter's most
    recent session (which belongs to a different end-user in a shared
    adapter)."""

    async def test_recall_without_bound_session_returns_not_ready(
        self, mock_async_clients,
    ) -> None:
        import json as _json

        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        # Prime the adapter with a session so _last_session_id is non-None;
        # the leak would surface that session to a tool that has no binding.
        adapter._sessions["other-session"] = MagicMock()
        tool = adapter.make_context_retrieval_tool()  # no session_id bound
        raw = await _invoke_openai_tool(tool)
        payload = _json.loads(raw)
        assert payload["status"] == "not_ready"

    async def test_pack_tool_without_bound_session_returns_not_ready(
        self, mock_async_clients,
    ) -> None:
        import json as _json

        adapter = OpenAIAdapter(agent_id="a", owner_id="o", **CREDS)
        adapter._sessions["other-session"] = MagicMock()
        adapter._allowed_pack_ids = {"profile"}
        tool = adapter.make_get_context_pack_tool(
            ["profile"],  # no session_id bound
        )
        raw = await _invoke_openai_tool(tool, context_pack_id="profile", params="")
        payload = _json.loads(raw)
        assert payload["status"] == "not_ready"
        assert "session bound to tool" in payload["detail"]

    async def test_recall_cache_hit_rejects_owner_mismatch(
        self, mock_async_clients,
    ) -> None:
        """A seeded cache must NOT short-circuit the session↔owner check.
        After session-A is seeded for owner-A, a tool bound to (session=A,
        owner=B) must refuse to serve owner-A's cached context."""
        import json as _json

        adapter = OpenAIAdapter(agent_id="a", owner_id="owner-A", **CREDS)
        # Session-A is bound to owner-A and has a seeded payload.
        adapter._get_or_create_manager("ses-A", "owner-A")
        adapter._sessions["ses-A"]._owner_id = "owner-A"
        adapter._seeded_contexts["ses-A"] = {"userContext": [{"secret": "A"}]}

        # A tool bound to session-A but owner-B must be rejected.
        tool = adapter.make_context_retrieval_tool(
            session_id="ses-A", owner_id="owner-B",
        )
        raw = await _invoke_openai_tool(tool)
        payload = _json.loads(raw)
        assert payload["status"] == "not_ready"
        # And the owner-A cache payload is never surfaced.
        assert "secret" not in raw


class TestRecallToolDropsOwnerIdFromModelParams:
    """Finding #2: the model's free-form ``params`` JSON must never place
    ``owner_id`` on the outgoing request — the parsing boundary strips it.
    """

    async def test_pack_tool_strips_model_owner_id(
        self, mock_async_clients,
    ) -> None:
        import json as _json

        adapter = OpenAIAdapter(agent_id="a", owner_id="real-owner", **CREDS)
        adapter._get_or_create_manager("ses-pack")
        mgr = adapter._sessions["ses-pack"]
        mgr._owner_id = "real-owner"
        adapter._allowed_pack_ids = {"profile"}
        tool = adapter.make_get_context_pack_tool(
            ["profile"], session_id="ses-pack",
        )
        mock_async_clients["retrieval"].get_context_pack = AsyncMock(
            return_value={"ok": True},
        )
        await _invoke_openai_tool(
            tool,
            context_pack_id="profile",
            params=_json.dumps({"owner_id": "attacker"}),
        )
        _, kwargs = mock_async_clients["retrieval"].get_context_pack.call_args
        sent_params = kwargs["params"]
        # owner_id was stripped at the parsing boundary — not present in the
        # outgoing params. (The retrieval client re-stamps it from the trusted
        # argument on the request body itself, but params stays clean.)
        assert "owner_id" not in sent_params


class TestCachedManagerOwnerMismatch:
    """Finding #5: once a session_id is bound to an owner, reusing the same
    session_id for a different owner must be refused — otherwise the second
    user's transcript would be attributed to the first owner on Beacon."""

    async def test_cached_manager_rejects_owner_mismatch(
        self, mock_async_clients,
    ) -> None:
        adapter = OpenAIAdapter(agent_id="a", owner_id="alice", **CREDS)
        with patch(
            "agentic_context_openai.adapter.AsyncSessionManager"
        ) as MockMgr:
            mgr = MagicMock()
            mgr._owner_id = "alice"
            mgr.ingest = AsyncMock(return_value=None)
            MockMgr.return_value = mgr
            await adapter.push(
                "ses-shared", messages=[{"role": "user", "content": "hi"}],
                owner_id="alice",
            )
            with pytest.raises(ValueError, match="already bound to owner"):
                await adapter.push(
                    "ses-shared", messages=[{"role": "user", "content": "hi"}],
                    owner_id="bob",
                )
