# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for GoogleAdkAdapter — verifies Record output per callback."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_context import AsyncIngestionClient, AsyncRetrievalClient, AsyncSessionManager, Capabilities, Record
from agentic_context_adk import GoogleAdkAdapter


def make_adapter() -> tuple[GoogleAdkAdapter, AsyncMock]:
    """Return adapter wired to a real AsyncSessionManager backed by a mock HTTP client."""
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_response({}))
    http.aclose = AsyncMock()

    client = AsyncIngestionClient(
        base_url="https://beacon.test",
        api_key="test-key",
        client=http,
    )
    mgr = AsyncSessionManager(client, agent_id="test-agent", owner_id="user@test.com")
    adapter = GoogleAdkAdapter(
        agent_id="test-agent",
        owner_id="user@test.com",
        session_manager=mgr,
        # These are record-plumbing tests: pin manual so the adapter injects no
        # context/instructions (auto — the default — would fast-seed at open and
        # embed the sufficiency instruction, perturbing the assertions here).
        context_mode="manual",
    )
    return adapter, http


def _mock_response(body: dict, status: int = 200):
    """httpx-compatible mock response."""
    import json as _json
    resp = MagicMock()
    resp.status_code = status
    resp.text = _json.dumps(body)
    return resp


def make_ctx(session_id: str = "sess-1") -> SimpleNamespace:
    """Model the real ADK CallbackContext: session_id lives at ctx.session.id."""
    return SimpleNamespace(session=SimpleNamespace(id=session_id))


def make_tool_ctx(session_id: str = "sess-1") -> SimpleNamespace:
    """Model the real ADK ToolContext: session_id lives at ctx.session.id."""
    return SimpleNamespace(session=SimpleNamespace(id=session_id))


def make_llm_request(user_text: str = "") -> MagicMock:
    req = MagicMock()
    if user_text:
        part = MagicMock()
        part.text = user_text
        block = MagicMock()
        block.role = "user"
        block.parts = [part]
        req.contents = [block]
    else:
        req.contents = []
    return req


def make_llm_response(text: str = "", is_tool_call: bool = False) -> MagicMock:
    resp = MagicMock()
    resp.usage_metadata = None
    part = MagicMock()
    part.text = text or None
    if is_tool_call:
        part.function_call = MagicMock()
    else:
        part.function_call = None
    resp.content.parts = [part]
    return resp


def make_tool(name: str = "my_tool") -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


# ── capabilities ─────────────────────────────────────────────────────────────

def test_capabilities():
    adapter, _ = make_adapter()
    assert adapter.capabilities == Capabilities(working=True, persistent=True)


# ── session_id resolved from ctx.session.id ───────────────────────────────────

def test_session_id_read_from_ctx_session_id():
    """Adapter must read ctx.session.id, not ctx.session_id."""
    adapter, _ = make_adapter()
    ctx = make_ctx("real-session-42")
    adapter._on_before_model(ctx, make_llm_request("Hello"))
    assert "real-session-42" in adapter._buffers


def test_no_session_is_silent_noop():
    """When ctx.session is None the adapter does nothing."""
    adapter, _ = make_adapter()
    ctx = SimpleNamespace(session=None)
    adapter._on_before_model(ctx, make_llm_request("hi"))
    assert adapter._buffers == {}


# ── user_message record ───────────────────────────────────────────────────────

def test_user_message_buffered_in_before_model():
    adapter, _ = make_adapter()
    adapter._on_before_model(make_ctx("s1"), make_llm_request("Hello agent"))

    buf = adapter._buffers["s1"]
    user_records = [r for r in buf if r.event_type == "user_message"]
    assert len(user_records) == 1
    assert user_records[0].content == "Hello agent"
    assert user_records[0].participant_id == "user@test.com"
    assert user_records[0].category == "Message"
    # Every buffered record is a core Record stamped with the turn's session.
    for r in buf:
        assert isinstance(r, Record)
        assert r.session_id == "s1"


def test_no_user_message_when_contents_empty():
    adapter, _ = make_adapter()
    adapter._on_before_model(make_ctx("s2"), make_llm_request(""))
    buf = adapter._buffers.get("s2", [])
    assert not any(r.event_type == "user_message" for r in buf)


# ── interaction_id is fresh per turn ─────────────────────────────────────────

async def test_interaction_id_shared_within_turn_and_fresh_per_turn():
    adapter, _ = make_adapter()
    ctx = make_ctx("s3")
    tool_ctx = make_tool_ctx("s3")

    adapter._on_before_model(ctx, make_llm_request("Do it"))
    adapter._on_after_model(ctx, make_llm_response("Thinking", is_tool_call=True))
    adapter._on_before_tool(make_tool("run"), {"x": 1}, tool_ctx)
    adapter._on_after_tool(make_tool("run"), {}, tool_ctx, "done")
    id1 = adapter._interaction_ids["s3"]
    # All records in one turn share the turn's interaction_id.
    assert {r.interaction_id for r in adapter._buffers["s3"]} == {id1}

    await adapter._on_after_model_async(ctx, make_llm_response("Reply"))
    adapter._on_before_model(ctx, make_llm_request("Hi again"))
    assert adapter._interaction_ids["s3"] != id1


# ── set_owner ─────────────────────────────────────────────────────────────────

def test_set_owner_without_turn_updates_default_owner_and_resets_sessions():
    # No turn context (no _current_session_id set) → set_owner updates the
    # default owner, which the owner_id property resolves to, and re-arms
    # conversation-open for every session. Per-session manager rebinding under
    # an active turn is covered by the interleaving regression tests below.
    adapter, _ = make_adapter()
    adapter._sessions_initialised.add("old-sess")
    adapter.set_owner("alice@example.com")
    assert adapter.owner_id == "alice@example.com"
    assert adapter._owner_participant_id == "alice@example.com"
    assert adapter._default_owner == "alice@example.com"
    assert len(adapter._sessions_initialised) == 0


def test_set_owner_before_mgr_built():
    # Regression: bootstrapping owner_id at construction (before the first turn
    # has fired _ensure_clients) used to crash. set_owner should update the
    # default owner and NOT touch the event loop / build any manager;
    # _ensure_clients then builds the shared clients on the first turn, and
    # managers are built lazily per session.
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy",
        consumer_key="dummy",
        subject="dummy@example.com",
        login_url="login.example.com",
    )
    adapter = GoogleAdkAdapter(agent_id="test-agent", auth=auth)
    assert adapter._client is None  # confirm lazy-init path — no clients yet
    assert not adapter._managers  # no per-session managers yet

    adapter.set_owner("alice@example.com")

    assert adapter.owner_id == "alice@example.com"
    assert adapter._owner_participant_id == "alice@example.com"
    assert adapter._client is None  # still not built
    assert not adapter._managers  # still no managers


def test_session_manager_carries_framework_agent_label():
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy",
        consumer_key="dummy",
        subject="dummy@example.com",
        login_url="login.example.com",
    )
    adapter = GoogleAdkAdapter(agent_id="test-agent", owner_id="user@test.com", auth=auth)
    adapter._client = MagicMock()  # skip the token exchange; manager build is local
    mgr = adapter._manager_for("sess-1")
    assert mgr._agent_participant_variable["AiAgentType"] == "GoogleADK"
    assert mgr._agent_participant_variable["AiAgentApiName"] == "test-agent"


def test_agent_participant_variable_full_override_wins():
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy",
        consumer_key="dummy",
        subject="dummy@example.com",
        login_url="login.example.com",
    )
    custom = {"AiAgentType": "MyRuntime", "AiAgentApiName": "override"}
    adapter = GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com", auth=auth,
        agent_participant_variable=custom,
    )
    adapter._client = MagicMock()
    assert adapter._manager_for("sess-1")._agent_participant_variable == custom


def test_agent_participant_variable_rejected_with_injected_manager():
    with pytest.raises(ValueError, match="agent_participant_variable"):
        GoogleAdkAdapter(
            agent_id="test-agent", owner_id="user@test.com",
            session_manager=AsyncMock(), agent_participant_variable={},
        )


def test_set_owner_patches_buffered_records():
    # Simulate a turn where the user sends their email. before_model buffers it
    # under placeholder; set_owner fires mid-turn (before after_model
    # flushes) and should patch the buffer so the flush uses the real ID.
    adapter, _ = make_adapter()
    adapter.owner_id = "placeholder"  # routes to _default_owner; _owner_participant_id derives from it

    ctx = make_ctx("s-patch")
    adapter._on_before_model(ctx, make_llm_request("alice@example.com"))

    # Buffer has placeholder — not yet flushed.
    pre_ids = {r.participant_id for r in adapter._buffers.get("s-patch", [])}
    assert "placeholder" in pre_ids

    # set_owner fires mid-turn (e.g. LLM calls start_session tool).
    adapter.set_owner("alice@example.com")

    # Buffer should now have the real participant ID before after_model flushes.
    post_ids = {r.participant_id for r in adapter._buffers.get("s-patch", [])}
    assert "placeholder" not in post_ids
    assert "alice@example.com" in post_ids


# ── agent_reasoning vs agent_response ────────────────────────────────────────

def test_tool_call_model_output_is_buffered_agent_reasoning_trace():
    """A model turn that calls a tool emits agent_reasoning (not agent_response),
    as a Trace with the text in ``variable``, and is NOT flushed mid-turn."""
    adapter, _ = make_adapter()
    ctx = make_ctx("s5")
    adapter._on_before_model(ctx, make_llm_request("q"))
    adapter._on_after_model(ctx, make_llm_response("reasoning text", is_tool_call=True))

    assert "s5" in adapter._buffers  # not flushed mid-turn
    buf = adapter._buffers["s5"]
    assert not any(r.event_type == "agent_response" for r in buf)
    reasoning = next(r for r in buf if r.event_type == "agent_reasoning")
    assert reasoning.content is None
    assert reasoning.variable == {"OutputValueText": "reasoning text"}
    assert reasoning.category == "Trace"


# ── tool_call and tool_result records ────────────────────────────────────────

def test_tool_call_and_result_buffered_as_traces():
    adapter, _ = make_adapter()
    ctx = make_ctx("s7")
    tool_ctx = make_tool_ctx("s7")
    adapter._on_before_model(ctx, make_llm_request("do x"))
    adapter._on_before_tool(make_tool("lookup"), {"id": "123"}, tool_ctx)

    calls = [r for r in adapter._buffers["s7"] if r.event_type == "tool_call"]
    assert len(calls) == 1
    assert calls[0].variable == {"Name": "lookup", "InputValueText": '{"id": "123"}'}
    assert calls[0].category == "Trace"
    assert calls[0].content is None

    adapter._on_after_tool(make_tool("lookup"), {}, tool_ctx, "found it")

    results = [r for r in adapter._buffers["s7"] if r.event_type == "tool_result"]
    assert len(results) == 1
    assert results[0].variable == {"Name": "lookup", "OutputValueText": "found it"}
    assert results[0].category == "Trace"
    assert results[0].content is None
    # The tool name matches on the call and its result.
    assert calls[0].variable["Name"] == results[0].variable["Name"] == "lookup"


# ── turn flush ────────────────────────────────────────────────────────────────

async def test_buffer_cleared_after_turn_end():
    adapter, _ = make_adapter()
    ctx = make_ctx("s10")
    adapter._on_before_model(ctx, make_llm_request("q"))
    await adapter._on_after_model_async(ctx, make_llm_response("done"))

    assert "s10" not in adapter._buffers
    assert "s10" not in adapter._interaction_ids


# ── full turn record sequence ─────────────────────────────────────────────────

async def test_full_turn_with_tool_produces_five_records():
    adapter, _ = make_adapter()
    ctx = make_ctx("s12")
    tool_ctx = make_tool_ctx("s12")

    adapter._on_before_model(ctx, make_llm_request("Create ticket"))
    await adapter._on_after_model_async(ctx, make_llm_response("I'll call the tool", is_tool_call=True))
    adapter._on_before_tool(make_tool("create_ticket"), {"title": "Bug"}, tool_ctx)
    adapter._on_after_tool(make_tool("create_ticket"), {}, tool_ctx, "ticket-42")

    buf = list(adapter._buffers["s12"])
    types = [r.event_type for r in buf]
    assert types == ["user_message", "agent_reasoning", "tool_call", "tool_result"]

    await adapter._on_after_model_async(ctx, make_llm_response("Done, ticket created"))
    assert "s12" not in adapter._buffers


# ── existing callback composition ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_existing_before_model_callback_called():
    adapter, _ = make_adapter()
    called = []
    existing = lambda ctx, req: called.append(True)

    cb = adapter.wrap_before_model(existing)
    await cb(make_ctx("s13"), make_llm_request("hi"))

    assert called == [True]


# ── multi-session lifecycle ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_conversation_creates_fresh_per_session():
    """Each ADK session gets conversation_id = session_id (matches LangGraph/Claude convention)."""
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()

    adapter = GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com",
        session_manager=mgr,
    )

    await adapter._ensure_conversation("new-sess-1")

    mgr.create_session.assert_awaited_once_with("new-sess-1")


@pytest.mark.asyncio
async def test_wrap_before_model_only_calls_ensure_conversation_once():
    """The session-init guard in wrap_before_model fires only on first call for a session."""
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(return_value=None)

    adapter = GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com",
        session_manager=mgr,
    )

    async def noop(ctx, req): return None  # noqa: RUF029
    cb = adapter.wrap_before_model(existing=noop)
    ctx = make_ctx("sess-once")
    await cb(ctx, make_llm_request("hi"))
    await cb(ctx, make_llm_request("hi again"))

    # create_session should only be called once despite two callback invocations
    mgr.create_session.assert_awaited_once()


# ── retrieval wiring ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "with_retrieval", [True, False], ids=["with-retrieval-client", "no-retrieval-client"],
)
async def test_get_context_delegates_to_injected_session_manager(with_retrieval):
    """get_context() on the adapter delegates to the injected AsyncSessionManager.

    No conversation_id set yet — get_context returns None (best-effort), with or
    without a retrieval client wired on the manager. No live HTTP.
    """
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_response({}))
    http.aclose = AsyncMock()

    client = AsyncIngestionClient(
        base_url="https://beacon.test",
        api_key="test-key",
        client=http,
    )
    kwargs = {}
    if with_retrieval:
        kwargs["retrieval_client"] = AsyncRetrievalClient(
            base_url="https://beacon.test",
            api_key="test-key",
            client=AsyncMock(),
        )
    mgr = AsyncSessionManager(
        client, agent_id="test-agent", owner_id="user@test.com", **kwargs,
    )
    adapter = GoogleAdkAdapter(
        agent_id="test-agent",
        owner_id="user@test.com",
        session_manager=mgr,
    )

    result = await adapter.get_context()
    assert result is None


# ── make_start_session_tool ───────────────────────────────────────────────────
# google-adk uses _UnionGenericAlias which is deprecated in Python 3.14 and
# raises DeprecationWarning on import. Suppress it in tests that must import
# the ADK package — the warning is a google-adk/Python 3.14 compatibility issue,
# not something in our code.
import warnings as _warnings
_ADK_IMPORT_WARNING = ("ignore", ".*_UnionGenericAlias.*", DeprecationWarning)
_ADK_BASE_AGENT_CONFIG_WARNING = ("ignore", ".*BaseAgentConfig.*", DeprecationWarning)


def _make_tool_suppressed(adapter):
    with _warnings.catch_warnings():
        _warnings.filterwarnings(*_ADK_IMPORT_WARNING)
        _warnings.filterwarnings(*_ADK_BASE_AGENT_CONFIG_WARNING)
        return adapter.make_start_session_tool()


def test_make_start_session_tool_sets_owner_and_returns_success():
    adapter, _ = make_adapter()
    tool = _make_tool_suppressed(adapter)
    # Duck-type check — avoid re-importing FunctionTool to keep the warning isolated
    assert hasattr(tool, "func") and callable(tool.func)
    assert tool.__class__.__name__ == "FunctionTool"

    result = tool.func("alice@example.com")
    assert adapter.owner_id == "alice@example.com"
    assert adapter._owner_participant_id == "alice@example.com"
    assert result == {"status": "success", "customerEmail": "alice@example.com"}


def test_make_start_session_tool_docstring_mentions_own_email():
    """LLM uses the docstring to decide when to call the tool — it must distinguish
    'user's own email' from 'email mentioned in context'."""
    adapter, _ = make_adapter()
    tool = _make_tool_suppressed(adapter)
    doc = tool.func.__doc__ or ""
    assert "own" in doc.lower()
    assert "not" in doc.lower()  # must warn against false-positive cases


def test_make_start_session_tool_independent_per_adapter():
    """Two adapters each get their own closure — calling one does not affect the other."""
    adapter_a, _ = make_adapter()
    adapter_b, _ = make_adapter()
    tool_a = _make_tool_suppressed(adapter_a)
    tool_b = _make_tool_suppressed(adapter_b)
    tool_a.func("alice@example.com")
    tool_b.func("bob@example.com")
    assert adapter_a.owner_id == "alice@example.com"
    assert adapter_b.owner_id == "bob@example.com"


# ── make_context_retrieval_tool + default_tools ─────────────────────────────────────
# On-demand retrieval: the agent calls the get_context tool when it decides it
# needs cross-session memory, instead of the adapter injecting it every turn.

def _make_context_retrieval_tool_suppressed(adapter):
    with _warnings.catch_warnings():
        _warnings.filterwarnings(*_ADK_IMPORT_WARNING)
        _warnings.filterwarnings(*_ADK_BASE_AGENT_CONFIG_WARNING)
        return adapter.make_context_retrieval_tool()


def _make_default_tools_suppressed(adapter):
    with _warnings.catch_warnings():
        _warnings.filterwarnings(*_ADK_IMPORT_WARNING)
        _warnings.filterwarnings(*_ADK_BASE_AGENT_CONFIG_WARNING)
        return adapter.default_tools()


def _adapter_with_mock_mgr(get_context_return=None) -> tuple[GoogleAdkAdapter, AsyncMock]:
    """Adapter backed by an AsyncMock manager, for tool + injection tests.

    Pinned to context_mode="manual" so wrap_before_model injects nothing by
    default — these tests exercise the recall tool / the _should_inject_*
    helpers, and want a clean "adapter adds nothing" baseline (auto would seed
    context at open). Tests that need auto behaviour use _auto_adapter /
    _mode_adapter instead.
    """
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(return_value=get_context_return)
    adapter = GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com", session_manager=mgr,
        context_mode="manual",
    )
    return adapter, mgr


def make_llm_request_with_config(system_instruction: str = "") -> MagicMock:
    """LLM request whose config carries a real (mutable) system_instruction string."""
    req = make_llm_request("hi")
    req.config = SimpleNamespace(system_instruction=system_instruction)
    return req


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{}, {"topic": "billing history"}], ids=["no-topic", "with-topic"],
)
async def test_get_context_tool_delegates_to_adapter_get_context(kwargs):
    """The tool calls adapter.get_context() for an owner-scoped memory read.

    topic is an optional LLM-facing hint — the call works with or without it and
    is never forwarded to get_context.
    """
    adapter, _ = _adapter_with_mock_mgr()
    adapter.get_context = AsyncMock(
        return_value={"userContext": [{"entry": "PROFILE_FACT[1]\n  plan=Pro"}]},
    )
    tool = _make_context_retrieval_tool_suppressed(adapter)
    assert hasattr(tool, "func") and callable(tool.func)
    assert tool.__class__.__name__ == "FunctionTool"
    result = await tool.func(**kwargs)
    adapter.get_context.assert_awaited_once_with()
    assert result == {"userContext": [{"entry": "PROFILE_FACT[1]\n  plan=Pro"}]}


@pytest.mark.asyncio
async def test_get_context_tool_returns_empty_list_when_no_memory():
    """None from Beacon → {"userContext": []} so the LLM sees a clean 'nothing stored'."""
    adapter, _ = _adapter_with_mock_mgr()
    adapter.get_context = AsyncMock(return_value=None)
    tool = _make_context_retrieval_tool_suppressed(adapter)
    result = await tool.func()
    assert result == {"userContext": []}


def test_get_context_tool_docstring_guides_when_to_call():
    """The LLM decides when to recall from the docstring — it must mention memory/past."""
    adapter, _ = make_adapter()
    tool = _make_context_retrieval_tool_suppressed(adapter)
    doc = (tool.func.__doc__ or "").lower()
    assert "past" in doc or "remember" in doc
    assert "customer" in doc


def test_default_tools_includes_get_context_tool():
    """default_tools() exposes both the recall tool (owner-scoped profile read)
    and the pack-call tool (POST /context for data not in the recall)."""
    adapter, _ = make_adapter()
    tools = _make_default_tools_suppressed(adapter)
    assert isinstance(tools, list) and len(tools) == 2
    assert all(t.__class__.__name__ == "FunctionTool" for t in tools)
    names = {t.func.__name__ for t in tools}
    assert names == {"recall_customer_context", "get_context_pack"}


@pytest.mark.asyncio
async def test_get_context_tool_audit_logs_the_read(caplog):
    """The on-demand read must be audited like the per-turn injection path.

    Parity with wrap_before_model's "[ADK] get_context" log: the tool logs the
    owner it read for and whether context came back, so the on-demand path
    isn't a silent data-access hole relative to the always-inject path. The
    audit line logs a has_context bool only — never the retrieved memory text.
    """
    import logging
    secret = "PROFILE_FACT[1]\n  card_last_four=4242 plan=Enterprise"
    adapter, _ = _adapter_with_mock_mgr()
    adapter.get_context = AsyncMock(return_value={"userContext": [{"entry": secret}]})
    tool = _make_context_retrieval_tool_suppressed(adapter)
    with caplog.at_level(logging.INFO, logger="agentic_context_adk.adapter"):
        await tool.func(topic="billing history")
    audit = [r for r in caplog.records if "recall_customer_context" in r.getMessage()]
    assert len(audit) == 1, "expected exactly one audit log for the on-demand read"
    msg = audit[0].getMessage()
    assert "user@test.com" in msg          # owner attribution
    assert "has_context=True" in msg       # outcome recorded
    assert "4242" not in msg
    assert "PROFILE_FACT" not in msg


def test_get_context_tool_signature_only_takes_topic():
    """The recall tool exposes a single optional ``topic`` hint — no deep knobs."""
    import inspect
    adapter, _ = make_adapter()
    tool = _make_context_retrieval_tool_suppressed(adapter)
    params = inspect.signature(tool.func).parameters
    assert set(params) == {"topic"}
    assert params["topic"].default == ""


# ── auto-mode fast-once-at-open cadence (P1) ──────────────────────────────────
# context_mode="auto" seeds fast context ONLY at conversation-open (first turn of
# the session); after that the agent drives later reads itself via the recall
# tool. The _should_inject_fast_context helper is the single source of truth for
# cadence.

def _auto_adapter(get_context_return=None, **kwargs) -> tuple[GoogleAdkAdapter, AsyncMock]:
    """Adapter in context_mode="auto" with a fast get_context return.

    Exercises the fast-once-at-open cadence — the behaviour these tests assert.
    """
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(
        return_value=get_context_return or {"userContext": [{"entry": "plan=Pro"}]},
    )
    adapter = GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com",
        session_manager=mgr, context_mode="auto", **kwargs,
    )
    return adapter, mgr


# --- pure cadence-decision truth table (the _should_inject_fast_context helper) ---

@pytest.mark.parametrize(
    "make, at_open",
    [(_adapter_with_mock_mgr, False), (_auto_adapter, True)],
    ids=["manual-never", "auto-open-only"],
)
def test_should_inject_fast_context_cadence(make, at_open):
    """manual → never inject; auto → inject at open only (the agent drives later
    reads via the tool). Neither mode injects after open."""
    adapter, _ = make()
    assert adapter._should_inject_fast_context(is_conversation_open=True) is at_open
    assert adapter._should_inject_fast_context(is_conversation_open=False) is False


# --- end-to-end wrap_before_model behavior ---


@pytest.mark.asyncio
async def test_auto_injects_at_open_then_stops():
    """auto seeds fast context on the FIRST turn of a session, then stops
    injecting — the agent drives later reads."""
    adapter, mgr = _auto_adapter()
    cb = adapter.wrap_before_model()

    # Turn 1 (open): injects.
    req1 = make_llm_request_with_config("BASE PROMPT")
    await cb(make_ctx("s-open2"), req1)
    mgr.get_context.assert_awaited()                 # fast seeded at open
    assert "Cross-session context from Data Cloud" in req1.config.system_instruction
    assert "plan=Pro" in req1.config.system_instruction
    assert "BASE PROMPT" in req1.config.system_instruction
    mgr.get_context.reset_mock()

    # Turn 2 (same session, now open): no fast fetch/injection.
    req2 = make_llm_request_with_config("BASE PROMPT")
    await cb(make_ctx("s-open2"), req2)
    mgr.get_context.assert_not_awaited()
    assert "Cross-session context from Data Cloud" not in req2.config.system_instruction
    assert "plan=Pro" not in req2.config.system_instruction
    assert "BASE PROMPT" in req2.config.system_instruction


@pytest.mark.asyncio
async def test_auto_re_seeds_on_owner_switch_mid_session():
    """set_owner re-arms conversation-open, so a re-identified session re-seeds.

    Auto-detect fires set_owner when session.user_id resolves to a real owner;
    that discards the session from _sessions_initialised, which correctly makes
    the next turn look like a fresh conversation-open for the new owner.
    """
    adapter, mgr = _auto_adapter()
    cb = adapter.wrap_before_model()

    # Turn 1: opens under the construction-time owner.
    req1 = make_llm_request_with_config("BASE")
    await cb(make_ctx("s-reid"), req1)
    assert "plan=Pro" in req1.config.system_instruction
    mgr.get_context.reset_mock()

    # Turn 2: the real user identifies → set_owner re-arms open → re-seed.
    ctx2 = SimpleNamespace(session=SimpleNamespace(id="s-reid", user_id="real@user.com"))
    req2 = make_llm_request_with_config("BASE")
    await cb(ctx2, req2)
    mgr.get_context.assert_awaited()
    assert "plan=Pro" in req2.config.system_instruction


@pytest.mark.asyncio
async def test_manual_mode_does_not_inject_at_open():
    """manual → the adapter injects nothing, at open or after."""
    adapter, mgr = _adapter_with_mock_mgr()  # pinned manual
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("BASE PROMPT")
    await cb(make_ctx("s-noinstr"), req)
    assert req.config.system_instruction == "BASE PROMPT\n\n[Available context packs]\n{}"
    mgr.get_context.assert_not_awaited()


# ── multi-user owner race (contextvars + per-session managers) ────────────────
# One adapter instance is shared across all users; ADK interleaves turns as
# separate asyncio tasks on one loop. Owner identity is scoped to the turn's
# task via the _current_session_id ContextVar, and each session gets its own
# AsyncSessionManager (managers share the HTTP clients but not the mutable
# owner/session state), so concurrent turns for different users can't stomp
# each other's identity. These tests exercise the production FACTORY path
# (distinct managers per session), not the injected single-manager mode.
import asyncio as _asyncio

from agentic_context_adk.adapter import _current_session_id


def _factory_adapter() -> GoogleAdkAdapter:
    """Adapter on the auth/factory path: builds a DISTINCT AsyncSessionManager per
    session, sharing mock HTTP clients. This is the real production topology —
    unlike make_adapter()/_adapter_with_mock_mgr() which inject one shared manager.
    """
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy", consumer_key="dummy",
        subject="dummy@example.com", login_url="login.example.com",
    )
    adapter = GoogleAdkAdapter(agent_id="test-agent", owner_id="placeholder", auth=auth)

    # Inject mock HTTP clients and drop the auth so _ensure_clients is a pure
    # no-op (guard: _auth is None) — the factory path in _manager_for then builds
    # real per-session AsyncSessionManager instances over the mocks, no network.
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_response({}))
    http.aclose = AsyncMock()
    adapter._client = AsyncIngestionClient(base_url="https://beacon.test", api_key="k", client=http)
    adapter._retrieval = AsyncRetrievalClient(base_url="https://beacon.test", api_key="k", client=AsyncMock())
    # The bare AsyncMock transport can't build a real response; treat the
    # server prompt as unpublished so the bundled instruction is used.
    adapter._retrieval.get_system_prompt = AsyncMock(return_value=None)
    adapter._auth = None
    return adapter


def _ctx_with_user(session_id: str, user_id: str) -> SimpleNamespace:
    return SimpleNamespace(session=SimpleNamespace(id=session_id, user_id=user_id))


@pytest.mark.asyncio
async def test_interleaved_turns_do_not_cross_attribute_owner():
    """The core race regression: two users' turns interleave on one adapter, and
    each turn must bind to its OWN owner — never the other's.

    A 2-party barrier forces both turns to pause AFTER both have called set_owner,
    which is the worst case: with the old single-shared-owner design, whichever
    turn wrote last would win and BOTH turns would read that owner. With the fix,
    each turn resolves its owner from its own task-local ContextVar + the
    per-session owner map, so no cross-attribution occurs.
    """
    adapter = _factory_adapter()
    barrier = _asyncio.Barrier(2)
    captured: dict[str, dict] = {}

    def make_existing(session_id: str):
        # Runs inside wrap_before_model AFTER set_owner + _ensure_conversation.
        async def existing(ctx, req):
            await barrier.wait()  # both turns are now past their set_owner
            captured[session_id] = {
                "owner_id": adapter.owner_id,
                "participant": adapter._owner_participant_id,
                "mgr_owner": adapter._manager_for(session_id)._owner_id,
            }
        return existing

    async def run_turn(session_id: str, user_id: str):
        cb = adapter.wrap_before_model(existing=make_existing(session_id))
        await cb(_ctx_with_user(session_id, user_id), make_llm_request("hi"))

    await _asyncio.gather(
        run_turn("sess-alice", "alice@example.com"),
        run_turn("sess-bob", "bob@example.com"),
    )

    # Each turn saw its own owner — no stomping.
    assert captured["sess-alice"]["owner_id"] == "alice@example.com"
    assert captured["sess-bob"]["owner_id"] == "bob@example.com"
    # Participant ids derive from the per-turn owner.
    assert captured["sess-alice"]["participant"] == "alice@example.com"
    assert captured["sess-bob"]["participant"] == "bob@example.com"
    # The per-session managers each carry their own owner.
    assert captured["sess-alice"]["mgr_owner"] == "alice@example.com"
    assert captured["sess-bob"]["mgr_owner"] == "bob@example.com"


@pytest.mark.asyncio
async def test_per_session_managers_are_distinct_instances():
    """Two sessions must get different AsyncSessionManager objects with their own
    owners — proving the fix isn't a single shared manager being re-pointed."""
    adapter = _factory_adapter()
    adapter._session_owners["sess-a"] = "alice@example.com"
    adapter._session_owners["sess-b"] = "bob@example.com"

    mgr_a = adapter._manager_for("sess-a")
    mgr_b = adapter._manager_for("sess-b")

    assert mgr_a is not mgr_b
    assert mgr_a._owner_id == "alice@example.com"
    assert mgr_b._owner_id == "bob@example.com"
    # Resolving the same session again returns the SAME instance (cached).
    assert adapter._manager_for("sess-a") is mgr_a


def test_owner_id_property_falls_back_to_default_without_turn_context():
    """Outside any turn (no ContextVar set), owner_id reads the default owner."""
    adapter = _factory_adapter()
    assert _current_session_id.get() is None
    assert adapter.owner_id == "placeholder"
    adapter.set_owner("carol@example.com")  # no turn ctx → updates default
    assert adapter.owner_id == "carol@example.com"


@pytest.mark.asyncio
async def test_set_owner_inside_turn_binds_only_current_session():
    """set_owner called inside a turn (ContextVar set) rebinds ONLY that session,
    writing to the instance-level owner map (not a ContextVar) so the write
    survives the tool's throwaway child-task context and does not disturb the
    default owner or other sessions."""
    adapter = _factory_adapter()

    async def turn():
        _current_session_id.set("sess-x")
        adapter.set_owner("dana@company.com")

    await turn()

    # The session was bound in the shared instance map; default owner untouched.
    assert adapter._session_owners["sess-x"] == "dana@company.com"
    assert adapter._default_owner == "placeholder"
    # And a brand-new, unrelated session still resolves to the default.
    assert adapter._session_owners.get("sess-y") is None


@pytest.mark.asyncio
async def test_set_owner_does_not_patch_another_sessions_buffer():
    """set_owner's buffer patch must be scoped to the CURRENT turn's session.

    Regression for the cross-attribution the fix targets: both Alice and Bob
    buffer a message under placeholder (identity not yet known), then
    Bob's turn calls set_owner. Bob's buffered record must be rewritten to Bob;
    Alice's must be left untouched — a global patch would steal Alice's message
    into Bob's owner on flush."""
    adapter = _factory_adapter()

    # Two concurrent sessions each buffer a user message before identity is known.
    async def buffer_turn(session_id: str):
        _current_session_id.set(session_id)
        adapter._on_before_model(make_ctx(session_id), make_llm_request("hi there"))

    await buffer_turn("sess-alice")
    await buffer_turn("sess-bob")
    assert "placeholder" in {r.participant_id for r in adapter._buffers["sess-alice"]}
    assert "placeholder" in {r.participant_id for r in adapter._buffers["sess-bob"]}

    # Bob's turn establishes identity mid-turn (e.g. via the start_session tool).
    async def bob_sets_owner():
        _current_session_id.set("sess-bob")
        adapter.set_owner("bob@example.com")

    await bob_sets_owner()

    # Bob's buffer is corrected; Alice's is untouched (no cross-attribution).
    bob_ids = {r.participant_id for r in adapter._buffers["sess-bob"]}
    alice_ids = {r.participant_id for r in adapter._buffers["sess-alice"]}
    assert bob_ids == {"bob@example.com"}
    assert alice_ids == {"placeholder"}


@pytest.mark.asyncio
async def test_set_owner_write_propagates_from_tool_child_task():
    """Production relies on set_owner being called from an ADK tool that runs in a
    child asyncio task (a copy_context snapshot). The _current_session_id set in
    the parent turn must be visible in that child task, and the set_owner write
    (to the instance-level map, NOT a ContextVar) must survive back to the parent.
    Mirrors ADK's tool execution: set the var in the parent, then run set_owner in
    a child task via create_task (which snapshots the parent context)."""
    adapter = _factory_adapter()

    async def parent_turn():
        _current_session_id.set("sess-tool")
        # Child task inherits the parent's context snapshot (as ADK's tool runner
        # does). set_owner resolves sid from the inherited ContextVar and writes
        # the instance map, which is shared — so the binding survives task exit.
        await _asyncio.create_task(_child_tool_call(adapter))
        # Back in the parent: the child's write is visible, and owner_id resolves.
        assert adapter._session_owners["sess-tool"] == "erin@example.com"
        assert adapter.owner_id == "erin@example.com"

    await parent_turn()


async def _child_tool_call(adapter):
    # Simulates the body of an ADK-spawned tool (e.g. start_session).
    adapter.set_owner("erin@example.com")


def test_eviction_preserves_owner_binding_but_drops_manager():
    """LRU eviction must free the expensive manager but KEEP the cheap owner-id
    string — dropping it would silently re-default a paused user's identity to
    _default_owner and mis-attribute their next turn."""
    adapter = _factory_adapter()
    adapter._MGR_CAP = 1  # force eviction on the second session

    adapter._session_owners["sess-old"] = "old@example.com"
    mgr_old = adapter._manager_for("sess-old")
    assert mgr_old is not None

    adapter._session_owners["sess-new"] = "new@example.com"
    adapter._manager_for("sess-new")  # exceeds cap → evicts the idle sess-old mgr

    # Manager gone from the cache, but the owner binding survives.
    assert "sess-old" not in adapter._managers
    assert adapter._session_owners["sess-old"] == "old@example.com"
    # The owner resolves correctly for that session even after eviction.
    _current_session_id.set("sess-old")
    assert adapter.owner_id == "old@example.com"


# ── callback_kwargs ───────────────────────────────────────────────────────────

def test_callback_kwargs_returns_all_four_callables_without_existing():
    adapter, _ = make_adapter()
    kwargs = adapter.callback_kwargs()
    assert set(kwargs) == {
        "before_model_callback",
        "after_model_callback",
        "before_tool_callback",
        "after_tool_callback",
    }
    for cb in kwargs.values():
        assert callable(cb)
    # No existing callbacks to compose — before_tool_callback and
    # after_tool_callback are sync, so call them directly; they must not raise.
    kwargs["before_tool_callback"](make_tool("t"), {}, make_tool_ctx("s-kw2"))
    kwargs["after_tool_callback"](make_tool("t"), {}, make_tool_ctx("s-kw2"), "ok")


@pytest.mark.asyncio
async def test_callback_kwargs_composes_existing_before_model():
    adapter, _ = make_adapter()
    called = []
    kwargs = adapter.callback_kwargs(before_model=lambda ctx, req: called.append(True))
    await kwargs["before_model_callback"](make_ctx("s-kw"), make_llm_request("hi"))
    assert called == [True]


# ── register_shutdown ─────────────────────────────────────────────────────────

def test_register_shutdown_installs_signal_handlers():
    import signal as _signal
    adapter, _ = make_adapter()
    # Save originals so we can restore them after the test
    orig_int = _signal.getsignal(_signal.SIGINT)
    orig_term = _signal.getsignal(_signal.SIGTERM)
    try:
        adapter.register_shutdown()
        assert _signal.getsignal(_signal.SIGINT) not in (orig_int, _signal.SIG_DFL)
        assert _signal.getsignal(_signal.SIGTERM) not in (orig_term, _signal.SIG_DFL)
    finally:
        _signal.signal(_signal.SIGINT, orig_int)
        _signal.signal(_signal.SIGTERM, orig_term)


# ── add_participant ───────────────────────────────────────────────────────────

def _make_adapter_with_client() -> tuple[GoogleAdkAdapter, AsyncMock]:
    """Adapter with a live _client (mock HTTP) to test the add_participant HTTP path."""
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_response({}))
    http.aclose = AsyncMock()

    ingestion_client = AsyncIngestionClient(
        base_url="https://beacon.test",
        api_key="test-key",
        client=http,
    )
    mgr = AsyncSessionManager(ingestion_client, agent_id="test-agent", owner_id="user@test.com")
    adapter = GoogleAdkAdapter(
        agent_id="test-agent",
        owner_id="user@test.com",
        session_manager=mgr,
        retrieval_client=AsyncMock(),
    )
    # Manually set _client so add_participant takes the live HTTP path (not test-mode no-op)
    adapter._client = ingestion_client  # type: ignore[attr-defined]
    return adapter, http


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "participant_id, participant_type",
    [("billing-agent-participant", "agent"), ("dana@company.com", "human")],
    ids=["agent", "human"],
)
async def test_add_participant_calls_beacon(participant_id, participant_type):
    adapter, http = _make_adapter_with_client()
    # AsyncSessionManager unifies conversation + session under _session_id, so
    # it serves as both ids. add_participant resolves the current session's
    # manager (the injected one here, since no turn context is active).
    adapter._injected_mgr._session_id = "sess-123"  # type: ignore[attr-defined]

    await adapter.add_participant(participant_id, participant_type=participant_type)

    http.post.assert_called_once()
    import json as _json
    payload = _json.dumps(http.post.call_args, default=str)
    assert "sess-123" in payload  # used as both conversationId and sessionId
    assert participant_id in payload
    assert f'"{participant_type}"' in payload


@pytest.mark.asyncio
async def test_add_participant_no_op_before_first_turn():
    """Should not raise when conversation not yet initialised — just warns."""
    adapter, http = _make_adapter_with_client()

    await adapter.add_participant("billing-agent-participant")  # must not raise

    http.post.assert_not_called()


@pytest.mark.asyncio
async def test_add_participant_no_op_in_test_mode():
    """session_manager injection sets _client=None — should not raise."""
    adapter, _ = make_adapter()
    adapter._injected_mgr._session_id = "sess-123"  # type: ignore[attr-defined]

    await adapter.add_participant("billing-agent-participant")  # must not raise


# ── W2 — per-conversation config (adapter.hooks / adapter.apply) ───────────────
#
# W2a: owner_id optional at construction (per-conversation identity, not per-
# adapter). W2b: hooks()/apply() config surface bound at conversation-open.
# W2c: per-conversation system_prompt_override + per-conversation flag overrides.
# All three ride the race fix's per-session isolation (contextvar + owner map);
# these tests assert the new surface without disturbing the shipped defaults.


def _auth_adapter(**kwargs) -> GoogleAdkAdapter:
    """Auth/factory-path adapter with mock HTTP clients — no owner_id by default.

    Mirrors _factory_adapter but leaves owner_id unset (the modern per-
    conversation path) unless a test passes it, and forwards W2 kwargs. Builds
    real per-session AsyncSessionManagers over mock HTTP, so _manager_for's
    owner-None gate and the hook→set_owner wiring are exercised for real.
    """
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy", consumer_key="dummy",
        subject="dummy@example.com", login_url="login.example.com",
    )
    adapter = GoogleAdkAdapter(agent_id="test-agent", auth=auth, **kwargs)
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_response({}))
    http.aclose = AsyncMock()
    adapter._auth = None  # make _ensure_clients a no-op
    adapter._client = AsyncIngestionClient(base_url="https://beacon.test", api_key="k", client=http)
    adapter._retrieval = None
    return adapter


# ── W2a — owner_id optional / None ─────────────────────────────────────────────

def test_owner_id_optional_at_construction():
    """Constructing without owner_id leaves the owner unresolved (was 'placeholder');
    the old owner_id= call site keeps working silently (Option 1 — doc-only depr)."""
    import warnings

    adapter = _auth_adapter()
    assert adapter._default_owner is None
    assert adapter.owner_id is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning → failure, like the suite config
        adapter = _auth_adapter(owner_id="dana@corp.com")
    assert adapter.owner_id == "dana@corp.com"


def test_manager_for_returns_none_when_owner_unresolved():
    """The single upstream gate: no owner → no manager → downstream Beacon no-ops."""
    adapter = _auth_adapter()  # owner_id None, real client present
    assert adapter._manager_for("sess-anon") is None
    assert "sess-anon" not in adapter._managers  # nothing built/cached


@pytest.mark.asyncio
async def test_get_context_no_ops_when_owner_unresolved():
    """get_context returns None (not a crash, not a null-owner Beacon call)."""
    adapter = _auth_adapter()

    async def turn():
        _current_session_id.set("sess-anon")
        return await adapter.get_context()

    assert await turn() is None


@pytest.mark.asyncio
async def test_ensure_conversation_no_ops_when_owner_unresolved():
    """Owner-less conversation-open must not build a manager or raise."""
    adapter = _auth_adapter()

    async def turn():
        _current_session_id.set("sess-anon")
        await adapter._ensure_conversation("sess-anon")

    await turn()  # must not raise
    assert "sess-anon" not in adapter._managers


# ── W2b — hooks() / on_conversation_start ──────────────────────────────────────


@pytest.mark.asyncio
async def test_hook_binds_owner_at_conversation_open():
    """on_conversation_start returning {owner_id} scopes the conversation from open."""
    adapter = _auth_adapter()
    seen = {}

    def on_start(session_id, ctx):
        seen["session_id"] = session_id
        seen["user_id"] = ctx.get("user_id")
        return {"owner_id": "resolved@corp.com"}

    assert adapter.hooks(on_conversation_start=on_start) is adapter  # chainable
    cb = adapter.wrap_before_model()
    ctx = SimpleNamespace(session=SimpleNamespace(id="s-hook", user_id="raw-123"))

    await cb(ctx, make_llm_request_with_config("BASE"))

    assert seen == {"session_id": "s-hook", "user_id": "raw-123"}
    assert adapter._session_owners["s-hook"] == "resolved@corp.com"


@pytest.mark.asyncio
async def test_hook_fires_once_per_conversation_not_per_turn():
    """The hook runs at conversation-open only — not on every subsequent turn."""
    adapter = _auth_adapter()
    calls = []
    adapter.hooks(on_conversation_start=lambda sid, ctx: calls.append(sid) or {"owner_id": "x@y.com"})
    cb = adapter.wrap_before_model()
    ctx = make_ctx("s-once")

    await cb(ctx, make_llm_request_with_config())
    await cb(ctx, make_llm_request_with_config())  # second turn, same session

    assert calls == ["s-once"]  # fired exactly once


@pytest.mark.asyncio
async def test_hook_exception_is_swallowed():
    """A raising hook must never break the turn (best-effort)."""
    adapter = _auth_adapter()
    adapter.hooks(on_conversation_start=lambda sid, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    cb = adapter.wrap_before_model()

    await cb(make_ctx("s-boom"), make_llm_request_with_config())  # must not raise


@pytest.mark.asyncio
async def test_hook_non_dict_return_is_ignored():
    """A hook returning a non-dict is ignored, not applied."""
    adapter = _auth_adapter()
    adapter.hooks(on_conversation_start=lambda sid, ctx: "not-a-dict")
    cb = adapter.wrap_before_model()

    await cb(make_ctx("s-bad"), make_llm_request_with_config())  # must not raise
    assert adapter._session_owners.get("s-bad") is None


# ── W2b — apply() imperative push ──────────────────────────────────────────────


def test_apply_pre_seeds_owner_before_first_turn():
    """apply() before any turn binds the owner map + arms the lifecycle."""
    adapter = _auth_adapter()
    assert adapter.apply("s-preseed", owner_id="pre@corp.com") is adapter  # chainable
    assert adapter._session_owners["s-preseed"] == "pre@corp.com"
    # And a manager built for that session now carries the owner.
    mgr = adapter._manager_for("s-preseed")
    assert mgr is not None and mgr._owner_id == "pre@corp.com"


@pytest.mark.asyncio
async def test_apply_seeded_owner_used_at_conversation_open():
    """A pre-seeded owner scopes the conversation even with no hook registered."""
    adapter = _auth_adapter()
    adapter.apply("s-seed", owner_id="seed@corp.com")
    cb = adapter.wrap_before_model()

    await cb(make_ctx("s-seed"), make_llm_request_with_config())

    assert adapter._session_owners["s-seed"] == "seed@corp.com"


def test_apply_stores_unknown_overrides_verbatim():
    """Forward-compat: unrecognised keys are stored, not dropped."""
    adapter = _auth_adapter()
    adapter.apply("s-fwd", future_knob="xyz")
    assert adapter._session_configs["s-fwd"]["future_knob"] == "xyz"


# ── W2c — per-conversation flag overrides ──────────────────────────────────────

def test_config_flag_falls_back_to_instance_default_outside_turn():
    adapter = _auth_adapter(context_mode="auto")  # auto → inject_at_open=True
    assert adapter._config_flag("inject_at_open", adapter._inject_at_open) is True


@pytest.mark.asyncio
async def test_per_conversation_inject_at_open_override():
    """A conversation can flip inject_at_open independent of the instance flag."""
    adapter = _auth_adapter(context_mode="manual")  # instance default: inject_at_open OFF
    adapter.apply("s-auto", owner_id="u@c.com", inject_at_open=True)

    async def turn():
        _current_session_id.set("s-auto")
        # inject_at_open on for THIS session → fast injects only at open
        assert adapter._should_inject_fast_context(is_conversation_open=True) is True
        assert adapter._should_inject_fast_context(is_conversation_open=False) is False

    await turn()
    # A different session with no override still follows the instance default (OFF → never).
    async def other():
        _current_session_id.set("s-other")
        assert adapter._should_inject_fast_context(is_conversation_open=True) is False

    await other()


# ── W2c — system_prompt_override ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_prompt_override_replaces_base_instruction():
    """The override becomes the base system instruction for the conversation."""
    adapter = _auth_adapter()
    adapter.apply("s-persona", owner_id="u@c.com", system_prompt_override="You are Nova, a billing specialist.")
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("ORIGINAL AGENT PROMPT")

    await cb(make_ctx("s-persona"), req)

    assert "You are Nova, a billing specialist." in req.config.system_instruction
    assert "ORIGINAL AGENT PROMPT" not in req.config.system_instruction


@pytest.mark.asyncio
async def test_system_prompt_override_lets_memory_layer_on_top():
    """Memory blocks prepend ON TOP of the persona — override doesn't disable memory."""
    adapter = _auth_adapter(context_mode="auto")  # seeds fast context at open
    # Real per-session mgr; give it a get_context return via a mock manager instead.
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(return_value={"userContext": [{"entry": "plan=Pro"}]})
    adapter._injected_mgr = mgr
    adapter.apply("s-both", owner_id="u@c.com", system_prompt_override="PERSONA")
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("ORIGINAL")

    await cb(make_ctx("s-both"), req)

    si = req.config.system_instruction
    assert "PERSONA" in si                       # persona applied
    assert "plan=Pro" in si                       # memory layered on top
    assert si.index("plan=Pro") < si.index("PERSONA")  # memory prepended above persona


@pytest.mark.asyncio
async def test_system_prompt_override_idempotent_across_reentrant_before_model():
    """A re-entrant before_model (post tool result) must not clobber memory blocks."""
    adapter = _auth_adapter(context_mode="auto")
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(return_value={"userContext": [{"entry": "FACT"}]})
    adapter._injected_mgr = mgr
    adapter.apply("s-reentry", owner_id="u@c.com", system_prompt_override="PERSONA")
    cb = adapter.wrap_before_model()
    ctx = make_ctx("s-reentry")

    req1 = make_llm_request_with_config("ORIGINAL")
    await cb(ctx, req1)  # open turn: persona + memory
    first = req1.config.system_instruction

    # Simulate the re-entrant before_model after a tool result: same session, the
    # request already carries the prior instruction (persona + memory).
    req2 = make_llm_request_with_config(first)
    await cb(ctx, req2)

    # Persona applied exactly once — not stacked, memory not clobbered.
    assert req2.config.system_instruction.count("PERSONA") == 1
    assert "FACT" in req2.config.system_instruction


@pytest.mark.asyncio
async def test_no_override_leaves_instruction_untouched():
    """With no per-conversation override, the base instruction is unchanged.

    Pinned to context_mode="manual" so the ONLY variable under test is the
    system_prompt_override (W2c). Under the default "auto" mode the adapter would
    (correctly) also seed fast context at open; that path is covered by the
    context_mode tests, not this one.
    """
    adapter = _auth_adapter(context_mode="manual")
    adapter.apply("s-plain", owner_id="u@c.com")  # owner only, no override
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("AGENT PROMPT")

    await cb(make_ctx("s-plain"), req)

    assert req.config.system_instruction == f"AGENT PROMPT\n\n[Available context packs]\n{{}}"


# ── W3 — agent_api_key= constructor path (Context Builder demo wiring) ──────────
#
# The Context Builder front-end constructs the adapter with nothing but a
# per-agent key: GoogleAdkAdapter(agent_id="agent1", agent_api_key="<key>").
# These tests lock in that call shape against the shipped SDK: the kwarg must
# build an ApiKeyAuth internally, honour secret_server_url / its env fallback,
# yield to an explicit auth=, and do zero network I/O at construction (ApiKeyAuth
# fetches the bundle lazily on the first token request). A dummy key is used
# throughout — never a real Context Builder key literal.


def test_agent_api_key_matches_demo_call_shape():
    """The exact demo call — agent_id + agent_api_key, no owner_id — constructs clean,
    building an ApiKeyAuth internally (no wrapper needed)."""
    from agentic_context import ApiKeyAuth

    adapter = GoogleAdkAdapter(agent_id="agent1", agent_api_key="dummy-key")
    assert adapter.agent_id == "agent1"
    assert adapter.owner_id is None                 # per-conversation, not baked in
    assert isinstance(adapter._auth, ApiKeyAuth)
    assert adapter._auth._api_key == "dummy-key"


@pytest.mark.parametrize(
    "kwargs, expected",
    [({"secret_server_url": "https://sm.example.com"}, "https://sm.example.com"), ({}, None)],
    ids=["explicit-url", "omitted-url-env-fallback"],
)
def test_agent_api_key_secret_server_url(kwargs, expected):
    """secret_server_url= is threaded into the ApiKeyAuth (not dropped).

    Omitted → ApiKeyAuth normalises a falsy host to "" and reads
    AGENTIC_CONTEXT_SECRET_SERVER_URL lazily on the first token fetch; either
    way no host is baked in here (fail-closed if the env var is also unset).
    """
    adapter = GoogleAdkAdapter(agent_id="agent1", agent_api_key="dummy-key", **kwargs)
    if expected is None:
        assert not adapter._auth._secret_server_url
    else:
        assert adapter._auth._secret_server_url == expected


def test_explicit_auth_takes_precedence_over_agent_api_key():
    """auth= wins over agent_api_key= — an explicit instance is never overridden."""
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy", consumer_key="dummy",
        subject="dummy@example.com", login_url="login.example.com",
    )
    adapter = GoogleAdkAdapter(
        agent_id="agent1",
        auth=auth,
        agent_api_key="dummy-key",   # ignored — auth= present
    )
    assert adapter._auth is auth


def test_no_auth_and_no_key_still_raises():
    """The fail-closed contract holds: nothing to authenticate with → ValueError."""
    with pytest.raises(ValueError, match="requires auth"):
        GoogleAdkAdapter(agent_id="agent1")


def test_agent_api_key_construction_does_no_network(monkeypatch):
    """Construction is offline — ApiKeyAuth's bundle fetch is lazy, not at __init__.

    Guard against a future regression that eagerly fetches: poison httpx so any
    network attempt during construction would blow up loudly.
    """
    import httpx

    def _boom(*args, **kwargs):
        raise AssertionError("construction must not touch the network")

    monkeypatch.setattr(httpx, "Client", _boom)
    monkeypatch.setattr(httpx, "AsyncClient", _boom)

    adapter = GoogleAdkAdapter(agent_id="agent1", agent_api_key="dummy-key")  # must not raise
    from agentic_context import ApiKeyAuth
    assert isinstance(adapter._auth, ApiKeyAuth)


# ── context_mode facade (2026-07-16 two-mode decision) ──────────────────────────
#
# context_mode is the SOLE public context-behaviour selector. The mapping:
#
#   auto     → fast context seeded once at conversation-open, then the agent drives
#              later reads via the recall tool. Bare default.
#   manual   → the adapter injects NOTHING; the developer owns the whole prompt.
#
# context_mode defaults to "auto" and always resolves to one of the two slugs.
# The `session_manager=` seam is a pure auth-bypass for tests and no longer
# forces any legacy behaviour — helpers that need "inject nothing" pin
# context_mode="manual" explicitly.

from agentic_context_adk.adapter import _resolve_context_mode  # noqa: E402


def _mode_adapter(context_mode, **kwargs) -> GoogleAdkAdapter:
    """Auth-path adapter constructed via the context_mode facade (no session_manager).

    Uses the auth path (a real per-session manager over mock HTTP) so the whole
    wrap_before_model cadence runs for real under the selected mode.
    """
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy", consumer_key="dummy",
        subject="dummy@example.com", login_url="login.example.com",
    )
    adapter = GoogleAdkAdapter(agent_id="test-agent", auth=auth, context_mode=context_mode, **kwargs)
    http = AsyncMock()
    http.post = AsyncMock(return_value=_mock_response({}))
    http.aclose = AsyncMock()
    adapter._auth = None  # _ensure_clients no-op
    adapter._client = AsyncIngestionClient(base_url="https://beacon.test", api_key="k", client=http)
    adapter._retrieval = None
    return adapter


# --- pure resolver truth table (_resolve_context_mode) ---

@pytest.mark.parametrize(
    "slug, inject_at_open", [("auto", True), ("manual", False)], ids=["auto", "manual"],
)
def test_resolve_context_mode(slug, inject_at_open):
    """auto → inject_at_open=True; manual → False (the developer owns the prompt)."""
    assert _resolve_context_mode(slug) is inject_at_open


def _construct_with_mode(mode):
    from agentic_context import ContextAuth
    auth = ContextAuth(
        private_key="d", consumer_key="d", subject="d@e.com", login_url="login.e.com",
    )
    GoogleAdkAdapter(agent_id="a", auth=auth, context_mode=mode)


@pytest.mark.parametrize(
    "call",
    [
        lambda: _resolve_context_mode("aggressive"),
        # per-turn is no longer a valid ADK mode — it must raise like any bad slug.
        lambda: _resolve_context_mode("per-turn"),
        lambda: _construct_with_mode("turbo"),
        lambda: _mode_adapter("auto").set_context_mode("nope"),
    ],
    ids=["resolver", "resolver-per-turn-rejected", "constructor", "set_context_mode"],
)
def test_context_mode_bad_slug_raises(call):
    with pytest.raises(ValueError, match="must be one of"):
        call()


# --- construction resolution matrix (through the public surface) ---

def _injected_mgr_adapter():
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(return_value={"userContext": []})
    return GoogleAdkAdapter(
        agent_id="test-agent", owner_id="user@test.com", session_manager=mgr,
    )


@pytest.mark.parametrize(
    "make, mode, inject_at_open",
    [
        (lambda: _mode_adapter("auto"), "auto", True),
        (lambda: _mode_adapter("manual"), "manual", False),
        # No context_mode, no session_manager → the meeting default: auto.
        (_auth_adapter, "auto", True),
        # The session_manager= seam no longer overrides the mode — bare stays auto.
        # (Our shared helpers pin context_mode="manual" when they want "inject
        # nothing"; an injected manager on its own carries the normal default.)
        (_injected_mgr_adapter, "auto", True),
    ],
    ids=["explicit-auto", "explicit-manual", "bare-default", "injected-manager-default"],
)
def test_construction_resolves_context_mode(make, mode, inject_at_open):
    adapter = make()
    assert adapter.context_mode == mode
    assert adapter._inject_at_open is inject_at_open


# --- auto mode: fast at open only ---

@pytest.mark.asyncio
async def test_auto_mode_seeds_fast_at_open_only():
    """auto spec: fast context is auto-injected on turn 1, NOT on later turns.

    Turn 1 (conversation-open) seeds fast profile context. Turn 2+ do not
    auto-inject — the agent drives all subsequent reads via the tool.

    Also the only test of an owner-less adapter over an injected manager: with
    no resolved owner, _manager_for must not clobber the manager's own owner.
    """
    adapter = _mode_adapter("auto")
    mgr = AsyncMock()
    mgr.create_session = AsyncMock()
    mgr.get_context = AsyncMock(return_value={"userContext": [{"entry": "plan=Pro"}]})
    mgr._owner_id = "mgr-owner@test.com"
    adapter._injected_mgr = mgr
    cb = adapter.wrap_before_model()

    # Turn 1 (open): fast seeded.
    req1 = make_llm_request_with_config("BASE")
    await cb(make_ctx("s-auto-seed"), req1)
    assert "plan=Pro" in req1.config.system_instruction
    assert mgr._owner_id == "mgr-owner@test.com"
    mgr.get_context.reset_mock()

    # Turn 2 (already open): no auto fast injection — agent drives reads now.
    req2 = make_llm_request_with_config("BASE")
    await cb(make_ctx("s-auto-seed"), req2)
    mgr.get_context.assert_not_awaited()
    assert "plan=Pro" not in req2.config.system_instruction


# --- runtime set_context_mode ---

def test_set_context_mode_flips_flags_at_runtime():
    adapter = _mode_adapter("manual")
    assert adapter.context_mode == "manual"
    assert adapter._inject_at_open is False

    adapter.set_context_mode("auto")
    assert adapter.context_mode == "auto"
    assert adapter._inject_at_open is True

    adapter.set_context_mode("manual")
    assert adapter.context_mode == "manual"
    assert adapter._inject_at_open is False


def test_set_context_mode_logs_only_on_change(caplog):
    import logging
    adapter = _mode_adapter("auto")
    with caplog.at_level(logging.INFO, logger="agentic_context_adk.adapter"):
        adapter.set_context_mode("auto")  # no change
    assert not any("context_mode" in r.getMessage() for r in caplog.records)

    with caplog.at_level(logging.INFO, logger="agentic_context_adk.adapter"):
        adapter.set_context_mode("manual")
    assert any("context_mode" in r.getMessage() for r in caplog.records)


# --- per-conversation context_mode override (W2c) ---

@pytest.mark.parametrize(
    "extra, inject_at_open",
    [({}, True), ({"inject_at_open": False}, False)],
    ids=["facade-expands", "explicit-granular-key-wins"],
)
def test_per_conversation_context_mode_expands_to_flag(extra, inject_at_open):
    """apply(context_mode=...) expands the facade into the internal bool flag at merge
    time; a caller-supplied granular key in the SAME apply() wins over the expansion."""
    adapter = _auth_adapter(context_mode="manual")   # instance default: nothing on
    adapter.apply("s-pc", owner_id="u@c.com", context_mode="auto", **extra)
    assert adapter._session_configs["s-pc"]["inject_at_open"] is inject_at_open


@pytest.mark.asyncio
async def test_per_conversation_auto_overrides_manual_instance():
    """A single conversation can run auto injection even though the instance is manual."""
    adapter = _auth_adapter(context_mode="manual")   # instance never injects
    adapter.apply("s-pc-run", owner_id="u@c.com", context_mode="auto")

    async def turn():
        _current_session_id.set("s-pc-run")
        # auto override → fast injects at open for THIS session.
        assert adapter._should_inject_fast_context(is_conversation_open=True) is True

    await turn()


@pytest.mark.asyncio
async def test_per_conversation_override_does_not_leak_to_other_sessions():
    """The auto override is scoped to its session; others follow the manual instance."""
    adapter = _auth_adapter(context_mode="manual")
    adapter.apply("s-pc-scoped", owner_id="u@c.com", context_mode="auto")
    cb = adapter.wrap_before_model()

    # A different session with no override → manual → nothing injected.
    req = make_llm_request_with_config("BASE")
    await cb(make_ctx("s-other-manual"), req)

    assert req.config.system_instruction == f"BASE\n\n[Available context packs]\n{{}}"


# ── deterministic event_ids (stable across replay) ────────────────────────────

def _ctx_with_invocation(session_id: str, invocation_id: str) -> SimpleNamespace:
    """A CallbackContext carrying ADK's per-turn invocation_id.

    ADK stamps one invocation_id per turn and reuses it across the re-entrant
    before_model calls within that turn — the authoritative new-turn signal.
    """
    return SimpleNamespace(
        session=SimpleNamespace(id=session_id), invocation_id=invocation_id
    )


async def test_event_ids_are_deterministic_and_monotonic():
    """event_id = f"{prefix}-{session_id}-{seq}", monotonic per session.

    Regression: ids were minted with uuid4().hex[:8], so a re-read of the same
    source produced different ids.
    """
    adapter, _ = make_adapter()
    ctx = make_ctx("s-det")
    tool_ctx = make_tool_ctx("s-det")

    adapter._on_before_model(ctx, make_llm_request("Create ticket"))
    adapter._on_after_model(ctx, make_llm_response("thinking", is_tool_call=True))
    adapter._on_before_tool(make_tool("create_ticket"), {"title": "Bug"}, tool_ctx)
    adapter._on_after_tool(make_tool("create_ticket"), {}, tool_ctx, "ok")

    ids = [r.event_id for r in adapter._buffers["s-det"]]
    # user_message, agent_reasoning, tool_call, tool_result — in order.
    assert ids == [
        "msg-s-det-0",
        "step-s-det-1",
        "step-s-det-2",
        "step-s-det-3",
    ]


async def test_event_ids_reproduce_on_fresh_adapter_replay():
    """A fresh adapter replaying the same source reproduces the same ids.

    seq restarts at 0 for a new adapter, so the same event yields the same
    local id both times (event_id is not sent to Beacon).
    """
    def run_once() -> list[str]:
        adapter, _ = make_adapter()
        ctx = make_ctx("s-replay")
        adapter._on_before_model(ctx, make_llm_request("hi"))
        adapter._on_after_model(ctx, make_llm_response("thinking", is_tool_call=True))
        return [r.event_id for r in adapter._buffers["s-replay"]]

    assert run_once() == run_once()


def test_event_seq_is_per_session_independent():
    """Two sessions keep independent monotonic counters — ids never collide."""
    adapter, _ = make_adapter()
    adapter._on_before_model(make_ctx("s-a"), make_llm_request("a1"))
    adapter._on_before_model(make_ctx("s-b"), make_llm_request("b1"))

    assert adapter._buffers["s-a"][0].event_id == "msg-s-a-0"
    assert adapter._buffers["s-b"][0].event_id == "msg-s-b-0"


# ── invocation_id turn boundary + aborted-turn salvage ────────────────────────

def test_same_invocation_id_is_treated_as_reentry_not_new_turn():
    """ADK re-enters before_model after each tool result within a turn, reusing
    the invocation_id. That re-entry must NOT start a new turn / re-buffer."""
    adapter, _ = make_adapter()
    ctx = _ctx_with_invocation("s-reentry", "inv-1")

    adapter._on_before_model(ctx, make_llm_request("first"))
    interaction_1 = adapter._interaction_ids["s-reentry"]
    # Re-entry with the SAME invocation_id (post-tool) — no new user message.
    adapter._on_before_model(ctx, make_llm_request("first"))

    assert adapter._interaction_ids["s-reentry"] == interaction_1
    user_msgs = [r for r in adapter._buffers["s-reentry"] if r.event_type == "user_message"]
    assert len(user_msgs) == 1, "re-entry double-buffered the user message"


async def test_new_invocation_id_after_aborted_turn_salvages_prior_buffer():
    """A new turn (new invocation_id) after a turn that aborted before turn-end
    must flush the stale buffer, not append into it and cross-attribute records.

    Regression for #7: an aborted turn's records used to linger and get stamped
    with the NEXT turn's interaction_id.
    """
    adapter, _ = make_adapter()
    flushed: list[list[Record]] = []
    adapter._async_ingest = AsyncMock(side_effect=lambda sid, recs: flushed.append(list(recs)))

    # Turn 1 buffers a user message + reasoning, then "aborts" (never reaches
    # _on_after_model final → no turn-end).
    ctx1 = _ctx_with_invocation("s-abort", "inv-1")
    adapter._on_before_model(ctx1, make_llm_request("turn one"))
    adapter._on_after_model(ctx1, make_llm_response("mid", is_tool_call=True))
    aborted_interaction = adapter._interaction_ids["s-abort"]

    # Turn 2 arrives with a NEW invocation_id → salvage turn 1 first.
    ctx2 = _ctx_with_invocation("s-abort", "inv-2")
    adapter._on_before_model(ctx2, make_llm_request("turn two"))
    # Let the scheduled salvage task run.
    for t in list(adapter._flush_tasks):
        await t

    # The aborted turn's records were flushed under their OWN interaction_id.
    assert len(flushed) == 1
    assert all(r.interaction_id == aborted_interaction for r in flushed[0])
    # Turn 2's buffer holds only turn 2's user message, under a fresh interaction.
    buf2 = adapter._buffers["s-abort"]
    assert [r.content for r in buf2 if r.event_type == "user_message"] == ["turn two"]
    assert adapter._interaction_ids["s-abort"] != aborted_interaction


async def test_salvaged_and_new_turn_event_ids_never_collide():
    """_event_seq stays monotonic across a salvage — salvaged ids and the new
    turn's ids come from one counter, so they can never collide on the server."""
    adapter, _ = make_adapter()
    adapter._async_ingest = AsyncMock()

    ctx1 = _ctx_with_invocation("s-noclash", "inv-1")
    adapter._on_before_model(ctx1, make_llm_request("one"))  # msg-...-0
    adapter._on_after_model(ctx1, make_llm_response("mid", is_tool_call=True))  # step-...-1

    ctx2 = _ctx_with_invocation("s-noclash", "inv-2")
    adapter._on_before_model(ctx2, make_llm_request("two"))  # msg-...-2 (NOT reset to 0)

    new_turn_user = adapter._buffers["s-noclash"][0]
    assert new_turn_user.event_id == "msg-s-noclash-2"


# ─── HTTP observer (Beacon traffic observability) ──────────────────────────
#
# Mirrors the ClaudeAdapter http_observer feature: an opt-in callback fired for
# every Beacon HTTP call so a demo UI can render live traffic. These tests drive
# the httpx event hooks directly with real httpx.Request/Response objects (no
# network), plus the _ensure_clients / close() wiring with a mock auth.

import httpx  # noqa: E402

# Parity with Claude: consumers can type their observer via the package. The
# import itself is the re-export check — it fails collection if HttpObserver goes.
from agentic_context_adk import HttpObserver  # noqa: E402,F401


def _fake_auth(token: str = "tok-123", base_url: str = "https://beacon.test"):
    """Minimal ContextAuth-shaped mock: aget_token() + attach()."""
    auth = MagicMock()
    auth.aget_token = AsyncMock(return_value=(token, base_url))
    auth.attach = MagicMock()
    return auth


async def _drive_hooks(adapter, request, response):
    """Run the adapter's request+response hooks against one httpx exchange.

    Async so it runs on pytest-asyncio's loop — ``asyncio.run()`` from a sync
    test orphans that loop and trips ``filterwarnings = ["error"]`` later.
    """
    hooks = adapter._make_event_hooks()
    await hooks["request"][0](request)
    await hooks["response"][0](response)


async def test_observer_emits_paired_request_response_with_matching_callid():
    events: list[dict] = []
    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth(), http_observer=events.append)

    req = httpx.Request("POST", "https://beacon.test/v1/ingest?token=SECRET",
                        content=b'{"entry":"hi"}')
    resp = httpx.Response(200, content=b'{"ok":true}', request=req)
    await _drive_hooks(adapter, req, resp)

    assert [e["phase"] for e in events] == ["request", "response"]
    # callId ties request → response
    assert events[0]["callId"] == events[1]["callId"]
    assert events[0]["callId"]  # non-empty
    # request event shape
    assert events[0]["method"] == "POST"
    assert events[0]["requestBody"] is not None and "entry" in events[0]["requestBody"]
    assert "timestamp" in events[0]
    # response event shape
    assert events[1]["status"] == 200
    assert isinstance(events[1]["latencyMs"], int)
    assert events[1]["responseBody"] is not None and "ok" in events[1]["responseBody"]


async def test_observer_never_emits_headers_or_query_string():
    # CRITICAL privacy property: the bearer token lives in the Authorization
    # header and sometimes in query params — neither may reach the observer.
    events: list[dict] = []
    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth(), http_observer=events.append)

    req = httpx.Request(
        "GET", "https://beacon.test/v1/context?access_token=SECRET_TOKEN",
        headers={"Authorization": "Bearer SECRET_TOKEN"},
    )
    resp = httpx.Response(200, content=b"{}", request=req)
    await _drive_hooks(adapter, req, resp)

    for e in events:
        assert "headers" not in e
        assert "Authorization" not in e
        # url is path-only — query string (and any token in it) is stripped
        assert "SECRET_TOKEN" not in e["url"]
        assert "access_token" not in e["url"]
        assert e["url"] == "https://beacon.test/v1/context"


async def test_observer_may_be_async():
    # The observer can be a coroutine function — the adapter awaits it.
    events: list[dict] = []

    async def async_obs(event: dict) -> None:
        events.append(event)

    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth(), http_observer=async_obs)
    req = httpx.Request("POST", "https://beacon.test/x", content=b"{}")
    resp = httpx.Response(201, content=b"{}", request=req)
    await _drive_hooks(adapter, req, resp)

    assert len(events) == 2
    assert events[1]["status"] == 201


async def test_observer_that_raises_does_not_break_delivery():
    # Observability must never break Beacon delivery: a raising observer is
    # swallowed and both hooks still complete.
    def boom(event: dict) -> None:
        raise RuntimeError("observer blew up")

    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth(), http_observer=boom)
    req = httpx.Request("POST", "https://beacon.test/x", content=b"{}")
    resp = httpx.Response(200, content=b"{}", request=req)
    # Should not raise.
    await _drive_hooks(adapter, req, resp)


def test_decode_body_empty_json_and_truncation():
    dec = GoogleAdkAdapter._decode_body
    # empty → None
    assert dec(None) is None
    assert dec(b"") is None
    # JSON is pretty-printed (indent=2 introduces newlines)
    pretty = dec(b'{"a":1,"b":2}')
    assert pretty is not None and "\n" in pretty and '"a": 1' in pretty
    # non-JSON passes through as text
    assert dec(b"plain text") == "plain text"
    # non-utf8 bytes don't raise (errors="replace")
    assert dec(b"\xff\xfe") is not None
    # oversized bodies are capped with the truncation marker
    big = b'"' + b"x" * (GoogleAdkAdapter._BODY_CAP + 100) + b'"'
    capped = dec(big)
    assert capped is not None
    assert len(capped) <= GoogleAdkAdapter._BODY_CAP + len("\n… (truncated)")
    assert capped.endswith("… (truncated)")


@pytest.mark.asyncio
async def test_ensure_clients_injects_observer_wired_httpx_clients():
    # With an observer set, _ensure_clients builds its own httpx clients and
    # injects them via client= (so _owns_client is False on the core clients),
    # tracking them for close().
    events: list[dict] = []
    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth(), http_observer=events.append)

    assert adapter._owned_http_clients == []  # nothing built pre-first-turn
    await adapter._ensure_clients()

    assert len(adapter._owned_http_clients) == 2
    assert all(isinstance(c, httpx.AsyncClient) for c in adapter._owned_http_clients)
    # The core clients were handed our clients → they do NOT own them.
    assert adapter._client._owns_client is False
    assert adapter._retrieval._owns_client is False

    await adapter.close()


@pytest.mark.asyncio
async def test_ensure_clients_no_observer_owns_its_clients():
    # Regression / None path: without an observer, behaviour is unchanged — no
    # injected clients, core clients build + own their own httpx client.
    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth())  # no http_observer

    await adapter._ensure_clients()

    assert adapter._owned_http_clients == []
    assert adapter._client._owns_client is True
    assert adapter._retrieval._owns_client is True

    await adapter.close()


@pytest.mark.asyncio
async def test_close_acloses_owned_http_clients():
    adapter = GoogleAdkAdapter(agent_id="a", auth=_fake_auth(), http_observer=lambda e: None)
    await adapter._ensure_clients()

    owned = list(adapter._owned_http_clients)
    assert len(owned) == 2

    await adapter.close()

    # Each injected client is actually closed, and the tracking list is cleared.
    assert all(c.is_closed for c in owned)
    assert adapter._owned_http_clients == []


@pytest.mark.asyncio
async def test_concurrent_first_turns_build_one_client_pair_no_leak():
    # Regression: the adapter is a shared singleton; two users' first turns can
    # interleave across the aget_token() await inside _ensure_clients. Only ONE
    # client pair may be built (the lock + re-check serialises it), so the
    # observer's injected httpx clients can't leak (untracked → never aclosed).
    import asyncio

    # An auth whose aget_token suspends, widening the race window so both
    # coroutines are guaranteed to pass the fast guard before either assigns.
    auth = MagicMock()

    async def _slow_token():
        await asyncio.sleep(0.02)
        return ("tok", "https://beacon.test")

    auth.aget_token = AsyncMock(side_effect=_slow_token)
    auth.attach = MagicMock()

    adapter = GoogleAdkAdapter(agent_id="a", auth=auth, http_observer=lambda e: None)

    await asyncio.gather(adapter._ensure_clients(), adapter._ensure_clients())

    # aget_token ran exactly once; exactly one client pair is tracked.
    assert auth.aget_token.await_count == 1
    assert len(adapter._owned_http_clients) == 2
    first_client = adapter._client

    # A subsequent call is a no-op (fast guard) — no rebuild.
    await adapter._ensure_clients()
    assert adapter._client is first_client
    assert len(adapter._owned_http_clients) == 2

    await adapter.close()

@pytest.mark.asyncio
async def test_context_pack_list_injected_at_conversation_open():
    adapter, _ = _adapter_with_mock_mgr()  # context_mode="manual"
    adapter._retrieval = AsyncMock()
    adapter._retrieval.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]}
    )
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("BASE PROMPT")
    await cb(make_ctx("s-pack-open"), req)

    adapter._retrieval.get_context_pack_list.assert_awaited_once_with(
        agent_id="test-agent"
    )
    assert "[Available context packs]" in req.config.system_instruction
    assert "PlanRules" in req.config.system_instruction
    assert "BASE PROMPT" in req.config.system_instruction
    assert adapter._allowed_pack_ids == {"PlanRules"}


@pytest.mark.asyncio
async def test_context_pack_list_fetched_once_not_on_later_turns():
    adapter, _ = _adapter_with_mock_mgr()
    adapter._retrieval = AsyncMock()
    adapter._retrieval.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]}
    )
    cb = adapter.wrap_before_model()

    await cb(make_ctx("s-pack-cadence"), make_llm_request_with_config("BASE PROMPT"))
    adapter._retrieval.get_context_pack_list.assert_awaited_once()

    req2 = make_llm_request_with_config("BASE PROMPT")
    await cb(make_ctx("s-pack-cadence"), req2)
    adapter._retrieval.get_context_pack_list.assert_awaited_once()  # still once
    assert "[Available context packs]" in req2.config.system_instruction
    assert "PlanRules" in req2.config.system_instruction
    assert req2.config.system_instruction.count("[Available context packs]\n") == 1


@pytest.mark.asyncio
async def test_context_pack_list_refetched_on_new_session():
    adapter, _ = _adapter_with_mock_mgr()
    adapter._retrieval = AsyncMock()
    adapter._retrieval.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]}
    )
    cb = adapter.wrap_before_model()

    await cb(make_ctx("s-pack-a"), make_llm_request_with_config("BASE PROMPT"))
    adapter._retrieval.get_context_pack_list.assert_awaited_once()

    req_b = make_llm_request_with_config("BASE PROMPT")
    await cb(make_ctx("s-pack-b"), req_b)
    assert adapter._retrieval.get_context_pack_list.await_count == 2
    assert "[Available context packs]" in req_b.config.system_instruction


@pytest.mark.asyncio
async def test_context_pack_list_fetch_failure_degrades_silently():
    adapter, _ = _adapter_with_mock_mgr()
    adapter._retrieval = AsyncMock()
    adapter._retrieval.get_context_pack_list = AsyncMock(side_effect=RuntimeError("down"))
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("BASE PROMPT")

    await cb(make_ctx("s-pack-fail"), req)  # must not raise

    assert req.config.system_instruction == f"BASE PROMPT\n\n[Available context packs]\n{{}}"
    assert adapter._allowed_pack_ids == set()


@pytest.mark.asyncio
async def test_context_pack_list_not_duplicated_on_tool_reentry():
    adapter, _ = _adapter_with_mock_mgr()
    adapter._retrieval = AsyncMock()
    adapter._retrieval.get_context_pack_list = AsyncMock(
        return_value={"contextPacks": [{"contextPackId": "PlanRules", "description": "test"}]}
    )
    cb = adapter.wrap_before_model()
    ctx = _ctx_with_invocation("s-pack-reentry", "inv-1")
    req = make_llm_request_with_config("BASE PROMPT")

    await cb(ctx, req)
    assert req.config.system_instruction.count("[Available context packs]\n") == 1
    await cb(ctx, req)
    assert req.config.system_instruction.count("[Available context packs]\n") == 1


@pytest.mark.asyncio
async def test_context_pack_list_no_retrieval_client_is_noop():
    """Manager-injection with no owned retrieval client (self._retrieval is None)."""
    adapter, _ = _adapter_with_mock_mgr()
    assert adapter._retrieval is None
    cb = adapter.wrap_before_model()
    req = make_llm_request_with_config("BASE PROMPT")

    await cb(make_ctx("s-pack-none"), req)  # must not raise

    assert req.config.system_instruction == f"BASE PROMPT\n\n[Available context packs]\n{{}}"
