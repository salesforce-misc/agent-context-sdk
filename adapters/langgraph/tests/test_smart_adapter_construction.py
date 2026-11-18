# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Constructor validation for the self-driving (path-3) ``LangGraphAdapter``.

These tests do not run the graph or hit the network — they pin down the
shape of the public constructor contract:

- All-in-one path: ``LangGraphAdapter(graph=..., agent_id=..., owner_id=...,
  api_key=..., base_url=..., dataspace=...)`` succeeds and returns an
  adapter that owns its clients (verified by exposing ``__aenter__``).
- Mixing ``session_manager=`` with any connection parameter is a
  configuration mistake -> ``ValueError``.
- Missing one of the required connection parameters (``owner_id``,
  ``api_key``, ``base_url``, ``agent_id``) -> ``ValueError``. ``dataspace``
  is the one optional param and defaults to ``"default"``.
- ``pickle_path=`` is forbidden in self-driving mode (no live graph to drive).
- ``async with adapter:`` is a *self-driving* affordance — calling it on the
  legacy manager-injected adapter raises ``RuntimeError`` so customers
  don't accidentally double-own clients.
- ``context_mode`` → policy mapping lives in ``test_smart_adapter_context_mode``;
  ``resolve_policy`` (slugs, unknown slug, instance passthrough) in
  ``test_smart_adapter_policies``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agentic_context import AsyncSessionManager
from langgraph.graph import StateGraph

from agentic_context_langgraph import LangGraphAdapter

from .conftest import _build_graph

BASE_URL = "https://beacon.example.com"


def _conn_kwargs(**overrides):
    """Default connection params for a self-driving adapter."""
    base = {
        "agent_id": "bot",
        "owner_id": "user-1",
        "api_key": "tok",
        "base_url": BASE_URL,
        "dataspace": "default",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "env, drop, overrides",
    [
        ({}, (), {}),
        # ``dataspace`` is the one optional connection param — should default.
        ({}, ("dataspace",), {}),
        # ``api_key``/``base_url``/``dataspace`` fall back to CDP_* envs when the
        # kwargs are omitted, so 12-factor deployments don't have to thread
        # os.environ themselves.
        (
            {"CDP_TOKEN": "env-tok", "CDP_BASE_URL": BASE_URL, "CDP_DATASPACE": "from-env"},
            ("api_key", "base_url", "dataspace"),
            {},
        ),
        # An explicit kwarg always wins over the corresponding env var.
        (
            {"CDP_TOKEN": "env-tok", "CDP_BASE_URL": "https://wrong-host"},
            ("dataspace",),
            {"api_key": "explicit-tok"},
        ),
    ],
    ids=["all-required-kwargs", "dataspace-defaults", "env-var-fallback", "explicit-kwargs-win-over-env"],
)
def test_self_driving_construction_succeeds(
    monkeypatch: pytest.MonkeyPatch, env, drop, overrides
) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    kwargs = _conn_kwargs(**overrides)
    for k in drop:
        kwargs.pop(k)
    adapter = LangGraphAdapter(graph=_build_graph(), **kwargs)
    assert adapter is not None
    # Sanity: the self-driving path advertises the async-context-manager surface.
    assert hasattr(adapter, "__aenter__") and hasattr(adapter, "__aexit__")


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"api_key": "tok"},
        {"owner_id": "user-1"},
        {"base_url": BASE_URL},
        {"dataspace": "default"},
    ],
    ids=["api_key", "owner_id", "base_url", "dataspace"],
)
def test_mixing_session_manager_and_connection_params_raises(extra_kwargs) -> None:
    """``session_manager=`` is mutually exclusive with any connection param."""
    mgr = MagicMock(spec=AsyncSessionManager)
    with pytest.raises(ValueError):
        LangGraphAdapter(
            agent_id="bot",
            session_manager=mgr,
            graph=_build_graph(),
            **extra_kwargs,
        )


@pytest.mark.parametrize(
    "missing_key",
    ["api_key"],
)
def test_self_driving_missing_required_param_raises(missing_key: str) -> None:
    kwargs = _conn_kwargs()
    kwargs.pop(missing_key)
    with pytest.raises(ValueError):
        LangGraphAdapter(graph=_build_graph(), **kwargs)


def test_self_driving_without_owner_id_constructs_ok() -> None:
    """owner_id is optional at construction — a multi-tenant server
    supplies it per-turn via stream()/run(). Missing owner_id must NOT raise
    here; the ValueError moves to _resolve_owner() at turn time."""
    kwargs = _conn_kwargs()
    kwargs.pop("owner_id")
    adapter = LangGraphAdapter(graph=_build_graph(), **kwargs)
    assert adapter._owner_id == ""


def test_api_key_without_base_url_routes_to_secret_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_key + owner_id but NO base_url and NO raw creds → demo (ApiKeyAuth)
    path. This used to raise for a missing base_url; the static-token
    path now requires base_url alongside api_key, while api_key-only is the
    secret-server demo path."""
    from agentic_context import ApiKeyAuth

    monkeypatch.delenv("CDP_BASE_URL", raising=False)
    kwargs = _conn_kwargs()
    kwargs.pop("base_url")
    adapter = LangGraphAdapter(graph=_build_graph(), **kwargs)
    assert isinstance(adapter._auth, ApiKeyAuth)


def test_self_driving_with_pickle_path_raises(tmp_path) -> None:
    """Self-driving mode requires a live graph; pickle_path makes no sense."""
    pkl = tmp_path / "ses.pkl"
    pkl.write_bytes(b"")  # never read — constructor should reject before load
    with pytest.raises(ValueError):
        LangGraphAdapter(pickle_path=pkl, **_conn_kwargs())


async def test_async_with_on_manager_injected_adapter_raises() -> None:
    """The adapter doesn't own clients in path 1, so __aenter__ must reject."""
    mgr = MagicMock(spec=AsyncSessionManager)
    adapter = LangGraphAdapter(
        agent_id="bot", session_manager=mgr, graph=_build_graph()
    )
    with pytest.raises(RuntimeError):
        async with adapter:
            pass


def _manager_injected(**kwargs):
    return LangGraphAdapter(
        session_manager=MagicMock(spec=AsyncSessionManager), graph=_build_graph(), **kwargs
    )


def _self_driving(**kwargs):
    return LangGraphAdapter(graph=_build_graph(), **kwargs)


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (_self_driving, {k: v for k, v in _conn_kwargs().items() if k != "agent_id"}),
        (_manager_injected, {}),
    ],
    ids=["self-driving", "manager-injection"],
)
def test_missing_agent_id_raises_clear_error(factory, kwargs) -> None:
    """agent_id has no env-var fallback — missing it must surface a clear message.
    In manager-injection mode, agent_id is also required and must be flagged."""
    with pytest.raises(ValueError, match="agent_id"):
        factory(**kwargs)


@pytest.mark.parametrize(
    "partial_creds",
    [
        # Each set provides 3 of 4 inline credentials; the 4th is missing.
        {"private_key": "pk", "consumer_key": "ck", "subject": "s@example.com"},
        {"private_key": "pk", "consumer_key": "ck", "login_url": "login.example.com"},
        {"private_key": "pk", "subject": "s@example.com", "login_url": "login.example.com"},
        {"consumer_key": "ck", "subject": "s@example.com", "login_url": "login.example.com"},
    ],
    ids=["missing_login_url", "missing_subject", "missing_consumer_key", "missing_private_key"],
)
def test_partial_inline_credentials_raise(partial_creds: dict) -> None:
    """Inline ContextAuth creds are all-or-nothing — 3 of 4 must be rejected."""
    # Use connection params consistent with self-driving mode. The missing
    # api_key/base_url here means the four credential kwargs are the only
    # auth source; partial provisioning must raise rather than silently
    # fall through to the static-token branch.
    with pytest.raises(ValueError, match="all-or-nothing"):
        LangGraphAdapter(
            graph=_build_graph(),
            agent_id="bot",
            owner_id="user-1",
            **partial_creds,
        )


@pytest.mark.parametrize(
    "kwargs, expected_tools",
    [
        (_conn_kwargs(), ["recall_customer_context", "get_context_pack"]),
        (
            {"agent_id": "bot", "session_manager": MagicMock(spec=AsyncSessionManager)},
            [],
        ),
    ],
    ids=["self-driving", "manager-injection"],
)
def test_graph_builder_receives_default_tools(kwargs, expected_tools) -> None:
    """graph= may be a builder; it is called once with default_tools() and its
    return value becomes the driven graph."""
    compiled = _build_graph()
    received: list = []

    def build(sdk_tools):
        received.append(sdk_tools)
        return compiled

    adapter = LangGraphAdapter(graph=build, **kwargs)

    assert len(received) == 1
    assert [t.name for t in received[0]] == expected_tools
    assert adapter._graph is compiled
    assert adapter.mode == "stream"


def test_compiled_graph_is_not_treated_as_builder() -> None:
    compiled = _build_graph()
    adapter = LangGraphAdapter(graph=compiled, **_conn_kwargs())
    assert adapter._graph is compiled


@pytest.mark.parametrize(
    "build, session_owner, turn_owner, expected",
    [
        (lambda: _self_driving(**_conn_kwargs()), None, None, "user-1"),
        (lambda: _self_driving(**_conn_kwargs()), "per-turn-owner", None, "per-turn-owner"),
        # The turn's own owner beats the shared session slot (which a concurrent
        # turn for the same session under another owner may have overwritten).
        (lambda: _self_driving(**_conn_kwargs()), "other-tenant", "turn-owner", "turn-owner"),
        (
            lambda: _self_driving(**_conn_kwargs(), user_participant_id="explicit"),
            "x",
            "turn-owner",
            "explicit",
        ),
        (
            lambda: LangGraphAdapter(
                agent_id="bot",
                session_manager=MagicMock(spec=AsyncSessionManager, owner_id="mgr-owner"),
                graph=_build_graph(),
            ),
            None,
            None,
            "mgr-owner",
        ),
        (lambda: _manager_injected(agent_id="bot"), None, None, "user"),
    ],
    ids=[
        "constructor-owner",
        "session-owner",
        "turn-owner-beats-session-slot",
        "explicit-kwarg-wins",
        "manager-owner",
        "no-owner-fallback",
    ],
)
def test_user_participant_defaults_to_resolved_owner(
    build, session_owner, turn_owner, expected
) -> None:
    """Beacon marks an entry role="owner" only when participantId == ownerId,
    so user records default to the turn's resolved owner."""
    adapter = build()
    if session_owner is not None:
        adapter._session_owner["ses-1"] = session_owner
    assert adapter._user_participant_for("ses-1", turn_owner) == expected


@pytest.mark.parametrize(
    "returned",
    [None, StateGraph(dict)],
    ids=["none", "uncompiled-stategraph"],
)
def test_graph_builder_must_return_compiled_graph(returned) -> None:
    with pytest.raises(TypeError, match="compiled graph"):
        LangGraphAdapter(graph=lambda sdk_tools: returned, **_conn_kwargs())


async def _async_builder(sdk_tools):
    return _build_graph()


def test_async_graph_builder_raises_clear_error() -> None:
    with pytest.raises(TypeError, match="returned a coroutine"):
        LangGraphAdapter(graph=_async_builder, **_conn_kwargs())


def test_callable_snapshot_wrapper_is_not_treated_as_builder() -> None:
    """A callable with get_state but no astream_events (e.g. a snapshot-only
    proxy) is a graph, not a builder — it must not be invoked with tools."""

    class _SnapshotOnly:
        def __call__(self, *args):
            raise AssertionError("wrapper must not be called as a builder")

        def get_state(self, cfg):
            raise NotImplementedError

    wrapper = _SnapshotOnly()
    adapter = LangGraphAdapter(graph=wrapper, **_conn_kwargs())
    assert adapter._graph is wrapper
