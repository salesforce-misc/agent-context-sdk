# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""ContextAuth wiring for the self-driving LangGraph adapter.

These tests cover the auth-source resolution and the deferred client
construction that happens in ``__aenter__``. They do not exercise the
real JWT/CDP HTTP exchange — ``ContextAuth.aget_token`` is mocked out so
the adapter sees the same shape it would in production without needing
real credentials.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from agentic_context import ContextAuth

from agentic_context_langgraph import LangGraphAdapter

from .conftest import _build_graph

BASE_URL = "https://cdp.example.com"
TOKEN = "tok-1"


def _fake_auth() -> MagicMock:
    """A ContextAuth stand-in that returns deterministic (token, base_url)."""
    auth = MagicMock(spec=ContextAuth)
    auth.aget_token = AsyncMock(return_value=(TOKEN, BASE_URL))
    return auth


def test_auth_kwarg_defers_client_construction() -> None:
    """With auth=, no HTTP client is built until __aenter__, and api_key/base_url
    are not required up front — auth alone satisfies the constructor."""
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="u@o.com",
        auth=_fake_auth(),
    )
    assert adapter._owned_ingest_client is None
    assert adapter._owned_retrieval_client is None
    assert adapter._auth is not None


def test_inline_credentials_build_context_auth() -> None:
    """The four raw credential kwargs trigger an internal ContextAuth.

    The constructor never calls aget_token() — that's deferred to
    __aenter__ — so this stays a pure unit test with no crypto loaded.
    """
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="u@o.com",
        # Values are placeholders; ContextAuth.__init__ just stores them.
        private_key="-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n",
        consumer_key="3MVG9_consumer",
        subject="user@org.com",
        login_url="login.salesforce.com",
    )
    assert isinstance(adapter._auth, ContextAuth)
    assert adapter._owned_ingest_client is None


async def test_aenter_runs_token_exchange_and_attaches() -> None:
    """__aenter__ pulls (token, base_url), builds both clients, attaches both."""
    auth = _fake_auth()
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="u@o.com",
        auth=auth,
    )

    async with adapter:
        auth.aget_token.assert_awaited_once()
        assert adapter._owned_ingest_client is not None
        assert adapter._owned_retrieval_client is not None
        assert adapter._owned_ingest_client.base_url == BASE_URL
        assert adapter._owned_ingest_client.api_key == TOKEN
        assert adapter._owned_retrieval_client.base_url == BASE_URL
        assert adapter._owned_retrieval_client.api_key == TOKEN
        # Both clients must be registered so cached refreshes propagate.
        assert auth.attach.call_count == 2
        attached = {call.args[0] for call in auth.attach.call_args_list}
        assert adapter._owned_ingest_client in attached
        assert adapter._owned_retrieval_client in attached


def test_static_token_path_unchanged() -> None:
    """No auth=, just api_key/base_url — eager client construction stays."""
    adapter = LangGraphAdapter(
        graph=_build_graph(),
        agent_id="bot",
        owner_id="u@o.com",
        api_key="static-tok",
        base_url="https://beacon.example.com",
    )
    assert adapter._auth is None
    assert adapter._owned_ingest_client is not None
    assert adapter._owned_retrieval_client is not None
    assert adapter._owned_ingest_client.api_key == "static-tok"


def test_missing_credentials_still_raises_without_auth() -> None:
    """The static-token path still demands api_key + base_url when no auth=."""
    with pytest.raises(ValueError) as excinfo:
        LangGraphAdapter(
            graph=_build_graph(), agent_id="bot", owner_id="u@o.com"
        )
    msg = str(excinfo.value)
    assert "api_key" in msg
    assert "base_url" in msg
