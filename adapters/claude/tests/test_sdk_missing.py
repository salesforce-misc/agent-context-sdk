# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""The Claude Agent SDK is imported lazily inside hooks()/mcp_server()/the
get-context tools, not at module load — so ``ClaudeAdapter`` constructs (and the
pure ingest path works) even when ``claude-agent-sdk`` is absent. These tests
pin the two-part UX that makes that lazy failure friendly instead of a bare
``ModuleNotFoundError``:

* constructing without the SDK emits a non-fatal ``UserWarning`` at a
  predictable point (object creation), naming the exact ``pip install`` fix, and
* calling an SDK-backed method without the SDK raises an ``ImportError``
  carrying the same actionable hint, with the original error chained as
  ``__cause__``.

The tests monkeypatch the adapter module's own ``importlib`` handles rather than
touching ``sys.modules``, so they are independent of whether sibling test
modules stub ``claude_agent_sdk``.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

import agentic_context_claude.adapter as adapter_mod
from agentic_context_claude.adapter import (
    _SDK_INSTALL_HINT,
    _SDK_MODULE,
    _require_sdk,
)

CREDS = {
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
    "consumer_key": "3MVG.test",
    "subject": "user@org.com",
    "login_url": "login.salesforce.com",
}


def _block_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``importlib.import_module(_SDK_MODULE)`` fail as if the SDK is absent."""
    real = adapter_mod.importlib.import_module

    def _fake(name: str, *a: object, **k: object) -> object:
        if name == _SDK_MODULE:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real(name, *a, **k)

    monkeypatch.setattr(adapter_mod.importlib, "import_module", _fake)


def test_require_sdk_raises_friendly_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch)

    with pytest.raises(ImportError) as excinfo:
        _require_sdk("hooks")

    msg = str(excinfo.value)
    assert "ClaudeAdapter.hooks()" in msg
    assert _SDK_INSTALL_HINT in msg  # "pip install claude-agent-sdk"
    assert excinfo.value.__cause__ is not None


def test_require_sdk_is_noop_when_sdk_present() -> None:
    # claude-agent-sdk IS installed in the dev venv, so this must not raise.
    _require_sdk("mcp_server")


def _construct_with_mocked_transport() -> None:
    """Build a ClaudeAdapter with auth + clients mocked (no network, no real SDK)."""
    with patch("agentic_context_claude.adapter.AsyncIngestionClient"), \
         patch("agentic_context_claude.adapter.AsyncRetrievalClient"), \
         patch("agentic_context_claude.adapter.ContextAuth") as MockAuth:
        auth_inst = MagicMock()
        auth_inst.get_token.return_value = ("test-token", "https://beacon.test")
        MockAuth.return_value = auth_inst
        adapter_mod.ClaudeAdapter(agent_id="a", owner_id="u", **CREDS)


def test_constructor_warns_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # find_spec returning None simulates an uninstalled SDK without importing it.
    monkeypatch.setattr(adapter_mod.importlib.util, "find_spec", lambda name: None)

    with pytest.warns(UserWarning) as record:
        _construct_with_mocked_transport()

    text = str(record[0].message)
    assert _SDK_MODULE in text
    assert _SDK_INSTALL_HINT in text


def test_constructor_silent_when_sdk_present() -> None:
    # SDK present → construction must NOT emit the SDK-missing warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _construct_with_mocked_transport()

    sdk_warnings = [w for w in record if _SDK_INSTALL_HINT in str(w.message)]
    assert not sdk_warnings, f"unexpected SDK-missing warning(s): {sdk_warnings}"
