# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""The Google ADK is imported lazily inside the tool factories, not at module
load — so ``GoogleAdkAdapter`` constructs (and the pure callback/ingest path
works) even when ``google-adk`` is absent. These tests pin the two-part UX that
makes that lazy failure friendly instead of a bare ``ModuleNotFoundError``:

* constructing without the SDK emits a non-fatal ``UserWarning`` at a
  predictable point (object creation), naming the exact ``pip install`` fix, and
* calling a tool factory without the SDK raises an ``ImportError`` carrying the
  same actionable hint, with the original error chained as ``__cause__``.
"""
from __future__ import annotations

import warnings

import agentic_context_adk.adapter as adapter_mod
import pytest
from agentic_context_adk.adapter import _SDK_INSTALL_HINT, _SDK_MODULE, _require_sdk


def _block_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``importlib.import_module(_SDK_MODULE)`` fail as if google-adk is absent."""
    real = adapter_mod.importlib.import_module

    def _fake(name: str, *a: object, **k: object) -> object:
        if name == _SDK_MODULE or name.split(".", 1)[0] == _SDK_MODULE.split(".", 1)[0]:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real(name, *a, **k)

    monkeypatch.setattr(adapter_mod.importlib, "import_module", _fake)


def test_require_sdk_raises_friendly_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch)

    with pytest.raises(ImportError) as excinfo:
        _require_sdk("make_context_retrieval_tool")

    msg = str(excinfo.value)
    assert "GoogleAdkAdapter.make_context_retrieval_tool()" in msg
    assert _SDK_INSTALL_HINT in msg  # "pip install google-adk"
    # Original error preserved as the cause for debuggability.
    assert excinfo.value.__cause__ is not None


def test_require_sdk_is_noop_when_sdk_present() -> None:
    # google-adk IS installed in the dev venv, so this must not raise. Importing
    # it emits benign DeprecationWarnings (_UnionGenericAlias / BaseAgentConfig)
    # that the ADK test config escalates to errors — suppress those two here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*_UnionGenericAlias.*", DeprecationWarning)
        warnings.filterwarnings("ignore", ".*BaseAgentConfig.*", DeprecationWarning)
        _require_sdk("make_start_session_tool")


def test_constructor_warns_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # find_spec returning None simulates an uninstalled SDK without importing it.
    monkeypatch.setattr(adapter_mod.importlib.util, "find_spec", lambda name: None)
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy", consumer_key="dummy",
        subject="dummy@example.com", login_url="login.example.com",
    )
    with pytest.warns(UserWarning) as record:
        adapter_mod.GoogleAdkAdapter(agent_id="a", auth=auth)

    text = str(record[0].message)
    assert _SDK_MODULE in text
    assert _SDK_INSTALL_HINT in text


def test_constructor_silent_when_sdk_present() -> None:
    # SDK present → construction must NOT emit the SDK-missing warning. (Other
    # unrelated warnings are ignored; we only assert none mention the SDK.)
    from agentic_context import ContextAuth

    auth = ContextAuth(
        private_key="dummy", consumer_key="dummy",
        subject="dummy@example.com", login_url="login.example.com",
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        adapter_mod.GoogleAdkAdapter(agent_id="a", auth=auth)

    sdk_warnings = [w for w in record if _SDK_INSTALL_HINT in str(w.message)]
    assert not sdk_warnings, f"unexpected SDK-missing warning(s): {sdk_warnings}"
