# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for the unified adapter façade on the top-level ``agentic_context``.

The four framework adapters (Claude / ADK / LangGraph / Foundry) live in their own
import packages but are re-exported from ``agentic_context`` via a PEP 562
``__getattr__``. These tests pin the façade contract WITHOUT requiring any
framework SDK to be installed:

* unknown attributes still raise ``AttributeError`` (not ImportError),
* ``__dir__`` advertises the adapter names, and
* referencing an adapter whose framework SDK is absent raises an ``ImportError``
  carrying an actionable ``pip install <sdk>`` hint (the framework SDKs are not
  dependencies of agent-context-sdk, so we name the package to install directly).
"""
from __future__ import annotations

import builtins
import importlib

import pytest

import agentic_context


def test_unknown_attribute_raises_attributeerror() -> None:
    with pytest.raises(AttributeError):
        agentic_context.NotARealAdapter  # noqa: B018


def test_dir_lists_every_adapter() -> None:
    listed = dir(agentic_context)
    for name in (
        "ClaudeAdapter", "GoogleAdkAdapter", "LangGraphAdapter", "FoundryAdapter", "OpenAIAdapter",
    ):
        assert name in listed


def _block_sdk(
    monkeypatch: pytest.MonkeyPatch, sdk_module: str
) -> None:
    """Make imports of ``sdk_module`` (and its top-level package) fail, and drop
    any cached adapter/SDK modules so the import is genuinely re-attempted."""
    import sys

    root = sdk_module.split(".", 1)[0]
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == sdk_module or name.split(".", 1)[0] == root:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    for mod in list(sys.modules):
        if mod.startswith(("agentic_context_", root)):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)


def test_langgraph_missing_sdk_raises_friendly_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LangGraphAdapter's package imports langchain_core at MODULE load, so the
    # façade's importlib.import_module fails and we surface the actionable hint.
    _block_sdk(monkeypatch, "langchain_core")

    with pytest.raises(ImportError) as excinfo:
        agentic_context.LangGraphAdapter  # noqa: B018

    msg = str(excinfo.value)
    assert "LangGraphAdapter" in msg
    # Names the real framework packages to install (not an agent-context-sdk extra,
    # which no longer exists).
    assert "pip install langgraph langchain-core" in msg
    assert "agent-context-sdk[" not in msg
    # The original ImportError is preserved as the cause for debuggability.
    assert excinfo.value.__cause__ is not None


@pytest.mark.parametrize(
    ("attr", "sdk_module"),
    [
        ("ClaudeAdapter", "claude_agent_sdk"),
        ("GoogleAdkAdapter", "google.adk"),
        ("FoundryAdapter", "agent_framework")
    ],
)
def test_lazy_sdk_adapters_import_without_sdk(
    monkeypatch: pytest.MonkeyPatch, attr: str, sdk_module: str
) -> None:
    # ClaudeAdapter / GoogleAdkAdapter / FoundryAdapter
    # import their framework SDK LAZILY inside
    # methods, not at module load. So the façade can resolve (and return) the
    # class even when the SDK is absent — the ImportError is deferred to the
    # first method call that actually needs the SDK. Pin that contract so a
    # future refactor that moves the SDK import to module scope is noticed.
    _block_sdk(monkeypatch, sdk_module)

    cls = getattr(agentic_context, attr)
    assert cls.__name__ == attr
