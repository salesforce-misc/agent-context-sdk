# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Regression tests for the ADK adapter's framework-drift exception sites.

Each site used to swallow the underlying exception silently (bare
``except Exception: pass`` / ``return <default>``). The fix narrows (where the
call site only ever raises attribute/shape errors) or keeps the broad except
but always logs at WARNING with the function name, so a real framework-shape
drift is observable instead of silently eaten. These tests assert both halves
of that contract: the function still returns its documented default (never
raises to the caller) AND a WARNING naming the function lands in the logs.

Note: plain classes (not MagicMock + PropertyMock) are used to force the
AttributeError. A ``MagicMock`` defines ``__getattr__``, and Python's
attribute-lookup protocol falls back to ``__getattr__`` whenever the normal
descriptor lookup raises AttributeError — so a ``PropertyMock(side_effect=
AttributeError(...))`` set on ``type(some_magicmock)`` is silently swallowed
by the mock's own ``__getattr__`` and never reaches the adapter code. A plain
class with a raising ``@property`` has no such fallback, so the error
actually propagates.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_context_adk.adapter import _current_session_id

from .test_adapter import make_adapter

LOGGER_NAME = "agentic_context_adk.adapter"

# An immutable "config" stand-in: assigning to ``system_instruction`` raises
# AttributeError ("can't set attribute"), which is exactly the shape a
# genuinely incompatible llm_request.config would produce, and — unlike a
# MagicMock's __getattr__-swallowed reads — is NOT absorbed anywhere upstream
# of the adapter's own try/except.
_Config = namedtuple("_Config", ["system_instruction"])


class _BrokenContentsRequest:
    """Stand-in for llm_request whose .contents raises AttributeError."""

    @property
    def contents(self):
        raise AttributeError("no attribute 'contents'")


class _BrokenContentResponse:
    """Stand-in for llm_response whose .content raises AttributeError."""

    @property
    def content(self):
        raise AttributeError("no attribute 'content'")


class _BrokenUsageResponse:
    """Stand-in for llm_response whose .usage_metadata raises AttributeError."""

    @property
    def usage_metadata(self):
        raise AttributeError("no attribute 'usage_metadata'")


def test_extract_user_message_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    llm_request = _BrokenContentsRequest()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = adapter._extract_user_message(llm_request)

    assert result == ""
    assert "_extract_user_message" in caplog.text
    assert "AttributeError" in caplog.text


def test_extract_response_text_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    llm_response = _BrokenContentResponse()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = adapter._extract_response_text(llm_response)

    assert result == ""
    assert "_extract_response_text" in caplog.text
    assert "AttributeError" in caplog.text


def test_is_tool_call_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    llm_response = _BrokenContentResponse()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = adapter._is_tool_call(llm_response)

    assert result is False
    assert "_is_tool_call" in caplog.text
    assert "AttributeError" in caplog.text


def test_extract_tokens_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    llm_response = _BrokenUsageResponse()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = adapter._extract_tokens(llm_response)

    assert result == {}
    assert "_extract_tokens" in caplog.text
    assert "usage_metadata" in caplog.text


def test_inject_context_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    llm_request = SimpleNamespace(config=_Config(system_instruction="base"))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = adapter._inject_context(llm_request, {"userContext": []})

    assert result is None
    assert "_inject_context" in caplog.text
    assert "can't set attribute" in caplog.text


@pytest.mark.asyncio
async def test_inject_fast_context_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    adapter.get_context = AsyncMock(side_effect=AttributeError("no attribute 'userContext'"))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = await adapter._inject_fast_context(MagicMock(), "sess-1")

    assert result is None
    assert "_inject_fast_context" in caplog.text
    assert "userContext" in caplog.text


def test_apply_system_prompt_override_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    token = _current_session_id.set("sess-1")
    try:
        adapter._session_configs["sess-1"] = {"system_prompt_override": "persona X"}
        llm_request = SimpleNamespace(config=_Config(system_instruction="base"))

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            result = adapter._apply_system_prompt_override(llm_request)

        assert result is None
        assert "system_prompt_override not applied" in caplog.text
        assert "can't set attribute" in caplog.text
    finally:
        _current_session_id.reset(token)


@pytest.mark.asyncio
async def test_ensure_conversation_survives_attribute_error(caplog):
    adapter, _ = make_adapter()
    adapter._injected_mgr.create_session = AsyncMock(
        side_effect=AttributeError("no attribute 'create_session'")
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = await adapter._ensure_conversation("sess-1")

    assert result is None
    assert "_ensure_conversation" in caplog.text
    assert "create_session" in caplog.text
