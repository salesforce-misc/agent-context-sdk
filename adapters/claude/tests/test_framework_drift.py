# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Regression test for ClaudeAdapter.list_sessions' best-effort exception site.

Previously the failure was logged at DEBUG (easy to miss in production), even
though ``list_sessions`` still correctly degrades to ``[]`` rather than
raising to the caller. The fix bumps the log level to WARNING so a genuine
Beacon/framework-shape drift is observable without changing the best-effort
contract.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from agentic_context_claude import ClaudeAdapter

from .test_adapter import CREDS, mock_async_clients  # noqa: F401 (fixture import)

LOGGER_NAME = "agentic_context_claude.adapter"


@pytest.mark.asyncio
async def test_list_sessions_survives_attribute_error(mock_async_clients, caplog) -> None:
    adapter = ClaudeAdapter(
        agent_id="agent-support",
        owner_id="user-001",
        **CREDS,
    )
    mock_async_clients["retrieval"].list_conversations = AsyncMock(
        side_effect=AttributeError("no attribute 'list_conversations'")
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = await adapter.list_sessions()

    assert result == []
    assert "list_sessions" in caplog.text
    assert "list_conversations" in caplog.text
