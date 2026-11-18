# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Regression test for OpenAIAdapter.list_sessions' best-effort exception site.

Mirrors the Claude/ADK/LangGraph test of the same name: the failure must be
logged at WARNING (observable in production) while ``list_sessions`` still
correctly degrades to ``[]`` rather than raising to the caller — so a genuine
Beacon/framework-shape drift surfaces without changing the best-effort
contract.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from agentic_context_openai import OpenAIAdapter

from .test_adapter import CREDS, mock_async_clients  # noqa: F401 (fixture import)

LOGGER_NAME = "agentic_context_openai.adapter"


@pytest.mark.asyncio
async def test_list_sessions_survives_attribute_error(mock_async_clients, caplog) -> None:
    adapter = OpenAIAdapter(
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
