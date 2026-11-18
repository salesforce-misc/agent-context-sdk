# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Claude Agent SDK adapter for agent-context-sdk.

Public surface: ClaudeAdapter, plus the shared CONTEXT_PACK_INSTRUCTION guidance
(re-exported from core so ``from agentic_context_claude import ...`` mirrors the
ADK/LangGraph adapters).
One class, dual-mode (sync/async via use_async parameter).
"""
from __future__ import annotations

from agentic_context import CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_TIMEOUT_S

from .adapter import ClaudeAdapter

__all__ = ["CONTEXT_PACK_INSTRUCTION", "CONTEXT_PACK_TIMEOUT_S", "ClaudeAdapter"]
__version__ = "0.7.0"
