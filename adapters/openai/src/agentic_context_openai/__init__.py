# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""OpenAI adapter for agent-context-sdk.

Public surface: OpenAIAdapter plus CONTEXT_PACK_INSTRUCTION / CONTEXT_PACK_TIMEOUT_S
re-exported from core for a uniform import path across adapters.

Targets the OpenAI Agents SDK (``openai-agents``) as the primary runtime —
closest analog to claude-agent-sdk / google-adk / langgraph. Also accepts a
plain Chat Completions messages list for the snapshot path so teams that only
drive ``openai.chat.completions.create(...)`` can still ship their transcripts
to Beacon without adopting a new runtime.
"""
from __future__ import annotations

from agentic_context import CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_TIMEOUT_S

from .adapter import OpenAIAdapter

__all__ = [
    "CONTEXT_PACK_INSTRUCTION",
    "CONTEXT_PACK_TIMEOUT_S",
    "OpenAIAdapter",
]
__version__ = "0.7.0"
