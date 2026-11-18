# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""LangGraph adapter for agent-context-sdk.

Public surface (semver-committed):
    LangGraphAdapter, TurnEvent, ToolEvent, ErrorEvent,
    make_context_node, format_context_block,
    context_to_system_message, conversation_to_messages,
    ContextPolicy, SessionFirstPolicy, EveryTurnPolicy, TimeDecayPolicy,
    TopicShiftPolicy, NoFetchPolicy, resolve_policy.

``LangGraphAdapter`` additionally exposes the on-demand retrieval surface as
instance members (not separate module symbols, so they are not listed in
``__all__``):
    - ``make_context_retrieval_tool()`` — a LangChain ``StructuredTool`` the LLM
      calls to read the cross-session memory already loaded for this session.
      Zero-argument cache read; no new retrieval.
    - ``default_tools()`` — ``[recall_customer_context, get_context_pack]``; passed
      to a ``graph=`` builder ``(sdk_tools) -> compiled graph`` so they get bound.
    - ``make_get_context_pack_tool(valid_pack_ids)`` — a ``StructuredTool`` for
      the pack-scoped read (POST GetContext). Pair with the re-exported
      ``CONTEXT_PACK_INSTRUCTION`` guidance; opt-in (advertise your pack ids).

The adapter's policy-driven per-turn context injection (self-driving mode) is
unchanged — the tool is additive, not a replacement.
"""

from __future__ import annotations

from agentic_context import CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_TIMEOUT_S

from .adapter import LangGraphAdapter
from .events import ErrorEvent, ToolEvent, TurnEvent
from .policies import (
    ContextPolicy,
    EveryTurnPolicy,
    NoFetchPolicy,
    SessionFirstPolicy,
    TimeDecayPolicy,
    TopicShiftPolicy,
    resolve_policy,
)
from .retrieval_node import make_context_node
from .retrieval_normalize import (
    context_to_system_message,
    conversation_to_messages,
    format_context_block,
)

__all__ = [
    "CONTEXT_PACK_INSTRUCTION",
    "CONTEXT_PACK_TIMEOUT_S",
    "ContextPolicy",
    "ErrorEvent",
    "EveryTurnPolicy",
    "LangGraphAdapter",
    "NoFetchPolicy",
    "SessionFirstPolicy",
    "TimeDecayPolicy",
    "ToolEvent",
    "TopicShiftPolicy",
    "TurnEvent",
    "context_to_system_message",
    "conversation_to_messages",
    "format_context_block",
    "make_context_node",
    "resolve_policy",
]

__version__ = "0.7.0"
