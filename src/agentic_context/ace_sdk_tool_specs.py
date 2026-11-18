# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""AgentContextClient model tool schemas (Design.md §8). Used by ``tool_specs`` / ``call_tool``."""
from __future__ import annotations

from typing import Any

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "ace_get_context",
        "description": (
            "Retrieve the latest customer memory (profile, prior "
            "summaries, cross-session facts). Use to "
            "refresh memory seeded at session open. For structured pack "
            "data use ace_get_context_pack. conversation_id optional when "
            "bound."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "Omit when the conversation is already bound",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ace_get_context_pack",
        "description": (
            "Fetch one context pack. Pass context_pack_id from "
            "[Available context packs] and that pack's declared params only. "
            "Returns status ok | invalid_pack | not_ready. conversation_id "
            "optional when bound."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "Omit when the conversation is already bound",
                },
                "context_pack_id": {
                    "type": "string",
                    "description": "Pack id from the available-packs catalog",
                },
                "params": {
                    "type": "object",
                    "description": "Declared pack parameters only",
                    "additionalProperties": True,
                },
            },
            "required": ["context_pack_id"],
        },
    },
]
