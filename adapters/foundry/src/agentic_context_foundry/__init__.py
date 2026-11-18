# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Foundry (Microsoft Agent Framework) adapter for agent-context-sdk."""
from agentic_context import (
    CONTEXT_PACK_INSTRUCTION,
    CONTEXT_PACK_LIST_MARKER,
    CONTEXT_PACK_TIMEOUT_S,
)

from .adapter import FoundryAdapter

__all__ = [
    "CONTEXT_PACK_INSTRUCTION",
    "CONTEXT_PACK_LIST_MARKER",
    "CONTEXT_PACK_TIMEOUT_S",
    "FoundryAdapter",
    "FoundryContextProvider",
]


def __getattr__(name: str) -> object:
    if name == "FoundryContextProvider":
        from .context_provider import FoundryContextProvider

        return FoundryContextProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.7.0"
