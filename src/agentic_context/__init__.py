# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Framework-agnostic core for agent-context-sdk.

Public surface (semver-committed):
    Record, Capabilities, ContextSource,
    ContextAuth, UserContextAuth, ApiKeyAuth,
    AuthorizationRequest, UserLogin, ReauthRequired,
    LoginStateStore, RefreshTokenStore,
    InMemoryLoginStateStore, InMemoryRefreshTokenStore,
    FileLoginStateStore, FileRefreshTokenStore,
    IngestionClient, IngestionError, IngestionResponse,
    AsyncIngestionClient,
    RetrievalClient, AsyncRetrievalClient,
    SessionManager, AsyncSessionManager,
    IngestionCallbacks,
    CONTEXT_PACK_INSTRUCTION, CONTEXT_PACK_LIST_MARKER, CONTEXT_PACK_TIMEOUT_S,
    SYSTEM_PROMPT_TYPE_CONTEXT_PACK.

Preview (exported, not yet semver-committed):
    AgentContextClient, allocate_session_id.

Unified adapter façade (lazily loaded):
    ClaudeAdapter, GoogleAdkAdapter, LangGraphAdapter, FoundryAdapter,
    OpenAIAdapter.

The five framework adapters are re-exported here so a single
``from agentic_context import ClaudeAdapter`` works even though each adapter
physically lives in its own package (``agentic_context_claude``,
``agentic_context_adk``, ``agentic_context_langgraph``,
``agentic_context_foundry``, ``agentic_context_openai``). They are resolved
LAZILY via PEP 562 ``__getattr__`` for three reasons:

* each adapter module does ``from agentic_context import ...`` at its own import
  time, so eager imports here would create a circular import;
* importing core stays cheap — none of the heavy framework SDKs
  (claude-agent-sdk / google-adk / langgraph / agent-framework-foundry) load
  until the corresponding adapter attribute is first referenced; and
* the framework SDKs are NOT dependencies of ``agent-context-sdk`` (a single
  ``pip install agent-context-sdk`` installs core only) — they are expected to be
  present because the user already runs that framework. A user who only runs
  Claude can ``import agentic_context`` without the google-adk / langgraph /
  Foundry SDKs present; the ImportError surfaces only if they actually touch
  that adapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .auth import (
    ApiKeyAuth,
    AuthorizationRequest,
    ContextAuth,
    ReauthRequired,
    UserContextAuth,
    UserLogin,
)
from .callbacks import IngestionCallbacks
from .client import IngestionClient, IngestionError, IngestionResponse
from .client_async import AsyncIngestionClient
from .ace_sdk_client import AgentContextClient, allocate_session_id
from .models import Capabilities, ContextSource, Record
from .retrieval import (
    CONTEXT_PACK_INSTRUCTION,
    CONTEXT_PACK_LIST_MARKER,
    CONTEXT_PACK_TIMEOUT_S,
    CONTEXT_PACK_VERSION_GUIDANCE,
    SYSTEM_PROMPT_TYPE_CONTEXT_PACK,
    AsyncRetrievalClient,
    RetrievalClient,
    patch_pack_version_in_catalog,
    version_mismatch_entry,
)
from .session import AsyncSessionManager, SessionManager
from .stores import (
    FileLoginStateStore,
    FileRefreshTokenStore,
    InMemoryLoginStateStore,
    InMemoryRefreshTokenStore,
    LoginStateStore,
    RefreshTokenStore,
)

if TYPE_CHECKING:
    # Import-time only for type checkers/IDEs; never executed at runtime, so it
    # neither triggers the circular import nor requires the framework SDKs.
    from agentic_context_adk import GoogleAdkAdapter
    from agentic_context_claude import ClaudeAdapter
    from agentic_context_foundry import FoundryAdapter
    from agentic_context_langgraph import LangGraphAdapter
    from agentic_context_openai import OpenAIAdapter

# name exposed on `agentic_context` -> (module to import, attribute in it)
_LAZY_ADAPTERS = {
    "ClaudeAdapter": ("agentic_context_claude", "ClaudeAdapter"),
    "GoogleAdkAdapter": ("agentic_context_adk", "GoogleAdkAdapter"),
    "LangGraphAdapter": ("agentic_context_langgraph", "LangGraphAdapter"),
    "FoundryAdapter": ("agentic_context_foundry", "FoundryAdapter"),
    "OpenAIAdapter": ("agentic_context_openai", "OpenAIAdapter"),
}

# adapter attribute name -> the pip install command for its framework SDK.
# These SDKs are NOT dependencies of agent-context-sdk (see pyproject.toml): the
# SDK is expected to be present because the user already runs that framework.
# When it is not, we name the exact package to install rather than a bare
# `No module named ...`.
_ADAPTER_SDK_INSTALL = {
    "ClaudeAdapter": "pip install claude-agent-sdk",
    "GoogleAdkAdapter": "pip install google-adk",
    "LangGraphAdapter": "pip install langgraph langchain-core",
    "FoundryAdapter": "pip install agent-framework-foundry",
    "OpenAIAdapter": "pip install openai-agents",
}


# deprecated name -> current name; resolved in __getattr__ with a warning.
_DEPRECATED_ALIASES = {
    "AceSdkClient": "AgentContextClient",
}


def __getattr__(name: str) -> object:
    """PEP 562 lazy loader for deprecated aliases and the adapter façade.

    A name in ``_DEPRECATED_ALIASES`` emits a ``DeprecationWarning`` and
    returns the current class. A framework adapter name imports its package.
    Raises the standard AttributeError for any truly unknown attribute. If the
    adapter's framework SDK is not installed, re-raises the ImportError with an
    actionable hint naming the exact package to install, instead of a bare
    ``No module named 'langchain_core'``.

    Note: only ``LangGraphAdapter`` imports its SDK at module load, so only it
    reaches this handler. ``ClaudeAdapter`` / ``GoogleAdkAdapter`` /
    ``FoundryAdapter`` import their SDK lazily inside methods, so the class
    resolves here without the SDK and the (raw) ImportError surfaces from the
    first SDK-backed method call instead.
    """
    if name in _DEPRECATED_ALIASES:
        import warnings

        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{name} is deprecated; use {new_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    target = _LAZY_ADAPTERS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        install = _ADAPTER_SDK_INSTALL[name]
        raise ImportError(
            f"{name} requires its framework SDK, which is not installed. "
            f"Install it with:\n\n    {install}\n"
        ) from exc
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # Authentication
    "ApiKeyAuth",
    "AuthorizationRequest",
    "ContextAuth",
    "ReauthRequired",
    "UserContextAuth",
    "UserLogin",
    # Login/token storage for UserContextAuth (interfaces + built-in stores)
    "FileLoginStateStore",
    "FileRefreshTokenStore",
    "InMemoryLoginStateStore",
    "InMemoryRefreshTokenStore",
    "LoginStateStore",
    "RefreshTokenStore",
    # Core data model
    "Capabilities",
    "ContextSource",
    "Record",
    # Sync transport + lifecycle
    "IngestionClient",
    "IngestionError",
    "IngestionResponse",
    "SessionManager",
    # Async transport + lifecycle
    "AsyncIngestionClient",
    "AsyncSessionManager",
    # Read-side (consumption)
    "RetrievalClient",
    "AsyncRetrievalClient",
    # Context packs (POST GetContext) — shared agent guidance + catalog marker + timeout
    "CONTEXT_PACK_INSTRUCTION",
    "CONTEXT_PACK_LIST_MARKER",
    "CONTEXT_PACK_TIMEOUT_S",
    # Server-side system prompt (GET /prompts/system/{type}) — prompt-type id
    "SYSTEM_PROMPT_TYPE_CONTEXT_PACK",
    # Context-pack versioning — shared tool-docstring block + helpers adapters use to
    # recognize and recover from Beacon's version_expired envelope.
    "CONTEXT_PACK_VERSION_GUIDANCE",
    "patch_pack_version_in_catalog",
    "version_mismatch_entry",
    # Observability (optional)
    "IngestionCallbacks",
    # AgentContextClient
    "AgentContextClient",
    "allocate_session_id",
    # Framework adapters (lazy façade)
    "ClaudeAdapter",
    "GoogleAdkAdapter",
    "LangGraphAdapter",
    "FoundryAdapter",
    "OpenAIAdapter",
]

__version__ = "0.7.0"
