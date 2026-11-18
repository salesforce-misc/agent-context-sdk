# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""LangGraph node factory that fetches Beacon context and writes it into state.

Wraps ``AsyncRetrievalClient.get_context`` in a function shaped like a stock
LangGraph node — ``async (state, config) -> dict`` — so callers can plug it
straight into a graph::

    from agentic_context import AsyncRetrievalClient
    from agentic_context_langgraph import make_context_node

    client = AsyncRetrievalClient(base_url=..., api_key=..., dataspace=...)
    node = make_context_node(client, agent_id="cs-bot")
    workflow.add_node("recall_context", node)
    workflow.add_edge(START, "recall_context")

By default the node returns ``{"messages": [SystemMessage(...)]}`` so the
``add_messages`` reducer in standard LangGraph state merges the context into
the thread without any prompt-template surgery. Pass ``into="<field_name>"``
to land the formatted block as a plain string in a custom state field
instead — that's the drop-in for agents like the customer-support bot which
already interpolate ``{customer_context}`` into a system prompt template.

Best-effort: any exception (network, parse, ``IngestionError``) is logged
and swallowed; the node returns ``{}`` (no state change) so a Beacon outage
never breaks the user-facing run.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from agentic_context import AsyncRetrievalClient, IngestionError
from langchain_core.runnables import RunnableConfig

from .retrieval_normalize import (
    DEFAULT_URGENT_TYPES,
    context_to_system_message,
    format_context_block,
)

logger = logging.getLogger(__name__)


_MESSAGES_KEY = "messages"


def _payload_to_delta(
    payload: dict[str, Any],
    *,
    into: str,
    header: str,
    message_id: str | None,
    dedupe: bool,
    rank_urgent: bool,
    urgent_types: tuple[str, ...],
) -> dict[str, Any]:
    """Render a context payload into a LangGraph state delta.

    ``into="messages"`` returns ``{"messages": [SystemMessage(...)]}`` so the
    ``add_messages`` reducer merges it into the thread; any other value names a
    state field that receives the plain-string block. Either way an empty
    payload yields ``{}`` (no state change).
    """
    if into == _MESSAGES_KEY:
        msg = context_to_system_message(
            payload,
            header=header,
            message_id=message_id,
            dedupe=dedupe,
            rank_urgent=rank_urgent,
            urgent_types=urgent_types,
        )
        if msg is None:
            return {}
        return {_MESSAGES_KEY: [msg]}
    block = format_context_block(
        payload,
        header=header,
        dedupe=dedupe,
        rank_urgent=rank_urgent,
        urgent_types=urgent_types,
    )
    if not block:
        return {}
    return {into: block}


def _log_swallowed(node: str, exc: Exception) -> None:
    """Log a swallowed node failure. Auth errors (401/403) get an actionable
    WARNING; everything else gets a full-traceback ERROR. Called from inside an
    ``except`` block so ``logger.exception`` still captures the active
    exception.
    """
    if isinstance(exc, IngestionError) and exc.status_code in (401, 403):
        logger.warning(
            "%s auth failed (HTTP %d); returning empty state delta. "
            "Check your API key and dataspace permissions.",
            node,
            exc.status_code,
        )
    else:
        logger.exception("%s failed; returning empty state delta", node)


def make_context_node(
    client: AsyncRetrievalClient,
    *,
    agent_id: str,
    owner_id_key: str = "customer_id",
    session_id_from_config: bool | None = None,
    session_id_key: str | None = None,
    into: str = _MESSAGES_KEY,
    owner_type: str = "individual",
    header: str = "Previous context about this customer",
    message_id: str | None = "agentic-memory-context",
    dedupe: bool = False,
    rank_urgent: bool = False,
    urgent_types: tuple[str, ...] = DEFAULT_URGENT_TYPES,
    conversation_id_from_config: bool | None = None,
    conversation_id_key: str | None = None,
) -> Callable[[dict[str, Any], RunnableConfig], Awaitable[dict[str, Any]]]:
    """Return an async LangGraph node bound to ``client``.

    Three-way identifier mapping:
      - LangGraph runtime puts the session under ``config["configurable"]["thread_id"]``.
      - The core SDK calls the same identifier ``session_id`` (public API).
      - Beacon's HTTP surface still calls it ``conversation_id`` (wire format).

    This node reads ``thread_id`` from the LangGraph cfg (or a state field, if
    ``session_id_from_config=False``) and passes it to the retrieval client's
    ``get_context(conversation_id=...)`` — which is just the wire-level name
    for the same string.

    Parameters
    ----------
    client:
        Open ``AsyncRetrievalClient``. The factory does **not** take ownership
        — keep it alive and close it yourself (typically via
        ``async with AsyncRetrievalClient(...) as client:``).
    agent_id:
        Required by the Context Service GET. Identifies the calling agent for
        scoping.
    owner_id_key:
        State field that holds the owner id (the Beacon ``conversationOwnerId``
        / ``owner_id`` query param). Defaults to ``"customer_id"`` to match
        the customer-support agent.
    session_id_from_config:
        When ``True`` (default), pull the session id from
        ``config["configurable"][session_id_key]`` — that's where the
        LangGraph runtime puts ``thread_id``. When ``False``, read it from
        ``state[session_id_key]`` instead.
    session_id_key:
        Name of the session-id field in either config or state. Defaults to
        ``"thread_id"`` because that's LangGraph's term for the same identifier.
    into:
        Destination for the formatted context. ``"messages"`` (default) returns
        ``{"messages": [SystemMessage(...)]}`` so the standard ``add_messages``
        reducer merges it into the thread. Any other value names a state field
        that gets the plain-string render (drop-in for templated system
        prompts).
    owner_type:
        Beacon owner type — currently ``individual`` for customer scenarios.
    header:
        Override the header line in the formatted block.
    message_id:
        Stable ``id`` set on the ``SystemMessage`` returned in messages mode,
        defaulting to ``"agentic-memory-context"``. The default ensures that
        re-injecting context every turn replaces the previous copy (via the
        ``add_messages`` reducer) instead of accumulating one SystemMessage
        per turn. Pass ``None`` to fall back to LangChain's behavior of
        generating a fresh id per call. Ignored when ``into != "messages"``.

    Deprecated
    ----------
    ``conversation_id_from_config`` and ``conversation_id_key`` are the
    pre-0.2.0 names for ``session_id_from_config`` and ``session_id_key``.
    They are still accepted with a ``DeprecationWarning`` and forwarded
    to the new names. Passing both the old and new name for the same
    knob raises ``TypeError``. The shim will be removed in 0.4.0.
    """
    if conversation_id_from_config is not None:
        if session_id_from_config is not None:
            raise TypeError(
                "make_context_node() received both 'session_id_from_config' "
                "and the deprecated 'conversation_id_from_config'; pass only "
                "the new name."
            )
        warnings.warn(
            "make_context_node(conversation_id_from_config=) is deprecated; "
            "rename to 'session_id_from_config'. The old name will be "
            "removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        session_id_from_config = conversation_id_from_config
    if conversation_id_key is not None:
        if session_id_key is not None:
            raise TypeError(
                "make_context_node() received both 'session_id_key' and "
                "the deprecated 'conversation_id_key'; pass only the new name."
            )
        warnings.warn(
            "make_context_node(conversation_id_key=) is deprecated; rename "
            "to 'session_id_key'. The old name will be removed in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        session_id_key = conversation_id_key

    if session_id_from_config is None:
        session_id_from_config = True
    if session_id_key is None:
        session_id_key = "thread_id"

    async def context_node(
        state: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        try:
            if session_id_from_config:
                cfg = (config or {}).get("configurable") or {}
                session_id = cfg.get(session_id_key)
            else:
                session_id = state.get(session_id_key)
            owner_id = state.get(owner_id_key)
            if not session_id or not owner_id:
                logger.debug(
                    "context_node: missing session_id/owner_id "
                    "(session_id=%r, owner_id=%r); no-op",
                    session_id, owner_id,
                )
                return {}

            # Beacon's wire-level GET still uses the conversation_id= kwarg name;
            # under the unified model it's the same string as session_id.
            payload = await client.get_context(
                conversation_id=session_id,
                owner_id=owner_id,
                agent_id=agent_id,
                owner_type=owner_type,
            )
            return _payload_to_delta(
                payload,
                into=into,
                header=header,
                message_id=message_id,
                dedupe=dedupe,
                rank_urgent=rank_urgent,
                urgent_types=urgent_types,
            )
        except Exception as exc:
            _log_swallowed("context_node", exc)
            return {}

    return context_node


__all__ = ["make_context_node"]
