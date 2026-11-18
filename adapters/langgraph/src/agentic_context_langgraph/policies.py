# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Context-fetch policies for the self-driving LangGraph adapter.

A policy decides, on every ``stream()`` call, whether the adapter should
call ``GET /agentic/context`` this turn. The default
:class:`SessionFirstPolicy` fetches once per session and relies on the
LangGraph ``add_messages`` reducer to keep the injected ``SystemMessage``
in state across subsequent turns of the same thread.

The smart adapter selects a policy through the ``context_mode`` facade
(``auto`` → :class:`SessionFirstPolicy`, ``per-turn`` →
:class:`EveryTurnPolicy`, ``manual`` → :class:`NoFetchPolicy`). For a fetch
cadence the three modes don't express — :class:`TimeDecayPolicy`,
:class:`TopicShiftPolicy`, or a custom :class:`ContextPolicy` — resolve one
with :func:`resolve_policy` and wire it into your own retrieval node via
``make_context_node``; the adapter constructor does not take a policy kwarg.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ContextPolicy(Protocol):
    """Decides whether the adapter should re-fetch context on this turn.

    Implementations are pure with respect to their own state — the adapter
    feeds back what it knows about the conversation each turn.
    """

    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict[str, Any] | None,
        last_fetched_at: datetime | None,
    ) -> bool: ...


class SessionFirstPolicy:
    """Fetch on the first turn of a session, then never again.

    Cheapest policy. Relies on ``add_messages`` keeping the injected
    SystemMessage in state for the rest of the thread. Covers the
    day-zero CRM, multi-session, and cross-topic scenarios at one GET
    per session.
    """

    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict[str, Any] | None,
        last_fetched_at: datetime | None,
    ) -> bool:
        return last_fetched_at is None


class EveryTurnPolicy:
    """Fetch before every user turn.

    Highest fidelity, +1 GET per turn. The injected SystemMessage uses
    the stable id ``"agentic-memory-context"`` so the ``add_messages``
    reducer replaces (rather than duplicates) the prior block.
    """

    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict[str, Any] | None,
        last_fetched_at: datetime | None,
    ) -> bool:
        return True


class TimeDecayPolicy:
    """Cache the fetched payload for ``ttl_seconds`` and re-fetch when stale.

    Re-fetch fires on the first turn after the TTL elapses; intermediate
    turns reuse the in-state SystemMessage.
    """

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self.ttl_seconds = ttl_seconds

    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict[str, Any] | None,
        last_fetched_at: datetime | None,
    ) -> bool:
        if last_fetched_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last_fetched_at).total_seconds()
        return elapsed >= self.ttl_seconds


class TopicShiftPolicy:
    """Re-fetch when the conversation pivots to a new topic.

    Embeds the new user message and the cached payload's ``summary`` and
    re-fetches when cosine similarity falls below ``threshold``.
    Falls back to "fetch on first only" semantics whenever an embedder
    or required state is unavailable, so a missing model never causes
    a thrashing loop of GETs.
    """

    def __init__(
        self,
        embedder: Callable[[str], list[float]] | None = None,
        threshold: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold

    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict[str, Any] | None,
        last_fetched_at: datetime | None,
    ) -> bool:
        if last_fetched_at is None:
            return True
        if self.embedder is None or last_user_text is None:
            return False
        prior_summary = ""
        if isinstance(last_payload, dict):
            prior_summary = str(last_payload.get("summary") or "")
        if not prior_summary:
            return False
        try:
            new_vec = self.embedder(last_user_text)
            old_vec = self.embedder(prior_summary)
        except Exception as e:
            logger.warning(
                "TopicShiftPolicy.should_fetch: embedder failed (%s: %s); treating as no-shift",
                type(e).__name__, e,
            )
            return False
        sim = _cosine(new_vec, old_vec)
        return sim < self.threshold


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Zero when either vector is degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class NoFetchPolicy:
    """Never fetch. The adapter injects no context; retrieval is entirely
    customer-driven.

    Backs ``context_mode="manual"``: the ``recall_customer_context`` tool stays
    installed but the adapter seeds nothing and injects no prompt — the
    customer's own graph-node prompting drives every read. (``per-turn`` seeds
    fast context every turn via :class:`EveryTurnPolicy`, so it does *not* use
    this policy.)
    """

    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict[str, Any] | None,
        last_fetched_at: datetime | None,
    ) -> bool:
        return False


_POLICY_REGISTRY: dict[str, type[ContextPolicy]] = {
    "session_first": SessionFirstPolicy,
    "every_turn": EveryTurnPolicy,
    "time_decay": TimeDecayPolicy,
    "topic_shift": TopicShiftPolicy,
    "no_fetch": NoFetchPolicy,
}


def resolve_policy(policy: ContextPolicy | str | None) -> ContextPolicy:
    """Normalize a policy slug / instance / ``None`` to a :class:`ContextPolicy`.

    Public helper — the escape hatch for fetch cadences the ``context_mode``
    facade doesn't express. Resolve a policy here and wire it into your own
    retrieval node via ``make_context_node``. Also validates a slug before
    construction (e.g. when accepting policy names from a config file).

    ``None`` defaults to :class:`SessionFirstPolicy`. A string slug looks up
    the registry and instantiates with no arguments — pass an instance if
    the policy needs configuration. Any other value is returned as-is so
    third-party policies pass through unchanged.
    """
    if policy is None:
        return SessionFirstPolicy()
    if isinstance(policy, str):
        cls = _POLICY_REGISTRY.get(policy)
        if cls is None:
            valid = ", ".join(sorted(_POLICY_REGISTRY))
            raise ValueError(
                f"Unknown context_policy slug {policy!r}. Valid slugs: {valid}"
            )
        return cls()
    return policy


__all__ = [
    "ContextPolicy",
    "EveryTurnPolicy",
    "NoFetchPolicy",
    "SessionFirstPolicy",
    "TimeDecayPolicy",
    "TopicShiftPolicy",
    "resolve_policy",
]
