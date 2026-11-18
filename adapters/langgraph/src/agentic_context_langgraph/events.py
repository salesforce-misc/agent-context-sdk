# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Observation events emitted by ``LangGraphAdapter`` hooks.

Each event is an immutable, JSON-friendly snapshot of an adapter lifecycle
moment. Hook callables receive these as their sole argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class TurnEvent:
    """Fired when ``stream()`` is called — one per turn."""

    session_id: str
    inputs: Any
    turn_started_at: datetime


@dataclass(frozen=True)
class ToolEvent:
    """Fired at tool invocation start and end.

    ``output`` and ``ended_at`` are ``None`` on the start event; both are
    populated on the end event. ``args`` mirrors the tool's input dict when
    LangChain provides it; otherwise it falls back to ``{"input": <raw>}``.
    """

    session_id: str
    tool_name: str
    tool_call_id: str | None
    args: dict[str, Any] | None
    output: Any | None
    started_at: datetime
    ended_at: datetime | None


@dataclass(frozen=True)
class ErrorEvent:
    """Fired when the stream loop or a tool callback raises.

    ``phase`` distinguishes a graph-level error (``"stream"``), a
    tool-level error (``"tool"``), and an ingest problem (``"ingest"``): a
    turn cut short whose ingest failed (reported here rather than raised so
    it can't mask the error or early exit that ended the turn), or records
    produced after ``close()`` / ``end_session()`` already sent the turn.
    The original exception is attached so observers can re-raise, log, or
    attribute it.
    """

    session_id: str
    phase: Literal["stream", "tool", "ingest"]
    exception: BaseException
    occurred_at: datetime


__all__ = ["ErrorEvent", "ToolEvent", "TurnEvent"]
