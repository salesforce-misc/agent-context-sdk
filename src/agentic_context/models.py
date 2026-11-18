# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Core data model and adapter contract.

Stable public surface — adapters and user code import from here.

Record fields map 1:1 to the Beacon wire format. No translation layer needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_ace_timestamp(value: datetime | None = None) -> str:
    """Format a UTC instant the way Beacon's ingest converter stamps ``dateTime``.

    ``TransformServiceGrpcModelConverter.formatInstant`` uses
    ``uuuu-MM-dd'T'HH:mm:ss.SSSX`` in UTC, e.g. ``2026-01-01T12:00:00.000Z``.
    Conversation and session events always take that server stamp and land in
    DLOs. Entries and participants currently copy the client string as-is.

    Python's default ``datetime.isoformat()`` emits ``+00:00`` and microseconds
    (``2026-01-01T12:00:00.123456+00:00``). The Data Cloud stream transform
    drops that, leaving ``datetime__c`` null and failing
    ``MISSING_REQUIRED_SCHEMA_FIELDS``. Millisecond precision with a ``Z``
    suffix is what the converter already writes for conversation/session.
    """
    dt = datetime.now(timezone.utc) if value is None else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Record(BaseModel):
    """Normalized event shape every adapter emits.

    Fields map directly to Beacon's entry wire format:
      event_id       -> "id"
      event_type     -> "type"
      participant_id -> "participantId"
      content        -> "content"
      content_type   -> "contentType"
      attachments    -> "attachments"
      category       -> "category"
      interaction_id -> "interactionId"
      variable       -> "variable"
      timestamp      -> "timestamp"

    ``event_id`` is local only and not sent (Beacon assigns entry ids), so
    re-sending the same record writes a duplicate entry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str                                       # Local id. Not sent on the wire.
    session_id: str                                     # Beacon session identifier
    agent_id: str                                       # Registered agent ID in Beacon
    event_type: str                                     # Beacon: "type". One of: user_message | agent_response | agent_reasoning | tool_call | tool_result
    participant_id: str | None = None                   # Beacon: "participantId". Who produced this event.
    content: str | None = None                          # Beacon: "content". Text for Message entries; null for Traces.
    category: str = "Message"                           # Beacon: "category". "Message" or "Trace".
    interaction_id: str | None = None                   # Beacon: "interactionId". Groups entries in one turn. Optional per API.
    variable: dict[str, Any] | None = None              # Beacon: "variable". For Traces: {Name?, InputValueText?, OutputValueText?}
    content_type: str | None = None                     # Beacon: "contentType". e.g. "markdown", "text". Message entries only.
    attachments: list[dict[str, Any]] | None = None     # Beacon: "attachments". [{contentType, contentUrl}]. Message entries only.
    timestamp: datetime                                 # Beacon: "timestamp". ISO 8601 UTC.
    metadata: dict[str, Any] = Field(default_factory=dict)  # Adapter-internal extras (not sent to Beacon)

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Normalize ``timestamp`` to timezone-aware UTC (contract rule §2.3).

        A naive datetime is assumed to be UTC and stamped as such; an aware
        datetime in another zone is converted to UTC. This guarantees the
        wire value always serializes via :func:`to_ace_timestamp` (millisecond
        UTC with a ``Z`` suffix) so Data Cloud can coerce ``dateTime``. Adapters
        should still pass UTC directly; this is the backstop, not the happy path.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)



class Capabilities(BaseModel):
    """Declares which memory tiers an adapter produces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    working: bool = False
    persistent: bool = False


@runtime_checkable
class ContextSource(Protocol):
    """The contract every framework adapter satisfies."""

    agent_id: str
    capabilities: Capabilities

    def snapshot(self, session_id: str) -> list[Record]:
        """Return all records for ``session_id`` in chronological order."""
        ...
