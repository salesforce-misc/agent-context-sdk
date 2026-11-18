# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from agentic_context import Capabilities, ContextSource, Record


def test_record_round_trip(record: Record) -> None:
    dumped = record.model_dump(mode="json")
    rebuilt = Record.model_validate(dumped)
    assert rebuilt == record


def test_record_rejects_unknown_field(now: datetime) -> None:
    with pytest.raises(ValidationError):
        Record(
            event_id="evt-1",
            session_id="ses-1",
            agent_id="agent-1",
            event_type="user_message",
            category="Message",
            content="x",
            timestamp=now,
            extra_field="nope",  # type: ignore[call-arg]
        )


def test_record_is_immutable(record: Record) -> None:
    with pytest.raises(ValidationError):
        record.content = "new"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "field", "expected"),
    [
        ({"event_type": "agent_reasoning", "category": "Trace",
          "variable": {"OutputValueText": "thinking..."}}, "content", None),
        ({"event_type": "user_message", "category": "Message", "content": "hello"},
         "interaction_id", None),
        ({"event_type": "user_message", "content": "hi"}, "category", "Message"),
    ],
    ids=["content-optional", "interaction-id-optional", "category-defaults-to-message"],
)
def test_optional_field_defaults(now: datetime, kwargs: dict, field: str, expected) -> None:
    record = Record(event_id="evt-2", session_id="ses-1", agent_id="agent-1", timestamp=now, **kwargs)
    assert getattr(record, field) == expected


def test_variable_for_trace(now: datetime) -> None:
    record = Record(
        event_id="step-1",
        session_id="ses-1",
        agent_id="agent-1",
        event_type="tool_call",
        category="Trace",
        participant_id="agent-1",
        variable={"Name": "OrderLookup", "InputValueText": '{"q": "test"}'},
        timestamp=now,
    )
    assert record.variable is not None
    assert record.variable["Name"] == "OrderLookup"
    assert record.content is None


def test_event_type_values(now: datetime) -> None:
    """All five valid event types from Design.md §2.2 can be set."""
    valid_types = ["user_message", "agent_response", "agent_reasoning", "tool_call", "tool_result"]
    for etype in valid_types:
        r = Record(
            event_id=f"evt-{etype}",
            session_id="ses-1",
            agent_id="agent-1",
            event_type=etype,
            category="Trace" if etype in ("agent_reasoning", "tool_call", "tool_result") else "Message",
            timestamp=now,
        )
        assert r.event_type == etype



def test_protocol_satisfied_runtime(working_caps: Capabilities) -> None:
    class Adapter:
        agent_id = "agent-x"
        capabilities = working_caps

        def snapshot(self, session_id: str) -> list[Record]:
            return []

    assert isinstance(Adapter(), ContextSource)


def test_protocol_rejects_missing_method(working_caps: Capabilities) -> None:
    class NotAdapter:
        agent_id = "agent-x"
        capabilities = working_caps

    assert not isinstance(NotAdapter(), ContextSource)


def test_capabilities_default_to_false() -> None:
    caps = Capabilities()
    assert caps.working is False
    assert caps.persistent is False


# ── timestamp normalization (contract: tz-aware UTC) ──────────────────────────


@pytest.mark.parametrize(
    "given",
    [
        # A naive datetime is assumed to be UTC and made tz-aware.
        datetime(2026, 1, 1, 12, 0, 0),
        # A tz-aware non-UTC datetime is converted to UTC, preserving the instant.
        datetime(2026, 1, 1, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5))),
        datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ],
    ids=["naive-stamped-utc", "offset-converted-to-utc", "utc-passes-through"],
)
def test_timestamp_normalized_to_utc(given: datetime) -> None:
    """The Record contract requires tz-aware UTC. An adapter
    that hands in ``datetime.now()`` (naive) must not silently produce a record
    the server timestamps incorrectly.
    """
    record = Record(
        event_id="ts-1", session_id="ses-1", agent_id="agent-1",
        event_type="user_message", content="hi", timestamp=given,
    )
    assert record.timestamp.tzinfo is timezone.utc
    assert record.timestamp.hour == 12
    assert record.timestamp == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_timestamp_normalized_before_json_round_trip() -> None:
    """Normalization survives the JSON round trip the wire format uses."""
    naive = datetime(2026, 1, 1, 12, 0, 0)
    record = Record(
        event_id="ts-4", session_id="ses-1", agent_id="agent-1",
        event_type="user_message", content="hi", timestamp=naive,
    )
    rebuilt = Record.model_validate(record.model_dump(mode="json"))
    assert rebuilt.timestamp == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert rebuilt == record


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc), "2026-01-01T12:00:00.123Z"),
        (datetime(2026, 1, 1, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5))), "2026-01-01T12:00:00.000Z"),
    ],
    ids=["utc-zulu-millis", "offset-converted-to-zulu"],
)
def test_to_ace_timestamp(given: datetime, expected: str) -> None:
    """Wire stamps must match TransformServiceGrpcModelConverter.formatInstant."""
    from agentic_context.models import to_ace_timestamp

    assert to_ace_timestamp(given) == expected
