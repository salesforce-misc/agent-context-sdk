# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for RecordTransformer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_context_claude.transformer import RecordTransformer, FILTERED_TOOLS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def transformer() -> RecordTransformer:
    return RecordTransformer(agent_id="agent-support", participant_id="user-001")


def _entry(kind: str, uuid: str, content, ts: str = "2026-06-08T20:00:00Z") -> dict:
    return {"type": kind, "uuid": uuid, "timestamp": ts, "message": {"role": kind, "content": content}}


def _check(record, expected: dict) -> None:
    for attr, want in expected.items():
        got = getattr(record, attr)
        if attr == "variable":
            assert got is not None
            for key, val in want.items():
                # Tool inputs are JSON-encoded; compare decoded so key order is irrelevant.
                assert (json.loads(got[key]) if isinstance(val, dict) else got[key]) == val, key
        else:
            assert got == want, attr


_TOOL_USE = {"type": "tool_use", "id": "toolu_001", "name": "lookup_order",
             "input": {"order_id": "ORD-1001"}}


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (
            [_entry("user", "usr-001", "Hello!")],
            [{"event_type": "user_message", "content": "Hello!", "category": "Message",
              "participant_id": "user-001", "event_id": "usr-001", "session_id": "ses-001"}],
        ),
        # A user turn can arrive as content blocks, not just a plain string
        # (multimodal input / normalized transcripts). Regression: only the
        # ``isinstance(content, str)`` branch handled user turns, so a
        # block-shaped user message was silently dropped. The first block reuses
        # the bare uuid so it stays stable.
        (
            [_entry("user", "usr-blocks", [{"type": "text", "text": "Hello from a block!"}])],
            [{"event_type": "user_message", "content": "Hello from a block!",
              "category": "Message", "participant_id": "user-001", "event_id": "usr-blocks"}],
        ),
        # Multiple text blocks in one user entry each become a record with a
        # distinct, stable event_id (bare uuid, then uuid-suffixed).
        (
            [_entry("user", "usr-multi", [{"type": "text", "text": "first"},
                                          {"type": "text", "text": "second"}])],
            [{"event_type": "user_message", "content": "first", "event_id": "usr-multi"},
             {"event_type": "user_message", "content": "second", "event_id": "usr-multi-1"}],
        ),
        (
            [_entry("assistant", "ast-001", [{"type": "text", "text": "Hi! How can I help?"}])],
            [{"event_type": "agent_response", "content": "Hi! How can I help?",
              "category": "Message", "participant_id": "agent-support"}],
        ),
        # tool_use becomes a Trace keyed on the tool_use id, payload in variable.
        (
            [_entry("assistant", "ast-002", [_TOOL_USE])],
            [{"event_type": "tool_call", "category": "Trace", "content": None,
              "event_id": "toolu_001",
              "variable": {"Name": "lookup_order", "InputValueText": {"order_id": "ORD-1001"}}}],
        ),
        (
            [_entry("assistant", "ast-003", [{"type": "tool_use", "id": "toolu_002",
                                              "name": "ToolSearch", "input": {"query": "test"}}])],
            [],
        ),
        # tool_result takes its Name from the tool_use seen earlier.
        (
            [
                _entry("assistant", "ast-004", [{"type": "tool_use", "id": "toolu_010",
                                                 "name": "check_status", "input": {"id": "123"}}]),
                _entry("user", "usr-005", [{"type": "tool_result", "tool_use_id": "toolu_010",
                                            "content": [{"type": "text", "text": '{"status":"shipped"}'}]}]),
            ],
            [{"event_type": "tool_call"},
             {"event_type": "tool_result",
              "variable": {"Name": "check_status", "OutputValueText": '{"status":"shipped"}'}}],
        ),
        (
            [_entry("assistant", "ast-005", [
                {"type": "thinking", "thinking": "I should look up the order first."},
                {**_TOOL_USE, "id": "toolu_020"},
            ])],
            [{"event_type": "agent_reasoning", "category": "Trace", "event_id": "ast-005-thinking",
              "variable": {"OutputValueText": "I should look up the order first."}},
             {"event_type": "tool_call"}],
        ),
    ],
    ids=["user-text", "user-content-blocks", "user-multiple-text-blocks", "assistant-text",
         "tool-use", "filtered-tool-skipped", "tool-result", "thinking"],
)
def test_entries_map_to_records(transformer: RecordTransformer, entries, expected) -> None:
    records = transformer.transform(entries, "ses-001")

    assert len(records) == len(expected)
    for record, want in zip(records, expected):
        _check(record, want)


class TestPerCallOwner:
    """Owner is per-call so one shared transformer serves many users."""

    def _user_entry(self) -> dict:
        return _entry("user", "usr-x", "Hi")

    @pytest.mark.parametrize(
        ("owner_kwargs", "expected"),
        [({"owner_id": "alice@corp.com"}, "alice@corp.com"), ({}, "fallback")],
        ids=["owner-overrides-constructor", "falls-back-to-constructor"],
    )
    def test_user_message_owner(self, owner_kwargs, expected) -> None:
        t = RecordTransformer(agent_id="agent-support", participant_id="fallback")
        records = t.transform([self._user_entry()], "ses-1", **owner_kwargs)
        assert records[0].participant_id == expected

    def test_two_users_share_one_transformer(self) -> None:
        # A single process-wide transformer must attribute each turn to the
        # owner passed for THAT turn — never leak the previous user's id.
        t = RecordTransformer(agent_id="agent-support")
        a = t.transform([self._user_entry()], "ses-a", owner_id="alice@corp.com")
        b = t.transform([self._user_entry()], "ses-b", owner_id="bob@corp.com")
        assert a[0].participant_id == "alice@corp.com"
        assert b[0].participant_id == "bob@corp.com"

    def test_agent_response_owner_unaffected(self) -> None:
        # owner_id stamps user messages only; agent turns stay agent_id.
        t = RecordTransformer(agent_id="agent-support")
        entry = _entry("assistant", "a-1", [{"type": "text", "text": "Hi!"}], "2026-06-08T20:00:01Z")
        records = t.transform([entry], "ses-1", owner_id="alice@corp.com")
        assert records[0].participant_id == "agent-support"


class TestFullTranscript:
    def test_sample_fixture(self, transformer: RecordTransformer) -> None:
        """End-to-end test with the sample transcript fixture."""
        entries: list[dict] = []
        with open(FIXTURES / "sample_transcript.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        records = transformer.transform(entries, "ses-001")

        # Expected: user_message + thinking + tool_call + tool_result + agent_response
        types = [r.event_type for r in records]
        assert types == [
            "user_message",
            "agent_reasoning",
            "tool_call",
            "tool_result",
            "agent_response",
        ]

        # Verify tool_call and tool_result have matching Name
        tool_call = next(r for r in records if r.event_type == "tool_call")
        tool_result = next(r for r in records if r.event_type == "tool_result")
        assert tool_call.variable["Name"] == "lookup_order"
        assert tool_result.variable["Name"] == "lookup_order"
