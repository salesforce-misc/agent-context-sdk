# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for the OpenAI RecordTransformer."""
from __future__ import annotations

import json

import pytest

from agentic_context_openai.transformer import RecordTransformer


@pytest.fixture
def transformer() -> RecordTransformer:
    return RecordTransformer(agent_id="agent-support", participant_id="user-001")


# ─── Single entry → single record ─────────────────────────────────
# Covers both wire formats: Chat Completions messages and Responses API
# items (the RunResult.to_input_list() shape). Every emission must also
# satisfy the Record contract rules: tool payloads ride in ``variable``,
# not ``content`` (rule 2), and the timestamp is timezone-aware UTC (rule 3).

@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (
            {"role": "user", "content": "Hello!"},
            {"event_type": "user_message", "content": "Hello!", "category": "Message",
             "participant_id": "user-001", "session_id": "ses-001"},
        ),
        # Multimodal input: content is a list of parts; only text survives.
        (
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "..."}},
                {"type": "text", "text": " there"},
            ]},
            {"content": "Hello there"},
        ),
        (
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"event_type": "agent_response", "content": "Hi, how can I help?",
             "category": "Message", "participant_id": "agent-support"},
        ),
        (
            {"role": "assistant", "tool_calls": [{
                "id": "call_abc", "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "SF"})},
            }]},
            {"event_type": "tool_call", "category": "Trace", "event_id": "call_abc",
             "content": None,
             "variable": {"Name": "get_weather", "InputValueText": json.dumps({"city": "SF"})}},
        ),
        (
            {"type": "function_call", "name": "get_weather", "call_id": "fc_1",
             "arguments": json.dumps({"city": "SF"})},
            {"event_type": "tool_call", "event_id": "fc_1", "content": None,
             "variable": {"Name": "get_weather", "InputValueText": json.dumps({"city": "SF"})}},
        ),
        # Tool outputs can be structured — JSON-encode so the Beacon
        # ``OutputValueText`` is always a string (Record contract rule §2).
        (
            {"type": "function_call_output", "call_id": "fc_1",
             "output": {"status": "delivered", "eta": None}},
            {"variable": {"Name": "", "OutputValueText": json.dumps({"status": "delivered", "eta": None})}},
        ),
        (
            {"type": "reasoning", "content": "The user wants weather in SF."},
            {"event_type": "agent_reasoning", "category": "Trace", "content": None,
             "variable": {"OutputValueText": "The user wants weather in SF."}},
        ),
        # Current Responses API reasoning shape — ``summary`` is a list of
        # ``summary_text`` parts, not a plain string. Must emit a record, not
        # silently drop it.
        (
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "The user wants weather"},
                {"type": "summary_text", "text": " in SF."},
            ]},
            {"event_type": "agent_reasoning", "category": "Trace",
             "variable": {"OutputValueText": "The user wants weather in SF."}},
        ),
        # ``{"type": "message", "role": "assistant", "content": [...]}`` is
        # just a role-keyed message with a type tag — falls through to the
        # role branch.
        (
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Hi there."}]},
            {"event_type": "agent_response", "content": "Hi there."},
        ),
    ],
    ids=[
        "user-string", "user-content-parts", "assistant-text", "assistant-tool-call",
        "responses-function-call", "function-call-output-jsonified", "reasoning-content",
        "reasoning-summary-text", "responses-message-item",
    ],
)
def test_single_entry_emits_one_record(transformer, entry, expected) -> None:
    records = transformer.transform([entry], "ses-001")
    assert len(records) == 1
    r = records[0]
    for field, value in expected.items():
        assert getattr(r, field) == value, field
    assert r.timestamp.tzinfo is not None


@pytest.mark.parametrize(
    "entry",
    [
        # A content-less user message is noise — never emitted.
        {"role": "user", "content": ""},
        # System prompt is operator scaffolding, not a transcript event.
        {"role": "system", "content": "You are helpful."},
    ],
    ids=["empty-user-message", "system-message"],
)
def test_entry_is_dropped(transformer, entry) -> None:
    assert transformer.transform([entry], "ses-001") == []


# ─── Multi-entry sequences ────────────────────────────────────────

@pytest.mark.parametrize(
    ("entries", "event_types", "last_event_id", "last_variable"),
    [
        # A single assistant turn with content + tool_calls → one record
        # each, text first (chronologically it is the pre-tool reply).
        (
            [{"role": "assistant", "content": "Let me look that up.", "tool_calls": [{
                "id": "call_1", "function": {"name": "lookup", "arguments": "{}"},
            }]}],
            ["agent_response", "tool_call"], None, None,
        ),
        # ``{"role": "tool"}`` carries only tool_call_id, so Name must come
        # from the preceding assistant tool_calls entry (shared state across
        # entries). Mirrors the ClaudeAdapter convention.
        (
            [
                {"role": "assistant", "tool_calls": [{
                    "id": "call_42", "function": {"name": "lookup_order", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "call_42", "content": '{"status": "delivered"}'},
            ],
            ["tool_call", "tool_result"], "call_42-out",
            {"Name": "lookup_order", "OutputValueText": '{"status": "delivered"}'},
        ),
        # The Responses API ``function_call_output`` carries no name — the
        # transformer must remember the name from the paired ``function_call``.
        (
            [
                {"type": "function_call", "name": "ship_lookup", "call_id": "fc_42", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "fc_42", "output": '{"status": "delivered"}'},
            ],
            ["tool_call", "tool_result"], None,
            {"Name": "ship_lookup", "OutputValueText": '{"status": "delivered"}'},
        ),
        # One realistic conversation emits the full event_type sequence.
        (
            [
                {"role": "user", "content": "What's my order status?"},
                {"role": "assistant", "content": "Looking that up now.", "tool_calls": [{
                    "id": "call_1", "function": {"name": "get_order", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "delivered"}'},
                {"role": "assistant", "content": "Your order was delivered."},
            ],
            ["user_message", "agent_response", "tool_call", "tool_result", "agent_response"],
            None, None,
        ),
    ],
    ids=["text-and-tool-calls", "tool-reply-resolves-name", "function-call-output-resolves-name",
         "full-tool-using-conversation"],
)
def test_sequence(transformer, entries, event_types, last_event_id, last_variable) -> None:
    records = transformer.transform(entries, "ses-001")
    assert [r.event_type for r in records] == event_types
    if last_event_id is not None:
        assert records[-1].event_id == last_event_id
    if last_variable is not None:
        assert records[-1].variable == last_variable


# ─── event_id ─────────────────────────────────────────────────────

def test_event_ids_are_stable_across_rereads(transformer) -> None:
    """Record rule 1 — ``event_id`` is stable across re-reads. A re-read (same or
    fresh transformer) must produce the SAME local ids."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    first = [r.event_id for r in transformer.transform(messages, "ses-X")]
    repush = [r.event_id for r in transformer.transform(messages, "ses-X")]
    fresh = [
        r.event_id
        for r in RecordTransformer("agent-support", "user-001").transform(messages, "ses-X")
    ]
    assert first == repush == fresh
    assert len(set(first)) == 2


def test_fallback_event_id_derivation() -> None:
    """When OpenAI gives us no native id, the fallback id is derived from
    session_id + turn + position + type-suffix. Streaming callers pass a
    per-turn interaction_id, so slices from different turns must not collide
    even though each call resets idx; session_id is part of the id so two
    sessions never collide even though both calls start at idx=0."""
    t = RecordTransformer(agent_id="a", participant_id="o")
    ids = [
        t.transform([{"role": "user", "content": c}], sid, interaction_id=iid)[0].event_id
        for c, sid, iid in [
            ("hi", "ses-stable", None),
            ("first", "ses-stream", "turn-1"),
            ("second", "ses-stream", "turn-2"),
            ("hi", "ses-A", None),
            ("hi", "ses-B", None),
        ]
    ]
    assert ids == [
        "msg-ses-stable-snap-0-u",
        "msg-ses-stream-turn-1-0-u",
        "msg-ses-stream-turn-2-0-u",
        "msg-ses-A-snap-0-u",
        "msg-ses-B-snap-0-u",
    ]


@pytest.mark.parametrize(
    ("entries", "expected_ids"),
    [
        (
            [
                {"type": "message", "role": "assistant", "id": "__fake_id__",
                 "content": [{"type": "output_text", "text": "first"}]},
                {"type": "message", "role": "assistant", "id": "__fake_id__",
                 "content": [{"type": "output_text", "text": "second"}]},
            ],
            ["msg-ses-fake-snap-0-a", "msg-ses-fake-snap-1-a"],
        ),
        (
            [
                {"type": "function_call", "name": "a", "id": "__fake_id__", "arguments": "{}"},
                {"type": "function_call", "name": "b", "id": "__fake_id__", "arguments": "{}"},
            ],
            ["msg-ses-fake-snap-0-fc", "msg-ses-fake-snap-1-fc"],
        ),
        # call_id wins when real; a bare __fake_id__ gets the slot fallback.
        (
            [{"type": "function_call", "name": "a", "id": "__fake_id__",
              "call_id": "call_real", "arguments": "{}"}],
            ["call_real"],
        ),
        (
            [{"role": "assistant", "tool_calls": [
                {"id": "__fake_id__", "function": {"name": "a", "arguments": "{}"}},
                {"id": "__fake_id__", "function": {"name": "b", "arguments": "{}"}},
            ]}],
            ["msg-ses-fake-snap-0-tc", "msg-ses-fake-snap-1-tc"],
        ),
    ],
    ids=["responses-message", "function-call", "real-call-id-wins", "chat-completions-tool-calls"],
)
def test_fake_id_placeholder_falls_back_to_slot_id(transformer, entries, expected_ids) -> None:
    """The Agents SDK Chat Completions converter stamps every synthesised item
    with id="__fake_id__". Treating that as a stable id would give every
    item the same event_id."""
    ids = [r.event_id for r in transformer.transform(entries, "ses-fake")]
    assert ids == expected_ids
    assert len(set(ids)) == len(ids)


# ─── interaction_id / owner ───────────────────────────────────────

@pytest.mark.parametrize("interaction_id", ["turn-ses-i-abc123", None], ids=["stamped", "omitted"])
def test_interaction_id_threading(interaction_id) -> None:
    """A logical turn's records (user_message / agent_response / reasoning /
    tool_call / tool_result) must share one interaction_id so Beacon can
    group them; without one the field stays None."""
    t = RecordTransformer(agent_id="a", participant_id="o")
    records = t.transform(
        [
            {"role": "user", "content": "run the tool"},
            {"role": "assistant", "content": "doing it"},
        ],
        "ses-i",
        interaction_id=interaction_id,
    )
    assert len(records) == 2
    assert all(r.interaction_id == interaction_id for r in records)


def test_owner_is_per_call() -> None:
    """Owner is per-call so one shared transformer serves many users: an
    explicit owner_id overrides the constructor value without leaking into
    later calls, and an omitted one falls back to the constructor."""
    t = RecordTransformer(agent_id="a", participant_id="fallback")
    owners = [
        t.transform([{"role": "user", "content": "hi"}], sid, owner_id=owner)[0].participant_id
        for sid, owner in [("ses-a", "alice@corp.com"), ("ses-b", "bob"), ("ses-c", None)]
    ]
    assert owners == ["alice@corp.com", "bob", "fallback"]
