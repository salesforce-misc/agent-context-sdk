# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unit tests for retrieval_normalize.py — pure helpers, no I/O."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_context_langgraph.retrieval_normalize import (
    context_to_system_message,
    conversation_to_messages,
    format_context_block,
)


# ─── format_context_block ────────────────────────────────────────────


def test_format_empty_payload() -> None:
    assert format_context_block(None) == ""
    assert format_context_block({}) == ""
    assert format_context_block({"summary": "", "userContext": []}) == ""


@pytest.mark.parametrize(
    "payload, present, absent",
    [
        (
            {"summary": "User prefers email follow-ups."},
            ["In-session summary:", "User prefers email follow-ups."],
            ["- "],  # no facts → no bullets
        ),
        (
            {
                "userContext": [
                    {"text": "Owns Pro plan", "type": "account_info", "confidence": 0.91},
                    {"text": "Last issue: refund", "type": "past_issue", "confidence": 0.83},
                ]
            },
            [
                "Previous context about this customer",
                "- Owns Pro plan (type=account_info, confidence=0.91)",
                "- Last issue: refund (type=past_issue, confidence=0.83)",
            ],
            ["In-session summary:"],
        ),
    ],
    ids=["only-summary", "only-facts"],
)
def test_format_single_section(payload, present, absent) -> None:
    out = format_context_block(payload)
    for text in present:
        assert text in out
    for text in absent:
        assert text not in out


def test_format_summary_and_facts() -> None:
    out = format_context_block({
        "summary": "User prefers email.",
        "userContext": [{"text": "Owns Pro plan"}],
    })
    assert "- Owns Pro plan" in out
    assert "In-session summary:\nUser prefers email." in out
    # Section ordering: summary must come before the facts header.
    assert out.index("In-session summary:") < out.index("Owns Pro plan")


def test_format_section_order_summary_then_conversation_then_facts() -> None:
    """All three sections present — summary first, conversation second, facts last."""
    out = format_context_block({
        "summary": "in-session note",
        "conversation_summary": "longer thread summary",
        "userContext": [{"text": "a fact about the user"}],
    })
    i_summary = out.index("In-session summary:")
    i_conv = out.index("Conversation summary:")
    i_facts = out.index("a fact about the user")
    assert i_summary < i_conv < i_facts


def test_format_tolerates_malformed_entries() -> None:
    """Unknown keys, wrong types, and missing fields should not raise."""
    out = format_context_block({
        "summary": 42,  # non-string summary
        "userContext": [
            "not a dict",  # wrong type
            {"text": ""},  # empty text → skipped
            {"content": "alt-text key works"},  # 'content' fallback
            {"text": "no annotations"},  # no type/confidence
            {"text": "weird conf", "confidence": "high"},  # non-numeric conf
            {"junk": "no text"},  # no text → skipped
        ],
    })
    assert "alt-text key works" in out
    assert "- no annotations" in out
    assert "weird conf" in out and "confidence=" not in out.split("weird conf")[1].split("\n")[0]
    # Numeric coercion of summary still surfaces it.
    assert "42" in out


def test_format_custom_header() -> None:
    out = format_context_block(
        {"userContext": [{"text": "X"}]},
        header="Customer notes",
    )
    assert out.startswith("Customer notes")


# ─── context_to_system_message ───────────────────────────────────────


def test_system_message_none_for_empty() -> None:
    assert context_to_system_message(None) is None
    assert context_to_system_message({}) is None
    assert context_to_system_message({"summary": "", "userContext": []}) is None


def test_system_message_round_trips() -> None:
    msg = context_to_system_message({
        "summary": "S",
        "userContext": [{"text": "F1"}],
    })
    assert isinstance(msg, SystemMessage)
    assert "F1" in msg.content
    assert "In-session summary:\nS" in msg.content


def test_system_message_has_stable_id_for_dedup() -> None:
    """add_messages dedupes on id, so the SystemMessage MUST carry a stable id —
    otherwise multi-turn graphs accumulate one context block per turn."""
    from langgraph.graph.message import add_messages

    p1 = {"summary": "v1"}
    p2 = {"summary": "v2"}
    msg1 = context_to_system_message(p1)
    msg2 = context_to_system_message(p2)
    assert msg1 is not None and msg2 is not None
    assert msg1.id == msg2.id == "agentic-memory-context"

    combined = add_messages([msg1], [msg2])
    assert len(combined) == 1
    assert "v2" in combined[0].content


def test_system_message_id_can_be_disabled_or_overridden() -> None:
    disabled = context_to_system_message({"summary": "x"}, message_id=None)
    assert disabled is not None
    # LangChain auto-generates an id; just confirm it's not our stable one.
    assert disabled.id != "agentic-memory-context"

    custom = context_to_system_message({"summary": "x"}, message_id="custom-key")
    assert custom is not None
    assert custom.id == "custom-key"


# ─── conversation_to_messages ────────────────────────────────────────


def test_conv_empty() -> None:
    assert conversation_to_messages(None) == []
    assert conversation_to_messages({}) == []
    assert conversation_to_messages({"sessions": []}) == []
    assert conversation_to_messages({"sessions": [{"entries": []}]}) == []


def test_conv_simple_two_turn() -> None:
    payload = {
        "sessions": [
            {
                "entries": [
                    {"type": "user_message", "content": "hi"},
                    {"type": "agent_response", "content": "hello"},
                    {"type": "user_message", "content": "bye"},
                    {"type": "agent_response", "content": "goodbye"},
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert len(msgs) == 4
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "hi"
    assert isinstance(msgs[1], AIMessage) and msgs[1].content == "hello"
    assert isinstance(msgs[2], HumanMessage) and msgs[2].content == "bye"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "goodbye"


def test_conv_tool_call_and_result() -> None:
    payload = {
        "sessions": [
            {
                "entries": [
                    {"type": "user_message", "content": "what's the weather?"},
                    {
                        "type": "tool_call",
                        "id": "step-tool-call-1",
                        "variable": {
                            "Name": "get_weather",
                            "InputValueText": '{"city": "SF"}',
                        },
                        "metadata": {"tool_call_id": "tc-1"},
                    },
                    {
                        "type": "tool_result",
                        "variable": {"Name": "get_weather", "OutputValueText": "72F"},
                        "metadata": {"tool_call_id": "tc-1"},
                    },
                    {"type": "agent_response", "content": "It's 72F."},
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert len(msgs) == 4

    assert isinstance(msgs[0], HumanMessage)

    assert isinstance(msgs[1], AIMessage)
    assert len(msgs[1].tool_calls) == 1
    tc = msgs[1].tool_calls[0]
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "SF"}
    assert tc["id"] == "tc-1"

    assert isinstance(msgs[2], ToolMessage)
    assert msgs[2].tool_call_id == "tc-1"
    assert msgs[2].name == "get_weather"
    assert msgs[2].content == "72F"

    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "It's 72F."


def test_conv_parallel_tool_calls_collapse() -> None:
    """Two adjacent tool_call entries collapse into one AIMessage."""
    payload = {
        "sessions": [
            {
                "entries": [
                    {
                        "type": "tool_call",
                        "variable": {"Name": "a", "InputValueText": "{}"},
                        "metadata": {"tool_call_id": "tc-a"},
                    },
                    {
                        "type": "tool_call",
                        "variable": {"Name": "b", "InputValueText": "{}"},
                        "metadata": {"tool_call_id": "tc-b"},
                    },
                    {
                        "type": "tool_result",
                        "variable": {"Name": "a", "OutputValueText": "ra"},
                        "metadata": {"tool_call_id": "tc-a"},
                    },
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    # AIMessage(tool_calls=[a, b]), ToolMessage(a)
    assert len(msgs) == 2
    ai = msgs[0]
    assert isinstance(ai, AIMessage)
    assert [tc["name"] for tc in ai.tool_calls] == ["a", "b"]
    assert isinstance(msgs[1], ToolMessage) and msgs[1].tool_call_id == "tc-a"


def test_conv_orphan_tool_result_dropped() -> None:
    """tool_result without tool_call_id can't form a valid ToolMessage."""
    payload = {
        "sessions": [
            {
                "entries": [
                    {"type": "tool_result", "variable": {"OutputValueText": "x"}},
                ]
            }
        ]
    }
    assert conversation_to_messages(payload) == []


def test_conv_reasoning_folds_into_next_ai() -> None:
    payload = {
        "sessions": [
            {
                "entries": [
                    {"type": "agent_reasoning", "variable": {"OutputValueText": "thinking..."}},
                    {"type": "agent_response", "content": "answer"},
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert "[reasoning] thinking..." in msgs[0].content
    assert "answer" in msgs[0].content


def test_conv_reasoning_before_tool_call_attaches_to_ai() -> None:
    payload = {
        "sessions": [
            {
                "entries": [
                    {"type": "agent_reasoning", "variable": {"OutputValueText": "let me check"}},
                    {
                        "type": "tool_call",
                        "variable": {"Name": "lookup", "InputValueText": "{}"},
                        "metadata": {"tool_call_id": "tc-z"},
                    },
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert len(msgs) == 1
    ai = msgs[0]
    assert isinstance(ai, AIMessage)
    assert "[reasoning] let me check" in ai.content
    assert ai.tool_calls[0]["name"] == "lookup"


def test_conv_multi_session_flattens_in_order() -> None:
    payload = {
        "sessions": [
            {"entries": [{"type": "user_message", "content": "s1-u"}]},
            {"entries": [{"type": "user_message", "content": "s2-u"}]},
        ]
    }
    msgs = conversation_to_messages(payload)
    assert [m.content for m in msgs] == ["s1-u", "s2-u"]


def test_conv_unknown_type_skipped() -> None:
    payload = {
        "sessions": [
            {
                "entries": [
                    {"type": "wat", "content": "??"},
                    {"type": "user_message", "content": "ok"},
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert len(msgs) == 1
    assert msgs[0].content == "ok"


@pytest.mark.parametrize(
    "input_value, expected_args",
    [
        # InputValueText may already be a dict — don't re-decode it.
        ({"x": 1}, {"x": 1}),
        ("not json", {"input": "not json"}),
    ],
    ids=["dict-passthrough", "unparseable-string-wrapped"],
)
def test_conv_inputvalue_decoding(input_value, expected_args) -> None:
    payload = {
        "sessions": [
            {
                "entries": [
                    {
                        "type": "tool_call",
                        "variable": {"Name": "f", "InputValueText": input_value},
                        "metadata": {"tool_call_id": "tc-1"},
                    },
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert msgs[0].tool_calls[0]["args"] == expected_args


def test_conv_legacy_variable_keys_still_rehydrate() -> None:
    """Payloads ingested before the InputValueText rename still rehydrate."""
    payload = {
        "sessions": [
            {
                "entries": [
                    {
                        "type": "tool_call",
                        "variable": {"Name": "get_weather", "InputValue": '{"city": "SF"}'},
                        "metadata": {"tool_call_id": "tc-1"},
                    },
                    {
                        "type": "tool_result",
                        "variable": {"Name": "get_weather", "OutputValue": "72F"},
                        "metadata": {"tool_call_id": "tc-1"},
                    },
                    {
                        "type": "agent_reasoning",
                        "variable": {"OutputValue": "thinking..."},
                    },
                    {"type": "agent_response", "content": "It's 72F."},
                ]
            }
        ]
    }
    msgs = conversation_to_messages(payload)
    assert len(msgs) == 3
    assert msgs[0].tool_calls[0]["args"] == {"city": "SF"}
    assert msgs[1].content == "72F"
    assert "[reasoning] thinking..." in msgs[2].content
    assert "It's 72F." in msgs[2].content
