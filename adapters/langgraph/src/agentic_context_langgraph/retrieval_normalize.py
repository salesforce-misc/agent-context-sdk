# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Pure helpers that translate Beacon GET payloads into LangGraph-native shapes.

No I/O. No exceptions on shape drift — the read-side is best-effort, and a
malformed-but-mostly-fine payload should still surface what we can parse.

Two payload shapes are handled:

- ``GET /api/v1/agentic/context``
    {"summary": "...", "userContext": [{"text"|"content": ..., "type"?: ...,
    "confidence"?: ..., "timestamp"?: ...}, ...]}
  Rendered into a prompt-friendly string via ``format_context_block`` or
  wrapped in a ``SystemMessage`` via ``context_to_system_message``.

- ``GET /api/v1/agentic/conversations/{id}?scope=all``
    {"sessions": [{"entries": [{"type": "user_message"|..., "id"?: ...,
    "content"?: ..., "variable"?: {"Name"?, "InputValueText"?, "OutputValueText"?},
    "metadata"?: {"tool_call_id"?: ...}}, ...]}]}
  Rehydrated into ``list[BaseMessage]`` (HumanMessage / AIMessage with
  optional tool_calls / ToolMessage) by ``conversation_to_messages``.

The conversation rehydrator mirrors the inverse of the record-builders in
``adapter.py`` (``_record_from_human_message``, ``_record_from_tool_call``,
``_record_from_tool_message``).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


_DEFAULT_HEADER = "Previous context about this customer"
# Stable id so re-injecting the SystemMessage on every turn replaces the
# previous one via add_messages — without an id, LangChain treats each
# SystemMessage as new and the message list accumulates one copy per turn.
_DEFAULT_MESSAGE_ID = "agentic-memory-context"

# Default fact types treated as "urgent" when ``rank_urgent=True``. Exposed
# so callers can extend (e.g. add ``"compliance_flag"``) without forking
# the formatter.
DEFAULT_URGENT_TYPES: tuple[str, ...] = (
    "open_case",
    "sla_breach",
    "churn_risk",
    "escalation",
)


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _fact_text(entry: dict[str, Any]) -> str:
    """Pull the human-readable text out of a userContext entry.

    Beacon TSEs return one of three shapes for the fact body:

    - ``text``    — current SDK fixture
    - ``content`` — earlier preview shape
    - ``value``   — alternate spec shape
    - ``entry``   — what the production Context Service actually returns
      (``{"entry": "INTERACTIONS[..] ...", "sourceType": "Conversation"}``)
    """
    for key in ("text", "content", "value", "entry"):
        v = entry.get(key)
        # Accept non-None values that are either non-strings or non-empty strings
        if v is not None and (not isinstance(v, str) or v):
            return _coerce_str(v)
    return ""


def format_context_block(
    payload: dict[str, Any] | None,
    *,
    header: str = _DEFAULT_HEADER,
    dedupe: bool = False,
    rank_urgent: bool = False,
    urgent_types: tuple[str, ...] = DEFAULT_URGENT_TYPES,
) -> str:
    """Render a context payload as a deterministic prompt-friendly string.

    Returns ``""`` when the payload is empty or carries nothing renderable —
    callers can ``if block:`` to decide whether to inject anything.

    Security note: Beacon-returned ``text``/``type`` fields originate from
    user conversations and are interpolated verbatim. Treat the returned
    string as untrusted input when injecting into a system prompt — wrap
    it in your own delimiters or instruct the model to treat it as data
    if your threat model includes prompt injection.

    The exact format mirrors ``recall_customer_context`` in the customer-
    support agent so this helper is a drop-in for ``state["customer_context"]``.
    Summary sections come first to orient the model before the fact bullets::

        In-session summary:
        <summary>

        Conversation summary:
        <conversation_summary>

        Previous context about this customer (only relevant if it matches the
        current question):
        - <fact text> (type=<type>, confidence=<.2f>)
        - ...

    Parameters
    ----------
    dedupe:
        When ``True``, drop later occurrences of facts whose
        ``(text.lower().strip(), type)`` tuple has already been seen.
        Beacon can echo the same fact across working + persistent memory
        sources; opt in to render each fact at most once. Default off
        for backwards compatibility.
    rank_urgent:
        When ``True``, surface facts whose ``type`` is in ``urgent_types``
        before others, preserving original order within each group.
        Useful for day-zero CRM contexts (P1 case > location). Default
        off — the existing contract is "render in Beacon order".
    urgent_types:
        Override the set of fact types considered urgent.
    """
    if not payload:
        return ""

    summary = _coerce_str(payload.get("summary", "")).strip()
    conversation_summary = _coerce_str(
        payload.get("conversation_summary", "")
    ).strip()
    raw_facts = payload.get("userContext") or []
    if not isinstance(raw_facts, list):
        raw_facts = []

    # Pass 1 — coerce to (text, type, confidence) tuples and apply dedup +
    # ranking before stringifying so both knobs compose cleanly.
    parsed: list[tuple[str, Any, Any]] = []
    seen: set[tuple[str, Any]] = set()
    urgent_set = set(urgent_types or ())
    for entry in raw_facts:
        if not isinstance(entry, dict):
            continue
        text = _fact_text(entry).strip()
        if not text:
            continue
        ftype = entry.get("type")
        if dedupe:
            key = (text.lower(), ftype)
            if key in seen:
                continue
            seen.add(key)
        parsed.append((text, ftype, entry.get("confidence")))

    if rank_urgent and urgent_set:
        urgent = [p for p in parsed if p[1] in urgent_set]
        rest = [p for p in parsed if p[1] not in urgent_set]
        parsed = urgent + rest

    fact_lines: list[str] = []
    for text, ftype, conf in parsed:
        annotations: list[str] = []
        if ftype:
            annotations.append(f"type={ftype}")
        if isinstance(conf, (int, float)):
            annotations.append(f"confidence={float(conf):.2f}")
        if annotations:
            fact_lines.append(f"- {text} ({', '.join(annotations)})")
        else:
            fact_lines.append(f"- {text}")

    if not summary and not fact_lines and not conversation_summary:
        return ""

    # Section order: orient the model with summaries before facts.
    # 1) In-session summary, 2) Conversation summary, 3) Facts.
    parts: list[str] = []
    if summary:
        parts.append("In-session summary:")
        parts.append(summary)
    if conversation_summary:
        if parts:
            parts.append("")
        parts.append("Conversation summary:")
        parts.append(conversation_summary)
    if fact_lines:
        if parts:
            parts.append("")
        parts.append(
            f"{header} (only relevant if it matches the current question):"
        )
        parts.extend(fact_lines)
    return "\n".join(parts)


def context_to_system_message(
    payload: dict[str, Any] | None,
    *,
    header: str = _DEFAULT_HEADER,
    message_id: str | None = _DEFAULT_MESSAGE_ID,
    dedupe: bool = False,
    rank_urgent: bool = False,
    urgent_types: tuple[str, ...] = DEFAULT_URGENT_TYPES,
) -> SystemMessage | None:
    """Wrap ``format_context_block`` in a ``SystemMessage``.

    Returns ``None`` for empty payloads so the caller can skip injection
    cleanly (e.g. ``msg = context_to_system_message(p); if msg: ...``).

    The returned message carries a stable ``id`` (default
    ``"agentic-memory-context"``) so multi-turn graphs that re-inject the
    context every turn replace the prior copy via ``add_messages`` instead
    of accumulating duplicates. Pass ``message_id=None`` to opt out and get
    a fresh id per call (LangChain's default).

    ``dedupe``, ``rank_urgent``, and ``urgent_types`` pass through to
    :func:`format_context_block`.
    """
    block = format_context_block(
        payload,
        header=header,
        dedupe=dedupe,
        rank_urgent=rank_urgent,
        urgent_types=urgent_types,
    )
    if not block:
        return None
    if message_id is None:
        return SystemMessage(content=block)
    return SystemMessage(content=block, id=message_id)


# ─── Conversation rehydration ────────────────────────────────────────


def _entry_text(entry: dict[str, Any]) -> str:
    """Pull readable text from a Beacon entry.

    Message entries carry text in ``content``. Trace entries (tool_call /
    tool_result / agent_reasoning) carry it in ``variable.OutputValueText``
    or ``variable.InputValueText``. Legacy ``OutputValue`` / ``InputValue``
    keys are still accepted so pre-rename payloads rehydrate.
    """
    direct = entry.get("content")
    if direct:
        return _coerce_str(direct)
    var = entry.get("variable") or {}
    if isinstance(var, dict):
        for key in ("OutputValueText", "OutputValue", "InputValueText", "InputValue"):
            v = var.get(key)
            if v:
                return _coerce_str(v)
    return ""


def _tool_name(entry: dict[str, Any]) -> str:
    var = entry.get("variable") or {}
    if isinstance(var, dict):
        n = var.get("Name")
        if n:
            return _coerce_str(n)
    return "unknown"


def _tool_call_id(entry: dict[str, Any]) -> str | None:
    md = entry.get("metadata") or {}
    if isinstance(md, dict):
        v = md.get("tool_call_id")
        if v:
            return _coerce_str(v)
    return None


def _tool_args(entry: dict[str, Any]) -> dict[str, Any]:
    var = entry.get("variable") or {}
    if not isinstance(var, dict):
        return {}
    raw = var.get("InputValueText")
    if raw is None:
        raw = var.get("InputValue")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return {"input": raw}
        if isinstance(decoded, dict):
            return decoded
        return {"input": decoded}
    return {"input": raw}


def conversation_to_messages(
    payload: dict[str, Any] | None,
) -> list[BaseMessage]:
    """Rehydrate a ``GET /agentic/conversations/{id}`` payload into LangGraph
    messages.

    Walks ``payload["sessions"][*]["entries"]`` in order and dispatches on
    ``type``:

    - ``user_message``    → ``HumanMessage``
    - ``agent_response``  → ``AIMessage``
    - ``tool_call``       → ``AIMessage(content="", tool_calls=[...])``
    - ``tool_result``     → ``ToolMessage``
    - ``agent_reasoning`` → folded into the next AIMessage's content as a
      ``[reasoning] ...`` line (or dropped if no AIMessage follows)

    Unknown ``type`` values are skipped silently. Entries with neither a
    parseable ``type`` nor a usable text body are skipped.

    Consecutive ``tool_call`` entries between an AIMessage boundary collapse
    into one AIMessage with multiple ``tool_calls``, matching how LangChain
    represents parallel tool use.
    """
    if not payload:
        return []
    sessions = payload.get("sessions") or []
    if not isinstance(sessions, list):
        return []

    messages: list[BaseMessage] = []
    pending_reasoning: list[str] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        # Reasoning preface (if any) is folded into the AIMessage content.
        content = "\n".join(f"[reasoning] {r}" for r in pending_reasoning)
        messages.append(AIMessage(content=content, tool_calls=list(pending_tool_calls)))
        pending_tool_calls.clear()
        pending_reasoning.clear()

    for session in sessions:
        if not isinstance(session, dict):
            continue
        entries = session.get("entries") or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            etype = entry.get("type")

            if etype == "tool_call":
                tc_id = _tool_call_id(entry) or _coerce_str(entry.get("id") or "")
                # Generate synthetic id if both sources are missing to avoid empty string breaking LangChain pairing
                if not tc_id:
                    tc_id = f"rehydrated-tool-call-{uuid.uuid4().hex[:12]}"
                pending_tool_calls.append({
                    "name": _tool_name(entry),
                    "args": _tool_args(entry),
                    "id": tc_id,
                })
                continue

            # Any non-tool_call entry closes an open AIMessage(tool_calls=...)
            flush_tool_calls()

            if etype == "user_message":
                text = _entry_text(entry)
                if not text:
                    continue
                messages.append(HumanMessage(content=text))
            elif etype == "agent_response":
                text = _entry_text(entry)
                if not text and not pending_reasoning:
                    continue
                preface = (
                    "\n".join(f"[reasoning] {r}" for r in pending_reasoning) + "\n"
                    if pending_reasoning
                    else ""
                )
                pending_reasoning.clear()
                messages.append(AIMessage(content=preface + text))
            elif etype == "tool_result":
                text = _entry_text(entry)
                tc_id = _tool_call_id(entry)
                if not tc_id:
                    # ToolMessage requires a tool_call_id; drop orphans.
                    continue
                messages.append(
                    ToolMessage(
                        content=text,
                        tool_call_id=tc_id,
                        name=_tool_name(entry),
                    )
                )
            elif etype == "agent_reasoning":
                text = _entry_text(entry)
                if text:
                    pending_reasoning.append(text)
            # Unknown types: skip silently.

    # Trailing tool_calls / reasoning at end-of-payload still need to land.
    flush_tool_calls()
    if pending_reasoning:
        messages.append(
            AIMessage(content="\n".join(f"[reasoning] {r}" for r in pending_reasoning))
        )
        pending_reasoning.clear()

    return messages


__all__ = [
    "format_context_block",
    "context_to_system_message",
    "conversation_to_messages",
]
