# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Convert OpenAI conversation entries into normalized ``Record`` objects.

Accepts two wire formats — the transformer sniffs each entry and dispatches:

1. Chat Completions message dicts (``openai.chat.completions.create`` input):

       {"role": "user", "content": "..."}
       {"role": "assistant", "content": "...", "tool_calls": [...]}
       {"role": "tool",   "content": "...", "tool_call_id": "..."}
       {"role": "system", "content": "..."}

2. Responses API items (``RunResult.to_input_list()`` from the OpenAI Agents SDK):

       {"type": "message",              "role": "assistant", "content": [...]}
       {"type": "function_call",        "name": "...", "call_id": "...", "arguments": "..."}
       {"type": "function_call_output", "call_id": "...", "output": "..."}
       {"type": "reasoning",            "content": "..."}

A single entry can produce multiple ``Record``s — e.g. an assistant turn with
both a text answer and a ``tool_calls`` array yields one ``agent_response`` and
one ``tool_call`` per call. Event IDs favour the native OpenAI id (``call_id``,
``id``) and fall back to a composed key shaped
``msg-<session>-<interaction|snap>-<idx>-<suffix>``. The idx resets at every
``transform()`` call, so a snapshot re-push of the same transcript produces
the same local ids (not sent, so a re-push is not deduped). Streaming
callers pass a per-turn ``interaction_id`` so slices from different turns stay
distinct.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from agentic_context import Record

logger = logging.getLogger(__name__)

# The Agents SDK's Chat Completions → Responses converter stamps every item it
# builds (messages, reasoning, function_calls) with the literal "__fake_id__".
# Treating that as a stable id would give every item the same event_id (and
# the run hooks' seen-id set would drop all but the first). Ignore it like a
# missing id.
_SDK_PLACEHOLDER_ID = "__fake_id__"


def _real_id(value: Any) -> str:
    """Return a trimmed id, or empty if absent or the SDK placeholder."""
    if not value:
        return ""
    s = str(value)
    if s == _SDK_PLACEHOLDER_ID:
        return ""
    return s


class RecordTransformer:
    """Converts OpenAI entries (Chat Completions messages OR Responses items) to Records.

    One shared instance can serve many users: pass ``owner_id`` per call so each
    user_message is attributed to the right owner rather than the constructor
    default.
    """

    def __init__(self, agent_id: str, participant_id: str = "") -> None:
        self.agent_id = agent_id
        self.participant_id = participant_id
        # Tool-name resolution across entries: a Chat Completions ``tool`` reply
        # only carries ``tool_call_id`` (no name), so we remember the name from
        # the preceding assistant ``tool_calls`` entry. Same shape works for the
        # Responses API's function_call → function_call_output pair.
        self._tool_names: dict[str, str] = {}
        # Per-transform() fallback counter. Reset at the start of each call so a
        # snapshot re-push of the same transcript produces identical local ids
        # (not sent, so nothing is deduped). Streaming callers
        # pass a per-turn ``interaction_id`` so slices from different turns stay
        # distinct even though each call starts at idx=0, and ``fallback_offset``
        # so several slices of one turn don't reuse the same idx.
        self._fallback_idx: int = 0

    def transform(
        self,
        entries: list[dict],
        session_id: str,
        *,
        owner_id: str | None = None,
        interaction_id: str | None = None,
        fallback_offset: int = 0,
    ) -> list[Record]:
        """Normalize ``entries`` to Records in chronological order.

        Args:
            entries: Chat Completions messages or Responses items.
            session_id: The Beacon session these records belong to.
            owner_id: Per-call participant_id for user_message records. Overrides
                the constructor default — pass it in a multi-user process.
            interaction_id: Optional turn id stamped onto every emitted Record
                so Beacon can group one turn's user_message, agent_response,
                reasoning, tool_call and tool_result. Callers mint one id per
                logical turn (streaming) or per snapshot (full re-read).
            fallback_offset: First idx for fallback event ids. A caller that
                transforms one turn in several slices under the same
                ``interaction_id`` passes the count of fallback ids already
                used (:attr:`fallback_count` after the previous call).
        """
        participant = owner_id or self.participant_id
        self._fallback_idx = fallback_offset
        records: list[Record] = []
        for entry in entries:
            records.extend(
                self._transform_entry(
                    entry, session_id, participant, interaction_id,
                )
            )
        return records

    @property
    def fallback_count(self) -> int:
        """Next fallback idx after the last :meth:`transform` call."""
        return self._fallback_idx

    # ─── Dispatch ──────────────────────────────────────────────────

    def _transform_entry(
        self,
        entry: dict,
        session_id: str,
        participant_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Pick the right handler by format sniffing.

        The Responses API items carry a top-level ``type`` discriminator
        (``function_call`` / ``function_call_output`` / ``reasoning`` / ``message``).
        Chat Completions messages don't — they're keyed by ``role``. A message
        dict with ``type == "message"`` goes down the role branch (same shape).
        """
        entry_type = entry.get("type")

        if entry_type == "function_call":
            return self._handle_function_call(entry, session_id, interaction_id)
        if entry_type == "function_call_output":
            return self._handle_function_call_output(entry, session_id, interaction_id)
        if entry_type == "reasoning":
            return self._handle_reasoning(entry, session_id, interaction_id)
        # ``type == "message"`` falls through to the role branch — same payload.
        return self._handle_role_message(
            entry, session_id, participant_id, interaction_id,
        )

    # ─── Chat Completions / Responses `message` entries ───────────

    def _handle_role_message(
        self,
        entry: dict,
        session_id: str,
        participant_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Dispatch a role-keyed message (Chat Completions or Responses ``message``)."""
        role = entry.get("role", "")
        if role == "user":
            return self._handle_user(
                entry, session_id, participant_id, interaction_id,
            )
        if role == "assistant":
            return self._handle_assistant(entry, session_id, interaction_id)
        if role == "tool":
            return self._handle_tool_reply(entry, session_id, interaction_id)
        # ``system`` / ``developer`` roles are prompt scaffolding, not transcript
        # events. Dropping them keeps Beacon from ingesting operator-side setup
        # as if the user had said it.
        return []

    def _handle_user(
        self,
        entry: dict,
        session_id: str,
        participant_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        content = entry.get("content")
        text = self._extract_text(content)
        if not text:
            return []
        return [self._make_record(
            event_id=self._event_id(entry, session_id, suffix="u", interaction_id=interaction_id),
            session_id=session_id,
            event_type="user_message",
            participant_id=participant_id,
            content=text,
            timestamp=entry.get("timestamp"),
            category="Message",
            interaction_id=interaction_id,
        )]

    def _handle_assistant(
        self,
        entry: dict,
        session_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Assistant turn: zero or one text record, plus one tool_call per call.

        An assistant message with both text and tool_calls emits both — the
        text is the reasoning-visible pre-tool reply, the tool_calls are the
        traces. Order is text → tool_calls (matches ``agent_response`` before
        ``tool_call`` chronologically within one turn).
        """
        records: list[Record] = []
        content = entry.get("content")
        text = self._extract_text(content)
        if text:
            records.append(self._make_record(
                event_id=self._event_id(entry, session_id, suffix="a", interaction_id=interaction_id),
                session_id=session_id,
                event_type="agent_response",
                participant_id=self.agent_id,
                content=text,
                timestamp=entry.get("timestamp"),
                category="Message",
                interaction_id=interaction_id,
            ))

        for call in entry.get("tool_calls") or []:
            call_id = _real_id(call.get("id"))
            fn = call.get("function") or {}
            tool_name = fn.get("name", "") or ""
            arguments = fn.get("arguments", "") or ""
            if call_id:
                self._tool_names[call_id] = tool_name
            records.append(self._make_record(
                event_id=call_id or self._fallback_id(session_id, "tc", interaction_id),
                session_id=session_id,
                event_type="tool_call",
                participant_id=self.agent_id,
                content=None,
                timestamp=entry.get("timestamp"),
                category="Trace",
                variable={
                    "Name": tool_name,
                    "InputValueText": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
                interaction_id=interaction_id,
            ))

        return records

    def _handle_tool_reply(
        self,
        entry: dict,
        session_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Chat Completions ``{"role": "tool", "tool_call_id": "...", "content": "..."}``."""
        call_id = _real_id(entry.get("tool_call_id"))
        output = self._extract_text(entry.get("content"))
        resolved_name = self._tool_names.get(call_id, "")
        return [self._make_record(
            event_id=f"{call_id}-out" if call_id else self._fallback_id(session_id, "to", interaction_id),
            session_id=session_id,
            event_type="tool_result",
            participant_id=self.agent_id,
            content=None,
            timestamp=entry.get("timestamp"),
            category="Trace",
            variable={
                "Name": resolved_name,
                "OutputValueText": output,
            },
            interaction_id=interaction_id,
        )]

    # ─── Responses API typed entries ──────────────────────────────

    def _handle_function_call(
        self,
        entry: dict,
        session_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Responses API ``{"type": "function_call", "name": "...", "call_id": "...", "arguments": "..."}``."""
        call_id = _real_id(entry.get("call_id")) or _real_id(entry.get("id"))
        tool_name = entry.get("name", "") or ""
        arguments = entry.get("arguments", "") or ""
        if call_id:
            self._tool_names[call_id] = tool_name
        return [self._make_record(
            event_id=call_id or self._fallback_id(session_id, "fc", interaction_id),
            session_id=session_id,
            event_type="tool_call",
            participant_id=self.agent_id,
            content=None,
            timestamp=entry.get("timestamp"),
            category="Trace",
            variable={
                "Name": tool_name,
                "InputValueText": arguments if isinstance(arguments, str) else json.dumps(arguments),
            },
            interaction_id=interaction_id,
        )]

    def _handle_function_call_output(
        self,
        entry: dict,
        session_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Responses API ``{"type": "function_call_output", "call_id": "...", "output": "..."}``."""
        call_id = _real_id(entry.get("call_id"))
        output = entry.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output)
        resolved_name = self._tool_names.get(call_id, "")
        return [self._make_record(
            event_id=f"{call_id}-out" if call_id else self._fallback_id(session_id, "fco", interaction_id),
            session_id=session_id,
            event_type="tool_result",
            participant_id=self.agent_id,
            content=None,
            timestamp=entry.get("timestamp"),
            category="Trace",
            variable={
                "Name": resolved_name,
                "OutputValueText": output,
            },
            interaction_id=interaction_id,
        )]

    def _handle_reasoning(
        self,
        entry: dict,
        session_id: str,
        interaction_id: str | None,
    ) -> list[Record]:
        """Responses API reasoning items (o1/o3).

        Two shapes in the wild:

        * legacy ``{"type": "reasoning", "content": "..."}`` — a plain string.
        * current ``{"type": "reasoning", "summary": [{"type": "summary_text",
          "text": "..."}]}`` — a list of summary parts.

        Both route through ``_extract_text`` so new part types only need to be
        added once.
        """
        text = self._extract_text(entry.get("content") or entry.get("summary"))
        if not text:
            return []
        return [self._make_record(
            event_id=self._event_id(entry, session_id, suffix="r", interaction_id=interaction_id),
            session_id=session_id,
            event_type="agent_reasoning",
            participant_id=self.agent_id,
            content=None,
            timestamp=entry.get("timestamp"),
            category="Trace",
            variable={"OutputValueText": text},
            interaction_id=interaction_id,
        )]

    # ─── Shared helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Pull a string out of OpenAI's content polymorphism.

        Content can be a plain string, a list of content parts (``{"type":
        "text", "text": "..."}``), or ``None``. Multimodal parts (image_url,
        input_audio) have no text payload — joined only when they're text parts.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                # summary_text is the Responses API reasoning shape:
                # {"type": "summary_text", "text": "..."}
                if ptype in ("text", "input_text", "output_text", "summary_text"):
                    parts.append(part.get("text", "") or "")
                elif ptype == "refusal":
                    parts.append(part.get("refusal", "") or "")
            return "".join(parts)
        return ""

    def _event_id(
        self, entry: dict, session_id: str, *,
        suffix: str, interaction_id: str | None,
    ) -> str:
        """Prefer the native OpenAI id; fall back to a composed key.

        ``id`` is the Responses API item id; it's absent on raw Chat Completions
        messages, so the fallback keys off (session, interaction_id, idx,
        suffix). The idx resets at each ``transform()`` call so a snapshot
        re-push of the same transcript produces identical local ids. They are
        not sent, so a re-push is not deduped.
        """
        native = _real_id(entry.get("id"))
        if native:
            return native
        return self._fallback_id(session_id, suffix, interaction_id)

    def _fallback_id(
        self, session_id: str, suffix: str, interaction_id: str | None,
    ) -> str:
        """Compose a fallback id keyed by session + turn + per-call idx."""
        idx = self._fallback_idx
        self._fallback_idx += 1
        turn = interaction_id or "snap"
        return f"msg-{session_id}-{turn}-{idx}-{suffix}"

    def _make_record(
        self,
        *,
        event_id: str,
        session_id: str,
        event_type: str,
        participant_id: str,
        content: str | None,
        timestamp: Any,
        category: str,
        variable: dict[str, Any] | None = None,
        interaction_id: str | None = None,
    ) -> Record:
        return Record(
            event_id=event_id,
            session_id=session_id,
            agent_id=self.agent_id,
            event_type=event_type,
            participant_id=participant_id,
            content=content,
            category=category,
            variable=variable,
            timestamp=self._parse_timestamp(timestamp),
            interaction_id=interaction_id,
        )

    @staticmethod
    def _parse_timestamp(ts: Any) -> datetime:
        """Parse an ISO timestamp; fall back to now(UTC) on anything unusable."""
        if not ts:
            return datetime.now(timezone.utc)
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)
