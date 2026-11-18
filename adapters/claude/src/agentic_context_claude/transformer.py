# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Convert Claude transcript entries into normalized Record objects.

One JSONL entry can produce MULTIPLE records:
- An assistant entry with [tool_use, text] content blocks produces both
  a tool_call record and an agent_response record
- A user entry with [tool_result, tool_result] produces multiple tool_result records
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from agentic_context import Record

logger = logging.getLogger(__name__)

# SDK-internal tools that don't represent customer interactions
FILTERED_TOOLS = {"ToolSearch"}


class RecordTransformer:
    """Converts Claude transcript entries to Records.

    Handles all Claude content block types:
    - text -> Message (user_message or agent_response)
    - tool_use -> Trace (tool_call)
    - tool_result -> Trace (tool_result)
    - thinking -> Trace (agent_reasoning)
    """

    def __init__(self, agent_id: str, participant_id: str = "") -> None:
        self.agent_id = agent_id
        # Fallback owner used only when transform() isn't given a per-call
        # owner_id. In a multi-user process this default MUST be overridden per
        # call (see transform(owner_id=...)) or every user_message would be
        # mis-attributed to a single owner.
        self.participant_id = participant_id
        # Track tool_name by tool_use_id across entries so tool_result can resolve it
        self._tool_names: dict[str, str] = {}

    def transform(
        self,
        entries: list[dict],
        session_id: str,
        *,
        owner_id: str | None = None,
        interaction_id: str | None = None,
    ) -> list[Record]:
        """Transform all entries, preserving order. Filters SDK noise.

        Args:
            owner_id: The end-user this transcript belongs to. Stamped as
                ``participant_id`` on user messages. Overrides the constructor
                default; pass it per call in a multi-user process so each user's
                messages are attributed to the right owner.
            interaction_id: Stamped on every record, so Beacon groups one
                turn's entries together.
        """
        participant = owner_id or self.participant_id
        records: list[Record] = []
        for entry in entries:
            records.extend(self._transform_entry(entry, session_id, participant))
        if interaction_id is not None:
            records = [r.model_copy(update={"interaction_id": interaction_id}) for r in records]
        return records

    def _transform_entry(
        self, entry: dict, session_id: str, participant_id: str,
    ) -> list[Record]:
        """Transform a single JSONL line.

        Transcript format:
        {"type": "user"|"assistant", "uuid": "...", "timestamp": "...",
         "message": {"role": "...", "content": "..." | [...blocks...]}}
        """
        records: list[Record] = []
        msg = entry.get("message", {})
        content = msg.get("content", "")
        uuid = entry.get("uuid", "")
        ts = entry.get("timestamp", "")
        entry_type = entry.get("type", "")

        # String content = simple user message
        if isinstance(content, str):
            if entry_type == "user":
                records.append(self._make_record(
                    event_id=uuid,
                    session_id=session_id,
                    event_type="user_message",
                    participant_id=participant_id,
                    content=content,
                    timestamp=ts,
                    category="Message",
                ))
            return records

        # List content = content blocks (user, assistant, or tool_result)
        for block_idx, block in enumerate(content):
            block_type = block.get("type", "")

            if block_type == "text" and entry_type == "assistant":
                records.append(self._make_record(
                    event_id=uuid,
                    session_id=session_id,
                    event_type="agent_response",
                    participant_id=self.agent_id,
                    content=block.get("text", ""),
                    timestamp=ts,
                    category="Message",
                ))

            elif block_type == "text" and entry_type == "user":
                # A user turn can arrive as content blocks (not just a plain
                # str) — e.g. multimodal input or a normalized transcript. The
                # str branch above only handles the simple case, so without
                # this the user_message would be silently dropped. Suffix the
                # event_id with the block index so multiple text blocks in one
                # user entry stay distinct and stable.
                records.append(self._make_record(
                    event_id=f"{uuid}-{block_idx}" if block_idx else uuid,
                    session_id=session_id,
                    event_type="user_message",
                    participant_id=participant_id,
                    content=block.get("text", ""),
                    timestamp=ts,
                    category="Message",
                ))

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_id = block.get("id", "")
                if tool_name in FILTERED_TOOLS:
                    continue
                self._tool_names[tool_id] = tool_name
                records.append(self._make_record(
                    event_id=tool_id or uuid,
                    session_id=session_id,
                    event_type="tool_call",
                    participant_id=self.agent_id,
                    content=None,
                    timestamp=ts,
                    category="Trace",
                    variable={
                        "Name": tool_name,
                        "InputValueText": json.dumps(block.get("input", {})),
                    },
                ))

            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = " ".join(
                        b.get("text", "") for b in result_content
                        if b.get("type") == "text"
                    )
                # Name on tool_result matches tool_call (Beacon convention)
                resolved_name = self._tool_names.get(tool_use_id, "")
                records.append(self._make_record(
                    event_id=f"{uuid}-{tool_use_id}",
                    session_id=session_id,
                    event_type="tool_result",
                    participant_id=self.agent_id,
                    content=None,
                    timestamp=ts,
                    category="Trace",
                    variable={
                        "Name": resolved_name,
                        "OutputValueText": (
                            result_content if isinstance(result_content, str)
                            else json.dumps(result_content)
                        ),
                    },
                ))

            elif block_type == "thinking":
                records.append(self._make_record(
                    event_id=f"{uuid}-thinking",
                    session_id=session_id,
                    event_type="agent_reasoning",
                    participant_id=self.agent_id,
                    content=None,
                    timestamp=ts,
                    category="Trace",
                    variable={
                        "OutputValueText": block.get("thinking", ""),
                    },
                ))

        return records

    def _make_record(
        self,
        *,
        event_id: str,
        session_id: str,
        event_type: str,
        participant_id: str,
        content: str | None,
        timestamp: str,
        category: str,
        variable: dict[str, Any] | None = None,
    ) -> Record:
        """Build a Record with proper timestamp parsing."""
        ts = self._parse_timestamp(timestamp)
        return Record(
            event_id=event_id,
            session_id=session_id,
            agent_id=self.agent_id,
            event_type=event_type,
            participant_id=participant_id,
            content=content,
            category=category,
            variable=variable,
            timestamp=ts,
        )

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse ISO timestamp, fallback to now(UTC) if invalid."""
        if not ts:
            return datetime.now(timezone.utc)
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)
