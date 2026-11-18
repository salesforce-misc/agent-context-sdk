# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Map Microsoft Agent Framework Content into Beacon Records.

One Content → at most one Record; unknown types are skipped. Categorization:

- ``user_message``/ ``agent_response`` → Message (user input to ``agent.run``).
- ``agent_reasoning`` → Trace (``text_reasoning`` or intermediate LLM text
  when tools follow).
- ``tool_call`` / ``tool_result`` → Trace (function name + args`).
  ``_tool_names`` remembers the call so the result can reuse
  the tool name when MAF does not surface one.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agentic_context import Record

class RecordTransformer:
    """RecordTransformer converts Foundry ``Message``/``AgentResponseUpdate`` contents to Records."""

    def __init__(self, agent_id: str, owner_id: str) -> None:
        self.agent_id = agent_id
        self.owner_id = owner_id
        # Beacon registers participants as the bare owner/agent ids (no suffix).
        self._user_participant = owner_id
        self._agent_participant = agent_id
        self._tool_names: dict[str, str] = {}

    def from_message(
        self, message: Any, session_id: str, interaction_id: str, index: int,
    ) -> list[Record]:
        role = role_of(getattr(message, "role", None))
        message_id = getattr(message, "message_id", None)
        records: list[Record] = []
        for offset, content in enumerate(getattr(message, "contents", None) or []):
            event_id = f"{message_id}-{index}" if message_id else f"msg-{session_id}-{index}"
            if offset:
                event_id = f"{event_id}-{offset}"
            record = self._from_content(
                content,
                role=role,
                session_id=session_id,
                interaction_id=interaction_id,
                event_id=event_id,
            )
            if record is not None:
                records.append(record)
        return records

    def from_update(
        self, update: Any, session_id: str, interaction_id: str, index: int,
        *,
        tools_only: bool = False,
    ) -> list[Record]:
        """Map one stream update. When ``tools_only``, skip text / reasoning
        (those are coalesced once per turn after the stream finishes).
        """
        if not tools_only:
            return self.from_message(update, session_id, interaction_id, index)
        role = role_of(getattr(update, "role", None))
        message_id = getattr(update, "message_id", None)
        records: list[Record] = []
        for offset, content in enumerate(getattr(update, "contents", None) or []):
            kind = getattr(content, "type", None) or ""
            if kind not in ("function_call", "function_result"):
                continue
            if kind == "function_call" and not stream_tool_args_ready(
                getattr(content, "arguments", None)
            ):
                continue
            event_id = f"{message_id}-{index}" if message_id else f"msg-{session_id}-{index}"
            if offset:
                event_id = f"{event_id}-{offset}"
            record = self._from_content(
                content,
                role=role,
                session_id=session_id,
                interaction_id=interaction_id,
                event_id=event_id,
            )
            if record is not None:
                records.append(record)
        return records

    def _from_content(
        self,
        content: Any,
        *,
        role: str,
        session_id: str,
        interaction_id: str,
        event_id: str,
    ) -> Record | None:
        kind = getattr(content, "type", None) or ""
        match kind:
            case "text":
                text = getattr(content, "text", None) or ""
                if not text:
                    return None
                if role == "user":
                    return self._record(
                        event_id=event_id,
                        session_id=session_id,
                        event_type="user_message",
                        participant_id=self._user_participant,
                        content=text,
                        category="Message",
                        interaction_id=interaction_id,
                    )
                if role == "assistant":
                    return self.agent_response(
                        text, session_id, interaction_id, event_id=event_id,
                    )
                return None
            case "text_reasoning":
                return self.agent_reasoning(
                    getattr(content, "text", None) or "",
                    session_id,
                    interaction_id,
                    event_id=event_id,
                )
            case "function_call":
                call_id = getattr(content, "call_id", None) or event_id
                name = getattr(content, "name", None) or "unknown"
                return self.tool_call(
                    call_id, name, as_dict(getattr(content, "arguments", None)),
                    session_id, interaction_id,
                )
            case "function_result":
                call_id = getattr(content, "call_id", None) or event_id
                result = getattr(content, "result", None)
                return self.tool_result(call_id, result, session_id, interaction_id)
            case _:
                return None

    def agent_response(
        self,
        content: str,
        session_id: str,
        interaction_id: str,
        *,
        event_id: str | None = None,
    ) -> Record:
        return self._record(
            event_id=event_id or f"{interaction_id}-response",
            session_id=session_id,
            event_type="agent_response",
            participant_id=self._agent_participant,
            content=content,
            category="Message",
            interaction_id=interaction_id,
        )

    def agent_reasoning(
        self,
        text: str,
        session_id: str,
        interaction_id: str,
        *,
        event_id: str | None = None,
    ) -> Record:
        return self._record(
            event_id=event_id or f"{interaction_id}-reasoning",
            session_id=session_id,
            event_type="agent_reasoning",
            participant_id=self._agent_participant,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={"OutputValueText": text},
        )

    def tool_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
        interaction_id: str,
    ) -> Record:
        self._tool_names[call_id] = tool_name
        return self._record(
            event_id=call_id,
            session_id=session_id,
            event_type="tool_call",
            participant_id=self._agent_participant,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={"Name": tool_name, "InputValueText": json.dumps(arguments, default=str)},
        )

    def tool_result(
        self,
        call_id: str,
        result: Any,
        session_id: str,
        interaction_id: str,
    ) -> Record:
        name = self._tool_names.get(call_id, "unknown")
        # None -> "", str passed through raw, everything else JSON-encoded
        # (default=str covers unserializable values, skipkeys=True covers
        # unserializable dict keys — json.dumps never raises here).
        output = (
            "" if result is None
            else result if isinstance(result, str)
            else json.dumps(result, default=str, skipkeys=True)
        )
        return self._record(
            event_id=f"{call_id}-result",
            session_id=session_id,
            event_type="tool_result",
            participant_id=self._agent_participant,
            content=None,
            category="Trace",
            interaction_id=interaction_id,
            variable={"Name": name, "OutputValueText": output},
        )

    def _record(
        self,
        *,
        event_id: str,
        session_id: str,
        event_type: str,
        participant_id: str,
        content: str | None,
        category: str,
        interaction_id: str,
        variable: dict[str, Any] | None = None,
    ) -> Record:
        return Record(
            event_id=event_id or f"msg-{session_id}-{event_type}",
            session_id=session_id,
            agent_id=self.agent_id,
            event_type=event_type,
            participant_id=participant_id,
            content=content,
            category=category,
            interaction_id=interaction_id,
            variable=variable,
            timestamp=datetime.now(timezone.utc),
        )

def stream_tool_args_ready(value: Any) -> bool:
    """True when a streamed function_call fragment has a complete arguments payload.
    """
    if value is None:
        return False
    if isinstance(value, str):
        if not value.strip():
            return False
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict)
    return True


def as_dict(value: Any) -> dict[str, Any]:
    """Normalize tool call's arguments to a dict to feed into Record's ``variable`` field.
    """
    match value:
        case dict():
            return value
        case str() if value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        case _ if hasattr(value, "items"):
            return dict(value.items())
        case _:
            return {}


def role_of(value: Any) -> str:
    """Normalize Message.role / update.role (str or enum) to a lowercase string."""
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).lower()
