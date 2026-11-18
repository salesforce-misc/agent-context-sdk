# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Microsoft Agent Framework middleware that records a turn as Beacon Records.

``agent.run()`` is the turn boundary (returns once after the tool loop); ingest
is triggered there, not here. These hooks only capture:

- ``_ChatMiddleware`` — per LLM call; intermediate assistant text/reasoning as
  Trace when tools follow. Skips streaming so it never consumes a ResponseStream.
- ``_FunctionMiddleware`` — per tool call; ``tool_call`` before ``call_next``,
  ``tool_result`` after (name + args / result).

Session ids are caller-supplied ``AgentSession``s on the adapter. When MAF does
not expose a tool_call_id, tool ``event_id``s are derived from session/turn/name.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agent_framework import ChatMiddleware, FunctionMiddleware

from .transformer import RecordTransformer, as_dict

logger = logging.getLogger(__name__)


class _ChatMiddleware(ChatMiddleware):
    """Capture assistant text / reasoning from a non-streaming chat result.

    Streaming turns are recorded from ``AgentResponseUpdate`` in ``stream()``
    so this class does not iterate (or consume) a ResponseStream.
    """

    def __init__(
        self, transformer: RecordTransformer, session_id: str, interaction_id: str,
    ) -> None:
        self._transformer = transformer
        self._session_id = session_id
        self._interaction_id = interaction_id
        self.records: list[Any] = []

    async def process(self, context: Any, call_next: Any) -> None:
        await call_next()
        if getattr(context, "stream", False):
            return
        try:
            self._capture(getattr(context, "result", None))
        except Exception:
            logger.exception("[Foundry] chat middleware capture failed")

    def _capture(self, result: Any) -> None:
        if result is None:
            return
        finish = (getattr(result, "finish_reason", None) or "").lower()
        messages = getattr(result, "messages", None) or []
        texts: list[str] = []
        for message in messages:
            for content in getattr(message, "contents", None) or []:
                if getattr(content, "type", None) == "text" and getattr(content, "text", None):
                    texts.append(content.text)
                elif getattr(content, "type", None) == "text_reasoning":
                    self.records.append(
                        self._transformer.agent_reasoning(
                            content.text or "", self._session_id, self._interaction_id,
                        )
                    )
        joined = "\n".join(texts)
        if "tool" in finish:
            if joined:
                self.records.append(
                    self._transformer.agent_reasoning(
                        joined, self._session_id, self._interaction_id,
                    )
                )
            return
        if joined:
            self.records.append(
                self._transformer.agent_response(
                    joined, self._session_id, self._interaction_id,
                )
            )


def _resolve_call_id(context: Any, interaction_id: str, name: str, n: int) -> str:
    call_id = getattr(context, "call_id", None)
    if call_id:
        return str(call_id)
    function_call = getattr(context, "function_call", None)
    call_id = getattr(function_call, "call_id", None) if function_call is not None else None
    if call_id:
        return str(call_id)
    metadata = getattr(context, "metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("call_id"):
        return str(metadata["call_id"])
    return f"{interaction_id}-{name}-{n}"


class _FunctionMiddleware(FunctionMiddleware):
    """Emit tool_call before the function runs and tool_result after."""

    def __init__(
        self, transformer: RecordTransformer, session_id: str, interaction_id: str,
    ) -> None:
        self._transformer = transformer
        self._session_id = session_id
        self._interaction_id = interaction_id
        self.records: list[Any] = []

    async def process(self, context: Any, call_next: Any) -> None:
        function = getattr(context, "function", None)
        name = getattr(function, "name", None) or "unknown"
        call_id = _resolve_call_id(
            context, self._interaction_id, name, len(self.records),
        )
        try:
            self.records.append(
                self._transformer.tool_call(
                    call_id,
                    name,
                    as_dict(getattr(context, "arguments", None)),
                    self._session_id,
                    self._interaction_id,
                )
            )
        except Exception:
            logger.exception("[Foundry] function middleware tool_call capture failed")
        await call_next()
        try:
            self.records.append(
                self._transformer.tool_result(
                    call_id,
                    _result_text(getattr(context, "result", None)),
                    self._session_id,
                    self._interaction_id,
                )
            )
        except Exception:
            logger.exception("[Foundry] function middleware tool_result capture failed")

def _result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
                continue
            nested = getattr(item, "result", None)
            if nested is not None:
                parts.append(str(nested))
            elif item is None:
                parts.append("")
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, default=str, skipkeys=True))
        return "\n".join(parts)
    return json.dumps(result, default=str, skipkeys=True)