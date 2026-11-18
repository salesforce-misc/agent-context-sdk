# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""FoundryContextProvider: a ContextProvider is a built-in class for memory persistence in session.

Before each reply, FoundryContextProvider appends known customer context onto the agent's
instructions. Post reply, converts the turn's messages and tool
calls into Records and ingests into session.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_framework import ContextProvider

from .adapter import session_messages
from .transformer import RecordTransformer

if TYPE_CHECKING:
    from .adapter import FoundryAdapter

_CAPTURE_KEYS = (
    "transformer", "chat_mw", "func_mw", "input_records",
)

logger = logging.getLogger(__name__)

def collect_turn_records(capture: dict[str, Any], context: Any) -> list[Any]:
    """Input + response messages + middleware, first event_id wins.

    Final response comes from ``context.response``.
    Chat middleware keeps intermediate reasoning; its final ``agent_response``
    is skipped when the response already supplied one.
    """
    transformer = capture["transformer"]
    session_id = capture["session_id"]
    interaction_id = capture["interaction_id"]
    records: list[Any] = list(capture.get("input_records") or [])

    response = getattr(context, "response", None) if context is not None else None
    messages = getattr(response, "messages", None) or []
    index = int(capture.get("history_len") or 0) + int(
        capture.get("input_message_count") or 0
    )
    for message in messages:
        if (getattr(message, "role", None) or "").lower() == "user":
            continue
        records.extend(
            transformer.from_message(message, session_id, interaction_id, index)
        )
        index += 1

    chat_mw = capture.get("chat_mw")
    func_mw = capture.get("func_mw")
    if chat_mw is not None:
        has_response = any(
            getattr(record, "event_type", None) == "agent_response" for record in records
        )
        for record in chat_mw.records:
            if has_response and getattr(record, "event_type", None) == "agent_response":
                continue
            records.append(record)
    if func_mw is not None:
        records.extend(func_mw.records)
    return _dedupe(records)


class FoundryContextProvider(ContextProvider):
    """Cross-session memory + Record capture for one ``FoundryAdapter``."""

    after_run_once_per_turn = True

    def __init__(self, adapter: "FoundryAdapter", *, owner_id: str | None = None) -> None:
        super().__init__(source_id="agentic_context")
        self._adapter = adapter
        self._owner_id = owner_id

    async def before_run(self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]) -> None:
        from .middleware import _ChatMiddleware, _FunctionMiddleware

        adapter = self._adapter
        session_id = session.session_id
        owner_override = state.pop("owner_override", None)
        owner = owner_override or self._owner_id or adapter._owner_id
        await adapter._manager(session_id, owner)
        adapter._agent_sessions[session_id] = session

        n = adapter._turn.get(session_id, 0)
        adapter._turn[session_id] = n + 1
        interaction_id = f"turn-{session_id}-{n}"
        transformer = RecordTransformer(adapter.agent_id, owner)
        chat_mw = _ChatMiddleware(transformer, session_id, interaction_id)
        func_mw = _FunctionMiddleware(transformer, session_id, interaction_id)

        history_len = len(session_messages(session))
        input_messages = list(getattr(context, "input_messages", None) or [])

        _update_session_state(
            state,
            owner=owner,
            session_id=session_id,
            interaction_id=interaction_id,
            history_len=history_len,
            input_message_count=len(input_messages),
        )

        is_open = await adapter._claim_seed(session_id)
        if is_open:
            formatted = adapter._format_context(
                await adapter.get_context(session_id, owner_id=owner)
            )
            if formatted:
                context.extend_instructions(self.source_id, formatted)
        await adapter._maybe_inject_context_pack_list(is_open=is_open)
        allowed_pack_ids = set(adapter._allowed_pack_ids)
        # The instruction explains the catalog; with no packs it is only cost.
        if allowed_pack_ids:
            instruction = await adapter.get_context_pack_instruction(session_id)
            if not adapter._context_pack_prompt.present_in(
                adapter._agent_instructions_text(agent), instruction,
            ):
                context.extend_instructions(self.source_id, instruction)
        if adapter._allowed_context_pack_list:
            context.extend_instructions(self.source_id, adapter._allowed_context_pack_list)

        context.extend_tools(
            self.source_id,
            adapter.default_tools(session_id, owner, valid_pack_ids=sorted(allowed_pack_ids)),
        )
        context.extend_middleware(self.source_id, [chat_mw, func_mw])

        records = []
        for offset, message in enumerate(input_messages):
            records.extend(
                transformer.from_message(
                    message, session_id, interaction_id, history_len + offset,
                )
            )
        adapter._captures[session_id] = {
            "transformer": transformer,
            "chat_mw": chat_mw,
            "func_mw": func_mw,
            "input_records": records,
            "interaction_id": interaction_id,
            "session_id": session_id,
            "owner": owner,
            "history_len": history_len,
            "input_message_count": len(input_messages),
            "skip_after_run_ingest": False,
        }

    async def after_run(self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]) -> None:
        adapter = self._adapter
        session_id = state.get("session_id") or session.session_id
        capture = adapter._captures.get(session_id)
        if capture is None or capture.get("skip_after_run_ingest"):
            return
        owner = (
            capture.get("owner")
            or state.get("owner")
            or self._owner_id
            or adapter._owner_id
        )
        records = collect_turn_records(capture, context)
        if not records:
            return
        manager = await adapter._manager(session_id, owner)
        await manager.ingest(records)
        await manager.flush()

def _update_session_state(state: dict[str, Any], **values: Any) -> None:
    for key in _CAPTURE_KEYS:
        state.pop(key, None)
    state.update(values)


def _dedupe(records: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for record in records:
        event_id = getattr(record, "event_id", None)
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        out.append(record)
    return out
