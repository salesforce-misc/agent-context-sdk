# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Smoke: Microsoft Agent Framework / Foundry adapter (``agentic_context_foundry``).

Wiring under test is the documented ``adapter.run(agent, text, session=...)``
path with a real ``Agent`` and ``AgentSession``. With no provider attached,
the adapter seeds context, adds its tools and middleware, and ingests once
at the end of each turn.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any
from uuid import uuid4

import pytest

from harness import (
    AGENT_ID, OWNER_ID, SYSTEM_PROMPT, TOOL_NAME, TURNS,
    Capture, FakeAuth, LiveConfig, LiveContract, MockBeacon, RecordingPassThrough,
    SmokeContract, lookup_order, new_session_id, read_back, script_next,
)

pytestmark = pytest.mark.foundry

pytest.importorskip("agent_framework")
pytest.importorskip("agentic_context_foundry")

from agent_framework import (  # noqa: E402
    Agent, AgentSession, BaseChatClient, ChatMiddlewareLayer, ChatResponse,
    ChatResponseUpdate, Content, FunctionInvocationLayer, Message, tool,
)

from agentic_context_foundry import FoundryAdapter  # noqa: E402

_USER_TEXTS = {t.user for t in TURNS}


def _progress(messages: Sequence[Message]) -> tuple[int, bool]:
    """(scenario user turns seen, whether the latest one has a tool result).

    Only scenario texts count as turns, so injected context or instruction
    messages can't shift the script.
    """
    users, answered = 0, False
    for message in messages:
        for content in message.contents or []:
            if str(message.role) == "user" and content.type == "text" and content.text in _USER_TEXTS:
                users, answered = users + 1, False
            elif content.type == "function_result":
                answered = True
    return users, answered


class ScriptedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Offline model: calls lookup_order on turn 1, then replies from the script."""

    def _contents(self, messages: Sequence[Message]) -> list[Content]:
        action, payload = script_next(*_progress(messages))
        if action == "tool":
            return [Content.from_function_call(
                f"call_{uuid4().hex[:12]}", TOOL_NAME, arguments=json.dumps(payload),
            )]
        return [Content.from_text(payload)]

    def _inner_get_response(self, *, messages: Sequence[Message], stream: bool,
                            options: Mapping[str, Any], **kwargs: Any) -> Any:
        if stream:
            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                yield ChatResponseUpdate(role="assistant", contents=self._contents(messages))

            return self._build_response_stream(_stream())

        async def _get() -> ChatResponse:
            return ChatResponse(messages=[Message(role="assistant", contents=self._contents(messages))])

        return _get()


async def drive(adapter: FoundryAdapter, client: Any, capture: Capture) -> None:
    agent = Agent(client, instructions=SYSTEM_PROMPT, name="smoke-foundry", tools=[tool(lookup_order)])
    session = AgentSession(session_id=capture.session_id)
    try:
        for turn in TURNS:
            records = await adapter.run(agent, turn.user, session=session)
            replies = [r.content for r in records if r.event_type == "agent_response" and r.content]
            capture.replies.append(replies[-1] if replies else "")
    finally:
        await adapter.close()


# ─── Tier 1: offline ─────────────────────────────────────────────────


def _run_offline() -> Capture:
    capture = Capture("foundry", new_session_id("foundry"), OWNER_ID, AGENT_ID)
    with MockBeacon(capture):
        adapter = FoundryAdapter(AGENT_ID, owner_id=OWNER_ID, auth=FakeAuth())
        asyncio.run(drive(adapter, ScriptedChatClient(), capture))
    return capture


@pytest.fixture(scope="module")
def offline_capture() -> Capture:
    return _run_offline()


KNOWN_GAPS: dict[str, str] = {}


@pytest.mark.offline
class TestFoundryOffline(SmokeContract):
    agent_type = "MicrosoftAgentFramework"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, offline_capture: Capture) -> Capture:
        return offline_capture


# ─── Tier 2: live ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_capture(live_config: LiveConfig) -> Capture:
    capture = Capture("foundry", new_session_id("foundry"), live_config.owner_id, live_config.agent_id)

    async def run() -> None:
        with RecordingPassThrough(capture):
            adapter = FoundryAdapter(
                live_config.agent_id, owner_id=live_config.owner_id, auth=live_config.auth,
            )
            await drive(adapter, ScriptedChatClient(), capture)
        capture.readback = await read_back(live_config, capture.session_id)

    asyncio.run(run())
    return capture


@pytest.mark.live
class TestFoundryLive(LiveContract):
    agent_type = "MicrosoftAgentFramework"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, live_capture: Capture) -> Capture:
        return live_capture
