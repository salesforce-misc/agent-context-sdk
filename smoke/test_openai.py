# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Smoke: OpenAI Agents SDK adapter (``agentic_context_openai``).

Wiring under test is the documented streaming pattern: one
``Runner.run(..., hooks=adapter.run_hooks(sid, user_input=text))`` per turn,
then ``await hooks.flush()``, with the adapter's ``default_tools`` and
``instructions_provider`` on the agent.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

from harness import (
    AGENT_ID, OWNER_ID, SYSTEM_PROMPT, TOOL_NAME, TURNS,
    Capture, FakeAuth, LiveConfig, LiveContract, MockBeacon, RecordingPassThrough,
    SmokeContract, lookup_order, new_session_id, read_back, script_next,
)

pytestmark = pytest.mark.openai

agents = pytest.importorskip("agents")
pytest.importorskip("agentic_context_openai")

from agents import Agent, Runner, function_tool, set_tracing_disabled  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
)

from agentic_context_openai import OpenAIAdapter  # noqa: E402

# The Agents SDK otherwise tries to export traces to OpenAI.
set_tracing_disabled(True)


def _item(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else raw.model_dump(exclude_unset=True)


def _progress(input_items: str | list[Any]) -> tuple[int, bool]:
    """(user messages seen, whether the last one already has a tool result)."""
    if isinstance(input_items, str):
        return 1, False
    users, answered = 0, False
    for raw in input_items:
        item = _item(raw)
        if item.get("role") == "user" and item.get("type", "message") == "message":
            users, answered = users + 1, False
        elif item.get("type") == "function_call_output":
            answered = True
    return users, answered


def text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid4().hex[:12]}", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> ResponseFunctionToolCall:
    call_id = call_id or f"call_{uuid4().hex[:12]}"
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, type="function_call",
        name=name, arguments=json.dumps(args), status="completed",
    )


class ScriptedModel(Model):
    """Offline model: calls lookup_order on turn 1, then replies from the script."""

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, **_: Any) -> ModelResponse:
        action, payload = script_next(*_progress(input))
        out = tool_call(TOOL_NAME, payload) if action == "tool" else text_message(payload)
        return ModelResponse(output=[out], usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any):  # pragma: no cover — not used
        raise NotImplementedError


async def drive(adapter: OpenAIAdapter, model: Model, capture: Capture) -> None:
    sid = capture.session_id
    agent = Agent(
        name="smoke-openai",
        instructions=adapter.instructions_provider(SYSTEM_PROMPT, session_id=sid),
        model=model,
        tools=[*adapter.default_tools(sid), function_tool(lookup_order)],
    )
    history: list[Any] = []
    try:
        for turn in TURNS:
            history = [*history, {"role": "user", "content": turn.user}]
            hooks = adapter.run_hooks(sid, user_input=turn.user)
            try:
                result = await Runner.run(agent, history, hooks=hooks)
            finally:
                await hooks.flush()
            capture.replies.append(str(result.final_output))
            history = result.to_input_list()
    finally:
        await adapter.close()


# ─── Tier 1: offline ─────────────────────────────────────────────────


def _run_offline() -> Capture:
    capture = Capture("openai", new_session_id("openai"), OWNER_ID, AGENT_ID)
    with MockBeacon(capture):
        adapter = OpenAIAdapter(AGENT_ID, owner_id=OWNER_ID, auth=FakeAuth())
        asyncio.run(drive(adapter, ScriptedModel(), capture))
    return capture


@pytest.fixture(scope="module")
def offline_capture() -> Capture:
    return _run_offline()


KNOWN_GAPS: dict[str, str] = {}


@pytest.mark.offline
class TestOpenAIOffline(SmokeContract):
    agent_type = "OpenAI"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, offline_capture: Capture) -> Capture:
        return offline_capture


# ─── Tier 2: live ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_capture(live_config: LiveConfig) -> Capture:
    capture = Capture("openai", new_session_id("openai"), live_config.owner_id, live_config.agent_id)

    async def run() -> None:
        with RecordingPassThrough(capture):
            adapter = OpenAIAdapter(
                live_config.agent_id, owner_id=live_config.owner_id, auth=live_config.auth,
            )
            await drive(adapter, ScriptedModel(), capture)
        capture.readback = await read_back(live_config, capture.session_id)

    asyncio.run(run())
    return capture


@pytest.mark.live
class TestOpenAILive(LiveContract):
    agent_type = "OpenAI"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, live_capture: Capture) -> Capture:
        return live_capture
