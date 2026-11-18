# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Smoke: Google ADK adapter (``agentic_context_adk``).

Wiring under test is the documented pattern: ``LlmAgent(tools=[*adapter.default_tools(),
*my_tools], **adapter.callback_kwargs())`` driven by a real ADK ``Runner``. The
ADK session is created with ``user_id=owner`` so the adapter's owner
auto-detect binds the Beacon owner.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest

from harness import (
    AGENT_ID, OWNER_ID, SYSTEM_PROMPT, TOOL_NAME, TURNS,
    Capture, FakeAuth, LiveConfig, LiveContract, MockBeacon, RecordingPassThrough,
    SmokeContract, lookup_order, new_session_id, read_back, script_next,
)

pytestmark = pytest.mark.adk

pytest.importorskip("google.adk")
pytest.importorskip("agentic_context_adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_context_adk import GoogleAdkAdapter  # noqa: E402

APP_NAME = "smoke-adk"


def _progress(contents: list[types.Content]) -> tuple[int, bool]:
    """(user text turns seen, whether the latest one already has a tool result)."""
    users, answered = 0, False
    for content in contents:
        parts = content.parts or []
        if any(p.function_response for p in parts):
            answered = True
        elif content.role == "user" and any(p.text for p in parts):
            users, answered = users + 1, False
    return users, answered


class ScriptedLlm(BaseLlm):
    """Offline model: calls lookup_order on turn 1, then replies from the script."""

    model: str = "smoke-scripted"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        action, payload = script_next(*_progress(llm_request.contents))
        if action == "tool":
            part = types.Part(function_call=types.FunctionCall(name=TOOL_NAME, args=payload))
        else:
            part = types.Part(text=payload)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def drive(adapter: GoogleAdkAdapter, model: BaseLlm, capture: Capture) -> None:
    agent = LlmAgent(
        name="smoke_adk",
        model=model,
        instruction=SYSTEM_PROMPT,
        tools=[*adapter.default_tools(), lookup_order],
        **adapter.callback_kwargs(),
    )
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name=APP_NAME, user_id=capture.owner_id, session_id=capture.session_id,
    )
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=sessions)
    try:
        for turn in TURNS:
            reply = ""
            async for event in runner.run_async(
                user_id=capture.owner_id,
                session_id=capture.session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=turn.user)]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    reply = "".join(p.text or "" for p in event.content.parts)
            capture.replies.append(reply)
    finally:
        await adapter.close()


# ─── Tier 1: offline ─────────────────────────────────────────────────


def _run_offline() -> Capture:
    capture = Capture("adk", new_session_id("adk"), OWNER_ID, AGENT_ID)
    with MockBeacon(capture):
        adapter = GoogleAdkAdapter(AGENT_ID, owner_id=OWNER_ID, auth=FakeAuth())
        asyncio.run(drive(adapter, ScriptedLlm(), capture))
    return capture


@pytest.fixture(scope="module")
def offline_capture() -> Capture:
    return _run_offline()


KNOWN_GAPS: dict[str, str] = {}


@pytest.mark.offline
class TestAdkOffline(SmokeContract):
    agent_type = "GoogleADK"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, offline_capture: Capture) -> Capture:
        return offline_capture


# ─── Tier 2: live ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_capture(live_config: LiveConfig) -> Capture:
    capture = Capture("adk", new_session_id("adk"), live_config.owner_id, live_config.agent_id)

    async def run() -> None:
        with RecordingPassThrough(capture):
            adapter = GoogleAdkAdapter(
                live_config.agent_id, owner_id=live_config.owner_id, auth=live_config.auth,
            )
            await drive(adapter, ScriptedLlm(), capture)
        capture.readback = await read_back(live_config, capture.session_id)

    asyncio.run(run())
    return capture


@pytest.mark.live
class TestAdkLive(LiveContract):
    agent_type = "GoogleADK"
    known_gaps = KNOWN_GAPS

    @pytest.fixture
    def capture(self, live_capture: Capture) -> Capture:
        return live_capture
