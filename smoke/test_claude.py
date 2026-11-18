# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Smoke: Claude Agent SDK adapter (``agentic_context_claude``).

Wiring under test is the documented one-call path:
``adapter.apply_to(ClaudeAgentOptions(...), session_id=sid)`` on a ``ClaudeSDKClient``,
two turns on one client. Ingestion is driven by the adapter's Stop hook,
which reads the CLI's transcript JSONL.

On both tiers a fake CLI stands in for the ``claude`` subprocess. It talks the real
control protocol over the SDK's ``Transport`` seam: it answers ``initialize``,
calls ``lookup_order`` through the SDK's in-process MCP server
(``mcp_message``), writes a real transcript, and fires real ``hook_callback``
requests for UserPromptSubmit and Stop. The CLI is the only fake; the SDK,
the hooks, and the adapter all run for real.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from harness import (
    AGENT_ID, OWNER_ID, SYSTEM_PROMPT, TOOL_DESCRIPTION, TOOL_NAME, TURNS,
    Capture, FakeAuth, LiveConfig, LiveContract, MockBeacon,
    RecordingPassThrough, SmokeContract, lookup_order, new_session_id, read_back,
    script_next,
)

pytestmark = pytest.mark.claude

pytest.importorskip("claude_agent_sdk")
pytest.importorskip("agentic_context_claude")

from claude_agent_sdk import (  # noqa: E402
    ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, create_sdk_mcp_server, tool,
)
from claude_agent_sdk._internal.transport import Transport  # noqa: E402

from agentic_context_claude import ClaudeAdapter  # noqa: E402

SERVER_NAME = "smoke"
WIRE_TOOL_NAME = f"mcp__{SERVER_NAME}__{TOOL_NAME}"
MODEL = "claude-smoke-fake"
OFFLINE_TURN_TIMEOUT = 30.0
LIVE_TURN_TIMEOUT = 120.0  # real Beacon round trips inside the hooks


@tool(TOOL_NAME, TOOL_DESCRIPTION, {"order_id": str})
async def lookup_order_tool(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": lookup_order(args["order_id"])}]}


def smoke_mcp_server() -> Any:
    return create_sdk_mcp_server(SERVER_NAME, tools=[lookup_order_tool])


def build_options(adapter: ClaudeAdapter, sid: str, **extra: Any) -> ClaudeAgentOptions:
    base = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={SERVER_NAME: smoke_mcp_server()},
        allowed_tools=[WIRE_TOOL_NAME],
        **extra,
    )
    return adapter.apply_to(base, session_id=sid)


async def drive(adapter: ClaudeAdapter, client: ClaudeSDKClient, capture: Capture, timeout: float) -> None:
    async def one_turn(text: str) -> str:
        await client.query(text)
        result = ""
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                if msg.is_error:
                    raise AssertionError(f"turn ended in error: {msg.result or msg.errors}")
                result = msg.result or ""
        return result

    try:
        async with client:
            for turn in TURNS:
                capture.replies.append(await asyncio.wait_for(one_turn(turn.user), timeout))
    finally:
        await adapter.close()


# ─── Fake CLI transport ──────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class FakeCLITransport(Transport):
    """Plays the ``claude`` CLI side of the stream-json control protocol."""

    def __init__(self, transcript_dir: Path) -> None:
        self.cli_session_id = str(uuid4())
        self.transcript_path = transcript_dir / f"{self.cli_session_id}.jsonl"
        self.hook_ids: dict[str, list[str]] = {}
        self.errors: list[str] = []
        self._out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._ready = False
        self._req = 0
        self._users_seen = 0

    # ── Transport ABC ────────────────────────────────────────────────

    async def connect(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        pass

    async def close(self) -> None:
        self._ready = False
        for task in self._tasks:
            task.cancel()
        await self._out.put(None)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            msg = await self._out.get()
            if msg is None:
                return
            yield msg

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if line.strip():
                self._on_sdk_message(json.loads(line))

    # ── SDK → CLI ────────────────────────────────────────────────────

    def _on_sdk_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "control_request":
            request = msg["request"]
            response: dict[str, Any] = {}
            if request.get("subtype") == "initialize":
                for event, matchers in (request.get("hooks") or {}).items():
                    for m in matchers:
                        self.hook_ids.setdefault(event, []).extend(m["hookCallbackIds"])
                response = {"commands": [], "models": []}
            self._emit({
                "type": "control_response",
                "response": {"subtype": "success", "request_id": msg["request_id"], "response": response},
            })
        elif kind == "control_response":
            response = msg["response"]
            waiter = self._waiters.pop(response.get("request_id"), None)
            if waiter is not None and not waiter.done():
                waiter.set_result(response)
        elif kind == "user":
            task = asyncio.get_running_loop().create_task(self._run_turn(msg["message"]["content"]))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    # ── CLI → SDK ────────────────────────────────────────────────────

    def _emit(self, msg: dict[str, Any]) -> None:
        self._out.put_nowait(msg)

    async def _control(self, request: dict[str, Any]) -> dict[str, Any]:
        self._req += 1
        request_id = f"cli_req_{self._req}"
        waiter: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = waiter
        self._emit({"type": "control_request", "request_id": request_id, "request": request})
        response = await waiter
        if response.get("subtype") != "success":
            raise RuntimeError(f"SDK rejected {request['subtype']}: {response.get('error')}")
        return response.get("response") or {}

    async def _fire_hooks(self, event: str, payload: dict[str, Any]) -> None:
        base = {
            "hook_event_name": event,
            "session_id": self.cli_session_id,
            "transcript_path": str(self.transcript_path),
            "cwd": str(self.transcript_path.parent),
        }
        for callback_id in self.hook_ids.get(event, []):
            await self._control({
                "subtype": "hook_callback", "callback_id": callback_id,
                "input": {**base, **payload}, "tool_use_id": None,
            })

    async def _mcp(self, method: str, params: dict[str, Any], *, notify: bool = False) -> dict[str, Any]:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notify:
            self._req += 1
            message["id"] = self._req
        out = await self._control({"subtype": "mcp_message", "server_name": SERVER_NAME, "message": message})
        reply = out.get("mcp_response") or {}
        if "error" in reply:
            raise RuntimeError(f"MCP {method} failed: {reply['error']}")
        return reply.get("result") or {}

    async def _call_tool(self, args: dict[str, Any]) -> str:
        await self._mcp("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "fake-claude-cli", "version": "0"},
        })
        await self._mcp("notifications/initialized", {}, notify=True)
        result = await self._mcp("tools/call", {"name": TOOL_NAME, "arguments": args})
        return " ".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")

    def _transcript(self, kind: str, message: dict[str, Any]) -> str:
        uuid = str(uuid4())
        line = {
            "type": kind, "uuid": uuid, "sessionId": self.cli_session_id,
            "timestamp": _now(), "message": message,
        }
        with self.transcript_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return uuid

    def _assistant(self, content: list[dict[str, Any]], stop_reason: str) -> None:
        message = {
            "id": f"msg_{uuid4().hex[:12]}", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason,
        }
        uuid = self._transcript("assistant", message)
        self._emit({
            "type": "assistant", "message": message, "parent_tool_use_id": None,
            "session_id": self.cli_session_id, "uuid": uuid,
        })

    async def _run_turn(self, prompt: str) -> None:
        try:
            await self._fire_hooks("UserPromptSubmit", {"prompt": prompt})
            self._users_seen += 1
            self._transcript("user", {"role": "user", "content": prompt})
            tool_answered = False
            while True:
                action, payload = script_next(self._users_seen, tool_answered)
                if action == "reply":
                    break
                tool_use_id = f"toolu_{uuid4().hex[:12]}"
                self._assistant(
                    [{"type": "tool_use", "id": tool_use_id, "name": WIRE_TOOL_NAME, "input": payload}],
                    "tool_use",
                )
                output = await self._call_tool(payload)
                result_message = {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": output}],
                }]}
                uuid = self._transcript("user", result_message)
                self._emit({
                    "type": "user", "message": result_message, "parent_tool_use_id": None,
                    "session_id": self.cli_session_id, "uuid": uuid,
                })
                tool_answered = True
            self._assistant([{"type": "text", "text": payload}], "end_turn")
            # The real CLI runs Stop hooks before it reports the result.
            await self._fire_hooks("Stop", {"stop_hook_active": False, "last_assistant_message": payload})
            self._emit({
                "type": "result", "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
                "is_error": False, "num_turns": 1, "session_id": self.cli_session_id,
                "result": payload, "usage": {},
            })
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — surfaced as an error result
            self.errors.append(repr(e))
            self._emit({
                "type": "result", "subtype": "error_during_execution", "duration_ms": 1,
                "duration_api_ms": 1, "is_error": True, "num_turns": 1,
                "session_id": self.cli_session_id, "result": f"fake CLI: {e!r}", "usage": {},
            })


KNOWN_GAPS: dict[str, str] = {}


class _ClaudeContract:
    agent_type = "ClaudeAgentSDK"
    tool_wire_name = WIRE_TOOL_NAME
    known_gaps = KNOWN_GAPS

    def test_fake_cli_clean(self, capture: Capture) -> None:
        """The fake Claude CLI hit no errors and the adapter's hooks registered with it."""
        assert not capture.notes, capture.notes


async def run_fake_cli(capture: Capture, auth: Any, timeout: float) -> None:
    """Both tiers: the real SDK, hooks and adapter against the fake CLI."""
    with tempfile.TemporaryDirectory(prefix="smoke-claude-") as tmp:
        transport = FakeCLITransport(Path(tmp))
        adapter = ClaudeAdapter(capture.agent_id, owner_id=capture.owner_id, auth=auth)
        client = ClaudeSDKClient(
            build_options(adapter, capture.session_id), transport=transport,
        )
        await drive(adapter, client, capture, timeout)
        if transport.errors:
            capture.notes.extend(transport.errors)
        missing = {"UserPromptSubmit", "Stop"} - set(transport.hook_ids)
        if missing:
            capture.notes.append(f"hooks never registered with the CLI: {sorted(missing)}")


# ─── Tier 1: offline ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def offline_capture() -> Capture:
    capture = Capture("claude", new_session_id("claude"), OWNER_ID, AGENT_ID)
    with MockBeacon(capture):
        asyncio.run(run_fake_cli(capture, FakeAuth(), OFFLINE_TURN_TIMEOUT))
    return capture


@pytest.mark.offline
class TestClaudeOffline(_ClaudeContract, SmokeContract):
    @pytest.fixture
    def capture(self, offline_capture: Capture) -> Capture:
        return offline_capture


# ─── Tier 2: live ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_capture(live_config: LiveConfig) -> Capture:
    capture = Capture("claude", new_session_id("claude"), live_config.owner_id, live_config.agent_id)

    async def run() -> None:
        with RecordingPassThrough(capture):
            await run_fake_cli(capture, live_config.auth, LIVE_TURN_TIMEOUT)
        capture.readback = await read_back(live_config, capture.session_id)

    asyncio.run(run())
    return capture


@pytest.mark.live
class TestClaudeLive(_ClaudeContract, LiveContract):
    @pytest.fixture
    def capture(self, live_capture: Capture) -> Capture:
        return live_capture
