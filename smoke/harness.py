# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Shared smoke-test harness: one scenario, one Beacon recorder, one contract.

Every adapter's smoke module drives the same two-turn scenario through its
real framework and hands the captured Beacon traffic to the same checks, so
adapter drift shows up as a failing check rather than a silent difference.

Both tiers use the same scripted fake LLM, so every run sees the same model
replies and the checks are deterministic. No real LLM is ever called.

* Offline tier: mocked Beacon. No credentials.
* Live tier: real auth + real dev Beacon. Wire traffic is passed through and
  recorded, so the same checks run against Beacon's real responses.

Recorded calls never keep the bearer token, only whether one was sent, so a
failing check can't print it.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import respx

REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── Scenario ─────────────────────────────────────────────────────────

AGENT_ID = "SmokeAgent"
OWNER_ID = "005SMOKE00000000AA"
BASE_URL = "https://beacon.smoke.invalid"
FAKE_TOKEN = "smoke-token"

ORDER_ID = "A-1001"
TOOL_NAME = "lookup_order"
TOOL_DESCRIPTION = "Look up the shipping status of a customer order by its order id."
# What the mocked Beacon returns as the customer's fast context.
SEEDED_MEMORY = "Customer prefers email updates."
SYSTEM_PROMPT = "You are a customer-support agent."


@dataclass(frozen=True)
class Turn:
    user: str
    calls_tool: bool
    # What the scripted fake LLM answers for this turn (both tiers).
    reply: str


TURNS: tuple[Turn, ...] = (
    Turn(
        user=f"Where is my order {ORDER_ID}?",
        calls_tool=True,
        reply=f"Your order {ORDER_ID} shipped on 2026-11-10.",
    ),
    Turn(
        user="Thanks! Please send me updates by email.",
        calls_tool=False,
        reply="Noted, I will send you email updates.",
    ),
)


def lookup_order(order_id: str) -> str:
    """Look up the shipping status of a customer order by its order id."""
    return f"Order {order_id} shipped on 2026-11-10 via UPS."


TOOL_ARGS = {"order_id": ORDER_ID}
TOOL_OUTPUT = lookup_order(ORDER_ID)


def script_next(user_turns_seen: int, tool_answered: bool) -> tuple[str, Any]:
    """What the scripted fake LLM does next.

    ``user_turns_seen`` is how many user messages are in the model input;
    ``tool_answered`` is whether the latest one already has a tool result
    after it. Returns ``("tool", TOOL_ARGS)`` or ``("reply", text)``. A count
    outside the scenario (a duplicated or missing user message) fails loudly
    instead of getting a plausible answer.
    """
    if not 1 <= user_turns_seen <= len(TURNS):
        raise AssertionError(
            f"fake LLM saw {user_turns_seen} user messages; the scenario has {len(TURNS)} turns"
        )
    turn = TURNS[user_turns_seen - 1]
    if turn.calls_tool and not tool_answered:
        return "tool", TOOL_ARGS
    return "reply", turn.reply


def new_session_id(adapter: str) -> str:
    return f"smoke-{adapter}-{uuid4().hex[:12]}"


# ─── Offline auth double ──────────────────────────────────────────────


class FakeAuth:
    """Duck-typed ContextAuth: fixed token + base_url, no network."""

    default_owner_type = "individual"

    def get_token(self) -> tuple[str, str]:
        return FAKE_TOKEN, BASE_URL

    async def aget_token(self) -> tuple[str, str]:
        return self.get_token()

    def invalidate(self) -> None:
        pass

    def attach(self, client: Any) -> None:
        pass  # the token never rotates, so attached clients need no refresh


# ─── Beacon traffic capture ───────────────────────────────────────────

_PATH_KINDS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("POST", "create_conversation", re.compile(r"^/api/v1/agentic/conversations$")),
    ("POST", "create_session", re.compile(r"^/api/v1/agentic/conversations/[^/]+/sessions$")),
    ("POST", "add_participants", re.compile(r"^/api/v1/agentic/conversations/[^/]+/sessions/[^/]+/participants$")),
    ("POST", "add_entries", re.compile(r"^/api/v1/agentic/conversations/[^/]+/entries$")),
    ("POST", "close", re.compile(r"^/api/v1/agentic/conversations/.+/close$")),
    ("POST", "get_context", re.compile(r"^/api/v1/agentic/context$")),
    ("GET", "context_pack_catalog", re.compile(r"^/api/v1/agentic/runtime-context-packs$")),
    ("GET", "system_prompt", re.compile(r"^/api/v1/agentic/prompts/system/.+$")),
)

BEACON_PATH = re.compile(r"/api/v1/agentic/")

LIFECYCLE_KINDS = ("create_conversation", "create_session", "add_participants")
# 202 = accepted (Beacon answers add_participants and /entries this way);
# 409 = already exists, which the manager treats as success.
LIFECYCLE_OK = frozenset({200, 201, 202, 204, 409})
ENTRIES_OK = frozenset({200, 201, 202, 204})
# Statuses the SDK's session manager retries (fixed retry policy).
RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})


def classify(method: str, path: str) -> str:
    for m, kind, pattern in _PATH_KINDS:
        if m == method and pattern.match(path):
            return kind
    return "unexpected"


def _bearer(request: httpx.Request) -> str:
    """``"fake"``, ``"real"`` or ``"missing"``; the token itself is never kept."""
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer ") or len(header) <= len("Bearer "):
        return "missing"
    return "fake" if header == f"Bearer {FAKE_TOKEN}" else "real"


@dataclass
class Call:
    kind: str
    method: str
    path: str
    body: Any
    bearer: str
    status: int | None = None  # None: the send raised before a response came back
    response_text: str = ""

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in RETRYABLE


@dataclass(repr=False)
class Capture:
    """Everything one scenario run sent to Beacon, plus what the run returned."""

    adapter: str
    session_id: str
    owner_id: str
    agent_id: str
    calls: list[Call] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    readback: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        # Short on purpose: pytest prints this in assertion explanations.
        return f"Capture({self.adapter}, {self.session_id}, {len(self.calls)} calls)"

    def of(self, kind: str) -> list[Call]:
        return [c for c in self.calls if c.kind == kind]

    def logical(self, kind: str) -> list[Call]:
        """Calls of ``kind`` with retried attempts collapsed.

        An attempt that got a retryable status (or raised) and was re-sent
        with the same path and body is dropped, so one transient Beacon error
        doesn't look like a duplicate call.
        """
        calls = self.of(kind)
        return [
            c for i, c in enumerate(calls)
            if not (c.retryable and any(
                later.path == c.path and later.body == c.body for later in calls[i + 1:]
            ))
        ]

    def posts(self) -> list[list[dict[str, Any]]]:
        """The entries of each logical ``/entries`` POST, in send order."""
        return [list((c.body or {}).get("entries") or []) for c in self.logical("add_entries")]

    @property
    def entries(self) -> list[dict[str, Any]]:
        return [e for post in self.posts() for e in post]

    def summary(self) -> str:
        lines = [
            f"{c.method} {c.kind} -> {c.status if c.status is not None else c.response_text}"
            for c in self.calls
        ]
        types = [e.get("type") for e in self.entries]
        return "\n".join([*lines, f"entry types: {types}"])


def _decode_body(request: httpx.Request) -> Any:
    if not request.content:
        return None
    try:
        return json.loads(request.content)
    except ValueError:
        return request.content.decode("utf-8", "replace")


def _call(request: httpx.Request) -> Call:
    path = request.url.path
    return Call(classify(request.method, path), request.method, path, _decode_body(request), _bearer(request))


class MockBeacon:
    """respx router that answers every Beacon endpoint and records each call.

    Unknown Beacon paths get a 599 and are recorded as ``unexpected``; any
    non-Beacon host is rejected, so a smoke run can never leak to the network.
    """

    def __init__(self, capture: Capture) -> None:
        self.capture = capture
        self.router = respx.mock(assert_all_called=False, assert_all_mocked=True)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        call = _call(request)
        response = self._response_for(call)
        call.status = response.status_code
        call.response_text = response.text
        self.capture.calls.append(call)
        return response

    def _response_for(self, call: Call) -> httpx.Response:
        if call.kind in LIFECYCLE_KINDS:
            return httpx.Response(201, json={})
        if call.kind == "add_entries":
            return httpx.Response(201, json={"entries": []})
        if call.kind == "get_context":
            return httpx.Response(200, json={"userContext": [{"entry": SEEDED_MEMORY}]})
        if call.kind == "context_pack_catalog":
            return httpx.Response(200, json={"contextPacks": [], "hasMore": False})
        if call.kind == "system_prompt":
            # Definitive miss: adapters fall back to the bundled instruction.
            return httpx.Response(404, json={})
        return httpx.Response(599, json={"error": f"smoke: unmocked Beacon path {call.path}"})

    def __enter__(self) -> MockBeacon:
        self.router.__enter__()
        self.router.route(host="beacon.smoke.invalid").mock(side_effect=self._respond)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.router.__exit__(*exc)


class RecordingPassThrough:
    """Live tier: let every request through, record the Beacon ones.

    Wraps httpx's ``Client.send`` / ``AsyncClient.send`` instead of using
    respx: respx keeps no response for pass-through calls and logs each one
    twice, so statuses and call counts would both be wrong. A send that
    raises is recorded too (``status=None``), so retried attempts stay visible.
    """

    def __init__(self, capture: Capture) -> None:
        self.capture = capture
        self._originals: tuple[Any, Any] | None = None

    def _record(
        self, request: httpx.Request, response: httpx.Response | None, error: BaseException | None,
    ) -> None:
        if not BEACON_PATH.search(request.url.path):
            return
        call = _call(request)
        if response is not None:
            call.status = response.status_code
            try:
                call.response_text = response.text[:2000]
            except httpx.ResponseNotRead:  # streamed response the caller hasn't read
                call.response_text = ""
        else:
            call.response_text = f"send raised {type(error).__name__}"
        self.capture.calls.append(call)

    def __enter__(self) -> RecordingPassThrough:
        sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send
        record = self._record

        def send(client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
            try:
                response = sync_send(client, request, *args, **kwargs)
            except Exception as e:
                record(request, None, e)
                raise
            record(request, response, None)
            return response

        async def asend(
            client: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any,
        ) -> httpx.Response:
            try:
                response = await async_send(client, request, *args, **kwargs)
            except Exception as e:
                record(request, None, e)
                raise
            record(request, response, None)
            return response

        self._originals = (sync_send, async_send)
        httpx.Client.send, httpx.AsyncClient.send = send, asend  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._originals is not None
        httpx.Client.send, httpx.AsyncClient.send = self._originals  # type: ignore[method-assign]


# ─── Contract checks ──────────────────────────────────────────────────

EXPECTED_TYPES = [
    "user_message", "tool_call", "tool_result", "agent_response",
    "user_message", "agent_response",
]
# Entry types an adapter may add between the expected ones.
OPTIONAL_TYPES = frozenset({"agent_reasoning"})


def _turn_segments(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split entries at each user_message (a leading non-user run is its own segment)."""
    segments: list[list[dict[str, Any]]] = []
    for e in entries:
        if e.get("type") == "user_message" or not segments:
            segments.append([e])
        else:
            segments[-1].append(e)
    return segments


class SmokeContract:
    """Checks shared by every adapter and both tiers.

    Subclasses provide a ``capture`` fixture (one scenario run) and set:

    * ``agent_type`` — the ``AiAgentType`` label the adapter should send.
    * ``tool_wire_name`` — the tool name as the framework reports it.
    * ``known_gaps`` — ``{check_name: reason}`` for checks the adapter is
      known to fail today. A known gap that starts passing FAILS the run, so
      the entry gets removed rather than rotting.
    """

    agent_type: str = ""
    tool_wire_name: str = TOOL_NAME
    known_gaps: dict[str, str] = {}
    live: bool = False

    def _expected_failure(self, name: str, capture: Capture) -> str | None:
        """A reason a failing check should xfail for reasons outside the adapter."""
        return None

    def _check(self, name: str, fn: Any, capture: Capture) -> None:
        gap = self.known_gaps.get(name)
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a KeyError in a check is still a failed check
            if gap:
                pytest.xfail(f"known gap: {gap}")
            expected = self._expected_failure(name, capture)
            if expected:
                pytest.xfail(expected)
            raise AssertionError(
                f"{type(e).__name__}: {e}\n--- beacon traffic ---\n{capture.summary()}"
            ) from e
        if gap:
            pytest.fail(f"known gap {name!r} now passes; remove it from known_gaps ({gap})")

    # ── lifecycle ────────────────────────────────────────────────────

    def test_lifecycle_created_once(self, capture: Capture) -> None:
        """The conversation, the session and the participant list are each created exactly once."""
        def check() -> None:
            for kind in LIFECYCLE_KINDS:
                n = len(capture.logical(kind))
                assert n == 1, f"expected exactly one {kind}, saw {n}"
        self._check("lifecycle_created_once", check, capture)

    def test_lifecycle_accepted(self, capture: Capture) -> None:
        """Beacon accepted every setup call (2xx, or 409 for already exists)."""
        def check() -> None:
            for kind in LIFECYCLE_KINDS:
                calls = capture.logical(kind)
                assert calls, f"no {kind} call reached Beacon"
                for call in calls:
                    assert call.status in LIFECYCLE_OK, (
                        f"{kind} -> {call.status}: {call.response_text[:300]}"
                    )
        self._check("lifecycle_accepted", check, capture)

    def test_conversation_owner_set(self, capture: Capture) -> None:
        """The conversation uses our session id and is owned by the right customer."""
        def check() -> None:
            calls = capture.logical("create_conversation")
            assert calls, "no create_conversation call"
            body = calls[0].body or {}
            assert body.get("conversationId") == capture.session_id, body.get("conversationId")
            assert body.get("conversationOwnerId") == capture.owner_id, (
                f"conversationOwnerId={body.get('conversationOwnerId')!r}, owner={capture.owner_id!r}"
            )
        self._check("conversation_owner_set", check, capture)

    def test_no_close_calls(self, capture: Capture) -> None:
        """The adapter never closes the session on Beacon."""
        def check() -> None:
            closes = [c.path for c in capture.of("close")]
            assert not closes, f"sessions must never be closed on Beacon: {closes}"
        self._check("no_close_calls", check, capture)

    def test_no_unexpected_calls(self, capture: Capture) -> None:
        """Nothing calls a Beacon path outside the known endpoints."""
        def check() -> None:
            bad = [f"{c.method} {c.path}" for c in capture.of("unexpected")]
            assert not bad, f"Beacon calls outside the known endpoints: {bad}"
        self._check("no_unexpected_calls", check, capture)

    def test_auth_headers(self, capture: Capture) -> None:
        """Every Beacon call carries a bearer token (the fake one offline, a real one live)."""
        def check() -> None:
            assert capture.calls, "no Beacon calls captured"
            want = "real" if self.live else "fake"
            for call in capture.calls:
                assert call.bearer != "missing", f"{call.kind}: no bearer token"
                assert call.bearer == want, f"{call.kind}: {call.bearer} bearer token, expected {want}"
        self._check("auth_headers", check, capture)

    def test_agent_type_label(self, capture: Capture) -> None:
        """The agent participant is labelled with this framework's AiAgentType."""
        def check() -> None:
            agent = _agent_participant(capture)
            got = (agent.get("variable") or {}).get("AiAgentType")
            assert got == self.agent_type, f"AiAgentType={got!r}, expected {self.agent_type!r}"
        self._check("agent_type_label", check, capture)

    # ── entries ──────────────────────────────────────────────────────

    def test_entries_accepted(self, capture: Capture) -> None:
        """Beacon accepted every /entries POST."""
        def check() -> None:
            calls = capture.logical("add_entries")
            assert calls, "no add_entries call reached Beacon"
            for call in calls:
                assert call.status in ENTRIES_OK, (
                    f"add_entries -> {call.status}: {call.response_text[:300]}"
                )
        self._check("entries_accepted", check, capture)

    def test_entry_order(self, capture: Capture) -> None:
        """The saved entries are exactly the scenario's types, in order, with no extras or gaps."""
        def check() -> None:
            types = [e.get("type") for e in capture.entries if e.get("type") not in OPTIONAL_TYPES]
            assert types == EXPECTED_TYPES, f"entry types {types}, expected {EXPECTED_TYPES}"
        self._check("entry_order", check, capture)

    def test_user_messages_owned(self, capture: Capture) -> None:
        """The customer's messages are saved word for word and marked as the owner's."""
        def check() -> None:
            users = [e for e in capture.entries if e.get("type") == "user_message"]
            assert [u.get("content") for u in users] == [t.user for t in TURNS]
            for u in users:
                assert u.get("participantId") == capture.owner_id, (
                    f"user_message participantId={u.get('participantId')!r}, owner={capture.owner_id!r}"
                )
                assert u.get("role") == "owner", f"user_message role={u.get('role')!r}"
        self._check("user_messages_owned", check, capture)

    def test_tool_traces(self, capture: Capture) -> None:
        """The tool use is saved once, as a call and a result, with its name, input and output."""
        def check() -> None:
            calls = [e for e in capture.entries if e.get("type") == "tool_call"]
            results = [e for e in capture.entries if e.get("type") == "tool_result"]
            assert len(calls) == 1 and len(results) == 1, (
                f"expected one tool_call and one tool_result, got {len(calls)} and {len(results)}"
            )
            for e in (*calls, *results):
                assert e.get("category") == "Trace", f"{e.get('type')} category={e.get('category')!r}"
                assert not e.get("content"), "tool payloads must ride in variable, not content"
            call_var = calls[0].get("variable") or {}
            result_var = results[0].get("variable") or {}
            assert call_var.get("Name") == self.tool_wire_name, f"tool_call Name={call_var.get('Name')!r}"
            assert ORDER_ID in (call_var.get("InputValueText") or ""), call_var
            assert TOOL_OUTPUT in (result_var.get("OutputValueText") or ""), result_var
        self._check("tool_traces", check, capture)

    def test_agent_responses(self, capture: Capture) -> None:
        """The agent's replies are saved word for word and marked as the agent's."""
        def check() -> None:
            responses = [e for e in capture.entries if e.get("type") == "agent_response"]
            got = [r.get("content") for r in responses]
            assert got == [t.reply for t in TURNS], f"agent_response contents {got}"
            agent_id = _agent_participant(capture).get("id")
            for r in responses:
                assert r.get("category") == "Message", f"agent_response category={r.get('category')!r}"
                assert r.get("participantId") == agent_id, (
                    f"agent_response participantId={r.get('participantId')!r}, agent participant={agent_id!r}"
                )
        self._check("agent_responses", check, capture)

    def test_framework_replies(self, capture: Capture) -> None:
        """The framework returned the scripted replies to the caller, with the adapter in the loop."""
        def check() -> None:
            assert capture.replies == [t.reply for t in TURNS], f"replies {capture.replies}"
        self._check("framework_replies", check, capture)

    def test_turns_grouped(self, capture: Capture) -> None:
        """Every entry carries its turn's interactionId, and no two turns share one."""
        def check() -> None:
            segments = _turn_segments(capture.entries)
            assert len(segments) == len(TURNS), f"{len(segments)} turns in the entries, expected {len(TURNS)}"
            ids = []
            for seg in segments:
                assert seg[0].get("type") == "user_message", f"turn starts with {seg[0].get('type')!r}"
                iid = seg[0].get("interactionId")
                assert iid, "user_message has no interactionId"
                stray = [(e.get("type"), e.get("interactionId")) for e in seg if e.get("interactionId") != iid]
                assert not stray, f"entries outside their turn's interactionId {iid!r}: {stray}"
                ids.append(iid)
            assert len(set(ids)) == len(ids), f"turns share an interactionId: {ids}"
        self._check("turns_grouped", check, capture)

    def test_one_entries_post_per_turn(self, capture: Capture) -> None:
        """Each turn is sent in exactly one /entries POST."""
        def check() -> None:
            # Judged by content, not timing: async managers post in the background.
            posts = capture.posts()
            assert len(posts) == len(TURNS), f"{len(posts)} /entries POSTs for {len(TURNS)} turns"
            for i, post in enumerate(posts):
                users = sum(e.get("type") == "user_message" for e in post)
                assert users == 1 and post[0].get("type") == "user_message", (
                    f"POST {i} does not hold exactly one whole turn: {[e.get('type') for e in post]}"
                )
        self._check("one_entries_post_per_turn", check, capture)


def _agent_participant(capture: Capture) -> dict[str, Any]:
    calls = capture.logical("add_participants")
    assert calls, "no add_participants call"
    participants = (calls[0].body or {}).get("participants") or []
    agents = [p for p in participants if p.get("type") == "agent"]
    assert len(agents) == 1, f"expected one agent participant, got {participants}"
    return agents[0]


# Some dev Beacon tenants reject every /entries POST server-side with this
# signature (lifecycle and recall still work). Not an adapter bug, so live
# entry checks xfail on it instead of failing every adapter.
BEACON_ENTRIES_REJECTED = "reject-duplicate-conversation"


class LiveContract(SmokeContract):
    """Live tier adds a read-back through Beacon's GET conversation."""

    live = True

    def _expected_failure(self, name: str, capture: Capture) -> str | None:
        posts = capture.logical("add_entries")
        rejected = [c for c in posts if BEACON_ENTRIES_REJECTED in c.response_text]
        if not rejected:
            return None
        reason = f"Beacon rejected /entries server-side ({BEACON_ENTRIES_REJECTED})"
        if name == "entries_accepted":
            # Only when every failing POST is one of the rejected ones.
            failing = [c for c in posts if c.status not in ENTRIES_OK]
            return reason if all(c in rejected for c in failing) else None
        if name == "readback":
            return reason
        return None

    def test_readback(self, capture: Capture) -> None:
        """The conversation read back from Beacon holds every user message and reply."""
        def check() -> None:
            rb = capture.readback
            assert rb is not None, "GET conversation returned nothing"
            text = json.dumps(rb)
            missing = [s for s in readback_expected() if s not in text]
            assert not missing, f"missing from Beacon read-back: {missing}; got {text[:1000]}"
        self._check("readback", check, capture)


# ─── Live environment ─────────────────────────────────────────────────

ENV_FILE = Path(os.environ.get("SMOKE_ENV_FILE", REPO_ROOT / "examples" / "ace_agent.env"))
TOKEN_CACHE = Path.home() / ".cache" / "agentic_context_demos" / "refresh_token.json"
READBACK_TIMEOUT = 15.0


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """``KEY=value`` lines, parsed like the examples/ demos' loader."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if value and value[0] not in ("'", '"') and "#" in value:
            value = value.split("#", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


@contextmanager
def env_file_applied(path: Path = ENV_FILE) -> Iterator[None]:
    """Expose the env file through os.environ (the shell wins), then restore it.

    The SDK reads a few settings (e.g. ``CDP_DATASPACE``) from os.environ, so
    the live run needs them there, but only while it runs.
    """
    added = {k: v for k, v in read_env_file(path).items() if k not in os.environ}
    os.environ.update(added)
    try:
        yield
    finally:
        for key in added:
            os.environ.pop(key, None)


def live_enabled() -> bool:
    return os.environ.get("SMOKE_LIVE") == "1"


def require_env(env: Mapping[str, str], *names: str) -> dict[str, str]:
    missing = [n for n in names if not env.get(n)]
    if missing:
        pytest.skip(f"live smoke needs {', '.join(missing)} (set in {ENV_FILE.name} or the shell)")
    return {n: env[n] for n in names}


@dataclass
class LiveConfig:
    auth: Any
    agent_id: str
    owner_id: str


def build_live_config(env: Mapping[str, str] = os.environ) -> LiveConfig:
    """Resolve Beacon auth, or skip with the reason."""
    from agentic_context import ContextAuth, ReauthRequired, UserContextAuth

    base = require_env(env, "SF_CONSUMER_KEY", "SF_LOGIN_URL", "BEACON_AGENT_ID")
    agent_id = base["BEACON_AGENT_ID"]
    mode = env.get("SMOKE_AUTH") or ("ent" if env.get("SF_PRIVATE_KEY_PATH") else "emp")
    if mode not in ("ent", "emp"):
        pytest.fail(f"SMOKE_AUTH must be 'ent' or 'emp', got {mode!r}", pytrace=False)

    if mode == "ent":
        ent = require_env(env, "SF_PRIVATE_KEY_PATH", "SF_SUBJECT", "BEACON_OWNER_ID")
        owner_id = ent["BEACON_OWNER_ID"]
        auth: Any = ContextAuth(
            private_key=Path(ent["SF_PRIVATE_KEY_PATH"]).expanduser().read_text(),
            consumer_key=base["SF_CONSUMER_KEY"],
            subject=ent["SF_SUBJECT"],
            login_url=base["SF_LOGIN_URL"],
            aid=agent_id,
            xid=owner_id,
        )
    else:
        # Non-interactive: reuse the demos' cached refresh token, never a browser.
        try:
            refresh = json.loads(TOKEN_CACHE.read_text())["refresh_token"]
        except (OSError, ValueError, KeyError):
            pytest.skip(
                f"SMOKE_AUTH=emp needs a cached refresh token at {TOKEN_CACHE}; "
                "run an examples/ demo once to log in, or use SMOKE_AUTH=ent"
            )
        try:
            auth = UserContextAuth(
                refresh_token=refresh,
                consumer_key=base["SF_CONSUMER_KEY"],
                login_url=base["SF_LOGIN_URL"],
                aid=agent_id,
                on_refresh_token=_save_refresh_token,
            )
            auth.get_token()
        except ReauthRequired:
            pytest.skip("cached refresh token was revoked; re-run an examples/ demo to log in")
        owner_id = env.get("BEACON_OWNER_ID") or auth.user_id

    return LiveConfig(auth=auth, agent_id=agent_id, owner_id=owner_id)


def _save_refresh_token(token: str) -> None:
    """Persist a rotated refresh token so the next module (and the demos) can reuse it."""
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(json.dumps({"refresh_token": token}))
    try:
        TOKEN_CACHE.chmod(0o600)
    except OSError:
        pass


def readback_expected() -> list[str]:
    """Text the read-back must contain: every user message and scripted reply."""
    return [s for t in TURNS for s in (t.user, t.reply)]


async def read_back(cfg: LiveConfig, session_id: str) -> dict[str, Any] | None:
    """GET the conversation from Beacon, polling briefly while the write lands."""
    from agentic_context import AsyncRetrievalClient

    loop = asyncio.get_running_loop()
    deadline = loop.time() + READBACK_TIMEOUT
    token, base_url = await cfg.auth.aget_token()
    async with AsyncRetrievalClient(base_url=base_url, api_key=token, auth=cfg.auth) as client:
        while True:
            try:
                result: dict[str, Any] | None = await client.get_conversation(
                    session_id, cfg.owner_id, cfg.agent_id,
                    owner_type=getattr(cfg.auth, "default_owner_type", "individual"),
                )
            except Exception as e:  # noqa: BLE001 — surfaced through the readback check
                result = {"error": repr(e)}
            text = json.dumps(result)
            if all(s in text for s in readback_expected()) or loop.time() >= deadline:
                return result
            await asyncio.sleep(1.0)
