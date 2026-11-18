"""Interactive demo exercising the full LangGraph adapter loop.

What this shows:

  1. On start the adapter's `SessionFirstPolicy` fetches prior memory for
     `owner_id` from Beacon and injects it as a SystemMessage before the first
     LLM turn. If you ran this before with the same `--owner-id`, the agent
     recognises you.
  2. The LLM (Claude) can call `recall_customer_context`
     mid-turn when it needs deeper memory.
  3. Each user turn ingests one `user_message` + one `agent_response` (plus
     any tool records) to Beacon over `AsyncSessionManager`.
  4. `/quit` or Ctrl-D closes the adapter, which drains pending async POSTs.
     The next run with the same `--owner-id` picks up where we left off.

Config:

  Shares the ``examples/ace_agent.env`` file with ``ace_agent.py`` — see
  ``examples/ace_agent.env.example`` for the template. The script auto-loads
  it on startup; variables already set in the shell win over the file.

Auth (``--auth``):

  ``emp`` (default) — OAuth + PKCE loopback via :class:`UserContextAuth`, same
  model the basic demo uses. First run opens a browser; the refresh token is
  cached at ``~/.cache/agentic_context_demos/refresh_token.json`` (shared across
  all example demos so one login works for every one). The signed-in user's
  ``005…`` becomes the default owner id, so ``--owner-id`` is optional.

  ``ent`` — service-account JWT bearer via :class:`ContextAuth`. No browser;
  needs ``SF_PRIVATE_KEY_PATH`` + ``SF_SUBJECT``, and an owner id from
  ``--owner-id`` or ``$BEACON_OWNER_ID`` (ContextAuth has no signed-in user).

Output:

  Each turn opens with a ``══ turn N`` divider. Your message and the agent's
  reply are boxed (red / green), reasoning is a 💭 line, and every tool call
  is a magenta ``╔═ 🔧 TOOL CALL`` frame that closes with ✔ OK / ✖ FAILED.
  Every Beacon call is printed in full as a numbered ``▶ #n`` request and
  ``◀ #n`` response (headers, params, JSON body); calls made while a tool
  runs sit inside its frame. Pass ``--debug`` for additional httpx DEBUG
  logs (TLS handshakes, request-level timings).

Prereqs:

  - ``examples/ace_agent.env`` populated — only ``SF_CONSUMER_KEY``,
    ``SF_LOGIN_URL``, ``BEACON_AGENT_ID``, and ``ANTHROPIC_*`` are needed
    (no SF_PRIVATE_KEY_PATH / SF_SUBJECT — those are JWT-flow only).
  - The ECA must register ``http://localhost:53682/callback`` as a callback
    URL and have "Require Secret for Web Server Flow" turned off.
  - ``pip install langchain-anthropic`` (not vendored in the SDK — see note
    in root pyproject.toml on keeping framework SDKs out of our deps).
  - ``ANTHROPIC_API_KEY`` set (plus ``ANTHROPIC_BASE_URL`` for a custom
    Anthropic-compatible endpoint).

Run:

    python examples/interactive_agent/interactive_agent.py                      # fresh session_id
    python examples/interactive_agent/interactive_agent.py --session-id ses-1   # resume a chat
    python examples/interactive_agent/interactive_agent.py --auth ent           # service-account ContextAuth
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import shutil
import sys
import textwrap
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Shared auth helpers (see examples/_demo_auth.py); --auth picks one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _demo_auth import anthropic_api_key, build_auth, build_auth_service_account  # noqa: E402


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _load_env_file(path: Path) -> int:
    """Read a KEY=value / `export KEY=value` file into os.environ.

    Already-set env vars WIN — the file never overwrites an existing value.
    """
    if not path.is_file():
        return 0
    loaded = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value and value[0] not in ("'", '"') and "#" in value:
            value = value.split("#", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _apply_env_file(explicit: str | None) -> None:
    """Default search order: examples/ace_agent.env (one dir up — shared with
    ace_basic_demo.py), then .env in the current working directory."""
    here = Path(__file__).resolve().parent
    candidates = (
        [Path(explicit)] if explicit else [here.parent / "ace_agent.env", Path.cwd() / ".env"]
    )
    for p in candidates:
        if p.is_file():
            n = _load_env_file(p)
            if VERBOSE:
                print(f"[env] loaded {n} var(s) from {p}", file=sys.stderr)
            return
    if explicit:
        sys.exit(f"--env-file not found: {explicit}")


# Pre-peek --env-file AND --verbose before argparse so (a) env vars are loaded
# before _require() runs, (b) the env-load banner respects --verbose.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--env-file", default=None)
_pre.add_argument("--verbose", action="store_true", default=False)
_pre_args, _ = _pre.parse_known_args()
VERBOSE: bool = _pre_args.verbose
_apply_env_file(_pre_args.env_file)

if not VERBOSE:
    # Silence langgraph's v3-streaming BetaWarning (adapter uses astream_events
    # version="v3"). Visible with --verbose for debugging.
    import warnings
    warnings.filterwarnings(
        "ignore", message=r".*streaming protocol on Pregel is experimental.*"
    )


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"missing env var: {name}\n"
            f"Set it in examples/ace_agent.env (copy from ace_agent.env.example) "
            f"or export it in the shell."
        )
    return val


# Auth: delegated to the shared helpers in ``examples/_demo_auth.py``.
# ``--auth emp`` → build_auth (UserContextAuth); ``--auth ent`` →
# build_auth_service_account (ContextAuth).


def _parse_custom_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


def _build_graph(model_with_tools, tools):
    """Minimal 2-node graph: `agent` LLM node + `tools` ToolNode, looped.

    Compiled with `InMemorySaver` so prior-turn state (including the context
    SystemMessage the adapter injects on turn 1 under `context_mode="auto"`)
    persists across turns. Without a checkpointer each turn starts with only
    the new human message, so the model loses the injected context and reflex-
    calls `recall_customer_context` every turn to recover it. The adapter
    threads `session_id` as `configurable.thread_id`, so the checkpointer
    resumes the right conversation across input() calls.
    """

    async def agent_node(state: ChatState) -> dict[str, list[BaseMessage]]:
        if os.environ.get("DEBUG_LLM_INPUT"):
            print(f"\n[LLM state: {len(state['messages'])} messages]")
            for i, m in enumerate(state["messages"]):
                mid = getattr(m, "id", None)
                content = getattr(m, "content", "")
                content_str = str(content)[:200].replace("\n", " ")
                print(f"  [{i}] {m.__class__.__name__} id={mid!r} content={content_str}")
            tool_names = [getattr(t, "name", "?") for t in getattr(model_with_tools, "kwargs", {}).get("tools", [])] or [
                getattr(t, "name", "?") for t in tools
            ]
            print(f"[bound tools: {tool_names}]\n")
        reply = await model_with_tools.ainvoke(state["messages"])
        if os.environ.get("DEBUG_LLM_INPUT"):
            tcs = getattr(reply, "tool_calls", None) or []
            print(f"[LLM returned: {len(tcs)} tool_calls] {[tc.get('name') for tc in tcs]}")
        return {"messages": [reply]}

    builder: StateGraph = StateGraph(ChatState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


# ─── Terminal rendering ───────────────────────────────────────────────
#
# Layout per turn:
#
#   ══ turn 3 · 05:45:17 ════════════      divider
#   ┏━ 👤 YOU ━━━━━━━━━━━━━━━━━━━━━━━      user box (red)
#   💭 Let me check your claims…           agent_reasoning (italic cyan)
#   ╔═ 🔧 TOOL CALL #2 · get_context_pack  tool frame (magenta); wire calls
#   ║  ▶ #9 POST …  /  ◀ #9 404 · 210 ms   made while it is open sit inside
#   ╚═ ✖ FAILED · 0.4 s · not_ready …
#   ┏━ 🤖 AGENT ━━━━━━━━━━━━━━━━━━━━━━━     agent box (green)
#   🔧 this turn: get_context_pack(…) ✖    footer
#
# Every Beacon request/response is printed in full (numbered ▶/◀ so a late
# background response can be matched to its request). All output goes
# through _emit, which clears the spinner line, adds the open tool frame's
# "║" bar, and redraws a pending "you> " prompt below.

_ANSI = {
    "user": "\033[31;1m",      # bright red — you> prompt + user box
    "agent": "\033[32;1m",     # bright green — agent box
    "tool": "\033[35;1m",      # bright magenta — tool-call frame
    "req": "\033[34;1m",       # bright blue — ▶ request header
    "ok": "\033[32m",          # green — ◀ 2xx
    "err": "\033[31;1m",       # bright red — ◀ error, ✖ failures
    "think": "\033[36;3m",     # italic cyan — 💭 agent_reasoning
    "warn": "\033[33m",        # yellow — SDK log warnings
    "turn": "\033[1m",         # bold — turn divider
    "label": "\033[33;1m",     # bold yellow — "headers:" / "params:" labels
    "key": "\033[36m",         # cyan — header / param values
    "dim": "\033[2m",          # dim — verbose-only records
    "reset": "\033[0m",
}
_TTY = sys.stdout.isatty()
_COLOR = _TTY and os.environ.get("NO_COLOR") is None


def _color(text: str, name: str) -> str:
    if not _COLOR:
        return text
    code = _ANSI.get(name, "")
    return f"{code}{text}{_ANSI['reset']}" if code else text


# Tool hooks can fire from LangGraph worker threads, so output is serialized.
_out_lock = threading.Lock()
# Prompt currently waiting for a line (None when not reading).
_pending_prompt: str | None = None
_prompt_redrawn = False
_spinner_on = False
# Open tool frames: tool_call_id -> (display number, tool name, label, start).
_open_tools: dict[str, tuple[int, str, str, float]] = {}
_tool_seq = itertools.count(1)
_wire_seq = itertools.count(1)
_turn_tools: list[str] = []  # footer entries for the current turn


def _width() -> int:
    return max(40, min(shutil.get_terminal_size((100, 24)).columns, 120))


def _rule(head: str, fill: str, extra_cells: int = 0) -> str:
    """``head`` padded with ``fill`` to the terminal width (emoji = 2 cells)."""
    return head + fill * max(3, _width() - len(head) - extra_cells)


def _prompt_text(prompt: str) -> str:
    # Color stays on so the terminal echoes the typed utterance in red too;
    # _read_user resets it once the line is read.
    return f"{_ANSI['user']}{prompt}" if _COLOR else prompt


def _emit(*lines: str) -> None:
    """The single print path — see the layout note above."""
    global _prompt_redrawn
    text = "\n".join(lines)
    with _out_lock:
        out = []
        if _spinner_on and _TTY:
            out.append("\r\033[K")
        if _pending_prompt is not None:
            out.append((_ANSI["reset"] if _COLOR else "") + "\n")
        bar = _color("║", "tool") + "  " if _open_tools else ""
        out.append("\n".join(bar + ln for ln in text.split("\n")) + "\n")
        if _pending_prompt is not None:
            # Anything already typed stays in the terminal's line buffer and
            # is still submitted on Enter; it just isn't re-echoed.
            out.append(_prompt_text(_pending_prompt))
            _prompt_redrawn = True
        sys.stdout.write("".join(out))
        sys.stdout.flush()


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


async def _spin(label: str = "thinking") -> None:
    """Rotating spinner on the bottom line until cancelled (tty only).

    _emit clears the spinner line before printing, so logs that arrive
    mid-turn push the spinner down instead of being glued onto it.
    """
    global _spinner_on
    if not _TTY:
        await asyncio.Event().wait()  # park forever; caller cancels us
    i = 0
    try:
        while True:
            with _out_lock:
                _spinner_on = True
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                sys.stdout.write(f"\r  {frame} {label}...")
                sys.stdout.flush()
            i += 1
            await asyncio.sleep(0.1)
    finally:
        with _out_lock:
            if _spinner_on:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            _spinner_on = False


async def _stop_spinner(task: asyncio.Task) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _read_user(prompt: str) -> str:
    """Read one line from stdin without blocking the event loop.

    A blocking input() stalls the background ingests (they trip their 10 s
    ConnectTimeout). Running input() in a thread instead hands the line to
    readline, whose state gets corrupted by logs printed from the loop. So
    let the loop watch stdin and rely on the terminal's own echo/backspace.
    Raises EOFError on Ctrl-D.
    """
    global _pending_prompt, _prompt_redrawn
    loop = asyncio.get_running_loop()
    with _out_lock:
        sys.stdout.write(_prompt_text(prompt))
        sys.stdout.flush()
        _pending_prompt = prompt
        _prompt_redrawn = False
    try:
        try:
            fd = sys.stdin.fileno()
            done: asyncio.Future[str] = loop.create_future()
            loop.add_reader(fd, lambda: done.done() or done.set_result(sys.stdin.readline()))
        except (NotImplementedError, ValueError, OSError):
            # Windows proactor loop has no add_reader.
            line = await asyncio.to_thread(sys.stdin.readline)
        else:
            try:
                line = await done
            finally:
                loop.remove_reader(fd)
    finally:
        with _out_lock:
            _pending_prompt = None
            if _COLOR:
                sys.stdout.write(_ANSI["reset"])
                sys.stdout.flush()
    if not line:
        raise EOFError
    return line


def _erase_typed_line(prompt: str, typed: str) -> None:
    """Remove the echoed ``you> …`` line so the boxed copy replaces it."""
    if not (_TTY and sys.stdin.isatty()):
        return
    # After a mid-prompt redraw only the last row is the live prompt; never
    # erase further up than that or we'd eat wire-log lines.
    cols = shutil.get_terminal_size((100, 24)).columns
    rows = 1 if _prompt_redrawn else max(1, -(-(len(prompt) + len(typed)) // cols))
    sys.stdout.write("\033[F\033[K" * rows)
    sys.stdout.flush()


def _box(title: str, text: str, name: str) -> None:
    """Speaker box: ┏━ title / ┃ wrapped text / ┗━ in the speaker color."""
    w = _width()
    body: list[str] = []
    for para in text.strip().split("\n"):
        body += textwrap.wrap(para, w - 2, replace_whitespace=False) or [""]
    _emit(
        "",
        _color(_rule(f"┏━ {title} ", "━", extra_cells=1), name),
        *(_color(f"┃ {ln}", name) for ln in body),
        _color("┗" + "━" * (w - 1), name),
    )


def _turn_divider(n: int) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    _emit("", _color(_rule(f"══ turn {n} · {stamp} ", "═"), "turn"))


def _format_record(r) -> str:
    content = (getattr(r, "content", "") or "").strip()
    label = f"  [{r.event_type}] {getattr(r, 'participant_id', '-')}:"
    if not content:
        return label
    # Preserve newlines so markdown lists/code blocks stay readable; indent
    # continuation lines so they visually hang under the record label.
    lines = content.split("\n")
    return "\n".join([f"{label} {lines[0]}", *(f"    {ln}" for ln in lines[1:])])


def _pretty(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str, sort_keys=False)
    except (TypeError, ValueError):
        return repr(obj)


def _indent(text: str, pad: str = "  ") -> str:
    return "\n".join(pad + ln for ln in text.split("\n"))


# ─── Wire trace (▶ request / ◀ response) ──────────────────────────────


def _wire_label(path: str, body) -> str:
    """What a Beacon call is for, shown next to its ▶ header."""
    if not isinstance(body, dict):
        body = {}
    if path.endswith("/entries"):
        types = [e.get("type", "?") for e in body.get("entries", [])]
        return "save " + ", ".join(types) if types else "save entries"
    if path.endswith("/participants"):
        return "add participants"
    if path.endswith("/sessions"):
        return "create session"
    if path.endswith("/conversations"):
        return "create conversation"
    if path.endswith("/agentic/context"):
        pack = body.get("context_pack")
        return f"context pack {pack}" if pack else "default context (memory)"
    if "runtime-context-packs" in path:
        return "list context packs"
    if "/prompts/system/" in path:
        return "system prompt"
    return ""


def _kv_line(label: str, value) -> str:
    """``headers: {...}`` — raw dict, bold label, cyan (not dimmed) value."""
    return f"  {_color(label + ':', 'label')} {_color(str(value), 'key')}"


def _log_request(method: str, url: str, path: str, headers: dict,
                 params=None, body=None) -> tuple[int, float]:
    n = next(_wire_seq)
    label = _wire_label(path, body)
    lines = [
        "",
        _color(f"▶ #{n} {method} {url}", "req")
        + (_color(f"   · {label}", "dim") if label else ""),
        _kv_line("headers", headers),
    ]
    if params is not None:
        lines.append(_kv_line("params", params))
    if body is not None:
        lines.append(_indent(_pretty(body)))
    _emit(*lines)
    return n, time.monotonic()


def _log_response(n: int, t0: float, resp=None, error: Exception | None = None) -> None:
    ms = int((time.monotonic() - t0) * 1000)
    if error is None:
        _emit(_color(f"◀ #{n} 2xx · {ms} ms", "ok"), _indent(_pretty(resp)))
        return
    status = getattr(error, "status_code", None)
    head = f"◀ #{n} HTTP {status} · {ms} ms" if status else f"◀ #{n} ERROR · {ms} ms"
    body = getattr(error, "body", None)
    if body:
        try:
            detail = _pretty(json.loads(body))
        except ValueError:
            detail = str(body)
    else:
        detail = repr(error)
    _emit(_color(head, "err"), _color(_indent(detail), "err"))


def _install_pretty_wire_hooks() -> None:
    """Print every Beacon call in full: numbered ▶ request (headers minus
    Authorization, params, JSON body) and ◀ response (JSON body or error).

    Always on — the interactive demo is primarily a plumbing-visibility tool.
    Pass ``--debug`` on top for httpx-level logging (TLS handshakes, timings).
    """
    from agentic_context.client_async import AsyncIngestionClient
    from agentic_context.retrieval import AsyncRetrievalClient

    def safe(headers: dict) -> dict:
        return {k: v for k, v in headers.items() if k.lower() != "authorization"}

    original_post = AsyncIngestionClient._post

    async def logged_post(self, path, payload, *, owner_id, agent_id):
        n, t0 = _log_request(
            "POST", f"{self.base_url}{path}", path,
            {"agent_id": agent_id, "owner_id": owner_id}, body=payload,
        )
        try:
            resp = await original_post(self, path, payload, owner_id=owner_id, agent_id=agent_id)
        except Exception as e:
            _log_response(n, t0, error=e)
            raise
        _log_response(n, t0, resp)
        return resp

    AsyncIngestionClient._post = logged_post

    original_get = AsyncRetrievalClient._get

    async def logged_get(self, path, *, params, headers):
        n, t0 = _log_request(
            "GET", f"{self.base_url}{path}", path, safe(headers), params=params,
        )
        try:
            resp = await original_get(self, path, params=params, headers=headers)
        except Exception as e:
            _log_response(n, t0, error=e)
            raise
        _log_response(n, t0, resp)
        return resp

    AsyncRetrievalClient._get = logged_get

    original_send = AsyncRetrievalClient._send

    async def logged_send(self, method, path, *, headers, params=None, json=None, timeout=None):
        n, t0 = _log_request(
            method, f"{self.base_url}{path}", path, safe(headers),
            params=params, body=json,
        )
        try:
            resp = await original_send(
                self, method, path, headers=headers, params=params, json=json, timeout=timeout
            )
        except Exception as e:
            _log_response(n, t0, error=e)
            raise
        _log_response(n, t0, resp)
        return resp

    AsyncRetrievalClient._send = logged_send


# ─── Tool-call frames (adapter on_tool_start / on_tool_end / on_error) ─


def _tool_label(name: str, args) -> str:
    pack = args.get("context_pack_id") if isinstance(args, dict) else None
    return f"{name}({pack})" if pack else name


def _tool_outcome(output) -> tuple[bool, str]:
    """(succeeded, one-line summary) from a tool's output / ToolMessage."""
    if getattr(output, "status", None) == "error":
        return False, str(getattr(output, "content", output))[:160]
    data = getattr(output, "content", output)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return True, data.replace("\n", " ")[:160]
    if isinstance(data, dict) and "status" in data:
        status = data["status"]
        if status == "ok":
            ctx = data.get("context")
            srcs = [e.get("sourceType", "?") for e in ctx.get("userContext", [])] \
                if isinstance(ctx, dict) else []
            return True, f"ok · {len(srcs)} source(s): {', '.join(srcs) or '-'}"
        detail = data.get("detail") or data.get("message") or ""
        return False, f"{status}" + (f" · {detail}" if detail else "")
    return True, str(data).replace("\n", " ")[:160]


def _on_tool_start(ev) -> None:
    n = next(_tool_seq)
    label = _tool_label(ev.tool_name, ev.args)
    _emit("", _color(_rule(f"╔═ 🔧 TOOL CALL #{n} · {label} ", "═", extra_cells=1), "tool"))
    _open_tools[ev.tool_call_id or f"tool-{n}"] = (n, ev.tool_name, label, time.monotonic())
    _emit(_color("args:", "tool"), _pretty(ev.args))


def _close_tool(key: str, ok: bool, summary: str) -> None:
    _, _, label, t0 = _open_tools.pop(key)
    mark = "✔ OK" if ok else "✖ FAILED"
    secs = time.monotonic() - t0
    _emit(_color(_rule(f"╚═ {mark} · {secs:.1f} s · {summary} ", "═"), "tool" if ok else "err"))
    _turn_tools.append(f"{label} {'✔' if ok else '✖'}" + ("" if ok else f" {summary}"))


def _on_tool_end(ev) -> None:
    key = ev.tool_call_id
    if key not in _open_tools:
        return
    ok, summary = _tool_outcome(ev.output)
    _close_tool(key, ok, summary)


def _on_adapter_error(ev) -> None:
    # ErrorEvent carries no tool_call_id (and phase="tool" also covers failed
    # tool-record ingests), so report it in place; frames a raising tool left
    # open are closed at the end of the turn.
    where = "tool" if ev.phase == "tool" else "turn"
    _emit(_color(f"✖ {where} error: {type(ev.exception).__name__}: {ev.exception}", "err"))


def _close_leftover_tools() -> None:
    for key in list(_open_tools):
        _close_tool(key, False, "no result (tool raised)")


class _EmitLogHandler(logging.Handler):
    """SDK warnings as one yellow line (no traceback) via _emit, so they sit
    inside an open tool frame instead of breaking it. --debug keeps the full
    stderr logging instead."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            exc = record.exc_info[1] if record.exc_info else None
            if exc is not None:
                msg += f" ({type(exc).__name__}: {exc})"
            _emit(_color(f"⚠ {msg}", "warn"))
        except Exception:  # logging must never raise
            self.handleError(record)


def _stub_add_entries() -> None:
    """Short-circuit AsyncIngestionClient.add_entries to a local no-op.

    Opt-in escape hatch for tenants whose Beacon rejects /entries POSTs
    server-side. Stubs that one endpoint so the demo can still exercise:
      - ContextAuth JWT→CDP exchange
      - create_conversation / create_session / add_participants
      - LLM chat
      - GET /context recall
    Lifecycle and recall are unaffected; nothing lands in Beacon storage per
    turn, so cross-run recall won't surface new facts, but the plumbing runs.
    """
    from agentic_context.client_async import AsyncIngestionClient

    async def stubbed_add_entries(self, conversation_id, session_id, entries, owner_id, agent_id):
        if VERBOSE:
            print(f"  [ingest stubbed] would POST /entries conv={conversation_id} entries={len(entries)}")
        return {"stubbed": True, "entries": len(entries)}

    AsyncIngestionClient.add_entries = stubbed_add_entries


async def run_interactive(
    owner_id: str | None,
    session_id: str,
    agent_id: str,
    model_id: str,
    debug: bool,
    skip_ingest: bool,
    auth_mode: str = "emp",
) -> None:
    _install_pretty_wire_hooks()
    if debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)s %(levelname)s %(message)s",
        )
        logging.getLogger("httpx").setLevel(logging.DEBUG)
    else:
        logging.getLogger().addHandler(_EmitLogHandler(logging.WARNING))
    if skip_ingest:
        _stub_add_entries()
        if VERBOSE:
            print("[ingest disabled] /entries will be stubbed locally")
    from agentic_context_langgraph import LangGraphAdapter

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        sys.exit(
            "langchain-anthropic not installed; run `pip install langchain-anthropic` (or ./examples/setup.sh)"
        )

    if auth_mode == "ent":
        if owner_id is None:
            sys.exit("--auth ent needs --owner-id or $BEACON_OWNER_ID "
                     "(ContextAuth has no signed-in user to default to)")
        auth = build_auth_service_account(agent_id, owner_id)
        print(f"[auth] ContextAuth service account (owner_id={owner_id})")
    else:
        auth = build_auth(agent_id)
        if owner_id is None:
            owner_id = auth.user_id
        print(f"[auth] signed in as {auth.user_id} (owner_id={owner_id})")

    llm_kwargs = {}
    if os.environ.get("ANTHROPIC_BASE_URL"):
        llm_kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    llm = ChatAnthropic(
        model=model_id,
        api_key=anthropic_api_key(),
        default_headers=_parse_custom_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS")),
        max_tokens=1024,
        **llm_kwargs,
    )

    system_prompt = (
        "You are a friendly customer-support agent. On the first turn, greet "
        "the user by name if the injected context block names them; otherwise "
        "greet them as a new customer. Call `recall_customer_context` when you "
        "need deeper memory beyond what's already in context."
    )

    if VERBOSE:
        print(f"\n[session_id={session_id}  owner_id={owner_id}  agent_id={agent_id}]")
        print("Type a message, '/quit' to exit. First turn triggers memory recall.\n")
    else:
        print("\nType a message. '/quit' to exit.\n")

    adapter = LangGraphAdapter(
        # The adapter calls this once with its SDK tools (recall_customer_context
        # + get_context_pack), so they're bound to the model before compiling.
        graph=lambda tools: _build_graph(llm.bind_tools(tools), tools),
        agent_id=agent_id,
        owner_id=owner_id,
        auth=auth,
        # Draw tool-call frames. The adapter ingests tool_call / tool_result
        # records with the rest of the turn whether or not these are set.
        on_tool_start=_on_tool_start,
        on_tool_end=_on_tool_end,
        on_error=_on_adapter_error,
    )
    async with adapter:
        turn = 0
        while True:
            try:
                raw = await _read_user("you> ")
            except EOFError:
                print()
                break
            user = raw.strip()
            if not user:
                continue
            if user in {"/quit", "/exit", "/close"}:
                break

            inputs = (
                {"messages": [("system", system_prompt), ("human", user)]}
                if turn == 0
                else user
            )
            turn += 1
            _erase_typed_line("you> ", raw.rstrip("\n"))
            _turn_divider(turn)
            _box("👤 YOU", user, "user")
            _turn_tools.clear()

            spinner = asyncio.create_task(_spin())
            try:
                async for record in adapter.stream(inputs, session_id=session_id):
                    if record.event_type == "user_message":
                        continue  # already shown in the YOU box
                    if record.event_type == "agent_response":
                        await _stop_spinner(spinner)
                        _box(f"🤖 AGENT · {agent_id}", record.content or "", "agent")
                    elif record.event_type == "agent_reasoning":
                        text = record.content or (record.variable or {}).get("OutputValueText", "")
                        if text:
                            _emit("", _color(f"💭 {text.strip()}", "think"))
                    elif VERBOSE:
                        _emit(_color(_format_record(record), "dim"))
            except Exception as e:  # noqa: BLE001 — demo
                await _stop_spinner(spinner)
                _emit(_color(f"✖ turn failed: {e!r}", "err"))
            finally:
                await _stop_spinner(spinner)
                _close_leftover_tools()
                if _turn_tools:
                    _emit(_color("🔧 tools this turn: " + " · ".join(_turn_tools), "tool"))
                _emit("")

    if VERBOSE:
        print("\nclosed. Re-run with the same --owner-id to be recognised.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive LangGraph + ACE demo.")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a KEY=value env file. Defaults to the shared "
             "examples/ace_agent.env (one dir up), then .env in CWD.",
    )
    parser.add_argument(
        "--auth",
        choices=("emp", "ent"),
        default="emp",
        help="emp = UserContextAuth (browser OAuth login, default); "
             "ent = ContextAuth (service-account JWT, needs SF_PRIVATE_KEY_PATH, "
             "SF_SUBJECT and an owner id).",
    )
    parser.add_argument(
        "--owner-id",
        default=os.environ.get("BEACON_OWNER_ID"),
        help="Override owner id. Defaults to $BEACON_OWNER_ID, or when neither "
             "is set, to the signed-in UserContextAuth user's 005… after login "
             "(--auth emp only; --auth ent requires one).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Reuse an existing session_id to resume. Defaults to a fresh uuid.",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("BEACON_AGENT_ID"),
        help="Registered Beacon agent. Defaults to $BEACON_AGENT_ID.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        help="Anthropic model id.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show setup banners (env load, session info, ingest stub notes) "
             "and every record type (agent_reasoning, tool_call, tool_result). "
             "Default output is just user_message + agent_response.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable httpx DEBUG logging (TLS handshakes, request-level timings) "
             "ON TOP of the always-on pretty JSON wire trace.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        default=False,
        help="Stub out /entries POSTs locally instead of hitting Beacon. "
             "Opt-in escape hatch for when the server pipeline rejects entries "
             "(e.g. reject-duplicate-conversation); off by default so "
             "the demo exercises the real /entries path.",
    )
    args = parser.parse_args()

    # owner_id is now optional at CLI-parse time — UserContextAuth supplies
    # the signed-in user's 005… after login if --owner-id and $BEACON_OWNER_ID
    # are both absent.
    if not args.agent_id:
        sys.exit("missing --agent-id or $BEACON_AGENT_ID")

    session_id = args.session_id or f"ses-{uuid.uuid4().hex[:12]}"
    try:
        asyncio.run(
            run_interactive(
                owner_id=args.owner_id,
                session_id=session_id,
                agent_id=args.agent_id,
                model_id=args.model,
                debug=args.debug,
                skip_ingest=args.skip_ingest,
                auth_mode=args.auth,
            )
        )
    except KeyboardInterrupt:
        # Ctrl-C cancels the main task; `async with adapter` has already
        # drained and closed by the time asyncio.run re-raises here.
        if _COLOR:
            sys.stdout.write(_ANSI["reset"])
        print()


if __name__ == "__main__":
    main()
