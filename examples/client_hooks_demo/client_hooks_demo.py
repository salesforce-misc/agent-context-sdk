#!/usr/bin/env python3
"""AgentContextClient via hooks + MCP tools (Design.md §8).

One of two AgentContextClient demos: lifecycle in hooks, mid-turn via MCP
call_tool. This demo uses Claude Agent SDK; see client_direct_demo for
the direct method-call path.

Env: examples/ace_agent.env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from agentic_context import AgentContextClient, CONTEXT_PACK_INSTRUCTION

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _demo_auth import anthropic_api_key, build_auth  # noqa: E402


# ── Env / gateway helpers ────────────────────────────────────────────────


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
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


_load_env_file(Path(__file__).resolve().parent.parent / "ace_agent.env")


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"missing env var: {name}")
    return val


# ── Console extras: HTTP dumps, spinner, seed banners ────────────────────


def _pretty(obj: object) -> str:
    try:
        return json.dumps(obj, indent=2, default=str, sort_keys=False)
    except (TypeError, ValueError):
        return repr(obj)


def _install_pretty_wire_hooks() -> None:
    """Print every Beacon POST/GET (like interactive_agent)."""
    from agentic_context.client import IngestionClient
    from agentic_context.retrieval import RetrievalClient

    original_post = IngestionClient._post

    def logged_post(self, path, payload, *, owner_id, agent_id):
        print(f"\n>>> POST {self.base_url}{path}")
        print(f">>> headers: agent_id={agent_id}  owner_id={owner_id}")
        if payload is not None:
            print(">>> body:")
            print(_pretty(payload))
        try:
            resp = original_post(
                self, path, payload, owner_id=owner_id, agent_id=agent_id
            )
            print("<<< 2xx response:")
            print(_pretty(resp))
            return resp
        except Exception as e:
            print(f"<<< error: {e!r}")
            raise

    IngestionClient._post = logged_post

    original_get = RetrievalClient._get

    def logged_get(self, path, *, params, headers):
        print(f"\n>>> GET {self.base_url}{path}")
        safe_headers = {
            k: v for k, v in headers.items() if k.lower() != "authorization"
        }
        print(f">>> headers: {safe_headers}")
        print(f">>> params: {params}")
        try:
            resp = original_get(self, path, params=params, headers=headers)
            print("<<< 2xx response:")
            print(_pretty(resp))
            return resp
        except Exception as e:
            print(f"<<< error: {e!r}")
            raise

    RetrievalClient._get = logged_get

    original_send = RetrievalClient._send

    def logged_send(
        self, method, path, *, headers, params=None, json=None, timeout=None
    ):
        print(f"\n>>> {method} {self.base_url}{path}")
        safe_headers = {
            k: v for k, v in headers.items() if k.lower() != "authorization"
        }
        print(f">>> headers: {safe_headers}")
        if params is not None:
            print(f">>> params: {params}")
        if json is not None:
            print(">>> body:")
            print(_pretty(json))
        try:
            resp = original_send(
                self,
                method,
                path,
                headers=headers,
                params=params,
                json=json,
                timeout=timeout,
            )
            print("<<< 2xx response:")
            print(_pretty(resp))
            return resp
        except Exception as e:
            print(f"<<< error: {e!r}")
            raise

    RetrievalClient._send = logged_send


# ── Inject start_conversation seed into UserPromptSubmit ─────────────────


def _format_context_for_injection(context_payload: dict) -> str:
    """Format userContext entries (same shape as ClaudeAdapter)."""
    user_context = context_payload.get("userContext") or []
    if not user_context:
        return ""
    lines = ["[Context from prior conversations]"]
    for entry in user_context:
        source_type = entry.get("sourceType", "Unknown")
        content = entry.get("entry", "")
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        lines.append(f"- [{source_type}] {content}")
    return "\n".join(lines)


def _format_seed_context(opened: dict) -> str:
    """context + system_prompt_suffix for additionalContext."""
    parts: list[str] = []
    formatted = _format_context_for_injection(opened.get("context") or {})
    if formatted:
        parts.append(formatted)
    suffix = opened.get("system_prompt_suffix") or opened.get("pack_catalog_text")
    if suffix:
        parts.append(str(suffix))
    return "\n\n".join(parts)


def _looks_empty_ref(entry: str) -> bool:
    """True for stubs like ``employeeIdentity[0]``."""
    s = (entry or "").strip()
    return bool(s) and s.endswith("[0]") and "\n" not in s and len(s) < 80


def _print_seed_summary(opened: dict) -> None:
    """Console summary of what was seeded."""
    ctx = opened.get("context") or {}
    entries = ctx.get("userContext") or []
    packs = [p for p in (opened.get("pack_ids") or []) if p]
    print(
        f"[seed] injected {len(entries)} userContext entr"
        f"{'y' if len(entries) == 1 else 'ies'}; "
        f"pack catalog ids={packs or '[]'}"
    )
    useful = 0
    for item in entries:
        text = str(item.get("entry") or "")
        src = item.get("sourceType") or "?"
        if _looks_empty_ref(text):
            print(
                f"[seed]   [{src}] {text!r} — empty/ref stub "
                "(no personalizable fields for the model)"
            )
            continue
        useful += 1
        preview = text.replace("\n", " ").strip()
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"[seed]   [{src}] {preview}")
    if not useful:
        print(
            "[seed] no usable identity/memory facts — first reply will likely "
            "be a generic greeting (Beacon returned stubs or unrelated profile)"
        )


def _summarize_tool_payload(out: dict) -> str:
    """One-line tool result (full body is in the wire dump)."""
    ctx = (out or {}).get("context") or out or {}
    entries = ctx.get("userContext") if isinstance(ctx, dict) else None
    if not isinstance(entries, list):
        return f"keys={list((out or {}).keys())}"
    bits: list[str] = []
    for item in entries:
        src = item.get("sourceType") or "?"
        text = str(item.get("entry") or "")
        if _looks_empty_ref(text):
            bits.append(f"{src}=empty")
        else:
            bits.append(f"{src}=ok")
    return f"userContext[{len(entries)}]: " + (", ".join(bits) if bits else "none")


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


async def _spin(label: str = "waiting on model") -> None:
    """Spinner while waiting on the model."""
    if not sys.stdout.isatty():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise
        return
    i = 0
    try:
        while True:
            sys.stdout.write(
                f"\r  {_SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]} {label}..."
            )
            sys.stdout.flush()
            i += 1
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        sys.stdout.write("\r" + " " * (len(label) + 20) + "\r")
        sys.stdout.flush()
        raise


async def _stop_spinner(task: asyncio.Task) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── Host on Claude hooks ─────────────────────────────────────────────────


def _build_ace_hooks(
    *,
    ace: AgentContextClient,
    session_id: str,
    auth,
    agent_id: str,
    title: str,
    owner_id: str | None = None,
    owner_type: str | None = None,
):
    """UserPromptSubmit / Stop → start_conversation + ingest_customer_message / ingest_agent_message."""
    from claude_agent_sdk import HookMatcher

    state = {"opened": False, "pack_ids": [], "last_opened": None}

    async def on_user_prompt_submit(input_data, tool_use_id, context):
        parts: list[str] = []
        if not state["opened"]:
            print(f"[hook UserPromptSubmit] start_conversation session={session_id}")
            open_kwargs: dict = {
                "conversation_id": session_id,
                "title": title,
                "auth": auth,
                "agent_id": agent_id,
            }
            if owner_id:
                open_kwargs["owner_id"] = owner_id
            if owner_type:
                open_kwargs["owner_type"] = owner_type
            opened = await asyncio.to_thread(ace.start_conversation, **open_kwargs)
            state["opened"] = True
            state["pack_ids"] = [p for p in (opened.get("pack_ids") or []) if p]
            state["last_opened"] = opened
            print(
                f"[hook UserPromptSubmit] opened pack_ids={state['pack_ids']} "
                f"context_keys={list((opened.get('context') or {}).keys())}"
            )
            _print_seed_summary(opened)

        # Fresh query() each turn — reinject seed + pack catalog.
        seed = _format_seed_context(state["last_opened"] or {})
        if seed:
            parts.append(seed)
            print(
                f"[hook UserPromptSubmit] inject additionalContext "
                f"({len(seed)} chars)"
            )

        prompt = (input_data.get("prompt") or "").strip()
        if prompt:
            print(f"[hook UserPromptSubmit] ingest_customer_message {prompt!r}")
            await asyncio.to_thread(ace.ingest_customer_message, prompt, session_id)

        if parts:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n\n".join(parts),
                }
            }
        return {}

    async def on_stop(input_data, tool_use_id, context):
        msg = (input_data.get("last_assistant_message") or "").strip()
        if msg and state["opened"]:
            print(f"[hook Stop] ingest_agent_message {msg[:120]!r}...")
            await asyncio.to_thread(ace.ingest_agent_message, msg, session_id)
        return {}

    return {
        "UserPromptSubmit": [
            HookMatcher(matcher=".*", hooks=[on_user_prompt_submit]),
        ],
        "Stop": [
            HookMatcher(matcher=".*", hooks=[on_stop]),
        ],
    }, state


# ── Model tools on MCP ───────────────────────────────────────────────────


def _mcp_tools_from_ace(ace: AgentContextClient):
    """ace_get_context / ace_get_context_pack only; server name ``context``."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    pack_spec = next(s for s in ace.tool_specs() if s["name"] == "ace_get_context_pack")
    ctx_spec = next(s for s in ace.tool_specs() if s["name"] == "ace_get_context")

    # Tools from one model reply may run concurrently; the client isn't safe
    # for concurrent calls, so serialize them.
    tool_lock = asyncio.Lock()

    async def _call_tool(name: str, payload: dict) -> dict:
        # Blocking HTTP off the event loop; bad tool calls go back to the model.
        try:
            async with tool_lock:
                return await asyncio.to_thread(ace.call_tool, name, payload)
        except (ValueError, RuntimeError) as e:
            return {"status": "error", "message": str(e)}

    @tool(
        "ace_get_context",
        ctx_spec["description"],
        {"conversation_id": str},
    )
    async def ace_get_context(args: dict) -> dict:
        payload: dict = {}
        cid = (args.get("conversation_id") or "").strip()
        if cid:
            payload["conversation_id"] = cid
        print(f"[tool] ace_get_context args={payload}")
        out = await _call_tool("ace_get_context", payload)
        print(f"[tool] ace_get_context result: {_summarize_tool_payload(out)}")
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    @tool(
        "ace_get_context_pack",
        pack_spec["description"],
        {"context_pack_id": str, "params": str},
    )
    async def ace_get_context_pack(args: dict) -> dict:
        raw_params = args.get("params") or ""
        params: object = {}
        if isinstance(raw_params, dict):
            params = raw_params
        elif isinstance(raw_params, str) and raw_params.strip():
            try:
                params = json.loads(raw_params)
            except json.JSONDecodeError:
                params = {}
        payload = {
            "context_pack_id": args.get("context_pack_id") or "",
            "params": params,
        }
        print(f"[tool] ace_get_context_pack args={payload}")
        out = await _call_tool("ace_get_context_pack", payload)
        print(f"[tool] ace_get_context_pack result: {_summarize_tool_payload(out)}")
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    server = create_sdk_mcp_server(
        name="context",
        version="1.0.0",
        tools=[ace_get_context, ace_get_context_pack],
    )
    allowed = [
        "mcp__context__ace_get_context",
        "mcp__context__ace_get_context_pack",
    ]
    return server, allowed


# ── REPL / scripted loop ─────────────────────────────────────────────────


def _scripted_turns(pack_ids: list[str]) -> list[str]:
    turns = ["hi"]
    if pack_ids:
        turns.append(
            f"Please use context pack {pack_ids[0]} if helpful, "
            "otherwise say what you know from context."
        )
    else:
        turns.append("What do you already know about me from context?")
    return turns


async def _run_turn(*, options, user_text: str) -> str:
    """One Claude Agent SDK ``query()`` turn."""
    from claude_agent_sdk import query

    answer = ""
    tool_names: list[str] = []
    print(
        "[llm] calling model (hooks already ran; quiet = model/tool latency)",
        flush=True,
    )
    spinner = asyncio.create_task(_spin("waiting on model"))
    try:
        async for message in query(prompt=user_text, options=options):
            await _stop_spinner(spinner)
            name = type(message).__name__
            if name == "AssistantMessage":
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        answer = text
                    tname = getattr(block, "name", None)
                    if tname:
                        tool_names.append(str(tname))
                        print(f"[llm] tool use: {tname}", flush=True)
                        print(
                            "[llm] tool HTTP + model continue next "
                            "(may take a while)...",
                            flush=True,
                        )
            elif name == "ResultMessage":
                result = getattr(message, "result", None)
                if isinstance(result, str) and result.strip():
                    answer = result.strip()
    finally:
        await _stop_spinner(spinner)

    if tool_names:
        print(f"[llm tools called] {tool_names}")
    return answer or "(no assistant text)"


async def main_async(
    *,
    scripted: bool,
    owner_id: str | None = None,
    owner_type: str | None = None,
) -> None:
    from claude_agent_sdk import ClaudeAgentOptions

    _install_pretty_wire_hooks()

    agent_id = _require("BEACON_AGENT_ID")
    auth = build_auth(agent_id)
    ace = AgentContextClient()
    session_id = f"client-hooks-{uuid.uuid4().hex[:8]}"

    if owner_id and not owner_type:
        owner_type = "individual" if owner_id.startswith("003") else None

    hooks, hook_state = _build_ace_hooks(
        ace=ace,
        session_id=session_id,
        auth=auth,
        agent_id=agent_id,
        title="client-hooks-demo",
        owner_id=owner_id,
        owner_type=owner_type,
    )
    server, allowed = _mcp_tools_from_ace(ace)

    token = anthropic_api_key()
    gateway_env = {"ANTHROPIC_API_KEY": token}
    # Some custom endpoints (e.g. model gateways) only accept a bearer token,
    # so send it too; unset means api.anthropic.com, which takes
    # ANTHROPIC_API_KEY as is.
    if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
        gateway_env["ANTHROPIC_BASE_URL"] = base_url
        gateway_env["ANTHROPIC_AUTH_TOKEN"] = token
    model_id = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=(
            "You are a helpful customer assistant with Salesforce Data 360 Context.\n"
            "Stay in that role. Do not talk about coding agents, MCP auth, or repos.\n"
            "On the first turn, greet the user by name if the injected context "
            "block names them; otherwise greet them as a new customer.\n"
            f"conversation_id={session_id}\n"
            "Use ace_get_context for on-demand customer memory when needed.\n"
            "Use ace_get_context_pack for structured pack data (ids from the "
            "available-packs block injected at open).\n"
            f"\n{CONTEXT_PACK_INSTRUCTION}\n"
        ),
        mcp_servers={"context": server},
        allowed_tools=allowed,
        hooks=hooks,
        env=gateway_env,
    )

    print("\n=== AgentContextClient — hooks + MCP tools ===")
    print(f"session={session_id} (open on first UserPromptSubmit)")
    if owner_id:
        print(
            f"owner_id={owner_id} owner_type={owner_type or '(auth default)'} "
            f"(auth user={getattr(auth, 'user_id', None)!r})"
        )
    else:
        print(
            f"owner_id=(auth user {getattr(auth, 'user_id', None)!r}) "
            "— pass --owner-id for a customer id"
        )
    print(
        "Wire dumps show Beacon HTTP. Between dumps, a spinner means the "
        "model (or a mid-turn tool) is still working — not a hang."
    )
    print("Type a message, or /quit to exit.\n")

    turns = _scripted_turns([]) if scripted else None
    turn_i = 0
    while True:
        if turns is not None:
            if turn_i >= len(turns):
                break
            if turn_i == 1 and hook_state["pack_ids"]:
                turns = _scripted_turns(hook_state["pack_ids"])
            text = turns[turn_i]
            turn_i += 1
            print(f"you> {text}")
        else:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text or text in ("/quit", "/exit", "quit"):
                break

        reply = await _run_turn(options=options, user_text=text)
        print(f"agent> {reply}\n")

    if scripted and hook_state["opened"]:
        print("[verify] ace_get_context")
        ctx = ace.get_context()
        print(
            f"  status={ctx.get('status')} "
            f"has_userContext={bool((ctx.get('context') or {}).get('userContext'))}"
        )
        packs = hook_state["pack_ids"]
        if packs:
            print(f"[verify] ace_get_context_pack {packs[0]!r}")
            pack = ace.get_context_pack(context_pack_id=packs[0], params={})
            print(
                f"  status={pack.get('status')} "
                f"keys={list((pack.get('context') or {}).keys())}"
            )

    print(f"Done. session={session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Non-interactive: hi + pack follow-up then exit (CI / recording)",
    )
    parser.add_argument(
        "--owner-id",
        default=os.environ.get("BEACON_OWNER_ID"),
        help="Customer owner_id for AgentContextClient (defaults to $BEACON_OWNER_ID, "
        "else the signed-in UserContextAuth user).",
    )
    parser.add_argument(
        "--owner-type",
        default=None,
        help="Owner type (e.g. individual, employee). "
        "Defaults to individual when --owner-id starts with 003.",
    )
    args = parser.parse_args()
    asyncio.run(
        main_async(
            scripted=args.scripted,
            owner_id=args.owner_id,
            owner_type=args.owner_type,
        )
    )


if __name__ == "__main__":
    main()
