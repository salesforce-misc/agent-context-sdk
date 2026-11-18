#!/usr/bin/env python3
"""Quickstart demo — Claude Agent SDK variant, follows docs/QUICKSTART.md.

Three steps:
  1. pip install agent-context-sdk  (done by examples/setup.sh)
  2. UserContextAuth.login()      (browser-interactive OAuth + PKCE)
  3. ClaudeAdapter(...).apply_to(ClaudeAgentOptions(...)) → claude_agent_sdk.query(...)

Usage:
  python examples/quickstart_demo/quickstart_demo.py "hello"
"""
import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path


# ── env loader (same convention as other demos: examples/ace_agent.env) ──
_ENV_FILE = Path(__file__).parent.parent / "ace_agent.env"
if _ENV_FILE.exists():
    for raw in _ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var: {name} (set in examples/ace_agent.env)")
    return v


async def main(user_input: str) -> None:
    # Imports kept local so missing-adapter errors surface clearly.
    from agentic_context import ClaudeAdapter
    from claude_agent_sdk import ClaudeAgentOptions, query

    # Shared auth helper (see examples/_demo_auth.py). To swap to the
    # service-account flow, change this import to ``build_auth_service_account``.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _demo_auth import anthropic_api_key, build_auth

    agent_id = _require("BEACON_AGENT_ID")

    # ─── QUICKSTART Step 2: credentials ───────────────────────────────
    auth = build_auth(agent_id)
    # Same rule as interactive_agent: $BEACON_OWNER_ID wins, else the
    # signed-in user's 005…, so both demos share one owner's memory.
    owner_id = os.environ.get("BEACON_OWNER_ID") or auth.user_id
    print(f"[auth] signed in as {auth.user_id} (owner_id={owner_id})")

    # ─── QUICKSTART Step 3: build the adapter and wire it into options ─
    adapter = ClaudeAdapter(
        agent_id=agent_id,
        owner_id=owner_id,
        auth=auth,
        context_mode="auto",
    )

    session_id = f"quickstart-{uuid.uuid4().hex[:8]}"
    print(f"[session] {session_id}")

    # The Claude CLI subprocess reads ANTHROPIC_API_KEY (and, for a custom
    # endpoint, ANTHROPIC_BASE_URL) from the environment.
    api_key = anthropic_api_key()
    model_id = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    # Machine-managed settings (/Library/Application Support/ClaudeCode/
    # managed-settings.json) can force Bedrock mode, where the CLI ignores
    # ANTHROPIC_API_KEY and silently retries 401s for minutes. Hand it the same
    # key as a Bedrock bearer token; outside Bedrock mode this is unused.
    cli_env = {"AWS_BEARER_TOKEN_BEDROCK": os.environ.get("AWS_BEARER_TOKEN_BEDROCK", api_key)}
    # Some custom endpoints (e.g. model gateways) only accept a bearer token
    # (ANTHROPIC_AUTH_TOKEN), so hand the CLI the key that way too.
    if os.environ.get("ANTHROPIC_BASE_URL"):
        cli_env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("ANTHROPIC_AUTH_TOKEN", api_key)

    scratch_dir = tempfile.mkdtemp(prefix="quickstart-")
    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=(
            "You are a friendly assistant. On the first turn, greet the "
            "user by name if the injected context block names them. Stay in "
            "that role: don't discuss the machine, directory, files or tools "
            "you run with."
        ),
        # Don't load ~/.claude / .claude settings into the CLI subprocess:
        # a developer's own plugins, MCP servers, hooks and provider flags
        # (e.g. CLAUDE_CODE_USE_BEDROCK) otherwise apply, and slow or hang
        # the turn. The demo is configured purely by env vars.
        setting_sources=[],
        env=cli_env,
        # The CLI is Claude Code: by default it hands the model file/shell
        # tools and describes the machine it runs on (cwd, user). A customer
        # assistant needs neither — disable the built-ins (the adapter's MCP
        # context tool stays) and run from an empty scratch dir.
        tools=[],
        cwd=scratch_dir,
        stderr=lambda line: print(f"[claude] {line}", file=sys.stderr),
        # QUICKSTART_DEBUG=1 surfaces the CLI's own debug log (API retries,
        # auth/TLS errors) — the turn is otherwise silent until it finishes.
        extra_args={"debug-to-stderr": None} if os.environ.get("QUICKSTART_DEBUG") else {},
    )
    options = adapter.apply_to(
        options, session_id=session_id, owner_id=owner_id
    )

    # ─── Drive one turn and print assistant messages ──────────────────
    print(f"\n[you] {user_input}")
    try:
        async for message in query(prompt=user_input, options=options):
            name = type(message).__name__
            if name == "AssistantMessage":
                # content is a list of TextBlock / ToolUseBlock
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        print(f"[agent] {text}")
            elif name == "ResultMessage":
                print(f"[done] turns={getattr(message, 'num_turns', '?')}")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "message",
        nargs="?",
        default="Hi — what do you remember about me?",
        help="user message for the single turn",
    )
    args = ap.parse_args()
    asyncio.run(main(args.message))
