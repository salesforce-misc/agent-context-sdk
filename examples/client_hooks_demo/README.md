# client_hooks_demo — AgentContextClient via hooks + MCP tools

One of two AgentContextClient demos (Design.md §8). This path wires the client
through host hooks (lifecycle) and MCP tools (mid-turn). The other path calls
the same client methods directly:
[`../client_direct_demo/`](../client_direct_demo/).

Claude Agent SDK is only the host chosen for this demo — any runtime with
hooks (or equivalent callbacks) and tool registration can use AgentContextClient
the same way.

Exercises:

- Lifecycle: `start_conversation`, `ingest_customer_message`,
  `ingest_agent_message` from hooks (`UserPromptSubmit` / `Stop`),
- Mid-turn tools: `tool_specs()` / `call_tool()` exposed as MCP tools,
- Beacon HTTP visibility (request/response dumps) and a model-wait spinner.

## Architecture at a glance

```
   client_hooks_demo.py
         │
         ├── Host (Claude Agent SDK query() in this demo)
         │     ├── hooks UserPromptSubmit / Stop
         │     │         └── AgentContextClient.start_conversation
         │     │             AgentContextClient.ingest_customer_message
         │     │             AgentContextClient.ingest_agent_message
         │     └── MCP server "context"
         │               └── AgentContextClient.call_tool
         │                   (ace_get_context / ace_get_context_pack)
         │
         └── AgentContextClient
               ├── POST /api/v1/agentic/context                  ──→ Beacon
               ├── GET  /api/v1/agentic/runtime-context-packs    ──→ Beacon
               └── POST conversations / sessions / entries      ──→ Beacon
```

The LLM never calls Beacon directly. Hooks own open + ingest; MCP handlers
own mid-turn reads. Same `AgentContextClient` instance for both.

## AgentContextClient mapping

### Hooks → AgentContextClient

See `_build_ace_hooks()` in `client_hooks_demo.py`.

| When | Hook | AgentContextClient | What it does |
|------|------|--------------|--------------|
| First user message | `UserPromptSubmit` | `start_conversation(...)` | Auth (if needed); `get_initial_context` → `POST /api/v1/agentic/context` + `GET …/runtime-context-packs`; create conversation/session; return seed |
| Every user message | `UserPromptSubmit` | `ingest_customer_message(text)` | POST user turn to `/entries` |
| After assistant reply | `Stop` | `ingest_agent_message(text)` | POST agent turn to `/entries` |

After open, the hook injects `opened["system_prompt_suffix"]` (and formatted
context) as `additionalContext` via `_format_seed_context()`.

```python
opened = ace.start_conversation(
    conversation_id=session_id, title=..., auth=auth, agent_id=agent_id,
)
# inject opened["system_prompt_suffix"] / context into the model
ace.ingest_customer_message(user_text, session_id)
# model runs; may call ace_get_context / ace_get_context_pack via MCP
ace.ingest_agent_message(agent_reply, session_id)
```

### MCP tools → AgentContextClient

See `_mcp_tools_from_ace()`. Schemas from `ace.tool_specs()`; handlers call
`ace.call_tool`. Wired on `ClaudeAgentOptions` in `main_async()` as
`mcp_servers={"context": server}`.

| Tool | AgentContextClient | Example |
|------|--------------|---------|
| `ace_get_context` | `call_tool(...)` | `ace.call_tool("ace_get_context", {})` |
| `ace_get_context_pack` | `call_tool(...)` | `ace.call_tool("ace_get_context_pack", {"context_pack_id": "<pack-id>", "params": {}})` |

Visible to the model as `mcp__context__ace_get_context` and
`mcp__context__ace_get_context_pack`. Pack ids come from the
`[Available context packs]` block seeded at open.

## Prereqs

1. **Python 3.10+ and the shared venv.** From the repo root (once):
   ```bash
   ./examples/setup.sh
   source .venv/bin/activate
   ```
2. **Shared param file** `examples/ace_agent.env` populated — same file
   `ace_basic_demo/` uses. **Skip the `cp` if the file already exists**:
   ```bash
   ls examples/ace_agent.env    # if this exists, skip the cp
   cp examples/ace_agent.env.example examples/ace_agent.env
   # edit: SF_CONSUMER_KEY, SF_LOGIN_URL, BEACON_AGENT_ID
   # (+ ANTHROPIC_BASE_URL when using a model gateway)
   ```
3. **`devbar` on PATH** (or set `ANTHROPIC_API_KEY`):
   ```bash
   which devbar && devbar auth claude | head -c 20
   ```
4. **VPN up** (SF model gateway only) — the gateway host is corp-only. Quick sanity:
   ```bash
   curl -sS -o /dev/null -w "http_code=%{http_code}\n" --max-time 10 \
     "$(grep '^ANTHROPIC_BASE_URL=' examples/ace_agent.env | cut -d= -f2-)"
   # expect: http_code=302 or 401 — both prove reachability
   ```

## Run

From the repo root:

```bash
python examples/client_hooks_demo/client_hooks_demo.py

python examples/client_hooks_demo/client_hooks_demo.py --scripted

python examples/client_hooks_demo/client_hooks_demo.py \
  --owner-id 003AAC00005zZTZYB2
```

Type at the `you>` prompt; `/quit` exits. First run may open a browser for
OAuth; refresh token caches under
`~/.cache/agentic_context_demos/refresh_token.json` (shared with other demos).

Example turn (abbreviated):

```
=== AgentContextClient — hooks + MCP tools ===
session=ace-sdk-9a13c5e3 (open on first UserPromptSubmit)
you> hi
[hook UserPromptSubmit] start_conversation session=ace-sdk-9a13c5e3
>>> POST /api/v1/agentic/context
<<< 2xx response: { "context": ... }
>>> GET /api/v1/agentic/runtime-context-packs
[hook UserPromptSubmit] ingest_customer_message 'hi'
>>> POST /api/v1/agentic/conversations/.../sessions/.../entries
[hook Stop] ingest_agent_message 'Hello! How can I help...'
agent> Hello! How can I help...
```

Console also dumps every Beacon `POST`/`GET` body and shows a spinner while
waiting on the model (quiet stretches are LLM/tool latency, not a hang).

## Flow of a single turn

1. `query(prompt=user_text)` starts a Claude Agent SDK turn.
2. `UserPromptSubmit` — first turn only: `start_conversation` (seed context +
   pack catalog + create conversation/session). Every turn:
   `ingest_customer_message`, reinject seed as `additionalContext`.
3. Model may call `mcp__context__ace_get_context` /
   `mcp__context__ace_get_context_pack` → `ace.call_tool`.
4. `Stop` — `ingest_agent_message`.

## Options

| Flag / env | Default | Purpose |
|---|---|---|
| `--owner-id` / `$BEACON_OWNER_ID` | signed-in auth user | Customer `owner_id` for AgentContextClient (distinct from auth subject when set). |
| `--owner-type` | `individual` if owner id starts with `003`, else auth default | Beacon owner type. |
| `--scripted` | off | Non-interactive: `hi` + pack follow-up, then exit. |
| `BEACON_AGENT_ID` | required | Registered Beacon agent. |
| `ANTHROPIC_BASE_URL` | `api.anthropic.com` | Optional Anthropic-compatible endpoint (e.g. the SF model gateway). |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5-20250929` | Model id for the gateway. |

## Env vars

| Var | Purpose |
|---|---|
| `SF_CONSUMER_KEY` | Connected App consumer key. |
| `SF_LOGIN_URL` | Salesforce login host, no scheme. |
| `BEACON_AGENT_ID` | `agent_id` header + a360 `aid`. |
| `BEACON_OWNER_ID` | Optional customer owner override (same as `--owner-id`). |
| `ANTHROPIC_API_KEY` | Anthropic or gateway key; falls back to `devbar auth claude`. |
| `ANTHROPIC_BASE_URL` | Optional Anthropic-compatible endpoint (e.g. the SF model gateway); defaults to `api.anthropic.com`. |
| `ANTHROPIC_MODEL` | Optional model override. |

## Troubleshooting

| Symptom | Cause |
|---|---|
| `missing env var: SF_CONSUMER_KEY` (or similar) | Fill `examples/ace_agent.env` (or export the var). |
| `` `devbar` not on PATH `` | Install devbar or set `ANTHROPIC_API_KEY`. |
| ImportError for `claude_agent_sdk` | Run `./examples/setup.sh` and activate `.venv` (it installs it). |
| Generic greeting / empty pack fields | Seed or pack returned stubs (`employeeIdentity[0]`, empty enrollments). Check `[seed]` lines and wire dumps; try `--owner-id` for a customer with data. |
| Spinner / quiet stretch after ingest | Model or MCP tool still running — watch for `[llm]` / `[tool]` lines and Beacon dumps. |
| OAuth / callback errors | Same ECA setup as `ace_basic_demo` (`http://localhost:53682/callback`, require-secret off). |
