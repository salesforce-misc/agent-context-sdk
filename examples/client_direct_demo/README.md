# client_direct_demo — AgentContextClient via direct method calls

One of two AgentContextClient demos (Design.md §8). This path calls client methods
directly from the host. The other path wires the same client through hooks:
[`../client_hooks_demo/`](../client_hooks_demo/).

LangGraph is only the host chosen for this demo — any agent runtime can call
the same AgentContextClient APIs the same way.

Exercises:

- Lifecycle: `start_conversation`, `ingest_customer_message`,
  `ingest_agent_message` from host nodes (`pre_turn` / `post_turn`),
- Mid-turn tools: `tool_specs()` → tools on a `ToolNode` → `call_tool`,
- Beacon HTTP visibility (request/response dumps).

## Architecture at a glance

```
   client_direct_demo.py
         │
         ├── Host (LangGraph StateGraph in this demo)
         │     ├── pre_turn   ──→ AgentContextClient.start_conversation
         │     │                  AgentContextClient.ingest_customer_message
         │     ├── agent      ──→ ChatAnthropic ──→ SF AI gateway ──→ Claude
         │     ├── tools      ──→ ToolNode → AgentContextClient.call_tool
         │     │                  (ace_get_context / ace_get_context_pack)
         │     └── post_turn  ──→ AgentContextClient.ingest_agent_message
         │
         └── AgentContextClient
               ├── POST /api/v1/agentic/context                  ──→ Beacon
               ├── GET  /api/v1/agentic/runtime-context-packs    ──→ Beacon
               └── POST conversations / sessions / entries      ──→ Beacon
```

Tools are ordinary host-bound tools; each tool body calls
`ace.call_tool(name, args)`.

## AgentContextClient mapping

### Lifecycle → AgentContextClient

See `_build_graph()` in `client_direct_demo.py`.

| When | Graph node | AgentContextClient | What it does |
|------|------------|--------------|--------------|
| First user message | `pre_turn` | `start_conversation(...)` | Auth (if needed); `get_initial_context` → `POST /api/v1/agentic/context` + `GET …/runtime-context-packs`; create conversation/session; seed is kept for the system prompt |
| Every user message | `pre_turn` | `ingest_customer_message(text)` | POST user turn to `/entries` |
| After assistant reply | `post_turn` | `ingest_agent_message(text)` | POST agent turn to `/entries` |

```python
opened = ace.start_conversation(
    conversation_id=session_id, title=..., auth=auth, agent_id=agent_id,
)
# keep opened["system_prompt_suffix"] / context for the system prompt
ace.ingest_customer_message(user_text, session_id)
# agent node runs; may call ace_get_context / ace_get_context_pack
ace.ingest_agent_message(agent_reply, session_id)
```

### Mid-turn tools → AgentContextClient

Built in `main_async()`: `ace.tool_specs()` → LangChain `StructuredTool` →
`ToolNode`. Each `_fn` calls `ace.call_tool`.

| Tool | AgentContextClient | Example |
|------|--------------|---------|
| `ace_get_context` | `call_tool(...)` | `ace.call_tool("ace_get_context", {})` |
| `ace_get_context_pack` | `call_tool(...)` | `ace.call_tool("ace_get_context_pack", {"context_pack_id": "<pack-id>", "params": {}})` |

The LLM decides when to call them (`model.bind_tools(tools)`).

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
python examples/client_direct_demo/client_direct_demo.py

python examples/client_direct_demo/client_direct_demo.py --scripted

python examples/client_direct_demo/client_direct_demo.py \
  --owner-id 003AAC00005zZTZYB2
```

Type at the `you>` prompt; `/quit` exits. First run may open a browser for
OAuth; refresh token caches under
`~/.cache/agentic_context_demos/refresh_token.json` (shared with other demos).

Example turn (abbreviated):

```
=== AgentContextClient — direct method calls ===
session=ace-sdk-abc12345 (open on first pre_turn)
you> hi
[node pre_turn] ace_start_conversation session=ace-sdk-abc12345
>>> POST /api/v1/agentic/context
<<< 2xx response: { "context": ... }
>>> GET /api/v1/agentic/runtime-context-packs
[node pre_turn] ace_ingest_customer_message 'hi'
>>> POST /api/v1/agentic/conversations/.../sessions/.../entries
[node post_turn] ace_ingest_agent_message 'Hello! How can I help...'
agent> Hello! How can I help...
```

Console dumps every Beacon `POST`/`GET` body (same idea as
`interactive_agent` wire tracing).

## Flow of a single turn

```
START → pre_turn → agent ⇄ tools → post_turn → END
```

1. `pre_turn` — first turn only: `start_conversation` and keep the seed for
   the system prompt. Every turn: `ingest_customer_message`.
2. `agent` — prepends the system prompt as a `SystemMessage` (not stored in
   checkpointed state), then calls `ChatAnthropic` via the SF gateway; may
   emit tool calls.
3. `tools` — `ToolNode` runs bound tools → `ace.call_tool`. Bad tool calls
   return `{"status": "error"}` to the model instead of aborting the run.
4. `post_turn` — `ingest_agent_message`.

## Options

| Flag / env | Default | Purpose |
|---|---|---|
| `--owner-id` / `$BEACON_OWNER_ID` | signed-in auth user | Customer `owner_id` for AgentContextClient (distinct from auth subject when set). |
| `--owner-type` | `individual` if owner id starts with `003`, else auth default | Beacon owner type. |
| `--scripted` | off | Non-interactive: `hi` + pack follow-up, then exit. |
| `BEACON_AGENT_ID` | required | Registered Beacon agent. |
| `ANTHROPIC_BASE_URL` | `api.anthropic.com` | Optional Anthropic-compatible endpoint (e.g. the SF model gateway). |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model id for the gateway. |

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
| `ModuleNotFoundError: langchain_anthropic` / `langgraph` | Run `./examples/setup.sh` and activate `.venv` (it installs both). |
| Generic greeting / empty pack fields | Seed or pack returned stubs or no rows for this owner. Check wire dumps; try `--owner-id` for a customer with data. |
| OAuth / callback errors | Same ECA setup as `ace_basic_demo` (`http://localhost:53682/callback`, require-secret off). |
| `AnthropicConnectionError` | VPN flap or stale devbar token. Rerun `devbar auth claude`; verify gateway with the curl probe above. |
