# interactive_agent — LangGraph + Claude + ACE memory recall

Full-stack interactive chat demo. Exercises:

- the LangGraph adapter (`agentic_context_langgraph`) in `context_mode="auto"`,
- Claude via the Anthropic API (or any Anthropic-compatible endpoint),
- Beacon memory recall across runs (SessionFirstPolicy fetches prior memory
  for `owner_id` and injects it as a SystemMessage before turn 1),
- the `recall_customer_context` tool for deeper, mid-turn lookups.

## Quick run
   ```bash
   cp examples/ace_agent.env.example examples/ace_agent.env

  # Update (or replace all) following in ace_agent.env
   SF_CONSUMER_KEY=<Consumer key from External Client App (ECA)>
   SF_LOGIN_URL=<Salesforce org url>

   BEACON_AGENT_ID=<agent_api_name>
   ANTHROPIC_API_KEY=<your Anthropic API key>
   ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"

   # Source env file
   source examples/ace_agent.env

  # Run script
  ./examples/setup.sh
   source .venv/bin/activate
   python examples/interactive_agent/interactive_agent.py --verbose
   ```

## Architecture at a glance

Three independent clients talking to three independent destinations:

```
   interactive_agent.py (your laptop)
         │
         ├── LangGraph StateGraph
         │     ├── "agent" node  ──→  ChatAnthropic ──→  Anthropic API ──→ Claude
         │     └── "tools" node  ──→  recall_customer_context
         │
         └── LangGraphAdapter
               ├── fetch memory  (GET  /agentic/context)   ──→  Beacon (ACE)
               └── stream records (POST /entries)          ──→  Beacon (ACE)
```

The LLM and the ACE SDK don't talk to each other over the network — they
meet only in your process memory. The adapter fetches memory **before**
the LLM call (injected as SystemMessage), POSTs records **after** the
LLM call. Swap `ChatAnthropic` for `ChatOpenAI` and the ACE wiring stays
intact; swap LangGraph for Claude Agent SDK and the LLM wiring stays
intact.

## Setup Details

1. **Python env + deps** — run the setup script once from the repo root.
   It creates `.venv` (pyenv `3.11.14` if installed, else `python3.11`;
   override with `PYTHON=/path/to/python3.11`), installs the SDK in
   editable mode plus the LangGraph/Anthropic stack, checks the imports,
   and creates `examples/ace_agent.env` from the template if it's
   missing. Safe to re-run.
   ```bash
   ./examples/setup.sh
   source .venv/bin/activate
   ```
   <details><summary>What it runs (manual equivalent)</summary>

   ```bash
   pyenv shell 3.11.14
   python3.11 -m venv .venv            # only if .venv/ doesn't exist yet
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .                                          # the SDK itself
   pip install langgraph langchain-anthropic langchain-core  # LangGraph stack
   pip install anyio httpx                                   # SDK runtime deps the demo touches
   ```
   </details>
2. **Shared param file** `examples/ace_agent.env` populated — same file
   `ace_basic_demo/` uses. The script auto-loads it; variables already
   set in the shell win over the file. `setup.sh` creates it from
   `ace_agent.env.example` if it's missing (never overwrites an existing
   one) — then fill in your values:
   ```bash
   $EDITOR examples/ace_agent.env
   ```
   Extra vars live in this file that `ace_basic_demo/` ignores:

   | Var | What it is |
   |---|---|
   | `ANTHROPIC_API_KEY` | Anthropic API key (**required**) |
   | `ANTHROPIC_BASE_URL` | Optional Anthropic-compatible endpoint; defaults to `api.anthropic.com` |
   | `ANTHROPIC_CUSTOM_HEADERS` | Optional extra headers, one per line |
   | `ANTHROPIC_MODEL` | Optional override; default is `claude-sonnet-4-5-20250929` |


## Run

From the repo root, with `.venv` activated:

```bash
python examples/interactive_agent/interactive_agent.py                      # fresh session, quiet
python examples/interactive_agent/interactive_agent.py --session-id ses-1   # resume
python examples/interactive_agent/interactive_agent.py --verbose            # full plumbing output
python examples/interactive_agent/interactive_agent.py --skip-ingest        # stub broken /entries
python examples/interactive_agent/interactive_agent.py --auth ent           # service-account ContextAuth
```

Type a message at the red `you>` prompt. A braille spinner (`⠋⠙⠹…`) sits
on the bottom line while the turn runs. `/quit`, Ctrl-D, or Ctrl-C exits
cleanly and drains any pending async POSTs.

Example turn with a tool call (abridged; JSON bodies are printed in full):

```
══ turn 4 · 05:45:17 ═══════════════════════════════════════
┏━ 👤 YOU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ← red
┃ Was my last cleaning covered?
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💭 Let me check the status of your dental claims.          ← italic cyan

╔═ 🔧 TOOL CALL #2 · get_context_pack(SA_Claims_Status) ═══   ← magenta
║  args: { "context_pack_id": "SA_Claims_Status", … }
║  ▶ #23 POST https://…/api/v1/agentic/context   · context pack SA_Claims_Status
║    { "context_pack": "SA_Claims_Status", … }
║  ◀ #23 HTTP 404 · 210 ms                                     ← red
║    Context pack not found.
╚═ ✖ FAILED · 0.4 s · not_ready · request failed: IngestionError ═

┏━ 🤖 AGENT · SA_Benefits_Assistant ━━━━━━━━━━━━━━━━━━━━━━━━   ← green
┃ I'm unable to retrieve your claims information right now…
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ #24 POST https://…/conversations/ses-…/entries   · save user_message, agent_reasoning, tool_call, tool_result, agent_response
  headers: {'agent_id': 'SA_Benefits_Assistant', 'owner_id': '005…'}   ← yellow label, cyan dict
  { "entries": [ … ] }
◀ #24 2xx · 180 ms
🔧 tools this turn: get_context_pack(SA_Claims_Status) ✖ not_ready · …
```

- **Turn divider:** `══ turn N · time ══` marks where each question/answer starts.
- **Speakers:** the 👤 YOU box is red, the 🤖 AGENT box green.
- **Wire calls:** `▶ #n` (blue) is a request, `◀ #n` is its response — green for
  2xx, red for errors, with latency. The number pairs a late background
  response with its request; the `· label` says what the call is for.
- **Tool frames:** `╔═ 🔧 TOOL CALL` … `╚═ ✔ OK / ✖ FAILED` in magenta. Beacon
  calls made while the tool runs (such as recall reads) are shown inside the
  `║` bar. The `tool_call` / `tool_result` records are saved with the rest of
  the turn in one POST when the turn ends. The footer lists the turn's tools.
- **SDK warnings:** one yellow `⚠` line (full tracebacks with `--debug`).

### Output levels

| Flag | What you see |
|---|---|
| *(default)* | Everything above: turn dividers, boxed YOU / AGENT messages, 💭 reasoning, tool frames, and every Beacon request/response in full. |
| `--verbose` | Everything above + env-load banner, session_id banner, `[ingest stubbed]` notes, and any other record type the adapter yields (dim). |
| `--debug` | Full HTTP request/response for every Beacon POST, **plus** one-line httpx logs for every outbound HTTP call in the process — LLM, `ContextAuth` leg-1/leg-2, Beacon GETs (recall), Beacon POSTs. Independent of `--verbose`. |

## Flow of a single run

1. The shared param file is loaded (silent — pass `--verbose` to see the
   `[env] loaded …` banner).
2. Auth runs the CDP token exchange and the tenant-specific `base_url` comes
   back. With `--auth emp` (default), that is `UserContextAuth`: browser OAuth the
   first time, then the cached refresh token. With `--auth ent`, it is `ContextAuth`
   (service-account JWT → CDP exchange).
3. Adapter lifecycle fires: `create_conversation` →
   `create_session` → `add_participants` (409 on resume — treated as success).
4. First turn triggers `SessionFirstPolicy`: `GET /api/v1/agentic/context`
   for the owner. Empty on a first-ever run; populated on subsequent runs
   with the same `--owner-id`.
5. The LLM responds via Claude. If memory was present,
   it greets you by name.
6. Each turn, one `user_message` + one `agent_response` (plus any tool
   trace records) are produced by the adapter and POSTed to
   `/entries` — unless you pass `--skip-ingest` to stub that call
   locally (see below).
7. `/quit` → the adapter drains pending async POSTs and closes locally
   (no `close_session` POST).

## Skipping ingest

If your tenant's Beacon rejects `POST /entries` server-side (for example
with `reject-duplicate-conversation`), pass `--skip-ingest` to stub
`/entries` locally (opt-in escape hatch):

```bash
python examples/interactive_agent/interactive_agent.py --skip-ingest
```

Lifecycle, auth, LLM, and recall still run; only the per-turn ingest POST
is suppressed. Without the flag, the "recognise returning user" loop works
end to end: first run fills memory, a second run with the same `--owner-id`
sees the injected SystemMessage pre-turn and the LLM greets you by name +
remembered facts.

## Options

| Flag | Default | Purpose |
|---|---|---|
| `--auth` | `emp` | `emp` = `UserContextAuth` (browser OAuth login; owner defaults to the signed-in user). `ent` = `ContextAuth` (service-account JWT; needs `SF_PRIVATE_KEY_PATH`, `SF_SUBJECT`, and `--owner-id` / `$BEACON_OWNER_ID`). |
| `--owner-id` | `$BEACON_OWNER_ID`, else signed-in user (`emp` only) | Beacon owner id (user identity). Required with `--auth ent`. |
| `--session-id` | fresh uuid | Reuse to resume a prior chat. |
| `--agent-id` | `$BEACON_AGENT_ID` | Registered Beacon agent id. |
| `--model` | `$ANTHROPIC_MODEL` or `claude-sonnet-4-5-20250929` | Anthropic model id. |
| `--verbose` | off | Show setup banners + every record type (see "Output levels"). |
| `--debug` | off | Full Beacon POST payloads + one-line httpx logs for every outbound HTTP call (LLM, auth exchange, Beacon GET recall, Beacon POST). |
| `--skip-ingest` | off | Stub `/entries` POSTs locally (opt-in; use when server pipeline rejects entries). |
| `--env-file` | `../ace_agent.env` | Override the shared param file. |

## Troubleshooting

| Symptom | Cause |
|---|---|
| `missing env var: ANTHROPIC_API_KEY` | Add it to `examples/ace_agent.env`. |
| `ModuleNotFoundError: langchain_core` or `langgraph` | re-run `./examples/setup.sh`, then `source .venv/bin/activate`. |
| `AnthropicConnectionError('Connection error.')` | Network issue or bad `ANTHROPIC_BASE_URL`. Rerun the demo; check the URL and your API key. |
| Every turn logs a 400 on `/entries` with `reject-duplicate-conversation` | Server-side rejection — chat still works, memory just isn't persisted. Pass `--skip-ingest` to suppress the POST. |
| Spinner stays on forever / never clears | LLM call hung (endpoint timeout) or the stream raised before any record. `/quit` + rerun; check with `--debug`. |
