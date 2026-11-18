# examples/

Six demos live here, each in its own folder with its own README. They're
layered — the first two isolate failures, the next two exercise real
agents, and the last two show two ways to use AgentContextClient (direct method
calls vs hooks). Walk them top-to-bottom when onboarding or debugging.

- [`non_sdk_basic_demo/`](non_sdk_basic_demo/README.md) — same two HTTP legs
  with pure `httpx`, no `agentic_context` import. If this works and
  `ace_basic_demo` doesn't, blame the SDK; if both fail, blame org/ECA/tenant.
- [`ace_basic_demo/`](ace_basic_demo/README.md) — SDK-backed lifecycle
  shakedown (`create`/`ingest`/`read`/`list`) with no LLM or gateway. Isolates
  "is it auth/transport?" vs "is it Claude/adapter?" when quickstart fails.
- [`quickstart_demo/`](quickstart_demo/README.md) — one-turn Claude Agent SDK
  integration; proves auth + Beacon + Claude gateway + adapter + context pack
  all work end-to-end. The right thing to put in front of a new customer.
- [`interactive_agent/`](interactive_agent/README.md) — multi-turn LangGraph
  REPL with `recall_customer_context` and a checkpointer. Validates
  agent-shape behavior and the streaming/per-turn ingest path, not just
  "one write succeeded".
- [`client_direct_demo/`](client_direct_demo/README.md) — `AgentContextClient` via
  direct method calls (LangGraph host in this demo).
- [`client_hooks_demo/`](client_hooks_demo/README.md) — `AgentContextClient` via
  hooks + MCP tools (Claude Agent SDK host in this demo).

Both scripts use the SDK's auth classes to handle token issuance, so
you never paste a bearer token:

- `agentic_context.ContextAuth` (default in both demos) — JWT bearer
  two-leg exchange, service-account identity.
- `agentic_context.UserContextAuth.login()` (`--user-login`, basic demo
  only for now) — OAuth + PKCE loopback, end-user identity.

## Setup — do this once

You need Python 3.10+ (the setup script prefers 3.11). From the repo root,
run the shared setup script. It creates `.venv`, installs the SDK in
editable mode plus every demo's deps (LangGraph/Anthropic,
`claude-agent-sdk`, `anyio`, `httpx`) with pip, and creates
`examples/ace_agent.env` from the template if it's missing. Safe to re-run.

```bash
./examples/setup.sh
source .venv/bin/activate
python examples/<demo>/<demo>.py
```

Every command below assumes `.venv` is activated.

## Shared param file

Both demos read from the same file: **`examples/ace_agent.env`**. Copy
the template once, fill in your values, then run either demo:

```bash
cp examples/ace_agent.env.example examples/ace_agent.env
# then edit examples/ace_agent.env in any text editor
```

The file is gitignored (via `*.env` + `!*.env.example` rules) so your
secrets never get committed. Variables already set in the shell win
over the file, so you can override one value ad-hoc without editing:

```bash
BEACON_AGENT_ID=OtherAgent python examples/ace_basic_demo/ace_basic_demo.py list
```

### What's in the file

| Var | Who reads it |
|---|---|
| `SF_PRIVATE_KEY_PATH`, `SF_SUBJECT` | both — JWT bearer flow (default); unused with `--user-login` |
| `SF_CONSUMER_KEY`, `SF_LOGIN_URL` | both — needed for either auth flow |
| `BEACON_AGENT_ID` | both — Beacon-registered agent id |
| `BEACON_OWNER_ID` | both — Beacon owner; `--user-login` resolves it from the signed-in user instead |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_MODEL` | `interactive_agent/` only (LLM gateway) |

See `ace_agent.env.example` for inline descriptions and format rules
(KEY=value, `export` prefix tolerated, quotes stripped).

### Point at a different file

```bash
python examples/ace_basic_demo/ace_basic_demo.py --env-file ~/configs/beacon.env list
python examples/interactive_agent/interactive_agent.py --env-file ~/configs/beacon.env
```

## Ownership gotcha (shared)

Always set `BEACON_OWNER_ID`. Beacon accepts a create without
`conversationOwnerId` and returns 200, but list filters by `owner_id` —
a conversation ingested with no owner becomes unfetchable.
