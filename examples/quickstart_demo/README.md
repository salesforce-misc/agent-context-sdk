# quickstart_demo — minimum viable Claude integration

The ~30-line Claude Agent SDK integration from
[docs/QUICKSTART.md](../../docs/QUICKSTART.md), as a runnable one-file script.
Three steps: install → `UserContextAuth.login()` → `ClaudeAdapter(...).apply_to(ClaudeAgentOptions(...))`,
then one turn through `claude_agent_sdk.query(...)`.

Use this when onboarding to confirm the full stack works end-to-end for
Claude: Salesforce auth → Beacon lifecycle → Claude turn with the context pack
injected.

## Prereqs

1. **Python 3.10+ and the shared venv.** From the repo root (once):
   ```bash
   ./examples/setup.sh
   source .venv/bin/activate
   ```
   This installs the SDK and `claude-agent-sdk` into `.venv` with pip. See
   "Manual install" below if you'd rather not use the script.

2. **Shared param file** `examples/ace_agent.env` populated. `setup.sh`
   creates it from the template; then edit:
   ```bash
   # fill in SF_CONSUMER_KEY, SF_LOGIN_URL, BEACON_AGENT_ID, ANTHROPIC_API_KEY
   ```
   The file is gitignored. The `ANTHROPIC_*` block is required for this demo
   (unlike `ace_basic_demo`, which ignores it).

## Run

```bash
python examples/quickstart_demo/quickstart_demo.py "hello"
```

Pass any prompt as the single positional argument. If omitted the demo asks
"Hi — what do you remember about me?" so the context pack has something to
reflect back.

The **first** invocation opens a browser popup for OAuth sign-in; the refresh
token caches to `~/.cache/agentic_context_demos/refresh_token.json` (shared
with `ace_basic_demo` and `interactive_agent` — one login works for all three),
and every later run reuses it silently.

## What success looks like

```
[auth] user login via UserContextAuth (OAuth + PKCE loopback)   # first run only
[auth] signed in as 005xx000001X8UzAAK (owner_id=005xx000001X8UzAAK)   # or $BEACON_OWNER_ID if set
[session] quickstart-abc12345
[you] hello
[agent] Hi — I don't have any prior context on you yet. What should I know?
[done] turns=1
```

Subsequent runs under the same `BEACON_AGENT_ID` + `005…` get richer answers
as Beacon accumulates history.

## Env vars

| Var | Purpose |
|---|---|
| `SF_CONSUMER_KEY` | Connected App consumer key (client_id). |
| `SF_LOGIN_URL` | Salesforce login host, no scheme. |
| `BEACON_AGENT_ID` | `agent_id` header + a360 exchange `aid`. |
| `ANTHROPIC_API_KEY` | Anthropic API key, read by the Claude CLI subprocess. |
| `ANTHROPIC_BASE_URL` | Optional Anthropic-compatible endpoint. |
| `ANTHROPIC_MODEL` | Optional. Defaults to `claude-sonnet-4-5-20250929`. |

Owner id is **not** configured — resolved from the signed-in user via
`auth.user_id`.

## ECA setup

Same as `ace_basic_demo` — the Connected App behind `SF_CONSUMER_KEY` must have:

- Callback URL `http://localhost:53682/callback`
- Web-server flow enabled, **Require secret for web server flow** → off
- Scopes granted to cover `api refresh_token cdp_ingest_api`

## Swap to the service-account (`ContextAuth`) flow

The demo imports `build_auth` from the shared [`examples/_demo_auth.py`](../_demo_auth.py).
To run without a browser popup:

1. In `_demo_auth.py`, uncomment the `ContextAuth` import and the
   `build_auth_service_account` function.
2. In `quickstart_demo.py` change the import to
   `from _demo_auth import build_auth_service_account as build_auth`.
3. Add `SF_PRIVATE_KEY_PATH`, `SF_SUBJECT`, `BEACON_OWNER_ID` to `ace_agent.env`.

## Manual install

What `setup.sh` does, for this demo only:


```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install claude-agent-sdk anyio httpx
python examples/quickstart_demo/quickstart_demo.py "hello"
```

If `pip install -e .` fails with `No module named 'hatchling'`, run
`pip install "hatchling>=1.24"` first.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `missing env var: BEACON_AGENT_ID` | `examples/ace_agent.env` still has placeholders. |
| `missing env var: ANTHROPIC_API_KEY` | API key unset. The quickstart demo needs it; the basic demo does not. |
| Hangs with no output after `[session]` | A custom `ANTHROPIC_BASE_URL` that expects a Bearer token: set `ANTHROPIC_AUTH_TOKEN` instead of `ANTHROPIC_API_KEY`. |
| 403 from Beacon with `aid` message | `BEACON_AGENT_ID` not registered on this tenant. |
| Browser lands on `redirect_uri_mismatch` | ECA missing `http://localhost:53682/callback`. |

## How this differs from the other demos

| | `ace_basic_demo` | **`quickstart_demo`** | `interactive_agent` |
|---|---|---|---|
| Framework | none (raw SDK) | Claude Agent SDK | LangGraph |
| LLM call | no | yes (one turn) | yes (loop) |
| Context pack injected | no | yes (via `apply_to`) | yes (via `recall_customer_context`) |
| Best for | auth/transport shakedown | onboarding — "does Claude work?" | agent-shape eval |
