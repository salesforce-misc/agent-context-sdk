# ace_basic_demo — thin ACE Agentic API demo

A single-file CLI that walks four Beacon lifecycle calls end-to-end —
`create`, `ingest`, `read`, `list` — using `UserContextAuth.login()`
(OAuth + PKCE loopback). End-user identity; opens a browser popup; the
signed-in user's `005…` becomes the owner.

Simplest possible shakedown of the auth + transport path. No LLM, no
LangGraph, no streaming — just the OAuth login and the SDK's lifecycle
helpers.

## Prereqs

1. **Python 3.10+ and the shared venv.** From the repo root (once):
   ```bash
   ./examples/setup.sh
   source .venv/bin/activate
   ```
2. **Shared param file** `examples/ace_agent.env` populated. Copy the
   template once, then edit:
   ```bash
   cp examples/ace_agent.env.example examples/ace_agent.env
   # then open examples/ace_agent.env and fill in SF_CONSUMER_KEY,
   # SF_LOGIN_URL, BEACON_AGENT_ID
   ```
   The file is gitignored. See `examples/ace_agent.env.example` for the
   schema. The `ANTHROPIC_*` block in the template is only read by the
   interactive demo and can be ignored here.

## Run

From the repo root:

```bash
# One full round-trip: create a conversation, ingest a user+agent turn,
# read it back, confirm it's in the list.
python examples/ace_basic_demo/ace_basic_demo.py create --name "trip-planner" --conversation-id conv-demo-1
python examples/ace_basic_demo/ace_basic_demo.py ingest --conversation-id conv-demo-1 \
    --user-message "What's my plan?" \
    --agent-response "Your plan is Gold PPO; deductible is \$500."
python examples/ace_basic_demo/ace_basic_demo.py read   --conversation-id conv-demo-1
python examples/ace_basic_demo/ace_basic_demo.py list   --status ACTIVE
```

The **first** invocation opens a browser popup for OAuth sign-in; after you
log in, the refresh token is cached in `~/.cache/agentic_context_demos/refresh_token.json`
(0600) and every subsequent run reuses it silently — no browser.

```
# first run
[auth] user login via UserContextAuth (OAuth + PKCE loopback)
<browser opens, you sign in>
[auth] signed in as 005xx000001X8UzAAK (owner_type=employee from auth default)

# every later run
[auth] cached refresh token reused (~/.cache/agentic_context_demos/refresh_token.json)
[auth] signed in as 005xx000001X8UzAAK (owner_type=employee from auth default)
```

To force a fresh login (new user, new org, revoked token), delete the cache
file:

```bash
rm ~/.cache/agentic_context_demos/refresh_token.json
```

When Salesforce rotates the refresh token on a refresh, the demo writes the
new value back to that same file automatically.

## Env vars

| Var | Purpose |
|---|---|
| `SF_CONSUMER_KEY` | Connected App consumer key (client_id). |
| `SF_LOGIN_URL` | Salesforce login host, no scheme (e.g. `acme.my.salesforce.com`). |
| `BEACON_AGENT_ID` | Goes into the `agent_id` header and the a360 exchange as `aid`. |

Owner id is **not** configured — it is resolved from the signed-in user
after login.

## ECA setup

The Connected App referenced by `SF_CONSUMER_KEY` must have, in addition to
whatever's already configured:

- A callback URL `http://localhost:53682/callback`
- The web-server flow enabled with **Require secret for web server flow** → off
- Scopes granted to cover `api refresh_token cdp_ingest_api`

If any of those are missing, the browser will land on a Salesforce error page
or the loopback listener will time out. Verify the ECA settings with your org
admin before relying on the flag.

## What each command does

| Subcommand | Transport | Purpose |
|---|---|---|
| `create [--name NAME] [--conversation-id CID]` | `AsyncIngestionClient.create_conversation(...)` | Creates a new conversation on Beacon owned by the signed-in user. |
| `ingest --conversation-id CID [--session-id SID] [--user-message MSG] [--agent-response MSG]` | `AsyncSessionManager.ingest([Record, ...])` | Writes one `user_message` + one `agent_response` under a shared `interaction_id`. SessionManager handles create_conversation → create_session → add_participants → add_entries lazily on the first call. |
| `read --conversation-id CID [--scope all\|TRACES]` | `AsyncRetrievalClient.get_conversation(...)` | Reads back the full message history for the conversation across every session. |
| `list [--status ACTIVE] [--page-size 20]` | raw `httpx` → `GET /api/v1/agentic/conversations` | Lists conversations for the signed-in user. The SDK's `AsyncRetrievalClient.list_conversations` exists but doesn't accept `status`/`page_size` filters, so this call stays raw. |

Every subcommand runs `UserContextAuth.aget_token()` first for the
`(bearer, base_url)` pair; the SDK paths additionally re-use `auth` on the
client so a mid-call 401 refreshes transparently.

## Overrides

Override the `owner_type` (defaults to `auth.default_owner_type` —
`"employee"` for UserContextAuth, `"individual"` for ContextAuth). Beacon
filters list reads by `(owner_id, owner_type)` together, so point this at
whichever bucket the conversations you want to see were written under:

```bash
# Read conversations the same 005… user wrote under a ContextAuth flow:
python examples/ace_basic_demo/ace_basic_demo.py --owner-type individual list
```

Point at a different param file:

```bash
python examples/ace_basic_demo/ace_basic_demo.py --env-file ~/configs/beacon.env list
```

Override a single value for one invocation (shell env wins over file):

```bash
BEACON_AGENT_ID=OtherAgent python examples/ace_basic_demo/ace_basic_demo.py list
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `missing env var: SF_...` | `examples/ace_agent.env` still has placeholder values. Edit it. |
| HTTP 403 with `aid` message | `BEACON_AGENT_ID` isn't registered on this tenant. |
| Browser lands on `error=redirect_uri_mismatch` | ECA is missing `http://localhost:53682/callback`. Add it. |
| Browser lands on `invalid_client_id` or `This app is blocked` | ECA secret requirement is on, or the user isn't permitted to use the ECA. Turn **Require secret** off and check user permissions. |
| Times out waiting for callback | Browser never redirected to loopback — port 53682 blocked, firewall ate the callback, or user closed the tab. Rerun. |
