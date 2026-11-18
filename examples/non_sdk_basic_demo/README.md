# non_sdk_basic_demo — zero-SDK Salesforce auth + Beacon smoke test

Proves the full auth + transport path works end-to-end **without importing
`agentic_context`**. Every HTTP call is raw `httpx` so you can see what the
SDK does on your behalf.

Use this when you want to isolate "is the SDK broken?" from "is my Salesforce
org / ECA / Beacon tenant configured right?". If this demo works and
`ace_basic_demo` doesn't, the problem is on the SDK side.

## What it does

```
┌─ Leg 1 ─────────────────────────────────────────────┐
│ POST https://{SF_LOGIN_URL}/services/oauth2/token   │
│  grant_type=authorization_code (+ PKCE verifier)    │
│  → SF access_token, instance_url                    │
└─────────────────────────────────────────────────────┘
┌─ Leg 2 ─────────────────────────────────────────────┐
│ POST {instance_url}/services/a360/token             │
│  grant_type=urn:salesforce:grant-type:external:cdp  │
│  subject_token=<SF access_token>                    │
│  aid=<BEACON_AGENT_ID>                              │
│  → CDP access_token, Tenant-Specific Endpoint (TSE) │
└─────────────────────────────────────────────────────┘
┌─ Beacon call ───────────────────────────────────────┐
│ create : POST {TSE}/api/v1/agentic/conversations    │
│ list   : GET  {TSE}/api/v1/agentic/conversations    │
└─────────────────────────────────────────────────────┘
```

Owner id (`005…`) comes from `/services/oauth2/userinfo` on the signed-in SF
token — same user that just logged in through the browser.

## Prereqs

- Python 3.10+
- `pip install httpx` — the only dependency
- Same `examples/ace_agent.env` the basic demo uses (`SF_CONSUMER_KEY`,
  `SF_LOGIN_URL`, `BEACON_AGENT_ID`)
- Same ECA config (callback `http://localhost:53682/callback`,
  "Require secret for web server flow" off, scopes
  `api refresh_token cdp_ingest_api`)

## Run

```bash
python examples/non_sdk_basic_demo/non_sdk_basic_demo.py create \
    --conversation-id raw-demo-1 --name "raw-api-check"

python examples/non_sdk_basic_demo/non_sdk_basic_demo.py list --status ACTIVE
```

Each invocation opens a browser popup — this demo does **not** cache the
refresh token (keeping the file dependency-light; see `ace_basic_demo` for
the caching pattern).

## What success looks like

```
─── Leg 1: Salesforce OAuth (PKCE loopback) ───
[auth] opening browser → https://acme.my.salesforce.com/services/oauth2/authorize?...
[leg1] POST https://acme.my.salesforce.com/services/oauth2/token  → 200
[auth] signed in as 005xx000001X8UzAAK
[auth] core instance_url: https://acme.my.salesforce.com

─── Leg 2: a360 CDP exchange ───
[leg2] POST https://acme.my.salesforce.com/services/a360/token  → 200
[auth] TSE: https://....c360a.salesforce.com

─── Beacon call (create) ───
[create] POST https://....c360a.salesforce.com/api/v1/agentic/conversations  → 200
[create] conversationId: raw-demo-1
[create] body: {"conversationId":"raw-demo-1", ...}
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `port 53682 is in use` | A stale listener (e.g. a previous run) is still on it. `lsof -iTCP:53682` to find it. |
| Browser lands on `error=redirect_uri_mismatch` | ECA missing `http://localhost:53682/callback`. |
| Leg 1 returns `invalid_client_id` | `SF_CONSUMER_KEY` wrong, or ECA requires secret and this flow omits it. |
| Leg 2 returns `access_denied` with `aid` message | `BEACON_AGENT_ID` is not registered on this tenant. |
| Leg 2 returns 200 with no `access_token` | Transient a360 issue — the real SDK retries this once; this demo does not. Rerun. |
| `create` returns 403 from Beacon | Owner id / agent id combo isn't permitted to write on this tenant. |
| `list` returns `[]` after a successful `create` | `owner_type` mismatch — this demo hardcodes `employee` (matches `UserContextAuth` default). Conversations written under a service-account flow use `individual` and won't show up here. |
