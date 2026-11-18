# Agentic Context SDK — Developer Guide

**Version:** 0.7.0  
**Last Updated:** 2026-11-12  
**Audience:** Developers integrating AI agent conversation context into Salesforce Data Cloud

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Claude Agent SDK Adapter](#3-claude-agent-sdk-adapter)
4. [LangGraph Adapter](#4-langgraph-adapter)
5. [Google ADK Adapter](#5-google-adk-adapter)
6. [OpenAI Agents SDK Adapter](#6-openai-agents-sdk-adapter)
7. [Microsoft Foundry Adapter](#7-microsoft-foundry-adapter)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Introduction

The **Agentic Context SDK** gives your AI agent cross-session context powered by Salesforce Data Cloud. After integration, your agent can:

- **Remember** returning users across conversations
- **Recall** prior context (order numbers, preferences, past issues) without the user repeating themselves
- **Ingest** every conversation turn (messages, tool calls, reasoning) automatically

### Supported Frameworks

| Framework | Adapter class | Adapter package (import name) | Framework SDK (bring your own) |
|-----------|---------------|-------------------------------|--------------------------------|
| Claude Agent SDK | `ClaudeAdapter` | `agentic_context_claude` | `claude-agent-sdk` |
| LangGraph | `LangGraphAdapter` | `agentic_context_langgraph` | `langgraph`, `langchain-core` |
| Google ADK | `GoogleAdkAdapter` | `agentic_context_adk` | `google-adk` |
| OpenAI Agents SDK | `OpenAIAdapter` | `agentic_context_openai` | `openai-agents` |
| Microsoft Foundry (Microsoft Agent Framework) | `FoundryAdapter` | `agentic_context_foundry` | `agent-framework-foundry` |

All five adapters ship inside the single `agent-context-sdk` distribution, so
`pip install agent-context-sdk` is the whole install. There are no per-framework
extras (no `agent-context-sdk[claude]` etc.), and the per-adapter distributions
(`agentic-context-claude`, …) are not published. The package's own runtime
dependencies are `pydantic`, `requests`, `httpx` and `cryptography`. The
framework SDKs are NOT dependencies — you're expected to already have the one
your agent runs on (if not, `pip install <sdk>` using the package name from the
table above). Every adapter is re-exported from the top-level package, so one
import line reaches all five:

```python
from agentic_context import (
    ClaudeAdapter,
    FoundryAdapter,
    GoogleAdkAdapter,
    LangGraphAdapter,
    OpenAIAdapter,
)
```

Per-package imports work too, e.g. `from agentic_context_claude import ClaudeAdapter`.

Names resolve lazily (PEP 562 module `__getattr__`) — touching one name imports
only that adapter's package, so you can import `ClaudeAdapter` without the ADK /
LangGraph / Foundry / OpenAI SDKs present. What happens when the framework SDK is
missing depends on the adapter:

| Adapter | Framework SDK missing |
|---------|-----------------------|
| `LangGraphAdapter` | The import itself fails (the adapter module imports `langchain_core` at load time). Via `from agentic_context import LangGraphAdapter` you get `ImportError: LangGraphAdapter requires its framework SDK, which is not installed. Install it with:` followed by `pip install langgraph langchain-core`. |
| `ClaudeAdapter`, `GoogleAdkAdapter`, `OpenAIAdapter`, `FoundryAdapter` | The class imports and can be constructed (a `UserWarning` names the missing package). Ingestion-only use works; the first method that needs the SDK raises `ImportError: <Class>.<method>() requires … which is not installed. Install it with:` followed by `pip install claude-agent-sdk` / `pip install google-adk` / `pip install openai-agents` / `pip install agent-framework-foundry`. |

Pick the adapter for your framework and follow the corresponding section below.

---

## 2. Prerequisites

### What you need

| Value | Where to get it |
|-------|----------------|
| **Credentials** | One of the auth paths below: Connected App credentials (`ContextAuth`), a signed-in Salesforce user (`UserContextAuth`), or a single API key for a secret server (`ApiKeyAuth`). For every path except the LangGraph static-token path, the SDK obtains the CDP Bearer token for you and **discovers the Beacon `base_url` from the token exchange** — you do not pass a URL. |
| **Beacon URL** (`base_url`) | Only needed on the LangGraph **static-token** path (pre-minted token). It is the CDP tenant host returned as `instance_url` by the token exchange (see [Obtaining a CDP API Token manually](#obtaining-a-cdp-api-token-manually-static-token-path)). |
| **CDP API Token** | Only needed on the LangGraph static-token path. Otherwise the SDK mints and refreshes it. |
| **Agent ID** | The `agent_id` of an agent registered in Beacon — pass it to the adapter exactly as registered (e.g. `"<your-agent-id>"`) |
| **Owner ID** | The end user's identity (`owner_id`). Required by `ClaudeAdapter`, `OpenAIAdapter` and `FoundryAdapter` at construction; resolved per conversation or per call for `GoogleAdkAdapter` and `LangGraphAdapter` (see each adapter's section). |
| **Dataspace** | Your Beacon dataspace name — default is `"default"` unless your org uses a custom one. Set with the adapter's `dataspace=` argument or, on every adapter except `FoundryAdapter`, the `CDP_DATASPACE` environment variable (see [Environment variables](#environment-variables)). |
| **Python** | 3.10+ for all five adapters (the `agent-context-sdk` distribution declares `requires-python = ">=3.10"`) |

### Which credential arguments each adapter accepts

All five adapters accept a shared `auth=` object, and the same four inline
Connected App credentials (`private_key`, `consumer_key`, `subject`,
`login_url`). They differ in how the API-key path is spelled and in what wins
when more than one is supplied:

| Adapter | API-key arguments | Precedence when several are given |
|---------|-------------------|-----------------------------------|
| `ClaudeAdapter` | `secrets_api_key=`, `secrets_url=` | `auth=` > inline credentials > `secrets_api_key=` |
| `OpenAIAdapter` | `secrets_api_key=`, `secrets_url=` | `auth=` > inline credentials > `secrets_api_key=` |
| `FoundryAdapter` | `secrets_api_key=`, `secrets_url=` | `auth=` > inline credentials > `secrets_api_key=` |
| `GoogleAdkAdapter` | `agent_api_key=`, `secret_server_url=` | `auth=` > `agent_api_key=` > inline credentials |
| `LangGraphAdapter` | `api_key=`, `secret_server_url=` (see the note below) | `auth=` > inline credentials > `api_key=` as a secret-server key > static token |

- Inline credentials are only used when **all four** of `private_key`,
  `consumer_key`, `subject`, `login_url` are non-empty. `LangGraphAdapter`
  is stricter: a partial set raises `ValueError` ("Inline ContextAuth
  credentials are all-or-nothing …").
- If none of the paths applies, the constructor raises `ValueError`
  (`<Class> requires credentials. Pass auth=ContextAuth(...), inline private_key+consumer_key+subject+login_url, or secrets_api_key= …` — the ADK message is `GoogleAdkAdapter requires auth. Pass agent_api_key=…`; `LangGraphAdapter` raises `Self-driving LangGraphAdapter requires: <missing fields, e.g. api_key, base_url>. Either supply these connection parameters or pass session_manager= directly.`).
- `LangGraphAdapter(api_key=…)` alone is treated as a **secret-server API key**
  only when neither `base_url=` nor the `CDP_BASE_URL` environment variable is
  set; otherwise `api_key=` is taken as a pre-minted CDP token (static-token
  path).
- When the adapter builds the auth object for you, the token exchange runs at
  **construction** for `ClaudeAdapter`, `OpenAIAdapter` and `FoundryAdapter`
  (so bad credentials fail in the constructor), and **lazily on first use** for
  `GoogleAdkAdapter` (first turn) and `LangGraphAdapter` (on `async with adapter:`, which
  opens its clients).
- Every adapter also takes `owner_type=`. When omitted it defaults to the auth
  object's `default_owner_type`: `"individual"` for `ContextAuth` and
  `ApiKeyAuth` (and for the LangGraph static-token path), `"employee"` for
  `UserContextAuth`.

### Authentication: let the SDK manage your token (recommended)

The CDP Bearer token is obtained via a two-step OAuth JWT flow (sign a JWT →
core access token → exchange for a CDP token). **You no longer have to do this
yourself.** Hand the SDK your Connected App credentials once and it performs the
exchange, discovers your Beacon `base_url`, and refreshes the token
automatically — 5 minutes before it expires, and again (once, with a retry of the
request) if Beacon answers `401`:

```python
from agentic_context_claude import ClaudeAdapter

adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id="<end-user-id>",
    private_key=open("server.key").read(),       # PEM RSA private key
    consumer_key="3MVG...",                       # Connected App consumer key
    subject="your-username@example.com",          # Salesforce username for JWT bearer
    login_url="login.salesforce.com",             # login host (no scheme); test orgs: login.test.salesforce.com
)
# No base_url, no api_key, no refresh loop. The SDK exchanges + refreshes for you.
```

`login_url` must be a bare `*.salesforce.com` host: a scheme, path, port or
userinfo is rejected with `ValueError: login_url is outside the allowlist …`. The
`instance_url` values returned by the two exchange legs are validated the same
way (must not be `http://`, must be a `*.salesforce.com` host, no port/path).

This requires the `cryptography` package, which is a core dependency of
`agent-context-sdk` — so `pip install agent-context-sdk` already includes it.

Want a single token exchange shared across multiple adapters in the same
process? Construct one `ContextAuth` and pass it to each:

```python
from agentic_context import ContextAuth

auth = ContextAuth(private_key=..., consumer_key=..., subject=..., login_url=...)
adapter = ClaudeAdapter(agent_id="support-bot", owner_id="<end-user-id>", auth=auth)
```

`ContextAuth` also accepts `token_ttl=` (requested JWT lifetime in seconds,
default `3600`, also the fallback lifetime when the response omits
`expires_in`) and optional `aid=` / `xid=` (agent id and participant external id
sent on the second exchange leg; the CDP exchange rejects an unknown `aid` with
`access_denied`). Adapters that build a `ContextAuth` from inline credentials do
not set `aid` / `xid` — construct the `ContextAuth` yourself if you need them.

> **Need a Connected App?** Go to Salesforce Setup → App Manager → New Connected App. Enable OAuth, add the `cdp_ingest_api cdp_profile_api cdp_query_api` scopes, and use certificate-based authentication with an RSA key pair you generate. Your Salesforce admin can also provision this for you.

### Authentication: a signed-in Salesforce user (`UserContextAuth`)

`UserContextAuth` has the same public surface as `ContextAuth` (`get_token`,
`aget_token`, `get_core_creds`, `aget_core_creds`, `invalidate`, `attach`), so it
is passed as `auth=` the same way. The difference is leg 1 of the exchange: it
uses the user's own OAuth **refresh token** (Authorization Code + PKCE) instead
of a JWT signed with a private key, so the CDP token acts **as that user**
rather than as a service account. Its `default_owner_type` is `"employee"`, and
the signed-in user's `005…` id is available as `auth.user_id`. One instance
represents exactly one user — build one per user.

Local tool (CLI, notebook, desktop) — opens the browser and catches the callback
on a loopback listener (`127.0.0.1:53682` by default):

```python
from agentic_context import UserContextAuth

auth = UserContextAuth.login(
    consumer_key="3MVG...",              # Connected App / ECA consumer key (no secret needed)
    login_url="login.salesforce.com",    # login host, no scheme
    aid="<your-agent-id>",               # optional
    # port=53682, scope="api refresh_token cdp_ingest_api", timeout=300.0   (defaults)
    # on_refresh_token=save_fn,          # persist rotated refresh tokens
)
owner_id = auth.user_id
```

The app must register `http://localhost:<port>/callback` as a callback URL,
enable the web-server flow with "Require secret" off, and grant every requested
scope. `login()` blocks until the browser callback arrives or `timeout` expires.

Backend serving many users — host your own callback instead of `login()`:

```python
from agentic_context import FileLoginStateStore, FileRefreshTokenStore, UserContextAuth

# /login route
req = UserContextAuth.authorization_request(
    consumer_key=KEY, login_url=MY_DOMAIN_HOST, redirect_uri=CALLBACK_URL,
)
login_states.save(req)               # e.g. FileLoginStateStore("<dir>"); redirect the browser to req.url

# /callback route (code and state come from the query string)
saved = login_states.consume(state)  # None if unknown, expired (300 s default) or already used
login = UserContextAuth.redeem_code(
    code=code, state=state, expected_state=saved.state,
    code_verifier=saved.code_verifier, redirect_uri=saved.redirect_uri,
    consumer_key=KEY, consumer_secret=SECRET, login_url=MY_DOMAIN_HOST,
)
token_store.save(login.org_id, login.user_id, login.refresh_token)   # e.g. FileRefreshTokenStore("<dir>")

# any later request: one auth per user, refresh token kept in the store
auth = UserContextAuth.from_store(
    token_store, org_id=login.org_id, user_id=login.user_id,
    consumer_key=KEY, consumer_secret=SECRET, login_url=MY_DOMAIN_HOST,
)
```

`aredeem_code` is the async twin of `redeem_code`. When Salesforce rejects the
refresh token (expired, revoked, user deactivated) — or the code on redeem —
the SDK raises `ReauthRequired`; send the user back through login. The stores
(`LoginStateStore`, `RefreshTokenStore`, `InMemory…`, `File…`) are exported from
`agentic_context`.

### Authentication: the API-key path (`ApiKeyAuth`)

When your credentials are provisioned by a hosting platform — the demo's
**Context Builder / Secret Manager**, or any service that vends a credential
bundle — your app never sees the Connected App private key at all. It holds a
single **API key** and lets the SDK exchange it for the full credential bundle
at runtime.

**The one-liner.** Pass the key straight to the adapter — it builds the
`ApiKeyAuth` for you. The argument name differs per adapter (see the table
above); for `GoogleAdkAdapter` it is `agent_api_key=`:

```python
from agentic_context import GoogleAdkAdapter

adapter = GoogleAdkAdapter(
    agent_id="<your-agent-id>",
    agent_api_key="<the key the platform gave you>",
    # secret_server_url="https://<host>",  # optional; else AGENTIC_CONTEXT_SECRET_SERVER_URL
)
```

The same wiring for the other adapters is `secrets_api_key="<key>"` (+ optional
`secrets_url=`) on `ClaudeAdapter`, `OpenAIAdapter` and `FoundryAdapter`, and
`api_key="<key>"` (+ optional `secret_server_url=`, no `base_url`) on
`LangGraphAdapter`.

If you'd rather hold the auth object yourself (to share one auth across several
adapters, or to read `agent_id` off the bundle — see the note below), construct
`ApiKeyAuth` explicitly and pass it as `auth=`:

```python
from agentic_context import ApiKeyAuth

auth = ApiKeyAuth(
    api_key="<the key the platform gave you>",
    secret_server_url="https://<your-secret-manager-host>",  # see note below
)
adapter = GoogleAdkAdapter(agent_id="<your-agent-id>", auth=auth)
```

Both forms are identical downstream; the adapter argument is just the shorthand.
`ApiKeyAuth("")` raises `ValueError: ApiKeyAuth requires a non-empty api_key.`

**How the handshake works.** On the first token fetch (lazy — nothing happens
when `ApiKeyAuth` is constructed; see the table above for when each adapter
triggers it), `ApiKeyAuth` does a `GET {secret_server_url}/api/secret` with the
key in an `x-api-key` header (retrying transient failures). The Secret Manager
returns a bundle:

```jsonc
{
  "secret": {
    "privateKey":  "...",   // PEM RSA private key
    "consumerKey": "3MVG...",
    "subject":     "user@example.com",
    "loginUrl":    "login.example.salesforce.com",   // "aud" is accepted as a fallback key
    "agentId":     "<agent-id>"   // optional; may be empty
  }
}
```

`privateKey`, `consumerKey`, `subject` and a login host are required; a missing
`secret` object, an incomplete bundle, or a login host that is not
`*.salesforce.com` raises `ValueError` (the response body is never echoed).
`ApiKeyAuth` builds an inner `ContextAuth` from those fields and runs the *same*
JWT → core → CDP exchange as the credential path — from that point on the two
paths are identical. This is why it's called a **dual-auth** design: production
apps pass `ContextAuth(private_key=…)` directly and never touch a secret server;
platform-hosted apps pass `ApiKeyAuth(api_key=…)` and the private key stays on
the server. The SDK code downstream of auth is the same either way.

> **The secret-server host is not baked into the SDK.** `ApiKeyAuth` **fails
> closed**: if you pass neither `secret_server_url=` nor set the
> `AGENTIC_CONTEXT_SECRET_SERVER_URL` environment variable, the first token fetch
> raises `ValueError` rather than calling an unknown host. The host must also be
> `https://`. This is deliberate — the SDK never phones home to a default
> endpoint.

**One API key ↔ one `agentId`.** The bundle carries a single `agentId` for the
key. A multi-agent app therefore holds one key per agent (see
[§5.8](#58-multi-agent-handoff-one-key-per-agent)).

> **You still pass `agent_id` to the adapter.** `ClaudeAdapter`,
> `GoogleAdkAdapter`, `OpenAIAdapter` and `FoundryAdapter` do not read the
> bundle's `agentId` — construct them with an explicit `agent_id=`. If you want
> to source it from the bundle, resolve it after the first token fetch:
>
> ```python
> auth = ApiKeyAuth(api_key=KEY, secret_server_url=URL)
> await auth.aget_token()                      # triggers the bundle download
> agent_id = auth.agent_id or "<your-agent-id>"   # bundle may leave it empty
> adapter = GoogleAdkAdapter(agent_id=agent_id, auth=auth)
> ```
>
> The one exception is `LangGraphAdapter` on the API-key path: if you omit
> `agent_id` it adopts the bundle's `agentId` when its clients open, and raises
> `ValueError` if the bundle has none. An explicit `agent_id=` always wins.

### Obtaining a CDP API Token manually (static-token path)

If you'd rather mint the token yourself — for testing, or to manage rotation
out-of-band — **only `LangGraphAdapter`** accepts a pre-minted `base_url` +
`api_key` (see its [Constructor](#43-constructor)); the other four adapters
require one of the auth paths above. The two-step flow the SDK automates is:

**Step 1 — Sign a JWT and exchange for a core access token:**

```python
import time, jwt, requests   # `jwt` is PyJWT (pip install pyjwt) — not an SDK dependency

# Your Connected App credentials (from Salesforce Setup → App Manager)
CONSUMER_KEY = "your-connected-app-consumer-key"
PRIVATE_KEY  = open("your-private-key.pem").read()
LOGIN_URL    = "https://login.salesforce.com"   # or test: https://login.test.salesforce.com
SUBJECT      = "your-salesforce-username@example.com"

payload = {
    "iss": CONSUMER_KEY,
    "sub": SUBJECT,
    "aud": LOGIN_URL,
    "exp": int(time.time()) + 300,
}
signed = jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")

resp = requests.post(f"{LOGIN_URL}/services/oauth2/token", data={
    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
    "assertion": signed,
})
core_token         = resp.json()["access_token"]
core_instance_url  = resp.json()["instance_url"]   # the org/core host — used only to POST Step 2
```

**Step 2 — Exchange the core token for a CDP Bearer token:**

```python
resp = requests.post(
    f"{core_instance_url}/services/a360/token",
    data={
        "grant_type": "urn:salesforce:grant-type:external:cdp",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "subject_token": core_token,
    },
)
cdp_json      = resp.json()
cdp_token     = cdp_json["access_token"]
cdp_base_url  = cdp_json.get("instance_url", "")     # the CDP tenant host — this is CDP_BASE_URL
if cdp_base_url and not cdp_base_url.startswith("http"):
    cdp_base_url = "https://" + cdp_base_url
```

Both exchanges use a form-encoded body (`data=`, matching the SDK's own `_exchange`), not JSON.

Set `cdp_token` as `CDP_TOKEN` and `cdp_base_url` as `CDP_BASE_URL` — these are the variables the LangGraph static-token path reads. Note `CDP_BASE_URL` comes from **Step 2's** response `instance_url` (the CDP tenant host), **not** Step 1's `core_instance_url` (the org/login host) — the two are different URLs. The response also carries `expires_in` (seconds); the SDK schedules its refresh from it. With the static-token path nothing refreshes the token: when it expires Beacon answers `401` and (on the async client) the batch is dropped, so you must build your own refresh mechanism (re-run the exchange) and swap in a freshly-built adapter at a session boundary. **The credential-based path above does all of this for you** and is the recommended approach for production.

### Environment variables

These are all of the environment variables the SDK reads (the adapters and
core were searched for `os.environ` / `os.getenv`). Explicit constructor
arguments always win. They are read when the adapter (or `ApiKeyAuth`) is
constructed, and an empty string counts as unset — except
`FAST_CONTEXT_OPEN_RETRIES` / `FAST_CONTEXT_OPEN_BACKOFF_S`, which
`GoogleAdkAdapter` reads at each conversation open and which must be a valid
int / float (an empty string raises `ValueError`).

| Variable | Read by | Effect |
|----------|---------|--------|
| `CDP_TOKEN` | `LangGraphAdapter` (self-driving mode, i.e. no `session_manager=`) | Fallback for `api_key=` — a pre-minted CDP Bearer token (static-token path). |
| `CDP_BASE_URL` | `LangGraphAdapter` (self-driving mode) | Fallback for `base_url=` — your CDP instance URL. Also: when set, `api_key=` alone is treated as a static token, not as a secret-server key. |
| `CDP_DATASPACE` | `ClaudeAdapter`, `GoogleAdkAdapter`, `OpenAIAdapter`, `LangGraphAdapter` (self-driving mode) | Fallback for `dataspace=`; if unset too, `"default"`. **`FoundryAdapter` does not read it** — its `dataspace=` argument defaults to `"default"`. |
| `AGENTIC_CONTEXT_SECRET_SERVER_URL` | `ApiKeyAuth` — i.e. the API-key path of every adapter | Fallback for the secret-server base URL (`secret_server_url=` / `secrets_url=`). No default host exists; must be `https://`. |
| `FAST_CONTEXT_OPEN_RETRIES` (default `0`), `FAST_CONTEXT_OPEN_BACKOFF_S` (default `1.2`) | `GoogleAdkAdapter` only | Optional tuning: how many times (and with what pause in seconds) to re-read context at conversation-open when the first read comes back empty. |

```bash
export CDP_TOKEN="your-cdp-bearer-token"                       # LangGraph static-token path
export CDP_BASE_URL="https://your-tenant.c360a.salesforce.com" # LangGraph static-token path
export CDP_DATASPACE="default"                                 # optional
export AGENTIC_CONTEXT_SECRET_SERVER_URL="https://<your-secret-manager-host>"   # API-key path
```

On the credential-based (`ContextAuth`), user (`UserContextAuth`) and API-key
(`ApiKeyAuth`) paths, the CDP token and `base_url` come from the token exchange,
not from the environment. `agent_id` and `owner_id` are **not** read from the
environment on any path — pass them to the adapter (or, on `LangGraphAdapter`'s
API-key path, `agent_id` may come from the secret bundle's `agentId`).

---

## 3. Claude Agent SDK Adapter

> **TL;DR** — Create **one** `ClaudeAdapter` for your whole app. Per conversation, wire memory into your options in a single call: `options = adapter.apply_to(options, session_id, owner_id)`. That installs the on-demand **tools** (`recall_customer_context` and `get_context_pack`) and the hooks (transcript ingest, context-pack instruction + catalog). Everything else in this section is optional detail.

> **What changed (v0.5.0):** memory behaviour is controlled by a single `context_mode` knob (`"auto"` / `"manual"`, default `"auto"`), replacing the earlier `inject_context_per_turn` flag. In every mode the agent gets the `recall_customer_context` and `get_context_pack` tools; the mode only changes what the adapter injects on top. Context packs (§3.5.3) and the required constructor `owner_id` are also new. See [§3.5.1](#351-on-demand-memory-tool-recommended)/[§3.5.2](#352-memory-modes-auto--manual).

### 3.1 Quick Start (the 90% case)

This is the complete, recommended integration. If you only read one code block, read this one.

```python
import asyncio

from claude_agent_sdk import ClaudeAgentOptions, query
from agentic_context_claude import ClaudeAdapter

# 1. ONE adapter for the whole process (not one per conversation).
#    Pass credentials — the SDK runs the JWT → CDP exchange, discovers your
#    base_url, and refreshes the token automatically.
adapter = ClaudeAdapter(
    agent_id="support-bot",                      # must be registered in Beacon
    owner_id="default",                          # REQUIRED fallback owner; override per-conversation below
    private_key=open("server.key").read(),       # PEM RSA private key
    consumer_key="3MVG...",                       # Connected App consumer key
    subject="your-username@example.com",          # Salesforce username for JWT bearer
    login_url="login.salesforce.com",             # login host (no scheme)
)
# Hold the API key only? Pass secrets_api_key= instead — the SDK downloads the
# full credential set from the secrets server. See §3.6.

async def main() -> None:
    # 2. Per conversation: wire memory in one call. apply_to() installs the
    #    memory MCP server (both tools), the allowed-tool entries, and the
    #    UserPromptSubmit / Stop / PreCompact hooks onto a copy of your options
    #    without clobbering anything already there.
    options = adapter.apply_to(
        ClaudeAgentOptions(
            model="claude-sonnet-4-20250514",
            system_prompt="You are a helpful support agent.",
        ),
        session_id="conv-123",
        owner_id="user@example.com",
    )

    # 3. Run the agent. The agent recalls memory on demand; ingestion is automatic.
    async for message in query(prompt="What's the status of my order?", options=options):
        if hasattr(message, "result") and message.result:
            print(message.result)

    # 4. On shutdown (once for the whole app).
    await adapter.close()

asyncio.run(main())
```

> **Prefer to place each piece yourself?** Skip `apply_to()` and set `mcp_servers` / `allowed_tools` / `hooks` by hand — identical result. See the manual wiring in [§3.5.1](#351-on-demand-memory-tool-recommended).

That's the full integration. No `start_session()`, no manual `get_context()`, no hand-written hooks. `apply_to()` gives you:

- **the `recall_customer_context` tool** — the agent calls it **when it decides** it needs the customer's history (owner-scoped profile lookup).
- **the `get_context_pack` tool** — the agent calls it to read a named, server-advertised context pack (see [§3.5.3](#353-context-packs)).
- **the `Stop` ingest hook** — ingests each turn's transcript into Beacon (fire-and-forget in async mode).
- **the `UserPromptSubmit` hook** — attaches the pack catalog and the context-pack instruction (and, in `"auto"` mode, seeds fast memory once) as `additionalContext`.
- **the `PreCompact` hook** — makes the instruction and catalog be attached again after Claude compacts the conversation.

`apply_to()` does **not** modify your `system_prompt`. The memory-tool guidance text that teaches the agent *when* to call `recall_customer_context` is opt-in: append `adapter.context_system_prompt_guidance()` to your prompt yourself ([§3.5.1](#351-on-demand-memory-tool-recommended)).

**Lifecycle is lazy** — the Beacon conversation + session are created automatically on the first `Stop` hook. You do not need to create anything up front.

> **Want a different cadence?** The default `context_mode="auto"` seeds fast context once at conversation-open, then lets the agent drive. Switch to `"manual"` (no memory seeding; you own retrieval). See [§3.5.2](#352-memory-modes-auto--manual).

### 3.2 Minimal Requirements

| You need | Notes |
|----------|-------|
| `pip install agent-context-sdk` | Python 3.10+. |
| `pip install claude-agent-sdk` | Required **only** by the methods that import it: `hooks()`, `apply_to()`, `mcp_server()`, `default_tools()`, `make_context_retrieval_tool()`, `make_get_context_pack_tool()`. Each raises `ImportError` (with the install command) if it is missing. `push`, `ingest`, `get_context`, `get_session`, `list_sessions`, `start_session`, `close` and `context_system_prompt_guidance()` work without it. It is not pulled in automatically; constructing the adapter without it emits a `warnings.warn` (non-fatal). |
| A registered `agent_id` in Beacon | Required for `get_context()` to return data. See [§2 Prerequisites](#2-prerequisites). |
| Connected App credentials **or** a secrets-server API key | Recommended: `private_key` + `consumer_key` + `subject` + `login_url` (or a shared `ContextAuth`) and the SDK handles exchange + refresh. Alternatively pass `secrets_api_key` and the SDK downloads the credential set for you. See [§2 Authentication](#authentication-let-the-sdk-manage-your-token-recommended). |
| `owner_id` | **Required** at construction (no default; `ValueError` if missing or empty). It is the fallback end-user identity; override per conversation in `apply_to(...)` / `hooks(...)`. |

The **only two things you must do per conversation** are: pick a `session_id` (your conversation/thread ID) and an `owner_id` (the end user), then pass both to `apply_to()` (or to `hooks()` if you wire the pieces manually).

### 3.3 One Adapter, Many Sessions (the core mental model)

A single `ClaudeAdapter` instance manages **N sessions internally**, keyed by `session_id`. You create the adapter once and reuse it for every conversation:

```python
# App startup — create once. owner_id is required at construction; it is only
# the fallback, so in a multi-user app pass the real owner per conversation.
adapter = ClaudeAdapter(agent_id="support-bot", owner_id="service-account",
                        private_key=PRIVATE_KEY, consumer_key=CONSUMER_KEY,
                        subject=SUBJECT, login_url=LOGIN_URL)

# Conversation A
hooks_a = adapter.hooks("conv-A", "alice@example.com")

# Conversation B (concurrent, same adapter, different owner)
hooks_b = adapter.hooks("conv-B", "bob@example.com")

# App shutdown — close once. Drains all sessions + clients.
await adapter.close()
```

Internally the adapter caches one `SessionManager` (or `AsyncSessionManager`) per `session_id` and shares a single ingestion + retrieval client pair across all of them. The shared client pair is also why token handling is process-wide: one `ContextAuth` (or `ApiKeyAuth`) refreshes the token for **every** session at once.

#### 3.3.1 The ownership model: owner is *per-call*, not *per-adapter*

A common misconception is that you create one adapter per user (because the constructor requires `owner_id`). **You don't.** One adapter serves *all* users. The owner identity travels with each per-conversation call, not with the adapter instance:

```
adapter.hooks(session_id, owner_id="alice@x.com")   ← owner enters HERE, per call
        │
        ▼
SessionManager(owner_id="alice@x.com")               ← stored as the manager's own state
        │
        ├─ read : retrieval.get_context(owner_id=manager.owner_id)      ← recall is scoped to Alice
        └─ write: ingested user messages are attributed to Alice too    ← never the constructor default
```

Each conversation gets its own session manager carrying its own `owner_id`. Alice's and Bob's managers coexist inside the **one** shared adapter, never colliding. The constructor `owner_id` is merely the fallback used when a call omits its own owner — handy for single-user bots, irrelevant for multi-user servers.

The per-call owner scopes **both directions**: recall reads Alice's memory, *and* the Stop-hook transcript ingestion stamps Alice's user messages with her `owner_id` (the transformer's owner is resolved per turn, not baked in at construction). So in a shared process one user's messages are never mis-attributed to another user — or to the constructor default.

A `session_id` is bound to **one owner** for the lifetime of the adapter. If you later pass a *different* explicit `owner_id` for a `session_id` the adapter has already seen, the call raises `ValueError("session_id '…' is already bound to owner '…'; refusing to serve owner '…'. Session ids must not be reused across owners.")`.

| Operation | Owner comes from | Notes |
|-----------|------------------|-------|
| `hooks(session_id, owner_id=...)` / `apply_to(options, session_id, owner_id)` / `mcp_server(session_id, owner_id)` | the call (first call for a `session_id` binds it) | The primary path. Always pass the real end user. |
| `start_session(session_id, owner_id=...)` | the call | Same. |
| `push` / `ingest` / `get_context` / `get_session` (by `session_id`) | the **session's** manager | A `session_id` belongs to exactly one owner, so keying by session implicitly keys by the right owner. Ingested user messages are attributed to that owner. |
| `list_sessions(owner_id=...)` | the call | **Owner-level** query (no session to key on) — you *must* pass `owner_id` in a multi-user app, or you'll get the wrong owner's conversations. |
| `get_context` / `get_session` for an **unseen** `session_id` | the call's `owner_id`, else the adapter default | A "cold" call has no cached manager. Pass `owner_id=...` to scope it correctly. |

> **Rule of thumb:** in any app with more than one user, treat the constructor `owner_id` as a placeholder (for example a service-account id) and pass the authenticated user's identity to every `apply_to()` / `hooks()` / `list_sessions()` call. If you call `get_context()` / `get_session()` directly on a session the adapter hasn't already seen via `hooks()` / `apply_to()` / `mcp_server()`, pass `owner_id` there too.

> **Migration note (from older versions):** the constructor no longer takes `delivery_mode`, `conversation_id`, or a construction-time `session_id`, and you no longer build a per-conversation adapter pool. If you previously ran an `AdapterManager` keyed by conversation, replace it with one shared adapter and per-conversation `hooks()` / `apply_to()` calls. `session_id` is now passed to the per-conversation methods (`hooks`, `push`, `get_context`, …), not to the constructor.

### 3.4 Does this work for single-session chat apps?

**Yes — and it's simpler than the multi-session case.** A single-session app (a CLI chat, a notebook, a one-user desktop app) is just the N=1 case:

```python
adapter = ClaudeAdapter(agent_id="bot", owner_id="me@example.com",
                        private_key=PRIVATE_KEY, consumer_key=CONSUMER_KEY,
                        subject=SUBJECT, login_url=LOGIN_URL)

options = ClaudeAgentOptions(model="claude-sonnet-4-20250514",
                             hooks=adapter.hooks("my-only-conversation"))

async for msg in query(prompt="hello", options=options):
    ...

await adapter.close()
```

Notes specific to single-session apps:

- You can omit `owner_id` in `hooks()` / `apply_to()` — it falls back to the adapter's constructor `owner_id`.
- The retrieval methods (`get_context()`, `get_session()`) can be called with **no** `session_id` — they fall back to the most recently *created* session manager (the last `session_id` the adapter first saw), which is unambiguous when there is only one.
- `use_async=False` exists for synchronous callers, but read the caveats in [§3.12](#312-sync-mode) first.

> **Multi-session caveat:** in a server handling concurrent conversations, **always pass `session_id` explicitly** to `get_context()` / `get_session()` (or just use `hooks()`, which always does). The no-argument fallback resolves to the *last created* session, which is racy under concurrency. Inside `hooks()` the `session_id` is always bound explicitly, so the recommended path is always safe regardless of app shape. (`mcp_server()` / `apply_to()` bind the `session_id` you pass; if you omit it, the tools use the same last-created fallback at call time.)

### 3.5 How It Works (under the hood)

Memory recall and ingestion are wired separately:

| Path | When | What happens |
|------|------|--------------|
| `recall_customer_context` tool (via `mcp_server`) | The **agent** calls it when it decides it needs memory | Owner-scoped profile lookup. Registered in every mode. |
| `get_context_pack` tool (via `mcp_server`) | The **agent** calls it to read a named context pack | Pack-scoped read; see [§3.5.3](#353-context-packs). Registered in every mode. |
| `UserPromptSubmit` hook | Before each user message — behaviour set by `context_mode` | `"auto"` seeds fast memory as `additionalContext` once at conversation-open; `"manual"` seeds none. In both modes, the context-pack catalog (and, when the catalog lists a pack, the instruction) is attached on the first turn of each Claude conversation. See [§3.5.2](#352-memory-modes-auto--manual) and [§3.5.3](#353-context-packs). |
| `PreCompact` hook | Before Claude compacts the conversation | Marks the instruction and catalog for re-attaching on the next turn, since the summary may drop them. |
| `Stop` hook | After each agent turn completes | Marks this turn's instruction/catalog as delivered, reads new transcript lines, transforms them into records, ingests to Beacon. Uses `last_assistant_message` for the reply (the transcript may not be flushed yet). Always on. |

You normally never touch these directly — `apply_to()` wires them for you (or `mcp_server()` + `hooks()` if you wire manually). The lower-level methods (`push`, `get_context`, `ingest`, …) exist for advanced/manual integrations and are documented in [§3.8](#38-method-reference).

#### 3.5.1 On-demand memory tool (recommended)

The recommended integration installs two tools: `recall_customer_context` (profile memory) and `get_context_pack` (named packs, [§3.5.3](#353-context-packs)). The agent calls them when it judges they would help; the tools are bound to the session/owner you pass, so the model never handles customer IDs.

**Recommended — `apply_to()` wires everything in one call.** Hand it your `ClaudeAgentOptions` and it installs the MCP server, the allowed tools, and the three hooks:

```python
options = adapter.apply_to(
    ClaudeAgentOptions(model="claude-sonnet-4-...", system_prompt=MY_PROMPT),
    "conv-123", "user@example.com",
)
```

It returns a **new** options object (via `dataclasses.replace`) — your input is never mutated — and preserves everything already on it:

- **mcp_servers / allowed_tools** — the memory server is *added alongside* your existing ones (no clobber), and both wire names `mcp__<server_name>__get_context_pack` and `mcp__<server_name>__recall_customer_context` are appended to `allowed_tools` (each only if not already present). If `mcp_servers` already uses the `"context"` key for something else, `ValueError` is raised; pass `server_name="…"` to rename (the wire tools become `mcp__<server_name>__recall_customer_context` / `mcp__<server_name>__get_context_pack`). If `mcp_servers` is a filesystem config path (not a dict) `apply_to` raises `TypeError`, and so it does if `options` is not a dataclass.
- **hooks** — merged via `hooks(existing_hooks=…)`; your matchers still fire.
- **system_prompt** — never modified. It is read only to check whether it already contains the context-pack instruction (see below).
- **Idempotent** — applying twice (same `server_name`) refreshes the server entry and replaces the adapter's hooks rather than duplicating them or the tool entries.

`apply_to()` wires the same pieces in **every** `context_mode`; the mode only changes what the `UserPromptSubmit` hook injects on top (see [§3.5.2](#352-memory-modes-auto--manual)). So it's the right one-call wiring regardless of mode.

**Manual wiring (identical result, minus `apply_to`'s merging).** If you'd rather place each piece yourself, set the fields directly:

```python
options = ClaudeAgentOptions(
    system_prompt=MY_PROMPT + adapter.context_system_prompt_guidance(),   # opt-in text (see below)
    mcp_servers={"context": adapter.mcp_server("conv-123", "user@example.com")},
    allowed_tools=[                                                      # wire name: mcp__<server>__<tool>
        "mcp__context__recall_customer_context",
        "mcp__context__get_context_pack",
    ],
    hooks=adapter.hooks("conv-123", "user@example.com", system_prompt=MY_PROMPT),
)
```

Composition helpers for building your own server: `default_tools(session_id, owner_id)` returns the tool list (`[recall_customer_context, get_context_pack]`), `make_context_retrieval_tool(...)` returns the single recall tool, and `make_get_context_pack_tool(...)` returns the single pack tool (the only one that takes `param_defaults` / `timeout`; see [§3.5.3](#353-context-packs)).

The `recall_customer_context` tool issues a single owner-scoped `get_context()` read on every call and returns the payload as pretty-printed JSON text (`{"userContext": []}` when nothing came back). An optional `topic` argument is an LLM-visible hint only; it is logged but not sent on the wire.

**System-prompt text.** The memory-tool guidance that teaches the agent *when* to reach for `recall_customer_context` is opt-in: add it yourself via `adapter.context_system_prompt_guidance()`, which returns a newline-prefixed string you can concatenate in either direction; it never overwrites your prompt. The context-pack instruction is *not* put in the system prompt — the `UserPromptSubmit` hook attaches it to `additionalContext` (see [§3.5.3](#353-context-packs)).

#### 3.5.2 Memory modes: `auto` / `manual`

One knob, `context_mode`, controls how the adapter feeds memory into the conversation. It's re-read inside the `UserPromptSubmit` hook every turn, so `adapter.set_context_mode(...)` takes effect on the next turn. In **both** modes the tools are registered and the `Stop`-ingest hook runs — only the hook's memory-seeding behaviour differs. There are exactly two values; no other mode exists.

| `context_mode` | What the `UserPromptSubmit` hook does | Who drives recall |
|---------------|----------------------------------------|-------------------|
| `"auto"` (default) | Seeds **fast** context once, at conversation-open. | The agent — after the seed, it calls the tool when it needs more. |
| `"manual"` | **No memory seeding.** The context-pack catalog and instruction are still attached. | You — the tools are available, but no memory is auto-injected. |

```python
adapter = ClaudeAdapter(..., context_mode="auto")   # or "manual"
options = adapter.apply_to(options, "conv-123", "user@example.com")
```

**`"auto"` (default) — seed once, then agent-driven.** The first turn this process sees for a `session_id` injects the customer's fast profile as `additionalContext`; subsequent turns inject no memory and the agent pulls fresh memory itself via the tool. Best of both: a warm start without paying a retrieval on every turn.

**`"manual"` — you own it.** The adapter registers the tools and the ingest hook but seeds no memory. Wire your own retrieval steps, or rely entirely on the agent calling the tools.

Notes:

- **Where the text goes:** the seeded context rides the hook's per-turn **`additionalContext`** — *not* the system prompt. The context-pack catalog/instruction the hook attaches, and the opt-in `context_system_prompt_guidance()` text, are unchanged by the mode.
- **"Open" is per-process, per `session_id`.** The open turn is tracked in an in-memory set keyed by the `session_id` you passed to `hooks()`, and the turn is marked open in **either** mode. So "open" means the first turn *this adapter process* has seen for that `session_id`. If the process restarts mid-conversation, the next turn re-seeds fast context (auto) and re-fetches the pack catalog. Switching `manual` → `auto` later does **not** seed for a `session_id` that already had a turn. For a single long-lived adapter that's exactly seed-once; a restart-heavy or horizontally-scaled deployment may seed more than once.
- **Adapter injection is one-shot.** When the adapter injects memory itself (auto's open turn), it's a single fast read; further recall is always the agent's considered choice via the tool.
- Flip it live — `adapter.set_context_mode("auto"|"manual")`. Invalid values raise `ValueError` (at construction, and in `set_context_mode`). Input is normalised (`strip()`, lower-case, `_`/space → `-`), so `"MANUAL"` and `" auto "` are accepted.
- The Google ADK and OpenAI adapters also expose a `context_mode` knob with the same two values.

#### 3.5.3 Context packs

A **context pack** is a named, server-published bundle of structured data (enrollments, dependents, handbook passages, …) that the agent can read on demand. Beacon advertises the packs available to your agent as a **catalog**; the agent picks one and calls the `get_context_pack` tool. The Claude adapter wires all of this for you when you use `apply_to()` (or `hooks()` + `mcp_server()`).

**Catalog.** On the first turn the adapter sees for a `session_id` (the same "open" turn as [§3.5.2](#352-memory-modes-auto--manual), in both context modes), the `UserPromptSubmit` hook fetches the catalog with `GET /api/v1/agentic/runtime-context-packs` for your `agent_id`. It renders it as the marker line `[Available context packs]` followed by a JSON object keyed by pack id; each entry has `description`, `dataNeeds` (a list of `{name, dataNeedDescription, parameters}`) and, when Beacon reports an integer revision, `version`. The fetch is best-effort: on failure a warning is logged and the previous catalog (initially the empty `[Available context packs]\n{}`) stays in effect. The catalog and the allowed pack ids are **adapter-wide** state, shared by all sessions, and are refreshed only at an open turn (or patched by a `version_expired` result, below).

**Instruction.** `CONTEXT_PACK_INSTRUCTION` is the guidance that teaches the model how to use the catalog and `get_context_pack`. The hook attaches the server-published copy (`GET /api/v1/agentic/prompts/system/context_pack_instruction`, fetched once per session and pinned for it) and falls back to the bundled `CONTEXT_PACK_INSTRUCTION` (importable from `agentic_context_claude` or `agentic_context`) if the fetch misses or fails. It is attached **ahead of the catalog**, and **only** when the catalog lists at least one pack, `inject_context_pack_instruction` is true (`apply_to(..., add_system_prompt_guidance=True)`, the default), and your `system_prompt` does not already contain it (the check looks at a `str`, a preset's `append`, or a custom `prompt`, for either the server or bundled text). `adapter.get_context_pack_instruction(session_id)` (a coroutine) returns the text pinned for a session if you want to place it in your own prompt; pair it with `inject_context_pack_instruction=False`.

**Attach once, re-attach when needed.** Hook `additionalContext` stays in Claude's history, so the instruction and catalog are attached once per *Claude conversation* (keyed on the `session_id` in the hook input, which is Claude's own id, not yours). They are attached again when:

- a new Claude conversation id appears;
- a compaction happened (the `PreCompact` hook forgets the conversation's state, so the next turn re-attaches);
- the text changed (the catalog was refreshed with different content, or a `version_expired` result patched it);
- the turn that carried them never reached `Stop` (they count as delivered only when that turn's `Stop` hook fires).

If a hook input carries no Claude conversation id, the blocks are attached every turn. When there is nothing new to attach (no memory to seed, blocks already delivered), the hook returns `{}`.

**The `get_context_pack` tool.** Input schema: `context_pack_id` (string, required), `params` (string — a JSON object whose keys are the parameter names from the pack's data needs), `context_pack_version` (integer, optional). The owner is bound server-side from the session's owner; the model never supplies it (an `owner_id` key in `params` is discarded). Blank parameter values are dropped. If the model omits `context_pack_version`, the adapter fills in the version cached from the catalog. The tool never raises; it returns a JSON text block with one of these `status` values:

| `status` | Meaning |
|----------|---------|
| `ok` | `{"status": "ok", "context_pack_id": …, "context": <payload>}`. |
| `invalid_pack` | `context_pack_id` is not in the catalog (also returned if called before the catalog has been fetched). Includes `valid_pack_ids`. |
| `invalid_version` | `context_pack_version` was not an integer. |
| `version_expired` | The pack was republished since the catalog was fetched. Includes `requested_version`, `latest_version`, `latest`. The adapter updates its cached version and catalog so the retry (and the next catalog render) pins the new revision; the instruction tells the model to retry at most once. |
| `not_ready` | The read failed (endpoint unavailable, timeout, no active conversation/owner, or `use_async=False`). The model is told to fall back to what it has. |

`make_get_context_pack_tool(session_id=None, owner_id=None, *, param_defaults=None, timeout=CONTEXT_PACK_TIMEOUT_S)` builds the tool yourself. `param_defaults` is a dict of fixed `params` merged into every call (a key the model also sends overrides the default; if the model's value is blank the key is dropped, default included); `timeout` is the per-call HTTP timeout in seconds (`CONTEXT_PACK_TIMEOUT_S`, currently `60.0`). `default_tools()` / `mcp_server()` / `apply_to()` use the defaults. To use `param_defaults` or `timeout`, build your own server with your own tool **instead of** the default one (don't register two tools named `get_context_pack`):

```python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server

server = create_sdk_mcp_server(
    name="context", version="1.0.0",
    tools=[
        adapter.make_context_retrieval_tool("conv-123", "user@example.com"),
        adapter.make_get_context_pack_tool(
            "conv-123", "user@example.com", param_defaults={"<param-name>": "<value>"},
        ),
    ],
)
options = ClaudeAgentOptions(
    system_prompt=MY_PROMPT,
    mcp_servers={"context": server},
    allowed_tools=[
        "mcp__context__recall_customer_context",
        "mcp__context__get_context_pack",
    ],
    hooks=adapter.hooks("conv-123", "user@example.com", system_prompt=MY_PROMPT),
)
```

---

### 3.6 Constructor

```python
from agentic_context_claude import ClaudeAdapter

# Recommended — credential-based: SDK handles token exchange + refresh.
adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id="default",               # REQUIRED — fallback owner (see §3.3.1)
    private_key=open("server.key").read(),
    consumer_key="3MVG...",
    subject="your-username@example.com",
    login_url="login.salesforce.com",
    dataspace="default",
    use_async=True,
    context_mode="auto",              # default — seed fast context once at open,
                                     # then let the agent recall via the tool.
                                     # Other mode: "manual" (no seeding; you own
                                     # retrieval — §3.5.2)
)

# Alternative — API key only: the SDK downloads the full credential set from
# the secrets server, then behaves exactly like the credential path above.
adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id="default",
    secrets_api_key="your-api-key",
    secrets_url="https://secrets.example.com",  # or AGENTIC_CONTEXT_SECRET_SERVER_URL env var
)
```

The adapter accepts auth in one of three ways, in precedence order: a shared
`auth=ContextAuth(...)` instance, inline credentials
(`private_key`+`consumer_key`+`subject`+`login_url` — **all four** must be non-empty),
or a `secrets_api_key` (the SDK builds an `ApiKeyAuth` that downloads the
credentials from the secrets server at `GET {secrets_url}/api/secret`). You must
supply one of these or construction raises `ValueError`. In every case the SDK
runs the JWT → CDP exchange, discovers the Beacon `base_url`, and refreshes the
token for you — there is no pre-minted-token path. The first token fetch (and
therefore the secrets download, for `secrets_api_key`) happens **inside the
constructor**.

> **API-key path.** The `secrets_api_key` path is for the "customer only holds
> an API key" case — the connected-app private key, consumer key, subject and
> login URL are downloaded for them from the secrets server on the first token
> exchange. Note this API key is **not** the same thing as a CDP
> `api_key` bearer token. `secrets_url` must be
> `https://` and falls back to the `AGENTIC_CONTEXT_SECRET_SERVER_URL` env var —
> no host is baked into the SDK; the server-supplied login host is allowlisted
> to `*.salesforce.com`. Production may keep using explicit credentials or a
> shared `ContextAuth`; both coexist with this path. The equivalent core
> primitive is `ApiKeyAuth(api_key, secret_server_url=...)`, usable from any
> adapter; it also exposes `.agent_id` from the fetched bundle.

`agent_id` is the only positional argument; **every other parameter is keyword-only**.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `agent_id` | `str` | ✅ | — | Must match a registered agent config in Beacon |
| `owner_id` | `str \| None` | ✅ | `None` (→ `ValueError`) | **Required, non-empty.** Omitting it or passing `""` raises `ValueError("ClaudeAdapter requires owner_id. …")`. It is the **fallback** end-user identity, used only when a per-call owner is *not* supplied. In a multi-user app pass a stable service-account id here and the real owner per conversation via `hooks(session_id, owner_id=...)` / `apply_to(...)`. See [§3.3.1](#331-the-ownership-model-owner-is-per-call-not-per-adapter). |
| `auth` | `ContextAuth \| ApiKeyAuth \| None` | — | `None` | A shared `ContextAuth` (or `ApiKeyAuth`) instance. Use when the same credentials back multiple adapters so a single token exchange/refresh is shared. Takes precedence over the inline credentials and `secrets_api_key`. |
| `private_key` | `str` | — | `""` | PEM RSA private key. With `consumer_key`+`subject`+`login_url`, the SDK builds `ContextAuth` and runs the JWT → CDP exchange itself. Uses `cryptography`, a core dependency. |
| `consumer_key` | `str` | — | `""` | Connected App consumer key (the JWT `iss`). |
| `subject` | `str` | — | `""` | Salesforce username authorised for the JWT bearer flow (the JWT `sub`). |
| `login_url` | `str` | — | `""` | Salesforce login host, no scheme (e.g. `login.salesforce.com`; test orgs `login.test.salesforce.com`). |
| `secrets_api_key` | `str` | — | `""` | Secrets-server API key. Used when neither `auth` nor a complete set of inline credentials is given; the SDK builds an `ApiKeyAuth` that downloads the full connected-app credential set. **Not** a CDP bearer token — see the API-key note above. |
| `secrets_url` | `str` | — | `""` (→ `AGENTIC_CONTEXT_SECRET_SERVER_URL`) | Base URL of the secrets server used by `secrets_api_key`. Must be `https://`. Falls back to the `AGENTIC_CONTEXT_SECRET_SERVER_URL` env var when empty; if neither is set the first token fetch (i.e. construction) fails rather than sending the key anywhere. |
| `dataspace` | `str \| None` | — | `None` (→ `CDP_DATASPACE` env var → `"default"`) | Beacon dataspace name. An explicit non-empty value wins; otherwise the `CDP_DATASPACE` environment variable; otherwise `"default"`. |
| `owner_type` | `str \| None` | — | `None` | Owner entity type passed with every retrieval/lifecycle request. When `None`, the auth object's default is used: `"individual"` for `ContextAuth` / `ApiKeyAuth` (service-account auth), `"employee"` for interactive user auth (`UserContextAuth`). |
| `use_async` | `bool` | — | `True` | `True` selects the async clients/managers (non-blocking ingestion). `False` selects the sync ones. See [§3.12](#312-sync-mode) before using `False`. |
| `context_mode` | `str` | — | `"auto"` | Memory-behaviour switch, one of `"auto"` / `"manual"`. `"auto"` seeds fast context once at conversation-open, then the agent drives reads via the tool. `"manual"` seeds no memory (you own retrieval). Input is normalised; invalid values raise `ValueError`. See [§3.5.2](#352-memory-modes-auto--manual). |
| `http_observer` | `Callable[[dict], Any] \| None` | — | `None` | Optional. Fired for every Beacon HTTP call (request + response) so you can surface live traffic in a UI/console or tracer. Async-mode only (ignored with a logged warning when `use_async=False`). See [§3.11](#311-http-observability). |
| `agent_participant_variable` | `dict[str, Any] \| None` | — | `None` | **Full override** of Beacon's `AiAgentParticipant` block. Default: `{"AiAgentType": "ClaudeAgentSDK", "AiAgentVersionApiName": "v1", "AiAgentTemplateApiName": "default", "AiAgentApiName": <agent_id>}`. |

**Memory mode defaults to `"auto"`.** Out of the box the adapter seeds fast context once at conversation-open and then lets the agent recall memory by calling the tools when it needs them. Set `context_mode="manual"` to seed nothing (see [§3.5.2](#352-memory-modes-auto--manual)).

**Mode is locked at construction.** With `use_async=True` (default) the per-conversation methods use the async session manager, which schedules delivery in the background. `use_async=False` uses the blocking session manager (see [§3.12](#312-sync-mode) for the exact calling convention). `hooks()` is always synchronous to *call*; it returns `async def` or plain `def` hook callbacks to match the adapter's mode. The adapter exposes no `flush()`; ingestion is drained by `close()`.

> **Construct the adapter at startup, not inside a request handler.** With
> credential-based auth the constructor performs the JWT → CDP exchange
> synchronously (one-time) to obtain the first token. That's ideal at app
> startup, but if you build the adapter *inside* an already-running event loop it
> briefly blocks that loop. Create the one shared adapter once during startup
> (as the examples show) and this never affects request latency. After
> construction, token refresh is automatic: a token within 5 minutes of expiry is refreshed on the next request, and a `401` triggers one invalidate-and-retry.

> **What changed:** there is no `delivery_mode` and no `conversation_id`/`session_id` constructor argument anymore. Delivery is always per-turn (the `Stop` hook fires every turn). The session identity is supplied per call, not at construction.

---

### 3.7 `hooks()` — the ingestion + injection hook builder

Returns a ready-made hooks dict for `ClaudeAgentOptions(hooks=...)`, wiring both memory/context-pack injection and ingestion in one call. `apply_to()` calls this for you; reach for `hooks()` directly only when wiring the pieces manually or composing with your own hooks.

**Signature:**

```python
def hooks(
    session_id: str,
    owner_id: str | None = None,
    *,
    context_formatter: Callable[[dict], str] | None = None,
    existing_hooks: dict[str, list[HookMatcher]] | None = None,
    inject_context_pack_instruction: bool = True,
    system_prompt: Any = None,
) -> dict[str, list[HookMatcher]]
```

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `session_id` | `str` | ✅ | — | The Beacon session ID for this conversation. Bound into all hook closures. |
| `owner_id` | `str \| None` | — | adapter's `owner_id` | Owner override for this conversation. |
| `context_formatter` | `Callable[[dict], str] \| None` | — | built-in formatter | Custom formatter: turns the Beacon context payload into the string injected as `additionalContext` (auto-mode seed only). The built-in one renders `[Context from prior conversations]` followed by one `- [<sourceType>] <entry>` line per `userContext` entry, or an empty string when there are none. |
| `existing_hooks` | `dict[str, list[HookMatcher]] \| None` | — | `None` | Your own hooks dict. The memory matchers are **appended alongside** yours (both run; nothing overwritten); your dict is not mutated. Matchers this adapter added earlier are replaced, not duplicated. See [Composing with your own hooks](#composing-with-your-own-hooks). |
| `inject_context_pack_instruction` | `bool` | — | `True` | Attach the context-pack instruction ahead of the catalog (only when the catalog lists at least one pack). `False` leaves it out. |
| `system_prompt` | `str \| dict \| None` | — | `None` | Your `ClaudeAgentOptions.system_prompt`. If it already contains the instruction (server or bundled text, in a `str`, a preset's `append`, or a custom `prompt`), the hook doesn't attach it again. `apply_to()` passes it for you. |

**Returns:** a dict suitable for `ClaudeAgentOptions(hooks=...)`. Without `existing_hooks`, it has exactly three keys — `"UserPromptSubmit"`, `"Stop"`, and `"PreCompact"` — each a list with one `HookMatcher(matcher=".*", hooks=[callback])`. With `existing_hooks`, it's your dict plus the memory matchers merged in.

**Behavior notes:**

- Calling `hooks()` eagerly registers a session manager for that `session_id` (so the session is tracked even before the first turn), but **no Beacon call is made** until the first `Stop` fires (lazy lifecycle).
- The hook callbacks are best-effort: a failed memory fetch, catalog fetch, or ingest logs a warning and the agent turn proceeds normally. Memory is an enhancement, never a blocker.
- The hook callbacks are `async def` when `use_async=True`, plain `def` when `use_async=False`. Both produce the same dict shape.
- Requires `claude_agent_sdk` installed; otherwise raises `ImportError` naming `pip install claude-agent-sdk`.
- `ValueError` is raised if `session_id` is already bound to a different `owner_id` (see [§3.3.1](#331-the-ownership-model-owner-is-per-call-not-per-adapter)).

**`UserPromptSubmit` return shape:**

```python
{
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "<seeded memory (auto mode, open turn only)>\n\n<context-pack instruction>\n\n[Available context packs]\n{...}",
    }
}
```

The parts present are joined with a blank line. With the built-in formatter the seeded memory looks like `[Context from prior conversations]\n- [Profile] Prefers vegetarian\n- ...`. If there's nothing new to attach (no memory to seed, and the instruction and catalog are already in this Claude conversation), it returns `{}`. The `Stop` and `PreCompact` hooks always return `{}`.

**Custom context formatter:**

```python
def my_formatter(payload: dict) -> str:
    entries = payload.get("userContext", [])
    if not entries:
        return ""
    lines = ["<memory>"]
    for e in entries:
        lines.append(f"  <fact source='{e.get('sourceType', 'unknown')}'>{e.get('entry', '')}</fact>")
    lines.append("</memory>")
    return "\n".join(lines)

hooks = adapter.hooks("conv-123", "user@example.com", context_formatter=my_formatter)
```

Return an empty string from your formatter to suppress the memory part (the pack instruction and catalog are unaffected).

#### Composing with your own hooks

If your agent already passes hooks to `ClaudeAgentOptions` (your own `PreToolUse` guardrails, a custom `Stop`, logging on `UserPromptSubmit`, …), hand them to `hooks()` as `existing_hooks` and the memory hooks are merged in for you — **no manual merge, nothing clobbered**:

```python
my_hooks = {
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[block_dangerous_cmds])],
    "Stop":       [HookMatcher(matcher=".*", hooks=[my_logging_stop])],
}

options = ClaudeAgentOptions(
    model="claude-sonnet-4-20250514",
    hooks=adapter.hooks("conv-123", "user@example.com", existing_hooks=my_hooks),
)
```

Resulting hooks dict:

| Event | Matchers (in run order) |
|-------|--------------------------|
| `PreToolUse` | `[block_dangerous_cmds]` (yours, untouched) |
| `Stop` | `[my_logging_stop, <memory push>]` (yours first, ours appended) |
| `UserPromptSubmit` | `[<memory + pack injection>]` (ours) |
| `PreCompact` | `[<re-attach after compaction>]` (ours) |

The Claude Agent SDK runs **every** matcher in an event's list, so for shared events (`Stop`, `UserPromptSubmit`) both your hook and ours fire. Your `existing_hooks` dict is copied, not mutated.

> **Why this matters:** without `existing_hooks` you'd have to merge by hand, and the obvious `{**my_hooks, **adapter.hooks(...)}` would *silently overwrite* your `Stop` with ours (or vice versa). Passing `existing_hooks` avoids that footgun.

---

### 3.8 Method Reference

> Most apps only need `apply_to()` (or `hooks()`) and `close()`. The methods below are for manual integrations (custom hook wiring, non-SDK event sources, or building your own UI lifecycle).

The per-conversation methods `start_session`, `push`, `ingest`, `get_context`, `get_session`, `list_sessions`, `get_context_pack_instruction` and `close` are declared `async def` (coroutines) **in both modes** — always `await` them. With `use_async=False` the coroutine simply performs the blocking call inside. `hooks()`, `apply_to()`, `mcp_server()`, `default_tools()`, `make_*_tool()`, `context_system_prompt_guidance()` and `set_context_mode()` are plain synchronous calls.

#### `start_session` *(optional)*

Eagerly creates the conversation + session + participants in Beacon **before** the first message. Only needed if you want the lifecycle objects to exist up front (e.g. a UI that displays the session ID immediately). With `hooks()`, lifecycle is lazy and this is **not required**.

```python
async def start_session(session_id: str, *, owner_id: str | None = None, title: str = "") -> dict[str, str]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | The conversation/session ID |
| `owner_id` | `str \| None` | Owner override (defaults to constructor `owner_id`) |
| `title` | `str` | Optional human-readable title |

**Returns:** `{"sessionId": ..., "ownerId": ..., "agentId": ...}`. A 409 (already exists) is treated as success — returning users just work. Any other lifecycle failure raises `IngestionError`.

```python
await adapter.start_session("conv-123", owner_id="user@example.com", title="Support Chat")
```

#### `push`

Ingests one turn's transcript into Beacon. Called automatically by the `Stop` hook from `hooks()` — call it directly only if you wire your own Stop hook.

```python
async def push(session_id: str, input_data: dict, *, owner_id: str | None = None) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | The Beacon session to push to |
| `input_data` | `dict` | The Claude Agent SDK `Stop` hook input dict |
| `owner_id` | `str \| None` | Owner override; only used to scope a previously-unseen (cold) session, otherwise the session's existing owner is used (a different explicit owner for a bound session raises `ValueError`) |

`input_data` keys used: `transcript_path`, `session_id` (the SDK's internal ID), `last_assistant_message`. If `transcript_path` or `session_id` is missing, `push()` is a no-op. Lifecycle is lazy — the first `push()` for a session creates it.

```python
async def on_stop(input_data, tool_use_id, context):
    await adapter.push("conv-123", input_data)
    return {}
```

#### `ingest`

Ingests a list of `Record` objects directly — for custom event sources that aren't Claude transcripts. Most integrations use `push()` instead.

```python
async def ingest(session_id: str, records: list[Record], *, owner_id: str | None = None) -> None
```

Empty list is a no-op. `owner_id` (keyword-only) only scopes a previously-unseen (cold) session; otherwise the session's existing owner is used.

```python
from datetime import datetime, timezone
from agentic_context import Record

await adapter.ingest("conv-123", [
    Record(
        event_id="msg-001",
        session_id="conv-123",
        agent_id="support-bot",
        event_type="user_message",
        participant_id="user@example.com",
        content="What's my order status?",
        category="Message",
        timestamp=datetime.now(timezone.utc),
    ),
])
```

#### `get_context`

Retrieves cross-session context from Beacon (`POST /api/v1/agentic/context`). Used automatically by the `UserPromptSubmit` hook (auto mode, open turn) and the `recall_customer_context` tool; call directly only for manual injection.

```python
async def get_context(session_id: str | None = None, *, owner_id: str | None = None) -> dict | None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str \| None` | The session to fetch context for. If `None`, falls back to the most recently created session manager (safe for single-session apps; **pass explicitly** in multi-session servers). |
| `owner_id` | `str \| None` | Only used for a **cold** call — a `session_id` the adapter hasn't seen before (no prior `hooks()`/`start_session()`). Scopes the new session to the right owner instead of the adapter default. For a warm session its manager already carries the owner. **Pass it in multi-user apps.** |

**Returns:** a context `dict` (entries under `"userContext"`), or `None` on any error or if no session is bound. Retrieval errors are logged and swallowed. Requires the `agent_id` to be registered in Beacon.

```python
context = await adapter.get_context("conv-123")
if context:
    system_prompt += f"\n\nPrior context:\n{context}"
```

#### `mcp_server` / `default_tools` / `make_context_retrieval_tool` / `make_get_context_pack_tool`

Build the on-demand tools (see [§3.5.1](#351-on-demand-memory-tool-recommended) and [§3.5.3](#353-context-packs)). All four require `claude_agent_sdk` installed.

```python
def mcp_server(session_id=None, owner_id=None, *, name="context")        # → MCP server (version "1.0.0") holding default_tools(), for ClaudeAgentOptions(mcp_servers=...)
def default_tools(session_id=None, owner_id=None) -> list                # → [recall_customer_context, get_context_pack]
def make_context_retrieval_tool(session_id=None, owner_id=None)          # → the recall_customer_context tool object
def make_get_context_pack_tool(session_id=None, owner_id=None, *,
                               param_defaults=None,
                               timeout=CONTEXT_PACK_TIMEOUT_S)           # → the get_context_pack tool object
```

`mcp_server()` is the batteries-included path. Use the `default_tools()` / `make_*` helpers to compose the tools into your own MCP server. The wire names are `mcp__<name>__recall_customer_context` and `mcp__<name>__get_context_pack`; `mcp_server()` does not touch `allowed_tools`, so list them yourself (or use `apply_to()`).

#### `apply_to`

One-call wiring over the primitives above — returns a **new** `ClaudeAgentOptions` (input never mutated) with the memory server, both allowed tools, and the Stop / UserPromptSubmit / PreCompact hooks merged in without clobbering what's already there (see [§3.5.1](#351-on-demand-memory-tool-recommended)). Requires `claude_agent_sdk`.

```python
def apply_to(options, session_id=None, owner_id=None, *,
             server_name="context", add_system_prompt_guidance=True)  # → new ClaudeAgentOptions
```

`add_system_prompt_guidance` is a legacy name: it controls whether the **context-pack instruction** is attached by the `UserPromptSubmit` hook (it is passed to `hooks(inject_context_pack_instruction=…)`). It does not touch your `system_prompt`. Raises `ValueError` if `server_name` is already used by a different MCP server, and `TypeError` if `options` is not a dataclass or its `mcp_servers` is a filesystem config rather than a dict.

#### `context_system_prompt_guidance`

Returns opt-in, append-safe system-prompt text teaching the agent *when* to call `recall_customer_context` (see [§3.5.1](#351-on-demand-memory-tool-recommended)). Pure text — no `claude_agent_sdk` needed. Concatenate onto your own prompt; it never overwrites it. Pass `tool_name=` only if you renamed the tool.

```python
def context_system_prompt_guidance(*, tool_name="recall_customer_context") -> str
```

#### `get_context_pack_instruction`

Returns the context-pack instruction pinned for a session (server-published text, or the bundled `CONTEXT_PACK_INSTRUCTION` on a miss/failure). `hooks()` / `apply_to()` already inject it, so you only need this when placing the instruction in your own prompt (with `inject_context_pack_instruction=False`). See [§3.5.3](#353-context-packs).

```python
async def get_context_pack_instruction(session_id: str) -> str
```

#### `context_mode` / `set_context_mode`

Read or flip the memory mode at runtime (see [§3.5.2](#352-memory-modes-auto--manual)). The mode is re-read inside the `UserPromptSubmit` hook, so a switch takes effect on the next turn.

```python
adapter.context_mode              # property → "auto" | "manual"
adapter.set_context_mode("manual") # re-read on the next turn
```

`set_context_mode` and the constructor both accept common spellings (`"MANUAL"`, `" auto "` → `"manual"` / `"auto"`); an unrecognised value raises `ValueError`.

#### `get_session`

Retrieves the full session data (all entries, metadata) from Beacon. Same `session_id` and `owner_id` semantics as `get_context` (`owner_id` only matters for a cold, previously-unseen session). Returns `dict | None`; retrieval errors are logged and `None` is returned.

```python
session = await adapter.get_session("conv-123")
```

#### `list_sessions`

Lists all conversations **for a given owner**. Returns `list[dict]`, or `[]` on error.

```python
async def list_sessions(owner_id: str | None = None) -> list[dict]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `owner_id` | `str \| None` | Owner to scope the listing to. Defaults to the adapter's fallback owner. **Always pass the current end user's `owner_id` in a multi-user app** — the adapter is a single process-wide instance, so an unscoped call returns whichever owner the adapter defaults to, *not* the current user's conversations. |

```python
for s in await adapter.list_sessions(owner_id="alice@example.com"):
    print(s.get("id"), s.get("name"))
```

#### Token refresh

You never rotate the token by hand. The SDK refreshes it automatically (on the next request once the token is within 5 minutes of expiry, and by invalidating and re-fetching on a `401`) and propagates the fresh token to every session's ingestion + retrieval clients — one `ContextAuth` (or `ApiKeyAuth`) covers **all** sessions at once.

#### `close`

Closes **all** session managers and the shared clients. Call once on app shutdown. In async mode it drains pending background deliveries (30 s budget per session, then cancels stragglers) before closing. Errors closing individual sessions are logged, not raised. After `close()`, the adapter's session cache is cleared; the shared clients are closed, so don't reuse the adapter.

```python
async def close() -> None
```

**In FastAPI:**

```python
@app.on_event("shutdown")
async def shutdown():
    await adapter.close()
```

#### Properties

```python
adapter.agent_id      # → str
adapter.session_id    # → str | None — the most recently created session manager's id (backward-compat shim)
adapter.context_mode  # → "auto" | "manual"
adapter.capabilities  # → Capabilities(working=True, persistent=False)  (class attribute)
```

> `session_id` returns the id of the last session manager the adapter created and exists mainly for backward compatibility. In a multi-session app, don't rely on it to identify "the" session — there isn't one.

---

### 3.9 End-to-End Example: FastAPI Agent (one shared adapter)

This is the production pattern: a **single** adapter created once at startup
(it handles token exchange + refresh itself), `apply_to()` per request, closed on
shutdown. No per-request token plumbing.

```python
import os
from fastapi import FastAPI
from pydantic import BaseModel
from claude_agent_sdk import ClaudeAgentOptions, query
from agentic_context_claude import ClaudeAdapter

app = FastAPI()

DATASPACE = os.environ.get("BEACON_DATASPACE", "default")
AGENT_ID = "support-bot"

# ── One adapter for the whole process, built once at startup ───────
# Credential-based: the SDK runs the JWT → CDP exchange, discovers base_url,
# and refreshes the token automatically (before expiry and on 401). No
# get_fresh_token() helper, no manual token plumbing.
_adapter = None

@app.on_event("startup")
async def startup():
    global _adapter
    _adapter = ClaudeAdapter(
        agent_id=AGENT_ID,
        # owner_id is required but is only the fallback: the real owner is
        # passed per request via apply_to(options, conversation_id, body.owner_id).
        owner_id="service-account",
        private_key=open(os.environ["BEACON_PRIVATE_KEY_FILE"]).read(),
        consumer_key=os.environ["BEACON_CONSUMER_KEY"],
        subject=os.environ["BEACON_SUBJECT"],
        login_url=os.environ.get("BEACON_LOGIN_URL", "login.salesforce.com"),
        dataspace=DATASPACE,
    )

class MessageRequest(BaseModel):
    text: str
    owner_id: str

@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, body: MessageRequest):
    # One call wires the memory + context-pack tools and the ingest/injection hooks.
    options = _adapter.apply_to(
        ClaudeAgentOptions(
            model="claude-sonnet-4-20250514",
            system_prompt="You are a helpful support agent.",
        ),
        conversation_id, body.owner_id,
    )

    response = ""
    async for message in query(prompt=body.text, options=options):
        if hasattr(message, "result") and message.result:
            response = message.result
    return {"response": response}

@app.on_event("shutdown")
async def shutdown():
    if _adapter:
        await _adapter.close()
```

### 3.10 Returning Users & Cross-Session Memory

Use the same `session_id` to continue a conversation; use the same `owner_id` (with a new `session_id`) to start a fresh conversation that still recalls the user across sessions. Beacon scopes cross-session context to `owner_id`.

```python
adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id="service-account",
    private_key=PRIVATE_KEY, consumer_key=CONSUMER_KEY,
    subject=SUBJECT, login_url=LOGIN_URL,
)

# Conversation 1 — order inquiry (owner passed per call)
hooks1 = adapter.hooks("conv-001", "user@example.com")
# ... agent runs; Stop hook ingests each turn ...

# Conversation 2 (days later) — SAME session_id continues the same conversation
hooks1_again = adapter.hooks("conv-001", "user@example.com")
ctx = await adapter.get_context("conv-001")   # owner-scoped cross-session context

# Or a NEW conversation for the same user — cross-session recall via owner_id
hooks2 = adapter.hooks("conv-002", "user@example.com")
ctx2 = await adapter.get_context("conv-002")   # Beacon owner cache returns prior facts
```

A 409 from Beacon (session already exists) is treated as success, so reusing a `session_id` for a returning user just works.

### 3.11 HTTP Observability

Pass `http_observer` to watch **every** Beacon HTTP call the adapter makes — both ingestion writes and retrieval reads. It's the seam for surfacing live Beacon traffic in a UI/console or wiring a request tracer. (Distinct from `IngestionCallbacks`, which are ingestion-only, per-batch, and carry records rather than HTTP metadata. The adapter does not accept `IngestionCallbacks`.)

```python
async def on_beacon_call(event: dict) -> None:   # may be sync or async
    # phase="request":  {"phase","callId","method","url","requestBody","timestamp"}
    # phase="response": {"phase","callId","method","url","status","latencyMs","responseBody","timestamp"}
    print(event["phase"], event["method"], event["url"], event.get("status"))

adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id="default",
    private_key=PRIVATE_KEY, consumer_key=CONSUMER_KEY,
    subject=SUBJECT, login_url=LOGIN_URL,
    http_observer=on_beacon_call,
)
```

- **`callId`** pairs a request event with its response event.
- **`url` has no query string** — it is the full request URL (scheme, host, path) with the query string removed before the event is built, so tokens / owner_ids in query params never reach your observer/UI.
- **Best-effort** — observer exceptions are swallowed; observability never disrupts delivery.
- **Async-mode only** — wired via httpx event hooks. In sync mode the argument is ignored with a logged warning.
- **Bodies are included** — the request event carries `requestBody` and the response event carries `responseBody`, both best-effort decoded (`None` when empty), JSON pretty-printed, and capped at 20,000 characters (larger bodies are truncated with a `… (truncated)` suffix). Headers (including the bearer token) and query strings are stripped, but the JSON payloads themselves are surfaced.
- **Labelling is yours** — map `method` + `url` to friendly labels (e.g. `…/entries` → "Add Entries", `…/context` → "Get Context").

The support demo uses this to drive a live "Beacon API Console" panel over SSE.

### 3.12 Sync Mode

`use_async=False` selects the blocking client/session-manager pair: ingestion blocks until the POST completes (with retry), instead of being scheduled as a background task. Read these caveats before choosing it:

- The adapter's per-conversation methods stay `async def` in this mode. `adapter.close()` without `await` returns an un-run coroutine and closes nothing — use `await adapter.close()` (or `asyncio.run(adapter.close())`).
- `hooks()` returns **plain `def`** callbacks. The Claude Agent SDK awaits every hook callback (it does `await callback(...)`, `claude_agent_sdk/_internal/query.py`, v0.2.148), and `query()` is an async generator, so a sync-mode hooks dict cannot be driven by `query()` / `ClaudeSDKClient` as-is. Use sync mode for the non-hook methods (`ingest`, `get_context`, `get_session`, `list_sessions`, `start_session`) from synchronous-style code, and use the default async mode with `apply_to()` / `hooks()` for agents.
- The `get_context_pack` tool returns `status: "not_ready"` in sync mode (the pack read needs the async retrieval client).

```python
import asyncio

adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id="user@example.com",
    private_key=PRIVATE_KEY, consumer_key=CONSUMER_KEY,
    subject=SUBJECT, login_url=LOGIN_URL,
    use_async=False,
)

async def main() -> None:
    ctx = await adapter.get_context("conv-123", owner_id="user@example.com")  # blocks inside the coroutine
    await adapter.close()

asyncio.run(main())
```

In async mode ingestion is non-blocking — the transcript is parsed and delivery is scheduled as a background task with automatic retry (4 attempts), so the next turn starts immediately; `close()` drains pending deliveries. If 1000 deliveries are already pending the new batch is dropped with a logged error.

### 3.13 Go-Live Checklist

- [ ] **One** adapter instance for the app (not one per conversation)
- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] `owner_id` is passed to the constructor (required) and the **real end-user** identity is passed to `apply_to()` / `hooks()` per conversation (constructor `owner_id` is just the fallback)
- [ ] `owner_id` passed to `list_sessions()` is the current user (an unscoped call returns the wrong owner's conversations — see [§3.3.1](#331-the-ownership-model-owner-is-per-call-not-per-adapter))
- [ ] `session_id` passed explicitly to `get_context()` / `get_session()` in multi-session servers (and `owner_id` too if the session may be cold)
- [ ] `claude_agent_sdk` installed (required for `apply_to()` / `hooks()` / `mcp_server()`)
- [ ] Auth configured — the credential path (`private_key`/`consumer_key`/`subject`/`login_url`), a shared `ContextAuth`, or `secrets_api_key` — all auto-refresh the token for every session; no manual token rotation needed
- [ ] `await adapter.close()` wired to your shutdown handler (one call drains all sessions)
- [ ] `use_async=False` only if you understand the sync-mode caveats in [§3.12](#312-sync-mode) (default is async)
- [ ] `get_context()` returns data (confirms the agent config exists in Beacon)
- [ ] If you use context packs: the catalog fetch succeeds (the first turn's `additionalContext` contains `[Available context packs]` with your pack ids) and `get_context_pack` returns `status: "ok"`

---

## 4. LangGraph Adapter

### 4.1 Installation

```bash
pip install agent-context-sdk
```

**Requirements:** Python 3.10+, plus the LangGraph stack you already run your agent on: `langgraph>=0.2` and `langchain-core>=0.3`. Those two are **not** installed by `agent-context-sdk` — if your environment lacks them, `pip install langgraph langchain-core`.

`LangGraphAdapter` ships inside the single `agent-context-sdk` distribution (there is no separate `agentic-context-langgraph` package to install). Import it from either place:

```python
from agentic_context_langgraph import LangGraphAdapter   # the adapter package
from agentic_context import LangGraphAdapter              # same class, re-exported from the top-level package
```

Everything else in this section (`TurnEvent`, `make_context_node`, the policy classes, …) is imported from `agentic_context_langgraph`.

### 4.2 How It Works

The LangGraph adapter is a single class — `LangGraphAdapter` — with **three** construction paths:

- **All-in-one / self-driving (recommended)** — pass your `graph=` plus connection params (`agent_id`, `api_key`, `base_url`, `dataspace`, and an `owner_id`) or an `auth=` object. The adapter constructs and owns the `AsyncIngestionClient`, the `AsyncRetrievalClient`, and one `AsyncSessionManager` per `(session_id, owner)` pair. Use it as an async context manager. Inside `stream()`, the adapter automatically calls `POST /api/v1/agentic/context` (subject to the policy behind `context_mode`), prepends a `SystemMessage(id="agentic-memory-context")` to your inputs, drives the graph, and ingests records per turn.
- **`graph=` + `session_manager=`** — pass a live compiled graph plus your own pre-built `SessionManager` (sync) or `AsyncSessionManager` (async). The adapter never auto-fetches memory in this mode (`context_mode` is inert) and `default_tools()` returns `[]` — you wire retrieval yourself via `make_context_node` (4.9) or by calling `AsyncRetrievalClient` directly. The adapter never POSTs to Beacon directly — all transport flows through your manager.
- **`pickle_path=` + `session_manager=` + `allow_pickle=True`** — load a serialized `StateSnapshot` from disk. `snapshot()` works; `stream()` raises `ValueError`. Used for cross-process file handoff. `pickle_path=` without `session_manager=` raises `ValueError`.

The all-in-one form is the path that drops customer code to **~6 lines** in the minimum case:

```python
import asyncio, os
from agentic_context_langgraph import LangGraphAdapter

async def main():
    async with LangGraphAdapter(
        graph=my_compiled_graph, agent_id="support-bot", owner_id="sarah@corp.com",
    ) as adapter:                                       # api_key / base_url come from env (CDP_TOKEN / CDP_BASE_URL)
        await adapter.run("Hey, I'm back", session_id="ses-2")

asyncio.run(main())
```

That is the entire integration. The customer **never imports `agentic_context`** (the middleware SDK) — only `agentic_context_langgraph`. They never construct `AsyncIngestionClient`, `AsyncRetrievalClient`, or `AsyncSessionManager`; they never call `get_context`, `ingest`, or `close`; they never normalize a LangChain message into a `Record`. The adapter wraps and hides all of it.

#### What you write vs. what the adapter handles

| Concern | You write | Adapter handles internally |
|---|---|---|
| Compiled `StateGraph` (your nodes, edges, model, tools) — or a `(sdk_tools) -> compiled graph` builder (4.3) | ✅ | — |
| Connection identity (`agent_id`, `owner_id`, `api_key`, `base_url`, `dataspace`) or `auth=` | ✅ (kwargs or env) | — |
| Per-call `session_id` (and, for multi-tenant servers, per-call `owner_id`) | ✅ | — |
| Construct `AsyncIngestionClient` | — | ✅ |
| Construct `AsyncRetrievalClient` | — | ✅ |
| Construct `AsyncSessionManager` (one per `(session_id, owner)`) | — | ✅ |
| Decide when to `POST /api/v1/agentic/context` | — | ✅ via `ContextPolicy` (chosen by `context_mode`) |
| Render Beacon payload → `SystemMessage(id="agentic-memory-context")` | — | ✅ via `context_to_system_message` |
| Provide the `recall_customer_context` and `get_context_pack` tools (`default_tools()`) | bind them in your graph (4.3) | builds them |
| Fetch the context-pack catalog and inject it, with the instruction, as `SystemMessage(id="agentic-context-pack-list")` | — | ✅ |
| Inject the recall-tool guidance as `SystemMessage(id="agentic-memory-recall-tool-instruction")` (`auto` / `per-turn` only) | — | ✅ |
| Drive `astream_events(version="v3")` | — | ✅ |
| Normalize each LangChain v3 message → `Record` (stable `event_id`, role mapping, JSON-safe `usage`) | — | ✅ |
| Ingest records per turn | — | ✅ via `AsyncSessionManager.ingest()` |
| Close all per-session managers, drain in-flight ingests (30 s budget) | — | ✅ via `__aexit__` |

#### Connection params: kwargs OR environment

The adapter resolves `api_key`, `base_url`, and `dataspace` from environment variables when the kwargs are omitted (static-token path):

| Kwarg | Env var | Required? |
|---|---|---|
| `api_key` | `CDP_TOKEN` | yes (kwarg or env) — unless `auth=` / inline credentials are used |
| `base_url` | `CDP_BASE_URL` | yes (kwarg or env) — unless `auth=` / inline credentials are used |
| `dataspace` | `CDP_DATASPACE` | no (defaults to `"default"`) |
| `agent_id` | — | yes (no env fallback) |
| `owner_id` | — | not at construction; must be resolvable by the time a turn runs (4.3, 4.4 `stream`) |

Explicit kwargs always win. Missing required values raise `ValueError` at construction time: a missing `agent_id` raises `"agent_id is required; pass agent_id= to LangGraphAdapter(...)"`; other missing values raise `"Self-driving LangGraphAdapter requires: <names>. Either supply these connection parameters or pass session_manager= directly."` Empty strings count as missing.

#### `stream()` vs `run()`

Both drive one turn end-to-end (fetch context → inject → drive graph → ingest records). Pick by what you do with the records:

| Method | Returns | Use when |
|---|---|---|
| `await adapter.run(text_or_inputs, session_id=...)` | `list[Record]` | You don't need per-record observability — you just want the turn to complete. |
| `async for record in adapter.stream(text_or_inputs, session_id=...)` | async iterator of `Record` | You're forwarding records to a UI, server-sent events, audit tap, or want early-termination logic. |

`stream()` and `run()` accept a plain string (wrapped to `{"messages": [("human", ...)]}` automatically) or a full inputs dict. In self-driving mode the adapter spins up one `AsyncSessionManager` per `(session_id, owner)` it sees, so a single adapter instance can drive any number of conversations — and, with the per-call `owner_id`, any number of owners.

The capability tier is fixed across all three paths:

```python
LangGraphAdapter.capabilities  # → Capabilities(working=True, persistent=False)
```

Working-tier turns (the messages flowing through `astream_events` or sitting in `StateSnapshot.values["messages"]`) are captured. Long-term store contents (`BaseStore`) are not — write a separate adapter against `store.search(...)` if you need that.

#### Terminology: LangGraph vs SDK

The adapter straddles two vocabularies. Keep them straight when reading the rest of this section — the LangGraph terms below are framework concepts you pass to the adapter; the SDK terms are what end up on every `Record` and on the wire to Beacon.

| LangGraph framework term | SDK / Beacon term | How the adapter maps them |
|--------------------------|-------------------|---------------------------|
| `thread_id` (the value under `config["configurable"]["thread_id"]`) | `session_id` on every emitted `Record`; `session_id` on `SessionManager` / `AsyncSessionManager` | The `session_id` you pass to `snapshot()` / `stream()` is interpolated into `{"configurable": {"thread_id": session_id}}` before driving the graph. One LangGraph thread = one Beacon session. |
| `StateGraph` / compiled graph | — | Passed via `graph=`. Adapter calls `graph.get_state(...)` (snapshot) and `graph.astream_events(..., version="v3")` (stream). |
| `StateSnapshot` (returned by `graph.get_state(cfg)`) | — | What `pickle_path=` mode loads. The adapter walks `StateSnapshot.values["messages"]`. |
| `BaseMessage` subclasses (`HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`) | `Record.event_type` (`user_message`, `agent_reasoning`, `tool_call`, `tool_result`, `agent_response`) | See the per-type mapping under `snapshot` in 4.4. |
| `BaseStore` (long-term store) | — | **Not** in scope. Adapter declares `Capabilities(working=True, persistent=False)`. |
| LangChain `@tool` runnables and their callbacks | `tool_call` + `tool_result` records (Trace category) | Always captured in streaming mode by the adapter's internal callback handler — no hook required (see 4.7). Plain Python functions called inside a node fire no callbacks and produce no tool records. |
| `astream_events(version="v3")` / `stream.messages` projection | — | Driven by `stream()`; one iteration = one fully-aggregated LLM utterance. |
| `add_messages` reducer | — | Not invoked by the adapter, but the `SystemMessage`s it injects (and the one `make_context_node` emits) carry stable ids (`"agentic-memory-context"`, `"agentic-context-pack-list"`, `"agentic-memory-recall-tool-instruction"`) so the reducer replaces (not duplicates) them on each turn. |

In short: the SDK's `session_id` is the only identifier you provide. The adapter owns the `thread_id` ↔ `session_id` translation; you should never need to construct a `{"configurable": {"thread_id": ...}}` dict yourself for the adapter's own calls.

#### What `stream()` injects into the graph's input messages

Before driving the graph, `stream()` prepends up to three `SystemMessage`s to `inputs["messages"]` (only when `inputs` is a dict). Final order, first to last:

1. `SystemMessage(id="agentic-memory-recall-tool-instruction")` — the built-in guidance on when to call `recall_customer_context`. Self-driving `context_mode="auto"` and `"per-turn"` only; not injected for `"manual"` or in manager-injection mode.
2. `SystemMessage(id="agentic-context-pack-list")` — injected on **every** turn in **every** mode. Content is the context-pack instruction text followed by the catalog block `[Available context packs]\n<json>`. The instruction is only included when the catalog is non-empty. When the adapter has no catalog (manager-injection mode, a failed catalog fetch, or an agent with no packs) the content is just `[Available context packs]\n{}`.
3. `SystemMessage(id="agentic-memory-context")` — the fetched memory block. Self-driving mode only, and only when the policy decides to fetch and the payload renders to a non-empty block.

The user message (the last item of `inputs["messages"]`) is then recorded as a `user_message`.

#### Retrieval — built in (all-in-one) or wired by you (manager-injection)

In the **all-in-one path**, retrieval is handled inside `stream()` automatically. Before each turn the adapter consults the policy behind `context_mode` (4.10), and on a fetch decision calls `POST /api/v1/agentic/context`, renders the payload via `context_to_system_message`, and prepends the result as a `SystemMessage` with the stable id `"agentic-memory-context"`. The LangGraph `add_messages` reducer keeps that `SystemMessage` in state across subsequent turns (when your graph has a checkpointer — see 4.10), so cross-session memory survives without any explicit re-injection. A failed fetch (network outage, 4xx, 5xx) is logged and swallowed — the graph still runs with no injected block, so a Beacon outage never breaks the user-facing turn.

In the **manager-injection paths** (`graph=` or `pickle_path=` + your own `session_manager=`), the adapter does **not** fetch memory context, and `default_tools()` returns `[]`. You wire retrieval yourself, two ways:

| Approach | Use when |
|----------|----------|
| `make_context_node(client, agent_id=...)` from `agentic_context_langgraph` (see 4.9) | You want context injected into state automatically before each LLM turn. The factory returns an async LangGraph node you can `add_node(...)` directly onto your `StateGraph`. |
| Call `AsyncRetrievalClient.get_context(...)` (or the sync `RetrievalClient`) yourself, from `agentic_context` | You want manual control — fetch once at session start, cache it, format the prompt yourself. |

All retrieval routes (all-in-one auto-fetch, `make_context_node`, manual `AsyncRetrievalClient.get_context`) hit the same Beacon endpoint (`POST /api/v1/agentic/context`) and take the same `owner_id` / `agent_id` / `conversation_id` (= the SDK `session_id`) / `owner_type` parameters.

### 4.3 Constructor

```python
from agentic_context_langgraph import LangGraphAdapter

# Path 1 (recommended) — all-in-one. Adapter owns its clients.
adapter = LangGraphAdapter(
    graph=my_compiled_graph,               # or a builder: see "graph= as a builder" below
    agent_id="my-agent",
    owner_id="user@example.com",           # optional here; may be passed per call instead
    api_key="...",
    base_url="https://beacon.example.com",
    dataspace="default",
    context_mode="auto",                   # "auto" (default) | "per-turn" | "manual"
)

# Path 2 — manager-injection (you own the clients, no auto-fetch).
from agentic_context import AsyncSessionManager
adapter = LangGraphAdapter(
    agent_id="my-agent",
    session_manager=mgr,
    graph=my_graph,
)

# Path 3 — pickle handoff. Always pairs with session_manager=.
adapter = LangGraphAdapter(
    agent_id="my-agent",
    session_manager=mgr,
    pickle_path="ses-123.pkl",
    allow_pickle=True,
)
```

| Parameter | Required | Default | Description |
|-----------|:--------:|---------|-------------|
| `agent_id` | yes¹ | `None` | Must match a registered agent config in Beacon. Stamped on every emitted `Record`. ¹Only exception: on the `ApiKeyAuth` demo path (see `api_key`) it may be omitted and is taken from the secret bundle's `agentId` when `async with adapter:` opens. Raises `ValueError` if absent everywhere. |
| `graph` | one of | `None` | A compiled LangGraph graph (`StateGraph(...).compile()`), **or** a builder `(sdk_tools) -> compiled graph` — see "`graph=` as a builder" below. Required for `stream()` and for the all-in-one path. Mutually exclusive with `pickle_path`. |
| `pickle_path` | one of | `None` | Path (`str` or `pathlib.Path`) to a pickled `StateSnapshot`. Mutually exclusive with `graph`. Requires `session_manager=` (without it, `ValueError`). |
| `session_manager` | path 2/3 | `None` | A `SessionManager` (sync) or `AsyncSessionManager` (async) from `agentic_context`. Pass this OR connection params (`owner_id` / `api_key` / `base_url` / `dataspace`), not both — passing both raises `ValueError`. In this mode `auth`, `private_key`, `consumer_key`, `subject`, `login_url`, `secret_server_url` and `owner_type` are accepted but unused — authenticate the client/manager you pass in. |
| `retrieval_client` | no | `None` | **Manager-injection only.** An already-open `AsyncRetrievalClient` the adapter uses solely to fetch the server-published context-pack instruction (`get_context_pack_instruction`). You open and close it. Passing it without `session_manager=` raises `ValueError`. |
| `owner_id` | no | `None` | End-user identity (email, user ID, etc.). Not required at construction: a multi-tenant server can build one adapter and pass a per-request `owner_id` to `stream()`/`run()` instead. It only needs to be resolvable by the time a turn runs (per-call → constructor → `ValueError`). Forbidden when `session_manager=` is set (the owner then comes from the manager's `owner_id` property). |
| `api_key` | path 1² | `None` | All-in-one mode: CDP Bearer token; falls back to env `CDP_TOKEN`. The adapter constructs an `AsyncIngestionClient` and `AsyncRetrievalClient` from this. Forbidden when `session_manager=` is set. ²Not needed when `auth=` or inline credentials are given. **Demo path:** if you pass only `api_key` — no `base_url`, no `CDP_BASE_URL` env var, no `auth=`, no inline credentials — the adapter treats it as a portal key and builds an `ApiKeyAuth(api_key, secret_server_url=...)`, resolving every credential from the secret server. |
| `base_url` | path 1² | `None` | All-in-one mode: Beacon base URL (must start with `https://` or `http://`); falls back to env `CDP_BASE_URL`. Forbidden when `session_manager=` is set. |
| `dataspace` | no | `"default"` | All-in-one mode: Beacon dataspace; falls back to env `CDP_DATASPACE`, then `"default"`. Forbidden when `session_manager=` is set. |
| `auth` | no | `None` | A `ContextAuth` or `ApiKeyAuth` instance (both from `agentic_context`). The recommended production all-in-one path: the adapter runs the JWT → CDP token exchange when `async with adapter:` opens, discovers `base_url`, and refreshes tokens automatically. Pass this instead of `api_key`/`base_url`. |
| `private_key` / `consumer_key` / `subject` / `login_url` | no | `""` | Inline `ContextAuth` credentials — the adapter builds the `ContextAuth` for you if you pass these instead of an `auth=` instance. **All-or-nothing:** pass all four or none; a partial set raises `ValueError`. Ignored when `auth=` is passed. |
| `secret_server_url` | no | `None` | Secret-server host for the `ApiKeyAuth` demo path; else read from `AGENTIC_CONTEXT_SECRET_SERVER_URL`. There is no built-in default host — if neither is set, the first token fetch fails. |
| `owner_type` | no | resolved from `auth` | Identity kind for `owner_id`. If omitted, uses the auth object's `default_owner_type` (`"individual"` for `ContextAuth` / `ApiKeyAuth`), or `"individual"` when no `auth` is used. Sent on context reads (`get_context`, `get_context_pack`) **and** to every per-session manager the adapter creates (so ingestion and retrieval agree). |
| `context_mode` | no | `"auto"` | The single knob that picks a memory behavior in all-in-one (self-driving) mode. `"auto"` seeds fast context once per session, then the agent can call the recall tool; `"per-turn"` seeds fast context every turn; `"manual"` never auto-seeds (tools present, no recall guidance injected). Any other value raises `ValueError` — in **every** mode, including manager-injection, where the value is validated but otherwise inert. For fetch cadences these three don't cover (time-decay, topic-shift, a custom `ContextPolicy`), drive retrieval yourself with `make_context_node`. See 4.10. |
| `ingestion_callbacks` | no | `None` | An `IngestionCallbacks` instance for per-batch ingestion observability. Self-driving mode only: it is forwarded to each per-session manager the adapter builds. In manager-injection mode wire callbacks on your own manager. |
| `agent_participant_variable` | no | `None` | **Full override** of Beacon's `AiAgentParticipant` block. Default: `{"AiAgentType": "LangGraph", "AiAgentVersionApiName": "v1", "AiAgentTemplateApiName": "default", "AiAgentApiName": <agent_id>}`. Self-driving mode only — combining it with `session_manager=` raises `ValueError` (set it on your own manager instead). |
| `context_header` | no | `"Previous context about this customer"` | Header line on the auto-injected SystemMessage block. |
| `context_message_id` | no | `"agentic-memory-context"` | Stable id on the injected `SystemMessage`. The `add_messages` reducer uses this id to replace (not duplicate) the block on subsequent injections. Pass `None` to opt out and get fresh ids. |
| `context_dedupe` | no | `False` | Drop repeated `(text, type)` facts in the rendered block. |
| `context_rank_urgent` | no | `False` | Surface facts with `type` in `("open_case", "sla_breach", "churn_risk", "escalation")` ahead of others. |
| `allow_pickle` | no | `False` | Must be set to `True` to actually load a `pickle_path` (checked when `snapshot()` runs, not at construction). Pickle deserialization executes arbitrary code; the gate forces a deliberate opt-in. See 4.8. |
| `user_participant_id` | no | the turn's owner | `participant_id` written to `user_message` records. Resolution: this kwarg if set → the turn's resolved owner (per-call `owner_id`, else constructor `owner_id`, else the injected manager's `owner_id`) → the literal `"user"` if no owner resolves. Defaults to the owner because Beacon marks an entry `role="owner"` only when `participantId == ownerId`. Override only if your Beacon setup registers a different participant. |
| `reasoning_classifier` | no | `None` | Callable `(AIMessage) -> "reasoning" \| "reply"` controlling how snapshot mode classifies an `AIMessage` that has both content and `tool_calls`. Default heuristic: any such message is reasoning. |
| `on_session_start` | no | `None` | Hook `(session_id: str) -> None`. Fires on **every** `stream()` call, before the graph runs. |
| `on_session_end` | no | `None` | Hook `(session_id: str \| None) -> None`. Fires from inside `close()` after the manager is closed, and from `end_session(session_id)`. |
| `on_turn_start` | no | `None` | Hook `(TurnEvent) -> None`. Fires once per `stream()` call before the graph runs. |
| `on_tool_start` | no | `None` | Hook `(ToolEvent) -> None`. Fires when a LangChain tool runnable starts. Observation only — tool records are emitted whether or not it is set (see 4.7). |
| `on_tool_end` | no | `None` | Hook `(ToolEvent) -> None`. Fires when a tool ends successfully. |
| `on_error` | no | `None` | Hook `(ErrorEvent) -> None`. Fires when the stream loop or a tool callback raises (`phase` `"stream"` / `"tool"`), or on an ingest problem the adapter reports instead of raising (`phase="ingest"`): a cut-short turn's ingest failed, or records were produced after `close()` / `end_session()` already sent the turn. |

You must pass **exactly one** of `graph=` or `pickle_path=`. Passing both, or neither, raises `ValueError`. Hooks are observation-only — exceptions inside a hook are caught and logged; the stream loop never aborts because of a bad hook.

#### `graph=` as a builder

`graph=` accepts either a compiled graph or a **builder function** `(sdk_tools) -> compiled graph`. The adapter calls the builder **once**, at the very end of `__init__`, passing `adapter.default_tools()` (a list of two LangChain tools in self-driving mode, `[]` in manager-injection mode). Use it to bind the SDK tools into your model and tool node before compiling:

```python
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

def build_graph(sdk_tools):
    tools = [*sdk_tools, *my_tools]
    model_with_tools = my_model.bind_tools(tools)
    builder = StateGraph(MyState)
    # ... add your agent node (calling model_with_tools) and ToolNode(tools) ...
    return builder.compile(checkpointer=my_checkpointer)

async with LangGraphAdapter(
    graph=build_graph, agent_id="support-bot", owner_id="sarah@corp.com",
) as adapter:
    await adapter.run("Hey, I'm back", session_id="ses-2")
```

Detection rule: a `graph=` value is treated as a builder when it is callable and has **neither** `astream_events` nor `get_state` attributes (a compiled graph has both). A builder must be a plain synchronous function and must return a compiled graph; otherwise the constructor raises `TypeError`:

| Mistake | Error |
|---|---|
| `async def` builder | `TypeError: graph= builder must be a plain (sync) function; it returned a coroutine. Remove `async` from the builder.` |
| Builder returns something without both `astream_events` and `get_state` (e.g. an uncompiled `StateGraph`) | `TypeError: graph= builder must return a compiled graph (call .compile() on your StateGraph); got <type name>` |

Do not add a second `make_get_context_pack_tool(...)` instance next to `default_tools()` — tool names must be unique.

#### Per-call `owner_id` (multi-tenant servers)

In self-driving mode the owner for a turn is resolved as: `owner_id=` passed to `stream()` / `run()` → constructor `owner_id` → `ValueError("owner_id is required but was not provided: pass owner_id= to stream()/run() for this turn, or set it once on LangGraphAdapter(owner_id=...).")`. Empty strings count as "not supplied". The adapter keeps a separate manager and separate policy/cache state per `(session_id, owner)`, so two tenants who reuse the same session string never share state, and the tools read the owner from the running turn's own config.

In manager-injection mode the owner lives on the injected manager. Passing `owner_id=` to `stream()` / `run()` there raises `ValueError("owner_id= is only valid in self-driving mode; ...")`.

#### Async context manager

The all-in-one path supports `async with adapter:` — `__aenter__` opens the owned ingest and retrieval clients (and, when `auth=` / inline credentials / the demo path are used, performs the first token exchange); `__aexit__` closes every per-session manager the adapter created, then closes the clients. Before closing managers it ingests any unfinished `stream()` turn (one the caller abandoned without `aclose()`). A turn still running in another task is cut off at that point: records it produces afterwards are not ingested and are reported to `on_error(phase="ingest")`. In self-driving mode `stream()` / `run()` raise `RuntimeError` if called before `async with adapter:` has been entered. Calling `async with adapter:` on a manager-injection adapter raises `RuntimeError` because the adapter does not own those clients.

---

### 4.4 Method Reference

---

#### `snapshot`

Returns all records for a session in chronological order. Available in both `graph=` and `pickle_path=` modes. Pure read — does **not** ingest; pass the result to `adapter.ingest(...)` (or directly to `mgr.ingest(...)`).

**Signature:**

```python
def snapshot(session_id: str) -> list[Record]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Used as the LangGraph `thread_id` (`{"configurable": {"thread_id": session_id}}`) and as the `session_id` on every emitted `Record`. |

**Returns:** `list[Record]` covering every message in the snapshot, sharing `interaction_id = f"turn-{session_id}-snapshot"`.

**Behavior:**

- `graph=` mode reads `graph.get_state({"configurable": {"thread_id": session_id}}).values["messages"]`. The graph needs a checkpointer for `get_state` to return history.
- `pickle_path=` mode opens the file and `pickle.load`s it. Requires `allow_pickle=True`; without it `snapshot()` raises `ValueError`.
- Branches per message type:
    - `HumanMessage` → `user_message` (Message)
    - `AIMessage` with content + `tool_calls`, `reasoning_classifier` returns `"reasoning"` (default) → `agent_reasoning` (Trace) **plus** one `tool_call` (Trace) per call
    - `AIMessage` with content + `tool_calls`, `reasoning_classifier` returns `"reply"` → `agent_response` (Message) **plus** one `tool_call` (Trace) per call
    - `AIMessage` with `tool_calls` only → one `tool_call` per entry
    - `AIMessage` with content only → `agent_response` (Message)
    - `AIMessage` with neither content nor `tool_calls` → no record
    - `ToolMessage` → `tool_result` (Trace)
    - Other `BaseMessage` (e.g. `SystemMessage`) → `agent_reasoning` (Trace fallback)
- `tool_call` `event_id`s are deterministic: `step-tool-call-{session_id}-{idx}-{tc_idx}-{tool_name}`. The framework `tool_call.id` (when present) is preserved in `metadata.tool_call_id` for cross-record correlation but is **not** used as the `event_id`, because LangGraph populates it lazily.
- Re-snapshotting the same session writes its entries again — `event_id`s are stable locally but not sent, so nothing is deduped.

**Usage (sync manager):**

```python
from agentic_context import IngestionClient, SessionManager
from agentic_context_langgraph import LangGraphAdapter

with IngestionClient(base_url="https://beacon.example.com", api_key="...") as client:
    mgr = SessionManager(client, agent_id="support-bot", owner_id="user@example.com")
    adapter = LangGraphAdapter(
        agent_id="support-bot",
        session_manager=mgr,
        graph=my_graph,
    )
    records = adapter.snapshot("ses-123")
    adapter.ingest(records)
    adapter.close()
```

---

#### `stream`

Drives the graph and yields one or more `Record`s per turn. **Async only.** Available only in `graph=` mode — calling on a `pickle_path=` adapter raises `ValueError`. In self-driving mode it must be called inside `async with adapter:` (otherwise `RuntimeError`).

**Signature:**

```python
async def stream(inputs: Any, session_id: str, *, owner_id: str | None = None) -> AsyncIterator[Record]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `Any` | A plain `str` (wrapped to `{"messages": [("human", text)]}`) or the input dict you would normally pass to `graph.astream_events(...)`. The adapter inspects `inputs["messages"][-1]` to extract the user message. Recognized shapes: `HumanMessage`, `("human", text)` tuples, and dicts of the form `{"role": "user"\|"human", "content": ...}` **or** `{"type": "user"\|"human", "content": ...}` (LangChain's standard alias for `role`). |
| `session_id` | `str` | Used as the `thread_id` and stamped on emitted records. |
| `owner_id` | `str \| None` | Keyword-only per-turn owner override for multi-tenant servers. **Self-driving mode only.** Resolution order: per-call `owner_id` → constructor `owner_id` → `ValueError` if neither is set. Passing it in manager-injection mode raises `ValueError`. |

**Yields:** `Record`s in the order they are produced. The adapter ingests every yielded record itself — the whole turn in one batch when the turn ends — so callers can `async for record in adapter.stream(...): pass` without doing their own ingest. **Do not call `adapter.ingest(...)` on records yielded by `stream()`** — they are already ingested, and re-ingesting writes duplicate entries (`event_id` is not sent, so nothing is deduped).

**Behavior:**

- Generates a fresh per-turn `interaction_id = f"turn-{session_id}-{turn_idx}"` (`turn_idx` increments per `session_id` across calls on the same adapter instance).
- In self-driving mode, consults the context policy and may fetch memory; then (in every mode) prepends the SystemMessages described under "What `stream()` injects" in 4.2. The context-pack catalog is (re)fetched on the first `stream()` call for each `session_id` on this adapter instance and reused for later turns.
- Captures the running event loop so tool callbacks (which LangGraph may invoke on a worker thread) can schedule ingest tasks back onto it via `run_coroutine_threadsafe`.
- Yields a `user_message` Record first if `inputs` carries a recognizable user message; otherwise skips that step. Its `participant_id` follows the `user_participant_id` resolution in 4.3.
- Drives `await graph.astream_events(inputs, cfg, version="v3")` and iterates `stream.messages`. Each iteration corresponds to one fully-aggregated LLM utterance. Utterances with empty text are skipped. The adapter classifies the awaited output:
    - Pending tool calls → `agent_reasoning` (Trace)
    - No pending tool calls → `agent_response` (Message), with metadata `{node, event: "utterance_complete", usage}` when available
- A single `stream()` call can yield multiple records — typically one `user_message`, any `tool_call` / `tool_result` records, and one or more reasoning/response records, in chronological order.
- Records are yielded in chronological order; a tool's records are yielded when the next model call starts streaming (or when the turn ends). The turn is ingested as **one batch — one `/entries` POST per turn** — when the turn ends. If the graph raises, the caller stops iterating early, or `close()` / `end_session()` runs while a turn is unfinished, the records captured so far are still ingested in that one batch. A process crash mid-turn loses the whole turn, user message included.
- If you `break` out of the loop, wrap the stream in `contextlib.aclosing(adapter.stream(...))` so the partial turn is ingested right away rather than at `close()`.
- If the underlying graph raises, the adapter fires `on_error(phase="stream")` and re-raises.
- Tool records (`tool_call`, `tool_result`) are captured by an internal callback handler that the adapter always registers on `cfg["callbacks"]`; no hook is required. See 4.7.
- The adapter constructs the LangGraph `RunnableConfig` itself (`{"configurable": {"thread_id": session_id, ...}, "callbacks": [...]}`; in self-driving mode `configurable` also carries a private `agentic_owner_id` key the SDK tools read) and **does not merge** any callbacks the caller may want. If you have your own LangChain tracer or callback handler, attach it at graph compile time (`graph.with_config(config={"callbacks": [...]})`), not via the `inputs` argument to `stream()`.

**Usage:**

```python
import asyncio
from agentic_context import AsyncIngestionClient, AsyncSessionManager
from agentic_context_langgraph import LangGraphAdapter

async def run_turn():
    async with AsyncIngestionClient(base_url="https://beacon.example.com", api_key="...") as client:
        mgr = AsyncSessionManager(
            client,
            agent_id="support-bot",
            owner_id="user@example.com",
            session_id="ses-123",
        )
        adapter = LangGraphAdapter(
            agent_id="support-bot",
            session_manager=mgr,
            graph=my_graph,
        )
        async for record in adapter.stream(
            {"messages": [("human", "what's my refund status?")]},
            session_id="ses-123",
        ):
            pass  # already ingested
        await adapter.close()

asyncio.run(run_turn())
```

> **Caveat — v3 is upstream beta.** The adapter pins `langgraph>=0.2` and uses `astream_events(version="v3")`. The protocol is stable enough for production but is marked `LangChainBetaWarning` upstream. Adapter patch releases will track upstream changes.

---

#### `run`

Drains `stream()` internally and returns the collected records. Use when you don't need per-record observability — server handlers, batch drivers, anywhere you just want one turn to complete and one POST per turn to land. **Async only.**

**Signature:**

```python
async def run(inputs: Any, *, session_id: str, owner_id: str | None = None) -> list[Record]
```

**Parameters:** identical to `stream()` — `inputs` accepts the same shapes (string or dict); `session_id` is keyword-only on `run()` to keep the call site readable, and `owner_id` is the same keyword-only per-turn owner override (self-driving mode only; per-call → constructor → `ValueError`).

**Returns:** `list[Record]` in the order they were emitted. Records are already in flight to Beacon by the time `run()` returns; the list is for inspection (e.g., audit metadata, response shaping) and **must not** be re-ingested.

**Behavior:** identical to `stream()` — same context fetch, same graph drive, same per-turn ingest. The only difference is that `run()` awaits the entire turn before returning instead of yielding records as they emit.

**Usage:**

```python
import asyncio
from agentic_context_langgraph import LangGraphAdapter

async def main():
    async with LangGraphAdapter(graph=my_graph, agent_id="bot", owner_id="x@y.com") as adapter:
        await adapter.run("Hey, I'm back", session_id="ses-2")

asyncio.run(main())
```

That is the entire integration in the all-in-one path. `api_key`, `base_url`, and `dataspace` come from `CDP_TOKEN` / `CDP_BASE_URL` / `CDP_DATASPACE` if not passed explicitly.

Multi-tenant variant — one adapter, owner supplied per request:

```python
async with LangGraphAdapter(graph=my_graph, agent_id="bot") as adapter:   # no owner_id here
    await adapter.run("Hi", session_id="ses-1", owner_id="alice@corp.com")
    await adapter.run("Hi", session_id="ses-9", owner_id="bob@corp.com")
```

---

#### `ingest`

Direct ingest of a `list[Record]` through the bound manager. Bound at construction to either a sync or async implementation depending on the manager type (always async in self-driving mode).

**Signature:**

```python
# When session_manager is a SessionManager
def ingest(records: list[Record]) -> None

# When session_manager is an AsyncSessionManager, or in self-driving mode
async def ingest(records: list[Record], owner: str | None = None) -> None   # owner: self-driving mode only; defaults to the owner last used for the session, else the constructor owner_id
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `records` | `list[Record]` | Records in chronological order. Empty list is a no-op. |

**Returns:** `None`

**Behavior:**

- Sync manager: blocks until the HTTP POST returns (with retry on transient failures).
- Async manager: schedules a background delivery task and returns; `close()` is the synchronization point.
- Self-driving mode: the batch is routed to the per-session manager for `records[0].session_id`, under the owner last used for that session (else the constructor `owner_id`).
- The adapter does no buffering — every non-empty call is a POST (sync) or a scheduled POST (async).

**Usage (sync):**

```python
records = adapter.snapshot("ses-123")
adapter.ingest(records)
```

**Usage (async):**

```python
records = adapter.snapshot("ses-123")
await adapter.ingest(records)
```

---

#### `close`

Closes the underlying session and releases resources. **Must be called on shutdown** — without it, async background deliveries may not finish before the process exits. `close()` is local cleanup only; no `close_session` is ever POSTed to Beacon (sessions are never explicitly closed — returning to the same `session_id` just works). In self-driving mode `async with adapter:` calls it for you.

**Signature:**

```python
# Sync (SessionManager)
def close() -> None

# Async (AsyncSessionManager, or self-driving)
async def close() -> None
```

**Behavior:**

- Async path: ingests any unfinished `stream()` turn, then closes the manager — in self-driving mode every per-session manager the adapter created, otherwise `await mgr.close()`. Before closing managers it ingests any unfinished `stream()` turn (one the caller abandoned without `aclose()`). A turn still running in another task is cut off at that point: records it produces afterwards are not ingested and are reported to `on_error(phase="ingest")`. Each manager drains its own delivery queue with a 30-second budget; none POSTs a `close_session`. It also clears the pinned context-pack instructions. In self-driving mode, `close()` does not close the owned HTTP clients — `__aexit__` does that.
- Sync path: calls `mgr.close()` directly.
- Always fires `on_session_end(last_session_id)` from a `finally` block — observability is delivered even when the manager close raises. The argument is `None` if `close()` runs before any `snapshot()` / `stream()` recorded a session.
- The underlying manager close is idempotent (`SessionManager.close()` / `AsyncSessionManager.close()` are both safe to call multiple times), but `adapter.close()` itself is **not** guarded against re-entry: calling it twice will fire `on_session_end` twice. Call it once in your shutdown path.

**Usage (sync):**

```python
adapter.close()
```

**Usage (async):**

```python
await adapter.close()
```

---

#### `end_session`

Closes and forgets the per-session manager(s) for one `session_id`. Long-running adapters (web servers, schedulers) that drive many sessions through one adapter instance should call this when a session ends so the manager pool doesn't grow unbounded. Drops every per-session `AsyncSessionManager` for that `session_id` (across owners), the cached fetch state, the turn counters, and the pinned context-pack instruction, so a later `stream(..., session_id)` starts fresh. Fires `on_session_end(session_id)`. No-op for sessions the adapter never saw (the hook still fires). **Self-driving (all-in-one) mode only** — raises `RuntimeError` on manager-injection paths, which don't own the manager. **Async only.**

**Signature:**

```python
async def end_session(session_id: str) -> None
```

---

#### Context-pack and memory-tool methods

These are public instance methods on `LangGraphAdapter` (not module-level exports). See 4.9 for how they fit together.

| Method | Signature | Returns / behavior |
|--------|-----------|--------------------|
| `default_tools()` | `() -> list[BaseTool]` | Self-driving mode: `[recall_customer_context, get_context_pack]` in **every** `context_mode`. Manager-injection mode: `[]`. |
| `make_context_retrieval_tool()` | `() -> StructuredTool` | The `recall_customer_context` tool (4.9). |
| `make_get_context_pack_tool(valid_pack_ids, *, param_defaults=None, timeout=CONTEXT_PACK_TIMEOUT_S)` | `(list[str], ...) -> StructuredTool` | The `get_context_pack` tool (4.9). |
| `await get_context_pack_instruction(session_id)` | `(str) -> str` | The context-pack instruction pinned for that session (4.9). |
| `await get_context_pack_list()` | `() -> dict \| None` | Fetches this agent's available context-pack catalog (`GET /api/v1/agentic/runtime-context-packs`). Returns `None` when the adapter has no owned retrieval client (manager-injection mode, or before `async with adapter:` under `auth=`) or the call fails (logged, never raised). |

---

#### Attributes and properties

```python
adapter.agent_id           # → str — the agent id passed at construction (or resolved on the ApiKeyAuth demo path)
adapter.capabilities       # → Capabilities(working=True, persistent=False)
adapter.mode               # → Literal["stream", "snapshot"]
adapter.context_mode       # → Literal["auto", "per-turn", "manual"]
adapter.context_policy     # → ContextPolicy — the resolved policy behind context_mode
adapter.is_self_driving    # → bool — True on the all-in-one path (adapter owns retrieval)
```

`adapter.mode` reflects the construction path: `"stream"` when the adapter holds a graph (`graph=`), `"snapshot"` when constructed with `pickle_path=`. Streaming-mode adapters can also serve `snapshot()` reads from the live graph; the label reflects construction, not capability. Use `adapter.mode == "stream"` as a guard before calling `stream()` if your code accepts either construction path from upstream. `context_mode`, `context_policy`, and `is_self_driving` are read-only introspection of the resolved memory behavior. In manager-injection mode `context_mode` reports the value you passed (default `"auto"`) but it is inert, and `context_policy` returns a default `SessionFirstPolicy` that is never consulted.

---

### 4.5 End-to-End Example: Snapshot

Snapshot is the simplest, most reliable model: let the agent run to completion, then ship the full session state in one batch. Use the sync stack — it removes a layer of concurrency you do not need.

```python
import os
from agentic_context import IngestionClient, SessionManager
from agentic_context_langgraph import LangGraphAdapter
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- Config ---
BEACON_URL = os.environ["CDP_BASE_URL"]
BEACON_TOKEN = os.environ["CDP_TOKEN"]
DATASPACE = os.environ.get("CDP_DATASPACE", "default")
AGENT_ID = "support-bot"
OWNER_ID = "user@example.com"
SESSION_ID = "ses-2026-06-18-001"

def build_graph() -> StateGraph:
    # ... your graph construction (returns the uncompiled StateGraph; it is compiled
    # below with a checkpointer so graph.get_state(cfg) returns the full message history).
    ...

def main() -> None:
    graph = build_graph().compile(checkpointer=MemorySaver())

    # 1. Run the agent to completion under thread_id == SESSION_ID.
    cfg = {"configurable": {"thread_id": SESSION_ID}}
    graph.invoke({"messages": [("human", "what's my refund status?")]}, cfg)

    # 2. Open transport + lifecycle (sync stack).
    with IngestionClient(base_url=BEACON_URL, api_key=BEACON_TOKEN, dataspace=DATASPACE) as client:
        mgr = SessionManager(
            client,
            agent_id=AGENT_ID,
            owner_id=OWNER_ID,
            session_id=SESSION_ID,
        )
        adapter = LangGraphAdapter(
            agent_id=AGENT_ID,
            session_manager=mgr,
            graph=graph,
        )

        try:
            records = adapter.snapshot(SESSION_ID)
            if records:
                adapter.ingest(records)
        finally:
            adapter.close()

if __name__ == "__main__":
    main()
```

`build_graph()` is a placeholder for your own graph factory — the example runs once you supply it. This makes four HTTP calls regardless of turn count: `create_conversation`, `create_session`, `add_participants`, one `add_entries`. No `close_session` is ever posted — `close()` is local cleanup only. Re-running the script writes duplicate entries — `event_id` is not sent, so nothing is deduped.

---

### 4.6 End-to-End Example: Streaming

Streaming is for live agents where you want each turn ingested as it happens. Use the async stack so the agent loop is never blocked by Beacon latency.

```python
import asyncio
import os
from agentic_context import AsyncIngestionClient, AsyncSessionManager
from agentic_context_langgraph import (
    ErrorEvent,
    LangGraphAdapter,
    ToolEvent,
    TurnEvent,
)
from langgraph.checkpoint.memory import MemorySaver

# --- Config ---
BEACON_URL = os.environ["CDP_BASE_URL"]
BEACON_TOKEN = os.environ["CDP_TOKEN"]
DATASPACE = os.environ.get("CDP_DATASPACE", "default")
AGENT_ID = "support-bot"
OWNER_ID = "user@example.com"
SESSION_ID = "ses-2026-06-18-002"

def log_turn(event: TurnEvent) -> None:
    print(f"turn start: {event.session_id} @ {event.turn_started_at}")

def log_tool_start(event: ToolEvent) -> None:
    print(f"tool start: {event.tool_name}({event.args})")

def log_tool_end(event: ToolEvent) -> None:
    duration = (event.ended_at - event.started_at).total_seconds()
    print(f"tool end: {event.tool_name} -> {event.output!r} ({duration:.3f}s)")

def log_error(event: ErrorEvent) -> None:
    print(f"error in {event.phase} for {event.session_id}: {event.exception!r}")

async def main() -> None:
    # build_graph(): your own factory returning an uncompiled StateGraph — see 4.5.
    graph = build_graph().compile(checkpointer=MemorySaver())

    async with AsyncIngestionClient(
        base_url=BEACON_URL, api_key=BEACON_TOKEN, dataspace=DATASPACE,
    ) as client:
        mgr = AsyncSessionManager(
            client,
            agent_id=AGENT_ID,
            owner_id=OWNER_ID,
            session_id=SESSION_ID,
        )
        adapter = LangGraphAdapter(
            agent_id=AGENT_ID,
            session_manager=mgr,
            graph=graph,
            # Wiring any tool/error hook is what enables tool record emission.
            on_turn_start=log_turn,
            on_tool_start=log_tool_start,
            on_tool_end=log_tool_end,
            on_error=log_error,
        )

        try:
            # Each call = one turn. Multi-turn sessions call stream() repeatedly.
            async for record in adapter.stream(
                {"messages": [("human", "what's my refund status?")]},
                session_id=SESSION_ID,
            ):
                # Records are already ingested before they're yielded.
                # Use this loop to forward the assistant's text to your client UI.
                if record.event_type == "agent_response":
                    print(f">> {record.content}")
        finally:
            await adapter.close()

if __name__ == "__main__":
    asyncio.run(main())
```

`build_graph()` is again a placeholder for your own factory. This example uses the manager-injection path, so the adapter does no memory auto-fetch here; for the all-in-one streaming form (adapter-owned clients and automatic context), see the `async with LangGraphAdapter(...)` examples in 4.2 and 4.4.

**Non-blocking guarantee.** Inside `stream()`, the one per-turn `await mgr.ingest(turn_records)` returns in well under a millisecond — `AsyncSessionManager.ingest` schedules an `asyncio.create_task` and returns. The agent loop is never stalled by Beacon latency. `close()` is the synchronization point: it ingests any unfinished turn, then calls `await mgr.close()`, which `asyncio.wait`s pending deliveries with a 30-second budget and cancels stragglers. No `close_session` is posted — close is local cleanup only.

---

### 4.7 Hooks

All hooks are optional keyword arguments to the constructor. Hook exceptions are caught and logged — a bad hook never aborts the stream.

| Hook | Signature | Fires when | Notes |
|------|-----------|------------|-------|
| `on_session_start` | `(session_id: str) -> None` | **Every** `stream(...)` call, before the graph runs. | Streaming only. The adapter does not deduplicate per-session — gate your own callback if you need once-per-session semantics. |
| `on_session_end` | `(session_id: str \| None) -> None` | Inside `close()`, after the manager closes (in a `finally` block); and from `end_session(session_id)`. | From `close()`: the last session id seen by the adapter, or `None` if `close()` is called before any `stream()`/`snapshot()`. From `end_session`: the session id passed in. |
| `on_turn_start` | `(event: TurnEvent) -> None` | Each `stream(...)` call, after `on_session_start`, before the graph runs. | `TurnEvent` carries `session_id`, `inputs` (the inputs dict **after** the adapter has prepended its `SystemMessage`s), and `turn_started_at`. |
| `on_tool_start` | `(event: ToolEvent) -> None` | A LangChain tool runnable starts (callback-driven). | Observation only. `tool_call` / `tool_result` records are emitted whether or not this hook is set. |
| `on_tool_end` | `(event: ToolEvent) -> None` | The same tool ends successfully. | Same `tool_call_id` and `started_at` as the matching `on_tool_start`. |
| `on_error` | `(event: ErrorEvent) -> None` | The graph or a tool callback raises, or a reported ingest problem. | `phase="stream"` for graph-level errors; `phase="tool"` when a tool raises (not for `interrupt()`); `phase="ingest"` when a cut-short turn's ingest fails or records arrive after `close()` / `end_session()` sent the turn. |

> **Important.** Tool records depend on the tool being a real LangChain runnable (e.g., decorated with `@tool`). Plain Python functions called from inside a node do not fire LangChain tool callbacks and therefore produce no tool records.

`TurnEvent`, `ToolEvent`, and `ErrorEvent` are immutable, frozen dataclasses exported from `agentic_context_langgraph`:

```python
@dataclass(frozen=True)
class TurnEvent:
    session_id: str
    inputs: Any
    turn_started_at: datetime

@dataclass(frozen=True)
class ToolEvent:
    session_id: str
    tool_name: str
    tool_call_id: str | None
    args: dict[str, Any] | None
    output: Any | None
    started_at: datetime
    ended_at: datetime | None  # None on the start event

@dataclass(frozen=True)
class ErrorEvent:
    session_id: str
    phase: Literal["stream", "tool"]
    exception: BaseException
    occurred_at: datetime
```

---

### 4.8 Cross-Process Snapshot (Pickle Mode)

Use `pickle_path=` when the agent runs in one process and ingestion runs in another, with a file handoff at session end (e.g., a worker writes to disk; a downstream cron job ships to Beacon).

> **Security warning.** `pickle.load` executes arbitrary Python code from the file. Only load pickles your own pipeline produced and stored under your control. If your input path is hostile or shared with untrusted writers, do **not** use this mode — switch to `graph=` (no deserialization) or wrap the producer/consumer in another transport. The adapter requires you to opt in explicitly via `allow_pickle=True`; calling `snapshot()` without that flag raises `ValueError`.

**Producer (agent process, at session end):**

```python
import pickle

# Inside the agent process, after the session has finished:
state = my_graph.get_state({"configurable": {"thread_id": "ses-123"}})
with open("/var/agent-state/ses-123.pkl", "wb") as f:
    pickle.dump(state, f)
```

**Consumer (SDK process, any time after):**

```python
from agentic_context import IngestionClient, SessionManager
from agentic_context_langgraph import LangGraphAdapter

with IngestionClient(base_url="https://beacon.example.com", api_key="...") as client:
    mgr = SessionManager(
        client,
        agent_id="support-bot",
        owner_id="user@example.com",
        session_id="ses-123",
    )
    adapter = LangGraphAdapter(
        agent_id="support-bot",
        session_manager=mgr,
        pickle_path="/var/agent-state/ses-123.pkl",
        allow_pickle=True,  # explicit opt-in
    )
    try:
        adapter.ingest(adapter.snapshot("ses-123"))
    finally:
        adapter.close()
```

The serialized file must be a `StateSnapshot` — the object returned by `graph.get_state(cfg)`. Anything else fails on load. Calling `adapter.stream(...)` on a pickle-mode adapter raises `ValueError`.

---

### 4.9 Retrieval Helpers and Context Packs

The adapter package exports four retrieval helpers for injecting prior-conversation memory back into the graph, plus the context-pack constants:

| Helper | Purpose |
|--------|---------|
| `make_context_node(client, *, agent_id, ...)` | LangGraph node factory for a **fast** profile read: calls Beacon's `POST /api/v1/agentic/context` (no query) and writes the result into state. |
| `format_context_block(payload, ...)` | Pure formatter: Beacon context payload → prompt-friendly string. |
| `context_to_system_message(payload, ...)` | Pure formatter: Beacon context payload → `SystemMessage` (or `None` if empty). |
| `conversation_to_messages(payload)` | Rehydrate a `GET /api/v1/agentic/conversations/{id}?scope=all` payload back into `list[BaseMessage]` (`HumanMessage` / `AIMessage(tool_calls=...)` / `ToolMessage`). |
| `CONTEXT_PACK_INSTRUCTION` | The bundled context-pack guidance text (re-exported from `agentic_context`). |
| `CONTEXT_PACK_TIMEOUT_S` | Default per-call timeout for a context-pack read, in seconds (`60.0`; re-exported from `agentic_context`). |

The agent-callable tools below are methods on `LangGraphAdapter`, not separate module exports.

#### `default_tools()` — the agent-callable memory and pack tools

In self-driving mode `adapter.default_tools()` returns two LangChain `StructuredTool`s, in every `context_mode`; under manager-injection it returns `[]` (the injected manager owns retrieval, so there is no owned client to back them). They are what a `graph=` builder receives as `sdk_tools` (4.3). This is the same on-demand `recall_customer_context` / `get_context_pack` surface documented for the Claude (§3) and ADK (§5) adapters.

| Tool name | LLM-visible arguments | Behavior |
|-----------|-----------------------|----------|
| `recall_customer_context` | none | Zero-argument **cache read** of the memory the adapter already seeded for this session. Returns `{"userContext": [...]}`. Under `auto` / `per-turn` it issues **no** HTTP call; with no cached memory it returns `{"userContext": []}`. Under `context_mode="manual"` (nothing is ever seeded) a cache miss triggers one `POST /api/v1/agentic/context` for that session and returns it. Never raises. |
| `get_context_pack` | `context_pack_id` (str), `params` (dict[str, str], optional), `query` (str, optional), `context_pack_version` (int, optional) | Pack-scoped read: `POST /api/v1/agentic/context` for one named context pack. Never raises; returns one of the `status` values below. `owner_id` and the session id are taken from the running turn, never from the model. |

Both tools resolve the session from the run's `configurable.thread_id` and the owner from the turn that is running, so concurrent sessions and tenants never read one another's identity.

`get_context_pack` return shapes:

| `status` | Meaning | Extra keys |
|----------|---------|------------|
| `"ok"` | Pack retrieved. | `context_pack_id`, `context` (the pack payload) |
| `"invalid_pack"` | `context_pack_id` is not in the advertised set (checked locally, no HTTP call). | `context_pack_id`, `valid_pack_ids`, `message` |
| `"invalid_version"` | `context_pack_version` was not an integer (bools are rejected). | `context_pack_id`, `message` |
| `"not_ready"` | Could not complete: no owned retrieval client (manager-injection mode, or before `async with adapter:`), no active session, no resolved owner, or the request failed/timed out (`detail` is `"request failed: <ExceptionType>"` in that case). | `context_pack_id`, `message`, `detail` |
| `"version_expired"` | The pack was republished since the catalog was read. The adapter updates its cached version so a retry pins to the new revision. | `context_pack_id`, `requested_version`, `latest_version`, `latest`, `message` |

**Versioning.** When the model omits `context_pack_version` and the cached catalog carries an integer `version` for that pack, the adapter fills it in automatically. Only integer versions are surfaced in the catalog; a `null` or boolean `version` is ignored.

**Which pack ids are valid.** `default_tools()` builds the pack tool with an empty id list, so valid ids come entirely from the server-side catalog the adapter fetches (`GET /api/v1/agentic/runtime-context-packs`) on the first `stream()` call for each `session_id` on an adapter instance; the catalog is cached on the instance. Until a catalog has been fetched (or when it is empty/failed), every id is rejected as `invalid_pack`. If you build the tool yourself with `make_get_context_pack_tool(valid_pack_ids, ...)`, your list is used only while no catalog has been fetched; once a non-empty catalog is cached it takes over.

**Catalog and instruction injection.** Every `stream()` turn prepends `SystemMessage(id="agentic-context-pack-list")` containing the context-pack instruction and the catalog (`[Available context packs]\n<json>`). The instruction is the server-published prompt (`GET /api/v1/agentic/prompts/system/context_pack_instruction`) fetched on a session's first turn and pinned for that session; a miss or failure pins the bundled `CONTEXT_PACK_INSTRUCTION` instead. After a fetch error, new sessions skip the fetch and use the bundled text for 10 seconds. `await adapter.get_context_pack_instruction(session_id)` returns the text pinned for a session (most integrations never need to call it); with no retrieval client it returns the bundled text without pinning.

**Manager-injection.** The pack tool and catalog cannot work in this mode (the adapter owns no retrieval client): `get_context_pack` returns `not_ready` for ids in the `valid_pack_ids` you passed (and `invalid_pack` for any other id), the catalog is always empty, and `default_tools()` is `[]`. If you pass `retrieval_client=` the adapter uses it **only** to fetch the context-pack instruction.

Tool factories (public, for callers who want a tool regardless of mode):

```python
adapter.make_context_retrieval_tool() -> StructuredTool
adapter.make_get_context_pack_tool(
    valid_pack_ids: list[str],
    *,
    param_defaults: dict[str, Any] | None = None,   # merged into every call's params; a model key overrides, a blank model value drops the key
    timeout: float | None = CONTEXT_PACK_TIMEOUT_S, # per-call HTTP timeout, seconds
) -> StructuredTool
```

Don't register a second `get_context_pack` tool alongside `default_tools()` — tool names must be unique.

#### `make_context_node`

```python
def make_context_node(
    client: AsyncRetrievalClient,
    *,
    agent_id: str,
    owner_id_key: str = "customer_id",
    session_id_from_config: bool | None = None,   # normalized to True
    session_id_key: str | None = None,            # normalized to "thread_id"
    into: str = "messages",
    owner_type: str = "individual",
    header: str = "Previous context about this customer",
    message_id: str | None = "agentic-memory-context",
    dedupe: bool = False,
    rank_urgent: bool = False,
    urgent_types: tuple[str, ...] = ("open_case", "sla_breach", "churn_risk", "escalation"),
) -> Callable[[dict, RunnableConfig], Awaitable[dict]]
```

> The identifier kwargs are `session_id_from_config` / `session_id_key` (the core SDK calls this id `session_id`; Beacon's wire format calls it `conversation_id`). The older `conversation_id_from_config` / `conversation_id_key` names are **deprecated aliases** that still work but emit a `DeprecationWarning` and are slated for removal; passing both the old and new name for the same identifier raises `TypeError`.

Returns an async LangGraph node `(state, config) -> dict`. The factory does not take ownership of `client`; you keep it open and close it yourself (typically via `async with AsyncRetrievalClient(...)`). The node takes no policy: it fetches every time it runs.

Default behavior (`into="messages"`) returns `{"messages": [SystemMessage(...)]}` so the standard `add_messages` reducer merges the context into the thread. The `SystemMessage` carries a stable `id="agentic-memory-context"` so re-injecting on every turn replaces the previous copy instead of accumulating duplicates. Pass `into="<state_field>"` to land the formatted block as a plain string in a custom state field instead — useful when your existing system prompt already interpolates a templated context variable.

The node is best-effort: any exception (network, parse, auth) is logged and swallowed. The node returns `{}` (no state change) so a Beacon outage never breaks the user-facing run. It also returns `{}` without fetching when the session id or the owner id cannot be found.

**Wiring example:**

```python
from agentic_context import AsyncRetrievalClient
from agentic_context_langgraph import make_context_node
from langgraph.graph import START, StateGraph

async def build_graph():
    client = AsyncRetrievalClient(
        base_url="https://beacon.example.com",
        api_key="...",
        dataspace="default",
    )

    workflow = StateGraph(MyState)
    workflow.add_node("recall_context", make_context_node(client, agent_id="support-bot"))
    workflow.add_node("llm", my_llm_node)
    workflow.add_edge(START, "recall_context")
    workflow.add_edge("recall_context", "llm")

    return workflow.compile(), client  # caller owns client lifetime
```

By default the node reads the session id from `config["configurable"]["thread_id"]` (where the LangGraph runtime puts it) and the owner id from `state["customer_id"]`. Override `session_id_key`, `owner_id_key`, or `session_id_from_config=False` (to read the id from `state[session_id_key]` instead) if your state schema differs.

#### `format_context_block` and `context_to_system_message`

Pure functions over a Beacon `POST /api/v1/agentic/context` payload. The keys they read are `summary` (rendered under "In-session summary:"), `conversation_summary` (rendered under "Conversation summary:"), and `userContext` (a list of fact entries; each fact's text is taken from the first non-empty of `text` / `content` / `value` / `entry`, plus optional `type` and `confidence`). Use them when you fetch context yourself rather than via `make_context_node`. `format_context_block` returns `""` when nothing is renderable; `context_to_system_message` returns `None` in that case, so you can write `if msg: state.append(msg)`.

Both accept `header`, `dedupe`, `rank_urgent`, and `urgent_types` (`context_to_system_message` also takes `message_id`, default `"agentic-memory-context"`; `None` gives a fresh id). `dedupe=True` drops repeats of `(text.lower().strip(), type)` tuples (Beacon can echo the same fact across working + persistent memory). `rank_urgent=True` surfaces facts whose `type` is in `urgent_types` (default: `open_case`, `sla_breach`, `churn_risk`, `escalation`) ahead of others, preserving original order within each group. Fact text comes from Beacon and originates in user conversations — treat the rendered block as untrusted input when placing it in a system prompt.

#### `conversation_to_messages`

Takes the response of `GET /api/v1/agentic/conversations/{id}?scope=all` (it walks `payload["sessions"][*]["entries"]`) and rebuilds `list[BaseMessage]`. Useful when you want to seed a fresh `StateGraph` thread with prior history rather than rely on a SystemMessage summary. `tool_call` / `tool_result` entries are paired into `AIMessage(tool_calls=...)` and `ToolMessage` instances; `agent_reasoning` entries are folded into the next `AIMessage`'s content as `[reasoning] ...` lines (reasoning with no following assistant message is emitted as a trailing `AIMessage`). Orphan `tool_result` entries (no `tool_call_id`) are dropped — `ToolMessage` requires the id. Unknown entry types are skipped.

---

### 4.10 Context Policies

The all-in-one path consults a `ContextPolicy` before each user turn to decide whether to call `POST /api/v1/agentic/context`. You select the behavior with the **`context_mode=`** constructor knob (`"auto"` | `"per-turn"` | `"manual"`), which the adapter maps to a policy internally:

| `context_mode` | Resolved policy | Memory seeding | Recall-tool guidance `SystemMessage` |
|----------------|-----------------|----------------|--------------------------------------|
| `"auto"` (default) | `SessionFirstPolicy` | Once per `session_id` (first turn only). | Injected (built-in text). |
| `"per-turn"` | `EveryTurnPolicy` | Before every turn. | Injected (built-in text). |
| `"manual"` | `NoFetchPolicy` | Never; the `recall_customer_context` tool falls back to one on-demand fetch on a cache miss. | Not injected — you own the prompting. |

Both tools in `default_tools()` are present in all three modes. The context-pack catalog `SystemMessage` (4.2) is independent of `context_mode` and is injected in all three.

> There is **no `context_policy=` constructor kwarg** — passing one raises `TypeError`. Use `context_mode=` for the three modes above. `.context_policy` exists only as a read-only property exposing the resolved policy. In manager-injection paths `context_mode` is validated but inert — you wire retrieval yourself.

Five `ContextPolicy` classes are exported from `agentic_context_langgraph` (slugs registered with `resolve_policy`). Three of them back `context_mode`. `TimeDecayPolicy` and `TopicShiftPolicy`, and any custom `ContextPolicy`, have **no constructor path** into the adapter; they are plain decision objects you call yourself from your own retrieval node (see below).

| Policy | Slug | Behavior | Use when |
|--------|------|----------|----------|
| `SessionFirstPolicy` (default) | `"session_first"` | Fetch when nothing has been fetched yet for the `(owner, agent_id, session_id)`, never again. The injected `SystemMessage(id="agentic-memory-context")` stays in state across subsequent turns via the `add_messages` reducer. | Backs `context_mode="auto"`. One fetch per session is enough for cross-session memory because the SystemMessage persists in state. |
| `EveryTurnPolicy` | `"every_turn"` | Fetch before every turn. The stable SystemMessage id makes the reducer replace (not duplicate) the prior block. | Backs `context_mode="per-turn"`. Highest fidelity, +1 fetch per turn. |
| `NoFetchPolicy` | `"no_fetch"` | Never fetch. | Backs `context_mode="manual"` — you own prompting/retrieval in your own graph nodes. |
| `TimeDecayPolicy(ttl_seconds=60.0)` | `"time_decay"` | Fetch if nothing was fetched yet, or if at least `ttl_seconds` have elapsed since the last fetch. | Bounded freshness with bounded cost. No adapter facade. |
| `TopicShiftPolicy(embedder=None, threshold=0.7)` | `"topic_shift"` | Fetch if nothing was fetched yet; afterwards embed the latest user text and the cached payload's `summary` and fetch when cosine similarity is below `threshold`. Never fetches again when no `embedder` is configured, or the user text / summary is missing, or the embedder raises. | Conversations that pivot. No adapter facade. |

`resolve_policy(policy)` normalizes a slug string, a policy instance, or `None` (→ `SessionFirstPolicy`) into a policy object. A slug is instantiated with no arguments (so `TopicShiftPolicy` from the slug has no embedder); an unknown slug raises `ValueError` listing the valid slugs; any other object is returned unchanged.

**Process-shared policy state.** Self-driving adapters share `last_fetched_at`, `last_payload`, `last_user_text`, and `turn_idx` through a class-level LRU cache keyed by `(owner_id, agent_id, session_id)` (capped at 1024 entries; least-recently-used evicted first). This is what makes `SessionFirstPolicy` honor its "one fetch per session" contract under the common per-request adapter pattern (a FastAPI handler constructs a fresh `LangGraphAdapter` per request) — without it, every request would see an empty state and always fetch. State is isolated per owner, so multi-tenant deployments don't leak across customers. `end_session(session_id)` drops the entry.

**Custom policies.** Implement the `ContextPolicy` Protocol (keyword-only arguments):

```python
from agentic_context_langgraph import ContextPolicy

class MyPolicy:
    def should_fetch(
        self,
        *,
        session_id: str,
        turn_idx: int,
        last_user_text: str | None,
        last_payload: dict | None,
        last_fetched_at,         # datetime | None
    ) -> bool:
        ...
```

`LangGraphAdapter` has no way to accept a custom policy, and `make_context_node` does not take one either. A custom (or `time_decay` / `topic_shift`) policy is a decision function: call `policy.should_fetch(...)` yourself in a node of your own and call `make_context_node`'s node (or `AsyncRetrievalClient.get_context`) only when it returns `True`.

For the two facade cadences beyond the default, use `context_mode`:

```python
# Re-seed context every turn.
adapter = LangGraphAdapter(graph=g, agent_id=..., owner_id=..., api_key=..., base_url=..., context_mode="per-turn")
```

For `time_decay` / `topic_shift`, resolve the policy object with `resolve_policy` or instantiate it directly:

```python
from agentic_context_langgraph import resolve_policy, TimeDecayPolicy
policy = resolve_policy("time_decay")        # or TimeDecayPolicy(ttl_seconds=120)
# call policy.should_fetch(...) from your own retrieval node.
```

A failed fetch (network outage, 4xx, 5xx) is logged and swallowed — `stream()` continues without an injected memory SystemMessage that turn. The graph still runs.

**Checkpointer requirement for `context_mode="auto"` (`session_first`).** It fetches once and relies on the LangGraph `add_messages` reducer to keep the injected `SystemMessage` in state for subsequent turns of the same `session_id`. That reducer only persists across `stream()` calls when your graph was compiled with a checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())   # or SqliteSaver / PostgresSaver
```

Without a checkpointer, every `stream()` call starts from a fresh state — the SystemMessage from turn 1 vanishes and turn 2 sees no injected context. If your deployment cannot run a checkpointer (stateless serverless, one-shot calls), pass `context_mode="per-turn"` so the adapter re-injects on every turn.

`every_turn` (`context_mode="per-turn"`) works equally well with or without a checkpointer — it re-injects the latest payload each time.

---

### 4.11 Delivery Modes

| Aspect | Snapshot (`graph=` or `pickle_path=`) | Streaming (all-in-one or `graph=`) |
|--------|---------------------------------------|---------------------------|
| Stack | Sync (`SessionManager` + `IngestionClient`) | Async (`AsyncSessionManager` + `AsyncIngestionClient`) |
| Trigger | One `snapshot()` + one `ingest()` after the agent finishes | Per-turn `async for` over `stream()` |
| HTTP calls per session | 4 (`create_conversation` + `create_session` + `add_participants` + 1×`add_entries`) | 3 + N (`create_conversation` + `create_session` + `add_participants` + N×`add_entries`), where N is the number of turns — one `add_entries` per turn; plus reads: 1 `POST /api/v1/agentic/context` per fetch decision and the catalog / instruction fetches in all-in-one mode. No `close_session` is posted. |
| Tool records | Always emitted from snapshot state | Always emitted, in the turn's batch |
| Crash recovery | Re-run the snapshot — duplicates already-written entries | Re-stream the session (or fall back to a snapshot) |
| Best for | End-of-session shipping, batch jobs, replays, file handoff between processes | Live agents needing near-real-time ingestion |

**Default to all-in-one streaming for new integrations.** The customer-facing surface is smallest and the adapter handles fetch + inject + post automatically. Drop to manager-injection when you need a sync stack, fine-grained control over the `SessionManager`, or pickle-based handoff between processes.

---

### 4.12 Go-Live Checklist

- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] `owner_id` (on the manager, passed to the constructor, or passed per call to `stream()`/`run()` in all-in-one mode) is the real end-user identity (not a placeholder)
- [ ] Picked the right construction path: all-in-one for new integrations (smallest surface), `graph=` + `session_manager=` for fine-grained control, `pickle_path=` only for cross-process file handoff with a trusted writer
- [ ] All-in-one only: chosen `context_mode` matches your latency-vs-freshness target (default `"auto"` = 1 context fetch per session; `"per-turn"` = 1 per turn; `"manual"` = none). Wired `async with adapter:` around your turn loop so owned clients open and close cleanly
- [ ] All-in-one only: if your graph uses the SDK tools, bind `adapter.default_tools()` via a `graph=` builder `(sdk_tools) -> compiled graph`; if you use `context_mode="auto"`, compile with a checkpointer (or use `"per-turn"`)
- [ ] All-in-one only: when connection params (`owner_id` / `api_key` / `base_url` / `dataspace`) are passed, `session_manager=` must NOT be passed (and vice versa) — passing both raises `ValueError`
- [ ] If using `pickle_path=`, you have set `allow_pickle=True` deliberately and reviewed the trust boundary on the file
- [ ] Picked the right delivery pattern: streaming is the default for the all-in-one path; snapshot for batch shipping
- [ ] Tools are real LangChain runnables (`@tool`-decorated or equivalent); plain Python functions called inside a node do not fire tool callbacks
- [ ] `close()` is wired to your shutdown handler (sync: `adapter.close()`; async: `await adapter.close()`; all-in-one: leaving the `async with` block). Async `close()` awaits each manager's `close()` (30-second drain budget for pending deliveries). Long-running servers call `await adapter.end_session(session_id)` when a session ends
- [ ] Token refresh strategy is in place. With `auth=` (or inline `ContextAuth` credentials / the `ApiKeyAuth` demo path) the adapter exchanges and refreshes tokens for you. With a static `api_key=` / `CDP_TOKEN`, **there is no `update_token()` on `LangGraphAdapter`** — the clients capture the token at construction. To rotate, build a fresh adapter (or client + manager + adapter) and swap it in at a session boundary
- [ ] If a 401 / 403 surfaces, verify the dataspace + token combination; the underlying clients log a warning when `base_url` uses `http://` (no TLS) and accept it only for testing
- [ ] If you expect persistent-tier (`BaseStore`) data in Beacon, that is not in scope — the adapter declares `Capabilities(working=True, persistent=False)`. Ship store contents via a separate adapter

**Further reading:** `docs/LANGGRAPH_LLD.md` covers the adapter's internal design.

---

## 5. Google ADK Adapter

### 5.1 Quickstart — 3 steps

**Step 1 — Install:**
```bash
pip install agent-context-sdk
pip install google-adk        # only if your project doesn't already have it
```

`google-adk` is not a dependency of `agent-context-sdk`. If it is missing, constructing the adapter only emits a warning, but `make_start_session_tool()`, `make_context_retrieval_tool()` and `make_get_context_pack_tool()` (and therefore `default_tools()`) raise `ImportError` naming `pip install google-adk`.

**Step 2 — Construct with explicit credentials:**
```python
from pathlib import Path
from agentic_context import GoogleAdkAdapter, ContextAuth

auth = ContextAuth(
    private_key=Path("server.key").read_text(),
    consumer_key="3MVG9...",
    subject="user@example.com",
    login_url="acme.my.salesforce.com",
)
adapter = GoogleAdkAdapter(
    agent_id="Agent_Data_Cloud",     # must match a registered agent config in Beacon
    auth=auth,
    dataspace="default",
)
```

Auth is supplied as a `ContextAuth` (or `ApiKeyAuth`) object via `auth=`, **not** as a raw `base_url`/`api_key` — the JWT → CDP token exchange runs lazily on the first model call. (You can also pass the four ContextAuth credentials inline as `private_key=`/`consumer_key=`/`subject=`/`login_url=` kwargs — all four are required.) `owner_id` is optional: leave it out and the adapter takes the owner from the ADK session's `user_id` on the first model call (it calls `set_owner()` for you). See [§5.2](#52-how-it-works) for the exact rules.

**Step 2 (platform-hosted variant) — construct from an API key:** If a hosting
platform (the demo's Context Builder / Secret Manager) vends your credentials,
hold only the key and let the SDK fetch the rest — see
[§2 Authentication: the API-key path](#authentication-the-api-key-path-apikeyauth):

```python
from agentic_context import GoogleAdkAdapter

adapter = GoogleAdkAdapter(agent_id="Agent_Data_Cloud", agent_api_key=API_KEY)
# secret_server_url= optional; else set AGENTIC_CONTEXT_SECRET_SERVER_URL
```

The private key never reaches your app; the secret-server download and the JWT → CDP exchange run lazily on the first model call. If neither `secret_server_url=` nor `AGENTIC_CONTEXT_SECRET_SERVER_URL` is set (or the URL is not `https://`), that first model call raises `ValueError` — there is no default host. `agent_id` is always required and is what the adapter uses; it is not read from the secret bundle. Everything below (Step 3, callbacks, ingestion) is identical.
(Prefer to hold the auth object yourself? Build `ApiKeyAuth(api_key, secret_server_url=…)` and pass `auth=` instead — same result.)

**Step 3 — Wire into LlmAgent:**
```python
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="my_agent",
    model="gemini-3-flash-preview",
    instruction="You are a helpful support agent.",
    tools=[*adapter.default_tools()],   # recall_customer_context + get_context_pack
    **adapter.callback_kwargs(),        # before/after model + before/after tool callbacks
)
```

That's it. In the default `"auto"` mode, fast profile context is fetched from Beacon and prepended to the system instruction **once, at conversation-open** (the first model call of an ADK session); on later turns the agent pulls context on demand through the `recall_customer_context` tool (which is why `tools=[*adapter.default_tools()]` must be wired in). Every turn is ingested automatically; delivery to Beacon is awaited (best-effort) at the end of each turn. Call `await adapter.close()` on shutdown (see [§5.2](#52-how-it-works) / [§5.7](#57-go-live-checklist)).

---

### 5.2 How It Works

The Google ADK adapter integrates via ADK's four callback hooks (`before_model`, `after_model`, `before_tool`, `after_tool`). It ingests every turn automatically and, in `"auto"` mode, seeds cross-session context at conversation-open. Wire in `tools=[*adapter.default_tools()]` (the on-demand recall tool and the context-pack tool) and unpack `callback_kwargs()`; no other wiring is needed.

- **`wrap_before_model`** — runs on every model call, in this order:
  1. Binds the ADK session id to the running asyncio task and builds the Beacon HTTP clients on the first call (this is where token exchange happens; auth failures raise here).
  2. **Owner auto-detect:** if `session.user_id` is non-empty, is not `"user"` or `"placeholder"`, and differs from the adapter's current owner for this turn, it calls `set_owner(session.user_id)`. A real `session.user_id` therefore **overrides** an `owner_id` passed at construction.
  3. **Conversation-open** (the first model call for an ADK session id, and again after any `set_owner()` on that session): runs the `hooks(on_conversation_start=…)` callback / `apply()` config if any, then eagerly creates the Beacon conversation + session + agent participant (`session_id` is used as both the conversation id and the session id; a 409 from Beacon counts as success). This step is best-effort and never raises.
  4. Applies a per-conversation `system_prompt_override` if one is configured (see `hooks` / `apply`).
  5. Injects the context-pack catalog (see [§5.10](#510-context-packs)).
  6. In `"auto"` mode, **only at conversation-open**, fetches fast (profile) context via `get_context()` and prepends it to the system instruction under the heading `[Cross-session context from Data Cloud]`. It does **not** re-fetch context on later turns, and it does not add any other guidance text.
  7. Runs your `existing=` callback (if any), then buffers the incoming user message as a `user_message` record (once per ADK invocation; re-entrant model calls after a tool result are skipped).
- **`wrap_after_model`** — runs your `existing=` callback, then buffers the model output. Text that accompanies a tool call is buffered as an `agent_reasoning` Trace; the final response is buffered as an `agent_response` Message. When a model response contains no tool call, the whole buffered turn (user message + reasoning + tool traces + response) is handed to the session's `AsyncSessionManager` as **one** batch and the adapter then **awaits delivery**: it calls `await mgr.ingest(records)` followed by `await mgr.flush()` (default 20 s budget). `ingest()` itself schedules a background task; `flush()` waits for that task (including its retries) so the records have normally reached Beacon before the turn returns. If delivery takes longer than the budget, it keeps running in the background and is drained by a later `flush()` or by `close()`. Ingestion errors are logged, never raised into your agent.
- **`wrap_before_tool` / `wrap_after_tool`** — buffer `tool_call` (`variable = {"Name", "InputValueText"}`, args JSON-encoded) and `tool_result` (`variable = {"Name", "OutputValueText"}`, output truncated to 2000 characters) Trace records. They are sent with the rest of the turn.
- **Aborted turns:** if a turn never reaches its final model response (it raised), the next turn's first `before_model` detaches the leftover buffer and delivers it in a background task under the aborted turn's own `interaction_id`.
- **Event ids** are deterministic per session (`msg-<session_id>-<n>` / `step-<session_id>-<n>`), so a replay from a fresh adapter reproduces the same local ids. They are not sent, so a replay writes duplicate entries.

**Owner identity — how it is resolved**

| Source | When to use | How |
|--------|-------------|-----|
| **`session.user_id`** (auto-detect) | Default. The ADK session is created with the real user id | Create sessions with `user_id=<verified user id>`; the adapter calls `set_owner()` on the first model call. One adapter can serve many users |
| **`hooks()` / `apply()`** | Your app resolves the owner some other way (lookup, header, JWT claim) | `adapter.hooks(on_conversation_start=…)` returns `{"owner_id": …}`; or `adapter.apply(session_id, owner_id=…)` |
| **`set_owner()` / `make_start_session_tool()`** | The user identifies themselves in conversation | Call `adapter.set_owner(email)` from your own tool, or use the ready-made tool |
| **`owner_id=` at construction** | Single-owner bots, scripts, tests | Fallback owner used when no per-session owner has been resolved. A real `session.user_id` still overrides it |

If **no owner** has been resolved for a session (no `owner_id`, no real `session.user_id`, no hook, no `set_owner()`), the adapter silently skips that session's Beacon writes and reads — no error is raised, and the turn's buffered records are discarded. Do not pass `owner_id="placeholder"`: that string is not special at construction and would be used literally as the owner id.

**One adapter, many sessions.** A single `GoogleAdkAdapter` instance can serve any number of concurrent users and ADK sessions. Owner identity is scoped per ADK session (tracked per asyncio task), and each ADK session gets its own `AsyncSessionManager` (up to 1024 are kept, least-recently-used idle ones are evicted) while all of them share the adapter's two HTTP clients. Construct one adapter per process (or per agent) and call `close()` once at shutdown; you do not need one adapter per user session. One adapter instance can only back one agent: `agent_id` is fixed at construction.

### 5.2.1 Deterministic User Identification (ADK Limitation)

Google ADK's web UI uses `user_id="user"` on its requests, so the real user identity does not reach the agent through standard ADK mechanisms there. The adapter ignores `session.user_id` values of `"user"` and `"placeholder"` for owner auto-detection.

If you run behind a layer you control (a FastAPI middleware, an authenticated session, a cookie), have that layer create the ADK session with the verified user id as `user_id`, or supply the owner explicitly with `hooks(on_conversation_start=…)` / `apply(session_id, owner_id=…)` / `set_owner()`.

> **Do not rely on the LLM to extract the user's email from their messages.** The LLM can misidentify a third-party email mentioned in conversation (e.g. "forward this to bob@...") as the user's own identity. Always derive the owner from an authenticated source — a session token, cookie, or middleware-rewritten header.

### 5.3 Constructor

```python
from agentic_context import GoogleAdkAdapter, ContextAuth

adapter = GoogleAdkAdapter(
    agent_id="Agent_Data_Cloud",
    auth=ContextAuth(private_key=..., consumer_key=..., subject=..., login_url=...),
    dataspace="default",
    context_mode="auto",      # "auto" (default) | "manual"
)
```

`agent_id` is the only positional-or-keyword parameter besides `owner_id`; every other parameter is keyword-only.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `agent_id` | `str` | ✅ | — | Must match a registered agent config in Beacon. Also used as the agent's participant id |
| `owner_id` | `str \| None` | — | `None` | Fallback owner for sessions that have no per-session owner. Optional: normally the owner comes from `session.user_id`, `hooks()`/`apply()` or `set_owner()` ([§5.2](#52-how-it-works)). Readable and assignable as `adapter.owner_id` (resolves to the current turn's owner inside a turn) |
| `auth` | `ContextAuth \| ApiKeyAuth \| None` | ✅¹ | `None` | Auth object; the JWT → CDP exchange runs lazily on the first model call. Takes precedence over `agent_api_key` and the inline credentials. ¹Alternatively pass the four inline credentials or `agent_api_key=` |
| `private_key` / `consumer_key` / `subject` / `login_url` | `str` | — | `""` | The four `ContextAuth` credentials, passed inline instead of `auth=`. **All four** must be non-empty, otherwise they are ignored |
| `agent_api_key` | `str` | — | `""` | Platform-hosted (Context Builder) path: the single per-agent key. The adapter builds an `ApiKeyAuth` from it, which downloads the credential bundle from the secret server on the first model call. Used only when `auth=` is not given; takes precedence over the inline credentials |
| `secret_server_url` | `str` | — | `""` | Secret-server base URL for the `agent_api_key` path. If empty, `ApiKeyAuth` reads the `AGENTIC_CONTEXT_SECRET_SERVER_URL` environment variable. One of the two must be set and must be `https://` — otherwise the first model call raises `ValueError` (fails closed, no default host). Ignored when `auth=` is given |
| `dataspace` | `str \| None` | — | `None` | Beacon dataspace. Falls back to the `CDP_DATASPACE` environment variable, then to `"default"` (an empty string counts as unset) |
| `owner_type` | `str \| None` | — | `None` | Owner type sent to Beacon on conversation / session / context-retrieval calls. When `None`, uses the auth object's default — `"individual"` for `ContextAuth` and `ApiKeyAuth` (and when no auth is configured) |
| `context_mode` | `"auto" \| "manual"` | — | `"auto"` | `"auto"`: seed fast profile context once at conversation-open. `"manual"`: inject no profile context (the recall tool and context-pack catalog are still present; you control prompting). Any other value — including `"per-turn"` — raises `ValueError("context_mode must be one of 'auto' or 'manual'; got …")` |
| `http_observer` | `HttpObserver \| None` | — | `None` | Optional callback that receives redacted Beacon request/response summaries — see [§5.9](#59-http-observability) |
| `agent_participant_variable` | `dict[str, Any] \| None` | — | `None` | **Full override** of Beacon's `AiAgentParticipant` block. Default: `{"AiAgentType": "GoogleADK", "AiAgentVersionApiName": "v1", "AiAgentTemplateApiName": "default", "AiAgentApiName": <agent_id>}`. Combining it with `session_manager=` raises `ValueError` |
| `session_manager` | `AsyncSessionManager \| None` | — | `None` | Test seam. One injected manager backs every session and skips the auth requirement. Not multi-user safe; do not use in production |
| `retrieval_client` | `AsyncRetrievalClient \| None` | — | `None` | Test seam, used together with `session_manager=` |

Raises `ValueError` at construction if no auth is configured (no `auth`, no `agent_api_key`, not all four inline credentials, and no `session_manager`):
`GoogleAdkAdapter requires auth. Pass agent_api_key="<key>" (demo api-key path), auth=ContextAuth(...)/ApiKeyAuth(...), or inline private_key/consumer_key/subject/login_url. For tests, inject session_manager= directly.`

`GoogleAdkAdapter.capabilities` is `Capabilities(working=True, persistent=True)`.

Two optional environment variables tune the conversation-open context read: `FAST_CONTEXT_OPEN_RETRIES` (default `0`) re-reads an empty result up to that many extra times, sleeping `FAST_CONTEXT_OPEN_BACKOFF_S` (default `1.2`) seconds between attempts.

**Exports.** `agentic_context_adk.__all__` is `CONTEXT_PACK_INSTRUCTION`, `CONTEXT_PACK_TIMEOUT_S`, `GoogleAdkAdapter`, `HttpObserver` (a type alias for `Callable[[dict], Any]`). `GoogleAdkAdapter`, `ContextAuth`, `ApiKeyAuth`, `CONTEXT_PACK_INSTRUCTION` and `CONTEXT_PACK_TIMEOUT_S` can all be imported from `agentic_context`.

### 5.4 Method Reference

---

#### `set_owner`

Binds a user identity. Call as soon as the user's identity is known — Beacon writes and `get_context` calls are then scoped to that user.

**Signature:**

```python
def set_owner(owner_id: str) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `owner_id` | `str` | End-user identity (email, user ID, etc.) |

**Notes:**
- **Inside a turn** (from a callback, or from a tool ADK runs for that turn, such as the one from `make_start_session_tool()`): rebinds **only the current ADK session**, so concurrent turns for other users are unaffected. The session's Beacon lifecycle is re-armed — the next model call for that session is treated as conversation-open again (new conversation/session/participant for the new owner, and in `"auto"` mode context is seeded again).
- **Outside any turn** (bootstrap): updates the fallback owner and re-arms every session.
- Records already buffered for the current turn under the previous owner id (or with no owner) are re-tagged with the new owner id before they are delivered, so even the turn in which the user identifies themselves lands with the correct participant id. Only the current session's buffer is touched when called inside a turn.
- Call it again if the user changes.

**Usage:**

```python
# In your own session-start tool — after the user provides their email:
def start_session(customer_email: str) -> dict:
    adapter.set_owner(customer_email)
    return {"status": "success"}
```

---

#### `get_context`

Retrieves context for the current owner from Beacon.

**Signature:**

```python
async def get_context() -> dict | None
```

**Returns:** the context dict from Beacon (it carries a `userContext` list), or `None` when there is nothing to read: no owner resolved for the session, no retrieval client yet (before the first model call), the session's Beacon conversation has not been opened yet (including right after `set_owner()`, until the next model call), or the Beacon read failed (failures are logged and swallowed).

**Notes:**
- In `"auto"` mode `wrap_before_model` calls it once at conversation-open, and the `recall_customer_context` tool calls it on demand. You only need to call it yourself for custom flows.
- Inside a turn it reads the current ADK session's manager. Outside a turn it only works if the adapter has an injected manager or exactly one live session manager; otherwise it returns `None`.

---

#### `make_start_session_tool`

Returns a ready-made ADK `FunctionTool` that binds the session to the user's identity. Drop it into `LlmAgent`'s `tools` list.

The LLM decides when to call it from the tool's docstring, which instructs it to fire only when the user provides **their own** email address — not when an email appears in any other context (e.g. "forward this to bob@...").

**Signature:**

```python
def make_start_session_tool() -> FunctionTool
```

**Returns:** an ADK `FunctionTool` wrapping `start_session(customer_email: str) -> {"status": "success", "customerEmail": customer_email}`; it calls `adapter.set_owner(customer_email)`. Raises `ImportError` if `google-adk` is not installed.

**Usage:**

```python
agent = LlmAgent(
    # ...your other LlmAgent arguments (name, model, instruction, ...)
    tools=[adapter.make_start_session_tool(), *adapter.default_tools()],
    **adapter.callback_kwargs(),
)
```

**Note:** `default_tools()` does **not** include this tool. If your `start_session` needs to do more than identity binding (e.g. create a CRM record, log metrics), write your own function and call `adapter.set_owner(email)` inside it instead.

---

#### `callback_kwargs`

Returns the four ADK callbacks — `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback` — ready to unpack into `LlmAgent`. These are exactly the keys returned; no agent/error callbacks are wired. This is the primary wiring call used throughout the examples.

**Signature:**

```python
def callback_kwargs(
    *,
    before_model: Callable | None = None,
    after_model: Callable | None = None,
    before_tool: Callable | None = None,
    after_tool: Callable | None = None,
) -> dict[str, Callable]
```

**Usage:**

```python
agent = LlmAgent(..., **adapter.callback_kwargs())
# compose with your own callbacks:
agent = LlmAgent(..., **adapter.callback_kwargs(before_model=my_metrics_cb))
```

**Note:** wires only the four callbacks — it does **not** register any tool. Register tools separately via `tools=[*adapter.default_tools()]` (ADK treats `tools` and the `*_callback` kwargs as distinct `LlmAgent` parameters).

---

#### `default_tools`

Returns the SDK-provided tools to spread into `LlmAgent(tools=[...])`. It returns exactly two tools, in this order:

1. `recall_customer_context` — on-demand owner-scoped memory read (`make_context_retrieval_tool()`).
2. `get_context_pack` — pack-scoped read of a named context pack (`make_get_context_pack_tool([])`, with an empty `valid_pack_ids` fallback). See [§5.10](#510-context-packs).

**Signature:**

```python
def default_tools(self) -> list[Any]
```

**Usage:**

```python
agent = LlmAgent(
    ...,
    tools=[*adapter.default_tools(), *my_tools],
    **adapter.callback_kwargs(),
)
```

Wire it in for `"auto"` mode to work as described: the conversation-open seed is the fast profile read, and the agent reaches for further history through these tools. Do not also add your own `make_get_context_pack_tool(...)` to the same agent — both are named `get_context_pack`. If you need `valid_pack_ids`, `param_defaults` or `timeout`, build the tools yourself instead of calling `default_tools()`.

---

#### `make_context_retrieval_tool`

Returns a single ADK `FunctionTool` wrapping `recall_customer_context(topic: str = "")` — the on-demand owner-scoped memory read. Use it if you want the recall tool without the context-pack tool.

**Signature:**

```python
def make_context_retrieval_tool() -> FunctionTool
```

The tool resolves the owner from the adapter's current identity, so the LLM never supplies a customer ID. The tool's docstring is the LLM-visible description. Internally it calls `get_context()`; `topic` is an LLM-visible hint only and is not sent to Beacon. It returns the context dict, or `{"userContext": []}` when `get_context()` returned nothing. Raises `ImportError` if `google-adk` is not installed.

---

#### `make_get_context_pack_tool`

Returns an ADK `FunctionTool` named `get_context_pack` that reads one named context pack. Full behaviour in [§5.10](#510-context-packs).

**Signature:**

```python
def make_get_context_pack_tool(
    valid_pack_ids: list[str],
    *,
    param_defaults: dict[str, Any] | None = None,
    timeout: float | None = 60.0,     # agentic_context.CONTEXT_PACK_TIMEOUT_S
) -> FunctionTool
```

---

#### `get_context_pack_instruction`

```python
async def get_context_pack_instruction(session_id: str) -> str
```

Returns the context-pack instruction text pinned for that session. See [§5.10](#510-context-packs). Most integrations never call it, because `before_model` adds it for you.

---

#### `set_context_mode` / `context_mode`

Switches the memory mode at runtime — a reversible operator/developer knob. The current mode is readable via the `context_mode` property (`"auto"` or `"manual"`).

**Signature:**

```python
def set_context_mode(mode: Literal["auto", "manual"]) -> None
```

Raises `ValueError` for any other value. The change applies from the next model call.

**Note:** flipping to `"auto"` mid-process does **not** retroactively seed already-open sessions — their conversation-open signal is already consumed, so those stay tool-only until they re-open (for example after `set_owner()` on that session).

---

#### `add_participant`

Adds a participant to the current Beacon conversation (async). Call it to register another agent before handing off, or a human who will send messages directly. Must be called from inside a turn (for example from a tool), after the session's first model call, so that the conversation exists.

**Signature:**

```python
async def add_participant(
    participant_id: str,
    participant_type: str = "agent",   # "agent" | "human"
    role: str = "participant",
) -> None
```

Pass the final ID directly (no suffix added) — agents use their plain `agent_id` and humans use their identifier (e.g. an email). It is best-effort: if there is no HTTP client yet, or the conversation has not been created yet, it logs a warning and returns; a failed Beacon call is logged and swallowed. It never raises for those cases.

---

#### `register_shutdown`

Installs `SIGINT` and `SIGTERM` handlers that try to run `close()` and then re-raise the signal with the default action.

**Signature:**

```python
def register_shutdown(timeout: float = 10.0) -> None
```

`timeout` is the number of seconds the handler waits for `close()` before giving up.

> **Caution (verified):** the handler schedules `close()` on the running loop and then blocks the signal-handling (main) thread waiting for it. If your event loop runs on the main thread (for example `asyncio.run(...)` in `main`, or uvicorn), `close()` cannot run during that wait; the handler times out after `timeout` seconds and the process then exits with the default signal action. If no event loop is running on the main thread (for example the loop runs in a worker thread), the handler cannot find a loop, silently skips `close()`, and re-raises the signal immediately. For a reliable drain, call `await adapter.close()` yourself (a `try/finally` around your run loop, or your framework's shutdown/lifespan hook). Treat `register_shutdown()` as a convenience only.

---

#### `hooks` / `apply`

Per-conversation configuration (advanced). Register **one** adapter and supply per-conversation identity/config instead of fixing an `owner_id` at construction. Both return the adapter (chainable).

**Signatures:**

```python
def hooks(*, on_conversation_start: Callable[[str, dict], dict | None] | None = None) -> GoogleAdkAdapter
def apply(session_id: str, *, owner_id: str | None = None, dataspace: str | None = None,
          system_prompt_override: str | None = None, **overrides: Any) -> GoogleAdkAdapter
```

`hooks()` registers a callback invoked at conversation-open (the first model call of an ADK session, and again whenever the owner is re-identified on that session). It receives `(session_id, ctx)` where `ctx` is `{"session_id", "user_id", "current_owner"}`, and returns a config dict (or `None`/`{}` for "no change"). It runs before the Beacon conversation is created, so an `owner_id` it returns applies from the first write. An exception in the hook is logged and swallowed; a non-dict return value is logged and ignored.

`apply()` is the push counterpart: it sets the same per-session config directly for a known `session_id`, before or during a session. Only the keys you pass are changed.

Recognised config keys:

| Key | Effect |
|-----|--------|
| `owner_id` | Binds the owner for that session (via `set_owner`) |
| `system_prompt_override` | Replaces the agent's base system instruction for that conversation on every model call (memory and context-pack blocks are still added on top) |
| `context_mode` | `"auto"` / `"manual"` for that session only; overrides the adapter-wide mode. An invalid value raises `ValueError` |
| `dataspace` | Stored but **not** used: the HTTP clients are built once with the constructor's dataspace |

Other keys are stored and ignored.

---

#### `wrap_before_model` / `wrap_after_model` / `wrap_before_tool` / `wrap_after_tool`

Return an ADK-compatible callback. Each accepts an optional `existing=` callable to compose with your own callback. `callback_kwargs()` calls these for you.

**Signature:**

```python
def wrap_before_model(existing: Callable | None = None) -> Callable
def wrap_after_model(existing: Callable | None = None) -> Callable
def wrap_before_tool(existing: Callable | None = None) -> Callable
def wrap_after_tool(existing: Callable | None = None) -> Callable
```

| Wrapper | Your `existing` callback | Return value |
|---------|--------------------------|--------------|
| `wrap_before_model` | Runs after the adapter's context injection and before it buffers the user message. May be sync or `async` | Always `None` — whatever `existing` returns is discarded, so it cannot short-circuit the model call |
| `wrap_after_model` | Runs first. Must be **sync** (it is called but not awaited) | Your callback's return value |
| `wrap_before_tool` / `wrap_after_tool` | Runs first. Must be **sync** | Your callback's return value |

---

#### `close`

Drains every session manager's pending deliveries, then closes the adapter's HTTP clients. Call when the process (or the adapter's life) ends. This is **local cleanup only** — no `close_session` is ever posted to Beacon; returning the same user to the same session later just works.

**Signature:**

```python
async def close() -> None
```

Each manager waits up to 30 s for pending deliveries and then cancels stragglers. Because every turn already awaits its own delivery (see [§5.2](#52-how-it-works)), `close()` mainly matters for deliveries that outlived the per-turn budget and for aborted-turn salvage. Do not keep using the adapter after `close()`: its session managers are closed, and later turns' ingests fail (the error is logged, not raised).

---

### 5.5 End-to-End Example: Google ADK Support Agent

Complete example — one shared adapter, credentials passed explicitly at construction, the owner taken from each ADK session's `user_id`:

```python
import asyncio
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from agentic_context import GoogleAdkAdapter, ContextAuth

BEACON_AGENT_ID = "your-registered-agent-id"
BEACON_DATASPACE = "default"
AUTH = ContextAuth(
    private_key=Path("server.key").read_text(),
    consumer_key="3MVG9...",
    subject="user@example.com",
    login_url="acme.my.salesforce.com",
)

# One adapter for the whole process. The owner is not fixed here: it is taken
# from session.user_id on each session's first model call.
adapter = GoogleAdkAdapter(
    agent_id=BEACON_AGENT_ID,
    auth=AUTH,
    dataspace=BEACON_DATASPACE,
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-3-flash-preview",
    instruction="You are a helpful support agent.",
    tools=[*adapter.default_tools()],       # recall_customer_context + get_context_pack
    **adapter.callback_kwargs(),
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="myapp", session_service=session_service)

async def run_session(user_email: str, user_message: str) -> str:
    # A new ADK session per conversation; user_id carries the verified identity.
    session = await session_service.create_session(
        app_name="myapp",
        user_id=user_email,
    )

    message = genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
    response = ""
    async for event in runner.run_async(
        user_id=user_email, session_id=session.id, new_message=message
    ):
        if hasattr(event, "content") and event.content:
            for part in (event.content.parts or []):
                if getattr(part, "text", None):
                    response += part.text
    return response

# Multi-session memory: same email, two separate sessions
async def main():
    try:
        # Session 1
        resp1 = await run_session("alice@example.com", "I'm on the Pro plan and have a billing issue")
        print(f"Session 1: {resp1}")

        # Session 2 — Beacon returns context from Session 1
        resp2 = await run_session("alice@example.com", "What did we talk about last time?")
        print(f"Session 2: {resp2}")
    finally:
        await adapter.close()   # once, at shutdown

asyncio.run(main())
```

The adapter handles context seeding and ingestion automatically on every turn. The verified user id on the ADK session is the only required step for identity binding. (You may instead create one adapter per user with `owner_id=user_email`; that works too, but is not required.)

### 5.6 Multi-Session Memory

Each ADK session creates its own Beacon conversation (`conversation_id = session_id`, the same convention as the LangGraph and Claude adapters). Cross-session memory comes from Beacon's owner-level context: `get_context()` is scoped to the `owner_id` across all of that user's conversations, so a new session for the same owner can see facts ingested in earlier sessions (context only appears once Beacon has processed what was ingested, so a very recent session may not be visible yet — `get_context()` returns `None` or an empty `userContext` in that case).

```python
# Both sessions are served by the same adapter; only the ADK session differs.
adapter = GoogleAdkAdapter(agent_id="my-agent", auth=AUTH)

# Session 1
s1 = await session_service.create_session(app_name="myapp", user_id="alice@example.com")
# ... run turns; each turn is delivered to Beacon at its end ...

# Session 2 — same user_id, new ADK session (new Beacon conversation)
s2 = await session_service.create_session(app_name="myapp", user_id="alice@example.com")
# In "auto" mode the adapter seeds Alice's context into the system instruction on
# this session's first model call; in any mode the agent can call
# recall_customer_context on demand.

await adapter.close()   # at shutdown
```

(`AUTH` is the `ContextAuth` from [§5.1](#51-quickstart--3-steps); `session_service` is an ADK session service as in [§5.5](#55-end-to-end-example-google-adk-support-agent).)

### 5.7 Go-Live Checklist

- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] `auth=` (a `ContextAuth`/`ApiKeyAuth`), the four inline credentials, or `agent_api_key=` (+ `secret_server_url=` / `AGENTIC_CONTEXT_SECRET_SERVER_URL`) passed at construction
- [ ] ADK sessions created with the real, verified user id as `user_id` (not `"user"` / `"placeholder"`) — or the owner supplied via `hooks()` / `apply()` / `set_owner()`. Without a resolved owner the adapter silently skips Beacon writes and reads
- [ ] `tools=[*adapter.default_tools()]` and `**adapter.callback_kwargs()` both wired into `LlmAgent`
- [ ] Auth handles its own token refresh — `ContextAuth`/`ApiKeyAuth` re-exchange the JWT automatically; no manual token plumbing
- [ ] `get_context()` returns `userContext` entries (confirms agent config exists in Beacon and prior data is ingested)
- [ ] `await adapter.close()` called on process shutdown (in a `try/finally` or your framework's shutdown hook — see the `register_shutdown()` caution in [§5.4](#54-method-reference)) to drain any delivery still in flight (see [§5.2](#52-how-it-works) and [§8](#8-troubleshooting) "Records Not Appearing")

### 5.8 Multi-Agent Handoff (one key per agent)

A multi-agent app — several specialised agents that hand a conversation off to
one another while sharing one end-user's memory — maps cleanly onto the SDK with
**no adapter changes**. The model is: **one API key → one agent → one adapter →
one Beacon conversation per ADK session, all under the same owner.**

```python
# Each agent gets its own key (the Secret Manager maps key → agentId 1:1),
# so each gets its own adapter. Pass the key straight in as agent_api_key= —
# the adapter builds the ApiKeyAuth internally. The owner is not set here: each
# ADK session carries the same user_id, which the adapter uses as the owner.
billing_adapter = GoogleAdkAdapter(agent_id="Agent_Billing", agent_api_key=BILLING_KEY)
support_adapter = GoogleAdkAdapter(agent_id="Agent_Support", agent_api_key=SUPPORT_KEY)
# (secret_server_url= optional per adapter; else AGENTIC_CONTEXT_SECRET_SERVER_URL.)

# One LlmAgent + Runner per specialised agent; route the next turn to whichever
# Runner the handoff selects. All sessions carry the same user_id (the owner).
```

**Topology that preserves per-agent governance:** give each agent its **own ADK
`Runner` and its own ADK session** (so each maps to its own Beacon conversation),
and implement handoff as **application-level routing** — send the next user turn
to a different Runner. Use a different session id per agent; two sessions that
share an id would share one Beacon conversation. Because every session shares one owner,
cross-session context (`get_context()` / `recall_customer_context`) surfaces the
*combined* memory to whichever agent is active, while each agent's authored turns
stay in its own conversation. If you want the receiving agent's participant
registered in the sending conversation, call `await adapter.add_participant(agent_id)`
from inside a turn ([§5.4](#add_participant)).

> **Do not use ADK's native `sub_agents` / `transfer_to_agent` for this.** Those
> keep all agents inside **one** ADK session → **one** Beacon conversation with
> mixed authorship, which collapses the per-agent boundary. Use independent
> Runners instead.

All mutable adapter state (owner map, session managers, buffers, auth, HTTP
clients, catalog cache) is held per adapter instance; the only module-level
state is a context variable carrying the current ADK session id, which is not
keyed by agent. N adapters therefore coexist in one process with linear cost
(N keys + N adapters + N sessions). This scales to any N, not just two.

### 5.9 HTTP Observability

Pass `http_observer` to watch **every** Beacon HTTP call the adapter makes — both
ingestion writes and retrieval reads (including the pack catalog and system-prompt
fetches). It's the seam for surfacing live Beacon traffic in a UI / API console.
Mirrors the Claude adapter's `http_observer`
([§3.11](#311-http-observability)); unlike Claude, ADK has no sync mode, so the
observer is always honoured (no warning).

```python
async def on_beacon_call(event: dict) -> None:   # may be sync or async
    # phase="request":  {"phase","callId","method","url","requestBody","timestamp"}
    # phase="response": {"phase","callId","method","url","status","latencyMs",
    #                    "responseBody","traceId","timestamp"}
    print(event["phase"], event["method"], event["url"], event.get("status"))

adapter = GoogleAdkAdapter(
    agent_id="Agent_Data_Cloud",
    auth=AUTH,                     # the ContextAuth from §5.1
    http_observer=on_beacon_call,
)
```

- **`callId`** pairs a request event with its response event.
- **`url` has no query string** — it is the request URL (scheme, host and path)
  with the query string (which can carry tokens / owner_ids) stripped before the
  event is built, so the query never reaches your observer/UI.
- **Headers are never emitted** — the bearer token lives in a header, so only
  the decoded request/response bodies are surfaced (JSON bodies are pretty-printed).
  Each body is capped at 20,000 characters and truncated with a trailing
  `… (truncated)`; an empty body is `None`.
- **`traceId`** (response events) is a dict of server trace / request-correlation
  response headers (for example `x-request-id`, `traceparent`), or `None` if the
  response carried none. It never includes `authorization` or cookie headers.
- **Token exchange bypasses the observer** — the JWT → CDP exchange uses its own
  client, so credentials never flow through the hook.
- **Best-effort** — observer exceptions are swallowed; observability never
  disrupts delivery.
- **Always active when set** — wired via httpx event hooks on the two clients
  the adapter builds (ADK is async-only).
- A transport-level failure (no HTTP response at all) produces a request event with no matching response event.

### 5.10 Context Packs

A context pack is a named, structured bundle of data (with declared "data needs" and parameters) that Beacon publishes for an agent. The ADK adapter supports packs end to end; with `tools=[*adapter.default_tools()]` and `**adapter.callback_kwargs()` there is nothing else to wire.

**The tool.** `default_tools()` includes `get_context_pack`. Its arguments, as the model sees them:

| Argument | Type | Meaning |
|----------|------|---------|
| `context_pack_id` | `str` | One id from the advertised catalog |
| `params` | `dict[str, str] \| None` | Parameter name → value, from the pack's data needs. Blank values are dropped; `owner_id` is stripped (the owner always comes from the adapter, never from the model) |
| `query` | `str` | Optional free text, sent as `params["query"]` only if `params` did not already set it |
| `context_pack_version` | `int \| None` | The catalog entry's `version`, when it has one. Whole-number floats (e.g. `3.0`) are accepted and converted; if omitted, the adapter fills in the catalog's version for that pack |

It returns one of these `status` values and never raises:

| `status` | When |
|----------|------|
| `ok` | Success: `{"status": "ok", "context_pack_id", "context": <Beacon payload>}` |
| `invalid_pack` | The id is not in the allowed set. Includes `valid_pack_ids` (sorted) and a message; no Beacon call is made |
| `invalid_version` | `context_pack_version` was not an integer |
| `version_expired` | The pack was republished since the catalog was read. Includes `requested_version`, `latest_version` and `latest` (the new description); the adapter updates its cached version so a retry uses the new one |
| `not_ready` | Any failure: client/auth init failed, no active conversation or owner, or the request failed or timed out (the `detail` field carries only the reason or the exception type) |

**Catalog injection (via `before_model`).** At conversation-open the adapter reads the agent's authorised packs (`GET /api/v1/agentic/runtime-context-packs`, filtered by `agent_id`) and caches the result on the adapter (it is refetched at every conversation-open; a failed fetch keeps the previous catalog and logs a warning). On every model call it then appends two things to the system instruction, unless the instruction already contains an `[Available context packs]` block:

1. the context-pack instruction — only when at least one pack is advertised, and skipped if your own prompt already contains it;
2. the catalog block: `[Available context packs]` followed by JSON of `{pack_id: {"description", "dataNeeds": [...], "version"?}}`. `version` is included only when the server reports an integer version. With no packs, the block is `[Available context packs]` followed by `{}` — it is present in both `"auto"` and `"manual"` modes.

**The instruction.** `await adapter.get_context_pack_instruction(session_id)` returns the instruction for a session. On a session's first call it fetches the server-published text (`GET /api/v1/agentic/prompts/system/context_pack_instruction`) and pins it for that session; a miss or failure pins the bundled `agentic_context.CONTEXT_PACK_INSTRUCTION` instead. Before the retrieval client exists (before the first model call) it returns the bundled text without pinning it.

**Versioning.** The instruction and the tool description tell the model to pass a catalog entry's `version` as `context_pack_version`. The adapter backs this up: it fills the version in when the model omits it, rejects non-integers, and handles `version_expired` as above (the model may retry once with `latest_version`).

**`valid_pack_ids` fallback.** `make_get_context_pack_tool(valid_pack_ids, *, param_defaults=None, timeout=60.0)`:
- `valid_pack_ids` is only a fallback allow-list: it is used **only when the fetched catalog is empty** (for example the catalog fetch failed, or no packs are published). Once a non-empty catalog has been fetched, the catalog decides and `valid_pack_ids` is ignored. `default_tools()` passes `[]`, so with an empty catalog every id is rejected as `invalid_pack`.
- `param_defaults` are fixed `params` merged into every call; a key the model also sends overrides the default, and a blank model value drops the key (default included).
- `timeout` is the per-call HTTP timeout for the pack read, default `CONTEXT_PACK_TIMEOUT_S` (60.0 s), independent of the 10 s default used for other reads.

```python
agent = LlmAgent(
    ...,
    instruction=my_prompt,
    tools=[
        adapter.make_context_retrieval_tool(),
        adapter.make_get_context_pack_tool(["<your-pack-id>"], param_defaults={"<param-name>": "<value>"}),
    ],
    **adapter.callback_kwargs(),
)
```

---

## 6. OpenAI Agents SDK Adapter

> **TL;DR** — Create **one** `OpenAIAdapter` for your whole app. Per conversation, wire three things into your Agents SDK run: `instructions=adapter.instructions_provider(base, session_id, owner_id)` (injects the context-pack catalog), `tools=adapter.default_tools(session_id, owner_id, valid_pack_ids=[...])` (the on-demand memory tool + the context-pack tool), and `hooks=adapter.run_hooks(session_id, owner_id, user_input=text)` on `Runner.run` (ingests the whole turn — user message, replies, reasoning, tools — in one POST under one `interaction_id`). Call `await hooks.flush()` in a `finally` so runs that end without a final output are still ingested. Everything else in this section is optional detail.

> **Sync vs async — read this first.** The adapter defaults to `use_async=True`, and `instructions_provider()` and `run_hooks()` **raise `RuntimeError` under `use_async=False`**. Every I/O method (`push`, `ingest`, `get_context`, `get_session`, `list_sessions`, `start_session`, `close`, `get_context_pack_instruction`) is a coroutine in **both** modes — always `await` them. See [§6.8](#68-sync-vs-async-use_async).

### 6.1 Quick Start (the 90% case)

```python
from pathlib import Path

from agents import Agent, Runner
from agentic_context_openai import OpenAIAdapter   # or: from agentic_context import OpenAIAdapter

# 1. ONE adapter for the whole process (not one per conversation).
#    owner_id is REQUIRED at construction (see §6.3 for how per-call owners work).
adapter = OpenAIAdapter(
    agent_id="support-bot",                          # must be registered in Beacon
    owner_id="<default-owner-id>",                   # fallback owner; override per call
    private_key=Path("server.key").read_text(),      # PEM RSA private key
    consumer_key="3MVG...",                           # Connected App consumer key
    subject="your-username@example.com",              # Salesforce username for JWT bearer
    login_url="login.salesforce.com",                 # login host (no scheme)
)

async def handle_turn(conversation_id: str, owner_id: str, user_text: str) -> str:
    # 2. Wire the Agent: dynamic instructions + the two memory tools.
    agent = Agent(
        name="support",
        instructions=adapter.instructions_provider(
            base="You are a helpful support agent.",
            session_id=conversation_id,
            owner_id=owner_id,
        ),
        tools=adapter.default_tools(
            conversation_id, owner_id,
            valid_pack_ids=["<your-pack-id>"],       # ids your agent may request
        ),
    )

    # 3. Run with the hooks. The user message and every agent_response /
    #    agent_reasoning / tool_call / tool_result record of this run share one
    #    interaction_id and are ingested in ONE POST when the run ends.
    hooks = adapter.run_hooks(conversation_id, owner_id, user_input=user_text)
    try:
        result = await Runner.run(agent, user_text, hooks=hooks)
    finally:
        await hooks.flush()   # no-op if on_agent_end already flushed
    return result.final_output

# 4. On shutdown (once for the whole app).
await adapter.close()
```

`valid_pack_ids` only matters if your agent has context packs published in Beacon; pass `valid_pack_ids=[]` (or omit it) if you have none — see [§6.7](#67-context-packs).

**Lifecycle is lazy** — the Beacon conversation + session are created automatically on the first ingest (`push` / `ingest` / run hooks). You do not need to call `start_session()`.

### 6.2 Minimal Requirements

| You need | Notes |
|----------|-------|
| `pip install agent-context-sdk` | Python 3.10+. The adapter ships inside the single `agent-context-sdk` wheel. |
| `pip install openai-agents` | Required **only** for `make_context_retrieval_tool()`, `make_get_context_pack_tool()`, `default_tools()` and `run_hooks()` (each raises `ImportError` with the install hint if it is missing). `run_hooks()` additionally needs `openai-agents >= 0.3` (older versions lack `RunHooks.on_llm_start` / `on_llm_end`) and raises `ImportError` otherwise. The pure ingestion path (`snapshot`, `push`, `ingest`, `get_context`, …) works without it. |
| A registered `agent_id` in Beacon | Required for `get_context()` to return data. See [§2 Prerequisites](#2-prerequisites). |
| Credentials | `auth=` (a shared `ContextAuth`), **or** `private_key` + `consumer_key` + `subject` + `login_url`, **or** `secrets_api_key`. There is no static-token path. See [§2 Authentication](#authentication-let-the-sdk-manage-your-token-recommended). |
| `owner_id` | **Required** at construction (no default). Pass the real end user per call as well in multi-user apps. |

If `openai-agents` is not installed, constructing the adapter still succeeds but emits a `warnings.warn(...)` naming `pip install openai-agents`.

The **only things you must do per conversation** are: pick a `session_id` (your conversation/thread ID) and an `owner_id` (the end user), pass both ids to `instructions_provider` / `default_tools` / `run_hooks`, pass the user message as `run_hooks(..., user_input=...)`, and `await hooks.flush()` after the run.

### 6.3 One Adapter, Many Sessions (ownership model)

A single `OpenAIAdapter` manages **N sessions internally**, keyed by `session_id` (one `SessionManager` / `AsyncSessionManager` per session, plus one shared ingestion client and one shared retrieval client). Create it once at startup; `close()` it once at shutdown.

**Owner is per call, not per adapter.** The constructor `owner_id` is only the fallback used when a call omits its own. The first time the adapter sees a `session_id` it binds that session to an owner (the call's `owner_id`, else the constructor default) and keeps that binding for the adapter's lifetime.

| Operation | Owner comes from |
|-----------|------------------|
| `push` / `ingest` / `start_session` / `get_context` / `get_session` | the session's manager. An explicit `owner_id=` only matters for a session the adapter has not seen yet ("cold"); for a known session it is checked against the binding. |
| `instructions_provider(base, session_id, owner_id)`, `default_tools(session_id, owner_id)`, `make_*_tool(..., session_id, owner_id)`, `run_hooks(session_id, owner_id)` | the call (falls back to the constructor `owner_id`) |
| `list_sessions(owner_id=...)` | the call — an **owner-level** query. In a multi-user app you must pass it; omitting it lists the constructor owner's conversations. |

**Session ↔ owner binding is enforced.** Passing an explicit `owner_id` that differs from the one a session is already bound to raises:

```
ValueError: session_id 'c' is already bound to owner 'u'; refusing to serve owner 'other'. Session ids must not be reused across owners.
```

Where that error surfaces: `push`, `ingest`, `start_session`, `get_context`, `get_session` and `instructions_provider(...)` raise it to you (`instructions_provider` raises at call time, i.e. at `Agent` construction, not on the first turn). The recall tool returns `{"status": "not_ready", ...}` instead. The run hooks catch it, log a warning, and drop the batch (hooks are best-effort).

The owner you pass also scopes **writes**: `user_message` records get `participant_id` = the resolved owner (Beacon marks an entry `role="owner"` only when `participantId == ownerId`). Agent-side records (`agent_response`, `agent_reasoning`, `tool_call`, `tool_result`) get `participant_id` = `agent_id`.

`adapter.session_id` returns the **most recently created** session's id (dict insertion order), not the most recently used one. `get_context()` / `get_session()` with no `session_id` fall back to it; in a multi-session server **always pass `session_id` explicitly**. The tools never fall back — a tool built without a `session_id` returns `not_ready` (§6.6).

### 6.4 How It Works (under the hood)

| Path | When | What happens |
|------|------|--------------|
| `adapter.push(...)` | You call it (a full transcript, or a slice outside `run_hooks`) | Normalizes Chat Completions messages / Responses items to `Record`s and ingests them. Under `context_mode="auto"`, the **first** push for a session also reads fast context once and caches it for the recall tool. |
| `instructions_provider(...)` | Agents SDK calls it every turn | First turn of a `session_id`: fetches the context-pack catalog from Beacon. Every turn: returns `base` + (context-pack instruction, when any pack id is allowed) + `[Available context packs]` block. |
| `recall_customer_context` tool | The **model** calls it | Returns the owner's cross-session profile context as JSON text. |
| `get_context_pack` tool | The **model** calls it | Returns one named context pack as JSON text. |
| `run_hooks(...)` | Agents SDK lifecycle events | Buffers the run: `user_input` → `user_message`; `on_llm_end` → records for the model's output items; `on_tool_end` → a `tool_result` record. `on_agent_end` (or `flush()`) ingests the buffer in one POST. Under `context_mode="auto"`, `on_agent_start` seeds fast context once per session. |

Ingestion is **non-blocking in async mode**: the awaited `ingest` returns once delivery is scheduled as a background task; `close()` drains pending deliveries (30 s budget). Data is therefore not guaranteed to be in Beacon before the next turn begins.

#### 6.4.1 Inputs, record mapping, and event ids

`push()` / `snapshot()` accept two input shapes (either or both; `messages` are processed first, then `items`, concatenated — not interleaved):

- `messages=` — Chat Completions dicts (`{"role": ..., "content": ...}`, assistant `tool_calls`, `{"role": "tool", "tool_call_id": ...}`).
- `items=` — Responses API items, e.g. `result.to_input_list()` from the Agents SDK (`message`, `function_call`, `function_call_output`, `reasoning`).

| Source entry | Record(s) emitted |
|--------------|-------------------|
| `role: "user"` with text | `user_message` (`category="Message"`, `participant_id` = owner) |
| `role: "assistant"` with text | `agent_response` (Message, `participant_id` = `agent_id`) |
| assistant `tool_calls[]` / `type: "function_call"` | one `tool_call` per call (`category="Trace"`, `variable={"Name", "InputValueText"}`, `content=None`) |
| `role: "tool"` / `type: "function_call_output"` | `tool_result` (Trace, `variable={"Name", "OutputValueText"}`; `Name` is resolved from the earlier call with the same id, else `""`) |
| `type: "reasoning"` (`content` or `summary`) | `agent_reasoning` (Trace, `variable={"OutputValueText"}`) |
| `system` / `developer` roles, other roles, empty-text user/assistant entries | dropped |

Text is extracted from a plain string or a list of `text` / `input_text` / `output_text` / `summary_text` parts (and `refusal`); image/audio parts contribute nothing. An entry's optional `"timestamp"` (ISO 8601 string or `datetime`) is used; otherwise the record is stamped `now(UTC)` at transform time.

**Event ids (local only — not sent to Beacon).** The native OpenAI id is used when present (`call_id` / `id`; the Agents SDK placeholder `"__fake_id__"` counts as absent; `tool_result` ids are `"<call_id>-out"`). Otherwise a fallback id `msg-<session_id>-<interaction_id or "snap">-<idx>-<suffix>` is used, where `idx` **restarts at 0 on every `push()` / `snapshot()` call**. The SDK omits `id` on the wire (Beacon assigns entry ids), so these ids are for local identity, logs and callbacks only. Consequences:

- **Snapshot style (push the full conversation every time)** — every push writes the whole transcript again (ids are not sent, so nothing is deduped). Push the full transcript once per session; for live sessions, use incremental pushes.
- **Incremental style (push only the new slice each turn)** — entries without a native id should be pushed with a unique `interaction_id` per slice. Without it, turn 2's first user message gets the same local id as turn 1's (`msg-<sid>-snap-0-u`), which makes logs and callbacks ambiguous. The run hooks do it for you: each `run_hooks(...)` instance mints one `turn-<session_id>-<12 hex>` for its run and keeps `idx` running across the run's model calls, so ids stay distinct within the turn.

#### 6.4.2 Memory modes: `auto` / `manual`

`context_mode` has two values. In **both** modes the `recall_customer_context` tool and the context-pack catalog (via `instructions_provider`) work; the mode only controls the adapter-side fast-context seed done in `push()`.

| `context_mode` | `push()` behaviour |
|----------------|--------------------|
| `"auto"` (default) | On the **first** `push` for a `session_id` in this process, `await`s one fast-context read (`get_context`) before ingesting; a non-empty result is cached and later served by the recall tool, saving a Beacon round trip. Later pushes for the session do nothing extra. |
| `"manual"` | No seed. The recall tool does a live `get_context` read on each call. |

Notes:

- The seeded context is **not** injected into the prompt — it only feeds the recall tool's cache. The model still has to call the tool to see it.
- The seed is awaited inline (not a background task) and is best-effort: a Beacon read failure never fails the push. `SessionManager.get_context()` swallows Beacon errors and returns `None`, so such a failure still counts as "seeded" and is not retried; the seed is retried on the next push only if the read itself raised.
- Once cached for a session, the recall tool returns that cached payload on **every** later call for the session; the cache is not refreshed.
- "First push" is tracked in an in-memory set — a process restart re-seeds.
- Under `manual`, a push still marks the session as seeded, so switching to `auto` later does not seed sessions that already had a push.
- `instructions_provider` and `push` use separate once-per-session gates, so calling them in either order is fine.
- Flip the mode live with `adapter.set_context_mode("auto"|"manual")`; `adapter.context_mode` reads it.

### 6.5 Constructor

```python
from agentic_context_openai import OpenAIAdapter

# Recommended — credential-based: the SDK handles token exchange + refresh.
adapter = OpenAIAdapter(
    agent_id="support-bot",
    owner_id="<default-owner-id>",
    private_key=Path("server.key").read_text(),
    consumer_key="3MVG...",
    subject="your-username@example.com",
    login_url="login.salesforce.com",
    dataspace="default",
    use_async=True,
    context_mode="auto",
)

# Alternative — API key only: the SDK downloads the credential set from the secrets server.
adapter = OpenAIAdapter(
    agent_id="support-bot",
    owner_id="<default-owner-id>",
    secrets_api_key="<your-secrets-api-key>",
    secrets_url="https://<your-secret-manager-host>",   # or AGENTIC_CONTEXT_SECRET_SERVER_URL
)
```

Auth precedence: a shared `auth=` instance, then inline credentials (**all four** of `private_key`, `consumer_key`, `subject`, `login_url` must be non-empty), then `secrets_api_key`. The constructor calls `auth.get_token()` once, synchronously, to discover the Beacon `base_url` (a network call; construct at startup, not inside a request handler). There is no pre-minted-token path and no `base_url` / `api_key` parameter.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `agent_id` | `str` | ✅ | — | Must match a registered agent in Beacon. Positional-or-keyword; every other parameter is keyword-only. |
| `owner_id` | `str \| None` | ✅ | `None` | **Required — an empty/missing value raises `ValueError`.** Fallback owner for calls that omit their own; also the transformer's default `participant_id`. See [§6.3](#63-one-adapter-many-sessions-ownership-model). |
| `auth` | `ContextAuth \| None` | — | `None` | A shared auth instance (annotated `ContextAuth`; it is used as-is with no `isinstance` check). Takes precedence over inline credentials and `secrets_api_key`. |
| `private_key` | `str` | — | `""` | PEM RSA private key. |
| `consumer_key` | `str` | — | `""` | Connected App consumer key. |
| `subject` | `str` | — | `""` | Salesforce username for the JWT bearer flow. |
| `login_url` | `str` | — | `""` | Salesforce login host, no scheme. |
| `secrets_api_key` | `str` | — | `""` | Secrets-server API key; the adapter builds an `ApiKeyAuth`. Used only when neither `auth` nor the full inline credential set is given. |
| `secrets_url` | `str` | — | `""` | Secrets-server base URL for `secrets_api_key`. Empty → `ApiKeyAuth` reads the `AGENTIC_CONTEXT_SECRET_SERVER_URL` env var (see [§2](#environment-variables)). |
| `dataspace` | `str \| None` | — | `None` → `CDP_DATASPACE` env var → `"default"` | Beacon dataspace name. An explicit value wins over the env var. |
| `owner_type` | `str \| None` | — | `None` → the auth class's default | Owner type sent with retrieval requests. When `None`, uses `auth.default_owner_type` (`"individual"` for `ContextAuth` / `ApiKeyAuth`, `"employee"` for `UserContextAuth`; `"individual"` if the auth object has none). |
| `use_async` | `bool` | — | `True` | `True` → async clients (required for `instructions_provider`, `run_hooks`, and the pack tool). `False` → sync clients; see [§6.8](#68-sync-vs-async-use_async). |
| `context_mode` | `str` | — | `"auto"` | `"auto"` or `"manual"` ([§6.4.2](#642-memory-modes-auto--manual)). Input is lower-cased and `_`/space become `-`, so `" AUTO "` works. Anything else raises `ValueError`. |
| `agent_participant_variable` | `dict[str, Any] \| None` | — | `None` | **Full override** of Beacon's `AiAgentParticipant` block. Default: `{"AiAgentType": "OpenAI", "AiAgentVersionApiName": "v1", "AiAgentTemplateApiName": "default", "AiAgentApiName": <agent_id>}`. |

**Validation errors (all `ValueError`):**

| Condition | Message |
|-----------|---------|
| `owner_id` empty / `None` | `OpenAIAdapter requires owner_id. Pass the end-user id (or a stable service-account id for single-tenant deployments). There is no default — a shared adapter must not attribute transcripts to a placeholder owner.` |
| invalid `context_mode` | `context_mode must be one of ('auto', 'manual'); got '<value>'.` |
| no `auth`, incomplete inline credentials, no `secrets_api_key` | `OpenAIAdapter requires credentials. Pass auth=ContextAuth(...), inline private_key+consumer_key+subject+login_url, or secrets_api_key= to fetch them from the secrets server.` |

(`owner_id` is checked first, then `context_mode`, then credentials.)

There is **no** `http_observer`, `delivery_mode`, `conversation_id`, or construction-time `session_id` parameter on this adapter.

### 6.6 Method Reference

Class attributes / properties:

```python
adapter.agent_id        # → str
adapter.capabilities    # → Capabilities(working=True, persistent=False)
adapter.context_mode    # → "auto" | "manual"   (read-only property)
adapter.session_id      # → str | None — the most recently CREATED session (see §6.3)
```

`OpenAIAdapter` satisfies the `ContextSource` protocol (`isinstance(adapter, ContextSource)` is `True`).

#### `snapshot` *(sync)*

Normalizes a conversation into `Record`s without sending anything (the `ContextSource` contract).

```python
def snapshot(session_id: str, *, messages: list[dict] | None = None, items: list[dict] | None = None,
             owner_id: str | None = None, interaction_id: str | None = None) -> list[Record]
```

`owner_id` is stamped on `user_message` records (falls back to the constructor owner). `interaction_id` is stamped on every emitted record. Event-id rules: [§6.4.1](#641-inputs-record-mapping-and-event-ids).

#### `push`

```python
async def push(session_id: str, *, messages: list[dict] | None = None, items: list[dict] | None = None,
               owner_id: str | None = None, interaction_id: str | None = None) -> None
```

Transforms and ingests. Returns immediately (no manager is created, no seed) if both `messages` and `items` are empty. If they are non-empty but produce no records (e.g. only `system` messages), the session is still bound to its owner and the auto-mode seed still runs, but nothing is ingested. The first push for a session creates the Beacon lifecycle lazily and — under `context_mode="auto"` — does the one-time fast-context seed ([§6.4.2](#642-memory-modes-auto--manual)). Raises `ValueError` on an owner mismatch ([§6.3](#63-one-adapter-many-sessions-ownership-model)).

#### `ingest`

```python
async def ingest(session_id: str, records: list[Record], *, owner_id: str | None = None) -> None
```

Ingests pre-built `Record`s, bypassing the transformer. Empty list is a no-op. This is what the run hooks call internally.

#### `start_session` *(optional)*

```python
async def start_session(session_id: str, *, owner_id: str | None = None, title: str = "") -> dict[str, str]
```

Eagerly creates the conversation + session + participants. Returns `{"sessionId": ..., "ownerId": ..., "agentId": ...}`. A 409 (already exists) is treated as success. Not required — lifecycle is lazy.

#### `get_context`

```python
async def get_context(session_id: str | None = None, *, owner_id: str | None = None) -> dict | None
```

Fast cross-session context read. Returns the context `dict`, or `None` on any Beacon error, or if no session is known. Beacon errors never raise; an owner mismatch for a known session does (`ValueError`). `owner_id` only matters for a cold session.

#### `get_session`

```python
async def get_session(session_id: str | None = None, *, owner_id: str | None = None) -> dict | None
```

Full session data (entries, metadata). Same semantics as `get_context`.

#### `list_sessions`

```python
async def list_sessions(owner_id: str | None = None) -> list[dict]
```

Lists conversations for an owner (default: constructor owner) and this `agent_id`. Returns `[]` on any error (logged as a warning). Not session-bound — always pass `owner_id` in a multi-user app.

#### `context_mode` / `set_context_mode`

```python
adapter.context_mode                 # property → "auto" | "manual"
def set_context_mode(mode: str) -> None   # same normalization/validation as the constructor
```

#### `instructions_provider`

```python
def instructions_provider(base: str, session_id: str, owner_id: str | None = None)
# → async callable (run_context, agent) -> str, for Agent(instructions=...)
```

Returns the dynamic-instructions callable the Agents SDK re-invokes on every turn. Behaviour:

1. **At call time:** raises `RuntimeError("OpenAIAdapter.instructions_provider() requires use_async=True; the Agents SDK instructions callable is async-only.")` if `use_async=False`; binds the session↔owner (so an owner mismatch raises `ValueError` here).
2. **First turn of each `session_id`** (adapter-wide gate): fetches the agent's context-pack catalog from Beacon and publishes it ([§6.7](#67-context-packs)). The gate is marked *before* the fetch, so a failed fetch is **not retried** for that session (a warning is logged).
3. **Every turn** it returns the parts joined with a blank line (`"\n\n"`), in this order:
   `base` → context-pack instruction → `[Available context packs]` block.
   - `base` is omitted if empty.
   - The **instruction** is included only when at least one pack id is currently allowed, and skipped if `base` already contains it (server text or the bundled `CONTEXT_PACK_INSTRUCTION`) — you never need to prepend it yourself.
   - The **catalog block** is always appended once the first turn has run, even when it is `[Available context packs]\n{}`.

You do not call `get_context_pack_instruction` yourself when using this.

#### `get_context_pack_instruction`

```python
async def get_context_pack_instruction(session_id: str) -> str
```

Returns the context-pack instruction pinned for `session_id`. First call per session fetches the server-published prompt (`GET /api/v1/agentic/prompts/system/context_pack_instruction`) and pins it for that session; a miss or failure pins the bundled `CONTEXT_PACK_INSTRUCTION` instead. After a fetch failure, other new sessions skip the fetch and use the bundled text for ~10 s. Only needed if you build `instructions` yourself.

#### `run_hooks`

```python
def run_hooks(session_id: str, owner_id: str | None = None, *,
              user_input: str | list[dict] | None = None, next_hooks: Any = None) -> RunHooks
```

Returns an Agents SDK `RunHooks` subclass instance bound to one `(session_id, owner_id)`; pass it as `Runner.run(..., hooks=...)`. **Create a fresh one per `Runner.run`** — one hooks instance is one turn: it mints one `interaction_id` (`turn-<session_id>-<12 hex>`, exposed as `hooks.interaction_id`), buffers every record of the run, and ingests them in **one** `/entries` POST. `user_input` (a string, or Responses-style input items) is recorded as the turn's `user_message`. `owner_id` falls back to the constructor owner. Raises `RuntimeError` under `use_async=False`; raises `ImportError` if `openai-agents` is missing or older than 0.3.

| Lifecycle event | Effect |
|-----------------|--------|
| `on_llm_end` | `response.output` items → `agent_response` / `agent_reasoning` / `tool_call` records, buffered and stamped with the run's `interaction_id`. Items with a real id are de-duplicated within the hooks instance. |
| `on_tool_end` | a buffered `tool_result` record. A non-`str` result is JSON-serialized (`default=str`), falling back to `str()`. If the SDK supplies no `tool_call_id`, a unique `tool-<uuid>` id is minted. |
| `on_agent_start` | under `context_mode="auto"`, seeds fast context once per session (best-effort) |
| `on_agent_end` | ingests the buffered turn in one POST (`flush()`) |
| `on_handoff`, `on_tool_start` | pass-through to `next_hooks` only |

- **Pass the user message as `user_input=`.** The hooks never see it otherwise; don't also `push` it, or it is written twice under a different `interaction_id`.
- **`await hooks.flush()` in a `finally`.** `on_agent_end` does not fire when the run raises (including max turns), trips an input guardrail, stops for approval or is cancelled mid-stream; `flush()` ingests whatever was captured. It is idempotent. Until it runs the turn is held in memory; `adapter.close()` posts any hooks still holding records and logs a warning for each. An output guardrail runs after `on_agent_end`, so a blocked output has already been ingested.
- **`Runner.run_streamed`:** it returns before the run executes, so put the `stream_events()` loop inside the `try` — flushing straight after `run_streamed` would post the user message alone and split the turn into two POSTs:
  ```python
  hooks = adapter.run_hooks(sid, owner_id, user_input=text)
  result = Runner.run_streamed(agent, text, hooks=hooks)
  try:
      async for event in result.stream_events():
          ...
  finally:
      await hooks.flush()
  ```
- **Resuming after an approval interruption:** create fresh hooks **without** `user_input` for `Runner.run(agent, state, hooks=...)` — the first run already recorded the user message. The resumed run is its own turn (its own `interaction_id`).
- **Fast-context seed (`context_mode="auto"`):** `on_agent_start` awaits the once-per-session `get_context` seed before the first model call, so the first turn's first model call waits on one Beacon round trip. A failed seed is retried on the next agent start.
- **Don't `close()` the adapter while runs are in flight.** Their buffered records are posted at `close()`, and anything they capture afterwards is lost (logged as a warning).
- **Best-effort:** an ingest failure is logged as a warning and never breaks the run.
- `next_hooks=` composes with your own `RunHooks`: the adapter ingests first (swallowing its own errors), then calls yours for every event above.
- Unlike `instructions_provider`, `run_hooks()` does not bind the session at call time; an owner mismatch is only seen (and logged, with the batch dropped) when records are ingested.

#### `make_context_retrieval_tool` / `make_get_context_pack_tool` / `default_tools`

```python
def make_context_retrieval_tool(session_id: str | None = None, owner_id: str | None = None)  # → FunctionTool
def make_get_context_pack_tool(valid_pack_ids: list[str], session_id: str | None = None,
                               owner_id: str | None = None, *,
                               param_defaults: dict[str, Any] | None = None,
                               timeout: float | None = 60.0)                                    # → FunctionTool
def default_tools(session_id: str | None = None, owner_id: str | None = None, *,
                  valid_pack_ids: list[str] | None = None) -> list                              # → [recall, pack]
```

All three require `openai-agents` (`ImportError` otherwise) and return Agents SDK `FunctionTool`s. `default_tools()` returns `[recall_customer_context, get_context_pack]` in that order. **Bind a `session_id`**: a tool built without one returns `not_ready` for every call (no fallback to "last session", which would leak between concurrent sessions).

**`recall_customer_context(topic: str = "")`** — returns the owner's cross-session context as `json.dumps(payload, indent=2)` text (`{"userContext": [...]}`; `{"userContext": []}` if there is none). `topic` is logged only; the Beacon read is scoped to owner + agent. Result is served from the `push()` seed cache when present, else a live `get_context`. `not_ready` is returned when no session is bound or the session is bound to a different owner.

**`get_context_pack(context_pack_id: str, params: str = "", context_pack_version: int | None = None)`** — see [§6.7](#67-context-packs).

`timeout` is the per-call HTTP timeout for the pack read; the default is `agentic_context.CONTEXT_PACK_TIMEOUT_S` (60.0 s). `param_defaults` are fixed pack `params` merged into every call; a key the model also sends overrides the default, and blank/`None` values are dropped after the merge (so a blank model value removes the default too).

#### `close`

```python
async def close() -> None
```

Closes every session manager (async mode: drains pending background deliveries, 30 s budget; errors closing one session are logged, not raised), clears the session map and the per-session prompt cache, then closes both clients. Call once on app shutdown. After `close()`, don't reuse the adapter.

### 6.7 Context Packs

A context pack is a named, server-defined bundle of structured data needs the agent can request on demand (as opposed to the profile read done by `recall_customer_context`). Wiring is three pieces that share one piece of adapter state:

1. `instructions_provider(...)` — tells the model which packs exist.
2. `get_context_pack` tool (`make_get_context_pack_tool` / `default_tools(valid_pack_ids=...)`) — lets it read one.
3. The **allowed set** — the pack ids the tool accepts. It is written together with the catalog block through a single code path, so the model is never shown an id the tool would reject and the tool never accepts an id the model wasn't shown.

**Instruction ordering** in the string `instructions_provider` returns: `base` → context-pack instruction → `[Available context packs]\n<json>`.

**Catalog block shape** — `{ "<pack id>": {"description": str, "dataNeeds": [{"name", "dataNeedDescription", "parameters"}], "version": int?} }`, JSON-encoded after the `[Available context packs]` marker. `version` appears only for packs whose server `version` is an integer (`null` and booleans are skipped).

**`valid_pack_ids` seeding and fallback semantics**

- `make_get_context_pack_tool(valid_pack_ids, ...)` (and `default_tools(..., valid_pack_ids=...)`) **eagerly publishes** the ids as the allowed set if none is set yet, so a tools-only integration (no `instructions_provider`) already works. Those entries have empty `description` / `dataNeeds` and no `version`.
- A **successful, non-empty server catalog** (fetched by `instructions_provider` on a session's first turn) **replaces** the allowed set and the block. `valid_pack_ids` is therefore a fallback, **not a filter** — server pack ids you didn't list become callable.
- An **empty** server catalog, or a **failed** fetch, degrades to the pinned ids (a failed fetch leaves an already-populated set untouched). With no pinned ids the block is `[Available context packs]\n{}` and the tool rejects every id.
- The catalog and allowed set are **adapter-wide**, not per session: the latest successful fetch wins for all sessions.
- **One pinned set per adapter.** Calling `make_get_context_pack_tool` / `default_tools` again with a *different* `valid_pack_ids` list (list equality — order matters; an omitted list counts as `[]`) raises `RuntimeError("make_get_context_pack_tool was already called on this adapter with pinned ids [...]; a second call with [...] would race on the shared catalog state. Build a second OpenAIAdapter instance for a second pack tool.")`. (The check only applies once a non-empty list has been pinned; a first call with `[]`/omitted does not lock anything.) In a per-request/per-session setup, pass the **same** `valid_pack_ids` every time.

**Tool call flow (`get_context_pack`)** — the tool never raises on Beacon failures; it returns a JSON string `{"status": ..., ...}` (indent 2):

| `status` | When | Extra fields |
|----------|------|--------------|
| `invalid_version` | `context_pack_version` is not an integer (bool counts as invalid) | `context_pack_id`, `message` |
| `invalid_pack` | `context_pack_id` not in the allowed set (no server call is made) | `context_pack_id`, `valid_pack_ids` (sorted), `message` |
| `not_ready` | adapter is `use_async=False`; no session bound to the tool; request raised (timeout, endpoint not deployed, auth error…) | `context_pack_id`, `message`, `detail` (e.g. `"request failed: <ExceptionTypeName>"` — only the exception type, never its text) |
| `version_expired` | the pack was republished since the catalog was read | `context_pack_id`, `requested_version`, `latest_version`, `latest` (new pack description), `message` |
| `ok` | success | `context_pack_id`, `context` (the Beacon payload) |

- **Owner mismatch is not caught by the pack tool:** if the tool is bound to `(session, owner)` and that session is already bound to a different owner, the `ValueError` from [§6.3](#63-one-adapter-many-sessions-ownership-model) propagates out of the tool to the Agents SDK (unlike `recall_customer_context`, which returns `not_ready`). Always bind the tool with the same owner as the session.
- **Owner and params:** the owner is taken from the session's bound owner, never from the model; an `owner_id` key in the model's `params` is stripped. `params` is a JSON-object **string**; blank values are dropped.
- **Versioning auto-fill:** if the model omits `context_pack_version` and the cached catalog has an integer version for that pack, the tool fills it in before calling Beacon. On `version_expired`, the cached version is updated to `latest_version` (or dropped if Beacon reports none) and the cached catalog block is patched, so the retry — and later `instructions_provider` turns — use the fresh version. The patched block is adapter-wide.
- Packs published by the server are the source of truth; pack ids in this guide's examples are placeholders.

### 6.8 Sync vs async (`use_async`)

`use_async` selects which clients and session managers the adapter builds; it does **not** change which methods are coroutines.

| | `use_async=True` (default) | `use_async=False` |
|---|---|---|
| Clients / managers | `AsyncIngestionClient`, `AsyncRetrievalClient`, `AsyncSessionManager` | `IngestionClient`, `RetrievalClient`, `SessionManager` |
| `push`, `ingest`, `get_context`, `get_session`, `list_sessions`, `start_session`, `close`, `get_context_pack_instruction` | coroutines — `await` them | **still coroutines — `await` them** (the sync call inside blocks the event loop, except `get_context_pack_instruction`, which runs it in a worker thread; ingestion blocks until the POST returns, with retries) |
| `snapshot`, `set_context_mode`, `make_*_tool`, `default_tools` | plain calls | plain calls |
| `instructions_provider`, `run_hooks` | work | **raise `RuntimeError`** |
| `get_context_pack` tool | works | returns `not_ready` (`"pack read requires async mode (use_async=True)"`) |
| `recall_customer_context` tool | works | works |

The Agents SDK tool loop is async-native, so use sync mode only for the pure ingestion path (e.g. `await adapter.push(..., items=result.to_input_list())` from a batch job). For the full integration use the default.

### 6.9 End-to-End Example: shared adapter in a request handler

```python
import os
from pathlib import Path

from agents import Agent, Runner
from agentic_context_openai import OpenAIAdapter

AGENT_ID = "support-bot"
PACK_IDS = ["<your-pack-id>"]           # same list on every call (one pinned set per adapter)

# Built ONCE at startup (the constructor does a blocking token exchange).
adapter = OpenAIAdapter(
    agent_id=AGENT_ID,
    owner_id="<default-owner-id>",       # fallback only; real owners are passed per call
    private_key=Path(os.environ["BEACON_PRIVATE_KEY_FILE"]).read_text(),
    consumer_key=os.environ["BEACON_CONSUMER_KEY"],
    subject=os.environ["BEACON_SUBJECT"],
    login_url=os.environ.get("BEACON_LOGIN_URL", "login.salesforce.com"),
    dataspace=os.environ.get("BEACON_DATASPACE", "default"),
)

async def send_message(conversation_id: str, owner_id: str, text: str) -> str:
    # The owner must come from your authenticated session, never from the model.
    agent = Agent(
        name="support",
        instructions=adapter.instructions_provider(
            base="You are a helpful support agent.",
            session_id=conversation_id,
            owner_id=owner_id,
        ),
        tools=adapter.default_tools(conversation_id, owner_id, valid_pack_ids=PACK_IDS),
    )
    hooks = adapter.run_hooks(conversation_id, owner_id, user_input=text)
    try:
        result = await Runner.run(agent, text, hooks=hooks)
    finally:
        await hooks.flush()
    return result.final_output

async def shutdown() -> None:
    await adapter.close()                # drains pending background deliveries
```

Each request builds a fresh `Agent`; the adapter's catalog fetch and instruction pinning are gated per `session_id`, so repeating `instructions_provider(...)` for the same conversation does not re-fetch. Prior-turn history for the model itself is your app's responsibility (for example pass an Agents SDK `session=` to `Runner.run`); Beacon ingestion is independent of it.

**Ship an already-finished transcript instead (snapshot):**

```python
result = await Runner.run(agent, "Hello")
await adapter.push("conv-123", items=result.to_input_list(), owner_id="user@example.com")
await adapter.close()
```

Push the finished transcript once. Re-running, or pushing the full transcript every turn, writes the earlier entries again: event ids are local only and not sent, so nothing is deduped ([§6.4.1](#641-inputs-record-mapping-and-event-ids)).

### 6.10 Returning Users & Cross-Session Memory

Reuse a `session_id` (with the same owner) to continue a conversation — a 409 from Beacon on re-create is treated as success. Use the same `owner_id` with a new `session_id` to start a fresh conversation that still recalls the user: `recall_customer_context` reads cross-session context scoped to `owner_id` + `agent_id`. A `session_id` cannot be reused with a different owner in the same adapter ([§6.3](#63-one-adapter-many-sessions-ownership-model)).

### 6.11 Go-Live Checklist

- [ ] **One** adapter instance for the app (not one per conversation), constructed at startup
- [ ] `agent_id` matches a registered agent in Beacon
- [ ] Constructor `owner_id` supplied (required) — and the real end-user `owner_id` passed on every `push` / `instructions_provider` / `default_tools` / `run_hooks` / `list_sessions` call
- [ ] The user message is passed as `run_hooks(..., user_input=...)` (not pushed separately)
- [ ] `run_hooks(...)` created fresh per `Runner.run`, with `await hooks.flush()` in a `finally` — one POST and one `interaction_id` per turn
- [ ] `openai-agents >= 0.3` installed (tools and `run_hooks` need it)
- [ ] `use_async=True` (default) — `instructions_provider`, `run_hooks`, and the pack tool need it
- [ ] `valid_pack_ids` is the **same** list on every `default_tools` / `make_get_context_pack_tool` call
- [ ] Auth configured (`auth=`, inline credentials, or `secrets_api_key`) — token refresh is automatic
- [ ] `await adapter.close()` wired to shutdown so background deliveries drain
- [ ] `get_context()` returns `userContext` entries (confirms the agent config exists and prior data is ingested)

---

## 7. Microsoft Foundry Adapter

The Foundry adapter targets the **Microsoft Agent Framework** (`agent_framework`, installed with `agent-framework-foundry`). It ingests every turn as Beacon records and gives the agent two function tools: `recall_customer_context` and `get_context_pack`.

### 7.1 Quick Start (the 90% case)

**Install.** The adapter ships inside the `agent-context-sdk` wheel (package `agentic_context_foundry`). It does **not** install the framework SDK for you:

```bash
pip install agent-context-sdk
pip install agent-framework-foundry     # Microsoft Agent Framework + Foundry client; Python 3.10+
```

**Wire it:**

```python
from pathlib import Path

from agent_framework import Agent, AgentSession
from agent_framework_foundry import FoundryChatClient
from agentic_context import ContextAuth, FoundryAdapter

auth = ContextAuth(
    private_key=Path("server.key").read_text(),
    consumer_key="3MVG9...",
    subject="user@example.com",
    login_url="acme.my.salesforce.com",
)
adapter = FoundryAdapter(
    agent_id="<your-registered-agent-id>",   # must match a registered agent config in Beacon
    owner_id="alice@example.com",            # REQUIRED: the end user (see 7.6)
    auth=auth,
)

agent = Agent(FoundryChatClient(), instructions=MY_PROMPT)   # MY_PROMPT: your own system prompt string
session = AgentSession(session_id="ses-123")

records = await adapter.run(agent, "Hi, I have a billing question", session=session)
reply = next(r.content for r in reversed(records) if r.event_type == "agent_response")
await adapter.close()   # on shutdown
```

`from agentic_context_foundry import FoundryAdapter` works too; both import paths return the same class.

Things to know up front:

- `run()` and `stream()` do **not** return the model's reply object. They return/yield `Record`s. The reply text is the `content` of the `agent_response` record.
- `owner_id` is required; there is no default and no placeholder (a missing value raises `ValueError`).
- There is no `context_mode` kwarg. The adapter always seeds fast context once per `session_id`, always registers both tools, and always ingests at the end of the turn.
- `FoundryAdapter(...)` is a **synchronous** constructor that calls `auth.get_token()` immediately, so the JWT exchange (and, for `ApiKeyAuth`, the secret-server fetch) happens during construction and any auth failure is raised there.

Import and `isinstance` facts: `FoundryAdapter.capabilities == Capabilities(working=True, persistent=False)`; `FoundryAdapter` also satisfies the core `ContextSource` shape (`agent_id`, `capabilities`, `snapshot(session_id)`).

---

### 7.2 Minimum Requirements

| Requirement | Detail |
|---|---|
| Python | 3.10+ (declared by `adapters/foundry/pyproject.toml`) |
| Framework SDK | `agent_framework` (`pip install agent-framework-foundry`). If it is missing, constructing `FoundryAdapter` emits a `UserWarning` that names the install command; `run`, `stream`, `snapshot`, `context_provider`, `make_get_context_tool`, `make_get_context_pack_tool` and `default_tools` then raise `ImportError` (message starts `FoundryAdapter.<method>() requires the Microsoft Foundry SDK, which is not installed.`). `get_context`, `get_context_pack_list` and `start_session` do not need the SDK |
| `agent_id` | Must match an agent config registered in Beacon |
| `owner_id` | The end-user identity (email / user id) or a stable service-account id; required |
| Credentials | One of: `auth=`, the four inline `ContextAuth` fields, or `secrets_api_key=` (see 7.4) |
| Session object | A MAF `AgentSession` with a non-empty `session_id` (`run`/`stream` raise `ValueError("AgentSession.session_id is required")` otherwise) |

---

### 7.3 How It Works (under the hood)

The adapter integrates through MAF's **context provider** and **middleware** extension points. It does not wrap or replace your `Agent`.

**Two wiring modes** (both produce identical records):

| Mode | How | Who ingests |
|---|---|---|
| **Implicit** (default) | `await adapter.run(agent, ...)` / `adapter.stream(agent, ...)` with a plain `Agent` | `stream()` itself, at the end of the turn |
| **Attached provider** | `Agent(client, instructions=..., context_providers=[adapter.context_provider()])`, then call `adapter.run`/`adapter.stream` **or** `agent.run(...)` directly | The provider's `after_run`, once per turn (`after_run_once_per_turn = True`) |

Attached-provider example:

```python
agent = Agent(
    FoundryChatClient(),
    instructions=MY_PROMPT,
    context_providers=[adapter.context_provider()],
)
```

In implicit mode `stream()` builds a `FoundryContextProvider` itself, runs its `before_run`, and passes the resulting tools, middleware and merged instructions to `agent.run(...)` for that call. The merged instructions are your agent's own instructions plus the adapter's additions, joined with a blank line, passed as `options={"instructions": ...}`. If the agent already has a `FoundryContextProvider` for **this** adapter in `agent.context_providers`, `stream()` detects it and does not add a second one.

**What `before_run` does on every turn:**

1. Resolves the owner and gets or creates the session's `AsyncSessionManager` (raises `ValueError` if the `session_id` is already bound to a different owner; see 7.6).
2. Remembers the `AgentSession` (this is what `snapshot()` reads later).
3. Allocates `interaction_id = f"turn-{session_id}-{n}"`, where `n` is a per-adapter-instance, in-memory counter starting at 0 for each `session_id`.
4. Converts the incoming user messages to `user_message` records.
5. Registers the chat and function middleware that capture the rest of the turn.
6. Adds the two tools (`recall_customer_context`, `get_context_pack`) to the run.

**What is seeded, and when:**

| Item | When | How |
|---|---|---|
| FAST profile context (`[Cross-session context]` block with one `- <entry>` line per `userContext` entry) | First turn for a `session_id` on this adapter instance only (tracked in memory) | `get_context()` is called and, if it yields any entries, they are added to that run's instructions. Not re-added on later turns; the model reaches for more through `recall_customer_context` |
| Context-pack catalog (the `[Available context packs]` block) | Fetched on the first turn for a `session_id` on this adapter instance; the block is added to the run's instructions on **every** turn | `get_context_pack_list` from Beacon for this `agent_id`. If the fetch fails, the previously cached catalog is kept (initially an empty `{}` block and no valid pack ids). The cache is shared by all sessions of the adapter instance, so the most recent session start wins |
| Context-pack instruction | Every turn, **only when the catalog has at least one pack** and the instruction is not already present in your agent's own instructions | Server-published prompt (`GET /api/v1/agentic/prompts/system/context_pack_instruction`) pinned per session on first use; a miss or error pins the bundled `CONTEXT_PACK_INSTRUCTION` for that session. Placed before the catalog block |

**What is ingested, and when.** Exactly one `add_entries` flow per turn: after the turn, the adapter calls `manager.ingest(records)` (non-blocking, with core retries) and then `manager.flush()`, which waits (up to 20 s, best effort) for the background delivery to finish. The manager stays open for the next turn. `close()` is the final drain.

| Record | Source | `category` | Notes |
|---|---|---|---|
| `user_message` | Input messages with `text` content | `Message` | `participant_id` = the turn's owner |
| `agent_response` | Assistant `text` content | `Message` | `participant_id` = `agent_id` |
| `agent_reasoning` | `text_reasoning` content (and, for non-streaming chat calls, assistant text that precedes tool calls) | `Trace` | `variable={"OutputValueText": ...}` |
| `tool_call` | Function middleware, before the tool runs | `Trace` | `variable={"Name": ..., "InputValueText": <JSON of arguments>}` |
| `tool_result` | Function middleware, after the tool runs | `Trace` | `variable={"Name": ..., "OutputValueText": ...}`; `event_id` = `<call_id>-result` |

Records are de-duplicated by `event_id` within a turn (first one wins) and unknown content types are skipped. `event_id` is `<message_id>-<index>` (or `msg-<session_id>-<index>` when MAF gives no `message_id`, with `-<n>` appended for the n-th content in a message); tool calls use the MAF `call_id`.

In streaming turns, `tool_call` records are emitted only once the tool arguments are a complete JSON object, so a partial argument delta is never ingested.

If the model call fails mid-turn in implicit mode, the records collected so far are still ingested and the original error is re-raised. If ingestion itself fails and there was no earlier error, that ingestion error is raised.

---

### 7.4 Constructor

```python
from agentic_context import ContextAuth, FoundryAdapter

adapter = FoundryAdapter(
    agent_id="<your-registered-agent-id>",
    owner_id="alice@example.com",
    auth=ContextAuth(private_key=..., consumer_key=..., subject=..., login_url=...),
    dataspace="default",
)
```

Everything after `agent_id` is keyword-only.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `agent_id` | `str` | yes | - | Must match a registered agent config in Beacon |
| `owner_id` | `str \| None` | yes | `None` | Default owner for every session this adapter creates. Empty/`None` raises `ValueError("FoundryAdapter requires owner_id. ...")`. Overridable per call (7.6) |
| `auth` | `ContextAuth \| ApiKeyAuth \| None` | yes\* | `None` | Auth object. Takes precedence over the inline fields and `secrets_api_key` |
| `private_key` / `consumer_key` / `subject` / `login_url` | `str` | yes\* | `""` | The four `ContextAuth` credentials inline. Used only when **all four** are non-empty and `auth` is not given |
| `secrets_api_key` | `str` | yes\* | `""` | API key; the adapter builds `ApiKeyAuth(secrets_api_key, secret_server_url=secrets_url or None)` |
| `secrets_url` | `str` | - | `""` | Secret-server URL for `secrets_api_key`. If empty, `ApiKeyAuth` reads the `AGENTIC_CONTEXT_SECRET_SERVER_URL` environment variable; there is no built-in host |
| `dataspace` | `str` | - | `"default"` | Beacon dataspace name. **Not read from any environment variable**; pass it explicitly if your org does not use `"default"` |
| `owner_type` | `str \| None` | - | `None` | Owner type sent with retrieval and lifecycle requests. If `None`, taken from the auth object's `default_owner_type` (`"individual"` for `ContextAuth` and `ApiKeyAuth`; `"individual"` if the auth object has none). A non-`None` value always wins |
| `agent_participant_variable` | `dict[str, Any] \| None` | - | `None` | **Full override** of Beacon's `AiAgentParticipant` block. Default: `{"AiAgentType": "MicrosoftAgentFramework", "AiAgentVersionApiName": "v1", "AiAgentTemplateApiName": "default", "AiAgentApiName": <agent_id>}` |

\* Exactly one credential route must resolve, otherwise `ValueError("FoundryAdapter requires credentials. ...")`. Precedence: `auth` > four inline fields > `secrets_api_key`.

Validation order: `owner_id` is checked first, then the missing-SDK warning is evaluated, then credentials.

Not available on this adapter: `context_mode`, `http_observer`, `callbacks`, `set_owner`, `hooks()`/`apply()`, `register_shutdown()`.

The adapter creates one `AsyncIngestionClient` and one `AsyncRetrievalClient` (both with the given `dataspace`) and one `AsyncSessionManager` per `session_id`, created lazily and kept until `close()`. One adapter can serve many sessions and many owners.

---

### 7.5 Method Reference

All methods below are `async def` unless marked **sync**.

---

#### `run`

Drives one turn and returns all records of that turn. It is `stream()` collected into a list; it always calls `agent.run(..., stream=True)` underneath.

```python
async def run(
    agent: Agent,
    messages: AgentRunInputs,        # str, Message, or a list of them (anything MAF's normalize_messages accepts)
    *,
    session: AgentSession,
    owner_id: str | None = None,
) -> list[Record]
```

- Raises `RuntimeError("FoundryAdapter is closed")` after `close()`, `ValueError` for a session without `session_id` or an owner conflict, `ImportError` if the SDK is missing.
- Returned records are in the order: input `user_message`s, streamed tool records, then the final response messages, then any records only the middleware saw.

---

#### `stream`

Same turn as `run`, as an async generator of `Record`s (records are yielded as they are produced). Ingestion happens once, after the generator is exhausted (implicit mode) or in the provider's `after_run` (attached mode).

```python
async def stream(
    agent: Agent,
    messages: AgentRunInputs,
    *,
    session: AgentSession,
    owner_id: str | None = None,
) -> AsyncIterator[Record]
```

```python
async for record in adapter.stream(agent, "What did we discuss last time?", session=session):
    if record.event_type == "agent_response":
        print(record.content)
```

Records are yielded per item; text is **not** token-streamed. The assistant text arrives as a single `agent_response` record after the model finishes.

---

#### `context_provider` (sync)

```python
def context_provider(*, owner_id: str | None = None) -> FoundryContextProvider
```

Returns a new `FoundryContextProvider` (a MAF `ContextProvider` with `source_id="agentic_context"`) bound to this adapter. Pass it in `Agent(..., context_providers=[...])` (see 7.3). `owner_id` here becomes that provider's default owner (overridden per call by `stream(owner_id=...)`). Requires the SDK. `FoundryContextProvider` is also importable as `from agentic_context_foundry import FoundryContextProvider`.

---

#### `default_tools` (sync)

```python
def default_tools(
    session_id: str,
    owner_id: str | None = None,
    *,
    valid_pack_ids: list[str] | None = None,
) -> list[Any]
```

Returns `[make_get_context_tool(session_id, owner_id), make_get_context_pack_tool(session_id, owner_id, valid_pack_ids=valid_pack_ids)]` (tool names `recall_customer_context`, `get_context_pack`). `run`/`stream`/the provider already add these every turn, so you only call this when you wire tools by hand. It does not accept `param_defaults` or `timeout`.

---

#### `make_get_context_tool` (sync)

```python
def make_get_context_tool(session_id: str, owner_id: str | None = None) -> FunctionTool
```

Builds the MAF tool `recall_customer_context(topic: str = "")` (`approval_mode="never_require"`) bound to `session_id` and `owner_id`. The model never supplies the owner. The tool returns `adapter.recall_customer_context(...)`.

---

#### `make_get_context_pack_tool` (sync)

```python
def make_get_context_pack_tool(
    session_id: str,
    owner_id: str | None = None,
    *,
    valid_pack_ids: list[str] | None = None,
    param_defaults: dict[str, Any] | None = None,
    timeout: float | None = CONTEXT_PACK_TIMEOUT_S,   # 60.0 seconds
) -> FunctionTool
```

Builds the MAF tool `get_context_pack(context_pack_id: str = "", params: Any = "", context_pack_version: int | None = None)` (`approval_mode="never_require"`). The model supplies `context_pack_id`, `params` and `context_pack_version`; everything else is fixed by the factory arguments.

| Argument | Behaviour |
|---|---|
| `valid_pack_ids` | **Replaces** the catalog for this tool. When it is not `None` (including `[]`), only these ids are accepted and the adapter's fetched catalog is ignored for validation. When `None`, the adapter's live cached catalog is used. This differs from the other adapters; see 7.7 |
| `param_defaults` | Fixed pack `params` merged into every call; any key the model passes overrides the default, and a blank value from the model removes that key entirely (the default is not kept) |
| `timeout` | HTTP budget in seconds for the pack request. Default `CONTEXT_PACK_TIMEOUT_S` (60 s); `None` is passed to the retrieval client, whose per-call `timeout` default is `None` |

---

#### `get_context`

```python
async def get_context(session_id: str | None = None, *, owner_id: str | None = None) -> dict | None
```

FAST profile read for the session's owner. Best-effort: returns `None` when `session_id` is falsy, when the session cannot be opened (for example an owner conflict), or when the read fails; it never raises. A non-`None` result is cached internally for the session. The returned dict is Beacon's payload; the profile entries are in `userContext`, each with an `entry` string.

---

#### `recall_customer_context`

```python
async def recall_customer_context(
    *, session_id: str | None = None, owner_id: str | None = None, topic: str = "",
) -> dict
```

The body of the `recall_customer_context` tool. Calls `get_context()` and returns the payload, or `{"userContext": []}` when there is nothing. Never raises. `topic` is only written to the log; it is not sent to Beacon.

---

#### `get_context_pack_list`

```python
async def get_context_pack_list() -> dict | None
```

Fetches this agent's context-pack catalog from Beacon (`contextPacks` list). Returns `None` on failure. It **does not** update the adapter's cached catalog or valid ids; only the first turn of a session does that.

---

#### `get_context_pack`

```python
async def get_context_pack(
    context_pack_id: str,
    *,
    session_id: str,
    owner_id: str | None = None,
    params: Any = None,
    param_defaults: dict[str, Any] | None = None,
    timeout: float | None = CONTEXT_PACK_TIMEOUT_S,
    valid_pack_ids: list[str] | None = None,
    context_pack_version: int | None = None,
) -> dict
```

The body of the `get_context_pack` tool; you can also call it directly. It never raises for retrieval problems; it returns a dict with a `status`:

| `status` | Meaning | Extra keys |
|---|---|---|
| `"ok"` | Pack retrieved | `context_pack_id`, `context` (the Beacon payload) |
| `"invalid_pack"` | `context_pack_id` is not in the allowed set | `context_pack_id`, `valid_pack_ids` (sorted), `message` |
| `"version_expired"` | The pack was republished since the catalog the agent saw | `context_pack_id`, `requested_version`, `latest_version`, `latest`, `message`. The cached catalog is patched so the next call uses the new version |
| `"not_ready"` | Session could not be opened, or the request failed | `context_pack_id`, `message`, `detail` |

Details: `params` is a dict or a JSON-object string; blank values are dropped, and an `owner_id` key in `params` is removed (the owner always comes from the session). `param_defaults` apply first, `params` override them, and blank values are dropped after the merge (so a blank model value removes the default too). If `context_pack_version` is `None` and the cached catalog has an integer version for the pack, that version is sent automatically.

---

#### `get_context_pack_instruction`

```python
async def get_context_pack_instruction(session_id: str) -> str
```

Returns the context-pack instruction pinned for `session_id` (server-published text, or the bundled `CONTEXT_PACK_INSTRUCTION` fallback). `run`/`stream` already add it for you, so call it only if you build the prompt yourself. Uses the retrieval client; on a miss or error it pins and returns the bundled text.

---

#### `start_session`

```python
async def start_session(
    session_id: str, *, owner_id: str | None = None, title: str = "",
) -> dict[str, str]
```

Optional eager lifecycle: creates the Beacon conversation, session and agent participant now instead of lazily on the first ingest. `run`/`stream` do not require it. A 409 (already exists) counts as success. Returns `{"sessionId": ..., "ownerId": ..., "agentId": ...}`. Raises `IngestionError` for any other non-2xx failure, and `RuntimeError`/`ValueError` for a closed adapter or an owner conflict.

---

#### `snapshot` (sync)

```python
def snapshot(session_id: str, *, session: AgentSession | None = None) -> list[Record]
```

Converts the conversation history stored in an `AgentSession` into records, in order. It **does not** send anything to Beacon.

- Without `session=`, it uses the `AgentSession` this adapter instance saw in a previous `run`/`stream` for that `session_id`. If there is none it raises `KeyError("no AgentSession stored for '<id>'; pass session=... to snapshot a session rehydrated from persisted storage")`.
- With `session=` (for example `AgentSession.from_dict(...)` from your own storage) it reads that object and does not need a prior run.
- History is read from `session.state["in_memory"]["messages"]` (the default `InMemoryHistoryProvider`), falling back to `session.state["messages"]`.
- Every record has `interaction_id="turn-<session_id>-snapshot"`, and `timestamp` is the time of the `snapshot()` call (not the original message time).

---

#### `close`

```python
async def close() -> None
```

Drains each session manager (30 s budget each, see core `AsyncSessionManager.close`), then closes the ingestion and retrieval HTTP clients. Idempotent. Local cleanup only; nothing is sent to Beacon. After `close()`, `run`, `stream` and `start_session` raise `RuntimeError("FoundryAdapter is closed")`, `get_context` returns `None`, `recall_customer_context` returns `{"userContext": []}`, and `get_context_pack` returns `not_ready` (for a valid pack id; an invalid id is reported as `invalid_pack` first).

---

### 7.6 Owner Identity

`owner_id` is a constructor default and can be overridden per call. Precedence (highest first):

| Entry point | Order |
|---|---|
| `adapter.run/stream(..., owner_id=X)` | `X`, then (attached provider only) `context_provider(owner_id=...)`, then the constructor `owner_id` |
| `adapter.get_context`, `recall_customer_context`, `get_context_pack`, `start_session` | the `owner_id=` argument, then the constructor `owner_id` |
| Tools from `default_tools` / `make_*_tool` | the `owner_id` given to the factory, then the constructor `owner_id` |

A `session_id` is bound to the **first** owner that opens it on this adapter instance. Opening it again with a different owner raises `ValueError("session_id '<id>' is bound to owner '<a>'; got '<b>'.")` from `run`/`stream`/`start_session`; the best-effort methods swallow it (`get_context` returns `None`, `recall_customer_context` returns `{"userContext": []}`, and `get_context_pack` returns `not_ready` for a valid pack id; an invalid id is reported as `invalid_pack` first). This isolates users when one adapter is shared across many users.

User-message records are tagged with the turn's owner as `participant_id`; agent records use `agent_id`.

**Do not take the owner from model output.** Derive `owner_id` from an authenticated source (session, token) and pass it per call.

---

### 7.7 Context Packs

Catalog, instruction, and the `get_context_pack` tool are wired automatically by `run`/`stream`/the provider; nothing extra is needed (see 7.3 for when each is fetched).

- Pack ids come from Beacon's catalog for your `agent_id`; do not invent them. Examples in this guide use `"<your-pack-id>"`.
- **Valid ids for the automatic tool** are the adapter's cached catalog as of the start of the turn (refetched on the first turn of each new `session_id`), copied so a concurrent session refreshing the shared cache cannot change it mid-turn. If the catalog is empty or its fetch failed, every pack id is rejected as `invalid_pack`.
- **`valid_pack_ids` replaces the catalog** when you build the tool yourself:

```python
pack_tool = adapter.make_get_context_pack_tool(
    "ses-123", "alice@example.com",
    valid_pack_ids=["<your-pack-id>"],     # only this id is accepted; the fetched catalog is ignored
    param_defaults={"<param-name>": "<fixed-value>"},
    timeout=30.0,
)
```

  Other adapters treat a passed list as a fallback for an empty catalog; Foundry does not. Note that `run`/`stream`/an attached provider always add their own `get_context_pack` tool (built with the turn's catalog and without `param_defaults`/`timeout`); a hand-built tool applies only where you wire it yourself, for example when you call `agent.run(...)` directly on an agent without the provider.
- Versions: if the catalog lists an integer `version` for a pack and the model omits `context_pack_version`, the adapter pins it. On a `version_expired` result the catalog and pin are updated in place.

---

### 7.8 End-to-End Example: One Shared Adapter, Two Sessions

```python
import asyncio
from pathlib import Path

from agent_framework import Agent, AgentSession
from agent_framework_foundry import FoundryChatClient
from agentic_context import ContextAuth, FoundryAdapter

MY_PROMPT = "You are a helpful support agent."
AUTH = ContextAuth(
    private_key=Path("server.key").read_text(),
    consumer_key="3MVG9...",
    subject="user@example.com",
    login_url="acme.my.salesforce.com",
)

async def main() -> None:
    adapter = FoundryAdapter(
        agent_id="<your-registered-agent-id>",
        owner_id="alice@example.com",
        auth=AUTH,
        dataspace="default",
    )
    agent = Agent(FoundryChatClient(), instructions=MY_PROMPT)

    try:
        # Session 1
        s1 = AgentSession(session_id="ses-1")
        records = await adapter.run(
            agent, "I'm on the Pro plan and have a billing issue", session=s1,
        )
        print(next(r.content for r in reversed(records) if r.event_type == "agent_response"))

        # Session 2: same owner, new session_id. FAST context from session 1 is seeded on its first turn.
        s2 = AgentSession(session_id="ses-2")
        records = await adapter.run(agent, "What did we talk about last time?", session=s2)
        print(next(r.content for r in reversed(records) if r.event_type == "agent_response"))

        # Inspect stored history without sending anything
        print(len(adapter.snapshot("ses-2")))
    finally:
        await adapter.close()

asyncio.run(main())
```

`FoundryChatClient()` reads `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL` from the environment (or takes `project_endpoint=` / `model=`); those belong to the Microsoft SDK, not to this SDK.

---

### 7.9 Returning Users and Cross-Session Memory

Each `session_id` is one Beacon conversation (`conversationId == sessionId`). Cross-session memory is returned by Beacon scoped to `owner_id`, so a new `session_id` for the same owner receives the earlier facts through the first-turn seed and through `recall_customer_context`. Seeding is tracked in memory per adapter instance: a new adapter instance (or a restarted process) seeds again on the first turn of any `session_id`, including one used before.

---

### 7.10 Go-Live Checklist

- [ ] `agent_framework` installed (`pip install agent-framework-foundry`), Python 3.10+
- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] Credentials passed at construction (`auth=` recommended); construction succeeds, since it performs the token exchange
- [ ] `dataspace=` set explicitly if it is not `"default"` (no environment lookup)
- [ ] `owner_id` comes from an authenticated source; per-call `owner_id=` used when one adapter serves many users
- [ ] A new `AgentSession` per conversation with a stable, unique `session_id`; not two concurrent turns on the same `session_id` (per-turn capture state is keyed by `session_id`)
- [ ] `await adapter.get_context("<session_id>")` returns `userContext` entries for a user with prior data
- [ ] `await adapter.close()` called on shutdown to drain pending deliveries (each turn already waits up to 20 s via `flush()`)
- [ ] If using context packs: `await adapter.get_context_pack_list()` returns your packs

---

## 8. Troubleshooting

### HTTP 401 Unauthorized

How a `401` from Beacon surfaces depends on how you authenticated:

- **`ContextAuth` / `UserContextAuth` / `ApiKeyAuth` (including every adapter's
  inline-credential and API-key path).** The SDK refreshes the token shortly
  before it expires, and on a `401` it invalidates the token, re-runs the
  exchange and retries the request once. A transient `401` therefore
  self-heals with no action from you.
- **Still failing after that retry:** the request raises
  `IngestionError(status_code=401, …)` (message `ingestion failed: HTTP 401: …`)
  on the sync stack. On the async stack (all five adapters ingest through
  `AsyncSessionManager`, except `ClaudeAdapter`/`OpenAIAdapter` with
  `use_async=False`) delivery is in the background: the batch is dropped and
  logged as `_deliver: dropping N record(s) after 1 attempt(s) — ingestion failed: HTTP 401 …`.
  Read calls such as `get_context()` are best-effort and return `None`.
  Because the exchange itself succeeded, the credentials are valid but Beacon
  rejected the resulting token — check with your Salesforce admin that the
  Connected App / user is permitted to use the agent and dataspace you are
  addressing.
- **Bad credentials** do not show up as a Beacon `401`; the token exchange
  itself fails (`httpx.HTTPStatusError` from the login or `/services/a360/token`
  call, or a `ValueError` described below). With `ClaudeAdapter`,
  `OpenAIAdapter` and `FoundryAdapter` this happens in the constructor; with
  `GoogleAdkAdapter` on the first turn and `LangGraphAdapter` on
  `async with adapter:`. Fix the values and construct a new adapter.
- **LangGraph static-token path** (`api_key=` + `base_url=` or
  `CDP_TOKEN` + `CDP_BASE_URL`): nothing refreshes the token. `AsyncIngestionClient`
  logs a warning at construction ("constructed with static api_key and no auth=
  …"); once the token expires you get `401`s and dropped records. Re-mint the
  token and build a new adapter, or switch to a credential-based path.

### Errors raised while building or exchanging credentials

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: ClaudeAdapter requires owner_id …` (same for `OpenAIAdapter`, `FoundryAdapter`) | No `owner_id=` passed; there is deliberately no default | Pass the end-user id (or a stable service-account id) |
| `ValueError: <Adapter> requires credentials …` / `GoogleAdkAdapter requires auth …` / `Self-driving LangGraphAdapter requires: …` | None of `auth=`, the four inline credentials, or the API-key argument was supplied | See [Which credential arguments each adapter accepts](#which-credential-arguments-each-adapter-accepts) |
| `ValueError: Inline ContextAuth credentials are all-or-nothing …` (`LangGraphAdapter`) | Only some of `private_key` / `consumer_key` / `subject` / `login_url` were given | Pass all four, or none |
| `ValueError: login_url is outside the allowlist (expected a bare *.salesforce.com host …)` | `login_url` has a scheme, path, port, or isn't a `*.salesforce.com` host | Use a bare host such as `login.salesforce.com` |
| `ValueError: ApiKeyAuth requires an explicit secret_server_url (or the AGENTIC_CONTEXT_SECRET_SERVER_URL env var) …` | API-key path with no secret-server URL | Pass `secret_server_url=` / `secrets_url=` or set the env var |
| `ValueError: secret_server_url must be https:// …` | Secret-server URL is `http://` | Use `https://` |
| `ValueError: secret server returned a malformed / incomplete credential bundle …` or `… a login host outside the allowlist …` | The `/api/secret` response lacked `secret`, `privateKey`, `consumerKey`, `subject` or a login host, or the host isn't `*.salesforce.com` | Check the key and the secret server's bundle |
| `httpx.HTTPStatusError` from `/api/secret` | Secret server rejected the key (retryable statuses are retried first) | Verify the API key and server URL |
| `ValueError: a360 CDP token exchange returned no access_token after 3 attempts (…)` | The CDP exchange answered 200 without a token three times; the parenthesised text carries the server's own `error` / `error_description` | Act on that text; an unknown `aid` is refused with `access_denied` |
| `ValueError: OAuth response returned … instance_url …` | An exchange response's `instance_url` was empty, `http://`, not `*.salesforce.com`, or had a port/path | Check the org / login host you are using |
| `ReauthRequired` (`UserContextAuth`) | Salesforce rejected the user's refresh token (or authorization code) | Send the user back through login |
| `ValueError: base_url must start with 'https://' or 'http://' …` | Core `IngestionClient` / `AsyncIngestionClient` built with a bare host (static-token path or direct client use) | Include the scheme |
| `ImportError: … requires its framework SDK …` / `<Class>.<method>() requires … not installed` | Framework SDK missing | See [§1](#1-introduction) for the `pip install` to run |

### `get_context()` Returns `None` or Empty `userContext`

`get_context()` exists on `ClaudeAdapter`, `GoogleAdkAdapter`, `OpenAIAdapter`
and `FoundryAdapter`; `LangGraphAdapter` reads context through its recall tool /
automatic seeding instead. Beacon read failures are best-effort: they are logged
as a warning (`AsyncSessionManager.get_context failed (best-effort, returning None): …`,
or `SessionManager.…` in sync mode) and `None` is returned. Foundry also catches
failures to open the session (including an owner mismatch), logs
`[Foundry] get_context failed (best-effort): …` and returns `None`. On `ClaudeAdapter` and `OpenAIAdapter` the owner-binding
`ValueError` (a `session_id` already bound to a different owner) is still raised.

| Cause | Fix |
|-------|-----|
| No session yet | The manager only knows its `session_id` after the first ingest or `start_session()`; before that `get_context()` returns `None`. Pass a `session_id` where the method takes one. |
| Agent not registered in Beacon / `agent_id` wrong | Use the exact registered `agent_id` |
| Credentials wrong / auth failing | See the 401 and credential tables above; check logs for the warning |
| Wrong or missing owner | Beacon keys context on `owner_id`, so a new `session_id` for the **same** `owner_id` recalls earlier conversations. `ClaudeAdapter` / `OpenAIAdapter` / `FoundryAdapter`: pass the real end-user id as `owner_id=` (constructor default, or per call). `GoogleAdkAdapter`: supply it via `adapter.set_owner(email)` or the `on_conversation_start` hook before the first turn — with no owner resolved the adapter skips its Beacon reads and writes for that session. `LangGraphAdapter`: pass `owner_id=` to `stream()`/`run()` or the constructor (`ValueError: owner_id is required …` otherwise). |
| `session_id` reused across owners | `ClaudeAdapter` / `OpenAIAdapter` raise `ValueError: session_id … is already bound to owner …`, `FoundryAdapter` raises `ValueError: session_id … is bound to owner …`; use a distinct `session_id` per conversation |
| No data ingested yet for this owner | Ingest at least one turn first |

### Records Not Appearing

| Check | Fix |
|-------|-----|
| Is `agent_id` correct? | Must match exactly (case sensitive) |
| Is `dataspace` correct? | Default is `"default"`. Resolution order: `dataspace=` argument, then `CDP_DATASPACE` (not read by `FoundryAdapter`), then `"default"` |
| Is `owner_id` the one you expect? | Entries are attributed to, and context is keyed on, the owner |
| Is `close()` being called? | Async delivery runs in background tasks; `close()` waits for pending deliveries (up to 30 s per manager, then cancels the rest). Call `await adapter.close()` (LangGraph: `async with adapter:` or `close()`) before the process exits |
| Are you getting errors in logs? | Delivery failures are logged at `ERROR`: `_deliver: dropping N record(s) after K attempt(s) — …` (retries exhausted or non-retryable status) and `ingest(): dropping N record(s) — M pending deliveries (cap=1000) …` (Beacon degraded). Enable debug logging (see below). On `LangGraphAdapter` you can also pass `ingestion_callbacks=` to get `on_success` / `on_retry` / `on_error` |

### Enable Debug Logging

The core logs under `agentic_context.*`; each adapter logs under its own
package name, which is not a child of `agentic_context`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
for name in (
    "agentic_context",
    "agentic_context_claude",
    "agentic_context_langgraph",
    "agentic_context_adk",
    "agentic_context_openai",
    "agentic_context_foundry",
):
    logging.getLogger(name).setLevel(logging.DEBUG)
```

(`basicConfig(level=DEBUG)` alone already turns every logger up, including
`httpx`; the loop is for when you set the root level higher and want only the
SDK's output.)

### "Session already closed" Error

`RuntimeError: Session already closed` is raised by `SessionManager.ingest()` /
`AsyncSessionManager.ingest()` when you ingest through a manager after its
`close()`. It reaches you when you drive a manager yourself (including one you
inject via `session_manager=` into `GoogleAdkAdapter` or `LangGraphAdapter`)
and keep using it after closing it. The lifecycle is one-directional:

```
create manager → ingest → ingest → ... → close  ✓
create manager → ingest → close → ingest        ✗ RuntimeError
```

Create a new manager (and, if the adapter owned it, a new adapter) for further
work. `FoundryAdapter` raises its own `RuntimeError: FoundryAdapter is closed` for
calls after `close()`; don't reuse any adapter after `close()` — construct a new one.

---

**End of Developer Guide**
