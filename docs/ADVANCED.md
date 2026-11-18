# Agentic Context SDK — Advanced Guide

**Version:** 0.4.0  
**Last Updated:** 2026-11-11  
**Audience:** Developers who have completed [QUICKSTART.md](QUICKSTART.md) and need the deep reference

> New here? Start with [QUICKSTART.md](QUICKSTART.md) (one screen to first
> success) and [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) (prerequisites +
> per-framework pointers). This document is the exhaustive reference: full
> constructor tables, token-exchange internals, HTTP observability, sync mode,
> the cross-process snapshot deep-dive, per-framework method references,
> go-live checklists, and troubleshooting.

**Install:** `pip install agent-context-sdk`. All five adapters ship in the one
distribution; the framework SDKs (`claude-agent-sdk`, `google-adk`,
`langgraph`/`langchain-core`, `agent-framework-foundry`, `openai-agents`) are
not bundled — install the one your agent runs on directly if it isn't already
present.

**Reading the code samples.** As in [QUICKSTART.md](QUICKSTART.md), imports
from your agent framework are tagged `# your existing <framework> code` — you
already have those. Lines tagged `# Agentic Context SDK` are what you add.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Claude Agent SDK Adapter](#3-claude-agent-sdk-adapter)
4. [LangGraph Adapter](#4-langgraph-adapter)
5. [Google ADK Adapter](#5-google-adk-adapter)
6. [AgentContextClient](#6-agentcontextclient)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Introduction

The **Agentic Context SDK** gives your AI agent cross-session memory powered by Salesforce Data Cloud. After integration, your agent can:

- **Remember** returning users across conversations
- **Recall** prior context (order numbers, preferences, past issues) without the user repeating themselves
- **Ingest** every conversation turn (messages, tool calls, reasoning) automatically

### Supported Frameworks

All five adapters ship inside the single `agent-context-sdk` distribution, so
`pip install agent-context-sdk` installs every one. The framework SDKs are not
bundled — install the one your agent runs on directly if it isn't already
present. Every adapter is re-exported from the top-level package
(`from agentic_context import ClaudeAdapter`), resolved lazily so importing one
adapter only loads that framework's SDK:

| Framework | Adapter (`from agentic_context import …`) | Per-package import (also works) | Framework SDK (bring your own) | Covered in |
|-----------|-------------------------------------------|---------------------------------|--------------------------------|------------|
| Claude Agent SDK | `ClaudeAdapter` | `agentic_context_claude` | `pip install claude-agent-sdk` | [§3](#3-claude-agent-sdk-adapter) |
| LangGraph | `LangGraphAdapter` | `agentic_context_langgraph` | `pip install langgraph langchain-core` | [§4](#4-langgraph-adapter) |
| Google ADK | `GoogleAdkAdapter` | `agentic_context_adk` | `pip install google-adk` | [§5](#5-google-adk-adapter) |
| OpenAI Agents SDK | `OpenAIAdapter` | `agentic_context_openai` | `pip install openai-agents` | [QUICKSTART](QUICKSTART.md#openai-agents-sdk), [`adapters/openai/README.md`](../adapters/openai/README.md) |
| Microsoft Foundry | `FoundryAdapter` | `agentic_context_foundry` | `pip install agent-framework-foundry` | [QUICKSTART](QUICKSTART.md#microsoft-foundry), [`adapters/foundry/README.md`](../adapters/foundry/README.md) |
| AgentContextClient (Design.md §8) | `AgentContextClient` | `agentic_context` | none | [§6](#6-agentcontextclient) |

Pick the adapter for your framework and follow the corresponding section below,
or use [AgentContextClient](#6-agentcontextclient) when you are not on an adapter.
Helpers beyond the adapter class (e.g. `make_context_node`, `resolve_policy`,
the LangGraph hook event types) still import from the per-package module.

---

## 2. Prerequisites

### What you need

| Value | Where to get it |
|-------|----------------|
| **Consumer key** | The OAuth consumer key (client id) of your External Client App (ECA) / Connected App — Salesforce Setup → App Manager, or ask your Salesforce admin |
| **Login URL** | Your org's login host, no scheme — the My Domain host (e.g. `acme.my.salesforce.com`), or `login.salesforce.com` / `test.salesforce.com` |
| **Agent ID** | An agent config registered in Beacon by your Salesforce admin — must match exactly (e.g. `Agent_Data_Cloud`) |
| **Dataspace** | Your Beacon dataspace name — default is `"default"` unless your org uses a custom one |
| **Python** | 3.10+ |

You do **not** need a Beacon URL or a CDP token. The auth classes below run the
token exchange, discover your Beacon `base_url` from it (the tenant's
`*.c360a.salesforce.com` host), and refresh the token before it expires (and on
a `401`). Both are only inputs on the LangGraph
[static-token path](#obtaining-a-cdp-api-token-manually-static-token-path).

### Authentication: `UserContextAuth` (recommended)

`UserContextAuth` signs the end user in with their own Salesforce credentials
(OAuth authorization code + PKCE). Every Beacon call then runs as **that user**,
so data access and audit follow their identity and permissions, and the owner
is simply the signed-in user (`auth.user_id`, their `005…`). It is the
preferred auth for every adapter.

For a local tool, notebook or CLI — one person on their own machine — one call
opens the browser and returns a ready `auth`:

```python
from agentic_context import ClaudeAdapter, UserContextAuth  # Agentic Context SDK

auth = UserContextAuth.login(
    consumer_key="3MVG...",                  # ECA consumer key
    login_url="acme.my.salesforce.com",      # login host, no scheme
    aid="support-bot",                       # Beacon-registered agent id
)
adapter = ClaudeAdapter(agent_id="support-bot", owner_id=auth.user_id, auth=auth)
# No base_url, no api_key, no refresh loop. The SDK exchanges + refreshes for you.
```

One `auth` instance can be passed to every adapter in the process (Claude + ADK +
LangGraph + OpenAI + Foundry) so they share one token exchange and refresh. The
ECA setup for this local flow, the refresh-token caching, and when to use it are
in [Local sign-in](#local-sign-in-usercontextauthlogin) below. A web backend that
serves many users hosts the OAuth callback itself instead — next section.

A `ReauthRequired` exception from any adapter call means the user's refresh
token is gone (expired or revoked) — send them back through sign-in.

### Acting as each signed-in user (`UserContextAuth`, backend-hosted login)

When the chat users are Salesforce users of the org and your app is a web
backend, use the backend-hosted form of `UserContextAuth`: the same
authorization-code + PKCE flow, with the OAuth callback hosted on **your
backend** (the browser is only redirected through it).

ECA settings: callback URL `https://<your-backend>/oauth/callback`, scopes
`api refresh_token cdp_ingest_api`, "Require secret for Web Server Flow" **on**,
"Require PKCE" **on**, admin-approved users pre-authorized by profile/permission
set. Use the org's My Domain host as `login_url`.

```python
from agentic_context import (
    FileLoginStateStore, FileRefreshTokenStore,   # or the InMemory* variants, or your own
    ReauthRequired, UserContextAuth,
)

ECA = dict(consumer_key=CK, login_url="acme.my.salesforce.com")
login_states = FileLoginStateStore("./.state", expiration_seconds=300)
token_store = FileRefreshTokenStore("./.tokens")       # plaintext: dev/demo only

# GET /oauth/login
req = UserContextAuth.authorization_request(
    **ECA, redirect_uri="https://chat.example.com/oauth/callback",
)
login_states.save(req)                    # holds state + code_verifier + redirect_uri
# → 302 to req.url

# GET /oauth/callback?code=...&state=...
tx = login_states.consume(query["state"])   # one-time; None if unknown/expired/reused → 400
login = await UserContextAuth.aredeem_code(
    **ECA, consumer_secret=CS, code=query["code"], state=query["state"],
    expected_state=tx.state, code_verifier=tx.code_verifier, redirect_uri=tx.redirect_uri,
)
token_store.save(login.org_id, login.user_id, login.refresh_token)
# Issue your own session cookie mapping to (login.org_id, login.user_id).

# Per request: one UserContextAuth (and one adapter) per user, cached by
# (org_id, user_id). The store is the source of truth for the refresh token.
auth = UserContextAuth.from_store(
    token_store, org_id=org_id, user_id=user_id,
    **ECA, consumer_secret=CS, aid="support-bot",
)
try:
    token, base_url = await auth.aget_token()
except ReauthRequired:                    # the dead token was already deleted from the store
    ...                                   # → HTTP 401 {"error": "reauth_required", "login_url": "/oauth/login?..."}
```

### Storage

Two small interfaces (`agentic_context.stores`), each with an in-memory and a file
implementation. Everything is synchronous, also from async code.

| Interface | Methods | Built-in |
|---|---|---|
| `LoginStateStore` | `save(request)`, `consume(state) -> request \| None` | `InMemoryLoginStateStore`, `FileLoginStateStore` |
| `RefreshTokenStore` | `load(org, user)`, `save(org, user, token)`, `delete(org, user, *, if_token=None)` | `InMemoryRefreshTokenStore`, `FileRefreshTokenStore` |

- `consume()` is one-time use: it returns the login at most once, and only before
  it expires (default 5 min). That is the replay / CSRF guard for the callback, so
  a custom store must claim the row atomically (e.g. DynamoDB `DeleteItem` with
  `ReturnValues=ALL_OLD`, or Redis `GETDEL`).
- The built-in stores are for tests, demos and a single process. The file stores
  write **plaintext** secrets (mode 0600). A production backend with several
  replicas should implement these few methods over its own database and encrypt
  refresh tokens at rest (e.g. KMS).
- `from_store` is a thin wrapper over `load_refresh_token=` / `on_refresh_token=`;
  use those two arguments directly if you don't want a store object.

- `ReauthRequired` means the grant is gone: the refresh token expired or was
  revoked, the user was deactivated, or the code was already used. It is never
  retried. Send the browser back through `/oauth/login`. Other OAuth errors,
  such as `invalid_client` (wrong consumer key or secret), still raise
  `httpx.HTTPStatusError`.
- `load_refresh_token` (what `from_store` uses) is called before every refresh. If Salesforce answers
  `invalid_grant`, it is called once more and the grant is retried when the
  stored value changed. This covers another replica rotating the same user's
  token. `on_refresh_token` runs before the CDP exchange and must persist the
  new token before it returns.
- The owner is the signed-in user. Take `owner_id` from your session
  (`auth.user_id`, their `005…`), never from LLM tool arguments. No `xid` is
  sent. The token is the employee's own, and `xid` is only for the
  integration-user model of `ContextAuth`.
- For local / desktop tools that have no web server of their own, see
  [Local sign-in](#local-sign-in-usercontextauthlogin) below.

#### Local sign-in (`UserContextAuth.login()`)

**Why it exists.** In the backend flow *you* host the callback: you pick the
`redirect_uri`, serve `/oauth/callback`, and the SDK never opens a socket.
A command-line tool, notebook or desktop script has no such server, yet
Salesforce still has to redirect the browser somewhere with the authorization
code. `login()` covers that case: it starts a short-lived listener on
`127.0.0.1`, opens the browser, captures the one redirect, and shuts the
listener down. This is the same loopback pattern the AWS CLI and `gcloud` use.
It is why `login()` has a `port` argument and the backend methods do not.

**When to use which.**

| | Backend-hosted (`authorization_request` / `redeem_code`) | Local sign-in (`login()`) |
|---|---|---|
| Who runs the callback | Your web app, at a `redirect_uri` you choose | The SDK, on `http://localhost:<port>/callback` |
| Fits | A site or chat app serving many users | One person on their own machine |
| ECA client | Confidential or public, your choice | Public: consumer key only, "Require secret" off |
| Needs a browser on the same machine | No | Yes |
| Port | None | `port=` (default `53682`) |

`login()` is not for servers: it blocks waiting for a browser on the host it
runs on, and it signs in exactly one user.

**Setup.**

```python
auth = UserContextAuth.login(
    consumer_key="3MVG9...",
    login_url="mydomain.my.salesforce.com",   # bare host, no scheme
    aid="my_agent",
    on_refresh_token=save_somewhere,          # optional, see below
)
```

- Register `http://localhost:53682/callback` as a callback URL on the ECA. The
  redirect URI must match a registered callback exactly, so `port=` has to
  match too. The SDK does not fall back to another port when the default is
  busy; it raises an `OSError` naming the port.
- The ECA must grant every scope you request (default
  `api refresh_token cdp_ingest_api`).
- The refresh token only lives in the process unless you persist it.
  `on_refresh_token` receives the first token and every rotated one.

### Alternative: service account (`ContextAuth`, JWT bearer)

Use `ContextAuth` only when there is no signed-in Salesforce user to act as — a
backend job, or a server whose end users are not users of the org. Every call
then runs under **one** integration identity, and you supply the owner per
conversation yourself. No browser; the Connected App's RSA private key proves
identity.

```python
from agentic_context import ClaudeAdapter, ContextAuth  # Agentic Context SDK

auth = ContextAuth(
    private_key=open("server.key").read(),   # PEM RSA private key
    consumer_key="3MVG...",                  # Connected App consumer key
    subject="integration-user@example.com",  # Salesforce username for JWT bearer
    login_url="login.salesforce.com",        # login host, no scheme; test orgs: test.salesforce.com
)
adapter = ClaudeAdapter(agent_id="support-bot", auth=auth)
```

The adapters also accept the four credentials inline (`private_key=`,
`consumer_key=`, `subject=`, `login_url=`) and build the `ContextAuth`
themselves; pass a shared instance instead when several adapters run in one
process. Signing the JWT uses the `cryptography` package, a core dependency of
`agent-context-sdk`, so no extra install is needed.

> **Need a Connected App for this flow?** Go to Salesforce Setup → App Manager → New Connected App. Enable OAuth, add the `cdp_ingest_api cdp_profile_api cdp_query_api` scopes, and use certificate-based authentication with an RSA key pair you generate. Your Salesforce admin can also provision this for you.

### Obtaining a CDP API Token manually (static-token path)

If you'd rather mint the token yourself — for testing, or to manage rotation
out-of-band — **only `LangGraphAdapter`** accepts a pre-minted `base_url` +
`api_key` (see its [Constructor](#43-constructor)); the other four adapters
require one of the auth paths above. The two-step flow the SDK automates is:

**Step 1 — Sign a JWT and exchange for a core access token:**

```python
import time, jwt, requests

# Your Connected App credentials (from Salesforce Setup → App Manager)
CONSUMER_KEY = "your-connected-app-consumer-key"
PRIVATE_KEY  = open("your-private-key.pem").read()
LOGIN_URL    = "https://login.salesforce.com"   # or test: login.test.salesforce.com
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
core_token    = resp.json()["access_token"]
instance_url  = resp.json()["instance_url"]
```

**Step 2 — Exchange the core token for a CDP Bearer token:**

```python
CDP_TENANT_ID = "a360/<your-tenant-id>"   # provided by your Salesforce admin

resp = requests.post(
    f"{instance_url}/services/a360/token",
    headers={"Authorization": f"Bearer {core_token}"},
    json={"grant_type": "urn:salesforce:grant-type:external:cdp", "subject_token_type": "urn:ietf:params:oauth:token-type:access_token", "subject_token": core_token},
)
cdp_token = resp.json()["access_token"]
```

Set the resulting `cdp_token` as `BEACON_API_KEY`. Tokens expire in ~2 hours — with the static-token path you must build your own refresh mechanism (re-run the exchange) and rebuild the LangGraph client + adapter with the new token. **`UserContextAuth` (or `ContextAuth`) does all of this for you** and is the recommended approach for production.

### Environment variables (static-token examples)

With `auth=` you need none of these. The static-token examples in this guide
(e.g. [§4.5](#45-end-to-end-example-snapshot)) read them:

```bash
export BEACON_BASE_URL="https://your-tenant.c360a.salesforce.com"
export BEACON_API_KEY="your-cdp-bearer-token"
export BEACON_DATASPACE="default"
export BEACON_AGENT_ID="your-registered-agent-id"
```

---

## 3. Claude Agent SDK Adapter

> **TL;DR** — Create **one** `ClaudeAdapter` per auth identity (one per signed-in user with `UserContextAuth`; one for the whole app with a service-account `ContextAuth`), then call `adapter.apply_to(options, session_id, owner_id)` once per conversation on the `ClaudeAgentOptions` you already build. That single call wires up memory recall, the memory tools, and ingestion. Everything else in this section is optional detail.

### 3.1 Quick Start

This is the complete, recommended integration. If you only read one code block, read this one.

```python
from claude_agent_sdk import ClaudeAgentOptions, query  # your existing Claude Agent SDK code
from agentic_context import ClaudeAdapter, UserContextAuth  # Agentic Context SDK

# 1. Sign the user in once — the SDK runs the token exchange, discovers your
#    base_url, and refreshes the token automatically. See §2.
auth = UserContextAuth.login(
    consumer_key="3MVG...",                       # ECA consumer key
    login_url="acme.my.salesforce.com",           # login host (no scheme)
    aid="support-bot",
)

# 2. ONE adapter for this user (not one per conversation).
adapter = ClaudeAdapter(
    agent_id="support-bot",                       # must be registered in Beacon
    owner_id=auth.user_id,                        # the signed-in user's 005…
    auth=auth,
)

# 3. Per conversation: apply_to() adds memory to the options you already build.
options = ClaudeAgentOptions(                     # your existing Claude Agent SDK code
    model="claude-sonnet-4-20250514",
    system_prompt="You are a helpful support agent.",
)
options = adapter.apply_to(options, session_id="conv-123", owner_id=auth.user_id)

# 4. Run the agent. Memory recall + ingestion happen automatically.
async for message in query(prompt="What's the status of my order?", options=options):  # your existing Claude Agent SDK code
    if hasattr(message, "result") and message.result:
        print(message.result)

# 5. On shutdown (once for the whole app).
await adapter.close()
```

That's the full integration. No `start_session()`, no manual `get_context()`, no hand-written Stop hook. `apply_to()` returns a new `ClaudeAgentOptions` (yours is not mutated) with:

- **A memory MCP server** (`mcp__context__…`) carrying the `recall_customer_context` and `get_context_pack` tools, both added to `allowed_tools`.
- **The adapter's hooks** from [`hooks()`](#37-hooks--the-primary-integration-point), merged alongside any you already have:
  - **`UserPromptSubmit`** → calls `get_context()` and injects prior memory as `additionalContext` before the model runs.
  - **`Stop`** → calls `push()` to ingest the turn's transcript into Beacon (fire-and-forget in async mode).
  - **`PreCompact`** → marks the conversation so the next turn re-attaches the context-pack instruction and catalog that compaction summarised away.

Your `system_prompt` is left alone. Prefer to wire the pieces yourself? Pass `hooks=adapter.hooks(session_id, owner_id)` and `mcp_servers={"context": adapter.mcp_server(...)}` to `ClaudeAgentOptions` directly — see [§3.7](#37-hooks--the-primary-integration-point).

**Lifecycle is lazy** — the Beacon conversation + session are created automatically on the first `Stop` hook. You do not need to create anything up front.

### 3.2 Minimal Requirements

| You need | Notes |
|----------|-------|
| `pip install agent-context-sdk` | Python 3.10+. |
| `claude-agent-sdk` | NOT bundled by `agent-context-sdk` — you're expected to already have it (you run Claude); if not, `pip install claude-agent-sdk`. It is lazy-imported only by the methods that build Claude Agent SDK objects (`hooks()`, `mcp_server()`, `apply_to()`); every other method works without it. |
| A registered `agent_id` in Beacon | Required for `get_context()` to return data. See [§2 Prerequisites](#2-prerequisites). |
| An `auth` instance (or inline service-account credentials) | Recommended: `auth=UserContextAuth` (signed-in user); `auth=ContextAuth` for a service account. The SDK handles exchange + refresh. See [§2 Authentication](#authentication-usercontextauth-recommended). |
| `owner_id` | The end-user identity — `auth.user_id` with `UserContextAuth`. Set a default in the constructor; override per conversation in `apply_to(...)` / `hooks(...)`. |

The **only two things you must do per conversation** are: pick a `session_id` (your conversation/thread ID) and an `owner_id` (the end user), then pass both to `apply_to()` (or `hooks()`).

### 3.3 One Adapter, Many Sessions (the core mental model)

A single `ClaudeAdapter` instance manages **N sessions internally**, keyed by `session_id`. You create the adapter once and reuse it for every conversation of the same auth identity. With `UserContextAuth` that is all of one signed-in user's conversations. With a service-account `ContextAuth`, one adapter can serve every end user, which is the case shown here:

```python
# App startup — create once. No owner_id needed; it's passed per conversation.
adapter = ClaudeAdapter(agent_id="support-bot", auth=service_auth)  # a ContextAuth — see §2

# Conversation A
hooks_a = adapter.hooks("conv-A", "alice@example.com")

# Conversation B (concurrent, same adapter, different owner)
hooks_b = adapter.hooks("conv-B", "bob@example.com")

# App shutdown — close once. Drains all sessions + clients.
await adapter.close()
```

Internally the adapter caches one `SessionManager` per `session_id` and shares a single ingestion + retrieval client pair across all of them. The shared client pair is also why token handling is process-wide: one auth instance refreshes the token for **every** session at once.

#### 3.3.1 The ownership model: owner is *per-call*, not *per-adapter*

A common misconception is that you create one adapter per user (because the constructor accepts `owner_id`). **You don't.** One adapter serves *all* users. The owner identity travels with each per-conversation call, not with the adapter instance:

```
adapter.hooks(session_id, owner_id="alice@x.com")   ← owner enters HERE, per call
        │
        ▼
SessionManager(owner_id="alice@x.com")               ← stored as the manager's own state
        │
        ▼
retrieval.get_context(owner_id=manager._owner_id)    ← Beacon calls read it internally
```

Each conversation gets its own `SessionManager` carrying its own frozen `owner_id`. Alice's and Bob's managers coexist inside the **one** shared adapter, never colliding. The constructor `owner_id` is merely the fallback used when a call omits its own owner — handy for single-user bots, irrelevant for multi-user servers.

| Operation | Owner comes from | Notes |
|-----------|------------------|-------|
| `hooks(session_id, owner_id=...)` | the call | The primary path. Always pass the real end user. |
| `start_session(session_id, owner_id=...)` | the call | Same. |
| `push` / `ingest` / `get_context` / `get_session` (by `session_id`) | the **session's** manager | A `session_id` belongs to exactly one owner, so keying by session implicitly keys by the right owner. |
| `list_sessions(owner_id=...)` | the call | **Owner-level** query (no session to key on) — you *must* pass `owner_id` in a multi-user app, or you'll get the wrong owner's conversations. |
| `get_context` / `get_session` for an **unseen** `session_id` | the call's `owner_id`, else the adapter default | A "cold" call has no cached manager. Pass `owner_id=...` to scope it correctly. |

> **With `UserContextAuth` (the recommended auth), owner and adapter go together.** The token itself is scoped to one Salesforce user, so build one `UserContextAuth` + one adapter **per signed-in user** (cache them by `(org_id, user_id)`), not one for the whole process. See [Acting as each signed-in user](#acting-as-each-signed-in-user-usercontextauth-backend-hosted-login).

> **Rule of thumb:** in any app with more than one user, treat the constructor `owner_id` as unset and pass the authenticated user's identity to every `hooks()` / `list_sessions()` call. If you call `get_context()` / `get_session()` directly on a session the adapter hasn't already seen via `hooks()`, pass `owner_id` there too.

> **Migration note (from older versions):** the constructor no longer takes `delivery_mode`, `conversation_id`, or a construction-time `session_id`, and you no longer build a per-conversation adapter pool. If you previously ran an `AdapterManager` keyed by conversation, replace it with one shared adapter and per-conversation `hooks()` calls. `session_id` is now passed to the per-conversation methods (`hooks`, `push`, `get_context`, …), not to the constructor.

### 3.4 Does this work for single-session chat apps?

**Yes — and it's simpler than the multi-session case.** A single-session app (a CLI chat, a notebook, a one-user desktop app) is just the N=1 case:

```python
auth = UserContextAuth.login(consumer_key=CONSUMER_KEY, login_url=LOGIN_URL, aid="bot")
adapter = ClaudeAdapter(agent_id="bot", owner_id=auth.user_id, auth=auth)

options = ClaudeAgentOptions(model="claude-sonnet-4-20250514")  # your existing Claude Agent SDK code
options = adapter.apply_to(options, session_id="my-only-conversation")

async for msg in query(prompt="hello", options=options):
    ...

await adapter.close()
```

Notes specific to single-session apps:

- You can omit `owner_id` in `apply_to()` / `hooks()` — it falls back to the adapter's constructor `owner_id`.
- The retrieval methods (`get_context()`, `get_session()`) can be called with **no** `session_id` — they fall back to the most recently used session, which is unambiguous when there is only one.
- `use_async=False` is available for purely synchronous apps (scripts, Streamlit). See [§3.12](#312-sync-mode).

> **Multi-session caveat:** in a server handling concurrent conversations, **always pass `session_id` explicitly** to `get_context()` / `get_session()` (or just use `hooks()`, which always does). The no-argument fallback resolves to the *last accessed* session, which is racy under concurrency. Inside `hooks()` the `session_id` is always bound explicitly, so the recommended path is always safe regardless of app shape.

### 3.5 How It Works (under the hood)

The Claude Agent SDK fires hooks around each agent turn. The adapter plugs into two of them:

| Hook | When | What the adapter does |
|------|------|-----------------------|
| `UserPromptSubmit` | Before each user message is processed | Calls `get_context()`, formats it, returns it as `additionalContext` so the model sees prior memory. |
| `Stop` | After each agent turn completes | Reads the new transcript lines, transforms them into records, and ingests them to Beacon. Uses `last_assistant_message` for the agent's reply (the transcript may not be flushed yet). |

You normally never touch these directly — `hooks()` builds them for you. The lower-level methods (`push`, `get_context`, `ingest`, …) exist for advanced/manual integrations and are documented in [§3.8](#38-method-reference).

---

### 3.6 Constructor

```python
from agentic_context import ClaudeAdapter  # Agentic Context SDK

# Recommended — UserContextAuth (signed-in user). SDK handles exchange + refresh.
adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id=auth.user_id,
    auth=auth,                              # UserContextAuth — see §2
    dataspace="default",
    use_async=True,
)

# Service account — a shared ContextAuth (or the four credentials inline):
adapter = ClaudeAdapter(agent_id="support-bot", auth=service_auth)
```

The adapter accepts auth in one of three ways, in precedence order: a shared
`auth=` instance (`UserContextAuth`, recommended, or `ContextAuth`), inline
service-account credentials (`private_key`+`consumer_key`+`subject`+`login_url`),
or `secrets_api_key=` to fetch those credentials from a secret server. You must
supply one of these or construction raises `ValueError`.

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `agent_id` | `str` | ✅ | — | Must match a registered agent config in Beacon |
| `owner_id` | `str` | — | `"default"` | **Fallback** end-user identity, used only when a per-call owner is *not* supplied. In a multi-user app leave this unset and pass the real owner per conversation via `hooks(session_id, owner_id=...)`. See [§3.3.1](#331-the-ownership-model-owner-is-per-call-not-per-adapter). |
| `auth` | `UserContextAuth \| ContextAuth \| None` | — | `None` | **Recommended.** A `UserContextAuth` (acts as the signed-in user) or a service-account `ContextAuth`. Share one instance across adapters so a single token exchange/refresh is shared. Takes precedence over the inline credentials. |
| `private_key` | `str` | — | `""` | Service-account path. PEM RSA private key. With `consumer_key`+`subject`+`login_url`, the SDK builds `ContextAuth` and runs the JWT → CDP exchange itself. Uses `cryptography`, a core dependency. |
| `consumer_key` | `str` | — | `""` | Connected App consumer key (the JWT `iss`). |
| `subject` | `str` | — | `""` | Salesforce username authorised for the JWT bearer flow (the JWT `sub`). |
| `login_url` | `str` | — | `""` | Salesforce login host, no scheme (e.g. `login.salesforce.com`; test orgs `login.test.salesforce.com`). |
| `dataspace` | `str` | — | `"default"` | Beacon dataspace name |
| `use_async` | `bool` | — | `True` | `True` for FastAPI/async agents (recommended). `False` for sync scripts. |
| `http_observer` | `Callable[[dict], Any] \| None` | — | `None` | Optional. Fired for every Beacon HTTP call (request + response) so you can surface live traffic in a UI/console or tracer. Async-mode only. See [§3.11](#311-http-observability). |

**Mode is locked at construction.** With `use_async=True` (default), the per-conversation methods (`push`, `get_context`, `get_session`, `list_sessions`, `start_session`, `close`) are coroutines — use `await`. With `use_async=False` they are plain calls. `hooks()` is always synchronous to *call*; `hooks()` returns async or sync hook callbacks to match the adapter's mode.

> **Construct the adapter at startup, not inside a request handler.** With
> `auth=` (or inline credentials) the constructor performs the core → CDP token
> exchange synchronously (one-time, ~1–2s) to obtain the first token. That's ideal at app
> startup, but if you build the adapter *inside* an already-running event loop it
> briefly blocks that loop. Create the one shared adapter once during startup
> (as the examples show) and this never affects request latency. After
> construction, all per-turn work (`push`, `get_context`) is non-blocking and
> token refresh happens in the background.

> **What changed:** there is no `delivery_mode` and no `conversation_id`/`session_id` constructor argument anymore. Delivery is always per-turn (the `Stop` hook fires every turn). The session identity is supplied per call, not at construction.

---

### 3.7 `hooks()` — the primary integration point

Returns a ready-made hooks dict for `ClaudeAgentOptions(hooks=...)`, wiring both memory retrieval and ingestion in one call.

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
| `session_id` | `str` | ✅ | — | The Beacon session ID for this conversation. Bound into both hook closures. |
| `owner_id` | `str \| None` | — | adapter's `owner_id` | Owner override for this conversation. |
| `context_formatter` | `Callable[[dict], str] \| None` | — | `_default_context_formatter` | Custom formatter: turns the Beacon context payload into the string injected as `additionalContext`. |
| `existing_hooks` | `dict[str, list[HookMatcher]] \| None` | — | `None` | Your own hooks dict. The memory matchers are **appended alongside** yours (both run; nothing overwritten); your dict is not mutated. Matchers this adapter added earlier are replaced, not duplicated. See [Composing with your own hooks](#composing-with-your-own-hooks). |
| `inject_context_pack_instruction` | `bool` | — | `True` | Attach the context-pack instruction ahead of the catalog (only when the catalog lists at least one pack). `False` leaves it out. |
| `system_prompt` | `str \| dict \| None` | — | `None` | Your `ClaudeAgentOptions.system_prompt`. If it already contains the instruction (server or bundled text, in a `str`, a preset's `append`, or a custom `prompt`), the hook doesn't attach it again. `apply_to()` passes it for you. |

**Returns:** a dict suitable for `ClaudeAgentOptions(hooks=...)`. Without `existing_hooks`, it has exactly three keys — `"UserPromptSubmit"`, `"Stop"`, and `"PreCompact"` — each a list with one `HookMatcher(matcher=".*", hooks=[callback])`. With `existing_hooks`, it's your dict plus the memory matchers merged in.

**Behavior notes:**

- Calling `hooks()` eagerly registers a `SessionManager` for that `session_id` (so the session is tracked even before the first turn), but **no Beacon call is made** until the first `Stop` fires (lazy lifecycle).
- Both hooks are **best-effort and never raise** — a Beacon outage or expired token logs a warning and the agent turn proceeds normally. Memory is an enhancement, never a blocker.
- The hook callbacks are `async def` when `use_async=True`, plain `def` when `use_async=False`. Both produce the same dict shape.
- Requires `claude_agent_sdk` installed; otherwise raises a clear `ImportError`.

**`UserPromptSubmit` return shape:**

```python
{
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "[Memory from prior conversations]\n- [Profile] Prefers vegetarian\n- ...",
    }
}
```

If there's nothing new to attach (no memory to seed, and the instruction and catalog are already in this Claude conversation), it returns `{}`.

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

Return an empty string from your formatter to suppress injection.

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
| `UserPromptSubmit` | `[<memory get_context>]` (ours) |

The Claude Agent SDK runs **every** matcher in an event's list, so for shared events (`Stop`, `UserPromptSubmit`) both your hook and ours fire. Your `existing_hooks` dict is copied, not mutated.

> **Why this matters:** without `existing_hooks` you'd have to merge by hand, and the obvious `{**my_hooks, **adapter.hooks(...)}` would *silently overwrite* your `Stop` with ours (or vice versa). Passing `existing_hooks` avoids that footgun.

---

### 3.8 Method Reference

> Most apps only need `hooks()` (or `apply_to()`) and `close()`. The methods below are for manual integrations (custom hook wiring, non-SDK event sources, or building your own UI lifecycle).

All per-conversation methods are coroutines in async mode (`await`), plain calls in sync mode.

#### `start_session` *(optional)*

Eagerly creates the conversation + session + participants in Beacon **before** the first message. Only needed if you want the lifecycle objects to exist up front (e.g. a UI that displays the session ID immediately). With `hooks()`, lifecycle is lazy and this is **not required**.

```python
# Async
async def start_session(session_id: str, *, owner_id: str | None = None, title: str = "") -> dict[str, str]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | The conversation/session ID |
| `owner_id` | `str \| None` | Owner override (defaults to constructor `owner_id`) |
| `title` | `str` | Optional human-readable title |

**Returns:** `{"sessionId": ..., "ownerId": ..., "agentId": ...}`. A 409 (already exists) is treated as success — returning users just work.

```python
await adapter.start_session("conv-123", owner_id="user@example.com", title="Support Chat")
```

#### `push`

Ingests one turn's transcript into Beacon. Called automatically by the `Stop` hook from `hooks()` — call it directly only if you wire your own Stop hook.

```python
# Async
async def push(session_id: str, input_data: dict) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | The Beacon session to push to |
| `input_data` | `dict` | The Claude Agent SDK `Stop` hook input dict |

`input_data` keys used: `transcript_path`, `session_id` (the SDK's internal ID), `last_assistant_message`. If `transcript_path` or `session_id` is missing, `push()` is a no-op. Lifecycle is lazy — the first `push()` for a session creates it.

```python
async def on_stop(input_data, tool_use_id, context):
    await adapter.push("conv-123", input_data)
    return {}
```

#### `ingest`

Ingests a list of `Record` objects directly — for custom event sources that aren't Claude transcripts. Most integrations use `push()` instead.

```python
# Async
async def ingest(session_id: str, records: list[Record]) -> None
```

Empty list is a no-op.

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

Retrieves cross-session context from Beacon. Used automatically by the `UserPromptSubmit` hook; call directly only for manual injection.

```python
# Async
async def get_context(session_id: str | None = None, *, owner_id: str | None = None) -> dict | None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str \| None` | The session to fetch context for. If `None`, falls back to the most recently used session (safe for single-session apps; **pass explicitly** in multi-session servers). |
| `owner_id` | `str \| None` | Only used for a **cold** call — a `session_id` the adapter hasn't seen before (no prior `hooks()`/`start_session()`). Scopes the new session to the right owner instead of the adapter default. Ignored for a warm session (its manager already carries the owner). **Pass it in multi-user apps.** |

**Returns:** a context `dict`, or `None` on any error or if no session is bound. **Never raises.** Requires the `agent_id` to be registered in Beacon.

```python
context = await adapter.get_context("conv-123")
if context:
    system_prompt += f"\n\nPrior context:\n{context}"
```

#### `get_session`

Retrieves the full session data (all entries, metadata) from Beacon. Same `session_id` and `owner_id` semantics as `get_context` (`owner_id` only matters for a cold, previously-unseen session). Returns `dict | None`, never raises.

```python
session = await adapter.get_session("conv-123")
```

#### `list_sessions`

Lists all conversations **for a given owner**. Returns `list[dict]`, or `[]` on error.

```python
# Async
async def list_sessions(owner_id: str | None = None) -> list[dict]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `owner_id` | `str \| None` | Owner to scope the listing to. Defaults to the adapter's fallback owner. **Always pass the current end user's `owner_id` in a multi-user app** — the adapter is a single process-wide instance, so an unscoped call returns whichever owner the adapter defaults to, *not* the current user's conversations. |

```python
for s in await adapter.list_sessions(owner_id="alice@example.com"):
    print(s.get("id"), s.get("name"))
```

#### `close`

Closes **all** session managers and the shared clients. Call once on app shutdown. In async + non-blocking mode it drains pending background deliveries before closing. Errors closing individual sessions are logged, not raised.

```python
# Async
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
adapter.session_id    # → str | None — the most recently accessed session (backward-compat shim)
adapter.capabilities  # → Capabilities(working=True, persistent=False)
```

> `session_id` returns the last-accessed session and exists mainly for backward compatibility. In a multi-session app, don't rely on it to identify "the" session — there isn't one.

---

### 3.9 End-to-End Example: FastAPI Agent (service account, one shared adapter)

This is the **service-account** production pattern: one `ContextAuth`
identity serves every end user, so a **single** adapter is created once at
startup (it handles token exchange + refresh itself), `apply_to()` per request
with that request's owner, and it's closed on shutdown. No per-request token
plumbing.

> **Acting as each signed-in user instead?** That's the recommended
> `UserContextAuth` path: one adapter per signed-in user (built from their
> `auth`), cached by `auth.user_id`. See
> [§2 — Authentication](#authentication-usercontextauth-recommended) and
> [§3.3](#33-one-adapter-many-sessions-the-core-mental-model).

```python
import os
from fastapi import FastAPI  # your existing app code
from pydantic import BaseModel  # your existing app code
from claude_agent_sdk import ClaudeAgentOptions, query  # your existing Claude Agent SDK code
from agentic_context import ClaudeAdapter, ContextAuth  # Agentic Context SDK

app = FastAPI()

DATASPACE = os.environ.get("BEACON_DATASPACE", "default")
AGENT_ID = "support-bot"

# ── One adapter for the whole process, built once at startup ───────
# Service account: the SDK runs the JWT → CDP exchange, discovers base_url,
# and refreshes the token automatically (before expiry and on 401). No
# get_fresh_token() helper.
_adapter: ClaudeAdapter | None = None

@app.on_event("startup")
async def startup():
    global _adapter
    _adapter = ClaudeAdapter(
        agent_id=AGENT_ID,
        # No owner_id here — the real owner is passed per request via
        # apply_to(..., owner_id=body.owner_id) below.
        auth=ContextAuth(
            private_key=open(os.environ["BEACON_PRIVATE_KEY_FILE"]).read(),
            consumer_key=os.environ["BEACON_CONSUMER_KEY"],
            subject=os.environ["BEACON_SUBJECT"],
            login_url=os.environ.get("BEACON_LOGIN_URL", "login.salesforce.com"),
        ),
        dataspace=DATASPACE,
    )


class MessageRequest(BaseModel):
    text: str
    owner_id: str


@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, body: MessageRequest):
    options = ClaudeAgentOptions(  # your existing Claude Agent SDK code
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful support agent.",
    )
    options = _adapter.apply_to(  # ← the whole integration
        options, session_id=conversation_id, owner_id=body.owner_id,
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
adapter = ClaudeAdapter(agent_id="support-bot", owner_id=auth.user_id, auth=auth)
owner = auth.user_id

# Conversation 1 — order inquiry
options1 = adapter.apply_to(options, session_id="conv-001", owner_id=owner)
# ... agent runs; Stop hook ingests each turn ...

# Conversation 2 (days later) — SAME session_id continues the same conversation
options1_again = adapter.apply_to(options, session_id="conv-001", owner_id=owner)
ctx = await adapter.get_context("conv-001")   # includes prior turns

# Or a NEW conversation for the same user — cross-session recall via owner_id
options2 = adapter.apply_to(options, session_id="conv-002", owner_id=owner)
ctx2 = await adapter.get_context("conv-002")   # Beacon owner cache returns prior facts
```

A 409 from Beacon (session already exists) is treated as success, so reusing a `session_id` for a returning user just works.

### 3.11 HTTP Observability

Pass `http_observer` to watch **every** Beacon HTTP call the adapter makes — both ingestion writes and retrieval reads. It's the seam for surfacing live Beacon traffic in a UI/console or wiring a request tracer. (Distinct from `IngestionCallbacks`, which are ingestion-only, per-batch, and carry records rather than HTTP metadata.)

```python
async def on_beacon_call(event: dict) -> None:   # may be sync or async
    # phase="request":  {"phase","callId","method","url","timestamp"}
    # phase="response": {"phase","callId","method","url","status","latencyMs","timestamp"}
    #                   (transport error → also "error")
    print(event["phase"], event["method"], event["url"], event.get("status"))

adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id=auth.user_id,
    auth=auth,
    http_observer=on_beacon_call,
)
```

- **`callId`** pairs a request event with its response event.
- **`url` is path-only** — query strings (which carry tokens / owner_ids) are stripped before the event is built, so they never reach your observer/UI.
- **Best-effort** — observer exceptions are swallowed; observability never disrupts delivery.
- **Async-mode only** — wired via httpx event hooks. In sync mode the argument is ignored with a warning.
- **Labelling is yours** — the event carries `method` + `url`, not payloads. Map those to friendly labels (e.g. `…/entries` → "Add Entries", `…/context` → "Get Context").

The support demo uses this to drive a live "Beacon API Console" panel over SSE.

### 3.12 Sync Mode

For purely synchronous apps (scripts, Streamlit, CLIs), pass `use_async=False`. The per-conversation methods become plain calls and `hooks()` returns synchronous callbacks.

```python
adapter = ClaudeAdapter(
    agent_id="support-bot",
    owner_id=auth.user_id,
    auth=auth,
    use_async=False,
)

options = ClaudeAgentOptions(
    model="claude-sonnet-4-20250514",
    hooks=adapter.hooks("conv-123"),
)

for message in query(prompt="Hello!", options=options):
    if hasattr(message, "result") and message.result:
        print(message.result)

adapter.close()   # no await
```

In sync mode, ingestion blocks until the POST completes (with retry). In async mode it's non-blocking — the transcript is parsed (<5ms) and delivery is scheduled as a background task with automatic retry, so the next turn starts immediately; `close()` drains pending deliveries.

### 3.13 Go-Live Checklist

- [ ] **One** adapter instance for the app (not one per conversation)
- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] `owner_id` passed to `apply_to()` / `hooks()` is the real end-user identity (`auth.user_id` with `UserContextAuth`) (constructor `owner_id` is just the default)
- [ ] `owner_id` passed to `list_sessions()` is the current user (an unscoped call returns the wrong owner's conversations — see [§3.3.1](#331-the-ownership-model-owner-is-per-call-not-per-adapter))
- [ ] `session_id` passed explicitly to `get_context()` / `get_session()` in multi-session servers (and `owner_id` too if the session may be cold)
- [ ] `claude_agent_sdk` installed (required for `apply_to()` / `hooks()`)
- [ ] Auth configured — `auth=UserContextAuth` (recommended) or `auth=ContextAuth` auto-refreshes; handle `ReauthRequired` by sending the user back through sign-in
- [ ] `adapter.close()` wired to your shutdown handler (one call drains all sessions)
- [ ] `use_async=False` only if running in a pure sync context (default is async)
- [ ] `get_context()` returns data (confirms the agent config exists in Beacon)

---

## 4. LangGraph Adapter

### 4.1 Installation

```bash
pip install agent-context-sdk
```

**Requirements:** Python 3.10+, plus `langgraph>=0.2` and `langchain-core>=0.3` — these are not bundled by `agent-context-sdk`; install them directly (`pip install langgraph langchain-core`) if your project doesn't already have them.

The distribution is `agent-context-sdk`. Import the adapter from the top-level package (`from agentic_context import LangGraphAdapter`); the per-package path `agentic_context_langgraph` also works and is where the helpers (`make_context_node`, `resolve_policy`, `TurnEvent`, …) live.

### 4.2 How It Works

The LangGraph adapter is a single class — `LangGraphAdapter` — with **three** construction paths:

- **All-in-one (recommended)** — pass `agent_id`, `owner_id`, `auth=` (or static `api_key`/`base_url`), optional `dataspace`, plus your `graph=`. The adapter constructs and owns the `AsyncIngestionClient`, `AsyncSessionManager` (one per `session_id`), and `AsyncRetrievalClient` internally. Use as an async context manager. Inside `stream()`, the adapter automatically calls `GET /agentic/context` (subject to the policy the resolved `context_mode` selects — see 4.11), prepends a `SystemMessage(id="agentic-memory-context")` to your inputs, drives the graph, and POSTs records per turn.
- **`graph=` + `session_manager=`** — pass a live `StateGraph` plus your own pre-built `SessionManager` (sync) or `AsyncSessionManager` (async). The adapter never auto-fetches in this mode — you wire retrieval yourself via `make_context_node` (4.9) or by calling `AsyncRetrievalClient` directly. The adapter never POSTs to Beacon directly — all transport flows through your manager.
- **`pickle_path=` + `session_manager=`** — load a serialized `StateSnapshot` from disk. `snapshot()` works; `stream()` raises `ValueError`. Used for cross-process file handoff.

The all-in-one form is the path that drops customer code to **~6 lines** in the minimum case:

```python
import asyncio
from agentic_context import LangGraphAdapter, UserContextAuth  # Agentic Context SDK

# my_compiled_graph = builder.compile()  — your existing LangGraph graph

async def main():
    auth = UserContextAuth.login(consumer_key=CONSUMER_KEY, login_url=LOGIN_URL, aid="support-bot")
    async with LangGraphAdapter(
        graph=my_compiled_graph, agent_id="support-bot", owner_id=auth.user_id, auth=auth,
    ) as adapter:
        await adapter.run("Hey, I'm back", session_id="ses-2")

asyncio.run(main())
```

That's the entire integration. The customer imports only the adapter and an auth class. They never construct `AsyncIngestionClient`, `AsyncRetrievalClient`, or `AsyncSessionManager`; they never call `get_context`, `ingest`, or `close`; they never normalize a LangChain message into a `Record`. The adapter wraps and hides all of it.

#### What you write vs. what the adapter handles

| Concern | You write | Adapter handles internally |
|---|---|---|
| Compiled `StateGraph` (your nodes, edges, model, tools) | ✅ | — |
| Connection identity (`agent_id`, `owner_id`, `auth`, `dataspace`) | ✅ (kwargs; static token may come from env) | — |
| Token exchange, `base_url` discovery, refresh | — | ✅ via `auth=` |
| Per-call `session_id` | ✅ | — |
| Construct `AsyncIngestionClient` | — | ✅ |
| Construct `AsyncRetrievalClient` | — | ✅ |
| Construct `AsyncSessionManager` (one per `session_id`) | — | ✅ |
| Decide when to GET `/agentic/context` | — | ✅ via `ContextPolicy` |
| Render Beacon payload → `SystemMessage(id="agentic-memory-context")` | — | ✅ via `context_to_system_message` |
| Inject the SystemMessage ahead of the graph | — | ✅ |
| Drive `astream_events(version="v3")` | — | ✅ |
| Normalize each LangChain v3 message → `Record` (stable `event_id`, role mapping, JSON-safe `usage`) | — | ✅ |
| POST records per turn | — | ✅ via `AsyncSessionManager.ingest()` |
| Close all per-session managers, drain in-flight POSTs (30 s budget) | — | ✅ via `__aexit__` |

#### Connection params: kwargs OR environment

With `auth=` (recommended), the token and `base_url` come from the auth instance — no `api_key`/`base_url` needed. On the static-token path, the adapter resolves `api_key`, `base_url`, and `dataspace` from environment variables when the kwargs are omitted:

| Kwarg | Env var | Required? |
|---|---|---|
| `auth` | — | recommended; replaces `api_key` + `base_url` |
| `api_key` | `CDP_TOKEN` | static path only (kwarg or env) |
| `base_url` | `CDP_BASE_URL` | static path only (kwarg or env) |
| `dataspace` | `CDP_DATASPACE` | no (defaults to `"default"`) |
| `agent_id` | — | yes (no env fallback) |
| `owner_id` | — | yes (no env fallback) |

Explicit kwargs always win. Missing required values raise `ValueError` at construction time, naming the missing kwarg.

#### `stream()` vs `run()`

Both drive one turn end-to-end (GET context → inject → drive graph → POST records). Pick by what you do with the records:

| Method | Returns | Use when |
|---|---|---|
| `await adapter.run(text_or_inputs, session_id=...)` | `list[Record]` | You don't need per-record observability — you just want the turn to complete. |
| `async for record in adapter.stream(text_or_inputs, session_id=...)` | async iterator of `Record` | You're forwarding records to a UI, server-sent events, audit tap, or want early-termination logic. |

`stream()` and `run()` accept a plain string (wrapped to `{"messages": [("human", ...)]}` automatically) or a full inputs dict. The adapter spins up one `AsyncSessionManager` per `session_id` it sees, so a single adapter instance can drive any number of conversations across the same owner.

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
| LangChain `@tool` runnables and their callbacks | `tool_call` + `tool_result` records (Trace category) | Always captured in streaming mode by the adapter's internal handler — no hook required (see 4.7). Plain Python functions called inside a node fire no callbacks and produce no tool records. |
| `astream_events(version="v3")` / `stream.messages` projection | — | Driven by `stream()`; one iteration = one fully-aggregated LLM utterance. |
| `add_messages` reducer | — | Not invoked by the adapter, but `make_context_node` (4.9) emits a `SystemMessage` with a stable `id="agentic-memory-context"` so the reducer replaces (not duplicates) the context block on each turn. |

In short: the SDK's `session_id` is the only identifier you provide. The adapter owns the `thread_id` ↔ `session_id` translation; you should never need to construct a `{"configurable": {"thread_id": ...}}` dict yourself for the adapter's own calls.

#### Retrieval — built in (all-in-one) or wired by you (manager-injection)

In the **all-in-one path**, retrieval is handled inside `stream()` automatically. Before each turn the adapter consults the policy its `context_mode` selected (4.10–4.11), and on a fetch decision calls `GET /agentic/context`, renders the payload via `context_to_system_message`, and prepends the result as a `SystemMessage` with the stable id `"agentic-memory-context"`. The LangGraph `add_messages` reducer keeps that `SystemMessage` in state across subsequent turns, so cross-session memory survives without any explicit re-injection. A failed GET (network outage, 4xx, 5xx) is logged and swallowed — the graph still runs with no injected block, so a Beacon outage never breaks the user-facing turn.

In the **manager-injection paths** (`graph=` or `pickle_path=` + your own `session_manager=`), the adapter does **not** fetch context. You wire retrieval yourself, two ways:

| Approach | Use when |
|----------|----------|
| `make_context_node(client, agent_id=...)` from `agentic_context_langgraph` (see 4.9) | You want context injected into state automatically before each LLM turn. The factory returns an async LangGraph node you can `add_node(...)` directly onto your `StateGraph`. |
| Call `AsyncRetrievalClient.get_context(...)` (or the sync `RetrievalClient`) yourself, from `agentic_context` | You want manual control — fetch once at session start, cache it, format the prompt yourself. |

All three retrieval routes (all-in-one auto-fetch, `make_context_node`, manual `AsyncRetrievalClient.get_context`) hit the same Beacon endpoint (`GET /api/v1/agentic/context`) and accept the same `owner_id` / `agent_id` / `conversation_id` parameters.

### 4.3 Constructor

```python
from agentic_context import LangGraphAdapter  # Agentic Context SDK

# Path 1 (recommended) — all-in-one. Adapter owns its clients.
adapter = LangGraphAdapter(
    graph=my_compiled_graph,               # your existing LangGraph graph
    agent_id="my-agent",
    owner_id=auth.user_id,
    auth=auth,                             # UserContextAuth (or ContextAuth) — see §2
    dataspace="default",
    context_mode="auto",                   # or "per-turn", "manual" (default "auto")
)

# Path 1, static token (testing) — instead of auth=:
#   api_key="...", base_url="https://beacon.example.com"

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
| `agent_id` | yes | — | Must match a registered agent config in Beacon. Stamped on every emitted `Record`. |
| `graph` | one of | `None` | A compiled LangGraph graph (`StateGraph(...).compile()`), or a builder `(sdk_tools) -> compiled graph` that the adapter calls once with `default_tools()`. Use the builder to bind the SDK tools. Required for `stream()` and for the all-in-one path. Mutually exclusive with `pickle_path`. |
| `pickle_path` | one of | `None` | Path (`str` or `pathlib.Path`) to a pickled `StateSnapshot`. Mutually exclusive with `graph`. Requires `session_manager=` (forbidden with the all-in-one connection params). |
| `session_manager` | path 2/3 | `None` | A `SessionManager` (sync) or `AsyncSessionManager` (async) from `agentic_context`. Pass this OR connection params (`owner_id`/`api_key`/`base_url`/`dataspace`), not both — passing both raises `ValueError`. |
| `owner_id` | no | `None` | All-in-one mode: end-user identity (email, user ID, etc.). **Optional at construction (see 4.13)** — a multi-tenant server can omit it here and pass a per-turn `owner_id=` to `stream()` / `run()` instead. It must be resolvable *somewhere* by the time a turn runs, or the turn raises `ValueError`. Forbidden when `session_manager=` is set. |
| `auth` | path 1 | `None` | **Recommended.** All-in-one mode: a `UserContextAuth` or `ContextAuth` instance. Supplies the token and `base_url` and refreshes automatically. Takes precedence over inline credentials and `api_key`/`base_url`. |
| `private_key`, `consumer_key`, `subject`, `login_url` | path 1 | `""` | All-in-one mode, service-account alternative to `auth=`: pass all four and the adapter builds a `ContextAuth` for you. |
| `api_key` | path 1 | `None` | All-in-one mode, static-token path: CDP Bearer token. The adapter constructs an `AsyncIngestionClient` and `AsyncRetrievalClient` from this. Forbidden when `session_manager=` is set. |
| `base_url` | path 1 | `None` | All-in-one mode, static-token path: Beacon base URL. Forbidden when `session_manager=` is set. |
| `dataspace` | no | `"default"` | All-in-one mode: Beacon dataspace. Forbidden when `session_manager=` is set. |
| `context_mode` | no | `"auto"` | The one knob for all-in-one context management: `"auto"` (seed fast once + tool + built-in prompt), `"per-turn"` (seed fast every turn + tool), `"manual"` (never seed + tool + no prompt — you own it). Inert under `session_manager=` (that manager owns retrieval). See 4.11. There is no `context_policy=` knob — for cadences the three modes don't cover, use `resolve_policy` + `make_context_node` (4.10). |
| `context_header` | no | `"Previous context about this customer"` | Header line on the auto-injected SystemMessage block. |
| `context_message_id` | no | `"agentic-memory-context"` | Stable id on the injected `SystemMessage`. The `add_messages` reducer uses this id to replace (not duplicate) the block on subsequent injections. Pass `None` to opt out and get fresh ids. |
| `context_dedupe` | no | `False` | Drop repeated `(text, type)` facts in the rendered block. |
| `context_rank_urgent` | no | `False` | Surface facts with `type` in `("open_case", "sla_breach", "churn_risk", "escalation")` ahead of others. |
| `allow_pickle` | no | `False` | Must be set to `True` to actually load a `pickle_path`. Pickle deserialization executes arbitrary code; the gate forces a deliberate opt-in. See 4.8. |
| `user_participant_id` | no | resolved `owner_id` | `participant_id` written to `user_message` records. Defaults to the turn's owner, because Beacon marks an entry `role="owner"` only when `participantId == ownerId`. It falls back to `"user"` if no owner resolves. Override only if your Beacon setup registers a different participant. |
| `reasoning_classifier` | no | `None` | Callable `(AIMessage) -> "reasoning" \| "reply"` controlling how snapshot mode classifies an `AIMessage` that has both content and `tool_calls`. Default heuristic: any such message is reasoning. |
| `on_session_start` | no | `None` | Hook `(session_id: str) -> None`. Fires on **every** `stream()` call, before the graph runs. |
| `on_session_end` | no | `None` | Hook `(session_id: str \| None) -> None`. Fires from inside `close()` after the manager is closed. |
| `on_turn_start` | no | `None` | Hook `(TurnEvent) -> None`. Fires once per `stream()` call before the graph runs. |
| `on_tool_start` | no | `None` | Hook `(ToolEvent) -> None`. Fires when a LangChain tool runnable starts. Observation only — tool records are emitted whether or not it is set (see 4.7). |
| `on_tool_end` | no | `None` | Hook `(ToolEvent) -> None`. Fires when a tool ends successfully. |
| `on_error` | no | `None` | Hook `(ErrorEvent) -> None`. Fires when the stream loop or a tool callback raises (`phase` `"stream"` / `"tool"`), or on an ingest problem the adapter reports instead of raising (`phase="ingest"`): a cut-short turn's ingest failed, or records were produced after `close()` / `end_session()` already sent the turn. |

You must pass **exactly one** of `graph=` or `pickle_path=`. Passing both, or neither, raises `ValueError`. Hooks are observation-only — exceptions inside a hook are caught and logged; the stream loop never aborts because of a bad hook.

**Async context manager.** The all-in-one path supports `async with adapter:` — `__aenter__` opens the owned ingest and retrieval clients; `__aexit__` closes every per-session manager the adapter created, then closes the clients. Before closing managers it ingests any unfinished `stream()` turn (one the caller abandoned without `aclose()`). A turn still running in another task is cut off at that point: records it produces afterwards are not ingested and are reported to `on_error(phase="ingest")`. Calling `async with adapter:` on a manager-injection adapter raises `RuntimeError` because the adapter does not own those clients.

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

- `graph=` mode reads `graph.get_state({"configurable": {"thread_id": session_id}}).values["messages"]`.
- `pickle_path=` mode opens the file and `pickle.load`s it. Requires `allow_pickle=True` at construction; otherwise raises `ValueError`.
- Branches per message type:
    - `HumanMessage` → `user_message` (Message)
    - `AIMessage` with content + `tool_calls`, `reasoning_classifier` returns `"reasoning"` (default) → `agent_reasoning` (Trace) **plus** one `tool_call` (Trace) per call
    - `AIMessage` with content + `tool_calls`, `reasoning_classifier` returns `"reply"` → `agent_response` (Message) **plus** one `tool_call` (Trace) per call
    - `AIMessage` with `tool_calls` only → one `tool_call` per entry
    - `AIMessage` with content only → `agent_response` (Message)
    - `ToolMessage` → `tool_result` (Trace)
    - Other `BaseMessage` (e.g. `SystemMessage`) → `agent_reasoning` (Trace fallback)
- `tool_call` `event_id`s are deterministic: `step-tool-call-{session_id}-{idx}-{tc_idx}-{tool_name}`. The framework `tool_call.id` (when present) is preserved in `metadata.tool_call_id` for cross-record correlation but is **not** used as the `event_id`, because LangGraph populates it lazily.
- Re-snapshotting the same session writes its entries again — `event_id`s are stable locally but not sent, so nothing is deduped.

**Usage (sync manager):**

```python
from agentic_context import IngestionClient, SessionManager
from agentic_context import LangGraphAdapter  # Agentic Context SDK

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

Drives the graph and yields one or more `Record`s per turn. **Async only.** Available only in `graph=` mode — calling on a `pickle_path=` adapter raises `ValueError`.

**Signature:**

```python
async def stream(inputs: Any, session_id: str, *, owner_id: str | None = None) -> AsyncIterator[Record]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `Any` | The input dict you would normally pass to `graph.astream_events(...)`. The adapter inspects `inputs["messages"][-1]` to extract the user message. Recognized shapes: `HumanMessage`, `("human", text)` tuples, and dicts of the form `{"role": "user"\|"human", "content": ...}` **or** `{"type": "user"\|"human", "content": ...}` (LangChain's standard alias for `role`). |
| `session_id` | `str` | Used as the `thread_id` and stamped on emitted records. |
| `owner_id` | `str \| None` | Per-turn end-user identity (all-in-one path only). Overrides the constructor `owner_id` for this turn — lets one adapter serve many tenants. Resolution order: this argument → constructor `owner_id` → `ValueError`. Empty strings count as "not supplied". Passing it in manager-injection mode raises `ValueError` (the manager carries its own owner). See 4.13. |

**Yields:** `Record`s in the order they are produced. The adapter ingests every yielded record itself — the whole turn in one batch when the turn ends — so callers can `async for record in adapter.stream(...): pass` without doing their own ingest. **Do not call `adapter.ingest(...)` on records yielded by `stream()`** — they are already ingested, and re-ingesting writes duplicate entries (`event_id` is not sent, so nothing is deduped).

**Behavior:**

- Generates a fresh per-turn `interaction_id = f"turn-{session_id}-{turn_idx}"` (`turn_idx` increments across calls on the same adapter instance).
- Captures the running event loop so tool callbacks (which LangGraph may invoke on a worker thread) can schedule ingest tasks back onto it via `run_coroutine_threadsafe`.
- Yields a `user_message` Record first if `inputs` carries a recognizable user message; otherwise skips that step.
- Drives `await graph.astream_events(inputs, cfg, version="v3")` and iterates `stream.messages`. Each iteration corresponds to one fully-aggregated LLM utterance. The adapter classifies the awaited output:
    - Pending tool calls → `agent_reasoning` (Trace)
    - No pending tool calls → `agent_response` (Message), with metadata `{node, event: "utterance_complete", usage}` when available
- A single `stream()` call can yield multiple records — typically one `user_message`, any `tool_call` / `tool_result` records, and one or more reasoning/response records, in chronological order.
- Records are yielded in chronological order; a tool's records are yielded when the next model call starts streaming (or when the turn ends). The turn is ingested as **one batch — one `/entries` POST per turn** — when the turn ends. If the graph raises, the caller stops iterating early, or `close()` / `end_session()` runs while a turn is unfinished, the records captured so far are still ingested in that one batch. A process crash mid-turn loses the whole turn, user message included.
- If you `break` out of the loop, wrap the stream in `contextlib.aclosing(adapter.stream(...))` so the partial turn is ingested right away rather than at `close()`.
- If the underlying graph raises, the adapter fires `on_error(phase="stream")` and re-raises.
- Tool records (`tool_call`, `tool_result`) are captured by an internal callback handler that the adapter always registers on `cfg["callbacks"]`; no hook is required. See 4.7.
- The adapter constructs the LangGraph `RunnableConfig` itself (`{"configurable": {"thread_id": session_id}, "callbacks": [...]}`) and **does not merge** any callbacks the caller may want. If you have your own LangChain tracer or callback handler, attach it at graph compile time (`graph.with_config(config={"callbacks": [...]})`), not via the `inputs` argument to `stream()`.

**Usage:**

```python
import asyncio
from agentic_context import AsyncIngestionClient, AsyncSessionManager
from agentic_context import LangGraphAdapter  # Agentic Context SDK

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

**Parameters:** identical to `stream()` — `inputs` accepts the same shapes (string or dict), and `session_id` is keyword-only on `run()` to keep the call site readable. `owner_id` (keyword-only) is forwarded to `stream()` for per-turn multi-tenant owner resolution — see `stream()` and 4.13.

**Returns:** `list[Record]` in the order they were emitted. Records are already in flight to Beacon by the time `run()` returns; the list is for inspection (e.g., audit metadata, response shaping) and **must not** be re-ingested.

**Behavior:** identical to `stream()` — same context fetch, same graph drive, same per-turn ingest. The only difference is that `run()` awaits the entire turn before returning instead of yielding records as they emit.

**Usage:**

```python
import asyncio
from agentic_context import LangGraphAdapter  # Agentic Context SDK

async def main():
    async with LangGraphAdapter(
        graph=my_graph, agent_id="bot", owner_id=auth.user_id, auth=auth,
    ) as adapter:
        await adapter.run("Hey, I'm back", session_id="ses-2")

asyncio.run(main())
```

That is the entire integration in the all-in-one path. Without `auth=`, `api_key`, `base_url`, and `dataspace` come from `CDP_TOKEN` / `CDP_BASE_URL` / `CDP_DATASPACE` if not passed explicitly.

---

#### `default_tools`

Returns the SDK-provided tools to splat into your agent's tool list. Returns `[recall_customer_context, get_context_pack]` in every self-driving `context_mode` (`auto` / `per-turn` / `manual`, including the bare `auto` default), and `[]` under manager-injection (the injected manager owns retrieval). See 4.11.

**Signature:**

```python
def default_tools() -> list[BaseTool]
```

**Usage:**

```python
# Pass a builder as graph=; the adapter calls it once with default_tools().
def build_graph(sdk_tools):
    # Called once by the adapter with default_tools(), before it drives the graph.
    tools = [*sdk_tools, *my_tools]
    return make_my_graph(model.bind_tools(tools), ToolNode(tools))  # returns compiled graph

adapter = LangGraphAdapter(graph=build_graph, agent_id="bot", owner_id="x@y.com")
```

A pre-compiled graph has already bound its tools, so calling `adapter.default_tools()` after construction can't reach it. Use the builder form whenever you wire the SDK tools. The builder always receives `default_tools()` at graph-build time; the construction mode decides whether a tool is actually installed (self-driving → yes, manager-injection → no). This means you don't rewrite your graph wiring to switch modes.

---

#### `make_context_retrieval_tool`

Returns a LangChain `StructuredTool` wrapping an async `recall_customer_context(topic="")` — the on-demand, **agent-driven** counterpart to the per-turn context injection. Public regardless of `context_mode`, for callers who want the tool without the auto-wiring. See 4.11.

**Signature:**

```python
def make_context_retrieval_tool() -> StructuredTool
```

**Behavior:**

- Issues the same owner-scoped `GET /agentic/context` the per-turn path uses (~260 ms, pre-indexed profile facts). The optional `topic` is an LLM-visible hint only — it is not sent on the wire today.
- Resolves `session_id` from the per-invocation `RunnableConfig` (`configurable.thread_id`), so it is race-safe across concurrent multi-session streams — the LLM never supplies a conversation id.
- **Best-effort:** returns `{"userContext": []}` (never raises) when there is no owned retrieval client (manager-injection mode), the session can't be resolved, or the GET fails. Emits exactly one INFO audit log recording `owner`/`session`/`topic`/`has_context` — never the retrieved payload.

---

#### `end_session`

Closes and forgets the per-session manager(s), policy state, and turn counters for a `session_id`. **Async, all-in-one path only** — the manager-injection paths don't own the manager, so calling it there raises `RuntimeError`. Long-running adapters (web servers, schedulers) that drive many sessions through one instance should call this when a session ends so the pools don't grow unbounded. Local cleanup only — no `close_session` POST to Beacon.

**Signature:**

```python
async def end_session(session_id: str) -> None
```

Because the manager pool is keyed by `(session_id, owner)` (see 4.13), `end_session(sid)` drops **every** manager for that `session_id` regardless of which owner(s) drove it.

---

#### `ingest`

Direct ingest of a `list[Record]` through the bound manager. Bound at construction to either a sync or async implementation depending on the manager type.

**Signature:**

```python
# When session_manager is a SessionManager
def ingest(records: list[Record]) -> None

# When session_manager is an AsyncSessionManager
async def ingest(records: list[Record]) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `records` | `list[Record]` | Records in chronological order. Empty list is a no-op. |

**Returns:** `None`

**Behavior:**

- Sync manager: blocks until the HTTP POST returns (with retry on transient failures).
- Async manager: schedules a background delivery task and returns; `close()` is the synchronization point.
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

Closes the underlying session and releases resources. **Must be called on shutdown** — without it, pending async background deliveries are not drained and on-disk state may not be flushed.

**Signature:**

```python
# Sync
def close() -> None

# Async
async def close() -> None
```

**Behavior:**

- Async path: ingests any unfinished `stream()` turn, then calls `await mgr.close()`. The manager itself drains its own queue with a 30-second budget. Local cleanup only — no `close_session` is posted. Before closing managers it ingests any unfinished `stream()` turn (one the caller abandoned without `aclose()`). A turn still running in another task is cut off at that point: records it produces afterwards are not ingested and are reported to `on_error(phase="ingest")`.
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

#### Properties

```python
adapter.agent_id        # → str — the agent id passed at construction
adapter.capabilities    # → Capabilities(working=True, persistent=False)
adapter.mode            # → Literal["stream", "snapshot"]
```

`adapter.mode` reflects the construction path: `"stream"` when constructed with `graph=`, `"snapshot"` when constructed with `pickle_path=`. Streaming-mode adapters can also serve `snapshot()` reads from the live graph; the label reflects construction, not capability. Use `adapter.mode == "stream"` as a guard before calling `stream()` if your code accepts either construction path from upstream.

---

### 4.5 End-to-End Example: Snapshot

Snapshot is the simplest, most reliable model: let the agent run to completion, then ship the full session state in one batch. Use the sync stack — it removes a layer of concurrency you do not need.

```python
import os
from agentic_context import IngestionClient, SessionManager
from agentic_context import LangGraphAdapter  # Agentic Context SDK
from langgraph.graph import StateGraph  # your existing LangGraph code
from langgraph.checkpoint.memory import MemorySaver  # your existing LangGraph code

# --- Config ---
BEACON_URL = os.environ["BEACON_BASE_URL"]
BEACON_TOKEN = os.environ["BEACON_API_KEY"]
DATASPACE = os.environ.get("BEACON_DATASPACE", "default")
AGENT_ID = "support-bot"
OWNER_ID = "user@example.com"
SESSION_ID = "ses-2026-06-18-001"


def build_graph() -> StateGraph:
    # ... your graph construction; ensure it has a checkpointer so
    # graph.get_state(cfg) returns the full message history.
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

This makes four HTTP calls regardless of turn count: `create_conversation`, `create_session`, `add_participants`, one `add_entries`. No `close_session` is ever posted — `close()` is local cleanup only. Re-running the script writes duplicate entries — `event_id` is not sent, so nothing is deduped.

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
from langgraph.checkpoint.memory import MemorySaver  # your existing LangGraph code

# --- Config ---
BEACON_URL = os.environ["BEACON_BASE_URL"]
BEACON_TOKEN = os.environ["BEACON_API_KEY"]
DATASPACE = os.environ.get("BEACON_DATASPACE", "default")
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
    # build_graph(): your own factory returning a StateGraph — see 4.5 for an example.
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

**Non-blocking guarantee.** Inside `stream()`, the one per-turn `await mgr.ingest(turn_records)` returns in well under a millisecond — `AsyncSessionManager.ingest` schedules an `asyncio.create_task` and returns. The agent loop is never stalled by Beacon latency. `close()` is the synchronization point: it ingests any unfinished turn, then calls `await mgr.close()`, which `asyncio.wait`s pending deliveries with a 30-second budget and cancels stragglers. No `close_session` is sent to Beacon on close.

---

### 4.7 Hooks

All hooks are optional keyword arguments to the constructor. Hook exceptions are caught and logged via the adapter's internal `_safe_call` — a bad hook never aborts the stream.

| Hook | Signature | Fires when | Notes |
|------|-----------|------------|-------|
| `on_session_start` | `(session_id: str) -> None` | **Every** `stream(...)` call, before the graph runs. | Streaming only. The adapter does not deduplicate per-session — gate your own callback if you need once-per-session semantics. |
| `on_session_end` | `(session_id: str \| None) -> None` | Inside `close()`, after the manager closes (in a `finally` block). | Argument is the last session id seen by the adapter, or `None` if `close()` is called before any `stream()`/`snapshot()`. |
| `on_turn_start` | `(event: TurnEvent) -> None` | Each `stream(...)` call, after `on_session_start`, before the graph runs. | `TurnEvent` carries `session_id`, `inputs`, `turn_started_at`. |
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

> **Security warning.** `pickle.load` executes arbitrary Python code from the file. Only load pickles your own pipeline produced and stored under your control. If your input path is hostile or shared with untrusted writers, do **not** use this mode — switch to `graph=` (no deserialization) or wrap the producer/consumer in another transport. The adapter requires you to opt in explicitly via `allow_pickle=True`; loading without that flag raises `ValueError`.

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
from agentic_context import LangGraphAdapter  # Agentic Context SDK

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

### 4.9 Retrieval Helpers

The adapter package re-exports four retrieval helpers for injecting prior-conversation memory back into the graph:

| Helper | Purpose |
|--------|---------|
| `make_context_node(client, *, agent_id, ...)` | LangGraph node factory that calls Beacon's `GET /agentic/context` and writes the result into state. |
| `format_context_block(payload, ...)` | Pure formatter: Beacon context payload → prompt-friendly string. |
| `context_to_system_message(payload, ...)` | Pure formatter: Beacon context payload → `SystemMessage` (or `None` if empty). |
| `conversation_to_messages(payload)` | Rehydrate a `GET /agentic/conversations/{id}?scope=all` payload back into `list[BaseMessage]` (`HumanMessage` / `AIMessage(tool_calls=...)` / `ToolMessage`). |

#### `make_context_node`

```python
def make_context_node(
    client: AsyncRetrievalClient,
    *,
    agent_id: str,
    owner_id_key: str = "customer_id",
    conversation_id_from_config: bool = True,
    conversation_id_key: str = "thread_id",
    into: str = "messages",
    owner_type: str = "individual",
    header: str = "Previous context about this customer",
    message_id: str | None = "agentic-memory-context",
    dedupe: bool = False,
    rank_urgent: bool = False,
    urgent_types: tuple[str, ...] = DEFAULT_URGENT_TYPES,
) -> Callable[[dict, RunnableConfig], Awaitable[dict]]
```

Returns an async LangGraph node `(state, config) -> dict`. The factory does not take ownership of `client`; you keep it open and close it yourself (typically via `async with AsyncRetrievalClient(...)`).

Default behavior (`into="messages"`) returns `{"messages": [SystemMessage(...)]}` so the standard `add_messages` reducer merges the context into the thread. The `SystemMessage` carries a stable `id="agentic-memory-context"` so re-injecting on every turn replaces the previous copy instead of accumulating duplicates. Pass `into="<state_field>"` to land the formatted block as a plain string in a custom state field instead — useful when your existing system prompt already interpolates a templated context variable.

The node is best-effort: any exception (network, parse, auth) is logged and swallowed. The node returns `{}` (no state change) so a Beacon outage never breaks the user-facing run.

**Wiring example:**

```python
from agentic_context import AsyncRetrievalClient
from agentic_context_langgraph import make_context_node
from langgraph.graph import START, StateGraph  # your existing LangGraph code

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

By default the node reads the conversation id from `config["configurable"]["thread_id"]` (where the LangGraph runtime puts it) and the owner id from `state["customer_id"]`. Override `conversation_id_key`, `owner_id_key`, or `conversation_id_from_config=False` if your state schema differs.

#### `format_context_block` and `context_to_system_message`

Pure functions over a Beacon `GET /agentic/context` payload (`{"summary": "...", "userContext": [...]}`). Use them when you fetch context yourself rather than via `make_context_node`. `context_to_system_message` returns `None` when the payload renders to an empty block, so you can write `if msg: state.append(msg)`.

Both accept the same `dedupe`, `rank_urgent`, and `urgent_types` knobs. `dedupe=True` drops repeats of `(text.lower().strip(), type)` tuples (Beacon can echo the same fact across working + persistent memory). `rank_urgent=True` surfaces facts whose `type` is in `urgent_types` (default: `open_case`, `sla_breach`, `churn_risk`, `escalation`) ahead of others, preserving original order within each group.

#### `conversation_to_messages`

Takes the response of `GET /api/v1/agentic/conversations/{id}?scope=all` and rebuilds `list[BaseMessage]`. Useful when you want to seed a fresh `StateGraph` thread with prior history rather than rely on a SystemMessage summary. `tool_call` / `tool_result` entries are paired into `AIMessage(tool_calls=...)` and `ToolMessage` instances; `agent_reasoning` entries are folded into the next `AIMessage`'s content as `[reasoning] ...` lines. Orphan `tool_result` entries (no `tool_call_id`) are dropped — `ToolMessage` requires the id.

---

### 4.10 Context Policies (the escape hatch under `context_mode`)

The all-in-one path consults a `ContextPolicy` before each user turn to decide whether to call `GET /agentic/context`. You don't select one directly — `context_mode` (4.11) picks it for you (`auto` → `SessionFirstPolicy`, `per-turn` → `EveryTurnPolicy`, `manual` → `NoFetchPolicy`). Five built-ins ship with the adapter. For a cadence the three modes don't express — `TimeDecayPolicy`, `TopicShiftPolicy`, or a custom policy — there is **no constructor knob**: resolve one with the public `resolve_policy` helper and wire it into your own retrieval node via `make_context_node` (4.9). In manager-injection paths retrieval is likewise yours to wire.

| Policy | Slug | Behavior | Use when |
|--------|------|----------|----------|
| `SessionFirstPolicy` (default) | `"session_first"` | Fetch on the first call for a `session_id`, never again. The injected `SystemMessage(id="agentic-memory-context")` stays in state across subsequent turns via the `add_messages` reducer. | Default. One GET per session is enough for cross-session memory because the SystemMessage persists in state. |
| `EveryTurnPolicy` | `"every_turn"` | Fetch before every turn. The stable SystemMessage id makes the reducer replace (not duplicate) the prior block. | Highest fidelity, +1 GET per turn. Use when freshness matters more than latency. |
| `TimeDecayPolicy(ttl_seconds=60)` | `"time_decay"` | Cache the fetched payload for `ttl_seconds`; re-fetch on the first turn after expiration. | Bounded freshness with bounded cost. Tune `ttl_seconds` to your update cadence. Pass an instance for non-default TTL. |
| `TopicShiftPolicy(embedder, threshold=0.7)` | `"topic_shift"` | Embed the new user message and the cached `summary`; re-fetch when cosine similarity falls below `threshold`. Falls back to "fetch on first only" when no embedder is configured. | Customer agents whose conversations pivot — re-fetches when the topic shifts but otherwise reuses the in-state block. |

**Process-shared policy state.** Adapter instances share `last_fetched_at`, `last_payload`, `last_user_text`, and `turn_idx` through a class-level cache keyed by `(owner_id, agent_id, session_id)`. This is what makes `SessionFirstPolicy` honor its "one GET per session" contract under the common per-request adapter pattern (FastAPI handler constructs a fresh `LangGraphAdapter` per request) — without it, every request would see an empty state and `last_fetched_at is None` would always be true. State is automatically isolated per `owner_id`, so multi-tenant deployments don't leak across customers.

**Custom policies.** Implement the `ContextPolicy` Protocol:

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

The adapter feeds `should_fetch` the per-session state it has tracked (`turn_idx`, `last_user_text`, `last_payload`, `last_fetched_at`) and routes the decision to whatever your implementation returns.

```python
from agentic_context_langgraph import resolve_policy, TimeDecayPolicy, make_context_node

# Slug form — defaults to no-arg constructor.
policy = resolve_policy("time_decay")

# Instance form — pass arbitrary configuration.
policy = resolve_policy(TimeDecayPolicy(ttl_seconds=120))

# Wire the resolved policy into your own retrieval node; the adapter
# constructor does not take a policy kwarg. Call policy.should_fetch(...)
# yourself inside the node body (make_context_node handles the GET + inject).
node = make_context_node(client, agent_id=...)
```

A failed fetch (network outage, 4xx, 5xx) is logged and swallowed — `stream()` continues without an injected SystemMessage that turn. The graph still runs.

**Checkpointer requirement for `session_first` and `time_decay`.** Both policies fetch once and rely on the LangGraph `add_messages` reducer to keep the injected `SystemMessage` in state for subsequent turns of the same `session_id`. That reducer only persists across `stream()` calls when your graph was compiled with a checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver  # your existing LangGraph code
graph = builder.compile(checkpointer=MemorySaver())   # or SqliteSaver / PostgresSaver
```

Without a checkpointer, every `stream()` call starts from a fresh state — the SystemMessage from turn 1 vanishes and turn 2 sees no injected context. If your deployment cannot run a checkpointer (stateless serverless, one-shot calls), use `context_mode="per-turn"` (backed by `EveryTurnPolicy`) so the adapter re-injects on every turn.

`every_turn` and `topic_shift` work equally well with or without a checkpointer — they re-inject the latest payload each time they decide to fetch.

Empirical comparison on the three bug-bash scenarios (with `MemorySaver`, 3 sessions × {1,2,3} turns each):

| Policy | Total GETs | Avg Recall | Notes |
|--------|------------|------------|-------|
| `session_first` (default) | 6 | 83% | One GET per session. Misses only within-session fact updates (same `session_id`, new fact appearing turn 2+). |
| `every_turn` | 8 | 100% | +33% GETs vs `session_first`. Always sees the freshest payload. |
| `time_decay` (`ttl=60`) | 6 | 83% | Same shape as `session_first` for short benchmarks; diverges as scenarios grow longer than the TTL. |
| `topic_shift` (with embedder) | 8 | 100% | Matches `every_turn` here because the bench scenarios pivot topics every turn. |

---

### 4.11 Memory mode (the one knob)

`context_mode=` is the whole API for all-in-one context management — one value, no granular knobs. It composes the adapter's two levers: the **policy-driven per-turn injection** (4.10, the automatic fast seed) and the **agent-driven `recall_customer_context` tool** (4.4). Each mode picks a point in the (seed cadence, tool, injected prompt) space:

| `context_mode` | Adapter seeds fast context? | `recall_customer_context` tool | Injected instruction | Backing policy |
|---|---|---|---|---|
| `"auto"` **(default)** | once, on turn 1 | installed | built-in — "call it when you need memory" | `SessionFirstPolicy` |
| `"per-turn"` | **every turn** | installed | built-in — "retrieve fresh context every turn" | `EveryTurnPolicy` |
| `"manual"` | never | installed | none — **you own the prompt** | `NoFetchPolicy` |

```python
def build_graph(sdk_tools):
    # Called once by the adapter with default_tools(), before it drives the graph.
    tools = [*sdk_tools, *my_tools]
    return make_my_graph(model.bind_tools(tools), ToolNode(tools))  # returns compiled graph

async with LangGraphAdapter(
    graph=build_graph, agent_id="bot", owner_id="x@y.com",
    context_mode="auto",              # the default — shown for clarity
) as adapter:
    await adapter.run("Hey, I'm back", session_id="ses-2")
```

**Design notes:**

- **`auto` (default)** seeds fast profile context once per session and hands the agent the tool + a built-in prompt so it can recall more context when it decides to. A bare `LangGraphAdapter(graph=…)` resolves to `auto`.
- **`per-turn`** seeds fast context on **every** turn (a deterministic fallback — the adapter cannot force a tool call, so even if the agent ignores the tool the turn still has fresh context).
- **`manual`** never seeds and injects no prompt — the tool is installed but you own all prompting in your graph nodes.
- **Manager-injection makes the mode inert.** `context_mode=` and `session_manager=` compose without error, but an injected manager owns retrieval, so the adapter seeds nothing and `default_tools()` returns `[]`. `.context_mode` still reports the constructed value.
- **You compose the tool into your graph; the adapter never reaches in.** Your `graph=` builder receives `default_tools()` and binds it via `bind_tools`/`ToolNode`.

The tool issues the same fast owner-scoped read every call (see `make_context_retrieval_tool` in 4.4); the optional `topic` is an LLM-visible hint only and is not sent on the wire.

---

### 4.12 Delivery Modes

| Aspect | Snapshot (`graph=` or `pickle_path=`) | Streaming (all-in-one or `graph=`) |
|--------|---------------------------------------|---------------------------|
| Stack | Sync (`SessionManager` + `IngestionClient`) | Async (`AsyncSessionManager` + `AsyncIngestionClient`) |
| Trigger | One `snapshot()` + one `ingest()` after the agent finishes | Per-turn `async for` over `stream()` |
| HTTP calls per session | 4 (`create_conversation` + `create_session` + `add_participants` + 1×`add_entries`) | 3 + N (`create_conversation` + `create_session` + `add_participants` + N×`add_entries`), where N is the number of turns — one `add_entries` per turn; +1 `GET /agentic/context` per fetch decision in all-in-one mode. No `close_session` is posted. |
| Tool records | Always emitted from snapshot state | Always emitted, in the turn's batch |
| Crash recovery | Re-run the snapshot — duplicates already-written entries | Re-stream the session (or fall back to a snapshot) |
| Best for | End-of-session shipping, batch jobs, replays, file handoff between processes | Live agents needing near-real-time ingestion |

**Default to all-in-one streaming for new integrations.** The customer-facing surface is smallest and the adapter handles fetch + inject + post automatically. Drop to manager-injection when you need a sync stack, fine-grained control over the `SessionManager`, or pickle-based handoff between processes.

---

### 4.13 Dynamic Per-Turn Owner (Multi-Tenant)

By default the all-in-one adapter binds one `owner_id` at construction. A multi-tenant server that fronts many end-users behind one process can instead construct **one adapter** and pass a per-turn `owner_id=` to `stream()` / `run()`:

```python
# One adapter for the whole process — owner_id omitted at construction.
# One identity serves every tenant, so this is the service-account case.
async with LangGraphAdapter(
    graph=my_graph, agent_id="bot",
    auth=service_auth,                     # a ContextAuth — see §2
) as adapter:
    # Each request scopes its own tenant.
    await adapter.run("hi", session_id="ses-a", owner_id="alice@corp.com")
    await adapter.run("hi", session_id="ses-b", owner_id="bob@corp.com")
```

**Resolution order** (per turn): the `owner_id=` passed to that turn → the constructor `owner_id` → `ValueError`. Empty strings count as "not supplied" at both levels (a blank owner is a wire-level footgun). If neither is set, the turn raises rather than silently posting to a blank owner — the check lives at turn time, so construction without an owner is allowed.

**What the resolved owner scopes:**

- The per-turn `GET /agentic/context` `owner_id` param and the injected context block.
- The per-session **manager pool**, which is keyed by `(session_id, owner)`. The same `session_id` driven for two different owners gets two managers — two tenants never share a manager, and `SessionFirstPolicy` state stays isolated per owner (keyed by `(owner, agent_id, session_id)`). This holds **under concurrency**: the owner each `stream()` resolves is carried explicitly through the ingest path (not re-derived from a shared per-session slot), so two same-session/different-owner turns running at once each POST under their own `owner_id` and never cross-route.
- The **owner-less read paths** — the `recall_customer_context` tool and the fast context read receive only a `session_id` from the `RunnableConfig`, never an owner. The adapter records each session's owner when `stream()` runs and recovers it for these paths, falling back to the constructor owner for a session it never streamed. Because that recovery reads a single per-session slot, prefer passing `session_id` (which these paths already do via `RunnableConfig`) and treat the recovered owner as best-effort when the *same* `session_id` is concurrently driven for different owners.

**Manager-injection mode does not accept a per-turn owner** — the injected manager carries its own `owner_id`, so passing `owner_id=` to `stream()` there raises `ValueError`.

**Cleanup:** call `end_session(session_id)` on long-running adapters; it drops every `(session_id, owner)` manager plus the session→owner mapping and policy state (see `end_session` in 4.4).

---

### 4.14 Go-Live Checklist

- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] `owner_id` (on the manager, passed at construction, or passed per-turn to `stream()`/`run()` in all-in-one mode) is the real end-user identity (not a placeholder). Multi-tenant servers: omit it at construction and pass per-turn — see 4.13
- [ ] All-in-one only: the agent-driven `recall_customer_context` tool is installed in every self-driving `context_mode` — pass `graph=` a builder that binds the `sdk_tools` it receives (see 4.11)
- [ ] Picked the right construction path: all-in-one for new integrations (smallest surface), `graph=` + `session_manager=` for fine-grained control, `pickle_path=` only for cross-process file handoff with a trusted writer
- [ ] All-in-one only: chosen `context_mode` matches your latency-vs-freshness target (default `"auto"` = 1 fast GET per session; `"per-turn"` = 1 fast GET per turn). Wired `async with adapter:` around your turn loop so owned clients open and close cleanly
- [ ] All-in-one only: when `agent_id`+`owner_id`+`api_key`+`base_url` are passed, `session_manager=` must NOT be passed (and vice versa) — passing both raises `ValueError`
- [ ] If using `pickle_path=`, you have set `allow_pickle=True` deliberately and reviewed the trust boundary on the file
- [ ] Picked the right delivery pattern: streaming is the default for the all-in-one path; snapshot for batch shipping
- [ ] Tools are real LangChain runnables (`@tool`-decorated or equivalent); plain Python functions called inside a node do not fire tool callbacks
- [ ] `close()` is wired to your shutdown handler (sync: `adapter.close()`; async: `await adapter.close()`). Async `close()` awaits `mgr.close()` (30-second drain budget for pending deliveries)
- [ ] Token refresh strategy is in place. **There is no `update_token()` on `LangGraphAdapter`** — the underlying `IngestionClient` / `AsyncIngestionClient` captures the token at construction. To rotate, build a fresh client + manager + adapter and swap them in at a session boundary
- [ ] If a 401 / 403 surfaces, verify the dataspace + token combination; the underlying clients log a warning when `base_url` uses `http://` (no TLS) and accept it only for testing
- [ ] If you expect persistent-tier (`BaseStore`) data in Beacon, that is not in scope — the adapter declares `Capabilities(working=True, persistent=False)`. Ship store contents via a separate adapter

**Further reading:** the LangGraph section of `docs/DEVELOPER_GUIDE.md` is the integration guide and includes verification recipes for each mode.

---

## 5. Google ADK Adapter

### 5.1 Quickstart — 3 steps

**Step 1 — Install:**
```bash
pip install agent-context-sdk   # + google-adk, if your project doesn't already have it
```

**Step 2 — Construct with `auth=`:**
```python
from agentic_context import GoogleAdkAdapter  # Agentic Context SDK

adapter = GoogleAdkAdapter(
    agent_id="Agent_Data_Cloud",
    owner_id=auth.user_id,           # omit to auto-detect from session.user_id
    auth=auth,                       # UserContextAuth (or ContextAuth) — see §2
    dataspace="default",
)
```

`auth=` supplies the token and Beacon URL and refreshes automatically. There is no static `base_url`/`api_key` path on this adapter. The alternatives are the four inline service-account credentials (`private_key`/`consumer_key`/`subject`/`login_url`) or `agent_api_key=`. If the user's identity isn't known at construction time, leave `owner_id` unset: `wrap_before_model` auto-detects the real identity from `session.user_id` on the first turn and calls `set_owner()` automatically.

**Step 3 — Wire into LlmAgent:**
```python
from google.adk.agents import LlmAgent  # your existing Google ADK code

agent = LlmAgent(
    name="my_agent",
    model="gemini-3-flash-preview",
    instruction="You are a helpful support agent.",
    tools=[*adapter.default_tools()],
    **adapter.callback_kwargs(),
)
```

That's it. Context is fetched from Beacon and injected into the system prompt before every turn. Every turn is ingested automatically.

---

### 5.2 How It Works

The Google ADK adapter integrates via ADK's four callback hooks (`before_model`, `after_model`, `before_tool`, `after_tool`). It automatically ingests every turn and retrieves cross-session context before each LLM call — no manual wiring needed beyond construction.

- **`wrap_before_model`** — auto-detects the real user identity from `session.user_id`, calls `set_owner()` if needed, fetches `get_context()` from Beacon and injects it into the system prompt, and buffers the incoming user message as a `user_message` record
- **`wrap_after_model`** — awaits ingest directly inside the callback, flushing the full turn (user message + agent response + tool traces) to Beacon before the turn response returns — data is in Beacon before the next turn starts
- **`wrap_before_tool` / `wrap_after_tool`** — capture tool call inputs and results as `tool_call` / `tool_result` records

**Owner identity — two patterns:**

| Pattern | When to use | How |
|---------|-------------|-----|
| **Pass at construction** ✅ recommended | Production — user identity is known when the session starts | `GoogleAdkAdapter(owner_id="alice@example.com", ...)` — one adapter per user session |
| **Auto-detect** | Testing / server processes — one shared adapter handles many users | Leave `owner_id` unset; `wrap_before_model` reads `session.user_id` and calls `set_owner()` on the first turn |

> **Note for implementors:** The current TaskFlow integration uses auto-detect for testing convenience (one shared server process, no per-session restarts). Switch to the pass-at-construction pattern before production — create a `GoogleAdkAdapter` per user session with `owner_id=auth.user_id` so identity is explicit from the start.

### 5.2.1 Deterministic User Identification (ADK Limitation)

Google ADK's web UI hardcodes `user_id="user"` on every HTTP request regardless of URL parameters — the Angular frontend creates the session before any `?userId=` query param is parsed, so the real user identity never reaches the agent via standard ADK mechanisms.

To work around this, the TaskFlow integration uses a **FastAPI Referer middleware** in `main.py`: the login page redirects to `/dev-ui/?app=taskflow&userId=<email>`, which keeps the email in the browser's URL bar. Every subsequent same-origin request from the ADK UI includes a `Referer` header containing that email — the middleware extracts it and rewrites both the session path (`/users/user/sessions` → `/users/<email>/sessions`) and the `run_sse` request body (`user_id: "user"` → `user_id: "<email>"`) before the request reaches ADK. This means `session.user_id` always carries the real email by the time `wrap_before_model` fires, so `set_owner()` gets the correct identity deterministically — no LLM prompt extraction involved.

> **Do not rely on the LLM to extract the user's email from their messages.** The LLM can misidentify a third-party email mentioned in conversation (e.g. "forward this to bob@...") as the user's own identity. Always derive `owner_id` from an authenticated source — a session token, cookie, or middleware-rewritten header as shown here.

### 5.3 Constructor

```python
from agentic_context import GoogleAdkAdapter  # Agentic Context SDK

# Production pattern — one adapter per signed-in user, identity explicit at construction
adapter = GoogleAdkAdapter(
    agent_id="Agent_Data_Cloud",
    owner_id=auth.user_id,
    auth=auth,                       # UserContextAuth (recommended) or ContextAuth
    dataspace="default",
)
```

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `agent_id` | `str` | ✅ | — | Must match a registered agent config in Beacon |
| `owner_id` | `str \| None` | — | `None` | End-user identity (`auth.user_id` with `UserContextAuth`). Pass it at construction in production. Leave unset only in shared-adapter setups — `wrap_before_model` auto-detects from `session.user_id`; with no owner resolved, Beacon reads/writes for that session are skipped |
| `auth` | `UserContextAuth \| ContextAuth \| None` | one of | `None` | **Recommended.** Supplies the token and Beacon URL; refreshes automatically. Takes precedence over the options below |
| `private_key`, `consumer_key`, `subject`, `login_url` | `str` | one of | `""` | Service-account alternative to `auth=`: pass all four and the adapter builds a `ContextAuth` |
| `agent_api_key` | `str` | one of | `""` | Platform-hosted API-key path (`secret_server_url=` or `AGENTIC_CONTEXT_SECRET_SERVER_URL`) |
| `dataspace` | `str \| None` | — | `"default"` | Beacon dataspace name (falls back to `CDP_DATASPACE`) |

With none of `auth=`, the inline credentials, or `agent_api_key=`, construction raises `ValueError`. There is no `base_url`/`api_key` static-token path on this adapter.

### 5.4 Method Reference

---

#### `set_owner`

Binds the adapter to a specific user identity. Call as soon as the user's identity is known — all subsequent Beacon writes and `get_context` calls are scoped to that user.

**Signature:**

```python
def set_owner(owner_id: str) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `owner_id` | `str` | End-user identity (email, user ID, etc.) |

**Notes:**
- Resets the adapter's conversation/session state so the next turn looks up or creates a Beacon conversation for the new owner
- Any records already buffered in the current turn under a placeholder ID are patched to the real owner ID before they flush — so even the email-collection turn lands with the correct participant ID
- Call this once per conversation when the user identifies themselves; call it again if the user changes (e.g. multi-tenant server handling a new user on the same process)

**Usage:**

```python
# In your session-start tool — after the user provides their email:
def start_session(customer_email: str) -> dict:
    adapter.set_owner(customer_email)
    return {"status": "success"}
```

---

#### `get_context`

Retrieves cross-session context for the current user from Beacon. Call before each LLM turn and inject the result into the system prompt.

**Signature:**

```python
async def get_context() -> dict | None
```

**Returns:** A context dict from Beacon, or `None` if no prior context exists (new user / DLO sync not yet run).

**Notes:**
- Called automatically by `wrap_before_model` — you only need to call this directly if you are not using `wrap_before_model`
- Requires `set_owner()` to have been called with the real user identity; returns `None` if still on a placeholder

---

#### `make_start_session_tool`

Returns a ready-made ADK `FunctionTool` that binds the adapter to the user's identity. Drop it directly into `LlmAgent`'s `tools` list — no boilerplate function required.

The LLM decides when to call it based on the tool's docstring, which explicitly instructs it to only fire when the user provides **their own** email address — not when an email appears in any other context (e.g. "forward this to bob@...").

**Signature:**

```python
def make_start_session_tool() -> FunctionTool
```

**Returns:** An ADK `FunctionTool` wrapping a `start_session(customer_email: str)` closure.

**Usage:**

```python
agent = LlmAgent(
    ...
    tools=[adapter.make_start_session_tool()],
    **adapter.callback_kwargs(),
)
```

**Note:** If your `start_session` needs to do more than identity binding (e.g. create a CRM record, log metrics), write your own function and call `adapter.set_owner(email)` inside it instead.

---

#### `wrap_before_model` / `wrap_after_model` / `wrap_before_tool` / `wrap_after_tool`

Returns an ADK-compatible callback that can be passed directly to `LlmAgent`. Each method accepts an optional `existing=` parameter to compose with your own callbacks.

**Signature:**

```python
def wrap_before_model(existing: Callable | None = None) -> Callable
def wrap_after_model(existing: Callable | None = None) -> Callable
def wrap_before_tool(existing: Callable | None = None) -> Callable
def wrap_after_tool(existing: Callable | None = None) -> Callable
```

Your existing callback runs first; the adapter's ingestion logic runs after. Both sync and async existing callbacks are supported.

---

#### `close`

Closes the Beacon session and HTTP client. Call when the conversation or process ends.

**Signature:**

```python
async def close() -> None
```

---

### 5.5 End-to-End Example: Google ADK Support Agent

Complete example — one adapter per signed-in user, `UserContextAuth` passed at construction:

```python
import asyncio
from google.adk.agents import LlmAgent  # your existing Google ADK code
from google.adk.runners import Runner  # your existing Google ADK code
from google.adk.sessions import InMemorySessionService  # your existing Google ADK code
from google.genai import types as genai_types  # your existing Google ADK code
from agentic_context import GoogleAdkAdapter, UserContextAuth  # Agentic Context SDK

CONSUMER_KEY = "3MVG..."
LOGIN_URL = "login.salesforce.com"
BEACON_AGENT_ID = "your-registered-agent-id"
BEACON_DATASPACE = "default"


async def run_session(auth: UserContextAuth, user_message: str) -> str:
    user_id = auth.user_id
    # One adapter per signed-in user — owner_id comes from the auth
    adapter = GoogleAdkAdapter(
        agent_id=BEACON_AGENT_ID,
        owner_id=user_id,
        auth=auth,
        dataspace=BEACON_DATASPACE,
    )

    agent = LlmAgent(
        name="support_agent",
        model="gemini-3-flash-preview",
        instruction="You are a helpful support agent.",
        **adapter.callback_kwargs(),
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="myapp",
        user_id=user_id,
    )

    runner = Runner(agent=agent, app_name="myapp", session_service=session_service)

    message = genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
    response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=message
    ):
        if hasattr(event, "content") and event.content:
            for part in (event.content.parts or []):
                if getattr(part, "text", None):
                    response += part.text

    await adapter.close()
    return response


# Multi-session memory: same signed-in user, two separate sessions
async def main():
    auth = UserContextAuth.login(
        consumer_key=CONSUMER_KEY, login_url=LOGIN_URL, aid=BEACON_AGENT_ID,
    )

    # Session 1
    resp1 = await run_session(auth, "I'm on the Pro plan and have a billing issue")
    print(f"Session 1: {resp1}")

    # Session 2 — Beacon returns context from Session 1
    resp2 = await run_session(auth, "What did we talk about last time?")
    print(f"Session 2: {resp2}")

asyncio.run(main())
```

The adapter handles context retrieval and ingestion automatically on every turn. Create a new `GoogleAdkAdapter` instance per signed-in user — passing `auth.user_id` as `owner_id` is the only required step for identity binding.

### 5.6 Multi-Session Memory

Each ADK session creates a fresh Beacon conversation (`conversation_id = session_id`), matching the convention used by the LangGraph and Claude adapters. Cross-session memory is returned by Beacon's owner-level context cache — `get_context()` scopes results to `owner_id` across all conversations for that user.

```python
# Session 1 — create adapter for alice, run conversation
adapter1 = GoogleAdkAdapter(agent_id="my-agent", owner_id="alice@example.com", ...)
# ... turns flushed to Beacon per turn ...
await adapter1.close()

# Session 2 — new adapter instance, same owner_id
# Beacon returns prior context from Session 1 automatically via wrap_before_model
adapter2 = GoogleAdkAdapter(agent_id="my-agent", owner_id="alice@example.com", ...)
# → get_context() returns facts from Session 1 on first turn
```

### 5.7 Go-Live Checklist

- [ ] `agent_id` matches a registered agent config in Beacon
- [ ] Auth supplied: `auth=` (`ContextAuth` / `ApiKeyAuth`), the four inline credentials, or `agent_api_key=`
- [ ] ADK sessions created with the real user email as `user_id` — `wrap_before_model` auto-detects it via `session.user_id`
- [ ] `callback_kwargs()` unpacked into `LlmAgent` — wires all four callbacks
- [ ] `get_context()` returns `userContext` entries (confirms agent config exists in Beacon and prior data is ingested)
- [ ] (Optional) `close()` called on process shutdown to release HTTP connections — not required for data safety since ingestion is synchronous per turn

---

## 6. AgentContextClient

> **Preview.** Exported from `agentic_context` but not yet semver-committed.

Deep reference for `agentic_context.AgentContextClient` — the adapter-free client
(Design.md §8 for why lifecycle vs mid-turn tools). Import from
`agentic_context`; no framework SDK required. Same Beacon stack and `auth`
objects as the adapters (`ContextAuth` / `UserContextAuth`). For the
customer-facing integration guide (API reference, wiring patterns), see
[AGENT_CONTEXT_CLIENT.md](AGENT_CONTEXT_CLIENT.md).

```python
from agentic_context import AgentContextClient

ace = AgentContextClient()
opened = ace.start_conversation(title="chat", auth=auth, agent_id=agent_id)
# Inject opened["system_prompt_suffix"]; register ace.tool_specs()
ace.ingest_customer_message(user_text)
# on model tool call: ace.call_tool(name, arguments)
ace.ingest_agent_message(agent_reply)
```

### 6.1 Lifecycle methods

| Method | Role | Notes |
|--------|------|-------|
| `authenticate(...)` | Bind auth / agent / owner | Usually folded into `start_conversation`. |
| `start_conversation(conversation_id=None, *, title="", auth=..., agent_id=..., owner_id=..., owner_type=..., ...)` | Open + seed | Auth if needed; `get_initial_context` → `POST /api/v1/agentic/context` + `GET …/runtime-context-packs`; create conversation/session; returns `system_prompt_suffix`, `context`, `pack_ids`. Idempotent rebound when already open. |
| `ingest_customer_message(content, conversation_id=None, ...)` | Pre-turn write | `POST …/entries` as customer. Omit `conversation_id` when bound. |
| `ingest_agent_message(content, conversation_id=None, ...)` | Post-turn write | `POST …/entries` as agent. |

Lower-level host helpers (prefer `start_conversation` unless splitting open):
`allocate_session_id`, `get_initial_context`, `open_conversation`,
`bind_conversation`, `ingest`, `list_conversations`, `get_conversation`.
`reset()` clears local sessions + auth (no Beacon close) — same hygiene
pattern as adapter `close()`.

### 6.2 Mid-turn tools

| Method | Role | Notes |
|--------|------|-------|
| `tool_specs()` | Schema list | Deep copy of `ace_get_context` + `ace_get_context_pack` only. |
| `call_tool(name, arguments=None)` | Dispatch | Host lifecycle methods are **not** valid here. Strips `owner_id` / `agent_id`; normalizes camelCase. |
| `tool_skill_name()` / `tool_skill_dir()` | Skill packaging | Skill id `ace-mid-turn-tools`. |
| `get_context` / `get_context_pack` | Direct equivalents | Same Beacon calls `call_tool` uses; prefer `call_tool` when wiring model tools. |

**`call_tool` / pack status values**

| `status` | Meaning |
|----------|---------|
| `ok` | Use `context` |
| `not_ready` | Transport / preview failure — fall back; do not retry in a loop |
| `invalid_pack` | Unknown id — see `valid_pack_ids` |

### 6.3 Identity and concurrency

- One `AgentContextClient` instance per auth identity (`agent_id` / `owner_id` /
  `owner_type` / `dataspace`). Passing a different identity after
  authenticate raises. A later `auth` object must be provably the same
  identity without a token fetch: same auth class, agent id, login host,
  consumer key, `subject` / `xid`, and, for `UserContextAuth`, a resolved
  `user_id` equal to the bound one. A fresh `UserContextAuth` (no `user_id`
  yet) raises. Pass the bound auth object, or omit `auth`.
- Not thread-safe. The active conversation and open-turn state have no
  locks, and the HTTP clients share one `requests.Session`. Don't run
  lifecycle calls on one instance from several threads (or concurrent
  `asyncio.to_thread` calls) at once. LangGraph's `ToolNode` runs parallel
  tool calls from one model reply in threads against the same instance.
  The mid-turn tools only read state, but `requests.Session` is not
  documented as thread-safe.
- Pass `conversation_id` when multiple conversations are open on the same
  client; omit it when a single conversation is bound.
- `owner_id` for Beacon customer context can differ from the auth subject
  (host-supplied); mid-turn tools must not override it.

### 6.4 Demos

- `examples/client_direct_demo/` — direct method calls from a LangGraph host
- `examples/client_hooks_demo/` — lifecycle hooks + MCP mid-turn tools on
  Claude Agent SDK

Also: [QUICKSTART.md](./QUICKSTART.md), Design.md §8.

---


## 7. Troubleshooting

### HTTP 401 Unauthorized

With `auth=` (`UserContextAuth` or `ContextAuth`) or inline credentials, the SDK
refreshes the token automatically, both before expiry and on a `401`. A
persistent 401 there means the credentials themselves are wrong: check the
consumer key, login URL, scopes, and that `agent_id` is registered.

With `UserContextAuth`, a `ReauthRequired` exception means the refresh token was
revoked or expired. Send the user back through `UserContextAuth.login()` (or your
backend authorize flow).

On the LangGraph static-token path (`api_key=` + `base_url=`), the token is
captured at construction. To rotate it, build a fresh client + manager + adapter
with the new token and swap them in at a session boundary.

### `get_context()` Returns `None` or Empty `userContext`

| Cause | Fix |
|-------|-----|
| Agent config not registered in Beacon | Register your `agent_id` in Beacon admin |
| Token expired | Automatic with `auth=`; on the LangGraph static-token path, rebuild the adapter with a fresh token |
| No data ingested yet | Ingest at least one turn first |
| `set_owner()` not called with real user identity | Call `set_owner(email)` before the first turn; context is keyed on `owner_id` |
| Cross-session `get_context()` returns empty | Known Beacon server limitation — owner-level context cache not yet active in test environments. In-session context works correctly. |

### Records Not Appearing

| Check | Fix |
|-------|-----|
| Is `agent_id` correct? | Must match exactly (case sensitive) |
| Is `dataspace` correct? | Default is `"default"` |
| Is `close()` being called? | Without it, background POSTs may not complete |
| Are you getting errors in logs? | Enable debug logging (see below) |

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("agentic_context").setLevel(logging.DEBUG)
```

### "Session already closed" Error

You called `push()` after `close()`. The lifecycle is one-directional:

```
start → push → push → ... → close  ✓
start → push → close → push        ✗ Error
```

---

**End of Advanced Guide**
