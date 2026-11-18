# Agentic Context SDK — Quickstart

Give your agent cross-session context in one screen. Deep reference lives in
[ADVANCED.md](ADVANCED.md); prerequisites in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## Install

```bash
pip install agent-context-sdk
```

That's the whole install. All five adapters ship inside this one distribution.
The agent-framework SDKs themselves (`claude-agent-sdk`, `google-adk`,
`langgraph`/`langchain-core`, `agent-framework-foundry`, `openai-agents`) are
**not** bundled — you're expected to already have the one your agent runs on.
If you don't, install it directly, e.g. `pip install claude-agent-sdk`.
Requires Python 3.10+.

The distribution is `agent-context-sdk`; the import package is `agentic_context`.
Every adapter is re-exported from the top-level package, so one import line
reaches all five:

```python
from agentic_context import (
    ClaudeAdapter, GoogleAdkAdapter, LangGraphAdapter, FoundryAdapter, OpenAIAdapter,
)
```

These are resolved lazily — importing a name only loads that framework's SDK, so
you can pull `ClaudeAdapter` without LangGraph/ADK present. If a framework SDK is
missing, the adapter tells you exactly which package to install. Per-package
imports work too if you prefer them — `from agentic_context_claude import
ClaudeAdapter`, `from agentic_context_adk import GoogleAdkAdapter`,
`from agentic_context_langgraph import LangGraphAdapter`,
`from agentic_context_foundry import FoundryAdapter`, and
`from agentic_context_openai import OpenAIAdapter`.

## Credentials

Every adapter authenticates through a Salesforce Connected App. Pick the
flow that matches **who is calling** — a backend service, or an end user —
then build one `auth` instance and share it across all five adapters below.

### Option A — `UserContextAuth.login()` (Salesforce user, OAuth + PKCE)

Opens the Salesforce login page in a browser, catches the callback on a short-lived
loopback listener, and returns an `auth` holding the user's refresh token.

```python
from agentic_context import UserContextAuth

auth = UserContextAuth.login(
    consumer_key="3MVG...",                  # Connected App consumer key
    login_url="login.salesforce.com",        # login host, no scheme
    aid="your_agent_id",                     # Beacon-registered agent id
)
end_user_id = auth.user_id                   # 005… of the signed-in user
```

The ECA referenced by `consumer_key` must register
`http://localhost:53682/callback` as a callback URL, enable the web-server
flow with **Require secret for web server flow** → off, and grant the scopes
`api refresh_token cdp_ingest_api`. Pass a different `port=` to
`UserContextAuth.login()` if 53682 is unavailable; register the matching
callback.

Salesforce may rotate the refresh token on each refresh. For a long-running
process, hand `UserContextAuth.login()` an `on_refresh_token=…` callback and
persist the new value — otherwise the token lives only in this process. For a
backend serving users (not a local CLI), skip `.login()` and host your own
callback with `UserContextAuth.authorization_request()` /
`UserContextAuth.redeem_code()`; see [ADVANCED.md](ADVANCED.md) for the full
backend flow and `RefreshTokenStore` wiring.

A `ReauthRequired` exception from any adapter call means the refresh token
has been revoked — route the user back through `UserContextAuth.login()`.

### Option B — `ContextAuth` (Service account, JWT bearer)

Backend agent acts for every call under one service identity. No browser; the
Connected App's RSA private key proves identity.

```python
from agentic_context import ContextAuth

auth = ContextAuth(
    private_key=open("server.key").read(),  # PEM RSA private key
    consumer_key="3MVG...",                  # Connected App consumer key
    subject="you@example.com",               # Salesforce username (JWT bearer)
    login_url="login.salesforce.com",        # login host, no scheme
    aid="your_agent_id",                     # Beacon-registered agent id
)
```

**Note:** `ContextAuth` has no `user_id` attribute, so pass the end user's id
explicitly as `owner_id` on each adapter. Optionally pass `xid=end_user_id` to
scope the token to that owner. One `ContextAuth` caches one token, so only do
that for a single-owner deployment; multi-user services need one instance per
owner.

## Wiring an adapter

### Claude Agent SDK

```python
from claude_agent_sdk import ClaudeAgentOptions, query  # your existing Claude Agent SDK code
from agentic_context import ClaudeAdapter  # Agentic Context SDK

adapter = ClaudeAdapter(agent_id=your_agent_id, owner_id=end_user_id, auth=auth)

options = ClaudeAgentOptions(model="<claude-model>")  # your existing Claude Agent SDK code
options = adapter.apply_to(options, session_id=session_id)

async for message in query(prompt=user_prompt, options=options):  # your existing Claude Agent SDK code
    ...


```

Prefer to wire the pieces yourself? Skip `apply_to`, build
`adapter.mcp_server(...)` and `adapter.hooks(session_id, owner_id)` yourself, and
pass them to `ClaudeAgentOptions`. `apply_to` does both for you and leaves your
`system_prompt` alone. `adapter.context_system_prompt_guidance()` is optional
text you can append to your own prompt to teach the agent when to recall
memory.

### LangGraph

```python
from agentic_context import LangGraphAdapter  # Agentic Context SDK

def build_graph(sdk_tools):  # your existing LangGraph code, wrapped in a function
    tools = [*sdk_tools, *my_tools]
    model_with_tools = model.bind_tools(tools)
    ...  # add nodes (e.g. ToolNode(tools)) and edges as usual
    return builder.compile()

async with LangGraphAdapter(
    graph=build_graph,
    agent_id=your_agent_id,
    owner_id=end_user_id,
    auth=auth,
) as adapter:
    await adapter.run(user_input, session_id=session_id)
```

`graph=` takes a builder: the adapter calls it once with its SDK tools
(`recall_customer_context` and `get_context_pack`) and drives the compiled graph
it returns. That way the tools are bound to your model before the graph is
compiled. Add your own tools alongside them in the same list. If your graph
doesn't need the SDK tools, you can pass an already-compiled graph instead.

Pass `auth=auth` just like the other adapters; one `auth` instance (either
kind) is shared across adapters. If you omit it, the adapter falls back to
`api_key`/`base_url`/`dataspace` from the `CDP_TOKEN`/`CDP_BASE_URL`/`CDP_DATASPACE`
env vars. `run()` drives one turn and ingests all of its records (user message,
reasoning, tool calls/results, reply) in one POST when the turn ends; use
`async for r in adapter.stream(...)` to see each record as it's produced.

### Google ADK

```python
from google.adk.agents import LlmAgent  # your existing Google ADK code
from agentic_context import GoogleAdkAdapter  # Agentic Context SDK

adapter = GoogleAdkAdapter(agent_id=your_agent_id, owner_id=end_user_id, auth=auth)

agent = LlmAgent(
    name="my_agent",
    model="<gemini-model>",
    tools=[*adapter.default_tools()],
    **adapter.callback_kwargs(),
)
```

Context is fetched from Beacon and every turn is ingested automatically via the
ADK callbacks. `default_tools()` adds the agent-driven `recall_customer_context`
and `get_context_pack` tools.

### Microsoft Foundry

```python
from agent_framework import Agent, AgentSession  # your existing Microsoft Foundry code
from agent_framework_foundry import FoundryChatClient  # your existing Microsoft Foundry code
from agentic_context import FoundryAdapter  # Agentic Context SDK

adapter = FoundryAdapter(agent_id=your_agent_id, owner_id=end_user_id, auth=auth)
agent = Agent(FoundryChatClient(), instructions=MY_PROMPT)
session = AgentSession(session_id=session_id)
await adapter.run(agent, user_input, session=session)
```

The first `run`/`stream` for a `session_id` seeds FAST context once. Every turn
registers `recall_customer_context` and `get_context_pack`. There is no
`context_mode` knob.

### OpenAI Agents SDK

```python
from agents import Agent, Runner  # your existing OpenAI Agents SDK code
from agentic_context import OpenAIAdapter  # Agentic Context SDK

adapter = OpenAIAdapter(agent_id=your_agent_id, owner_id=end_user_id, auth=auth)

agent = Agent(
    name="my_agent",
    model="<gpt-model>",
    instructions=adapter.instructions_provider(
        base=MY_PROMPT,
        session_id=session_id,
    ),
    tools=[*adapter.default_tools(session_id)],
)

hooks = adapter.run_hooks(session_id, user_input=user_input)
try:
    result = await Runner.run(agent, user_input, hooks=hooks)
finally:
    await hooks.flush()
```

`instructions_provider` fetches the agent's context-pack catalog once per
`session_id` and appends `[Available context packs]` to `base` on every turn.
`run_hooks(session_id, user_input=...)` captures the run as one turn: the user
message, the agent's replies, reasoning, and tool calls share one
`interaction_id` and go to Beacon in one POST when the run finishes. The hooks
never see the run's input, so pass it as `user_input`. `flush()` in `finally`
posts a run that ended without a final output (an exception, max turns); after
a normal finish it is a no-op. With `Runner.run_streamed`, put the
`stream_events()` loop inside the `try` instead, so `flush()` runs after the
stream is consumed. `default_tools()` adds
`recall_customer_context` and `get_context_pack`. Rebuild (or re-call
`instructions_provider`) per new `session_id`.

> **Why `base` flows through the adapter.** `Agent(instructions=…)` is a single
> slot — the string (or callable returning the string) is the whole system
> message. Claude's `ClaudeAgentOptions` splits this in two: `system_prompt`
> (customer, static) and the `UserPromptSubmit` hook's `additionalContext`
> (adapter, dynamic), merged by the SDK at turn time. OpenAI has no second
> slot, so whatever emits `instructions` emits the whole thing — the adapter
> needs `base` to produce `f"{base}\n\n{pack_list_block}"`.


## Context packs (preview)

A **context pack** is a named bundle of structured detail — a plan's rules, a
customer's claims — that the agent pulls on demand, in addition to the fast
context seeded at conversation open. Every adapter offers a `get_context_pack`
tool, so any agent built from this Quickstart can offer packs. The tools you
already wired include it: `default_tools()` on ADK, OpenAI, and Claude
(`apply_to` registers it for you), the `sdk_tools` passed to your `graph=`
builder on LangGraph, and `run` / `stream` on Foundry.
LangGraph and Foundry also have an adapter method, `get_context_pack_list()`,
that reads the catalog from `GET /api/v1/agentic/runtime-context-packs`. It is
a method, not a tool.

The tool accepts only the pack ids in the server-advertised catalog. An id
outside it is rejected without a network call. Where the adapter takes a
`valid_pack_ids` list, ADK, LangGraph, and OpenAI use it only as the fallback
for when the catalog is empty. Foundry is different: a `valid_pack_ids` you pass
to `make_get_context_pack_tool(...)` replaces the catalog, so an id that is in
the catalog but not in your list is rejected. Claude always uses the catalog,
which the `UserPromptSubmit` hook fetches. `make_get_context_pack_tool(...)` builds
the tool on its own if you assemble a custom tool list. Its arguments differ per
adapter:

| Adapter | `make_get_context_pack_tool` arguments |
|---|---|
| Google ADK, LangGraph | `(valid_pack_ids, *, param_defaults, timeout)` |
| OpenAI Agents SDK | `(valid_pack_ids, session_id, owner_id, *, param_defaults, timeout)` |
| Microsoft Foundry | `(session_id, owner_id, *, valid_pack_ids, param_defaults, timeout)` |
| Claude Agent SDK | `(session_id, owner_id, *, param_defaults, timeout)` — no id list |

The tool is **best-effort and never raises**. It returns one of:

- `{"status": "ok", "context": {...}}` — the pack payload.
- `{"status": "not_ready", "detail": ...}` — the pack endpoint isn't live here
  yet (this is a **preview** capability), or the read failed/timed out.
- `{"status": "invalid_pack", "valid_pack_ids": [...]}` — the id isn't advertised.
- `{"status": "invalid_version", "message": ...}` — the model passed a
  `context_pack_version` that isn't an integer (Claude, Google ADK, LangGraph,
  OpenAI; Foundry's tool schema types it as an integer).
- `{"status": "version_expired", "latest_version": ..., "latest": ...}` — the
  pack has been republished since this agent saw the catalog. See
  [Context-pack versioning](#context-pack-versioning-preview) below.

The owner/subject is bound **server-side** from the resolved session — the LLM
only ever supplies `context_pack_id`, an optional `params`, and a
`context_pack_version` (**required** whenever the catalog entry advertises a
`version`; see below).

Every adapter also injects `CONTEXT_PACK_INSTRUCTION` for you.
It is a shared, domain-neutral system-prompt block that tells the model when to
reach for a pack, to fetch once and reuse, to echo `context_pack_version`
whenever the catalog carries one, and to degrade gracefully on `not_ready` /
`version_expired`. It goes ahead of the `[Available context packs]` catalog and
after your own prompt. It's only added when the session's catalog lists at
least one pack, since otherwise there's nothing to fetch. **Don't add it to your
prompt yourself.**

The first turn of each session fetches the server-published copy
(`GET /api/v1/agentic/prompts/system/context_pack_instruction`) and pins it, so
the text never changes mid-conversation. If the fetch fails or the server has no
copy, the session uses the bundled constant instead and the turn carries on.
After a failed fetch, new sessions skip the fetch for 10 s, so a Beacon outage
doesn't slow every new session's first turn.

Where the instruction lands, per adapter:

| Adapter | Injected by | Turn it off |
|---|---|---|
| Google ADK | `callback_kwargs()` (the system instruction) | — |
| Claude Agent SDK | the `UserPromptSubmit` hook (`additionalContext`) | `apply_to(..., add_system_prompt_guidance=False)` or `hooks(..., inject_context_pack_instruction=False)` |
| LangGraph | the injected `SystemMessage` that carries the catalog | — |
| OpenAI Agents SDK | `instructions_provider(base=...)` | — |
| Microsoft Foundry | `context_provider()` / `run()` | — |

Most adapters add the instruction and catalog on every turn. Claude adds them
once per Claude conversation, because hook `additionalContext` stays in Claude's
history. It adds them again after a compaction, for a new conversation, or when
the catalog changes. A turn that ends before its `Stop` hook fires gets them
again on the next turn.

If your prompt already contains the instruction (the server copy or the bundled
text), the adapter doesn't add it a second time. Claude looks in
`system_prompt` when it is a `str`, a preset's `append`, or a custom `prompt` (not
a file prompt). `apply_to()` passes `system_prompt` to the hook for you; with
manual `hooks()` wiring, pass `system_prompt=` yourself. LangGraph can't see the
prompt inside your graph, so if you upgrade from an older setup, remove any
`CONTEXT_PACK_INSTRUCTION` you added to it yourself.

So wiring packs is just a matter of offering the tool. Use `default_tools()`
**or** your own `make_get_context_pack_tool(...)`, not both — both register a
tool named `get_context_pack`, and duplicate tool names are rejected or
ambiguous.

**Google ADK** — `default_tools()` already includes the tool. To set a fallback
id list, build the list yourself instead:

```python
agent = LlmAgent(
    name="my_agent",
    model="gemini-3-flash-preview",
    instruction=MY_PROMPT,
    tools=[
        adapter.make_context_retrieval_tool(),
        adapter.make_get_context_pack_tool(["PlanRules", "Claims"]),
    ],
    **adapter.callback_kwargs(),
)
```

**Claude Agent SDK** — `apply_to` (above) already registers and allows
`get_context_pack`. If you wire it yourself, register `default_tools()` on the
MCP server, allow both tools, and pass `hooks(...)` (with your `system_prompt`)
so the instruction and catalog are injected:

```python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server  # your existing Claude Agent SDK code

server = create_sdk_mcp_server(
    name="context", version="1.0.0",
    tools=[*adapter.default_tools(session_id, owner_id)],  # recall + get_context_pack
)
options = ClaudeAgentOptions(
    system_prompt=MY_PROMPT,
    mcp_servers={"context": server},
    allowed_tools=[
        "mcp__context__recall_customer_context",
        "mcp__context__get_context_pack",
    ],
    hooks=adapter.hooks(session_id, owner_id, system_prompt=MY_PROMPT),
)
```

**LangGraph** — nothing extra to wire. The `sdk_tools` the adapter passes to
your `graph=` builder already include `get_context_pack` (see
[LangGraph](#langgraph)), so don't add a second `make_get_context_pack_tool(...)`
to the list you bind. This is self-driving mode only. In manager-injection
mode, `default_tools()` returns `[]`, and a pack tool you build yourself
returns `not_ready` for every valid pack id, because the adapter has no
retrieval client of its own to call.
Passing `retrieval_client=` there only switches the injected instruction to the
server copy; it doesn't enable pack reads.

```python
def build_graph(sdk_tools):
    tools = [*sdk_tools, *my_tools]   # sdk_tools = recall + get_context_pack
    model_with_tools = model.bind_tools(tools)
    ...
```

**OpenAI Agents SDK** — `default_tools()` already includes the tool. Pass
`valid_pack_ids=` to set the fallback id list.
`instructions_provider` builds `base`, then the instruction, then the catalog,
on every turn:

```python
agent = Agent(
    name="my_agent",
    model="gpt-4.1",
    instructions=adapter.instructions_provider(
        base=MY_PROMPT,
        session_id=session_id,
    ),
    tools=[
        *adapter.default_tools(session_id, valid_pack_ids=["PlanRules", "Claims"]),
    ],
)
```

**Microsoft Foundry** — nothing extra. `run` / `stream` register the pack
tools and inject the instruction and catalog each turn:

```python
agent = Agent(FoundryChatClient(), instructions=MY_PROMPT)
```

Building your own prompt with injection turned off? Call
`await adapter.get_context_pack_instruction(session_id)` to get the text pinned
for that session. It never raises.

Two optional knobs on `make_get_context_pack_tool(...)` (all five adapters;
`default_tools()` doesn't take them, so build the tool yourself to use them):
`param_defaults={...}` merges fixed pack `params` into every call (a key the
model also sends overrides the default; if the model's value is blank the key
is dropped, default included), and `timeout=` overrides the per-call HTTP budget — a pack POST
fans out across slow sources, so it defaults to `CONTEXT_PACK_TIMEOUT_S`
(60s), independent of the fast-read client timeout.

> **Preview.** The pack read is `POST /api/v1/agentic/context`, which may not be
> deployed in every environment yet — hence the `not_ready` status and the
> "degrade, never fail" guidance baked into `CONTEXT_PACK_INSTRUCTION`.

### Context-pack versioning (preview)

A pack's shape (description, data needs, parameters) can change at any time
when the catalog owner republishes it. If the agent plans against one revision
and Beacon has moved on since, the fetched pack may no longer carry the data
needs the plan assumed. **Versioning** gives the agent a way to detect that
and recover. Every adapter — Claude, Google ADK, LangGraph, Foundry, and
OpenAI — wires this through the pack tool schema as a `context_pack_version`
argument, and the shared `CONTEXT_PACK_INSTRUCTION` teaches the LLM when to
pass it (**required** whenever the catalog entry carries a `version`;
omitted only when the entry has no `version` field).

The flow:

1. The pack catalog block (`[Available context packs]`) carries a `version`
   on every pack entry the server knows a revision for.
2. `CONTEXT_PACK_INSTRUCTION` tells the LLM to echo that `version` back to
   `get_context_pack` as `context_pack_version` so it only gets back the same
   revision it planned against.
3. If Beacon has republished the pack, it returns a
   `{"status": "version_expired", "latest_version": ..., "latest": "..."}`
   envelope — `latest` is a TOON description of the new pack (id,
   description, version, dataNeeds). The LLM compares its plan to the new
   dataNeeds and either retries with the new version or tells the user which
   piece is no longer available. Guidance caps the retry at once per turn.

Whenever the catalog entry for a pack carries a `version`, the LLM instruction
makes `context_pack_version` mandatory — it MUST be echoed back on every call so
the agent gets the exact revision it planned against. Only omit it when the
catalog entry has no `version` field. The SDK defends the MUST at the tool
boundary: if the model forgets to pass `context_pack_version`, the adapter
auto-fills it from the cached catalog so a pinned call still goes out. Beacon
also accepts a bare call without `context_pack_version` and serves the live
revision — that path is reserved for pre-versioned deployments where the server
has not yet started advertising `version` in the catalog.

## Shared defaults

The examples above pass only the essentials. The adapters share a set of
optional knobs, so you can stay on the defaults until you need to tune them.
These are the same across Claude, Google ADK, LangGraph, Foundry, and OpenAI
unless noted:

| Parameter | Default | What it controls |
| --- | --- | --- |
| `context_mode` | `"auto"` | `"auto"` seeds fast context once, then lets the agent call recall on demand. `"manual"` turns off automatic injection; the recall tool stays registered. Claude, ADK, LangGraph, and OpenAI accept both; LangGraph also accepts `"per-turn"`. Foundry has no mode knob — it always seeds once and always registers the tools. |
| `dataspace` | `CDP_DATASPACE`, then `"default"` | The CDP dataspace records are written to and read from. Foundry doesn't read `CDP_DATASPACE`; it defaults straight to `"default"`. |
| `owner_type` | from `auth` | Identity kind for `owner_id`: `"employee"` with `UserContextAuth`, `"individual"` with `ContextAuth`, `ApiKeyAuth`, or a plain `api_key`. Exposed on all five adapters. |

Claude and OpenAI also take `use_async` (default `True`). None of these are
required to get started; pass them only to override the default shown here.


## Without an adapter — `AgentContextClient` (preview)

If your runtime has no adapter, or you need to own the lifecycle yourself,
`AgentContextClient` exposes the same ingestion lifecycle and mid-turn tools the
adapters use internally. Same `auth` as above. See
[AGENT_CONTEXT_CLIENT.md](AGENT_CONTEXT_CLIENT.md).


## Next steps

- Prerequisites, Connected App setup, env vars → [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- Full constructor tables, sync mode, snapshot handoff, HTTP observability, go-live checklists, troubleshooting → [ADVANCED.md](ADVANCED.md)
