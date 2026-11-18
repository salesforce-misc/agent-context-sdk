# AgentContextClient — adapter-free integration (preview)

Use `AgentContextClient` when you can't use one of the framework adapters (custom
runtime, unsupported framework, or you need explicit control over when
ingestion happens). It runs on the same stack as the adapters; your host
calls it directly instead of an adapter doing it for you.

> **Preview.** `AgentContextClient` is exported from `agentic_context` but is not yet
> part of the semver-stable surface. Method names and return shapes may change.

## When to use it

| Use an adapter | Use `AgentContextClient` |
|---|---|
| Your agent runs on LangGraph, Claude Agent SDK, Google ADK, OpenAI Agents SDK, or Microsoft Foundry | Any other runtime, or you want to decide exactly when each ingest and retrieval call happens |
| Lifecycle and tools are wired for you | You wire two things: lifecycle calls in your host, and two tools for the model |

## The model: host owns lifecycle, model owns reads

| Layer | Method | Called by | When |
|---|---|---|---|
| Lifecycle | `start_conversation` | Host | Once, before the first model call |
| Lifecycle | `ingest_customer_message` | Host | Each user turn, before the model runs |
| Lifecycle | `ingest_agent_message` | Host | Each turn, after the model's final reply |
| Mid-turn tool | `ace_get_context` | Model | When it needs customer memory |
| Mid-turn tool | `ace_get_context_pack` | Model | When it needs structured pack data |

Lifecycle stays in the host on purpose: if the model were responsible for it,
a skipped tool call would mean an unopened conversation or a turn that never
gets ingested.

## Quick start

```python
import json
from agentic_context import AgentContextClient

ace = AgentContextClient()

# 1. Open — authenticates, retrieves seed context + pack catalog, creates the conversation.
opened = ace.start_conversation(
    title="support chat",
    auth=auth,                       # UserContextAuth or ContextAuth (see Credentials in QUICKSTART)
    agent_id="your_agent_id",        # registered agent id
    owner_id=end_user_id,            # required with ContextAuth; defaults to auth.user_id with UserContextAuth
)
conversation_id = opened["conversation_id"]

system_prompt = BASE_PROMPT + "\n\n" + opened["system_prompt_suffix"]
tools = ace.tool_specs()             # JSON Schema for the two mid-turn tools

# 2. Each turn
def on_tool_call(name: str, args: dict) -> str:
    try:
        return json.dumps(ace.call_tool(name, args), default=str)
    except (ValueError, RuntimeError) as e:     # bad tool call — tell the model
        return json.dumps({"status": "error", "message": str(e)})

if user_text.strip():
    ace.ingest_customer_message(user_text)

reply = run_model(system_prompt, history, tools, on_tool_call=on_tool_call)

if reply.strip():
    ace.ingest_agent_message(reply)

# 3. Shutdown — local cleanup only; nothing is closed on the ingestion service.
ace.reset()
```

`run_model` stands in for your runtime's model loop. Register `tools` with
the model; when it emits a tool call, pass the name and arguments to
`ace.call_tool` and send the serialized result back.

## API reference

All methods are synchronous. Import from the top-level package:

```python
from agentic_context import AgentContextClient, IngestionError
```

| Group | Method | Called by | Purpose |
|---|---|---|---|
| [Lifecycle](#1-lifecycle) | 1.1 [`start_conversation()`](#11-start_conversation--open-a-conversation) | Host | Authenticate, retrieve seed context, open the conversation |
| | 1.2 [`ingest_customer_message()`](#12-ingest_customer_message--ingest-the-users-message) | Host | Ingest the user's message for this turn |
| | 1.3 [`ingest_agent_message()`](#13-ingest_agent_message--ingest-the-agents-reply) | Host | Ingest the agent's reply for this turn |
| [Mid-turn tools](#2-mid-turn-tools) | 2.1 [`tool_specs()`](#21-tool_specs--get-tool-schemas) | Host | Tool schemas to register with the model |
| | 2.2 [`call_tool()`](#22-call_tool--run-a-model-tool-call) | Host, on the model's behalf | Run a tool call the model emitted |
| [Conversation history](#3-conversation-history) | 3.1 [`list_conversations()`](#31-list_conversations--list-past-conversations) | Host | List this owner's past conversations |
| | 3.2 [`get_conversation()`](#32-get_conversation--fetch-one-conversation) | Host | Fetch one conversation |
| [Cleanup](#4-cleanup) | 4.1 [`reset()`](#41-reset--release-connections-and-local-state) | Host | Release connections and clear local state |

Methods that take an optional `conversation_id` use the **active
conversation** when you omit it. `start_conversation` sets the active
conversation.

---

### 1. Lifecycle

#### 1.1 `start_conversation()` — Open a conversation

```python
start_conversation(
    conversation_id: str | None = None,
    *,
    title: str = "",
    prefix: str = "ses",
    auth: ContextAuth | UserContextAuth | None = None,
    agent_id: str | None = None,
    owner_id: str | None = None,
    owner_type: str | None = None,
    dataspace: str | None = None,
) -> dict
```

The signature is abridged. The method also accepts `consumer_key`,
`login_url`, `refresh_token`, and `on_refresh_token`, which build a
`UserContextAuth` for you. Prefer passing a ready `auth`.

Call once per conversation, before the first model call. On the first call it
authenticates. Then it retrieves the seed customer context and the
context-pack catalog, and creates the conversation, session, and agent
participant on the ingestion service.

**Parameters**

| Name | Required | Description |
|---|---|---|
| `conversation_id` | No | Your id for the conversation. Pass an existing id to resume it. If omitted, one is generated as `{prefix}-<12 hex>`. |
| `title` | No | Human-readable conversation name. |
| `auth` | Yes, on the first call | A `UserContextAuth` or `ContextAuth` instance (see *Credentials* in [QUICKSTART.md](QUICKSTART.md)). On later calls, omit it. A later `auth` must be verifiably the same identity (see *Identity per instance*), then is ignored: the first auth keeps serving and refreshing tokens. To switch credentials or identity, call `reset()` first. |
| `agent_id` | On the first call, unless the auth carries it | Your registered agent id. Defaults to the agent id the auth was built with. |
| `owner_id` | With `ContextAuth` | The end user the conversation belongs to. Defaults to `auth.user_id` with `UserContextAuth`. |
| `owner_type` | No | Identity kind of `owner_id`. Defaults to the auth's default (`"employee"` for `UserContextAuth`, `"individual"` for `ContextAuth`). If you use `UserContextAuth` but pass a customer's id as `owner_id`, set `owner_type="individual"`. Otherwise retrieval looks up the wrong identity kind and returns empty context. |
| `dataspace` | No | Data Cloud dataspace. Default: `"default"`. |

**Returns** `dict`:

| Key | Type | Description |
|---|---|---|
| `conversation_id` | `str` | The conversation id (yours or the generated one). |
| `system_prompt_suffix` | `str` | **Append this to your system prompt.** It holds the context-pack catalog and instructions for using the tools. |
| `context` | `dict` | Seed customer context (profile, prior-conversation summaries). Optionally add it to the prompt for first-turn personalization. |
| `pack_ids` | `list[str]` | Context-pack ids the model may request. Empty if the agent has no packs, or if the catalog couldn't be fetched (see below). |
| `already_open` | `bool` | `True` if this conversation was already open on this instance. The cached seed was returned and no network calls were made. |

**Raises**

The exceptions depend on which step fails.

| Step | Exception | When |
|---|---|---|
| Any | `ValueError` | `auth`, `agent_id`, or `owner_id` is missing or can't be resolved. Also raised when a later `auth` can't be verified as the bound identity (see *Identity per instance*), or `agent_id`, `owner_id`, `owner_type`, or `dataspace` differs from the identity this instance is already bound to (call `reset()` first). |
| Authentication | `ReauthRequired` | `UserContextAuth` only: the refresh token was rejected. The user must sign in again. |
| Authentication | `httpx.HTTPStatusError`, `httpx.TransportError` | The token exchange failed, e.g. a bad consumer key or a 401. Transient failures are retried first. |
| Authentication | `ValueError` | The token response was malformed or missing a field. |
| Seed context | `IngestionError` | The context read returned an error status, after retries on retryable statuses. |
| Seed context | `requests.RequestException` | A connection error or timeout. Not retried. |
| Conversation creation | `IngestionError` | An error status. An existing conversation (409) counts as success. Not retried. |
| Conversation creation | `requests.RequestException` | A connection error or timeout. Not retried. |

If the context-pack catalog can't be fetched, the call doesn't fail. The
two empty-`pack_ids` cases then behave differently:

| Case | `ace_get_context_pack` |
|---|---|
| The catalog loaded, but the agent has no packs | Every id returns `invalid_pack`. |
| The catalog couldn't be fetched | Pack-id validation is off: any id the model asks for is sent to the service. |

```python
opened = ace.start_conversation(
    title="support chat", auth=auth, agent_id="your_agent_id", owner_id=end_user_id,
)
system_prompt = BASE_PROMPT + "\n\n" + opened["system_prompt_suffix"]
```

#### 1.2 `ingest_customer_message()` — Ingest the user's message

```python
ingest_customer_message(
    content: str,
    conversation_id: str | None = None,
    *,
    event_id: str | None = None,
    interaction_id: str | None = None,
    metadata: dict | None = None,
) -> dict
```

Call once per user turn, **before** the model runs. Each call starts a new
turn.

#### 1.3 `ingest_agent_message()` — Ingest the agent's reply

```python
ingest_agent_message(
    content: str,
    conversation_id: str | None = None,
    *,
    event_id: str | None = None,
    interaction_id: str | None = None,
    metadata: dict | None = None,
) -> dict
```

Call **after** the model produces its final reply. It joins the turn opened
by the last `ingest_customer_message`. If the agent sends several messages
in one turn, call this once per message.

**Parameters** (1.2 and 1.3)

| Name | Required | Description |
|---|---|---|
| `content` | Yes | Message text. |
| `conversation_id` | No | Defaults to the active conversation. |
| `event_id` | No | Local id for this entry, generated if omitted. It is not sent: the ingestion service assigns its own entry id. Nothing is deduplicated, so passing the same `event_id` again writes a second entry. |
| `interaction_id` | No | Turn id. Generated and tracked for you; only set it to supply your own turn ids. |
| `metadata` | No | Accepted for compatibility. It is not sent to the ingestion service and has no effect. |

**Returns** `dict`:

| Key | Type | Description |
|---|---|---|
| `event_id` | `str` | The local id of the entry (see `event_id` above). |
| `interaction_id` | `str` | The turn id the entry belongs to. |
| `conversation_id` | `str` | The conversation it was ingested into. |

**Raises**

| Exception | When |
|---|---|
| `ValueError` | `conversation_id` was omitted and there is no active conversation. |
| `RuntimeError` | The conversation isn't open on this instance. Call `start_conversation` first. |
| `IngestionError` | The service returned an error status after retries, or right away on a non-retryable status (400/401/403/404). |
| `requests.RequestException` | A connection error or timeout persisted after retries. |
| `ReauthRequired`, `httpx.HTTPError` | On a 401 the call refreshes the token once. These are raised if that refresh fails (`ReauthRequired`: `UserContextAuth` refresh token rejected). |

If an ingest raises after a timeout, the entry may or may not have been
written. Calling it again can create a duplicate entry. Decide whether to
retry, or to log the error and accept a possible missing entry.

```python
ace.ingest_customer_message(user_text)
reply = run_model(...)
ace.ingest_agent_message(reply)
```

---

### 2. Mid-turn tools

The model calls these two tools while it works on a reply. Register them
with your model and route the model's tool calls through `call_tool`.

| Tool | Arguments | Returns |
|---|---|---|
| `ace_get_context` | `conversation_id?` | Latest customer memory for the conversation. |
| `ace_get_context_pack` | `context_pack_id` (required), `params?`, `conversation_id?` | Structured data from one context pack. Valid ids come from the catalog in `system_prompt_suffix`; `params` takes only the pack's declared parameters. |

#### 2.1 `tool_specs()` — Get tool schemas

```python
tool_specs() -> list[dict]
```

Returns one dict per tool: `{"name", "description", "parameters"}`.
`parameters` is a JSON Schema object. Convert these to your framework's
tool format when you register them. The list is a fresh copy on every call,
so it's safe to modify.

#### 2.2 `call_tool()` — Run a model tool call

```python
call_tool(name: str, arguments: dict | None = None) -> dict
```

Runs one tool call. Pass the tool name and arguments exactly as the model
emitted them. `call_tool` cleans up the arguments before running the tool:

- camelCase keys are accepted (`contextPackId` → `context_pack_id`)
- unknown keys are dropped
- `owner_id` / `agent_id` are dropped, because identity is fixed at `start_conversation`
- a missing `conversation_id` is filled from the active conversation

**Returns** a status envelope. Serialize it (e.g. `json.dumps`) and return
it to the model as the tool result.

| `status` | Other keys | What the model should do |
|---|---|---|
| `"ok"` | `context` (the data), `conversation_id`; packs also include `context_pack_id`, `params` | Use the data. |
| `"not_ready"` | `message`, `detail` | Retrieval failed. Continue with the context already in hand; don't retry repeatedly. |
| `"invalid_pack"` | `valid_pack_ids`, `message` | Pick an id from `valid_pack_ids`. |

**Raises**

| Exception | When |
|---|---|
| `ValueError` | `name` isn't one of the two tools, a required argument is missing, or there is no `conversation_id` and no active conversation. |
| `RuntimeError` | `start_conversation` hasn't been called on this instance, or (`ace_get_context_pack` only) the requested conversation isn't open. |

Return both to the model as a tool error rather than letting them end the
turn. Network and service errors never raise from `call_tool`. They come
back as `not_ready`.

```python
def on_tool_call(name: str, args: dict) -> str:
    try:
        return json.dumps(ace.call_tool(name, args), default=str)
    except (ValueError, RuntimeError) as e:
        return json.dumps({"status": "error", "message": str(e)})
```

The model may pass any `conversation_id`, not only the active one. Reads are
always scoped to the bound owner and agent, so it can reach only that
owner's conversations with this agent. If the model should only read the
active conversation, drop `conversation_id` from the arguments before
calling `call_tool`.

The host can also call the tool methods directly: `ace.get_context()` and
`ace.get_context_pack(context_pack_id=..., params=...)` return the same
envelopes.

---

### 3. Conversation history

Optional reads for hosts that show past conversations, e.g. a "recent chats"
list. Both use the identity bound by `start_conversation`. Before that,
both raise `RuntimeError`; after it, errors are logged and swallowed as
described below.

#### 3.1 `list_conversations()` — List past conversations

```python
list_conversations() -> list[dict]
```

Lists the bound owner's conversations with this agent. Returns `[]` on any
error.

#### 3.2 `get_conversation()` — Fetch one conversation

```python
get_conversation(conversation_id: str | None = None) -> dict | None
```

Fetches one conversation, including its entries. Defaults to the active
conversation. Returns `None` on any retrieval error. Raises `ValueError`
if `conversation_id` is omitted and there is no active conversation.

---

### 4. Cleanup

#### 4.1 `reset()` — Release connections and local state

```python
reset() -> dict
```

Releases HTTP connections and clears all local state: open conversations,
the bound identity, and cached seeds. Nothing is closed or deleted on the
ingestion service, and conversations can be resumed later by
`conversation_id`. Call it at shutdown, or before reusing the instance for a
different end user or agent.

## Things to know

- **Blocking calls.** All methods are synchronous. In an async host, call
  them with `await asyncio.to_thread(ace.ingest_customer_message, text)` so
  your event loop isn't blocked.
- **Retries.** Retries differ by call:

  | Call | Retries |
  |---|---|
  | `ingest_customer_message` / `ingest_agent_message` | Up to 4 attempts with exponential backoff, on retryable statuses (408, 429, 500, 502, 503, 504) and on connection errors or timeouts. |
  | Mid-turn tools, seed context in `start_conversation` | Up to 4 attempts on retryable statuses only. Connection errors are not retried. |
  | Conversation creation in `start_conversation` | Not retried. |
- **No dedup.** `event_id` is not sent and the ingestion service assigns
  its own entry ids, so nothing is deduplicated. Every successful ingest
  call writes a new entry. Don't re-send a turn that already went through.
- **Errors.** Lifecycle methods raise; mid-turn tools return `not_ready`
  instead (see *Raises* under each method). Decide whether a failed ingest
  should fail the turn or just be logged.
- **One identity per instance.** An instance is bound to one
  `auth` + `agent_id` + `owner_id`. Passing a different value to a later
  `start_conversation` raises `ValueError`. A later `auth` must be provably
  the same identity without fetching a token: the same auth class, agent id,
  login host, consumer key, `subject` and `xid`, and for a `UserContextAuth`
  a resolved `user_id` equal to the bound one. Anything it can't verify
  raises, including a fresh `UserContextAuth` that hasn't fetched a token
  yet. Pass the bound auth object, or omit `auth`. A different `owner_id`
  also raises. With `ContextAuth`, the
  end user is only identified by `owner_id`. If you omit it on a later call,
  the bound owner is used. Use one instance per end user, or `reset()`
  between users.
- **Not for concurrent lifecycle calls.** The instance tracks the active
  conversation and the open turn without locking. Don't run lifecycle
  calls for one instance from several threads at once. This includes
  `to_thread` calls from tasks that run concurrently. If you keep several
  conversations on one instance, pass `conversation_id` explicitly on
  every call and run them one at a time. Frameworks can run the tools from
  one model reply in parallel (LangGraph's `ToolNode` does), so the examples
  below serialize `call_tool` with a lock.
- **Conversation = session.** `conversation_id` is used as both the
  conversation id and the session id. Reusing a `conversation_id` resumes it.
- **Only messages are ingested.** Customer and agent messages are ingested;
  tool calls and intermediate model steps are not.

## Wiring patterns

There are two ways to connect the client to a host. Both use the same
`AgentContextClient` calls; only the glue differs. The snippets below are
based on the reference demos, simplified, with each blocking call wrapped in
`asyncio.to_thread` (see *Things to know*).

### 1. Hooks + MCP

Lifecycle runs in host hooks (e.g. "user prompt submitted" / "agent
stopped"). The two tools are served from an MCP server whose handlers call
`ace.call_tool(...)`.

**Example:** [`examples/client_hooks_demo/`](../examples/client_hooks_demo/),
with Claude Agent SDK as the host.

| Host event | AgentContextClient call |
|---|---|
| `UserPromptSubmit` hook, first turn | `start_conversation(...)`. Inject `system_prompt_suffix` as `additionalContext`. |
| `UserPromptSubmit` hook, every turn | `ingest_customer_message(prompt)` |
| `Stop` hook | `ingest_agent_message(last_assistant_message)` |
| MCP tools `ace_get_context` / `ace_get_context_pack` | `call_tool(name, args)` |

```python
import asyncio, json
from claude_agent_sdk import (
    ClaudeAgentOptions, HookMatcher, create_sdk_mcp_server, query, tool,
)
from agentic_context import AgentContextClient

ace = AgentContextClient()
state = {"opened": None}
tool_lock = asyncio.Lock()   # tools from one model reply may run concurrently

# Lifecycle in hooks
async def on_user_prompt_submit(input_data, tool_use_id, context):
    if state["opened"] is None:
        state["opened"] = await asyncio.to_thread(
            ace.start_conversation,
            conversation_id=session_id, title="support chat",
            auth=auth, agent_id=agent_id, owner_id=end_user_id,
        )
    prompt = (input_data.get("prompt") or "").strip()
    if prompt:
        await asyncio.to_thread(ace.ingest_customer_message, prompt)
    return {"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": state["opened"]["system_prompt_suffix"],
    }}

async def on_stop(input_data, tool_use_id, context):
    reply = (input_data.get("last_assistant_message") or "").strip()
    if reply and state["opened"]:
        await asyncio.to_thread(ace.ingest_agent_message, reply)
    return {}

# Mid-turn tools on MCP: one tool per spec, handler calls call_tool
def make_mcp_tool(spec: dict):
    name = spec["name"]

    @tool(name, spec["description"], spec["parameters"])   # JSON Schema passes through
    async def handler(args: dict) -> dict:
        try:
            async with tool_lock:
                out = await asyncio.to_thread(ace.call_tool, name, args)
        except (ValueError, RuntimeError) as e:   # bad tool call: tell the model
            out = {"status": "error", "message": str(e)}
        return {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}

    return handler

server = create_sdk_mcp_server(
    name="context", version="1.0.0",
    tools=[make_mcp_tool(s) for s in ace.tool_specs()],
)

options = ClaudeAgentOptions(
    system_prompt=BASE_PROMPT,
    mcp_servers={"context": server},
    allowed_tools=["mcp__context__ace_get_context", "mcp__context__ace_get_context_pack"],
    hooks={
        "UserPromptSubmit": [HookMatcher(matcher=".*", hooks=[on_user_prompt_submit])],
        "Stop": [HookMatcher(matcher=".*", hooks=[on_stop])],
    },
)

async for message in query(prompt=user_text, options=options):
    ...
```

Because the hook injects `system_prompt_suffix` through `additionalContext`
on every turn, the pack catalog stays available to the model even when each
turn starts a fresh `query()`.

A fresh `query()` per turn has no history of earlier turns. The model only
learns about them through `ace_get_context`. To keep the history in the
model's context, run all turns on one `ClaudeSDKClient`. In that case inject
`system_prompt_suffix` only on the first turn, because every
`additionalContext` stays in the transcript and would otherwise repeat.

### 2. Direct calls

Host code (graph nodes, middleware) calls the lifecycle methods. The
mid-turn tools are ordinary framework tools whose body is
`ace.call_tool(...)`.

**Example:** [`examples/client_direct_demo/`](../examples/client_direct_demo/),
with LangGraph as the host. The demo runs a single thread, so it opens the
conversation once behind a flag. The snippet below tracks open
conversations per `thread_id` instead.

| Graph node | AgentContextClient call |
|---|---|
| `pre_turn`, every turn | `start_conversation(conversation_id=thread_id, ...)`. Opens on the first turn; later turns return the cached result and make the thread active. Then `ingest_customer_message(user_text)`. |
| `agent` | Prepends `BASE_PROMPT` + `system_prompt_suffix` as a `SystemMessage` on each model call. The message is not stored in graph state. |
| `tools` (`ToolNode`) | `call_tool(name, args)` inside each tool |
| `post_turn` | `ingest_agent_message(reply)` |

Graph shape: `START → pre_turn → agent ⇄ tools → post_turn → END`

```python
import asyncio
import threading
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from agentic_context import AgentContextClient

ace = AgentContextClient()          # one instance per end user
opened: dict[str, dict] = {}        # thread_id -> start_conversation result
tool_lock = threading.Lock()        # ToolNode runs one reply's tool calls in parallel threads

# Mid-turn tools: one StructuredTool per spec, body calls call_tool
def make_tool(spec: dict) -> StructuredTool:
    name = spec["name"]

    def run(**kwargs) -> dict:
        try:
            with tool_lock:
                return ace.call_tool(name, kwargs)
        except (ValueError, RuntimeError) as e:   # bad tool call: tell the model, don't fail the run
            return {"status": "error", "message": str(e)}

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=spec["description"],
        args_schema=spec["parameters"],   # JSON Schema dict
    )

tools = [make_tool(s) for s in ace.tool_specs()]
model_with_tools = model.bind_tools(tools)  # any LangChain chat model

# Lifecycle in host nodes
async def pre_turn(graph_state, config):
    thread_id = config["configurable"]["thread_id"]
    # Idempotent: the first call opens the conversation; later calls return the
    # cached seed without network calls and make this thread the active one.
    opened[thread_id] = await asyncio.to_thread(
        ace.start_conversation,
        conversation_id=thread_id, title="support chat",
        auth=auth, agent_id=agent_id, owner_id=end_user_id,
    )
    text = last_user_text(graph_state)
    if text:
        await asyncio.to_thread(ace.ingest_customer_message, text, thread_id)
    return {}

async def agent(graph_state, config):
    suffix = opened[config["configurable"]["thread_id"]]["system_prompt_suffix"]
    # Prepend per call instead of storing it in checkpointed state, so it is
    # always first and never duplicated after a restart.
    system = SystemMessage(content=BASE_PROMPT + "\n\n" + suffix)
    reply = await model_with_tools.ainvoke([system, *graph_state["messages"]])
    return {"messages": [reply]}

async def post_turn(graph_state, config):
    text = last_ai_text(graph_state)
    if text:
        await asyncio.to_thread(
            ace.ingest_agent_message, text, config["configurable"]["thread_id"],
        )
    return {}

g = StateGraph(ChatState)        # ChatState: messages with add_messages reducer
g.add_node("pre_turn", pre_turn)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(tools))
g.add_node("post_turn", post_turn)
g.add_edge(START, "pre_turn")
g.add_edge("pre_turn", "agent")
g.add_conditional_edges("agent", lambda s: "tools" if tools_condition(s) == "tools" else "post_turn")
g.add_edge("tools", "agent")
g.add_edge("post_turn", END)
graph = g.compile(checkpointer=checkpointer)

# Each graph run is one turn for one thread:
# await graph.ainvoke({"messages": [HumanMessage(user_text)]},
#                     {"configurable": {"thread_id": session_id}})
```

Because `start_conversation` runs again after a restart, the cached
`opened` entry is rebuilt from the service and the seed is retrieved again.
Run one turn at a time per instance (see *Things to know*).

`last_user_text` / `last_ai_text` return the latest `HumanMessage` text and
the latest `AIMessage` text that has no tool calls. See the demo for both.

### Running the demos

Both demos read settings from `examples/ace_agent.env` and print every
request and response on the wire. Each demo folder's README covers
prerequisites and run commands.
