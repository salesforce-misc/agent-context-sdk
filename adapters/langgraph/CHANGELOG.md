# Changelog

Historical changes to the LangGraph adapter (`agentic_context_langgraph`).
Since 0.7.0 the adapter ships inside the `agent-context-sdk` distribution and
new changes are recorded in the root `CHANGELOG.md`. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] — shipped in agent-context-sdk 0.7.0

### Added

- `graph=` accepts a builder `(sdk_tools) -> compiled graph`. The adapter
  calls it once at construction with `default_tools()` so the SDK tools are
  bound before the graph compiles.

### Changed

- `user_participant_id` now defaults to the turn's resolved `owner_id`
  instead of `"user"`, because Beacon marks an entry `role="owner"` only when
  `participantId == ownerId`. It falls back to `"user"` only when no owner
  resolves. Pass the kwarg explicitly to keep the old label.

### Removed

- Deep-mode plumbing (`make_deep_context_node` factory, `get_deep_context`
  adapter method, `deep_context_latency_sensitivity` /
  `deep_context_token_budget` constructor kwargs, deep sufficiency instruction).
  The merged `GetContextRequest` schema is
  `additionalProperties: false` and defines only `conversation_id`,
  `owner_id`, `agent_id`, `owner_type` — the deep kwargs could never reach
  the wire. The `recall_customer_context` tool now takes an optional
  `topic` hint only.

## [0.3.0] - 2026-07-22

### Added

- `make_deep_context_node(client, *, agent_id, query_key="recall_query", ...)`
  — a node factory that is the **deep sibling** of `make_context_node`. Where
  `make_context_node` runs a fast, pre-indexed profile read with no query, this
  node reads a caller-generated `query` (the *intent*) from graph **state** (or
  the `RunnableConfig` via `query_from_config=True`) and issues
  `get_context(mode="deep", query=...)` — Beacon's LLM-driven hydration scoped
  to that intent, tuned by `latency_sensitivity` (default `"medium"`) and
  `token_budget` (default `4096`) factory kwargs. It is the primitive for a
  **structurally-enforced** recall pipeline: a router node decides *when* to
  recall, a `gen_query` node writes the intent into `state[query_key]`, plain
  code validates it, then this node runs the deep read with that query — so
  "generate the query before the read" is a code invariant (graph edges), not a
  matter of prompt compliance or a fused tool call. An empty/missing query is
  not special-cased client-side (sent as `mode="deep"` with an empty query;
  Beacon falls back to fast context); only a missing `session_id`/`owner_id`
  short-circuits with no HTTP call. Same best-effort semantics as
  `make_context_node` (any error logged and swallowed → `{}`), and a distinct
  default `SystemMessage` id (`agentic-memory-deep-context`) so a fast seed and
  a deep read coexist in one thread without clobbering. Exported from
  `agentic_context_langgraph` and added to `__all__`. Purely additive.

- On-demand `get_context` tool surface on `LangGraphAdapter`, mirroring the
  Google ADK adapter's `recall_customer_context` tool so all adapters behave
  identically:
  - `make_context_retrieval_tool()` — a LangChain `StructuredTool` wrapping an async
    `recall_customer_context(topic="", query="")` that the LLM calls when it
    decides it needs cross-session memory. The tool always issues a deep,
    agentic, intent-scoped read (`mode="deep"`): the optional `query` is the
    agent's *intent*, not a fast/deep switch. An empty intent is not
    special-cased client-side — it is sent through as `mode="deep"` with an
    empty query, and Beacon's `get_context` API falls back to fast profile
    context server-side, so there is a single client code path and the tool
    stays safe to call reflexively. The deep pass is tuned by the new
    `deep_context_latency_sensitivity` (default `"medium"`) and
    `deep_context_token_budget` (default `4096`) constructor knobs. Per-turn
    injection stays fast-only. The `session_id` is resolved from the
    per-invocation `RunnableConfig` (`configurable.thread_id`), so the tool is
    race-safe under concurrent multi-session streams. It is best-effort:
    returns `{"userContext": []}` (never raises) when there is no owned
    retrieval client, the session can't be resolved, or the read fails, and
    emits a single `mode=deep` + `has_intent` + `has_context`-boolean audit log
    (never the retrieved payload or the deep `query` text).
  - `get_deep_context(query, session_id=None)` — the deep read as a public
    method, for direct/manual use outside the tool.
  - `default_tools()` — returns `[make_context_retrieval_tool()]` to splat into your
    agent's tool list / `ToolNode`.

  The tool is **purely additive**: the adapter's existing policy-driven
  per-turn context injection (self-driving mode, default `SessionFirstPolicy`)
  is unchanged and remains the default. There is no behaviour change for
  existing integrations.

### Changed

- `context_mode=` (`"auto"` / `"per-turn"` / `"manual"`, default `"auto"`) is
  now the single knob for context management — there are no granular
  `context_policy=` / `enable_deep_context=` / `deep_context_instruction=`
  constructor knobs. A **bare** `LangGraphAdapter(graph=…)` (no `context_mode`,
  no injected `session_manager`) resolves to `"auto"`: it installs the
  `recall_customer_context` tool + the built-in instruction and seeds **fast**
  context once on turn 1 (`SessionFirstPolicy`). This is additive versus the
  prior bare default (it adds the tool and instruction; the fast seed still
  fires once) and cost-safe (a deep read happens only when the agent chooses).
  Inspect the resolved mode via `adapter.context_mode`. This is the behaviour
  change the `0.2.0 → 0.3.0` minor bump reflects. For a fetch cadence the three
  modes don't express, resolve a `ContextPolicy` with `resolve_policy(...)` and
  wire it into your own `make_context_node`.

## [0.2.0] - 2026-06-29

### Deprecated

- Renamed `make_context_node` keyword arguments to align with the SDK's
  "session_id is the wire identifier" terminology:
  - `conversation_id_from_config` → `session_id_from_config`
  - `conversation_id_key` → `session_id_key`

  The old names still work — they emit a `DeprecationWarning` and forward
  to the new names. Passing both the old and new name for the same knob
  raises `TypeError`. The shim will be removed in 0.4.0.

  The runtime behaviour is unchanged; only the parameter names moved.
  `session_id_key` continues to default to `"thread_id"` because that
  is the field LangGraph itself uses inside `config["configurable"]`.
