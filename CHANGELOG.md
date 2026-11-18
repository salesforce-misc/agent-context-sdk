# Changelog

All notable changes to `agent-context-sdk` (formerly `agentic-context`) are documented here. This project
follows [Semantic Versioning](https://semver.org/).

## [0.7.0]

### Changed — packaging
- The distribution is now `agent-context-sdk` (`pip install agent-context-sdk`).
  Import names are unchanged (`agentic_context`, `agentic_context_langgraph`, …).
- Released under the Salesforce Public Code License (`LICENSE.txt`), replacing
  the Apache-2.0 metadata. `LICENSE.txt` now ships in the sdist.

### Fixed — adapters label their agent type in Beacon
- The LangGraph, Claude, Google ADK and Foundry adapters now send their
  framework as Beacon's `AiAgentType` (`"LangGraph"`, `"ClaudeAgentSDK"`,
  `"GoogleADK"`, `"MicrosoftAgentFramework"`) instead of the SDK default
  `"Custom"`, so Beacon analytics can tell the frameworks apart. This
  matches what the OpenAI adapter already did.
- Each of the four constructors takes a new `agent_participant_variable=`
  kwarg, a full override of the `AiAgentParticipant` block (same semantics
  as `OpenAIAdapter`). LangGraph and ADK raise `ValueError` if it is
  combined with an injected `session_manager=`; set it on that manager
  instead.

### Fixed — LangGraph: a cancelled tool is not a tool error
- With newer `langchain-core`, which reports `CancelledError` to
  `on_tool_error`, a tool cancelled mid-turn was recorded as
  `"Tool raised CancelledError"` and fired `on_error(phase="tool")`. It now
  gets the documented `"Tool did not finish before the turn ended"` result
  and no `on_error`, whatever the `langchain-core` version.

### Fixed — a turn's entries share one `interaction_id`
- Claude: every record from one Stop hook now carries the same
  `interaction_id`, `turn-<session_id>-<first transcript uuid>`, so a re-read
  of the same entries gets the same id (a random suffix is used if no entry
  has a `uuid`). Entries written to the transcript after the Stop hook fires
  are grouped with the next turn. Previously `RecordTransformer` set none, so no
  entries were grouped. `RecordTransformer.transform()` takes a new
  `interaction_id=` kwarg.

### Changed — OpenAI `run_hooks`: one turn, one `interaction_id`, one POST
- One `run_hooks(...)` instance is one turn. It mints one `interaction_id`
  (exposed as `hooks.interaction_id`), buffers every record of the run, and
  ingests them in one `/entries` POST from `on_agent_end`. Previously it
  POSTed after every model call and every tool result and minted a new
  `interaction_id` per model call, so a tool turn made 4 POSTs across
  several groups.
- New `user_input=` kwarg records the user message in the same batch and
  group. Pass the user message here instead of a separate `push(...)`.
- New `await hooks.flush()` ingests whatever was captured. Call it in a
  `finally`, because `on_agent_end` does not fire when the run raises
  (including max turns), trips an input guardrail, stops for approval or is
  cancelled mid-stream. It is idempotent. With `Runner.run_streamed`, wrap the
  `stream_events()` loop in the `try`.
- **Migration:** partial runs used to be posted as they happened. Now they are
  held in memory until `flush()`. If you don't call it, `adapter.close()`
  posts them at shutdown, with a warning for each, and they are lost if the
  process crashes first. Existing code that pushes the user message itself
  keeps working, but that message gets its own `interaction_id`, so pass it
  as `user_input=` instead.
- Local fallback `event_id`s (Chat Completions `__fake_id__` items,
  reasoning without an id) stay distinct across a run's model calls.
  `RecordTransformer.transform()` takes a new `fallback_offset=` kwarg.
- Under `context_mode="auto"`, the once-per-session fast-context seed now
  runs from `on_agent_start`, since the user message no longer goes through
  `push()`. The first model call of a session's first run waits on it.

### Changed — LangGraph streaming matches the other adapters
- `LangGraphAdapter.stream()` now always captures `tool_call` /
  `tool_result` records. Previously they were captured only when an
  `on_tool_start`, `on_tool_end` or `on_error` hook was passed, so the
  documented basic setup silently dropped them. The tool hooks are now
  observation only.
- `stream()` ingests each turn as one batch — one `/entries` POST per turn,
  in chronological order — instead of one POST per record. Tool records are
  also yielded from `stream()` / returned by `run()` in order. If the graph
  raises, the caller stops iterating early, or `close()` / `end_session()`
  runs while a turn is unfinished, the records captured so far are still
  ingested in that batch. Beacon now receives a turn's records when the
  turn ends rather than as each one is produced, as with the Claude, ADK
  and Foundry adapters. A process crash mid-turn therefore loses the whole
  turn, including the user message, which used to be posted before the
  graph ran.
- Yielded / returned records now include tool records, which have
  `content=None`, and a turn can end on one. Code that reads
  `records[-1].content` or treats every non-user record as agent text
  should filter on `event_type`.
- A tool that raises now gets a `tool_result` record
  (`OutputValueText="Tool raised <ExceptionType>"`, `metadata["error"]=True`).
  Only the exception type is written to Beacon — messages can carry URLs,
  tokens or DSNs; the full exception still goes to `on_error`. A tool still
  running when the turn ends (for example, cancelled) gets a
  `"Tool did not finish before the turn ended"` error result. A tool that
  calls `interrupt()` is not treated as an error: its `tool_call` stays
  open, and on resume the tool runs again with a new call/result pair.
- `tool_result.OutputValueText` is now the tool's content rather than a
  repr — a `ToolMessage`'s content, a `Command`'s `ToolMessage` content,
  dicts and lists as JSON — capped at 2000 characters as in the ADK adapter
  (in both streaming and snapshot mode; snapshot list content was
  previously `str(list)`). This includes the SDK's own recall and
  context-pack tools, whose output is now always written to Beacon as
  traces.
- `close()`, `end_session()` and `__aexit__` now ingest any unfinished
  `stream()` turn before closing its manager. If another task is still
  driving that turn, it is cut off there: later records are not ingested.
- `ErrorEvent.phase` gains `"ingest"` (typing-visible: exhaustive `match` /
  `assert_never` over `phase` needs a new arm). It reports a failed ingest
  of a turn cut short by an error or early exit (logged, not raised, so it
  can't mask the original error), and records produced after
  `close()` / `end_session()` already sent the turn.
- The tool-capture handler ignores LangChain's chain, LLM, chat-model,
  retriever, retry and custom events, so always registering it adds no
  per-event overhead outside tool calls.

### Fixed
- A 2xx response with a non-JSON body from an ingestion POST is now treated as
  success (logged as a warning). Previously the sync client raised
  `requests.JSONDecodeError` (an `OSError`), so the retry layer re-sent the
  batch and wrote duplicate entries; the async client reported the delivered
  batch to `on_error`. A JSON body that isn't an object (`null`, a string, a
  list) is also returned as `{}`. The warning includes the content type and
  the first 200 bytes of the body.
- `RetrievalClient` / `AsyncRetrievalClient` now raise `IngestionError` (with
  the 2xx status, content type and a body snippet) on a 2xx non-JSON body,
  such as an HTML page from a proxy or a wrong `base_url`, instead of a raw
  `JSONDecodeError`. It is not retried. JSON objects and lists pass through;
  other JSON values (`null`, a string, a number) are returned as `{}`.

### Removed (breaking)
- The deprecated `BeaconAuth` / `UserBeaconAuth` aliases (from 0.6.0) are
  removed from `agentic_context` and `agentic_context.auth`. This is earlier
  than the 1.0 removal announced in 0.6.0. Import `ContextAuth` / `UserContextAuth` instead.

### Added — AgentContextClient (preview)
- `AgentContextClient`: an adapter-free client for custom agent loops. The
  host opens the conversation (`start_conversation`) and ingests each message
  (`ingest_customer_message` / `ingest_agent_message`); the model gets two
  mid-turn tools, `ace_get_context` and `ace_get_context_pack`, via
  `tool_specs()` / `call_tool()`. Preview: exported, not yet
  semver-committed.
- `start_conversation` raises `ValueError` unless a later `auth` is provably
  the bound identity without a token fetch: same auth class, agent id, login
  host, consumer key, `subject` / `xid`, and for `UserContextAuth` a resolved
  `user_id` equal to the bound one. Pass the bound auth object or omit `auth`.
- Guide: `docs/AGENT_CONTEXT_CLIENT.md`.

### Deprecated
- `AceSdkClient`, the pre-release name of `AgentContextClient`, still imports
  from `agentic_context` and `agentic_context.ace_sdk_client` but emits a
  `DeprecationWarning`. It is not in `__all__`, so `from agentic_context import *`
  does not bind it. Code that used it must also follow the stricter
  `start_conversation` auth check above: before, a later `auth` was ignored;
  now a fresh `UserContextAuth` (no token fetched yet) raises.

### Added — server-side system prompts
- `RetrievalClient.get_system_prompt` / `AsyncRetrievalClient.get_system_prompt`
  call `GET /api/v1/agentic/prompts/system/{type}`. They return the
  `text/plain` prompt body as-is, or `None` on 404/410.
- `SYSTEM_PROMPT_TYPE_CONTEXT_PACK` is public.
- Every adapter now injects the context-pack instruction automatically on each
  turn, in front of the `[Available context packs]` catalog and after the
  customer's own prompt, whenever the session's catalog lists at least one pack.
  No customer code is needed. The first turn of a session
  fetches the server-published instruction and pins it for the rest of that
  session. A 404/410 or a fetch error pins the bundled
  `CONTEXT_PACK_INSTRUCTION` for that session instead. After a fetch error, new
  sessions skip the fetch and use the bundled text for 10 s, so a Beacon outage
  doesn't stall every new session's first turn. If the customer's prompt already
  contains the instruction (server or bundled text), ADK, OpenAI, Foundry and
  Claude don't add it again. LangGraph can't see the prompt inside the graph, so
  it doesn't check (see the QUICKSTART note). Pinned instructions are capped per
  adapter (least recently used session dropped first) and cleared on `close()`.
- `get_context_pack_instruction(session_id)` on every adapter returns the pinned
  instruction. It is an escape hatch for customers who opt out of injection and
  build their own prompt. LangGraph gains `retrieval_client=` (manager-injection
  mode) for the prompt fetch.

### Changed — retrieval client and Claude `apply_to`
- All retrieval endpoints honour `Retry-After` on retryable responses (408/429/5xx),
  capped at 30 s, instead of always using jittered backoff.
- A 401 refresh only re-fetches the token if no concurrent caller has already
  rotated it.
- `get_context_pack_list` treats a page missing `contextPacks` / `hasMore` as
  empty / last instead of raising `KeyError`.
- Claude `apply_to` no longer edits `system_prompt`. The context-pack
  instruction now rides in the `UserPromptSubmit` hook's `additionalContext`,
  ahead of the catalog. `add_system_prompt_guidance=False` turns that off. The
  hook skips the instruction when `system_prompt` already contains it (server or
  bundled text, in a `str`, a preset's `append`, or a custom `prompt`). For
  manual wiring, the new `hooks(..., inject_context_pack_instruction=False)`
  kwarg turns it off, and `hooks(..., system_prompt=...)` enables the
  already-present check.
- Claude attaches the instruction and the `[Available context packs]` catalog
  once per Claude conversation instead of on every turn. `additionalContext`
  stays in Claude's history, so repeating it only cost tokens. The hook attaches
  them again for a new Claude conversation id, after a compaction (new
  `PreCompact` hook), and when the catalog text changes (a `version_expired`
  patch). They count as delivered only once that turn's `Stop` hook fires, so a
  turn that fails first gets them again on the next turn. If the hook input has
  no conversation id, it still attaches every turn.

### Fixed
- ADK: a prompt that already contained the context-pack instruction never got
  the `[Available context packs]` catalog, because the duplicate guard matched
  the instruction's quoted mention of the marker. The guard now matches only the
  catalog block header.
- Claude: calling `apply_to` (or `hooks(existing_hooks=...)`) twice on the same
  options registered the adapter's hooks twice, so every turn was ingested
  twice. The adapter now replaces its own earlier matchers and keeps yours.

### Changed — `get_context` is now a POST
- `RetrievalClient.get_context` and `AsyncRetrievalClient.get_context` send
  `POST /api/v1/agentic/context` with a JSON body (`conversation_id`,
  `owner_id`, `agent_id`, `owner_type`) instead of a GET with query string.
  Matches the merged `GetContextRequest` schema in
  the server API spec, which is `additionalProperties: false`
  and defines only those four identifier fields today.

### Removed — deep-mode kwargs and helpers
- `RetrievalClient.get_context` / `AsyncRetrievalClient.get_context` no longer
  accept `mode`, `query`, `latency_sensitivity`, or `token_budget`. The merged
  `GetContextRequest` schema (the server API spec) is
  `additionalProperties: false` so the kwargs could never reach the wire;
  retiring them matches the server contract. SessionManager delegates lose
  the same kwargs.
- LangGraph adapter: dropped `deep_context_latency_sensitivity`,
  `deep_context_token_budget` constructor kwargs; removed `get_deep_context`,
  `make_deep_context_node`, and the deep plumbing in the recall tool.
  `context_mode="per-turn"` now seeds fast context every turn without a
  parallel deep instruction.
- Claude and ADK adapters: dropped the equivalent `get_deep_context` methods,
  deep ctor kwargs, and the `[Context guidance]` sufficiency instruction
  injected under `context_mode="auto"`. ADK's `_resolve_context_mode` collapses
  to a single `inject_at_open` bool; the recall tool signature narrows to one
  optional `topic` hint.
- OpenAI adapter: dropped `deep_context_latency_sensitivity`,
  `deep_context_token_budget` constructor kwargs and `get_deep_context`; the
  `recall_customer_context` function tool now takes a single optional `topic`
  hint and always does the fast profile read (parity with the other adapters).

### Changed — participant ids on the wire
- Owner and agent participant ids are now sent to Beacon as the plain owner id /
  agent id, without the `-participant` suffix (previously `{owner_id}-participant`
  and `{agent_id}-participant`). Affects `create_session` participants in the
  sync and async clients, the session managers, and the ADK and LangGraph
  adapters' `participant_id` on emitted `Record`s. The adapter docs
  (`docs/DEVELOPER_GUIDE.md`) now describe the
  new convention.
- **Data compatibility:** anything keyed on the old `...-participant` ids
  (analytics, queries, governance scoping) will see old and new values side by
  side for sessions written before and after this change. There is no backfill.

### Added — consumer-key-only user login
- **`UserContextAuth.login(consumer_key=..., login_url=..., aid=...)`** signs the
  user in through their browser (OAuth web-server flow + PKCE) using only the ECA
  consumer key — no private key, no consumer secret. It is for local tools (CLI,
  notebook, desktop) with no web server of their own: the SDK runs a short-lived
  loopback listener for the one OAuth redirect, hence the `port` argument. A
  backend that serves users hosts its own callback instead (see "backend-hosted
  user login" below). The returned auth acts as that user. Requires the loopback callback
  `http://localhost:<port>/callback` registered on the ECA and every requested
  `scope` (default `api refresh_token cdp_ingest_api`) granted to it.
  The default port is `53682` (rarely occupied, unlike `8080`); pass `port=` to
  change it. If the port is busy, `login()` raises an `OSError` naming the port
  rather than trying another one, since only a registered callback URL works.
- `UserContextAuth(consumer_secret=...)` is now optional (public client).
- `on_refresh_token` callback on `UserContextAuth` / `login()` reports a rotated
  refresh token so callers can persist it; the instance also uses the new value
  for its next refresh.

### Added — backend-hosted user login
- **`UserContextAuth.authorization_request(consumer_key=..., login_url=..., redirect_uri=...)`**
  returns an `AuthorizationRequest` (authorize `url`, `state`, PKCE
  `code_verifier`) for a backend `/login` route. `redirect_uri` must be
  `https://` (`http://` only on loopback) and registered on the ECA.
- **`UserContextAuth.redeem_code(...)` / `aredeem_code(...)`** finish the flow in
  the backend `/callback`: they check `state` in constant time, send
  `client_secret` (confidential client) together with the PKCE verifier, and
  return a `UserLogin` (`refresh_token`, `user_id`, `org_id`, `instance_url`).
  Secrets are left out of `repr`.
- **`ReauthRequired`** is raised when Salesforce rejects the refresh token or
  code (`invalid_grant`, or a bare 401), and when no refresh token is stored.
  It is never retried. Previously this surfaced as a raw `httpx.HTTPStatusError`.
  `invalid_client` and other errors are unchanged.
- **Pluggable storage** (`agentic_context.stores`): `LoginStateStore`
  (`save` / one-time `consume` of state + PKCE verifier + redirect URI) and
  `RefreshTokenStore` (`load` / `save` / `delete` by `(org_id, user_id)`), each
  with in-memory and file implementations. The file stores keep plaintext
  secrets (mode 0600) and are for development only.
- **`UserContextAuth.from_store(store, org_id=..., user_id=..., ...)`** wires a
  `RefreshTokenStore` in: it loads before each refresh, saves a rotated token,
  and deletes a dead one before `ReauthRequired` propagates. The delete is a
  compare-and-delete (`delete(..., if_token=<rejected token>)`), so a fresh login
  saved in the meantime is never wiped; custom stores must implement `if_token` atomically.
- **Sign-in audit records**: `login()`, `redeem_code()` and `aredeem_code()` log an
  `audit: user_login` record (INFO on success, WARNING on denial) with method,
  outcome, org/user id, source IP and a failure reason — never codes or tokens.
  `redeem_code`/`aredeem_code` take an optional `client_ip` for the callback's
  source address (logged as `unknown` otherwise).
- `UserContextAuth` never sends an a360 `xid` and takes no `xid` argument: the
  token already belongs to the signed-in employee. `xid` stays on `ContextAuth`
  for the integration-user model.
- **`UserContextAuth(load_refresh_token=...)`** reads the latest stored refresh
  token before each refresh, and once more after an `invalid_grant`, so backend
  replicas that share a user's rotating token don't force a re-login. With a
  loader, `refresh_token=` is optional.

## [0.6.0]

### Added
- `FoundryAdapter` integrates Microsoft Agents (Foundry) with the context service.
    The adapter reads Foundry agent turns and emits them as `Records` for ingestion into 
    the Context Service, and also retrieves context for 
    the agent with `GET /context`.


### Changed — rename (BeaconAuth → ContextAuth)
- Auth class `BeaconAuth` renamed to **`ContextAuth`**. The old name is kept
  as a backwards-compatible alias (`BeaconAuth = ContextAuth`) so existing
  imports (`from agentic_context import BeaconAuth`) keep working — both names
  resolve to the same class object and instances are interchangeable.
- **Deprecation:** the `BeaconAuth` alias will be **removed in 1.0**. Migrate
  imports to `ContextAuth` before then.
- Test files renamed: `tests/test_beacon_auth.py` → `tests/test_context_auth.py`
  and `adapters/langgraph/tests/test_beacon_auth_wiring.py` →
  `adapters/langgraph/tests/test_context_auth_wiring.py`.

## [0.5.0]

The **memory → context** rebrand and single-install packaging release.

### Changed — packaging
- **One install, no extras.** `pip install agentic-context` now ships the core
  SDK plus all three framework adapters (ADK, Claude, LangGraph) in a single
  wheel. The old per-framework extras (`[adk]` / `[claude]` / `[langgraph]` /
  `[all]` / `[auth]`) are **removed** (#39).
- Framework SDKs (`google-adk`, `claude-agent-sdk`, `langgraph` /
  `langchain-core`) are **no longer runtime dependencies** — you already run
  your framework. They live in the `dev` extra for contributors only.
- `cryptography` is now a core dependency (JWT signing for `BeaconAuth`).
- Missing-SDK UX: constructing an adapter without its framework SDK installed
  emits a soft warning; the first method that needs the SDK raises a friendly
  `ImportError` naming the exact `pip install …` fix (original error chained).

### Changed — rename (memory → context)
- Distribution renamed `agentic-memory` → **`agentic-context`**; core package
  `agentic_memory` → **`agentic_context`**; adapters → `agentic_context_adk`,
  `agentic_context_claude`, `agentic_context_langgraph` (#35).
- ADK adapter class → **`GoogleAdkAdapter`** (was `GoogleADKAdapter`). A lazy
  top-level façade means `from agentic_context import GoogleAdkAdapter` works.
- `MemorySource` → **`ContextSource`**.
- **All three adapters** now share one recall surface: constructor kwarg
  `memory_mode` → **`context_mode`** and recall tool `recall_customer_memory` →
  **`recall_customer_context`**. ADK also renamed its system-prompt markers
  `[Memory guidance]` / `[Memory lookup required]` /
  `[Cross-session memory from Data Cloud]` → `[Context guidance]` /
  `[Context lookup required]` / `[Cross-session context from Data Cloud]`.
  Claude, ADK, and LangGraph are now identical on this surface.
- Secret-server env var `AGENTIC_MEMORY_SECRET_SERVER_URL` →
  **`AGENTIC_CONTEXT_SECRET_SERVER_URL`** (clean rename — the SDK is
  pre-release, so no old name to preserve).

### Changed — retrieval
- **Always-deep recall.** The recall tool always calls `get_deep_context(query)`;
  an empty `query` passes through and Beacon falls back to fast profile context
  **server-side**. The audit log line is now `mode=deep has_intent=<bool>`
  (was a `mode=fast|deep` client-side branch). Applied to ADK and LangGraph;
  Claude tracks this in a separate in-flight change.

### Added
- `context_mode="per-turn"` on ADK hard-forces the recall tool call every turn
  via Gemini `FunctionCallingConfigMode.ANY` + a per-session release latch, so
  the lookup is guaranteed to run (the deep-vs-fast choice stays the agent's).
