# Adapter smoke tests

One scenario, one set of checks, every adapter. Each adapter runs the same two-turn conversation through its **real framework**. The Beacon traffic it produces is checked against the core ingestion contract (`docs/Design.md`):

1. "Where is my order A-1001?" → the agent calls `lookup_order` → it replies.
2. "Thanks! Please send me updates by email." → it replies with no tool call.

No real LLM is ever called. Both tiers use the same scripted fake model, plugged into each framework's own model seam (Claude gets a fake `claude` CLI). Every run sees the same replies, so the checks are deterministic.

| Tier | Marker | Beacon | Needs |
|---|---|---|---|
| 1 — offline | `offline` | `respx` mock that records every call | nothing |
| 2 — live | `live` | real dev Beacon (traffic passed through and recorded) | `SMOKE_LIVE=1` + Beacon credentials |

## Running

```bash
.venv/bin/python smoke/run_smoke.py                         # all adapters, offline
.venv/bin/python smoke/run_smoke.py -a adk openai           # a subset
.venv/bin/python smoke/run_smoke.py -t live -a claude       # one adapter, live
.venv/bin/python smoke/run_smoke.py -t all                  # both tiers, everything
.venv/bin/python smoke/run_smoke.py --list
.venv/bin/python smoke/run_smoke.py -a foundry -- -x -vv    # extra args go to pytest
```

Plain pytest works too. You always need `-c smoke/pytest.ini`, because the root config turns warnings into errors:

```bash
.venv/bin/python -m pytest -c smoke/pytest.ini smoke/test_openai.py -m offline
SMOKE_LIVE=1 .venv/bin/python -m pytest -c smoke/pytest.ini smoke/ -m live
```

## What is checked

These checks live in `harness.SmokeContract` and are shared by every adapter and both tiers. They are exact: a duplicated, missing or misplaced entry fails.

- **Lifecycle**
  - Exactly one each of `create_conversation`, `create_session` and `add_participants`; all are 2xx (Beacon answers `add_participants` with 202), and 409 counts as success.
  - `conversationId` is the session id and `conversationOwnerId` is set.
  - No `close` calls ever.
- **Agent label:** `AiAgentType` matches the framework, e.g. `"LangGraph"`.
- **Entry order:** exactly user → tool_call → tool_result → agent_response, then user → agent_response. `agent_reasoning` entries may appear in between.
- **User messages:** the scripted texts, with `participantId == ownerId`, so `role == "owner"`.
- **Tool calls:** exactly one `tool_call` and one `tool_result`, both `Trace` with no `content`. The call's `variable.Name` is the tool name as the framework reports it and `InputValueText` holds the order id; the result's `OutputValueText` holds the tool output.
- **Replies:** the `agent_response` contents are exactly the scripted replies, posted as the agent participant. The framework also returns those replies to the caller.
- **Turns**
  - Every entry carries its turn's `interactionId`, and no two turns share one.
  - Exactly one `/entries` POST per turn, holding the whole turn.
- **Requests**
  - A bearer token on every call (the fake one offline, a real one live). Recorded calls keep only which kind was sent, never the token.
  - No calls to Beacon paths outside the known endpoints.

A request that fails with a status the SDK retries (408, 429, 5xx) or a connection error, and is then sent again, counts once. Retries don't look like duplicates.

The live tier also reads the conversation back with `GET /conversations/{id}`, polling for up to 15 s, and checks that every user message and reply is there.

### Known gaps

A check an adapter is known to fail goes in that adapter's `known_gaps` dict, along with the reason. It reports as **xfail**. If the adapter is fixed, that check **fails** with "known gap … now passes". Delete the entry at that point; gaps can't quietly linger.

On the live tier, `entries_accepted` also xfails when every failing `/entries` POST was answered with `reject-duplicate-conversation`, and the read-back check xfails when any was. Some dev tenants do this server-side for every write; it is not an adapter bug.

## Live tier setup

Put these variables in `examples/ace_agent.env`, or in your shell; shell values win. Set `SMOKE_ENV_FILE` to use a different file.

**Always required**

| Var | What it is |
|---|---|
| `SF_CONSUMER_KEY` | Connected-app consumer key |
| `SF_LOGIN_URL` | Salesforce login URL |
| `BEACON_AGENT_ID` | A Beacon agent registered on your tenant |

**Plus one auth mode** (pick with `SMOKE_AUTH`)

| Mode | Needs | Notes |
|---|---|---|
| `ent`: service-account JWT | `SF_PRIVATE_KEY_PATH`, `SF_SUBJECT`, `BEACON_OWNER_ID` | Default when `SF_PRIVATE_KEY_PATH` is set |
| `emp`: logged-in user | A cached refresh token in `~/.cache/agentic_context_demos/` | Default otherwise. Run any `examples/` demo once to log in; the smoke run never opens a browser. `BEACON_OWNER_ID` is optional and defaults to the logged-in user. |

If a variable is missing, the live tests skip and say which one. Any other `SMOKE_AUTH` value is an error.

## Adding an adapter

Copy `test_openai.py` and follow these steps:

1. Write `drive()` for the framework.
2. Write a fake model that calls `harness.script_next`; both tiers use it.
3. Subclass `SmokeContract` and `LiveContract`, setting `agent_type` and `known_gaps` (and `tool_wire_name` if the framework renames tools, as Claude's MCP does).
4. Add the adapter's marker to `pytest.ini` and its name to `run_smoke.py`.

Agentforce isn't covered yet; its adapter has no tracked source in this repo.
