# agent-context-sdk

Give your AI agent cross-session memory backed by Salesforce Data Cloud.

`agent-context-sdk` connects an agent you already run to the Salesforce
Context Service (Beacon). It does two things:

- **Ingests** each conversation turn — user messages, agent responses,
  reasoning, tool calls and results — so the service can build up context.
- **Recalls** that context on later turns and sessions, injecting it into the
  agent's prompt and exposing it as tools (`recall_customer_context`,
  context packs).

It ships with adapters for **LangGraph**, the **Claude Agent SDK**,
**Google ADK**, the **OpenAI Agents SDK** and **Microsoft Foundry**, all in
one package.

## Install

```bash
pip install agent-context-sdk
```

Requires Python 3.10+. All five adapters are included. The framework SDKs
themselves (`langgraph`, `claude-agent-sdk`, `google-adk`, `openai-agents`,
`agent-framework-foundry`) are **not** bundled — you already run the one your
agent uses. If it's missing, the adapter tells you exactly which package to
install.

## Quick start

**1. Authenticate** with a Salesforce Connected App. For an end user signing
in from a local tool:

```python
from agentic_context import UserContextAuth

auth = UserContextAuth.login(
    consumer_key="3MVG...",            # Connected App consumer key
    login_url="login.salesforce.com",  # login host, no scheme
    aid="your_agent_id",               # agent id registered with the Context Service
)
```

For a backend service account, use `ContextAuth` (JWT bearer) instead — see
the [Quickstart](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/QUICKSTART.md#credentials).

**2. Wrap your agent** with its adapter. For example, the Claude Agent SDK:

```python
from claude_agent_sdk import ClaudeAgentOptions, query
from agentic_context import ClaudeAdapter

adapter = ClaudeAdapter(agent_id="your_agent_id", owner_id=auth.user_id, auth=auth)

options = adapter.apply_to(ClaudeAgentOptions(model="<claude-model>"), session_id="ses-123")
async for message in query(prompt="Hi, I need help with my order", options=options):
    ...
```

Every turn is now ingested, and prior context is recalled automatically. The
other adapters follow the same pattern:

```python
from agentic_context import (
    ClaudeAdapter, GoogleAdkAdapter, LangGraphAdapter, FoundryAdapter, OpenAIAdapter,
)
```

Imports are lazy — importing one adapter only loads that framework's SDK.

## AgentContextClient (preview)

If you don't use one of the supported frameworks, `AgentContextClient` gives
your own agent loop the same ingest and recall without an adapter. Your code
opens the conversation and ingests each message; the model gets two mid-turn
tools (`ace_get_context`, `ace_get_context_pack`) via `tool_specs()` /
`call_tool()`. It is a preview API and not semver-committed — see the
[AgentContextClient guide](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/AGENT_CONTEXT_CLIENT.md).

## Documentation

| Guide | What's in it |
|---|---|
| [Quickstart](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/QUICKSTART.md) | Credentials and per-framework wiring for all five adapters |
| [Developer guide](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/DEVELOPER_GUIDE.md) | Full per-adapter guide: prerequisites, constructors, troubleshooting |
| [Advanced](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/ADVANCED.md) | Backend-hosted login, token storage, service accounts, static tokens |
| [AgentContextClient](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/AGENT_CONTEXT_CLIENT.md) | Adapter-free client for custom agent loops (preview) |
| [Examples](https://github.com/salesforce-misc/agent-context-sdk/tree/main/examples) | Runnable demos |
| [Changelog](https://github.com/salesforce-misc/agent-context-sdk/blob/main/CHANGELOG.md) | Release notes |

## How it works

```
┌─────────────────────────────────────────────┐
│         Framework adapter layer             │
│ LangGraph / Claude / ADK / OpenAI / Foundry │
│  Turns agent events into Record objects     │
└────────────────────┬────────────────────────┘
                     │ list[Record]
                     ▼
┌─────────────────────────────────────────────┐
│    SessionManager / AsyncSessionManager     │
│  Creates the conversation + session lazily  │
│  Retries transient failures                 │
│  Async path never blocks the agent          │
└────────────────────┬────────────────────────┘
                     │ HTTPS
                     ▼
┌─────────────────────────────────────────────┐
│              Context Service                │
└─────────────────────────────────────────────┘
```

Each `Record` carries a stable local `event_id` (for logs and callbacks). It
is not sent to the service, so re-sending a turn writes duplicate entries.

## Development

```bash
pip install -e ".[dev]"     # SDK plus the framework SDKs the adapter tests need
pytest                      # core tests
pytest adapters/<fw>/tests  # one adapter's tests (adk, claude, foundry, langgraph, openai)
ruff check src tests        # lint
mypy src/agentic_context    # type-check
```

Layout: the core lives in `src/agentic_context/`; each adapter lives in
`adapters/<fw>/src/` and is built into the same wheel. For adapter design
rules, see [docs/Design.md](https://github.com/salesforce-misc/agent-context-sdk/blob/main/docs/Design.md).

## Support

Report bugs and request features through
[GitHub Issues](https://github.com/salesforce-misc/agent-context-sdk/issues).
This project does not accept external code contributions; see
[CONTRIBUTING.md](https://github.com/salesforce-misc/agent-context-sdk/blob/main/CONTRIBUTING.md).

## Security

Do not report security issues in GitHub Issues. See
[SECURITY.md](https://github.com/salesforce-misc/agent-context-sdk/blob/main/SECURITY.md).

## License

Licensed under the Salesforce Public Code License — see
[LICENSE.txt](https://github.com/salesforce-misc/agent-context-sdk/blob/main/LICENSE.txt).
