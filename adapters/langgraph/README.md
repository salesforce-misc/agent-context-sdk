# agentic-context-langgraph

LangGraph adapter for the [agent-context-sdk](https://github.com/salesforce-misc/agent-context-sdk) SDK.

Provides `LangGraphAdapter` — a single class with two construction paths:

- **Snapshot mode** — read messages from a live `StateGraph` or a serialized `StateSnapshot` on disk. Implements `ContextSource`.
- **Streaming mode** — drive `graph.astream_events(version="v3")`, yield each `Record` of the turn in order (user message, tool calls and results, replies), and ingest the whole turn in one POST when it ends. Tool records are always captured; no hook is needed.

Optional observation hooks (`on_session_start`, `on_session_end`, `on_turn_start`, `on_tool_start`, `on_tool_end`, `on_error`) fire during a stream without blocking it; hook exceptions are logged and swallowed.

## Install

```bash
pip install agent-context-sdk
```

## Quick start

```python
import asyncio
from agentic_context_langgraph import LangGraphAdapter

async def main():
    # Smart (self-driving) path: the adapter owns its clients and must be
    # entered as an async context manager. Connection params come from
    # api_key=/base_url= or the CDP_TOKEN/CDP_BASE_URL env vars.
    async with LangGraphAdapter(
        graph=graph, agent_id="my-agent", owner_id="user@org.com",
    ) as adapter:
        async for r in adapter.stream(inputs, session_id="sid-1"):
            ...

asyncio.run(main())
```

For end-of-session snapshot drains, the manager-injection paths (which do not
require `async with`), and full integration guidance — required flow, hook
reference, mode constraints, common pitfalls, verification — see
[`docs/DEVELOPER_GUIDE.md`](../../docs/DEVELOPER_GUIDE.md).
