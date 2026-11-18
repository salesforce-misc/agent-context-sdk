# agentic-context-openai

OpenAI adapter for agent-context-sdk. Normalizes Chat Completions
messages or OpenAI Agents SDK Responses items into Beacon `Record`s and ships
them through the shared `SessionManager` lifecycle.

Primary runtime target: the OpenAI Agents SDK (`openai-agents`), the direct
analog of `claude-agent-sdk` / `google-adk` / `langgraph`. Also accepts a plain
Chat Completions messages list for the simple snapshot path — teams that only
drive `openai.chat.completions.create(messages=...)` can ship transcripts
without adopting a new runtime.

## Install

The adapter ships inside the root `agent-context-sdk` wheel. Install the core
plus the OpenAI Agents SDK:

```bash
pip install agent-context-sdk
pip install openai-agents  # only required for make_context_retrieval_tool()
```

## Usage — snapshot (Chat Completions)

```python
from agentic_context_openai import OpenAIAdapter

adapter = OpenAIAdapter(
    agent_id="support-bot",
    owner_id="customer@corp.com",
    private_key=open("server.key").read(),
    consumer_key="3MVG...",
    subject="user@org.com",
    login_url="login.salesforce.com",
)

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help?"},
]
await adapter.push("conv-123", messages=messages)
await adapter.close()
```

## Usage — snapshot (OpenAI Agents SDK)

```python
from agents import Agent, Runner

agent = Agent(name="support", instructions="…")
result = await Runner.run(agent, "Hello")
await adapter.push("conv-123", items=result.to_input_list())
```

Push a finished transcript once. Event ids are not sent, so nothing is
deduped: pushing the full transcript again every turn re-writes all earlier
turns. For per-turn ingestion, push only the new items (or use `run_hooks`).

## Shape conformance

```python
from agentic_context import ContextSource
assert isinstance(adapter, ContextSource)
```
