# Adapters

One subpackage per agent framework. Each adapter:

- Depends on `agentic_context` core for `Record`, `Capabilities`, `ContextSource`.
- Imports the framework's own SDK as its only other heavy dep.
- Is published as its own optional install — users who only need LangGraph don't pull in `anthropic`, etc.

Planned adapters:

| Subpackage | Framework | Status |
|---|---|---|
| `agentic_context_langgraph` | LangGraph | planned |
| `agentic_context_claude` | Claude Agent SDK | planned |
| `agentic_context_adk` | Google ADK | planned |
| `agentic_context_foundry` | Microsoft Agent Framework | FoundryAdapter |

## Contract

Every adapter MUST satisfy the `ContextSource` Protocol from `agentic_context.models`:

```python
class ContextSource(Protocol):
    agent_id: str
    capabilities: Capabilities
    def snapshot(self, session_id: str) -> list[Record]: ...
```

Verify with `isinstance(adapter, ContextSource)` — it's `@runtime_checkable`.

## Rules (non-negotiable)

1. **Order is the caller's contract.** Records returned from `snapshot()` MUST be in event order. Downstream consumers depend on it.
2. **`id` must be stable across re-reads.** Same event, same id. It is a local id (logs, callbacks) — not sent, so nothing is deduped on it.
3. **`content` is always a string.** Flatten structured payloads (tool-call args, JSON outputs); put the structured form in `metadata` if it matters.
4. **`metadata` must be JSON-serializable.** No model objects, no datetimes (use isoformat strings), no bytes. Coerce at the boundary.
5. **Don't import the transport.** Adapters never POST. They produce records; user code or `ship()` posts them.
6. **Don't catch errors silently.** If reading framework state fails, raise. The caller decides retry.

## Layout for a new adapter

```
adapters/
└── <framework>/
    ├── pyproject.toml          # name = agentic_context_<framework>
    ├── src/agentic_context_<framework>/
    │   ├── __init__.py         # re-export the adapter class
    │   └── adapter.py          # the ContextSource implementation
    └── tests/
```

The adapter's `pyproject.toml` should list `agent-context-sdk` as a dependency and the framework's SDK as its other runtime dep. Keep optional features (e.g., persistent-tier readers) behind `[project.optional-dependencies]`.

## Streaming

If your framework emits live turn events, expose an `async def stream(inputs, session_id) -> AsyncIterator[Record]` alongside `snapshot()`. Streaming is opt-in per adapter; the core does not require it. One record per turn — never per token.
