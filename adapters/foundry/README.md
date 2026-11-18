# agentic-context-foundry

Microsoft Foundry adapter for the [agent-context-sdk](https://github.com/salesforce-misc/agent-context-sdk) SDK.

This adapter reads Foundry agent turns and emits them as Records for ingestion into the Context Service, and retrieves context for the agent with GET /context.

NOTE: Foundry (`agent_framework`) is not bundled. If Foundry is missing, constructing `FoundryAdapter` warns and names the install needed.


```python
from agentic_context_foundry import FoundryAdapter

adapter = FoundryAdapter()
```
