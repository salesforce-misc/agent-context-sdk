# agentic-context-claude

Claude Agent SDK adapter for agent-context-sdk.

## Usage

```python
from agentic_context_claude import ClaudeAdapter

# Credential-based auth — the SDK runs the JWT → CDP exchange, discovers the
# base_url, and refreshes the token automatically.
adapter = ClaudeAdapter(
    agent_id="support-agent",
    owner_id="customer@corp.com",
    private_key=open("server.key").read(),   # PEM RSA private key
    consumer_key="3MVG...",                   # Connected App consumer key
    subject="you@example.com",                # Salesforce username for JWT bearer
    login_url="login.salesforce.com",         # login host (no scheme)
)

# In your Stop hook (async by default):
async def on_stop(input_data, tool_use_id, context):
    await adapter.push(input_data)
    return {}

# When the app shuts down:
await adapter.close()
```

See `docs/Design.md` §3 for the full design specification.
