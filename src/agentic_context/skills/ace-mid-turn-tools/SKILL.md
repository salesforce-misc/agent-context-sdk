---
name: ace-mid-turn-tools
description: >-
  AgentContextClient model tools: ace_get_context, ace_get_context_pack.
---

# ace-mid-turn-tools

Design.md §8.

**Lifecycle** (host): `start_conversation`, `ingest_customer_message`,
`ingest_agent_message`

**Mid-turn tools** (model via `tool_specs` / `call_tool`): `ace_get_context`,
`ace_get_context_pack`

```python
from agentic_context import AgentContextClient

ace = AgentContextClient()
opened = ace.start_conversation(conversation_id, title=title, auth=auth, agent_id=agent_id)
# Inject opened["system_prompt_suffix"]; register ace.tool_specs()

ace.ingest_customer_message(user_text, conversation_id)
# run model; on tool call: ace.call_tool(name, arguments)
ace.ingest_agent_message(agent_reply, conversation_id)
```

## Mid-turn tools

- `ace_get_context` — latest customer memory (`POST /api/v1/agentic/context`) (Claude adapter: `recall_customer_context`)
- `ace_get_context_pack` — id from `[Available context packs]`; `params` per
  `CONTEXT_PACK_INSTRUCTION` (Claude adapter: `get_context_pack`)
- Do not call lifecycle APIs. Do not pass `owner_id` or `agent_id`.

| `status` | Action |
|----------|--------|
| `ok` | Use `context` |
| `not_ready` | Use conversation context; do not retry repeatedly |
| `invalid_pack` | Choose from `valid_pack_ids` |
