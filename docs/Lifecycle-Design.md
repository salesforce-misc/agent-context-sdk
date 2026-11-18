# Design — Eager Lifecycle Methods

**Authors:** Dhruv Kachhadia, Sylvester Kpei  
**Date:** 2026-06-17  
**Status:** Implemented (v0.5.0)  
**Branch:** `feature/eager-lifecycle`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Two New Methods](#2-two-new-methods)
3. [Design Decisions](#3-design-decisions)
4. [API Reference](#4-api-reference)
5. [Usage Examples](#5-usage-examples)
6. [Interaction with Lazy Lifecycle](#6-interaction-with-lazy-lifecycle)
7. [Testing](#7-testing)

---

## 1. Problem Statement

The original `SessionManager` only supports **lazy** lifecycle creation — conversation and session are created on the first `ingest()` call. This works for simple agent loops, but breaks for **UI-driven agents** where:

1. The user clicks "+ New Conversation" and the frontend needs IDs **before** any message is sent
2. The agent needs to re-open a conversation with a **new session** (multi-session memory pattern)

### Lazy vs Eager

| Mode | How it works | Best for |
|------|-------------|----------|
| **Lazy** (existing) | `ingest()` auto-creates conversation + session on first call | Simple scripts, CLI agents — no upfront setup needed |
| **Eager** (new) | `create_conversation()` explicitly creates before any message | UI agents where frontend needs IDs immediately |

---

## 2. Two New Methods

| Method | Purpose | When to use |
|--------|---------|-------------|
| `create_conversation(conversation_id, session_id)` | Eagerly create conversation + session + participants | Before any message — e.g., user clicks "+ New" |
| `start_new_session(session_id)` | Create new session under existing conversation | Re-opening a conversation (multi-session memory) |

Both methods exist on `SessionManager` (sync) and `AsyncSessionManager` (async).

---

## 3. Design Decisions

### 3.1 Why These Live on SessionManager (Not the Adapter)

SessionManager is the **single owner** of all Beacon lifecycle. If adapters bypassed SessionManager for eager creation but used it for lazy creation, you'd get split ownership — some lifecycle state tracked inside SessionManager, some outside. Every adapter would need to replicate the same 409 handling, participant registration, etc.

By putting eager methods on SessionManager, adapters remain thin wrappers that delegate **everything** to the manager.

### 3.2 409 Handling

Same pattern as `_ensure_lifecycle` — treat HTTP 409 as success. This makes `create_conversation()` idempotent (safe to retry on network timeout).

### 3.3 Best-Effort Close on `start_new_session()`

If closing the old session fails (e.g., server timeout), we log at DEBUG and proceed to create the new session. Blocking on a close failure would wedge the agent. Beacon will eventually time out the old session anyway.

### 3.4 Lock Protection (Async)

Both methods acquire `_lifecycle_lock` to prevent races with concurrent `ingest()` calls. This ensures that eager creation and lazy creation never run simultaneously.

### 3.5 Two-Step Participant Registration

Following Sylvester's fix for `get_context` scoping: session is created with the owner only, then the agent is added as a separate participant via `add_participants`. This ensures context retrieval is properly scoped to the agent's perspective.

---

## 4. API Reference

```python
# Sync
class SessionManager:
    def create_conversation(
        self,
        conversation_id: str,
        session_id: str,
        *,
        title: str = "",
    ) -> dict[str, str]:
        """Eagerly create conversation + session + participants.

        Returns {"conversationId", "sessionId", "ownerId", "agentId"}

        Raises:
            IngestionError: If any lifecycle call fails (non-409).
            RuntimeError: If the manager is already closed.
        """

    def start_new_session(self, session_id: str) -> dict[str, str]:
        """Create new session under existing conversation (multi-session).

        Closes old session (best-effort), creates fresh session + participants.
        Returns {"conversationId", "sessionId", "ownerId", "agentId"}

        Raises:
            IngestionError: If lifecycle calls fail (non-409).
            RuntimeError: If closed or no conversation exists yet.
        """

# Async
class AsyncSessionManager:
    async def create_conversation(
        self,
        conversation_id: str,
        session_id: str,
        *,
        title: str = "",
    ) -> dict[str, str]: ...

    async def start_new_session(self, session_id: str) -> dict[str, str]: ...
```

---

## 5. Usage Examples

### 5.1 Eager Creation (UI Agent)

```python
import uuid
from agentic_context import SessionManager, IngestionClient, RetrievalClient

ingestion = IngestionClient(base_url=URL, api_key=TOKEN)
retrieval = RetrievalClient(base_url=URL, api_key=TOKEN)

mgr = SessionManager(
    ingestion,
    agent_id="support-bot",
    owner_id="user@example.com",
    retrieval_client=retrieval,
)

# User clicks "+ New Conversation" — need IDs before any message
conv_id = str(uuid.uuid4())
sess_id = str(uuid.uuid4())
result = mgr.create_conversation(conv_id, sess_id, title="Support Request")
# → {"conversationId": "...", "sessionId": "...", "ownerId": "...", "agentId": "..."}

# Now ingest() will skip lazy lifecycle — goes straight to add_entries
mgr.ingest(records)
```

### 5.2 Multi-Session Memory Pattern

```python
# Day 1: User chats, session closes
mgr.create_conversation("conv-001", "sess-day1")
mgr.ingest(day1_records)
mgr.close()  # closes sess-day1

# Day 2: User re-opens the same conversation
mgr2 = SessionManager(
    ingestion,
    agent_id="support-bot",
    owner_id="user@example.com",
    conversation_id="conv-001",    # existing conversation
    retrieval_client=retrieval,
)

# get_context() returns summaries from Day 1 session
context = mgr2.get_context()

# Start a fresh session under the same conversation
mgr2.start_new_session("sess-day2")

# Ingest Day 2 messages — agent has memory of Day 1 via context
mgr2.ingest(day2_records)
mgr2.close()
```

### 5.3 Async (FastAPI Agent)

```python
mgr = AsyncSessionManager(
    async_ingestion,
    agent_id="support-bot",
    owner_id=owner_id,
    retrieval_client=async_retrieval,
)

# Eager creation
result = await mgr.create_conversation(conv_id, sess_id)

# Later — user re-opens
await mgr.start_new_session(new_sess_id)
ctx = await mgr.get_context()  # includes prior session summaries
```

---

## 6. Interaction with Lazy Lifecycle

| Scenario | What happens |
|----------|-------------|
| `create_conversation()` then `ingest()` | `ingest()` sees `_conversation_created=True` and `_session_created=True`, skips `_ensure_lifecycle`, goes straight to `add_entries` |
| `ingest()` first (lazy) then `start_new_session()` | Works — `start_new_session()` requires `conversation_id` to be set, which `ingest()` does via lazy creation |
| `start_new_session()` without prior conversation | Raises `RuntimeError` — must call `create_conversation()` or `ingest()` first |
| `create_conversation()` with 409 | Treated as success — all flags set, method returns normally |

---

## 7. Testing

### 7.1 Test Coverage (22 tests)

| Category | Tests | Description |
|----------|-------|-------------|
| Sync `create_conversation` | 6 | Full lifecycle, state setting, skip lazy, 409 success, non-409 raises, closed raises |
| Sync `start_new_session` | 5 | Creates under existing, updates state, no-conv raises, after-ingest works, close-failure doesn't block |
| Async `create_conversation` | 6 | Same as sync + lock-protected |
| Async `start_new_session` | 5 | Same as sync + lock-protected + concurrent safety |

### 7.2 Running Tests

```bash
cd agent-context-sdk
.venv/bin/python -m pytest tests/test_session.py -v -k "EagerLifecycle"
```

---

## Changelog

### v0.5.0 (2026-06-17)

- **Added:** `SessionManager.create_conversation()` — eager lifecycle creation
- **Added:** `SessionManager.start_new_session()` — multi-session pattern
- **Added:** Same methods on `AsyncSessionManager` (lock-protected)
- **Added:** 22 new tests covering eager lifecycle (sync + async)
