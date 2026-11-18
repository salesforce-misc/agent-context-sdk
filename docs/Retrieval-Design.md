# Design — Retrieval Client & SessionManager Integration

**Authors:** Dhruv Kachhadia, Sylvester Kpei  
**Date:** 2026-06-17  
**Status:** Implemented (v0.4.0)  
**Branch:** `feature/retrieval-client`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [RetrievalClient (Sync)](#3-retrievalclient-sync)
4. [AsyncRetrievalClient](#4-asyncretrievalclient)
5. [SessionManager Integration](#5-sessionmanager-integration)
6. [Header Patterns](#6-header-patterns)
7. [Error Handling & Design Decisions](#7-error-handling--design-decisions)
8. [API Reference](#8-api-reference)
9. [Usage Examples](#9-usage-examples)
10. [Testing](#10-testing)

---

## 1. Overview

### 1.1 Problem Statement

The SDK originally only supported **writing** to Beacon (ingestion). Agents also need to **read** from Beacon to:

1. **Get context** — Retrieve in-session summary + cross-conversation user facts before each LLM turn (gives the agent memory)
2. **Get conversation** — Rehydrate a conversation on load, or verify what was written
3. **List conversations** — Populate a conversation picker UI, or resume prior sessions

### 1.2 Why a Separate Class?

The read-side API has **different header conventions** than the write-side:

| Operation | Header Style |
|-----------|-------------|
| POST (ingestion) | `dataspace`, `owner_id`, `registered_agent_id` |
| GET context | `ctx-dataspace-ds_name` |
| GET conversations (list/get) | `x-sfdc-dataspace`, `agent_id` |

A single class handling both would need mode switches for headers. Separation keeps each client focused:

- `IngestionClient` / `AsyncIngestionClient` → writes (POST)
- `RetrievalClient` / `AsyncRetrievalClient` → reads (GET)

`SessionManager` composes both, giving the application a single interface.

### 1.3 Design Principles

- **Best-effort retrieval:** Retrieval failure ≠ agent failure. The agent should still respond (just without memory) rather than crash.
- **Pre-bound identity:** SessionManager binds `conversation_id`, `owner_id`, `agent_id` at construction so callers never pass IDs for retrieval.
- **Normalization at the HTTP boundary:** `RetrievalClient` normalizes API response shapes (list wrappers) so callers always get predictable types.
- **No retries on retrieval:** Retrieval is on the critical user-facing path. Retries add latency; better to proceed without context than make the user wait.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Application / Agent                     │
│                                                              │
│   mgr.get_context()   mgr.ingest(records)   mgr.close()    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    SessionManager                             │
│                                                              │
│   Retrieval:                   Ingestion:                    │
│   • get_context() → None|dict  • ingest(records) → POST    │
│   • get_conversation()          • close() → close_session   │
│   • list_conversations()        • _ensure_lifecycle (409)   │
│         │                              │                     │
│         ▼                              ▼                     │
│   RetrievalClient              IngestionClient               │
│   (GET, read-only)             (POST, lifecycle)             │
└─────────────────────────────────────────────────────────────┘
```

**Key composition rule:** SessionManager holds both clients. The application only interacts with SessionManager.

---

## 3. RetrievalClient (Sync)

### 3.1 Construction

```python
from agentic_context import RetrievalClient

rc = RetrievalClient(
    base_url="https://beacon.example.com",  # required, must be http(s)
    api_key="Bearer token",                  # required
    dataspace="default",                     # optional, defaults to "default"
    timeout=10.0,                            # optional, seconds
    session=None,                            # optional, inject requests.Session
)
```

**Validations:**
- `base_url` must start with `https://` or `http://` (raises `ValueError` otherwise)
- `http://` emits a warning (credentials sent in clear text)
- Trailing slash is stripped

### 3.2 Endpoints

| Method | Endpoint | Returns |
|--------|----------|---------|
| `get_context(conversation_id, owner_id, agent_id)` | `POST /api/v1/agentic/context` | `dict[str, Any]` |
| `get_conversation(conversation_id, owner_id, agent_id)` | `GET /api/v1/agentic/conversations/{id}` | `dict[str, Any]` |
| `list_conversations(owner_id, agent_id)` | `GET /api/v1/agentic/conversations` | `list[dict[str, Any]]` |

### 3.3 Context Manager Support

```python
with RetrievalClient(base_url=..., api_key=...) as rc:
    ctx = rc.get_context(conv_id, owner_id, agent_id)
# connection pool closed automatically
```

---

## 4. AsyncRetrievalClient

Async sibling using `httpx.AsyncClient`. Same three endpoints, same validations, non-blocking I/O.

```python
async with AsyncRetrievalClient(base_url=..., api_key=...) as rc:
    ctx = await rc.get_context(conv_id, owner_id, agent_id)
```

---

## 5. SessionManager Integration

### 5.1 Construction (Optional Retrieval)

Retrieval is opt-in. Pass `retrieval_client` to enable it:

```python
from agentic_context import SessionManager, IngestionClient, RetrievalClient

ingestion = IngestionClient(base_url=..., api_key=...)
retrieval = RetrievalClient(base_url=..., api_key=...)

mgr = SessionManager(
    ingestion,
    agent_id="my-agent",
    owner_id="user-001",
    conversation_id="conv-123",
    retrieval_client=retrieval,       # ← optional
)
```

Without `retrieval_client`, the retrieval methods return `None` / `[]` immediately.

### 5.2 Retrieval Methods on SessionManager

| Method | Returns | On error | On no client |
|--------|---------|----------|--------------|
| `get_context()` | `dict \| None` | `None` | `None` |
| `get_conversation()` | `dict \| None` | `None` | `None` |
| `list_conversations()` | `list[dict]` | `[]` | `[]` |

**Zero arguments** — IDs are pre-bound from construction.

**Best-effort semantics:**
- Never raises exceptions
- Returns `None` or `[]` on any error (network, 4xx, 5xx)
- Logs at DEBUG level for observability
- Guards: returns immediately if no retrieval client or no `conversation_id` resolved yet

### 5.3 Async Variant

Same pattern on `AsyncSessionManager`:

```python
mgr = AsyncSessionManager(
    async_ingestion_client,
    agent_id="my-agent",
    owner_id="user-001",
    retrieval_client=async_retrieval_client,  # AsyncRetrievalClient
)

ctx = await mgr.get_context()        # best-effort, non-blocking
convos = await mgr.list_conversations()
```

### 5.4 Why No Retries on Retrieval

| | Ingestion | Retrieval |
|---|---|---|
| **Failure cost** | Data loss (permanent) | Agent responds without context (degraded, not broken) |
| **Who's waiting** | Nobody (background task) | The user (blocking LLM call) |
| **Retry acceptable?** | Yes — background, invisible | Risky — each retry adds latency |

Retrying retrieval means the user waits 1s + 2s + 4s before the agent starts generating. Worse UX than responding without memory.

---

## 6. Header Patterns

### 6.1 POST /api/v1/agentic/context

```
Authorization: Bearer {token}
ctx-dataspace-ds_name: {dataspace}
```

JSON body: `conversation_id`, `owner_id`, `agent_id`, `owner_type` (default `"individual"`). Matches the merged `GetContextRequest` schema, which is `additionalProperties: false` and defines only these four identifier fields today.

### 6.2 GET /api/v1/agentic/conversations/{id}

```
Authorization: Bearer {token}
x-sfdc-dataspace: {dataspace}
agent_id: {agent_id}
```

Query params: `owner_id`, `owner_type=individual`, `scope=all`

### 6.3 GET /api/v1/agentic/conversations (list)

```
Authorization: Bearer {token}
x-sfdc-dataspace: {dataspace}
agent_id: {agent_id}
```

Query params: `owner_id`, `owner_type=individual`

---

## 7. Error Handling & Design Decisions

### 7.1 HTTP 409 Handling in _ensure_lifecycle

**Problem:** If the server persists a resource (conversation/session) but the client doesn't observe the success (network timeout, connection reset), the client's `_conversation_created` flag stays `False`. On the next `ingest()` call, it retries `create_conversation` and gets 409 Conflict — permanently wedging the session.

**Solution:** Wrap lifecycle creation calls in try/except for `IngestionError` with `status_code == 409`. Treat 409 as success (set the flag to `True`). Non-409 errors still propagate.

```python
try:
    self._client.create_conversation(...)
except IngestionError as e:
    if e.status_code != 409:
        raise
self._conversation_created = True  # whether we created it or it already existed
```

### 7.2 URL Encoding

`conversation_id` values may contain special characters (slashes, spaces, Unicode). The `get_conversation` endpoint embeds the ID in the URL path, so it must be percent-encoded:

```python
from urllib.parse import quote
f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}"
```

### 7.3 List Response Normalization

The `list_conversations` API response shape varies across deployments:

| Shape | Example |
|-------|---------|
| Bare list | `[{"conversationId": "..."}, ...]` |
| Wrapped in "conversations" | `{"conversations": [...]}` |
| Wrapped in "data" | `{"data": [...]}` |
| Wrapped in "items" | `{"items": [...]}` |
| Empty | `{}` or `null` |

The `_normalize_list_response()` helper handles all shapes and always returns `list[dict]`. This normalization lives in `RetrievalClient` (HTTP boundary), not SessionManager.

### 7.4 base_url Validation

Both sync and async clients validate at construction:
- Must start with `https://` or `http://`
- `http://` logs a warning (credentials in clear text)
- Trailing slash stripped to prevent double-slash in URLs

---

## 8. API Reference

### 8.1 Public Exports (from `agentic_context`)

```python
from agentic_context import (
    RetrievalClient,         # Sync GET client
    AsyncRetrievalClient,    # Async GET client
    SessionManager,          # Sync manager (ingestion + retrieval)
    AsyncSessionManager,     # Async manager (ingestion + retrieval)
)
```

### 8.2 RetrievalClient

```python
class RetrievalClient:
    def __init__(self, base_url: str, api_key: str, *, dataspace: str = "default", timeout: float = 10.0, session: requests.Session | None = None) -> None
    def get_context(self, conversation_id: str, owner_id: str, agent_id: str, *, owner_type: str = "individual") -> dict[str, Any]
    def get_conversation(self, conversation_id: str, owner_id: str, agent_id: str, *, scope: str = "all", owner_type: str = "individual") -> dict[str, Any]
    def list_conversations(self, owner_id: str, agent_id: str, *, owner_type: str = "individual") -> list[dict[str, Any]]
    def close(self) -> None
```

### 8.3 AsyncRetrievalClient

```python
class AsyncRetrievalClient:
    async def get_context(...) -> dict[str, Any]
    async def get_conversation(...) -> dict[str, Any]
    async def list_conversations(...) -> list[dict[str, Any]]
    async def close(self) -> None
```

### 8.4 SessionManager Retrieval Methods

```python
class SessionManager:
    def get_context(self) -> dict[str, Any] | None
    def get_conversation(self) -> dict[str, Any] | None
    def list_conversations(self) -> list[dict[str, Any]]

class AsyncSessionManager:
    async def get_context(self) -> dict[str, Any] | None
    async def get_conversation(self) -> dict[str, Any] | None
    async def list_conversations(self) -> list[dict[str, Any]]
```

---

## 9. Usage Examples

### 9.1 Basic Agent Loop (Sync)

```python
from agentic_context import (
    IngestionClient, RetrievalClient, SessionManager
)

ingestion = IngestionClient(base_url=URL, api_key=TOKEN, dataspace=DS)
retrieval = RetrievalClient(base_url=URL, api_key=TOKEN, dataspace=DS)

mgr = SessionManager(
    ingestion,
    agent_id="support-bot",
    owner_id="user@example.com",
    conversation_id="conv-001",
    retrieval_client=retrieval,
)

# Before each LLM call — inject context into system prompt
context = mgr.get_context()  # None on first turn or error
system_prompt = build_prompt(context)

# After LLM responds — ingest the turn
mgr.ingest(records)

# When done
mgr.close()
retrieval.close()
```

### 9.2 Async Agent (FastAPI / Claude SDK)

```python
from agentic_context import (
    AsyncIngestionClient, AsyncRetrievalClient, AsyncSessionManager
)

async with AsyncIngestionClient(base_url=URL, api_key=TOKEN) as ingestion:
    async with AsyncRetrievalClient(base_url=URL, api_key=TOKEN) as retrieval:
        mgr = AsyncSessionManager(
            ingestion,
            agent_id="support-bot",
            owner_id=owner_id,
            conversation_id=conv_id,
            retrieval_client=retrieval,
        )

        # Get context (non-blocking, best-effort)
        ctx = await mgr.get_context()

        # Ingest (non-blocking, background delivery)
        await mgr.ingest(records)

        # Close (flushes pending, closes session)
        await mgr.close()
```

### 9.3 Listing Conversations for a UI

```python
mgr = SessionManager(
    ingestion_client,
    agent_id="my-agent",
    owner_id="user-42",
    retrieval_client=retrieval_client,
)

# No conversation_id needed for listing
conversations = mgr.list_conversations()
# Returns: [{"conversationId": "...", "name": "...", ...}, ...]
```

---

## 10. Testing

### 10.1 Test Coverage (23 tests)

| Category | Tests | Description |
|----------|-------|-------------|
| Sync headers/params | 5 | Verify correct headers for each endpoint |
| Async headers/params | 5 | Same for async client |
| Error handling | 2 | IngestionError on 4xx (sync + async) |
| base_url validation | 3 | Invalid scheme rejected, trailing slash stripped |
| URL encoding | 1 | Special chars in conversation_id |
| List normalization | 3 | Bare list, data wrapper, empty dict |
| SessionManager integration | 3 | Delegation, no-client fallback, error swallowing |
| 409 handling | 2 | 409 → success, non-409 → propagate |

### 10.2 Running Tests

```bash
cd agent-context-sdk
.venv/bin/python -m pytest tests/test_retrieval.py -v
```

### 10.3 Dependencies

- `responses` — mock sync HTTP (requests)
- `respx` — mock async HTTP (httpx)
- `pytest-asyncio` — async test support

---

## Changelog

### v0.4.0 (2026-06-17)

- **Added:** `RetrievalClient` and `AsyncRetrievalClient` for read-side API
- **Added:** `SessionManager.get_context()`, `.get_conversation()`, `.list_conversations()`
- **Added:** Same retrieval methods on `AsyncSessionManager`
- **Fixed:** 409 handling in `_ensure_lifecycle` (both sync and async)
- **Improved:** base_url validation on retrieval clients
- **Improved:** URL encoding for conversation_id in path segments
- **Improved:** `list_conversations` returns normalized `list[dict]` instead of raw response
