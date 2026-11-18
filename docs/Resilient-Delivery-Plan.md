# Resilient Delivery — Production-Ready Ingestion

> **Status:** Proposal  
> **Author:** Dhruv Kachhadia, Luke Shen, Sylvester Kpei  
> **Branch:** `feature/resilient-delivery`  
> **Depends on:** Design.md §2 (Core SDK)

---

## 1. Motivation

The current SDK ingests entries into Beacon **without retry logic or observability**. Additionally, the `AsyncSessionManager` blocks the calling coroutine until the HTTP round-trip completes — which stalls agent frameworks that call it from within their event loop.

**Production concerns:**

| Issue | Impact |
|-------|--------|
| No retries | A single Beacon 503 (transient) loses the entire turn's data |
| No backoff | Hammering a degraded Beacon makes recovery harder |
| No observability | Caller has no visibility into success/failure |
| AsyncSessionManager blocks the coroutine | Even though it uses `async/await`, the caller must wait for Beacon to respond before continuing — this stalls agent execution |

These are solved **once in the core SDK**. Adapters remain thin (parsing + transformation only).

---

## 2. Before (Current State)

### 2.1 SessionManager (sync)

```python
class SessionManager:
    def ingest(self, records):
        self._ensure_lifecycle()
        entries = self._to_beacon_entries(records)
        self._client.add_entries(...)  # blocks until HTTP response
        # no retry, no callback, throws on error
```

### 2.2 AsyncSessionManager (async — but still blocking the coroutine)

```python
class AsyncSessionManager:
    async def ingest(self, records):
        await self._ensure_lifecycle()
        entries = self._to_beacon_entries(records)
        await self._client.add_entries(...)  # caller waits here
        # no retry, no callback, throws on error
```

Despite using `async/await`, this is NOT non-blocking from the caller's perspective. The coroutine suspends until Beacon responds — meaning the agent framework is stalled during ingestion.

### 2.3 Summary (Before)

| Class | Blocks caller? | Retries | Callbacks | Production-ready? |
|-------|----------------|---------|-----------|-------------------|
| `SessionManager` | Yes (thread) | No | No | ❌ |
| `AsyncSessionManager` | Yes (coroutine) | No | No | ❌ |

---

## 3. After (Proposed State)

### 3.1 SessionManager — Sync, Blocking, With Retries

**Purpose:** Tests, scripts, debugging, and simple one-off usage where blocking is acceptable.

```python
class SessionManager:
    def __init__(self, client, *, agent_id, owner_id, callbacks=None):
        self._retry = _RetryPolicy()  # internal, not configurable
        self._callbacks = callbacks    # optional — None means no callbacks

    def ingest(self, records):
        """Blocking. Retries transient failures. Returns after delivery (or final failure)."""
        self._ensure_lifecycle()
        entries = self._to_beacon_entries(records)

        for attempt in range(self._retry.max_retries + 1):
            try:
                self._client.add_entries(...)
                if self._callbacks:
                    self._callbacks.on_success(records)
                return
            except IngestionError as e:
                if self._retry.is_retryable(e) and attempt < self._retry.max_retries:
                    delay = self._retry.backoff(attempt)
                    if self._callbacks:
                        self._callbacks.on_retry(records, e, attempt, delay)
                    time.sleep(delay)
                else:
                    if self._callbacks:
                        self._callbacks.on_error(records, e, attempt)
                    raise
```

### 3.2 AsyncSessionManager — Async, Non-Blocking, With Retries

**Purpose:** Production use by ALL agent framework adapters (Claude, LangGraph, ADK). The caller is never blocked by Beacon latency.

```python
class AsyncSessionManager:
    def __init__(self, client, *, agent_id, owner_id, callbacks=None):
        self._retry = _RetryPolicy()  # internal, not configurable
        self._callbacks = callbacks    # optional — None means no callbacks
        self._pending_tasks: set[asyncio.Task] = set()

    async def ingest(self, records):
        """Non-blocking. Fires a background task and returns immediately.
        
        The caller (adapter/agent framework) is never stalled by Beacon latency.
        Delivery happens in the background with automatic retries.
        """
        task = asyncio.create_task(self._deliver(records))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _deliver(self, records):
        """Background delivery with retry. Runs as a detached task."""
        await self._ensure_lifecycle()
        entries = self._to_beacon_entries(records)

        for attempt in range(self._retry.max_retries + 1):
            try:
                await self._client.add_entries(...)
                if self._callbacks:
                    self._callbacks.on_success(records)
                return
            except IngestionError as e:
                if self._retry.is_retryable(e) and attempt < self._retry.max_retries:
                    delay = self._retry.backoff(attempt)
                    if self._callbacks:
                        self._callbacks.on_retry(records, e, attempt, delay)
                    await asyncio.sleep(delay)
                else:
                    if self._callbacks:
                        self._callbacks.on_error(records, e, attempt)

    async def close(self):
        """Flush all pending background deliveries, then close session in Beacon.
        
        Must be called on shutdown to ensure no data is lost.
        """
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks, timeout=30.0)  # internal default
        if self._session_created:
            await self._client.close_session(...)
        self._closed = True
```

### 3.3 Summary (After)

| Class | Blocks caller? | Retries | Callbacks | Production-ready? |
|-------|----------------|---------|-----------|-------------------|
| `SessionManager` | Yes (intentional, for tests/scripts) | ✅ | Optional | ✅ |
| `AsyncSessionManager` | **No** (fire-and-forget) | ✅ | Optional | ✅ |

---

## 4. Which Adapter Uses Which

**All production adapters use `AsyncSessionManager`** — the non-blocking one.

| Adapter | Manager | Why |
|---------|---------|-----|
| **Claude** | `AsyncSessionManager` | Stop hook is `async def` inside SDK event loop. Non-blocking = next turn starts faster. |
| **LangGraph** | `AsyncSessionManager` | Callbacks are `async def` inside agent event loop. Blocking would stall the agent. |
| **ADK** | `AsyncSessionManager` | Same as LangGraph — async callbacks inside event loop. |
| **Test scripts** | `SessionManager` | Blocking is fine for tests. Simple to assert results immediately. |

```python
# ALL adapters (production) — same pattern:
manager = AsyncSessionManager(
    async_client,
    agent_id="...",
    owner_id="...",
)

await manager.ingest(records)  # returns immediately — agent not stalled
# ... agent continues processing ...
await manager.close()          # flush pending deliveries, then close session
```

---

## 5. Execution Flow (How Non-Blocking Works)

### 5.1 Step-by-Step Flow

```
TIME →

Agent / Adapter (caller)                 Background Task (same event loop)
──────────────────────────────           ─────────────────────────────────

1. await manager.ingest(records)
   │
   ├─ asyncio.create_task(_deliver)────▶ task scheduled on event loop
   ├─ adds task to _pending_tasks set
   │
   └─ RETURNS IMMEDIATELY ✓
       (adapter/agent continues)
                                         2. _deliver(records) starts:
                                            ├─ await _ensure_lifecycle()
                                            │   └─ create conv + session (if first time)
                                            │
                                            ├─ entries = _to_beacon_entries(records)
                                            │
                                            ├─ await client.add_entries(entries)
                                            │   └─ HTTP POST to Beacon (200ms)
                                            │
                                            ├─ SUCCESS? → callbacks.on_success() (if provided)
                                            │             → task removes itself from set
                                            │
                                            └─ FAILURE? → retry loop (see §5.2)

... later on shutdown ...

3. await manager.close()
   │
   ├─ asyncio.wait(pending_tasks)──────▶ waits for any in-flight deliveries
   ├─ await client.close_session()
   └─ self._closed = True
```

### 5.2 Retry Flow (Inside Background Task)

```
_deliver(records):
│
├─ Attempt 1: await client.add_entries()
│   └─ HTTP 503 (Beacon overloaded)
│   └─ is_retryable(503) → True
│   └─ callbacks.on_retry() (if provided)
│   └─ await asyncio.sleep(1.0)       ← does NOT block the adapter
│
├─ Attempt 2: await client.add_entries()
│   └─ HTTP 503 again
│   └─ callbacks.on_retry() (if provided)
│   └─ await asyncio.sleep(2.1)
│
├─ Attempt 3: await client.add_entries()
│   └─ HTTP 202 ✓
│   └─ callbacks.on_success() (if provided)
│   └─ return (task done)
│
│  ── OR if all retries fail ──
│
├─ Attempt 4: await client.add_entries()
│   └─ HTTP 503 again
│   └─ attempt(3) >= max_retries(3)
│   └─ callbacks.on_error() (if provided)
│   └─ return (task done, data lost)
```

### 5.3 Why `asyncio.create_task` Makes It Non-Blocking

```python
# BEFORE — blocks the caller:
async def ingest(self, records):
    await self._client.add_entries(...)
    # ↑ The adapter is SUSPENDED here. It cannot process anything
    #   until Beacon responds. This is like calling someone on the phone
    #   and waiting on hold — you can't do anything else.

# AFTER — fire-and-forget:
async def ingest(self, records):
    asyncio.create_task(self._deliver(records))
    # ↑ This is like dropping a letter in a mailbox. You walk away immediately.
    #   The postal service (event loop) delivers it in the background.
    #   You're free to do other things right now.
```

`asyncio.create_task()` schedules a coroutine to run independently on the event loop. The caller doesn't wait for it. The event loop runs it whenever it gets CPU time (immediately after the adapter yields).

---

## 6. Callbacks (Optional Observability Hooks)

Callbacks are **optional**. If not provided (`callbacks=None`), the SDK still works — retries still happen, delivery still works. Callbacks are only for customers who want custom observability (metrics, alerting, dead-letter queues).

### 6.1 Protocol Definition

```python
from typing import Protocol

class IngestionCallbacks(Protocol):
    """Optional observability hooks invoked during delivery.
    
    Rules for implementations:
    - Never raise exceptions (would crash the background task)
    - Never block (would delay retries/delivery)
    - Keep execution time minimal (< 10ms)
    """

    def on_success(self, records: list[Record], status_code: int) -> None: ...
    def on_error(self, records: list[Record], error: Exception, attempts: int) -> None: ...
    def on_retry(self, records: list[Record], error: Exception, attempt: int, delay: float) -> None: ...
```

### 6.2 What Each Callback Does

#### `on_success(records, status_code)`

**When called:** After Beacon accepts the entries (HTTP 2xx response).

**Typical uses:**
- Increment a success counter metric
- Track delivery latency
- Log delivery confirmation

#### `on_error(records, error, attempts)`

**When called:** After ALL retries are exhausted. The data will NOT be delivered.

**Typical uses:**
- Increment a failure counter metric
- Alert on-call if failure rate is elevated
- Persist failed records to a dead-letter file for manual recovery
- Trigger a custom recovery mechanism (re-queue, write to disk, etc.)

#### `on_retry(records, error, attempt, delay)`

**When called:** After a transient failure, BEFORE sleeping for the backoff delay.

**Typical uses:**
- Increment a retry counter metric
- Track which error codes trigger retries (identifies Beacon instability)

### 6.3 Usage

```python
# Without callbacks (default) — retries happen automatically, no custom hooks:
manager = AsyncSessionManager(client, agent_id=..., owner_id=...)

# With callbacks — retries still automatic + custom metrics/alerting:
class MyCallbacks:
    def on_success(self, records, status_code):
        datadog.increment("beacon.success")
    
    def on_error(self, records, error, attempts):
        datadog.increment("beacon.failure")
        pagerduty.alert(f"Beacon ingestion failed: {error}")
    
    def on_retry(self, records, error, attempt, delay):
        datadog.increment("beacon.retry")

manager = AsyncSessionManager(client, agent_id=..., owner_id=...,
                              callbacks=MyCallbacks())
```

### 6.4 Safety: Callbacks Never Crash Delivery

Callbacks are wrapped internally to ensure a buggy callback never breaks the delivery pipeline:

```python
def _safe_callback(self, fn, *args):
    """Call callback safely — never let callback errors crash delivery."""
    try:
        fn(*args)
    except Exception:
        pass  # callback failure is silently ignored
```

---

## 7. Shared Components (Core SDK)

### 7.1 RetryPolicy (Internal — Not Customer-Facing)

The retry policy is **internal to the SDK**. Customers do not configure it. We set the right values.

```python
class _RetryPolicy:
    """Internal retry configuration. Not exported. Not configurable by customers."""

    max_retries = 3
    backoff_base = 1.0        # seconds
    backoff_max = 30.0        # cap
    retryable_codes = frozenset({429, 500, 502, 503, 504})

    def is_retryable(self, error: IngestionError) -> bool:
        """Determine if an error is transient and worth retrying."""
        # Network-level errors are always retryable
        if isinstance(error.__cause__, (ConnectionError, TimeoutError, OSError)):
            return True
        # HTTP status code classification
        return error.status_code in self.retryable_codes

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter.
        
        attempt=0 → ~1.0s
        attempt=1 → ~2.0s  
        attempt=2 → ~4.0s
        (capped at backoff_max=30s)
        """
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        jitter = random.uniform(0, delay * 0.1)  # ±10% jitter
        return delay + jitter
```

**Fixed defaults:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| `max_retries` | 3 | Covers transient blips without excessive delay |
| `backoff_base` | 1.0s | Industry standard starting point |
| `backoff_max` | 30.0s | Never wait longer than 30s between retries |
| `retryable_codes` | 429, 500-504 | All transient server errors + rate limiting |

---

## 8. Error Classification

| Error Type | Retryable? | Action |
|------------|-----------|--------|
| HTTP 429 (rate limit) | ✅ | Backoff + retry |
| HTTP 500-504 (server error) | ✅ | Backoff + retry |
| Network timeout | ✅ | Backoff + retry |
| Connection refused | ✅ | Backoff + retry |
| HTTP 400 (bad request) | ❌ | `on_error` (if provided) → fail immediately |
| HTTP 401 (unauthorized) | ❌* | `on_error` (if provided) → fail (or token refresh if configured) |
| HTTP 403 (forbidden) | ❌ | `on_error` (if provided) → fail immediately |
| HTTP 404 (not found) | ❌ | `on_error` (if provided) → fail immediately |

---

## 9. Ordering & Lifecycle Guarantees

| Guarantee | How |
|-----------|-----|
| Lifecycle ordering (create conv → session → entries) | `_ensure_lifecycle()` uses an `asyncio.Lock` — only the first background task creates the conversation/session; subsequent tasks wait on the lock then proceed directly to add_entries |
| Graceful shutdown | `close()` awaits all pending tasks before sending close_session |
| At-most-once delivery | Default behavior. Beacon uses `entry.id` for idempotency (future enhancement) |

---

## 10. File Changes

```
src/agentic_context/
├── _retry.py        # NEW — _RetryPolicy (internal, not exported)
├── callbacks.py     # NEW — IngestionCallbacks protocol (optional, exported)
├── session.py       # MODIFIED — add internal retry + optional callbacks
├── session_async.py # MODIFIED — non-blocking ingest via create_task, internal retry
├── client.py        # unchanged
├── client_async.py  # unchanged
└── __init__.py      # export IngestionCallbacks only
```

---

## 11. Implementation Order

| Phase | What | Effort |
|-------|------|--------|
| 1 | `_retry.py` — Internal RetryPolicy | ~30 min |
| 2 | `callbacks.py` — IngestionCallbacks protocol (optional) | ~30 min |
| 3 | `session.py` — Add internal retry + optional callbacks to SessionManager | ~2 hr |
| 4 | `session_async.py` — Non-blocking ingest + internal retry in AsyncSessionManager | ~3 hr |
| 5 | Unit + integration tests | ~3 hr |

---

## 12. Backward Compatibility

| Change | Breaking? | Migration |
|--------|-----------|-----------|
| Internal retry logic added to SessionManager | No — transparent to callers | None |
| New `callbacks` param on SessionManager | No — defaults to `None` (no callbacks) | None |
| AsyncSessionManager.ingest becomes non-blocking | **Yes** — delivery now happens in background | Adapters must call `await close()` on shutdown to flush |

The async non-blocking change only affects internal adapters (not customer-facing APIs). All adapters already call `close()` on shutdown.

---

## 13. Open Questions

1. **Dead letter queue?** After max retries, should failed records be persisted to disk for later inspection? Or just drop?

2. **Concurrency limit?** If 20 rapid messages fire 20 background tasks, should we cap with a semaphore?
