# API Limits & How the SDK Handles Them

**Version:** 0.7.0  
**Last Updated:** 2026-06-17  
**Reference:** [Streaming Ingest API Limits](https://developer.salesforce.com/docs/data/data-cloud-int/references/data-cloud-ingestionapi-ref/c360-a-api-get-started.html#streaming-ingest-api-limits)

---

## Beacon (Context Service) Streaming Ingest API Limits

| Limit | Value | SDK Handling |
|-------|-------|--------------|
| Max payload size per request | 200 KB | ✅ Automatic payload chunking |
| Rate limit (all endpoints) | 250 requests/second | ✅ Retry with exponential backoff on 429 |
| Rate limit response | HTTP 429 Too Many Requests | ✅ Retried automatically (up to 3 attempts) |
| Server errors | HTTP 500, 502, 503, 504 | ✅ Retried automatically |
| Request timeout | HTTP 408 | ✅ Retried automatically |
| Data availability latency | ~30 seconds after ingestion | ⚠️ Not SDK-controlled (server-side processing) |

---

## Payload Chunking (200KB Limit)

### The Problem

The Beacon Streaming Ingest API rejects requests with payloads exceeding 200KB. A typical agent turn is 1-2KB, but turns with many tool calls or large tool outputs (database query results, generated code blocks) can exceed this.

### How the SDK Handles It

The SDK automatically splits large entry lists into multiple sequential requests, each under 180KB (safe margin below the 200KB hard limit):

```
Turn produces entries totaling ~300KB:

→ Chunk 1 (entries 1-3, ~170KB):  POST /entries → 202 ✅
→ Chunk 2 (entries 4-5, ~130KB):  POST /entries → 202 ✅
→ on_success callback fires
```

**Behavior details:**

- **Greedy accumulation** — entries are added to the current chunk until the next entry would exceed 180KB, then a new chunk starts
- **Order preserved** — entry chronological order is maintained across chunks
- **Sequential delivery** — chunks are sent one after another (not in parallel) to maintain ordering and avoid overwhelming Beacon
- **Fail-fast** — if a chunk fails after retries, delivery stops and `on_error` fires. Subsequent chunks are not sent
- **No-op for typical turns** — a normal agent turn is 1-2KB, so chunking produces a single chunk (zero overhead)
- **Logged** — when chunking activates, an info-level log message is emitted:
  ```
  Payload exceeded 180KB limit; split 12 entries into 3 chunks
  ```

### When Does Chunking Activate?

Rarely. These scenarios could trigger it:

| Scenario | Typical size | Chunks? |
|----------|-------------|---------|
| Normal chat turn (2-3 messages) | 1-2 KB | No (1 chunk) |
| Turn with 1 tool call + short result | 3-5 KB | No |
| Turn with 5 tool calls + medium results | 10-30 KB | No |
| Turn with tool returning large DB result set | 50-200+ KB | Maybe |
| Turn with code generation (long output) | 20-100 KB | Unlikely |
| Batch ingestion (end_session mode, full transcript) | 50-500+ KB | Likely |

### What Customers Need to Do

**Nothing.** Chunking is fully automatic and internal to the SDK. No configuration, no opt-in, no code changes required.

---

## Rate Limiting (429 Handling)

### The Problem

Beacon enforces 250 requests/second across all endpoints for a given org. Under heavy concurrent load (many agents, many conversations), you may hit this limit.

### How the SDK Handles It

The SDK retries HTTP 429 responses with **exponential backoff and full jitter**:

```
Attempt 0: POST → 429 Too Many Requests
  → backoff: random(0, 1.0s)
Attempt 1: POST → 429
  → backoff: random(0, 2.0s)
Attempt 2: POST → 429
  → backoff: random(0, 4.0s)
Attempt 3: POST → 429
  → on_error callback fires, delivery abandoned
```

**Configuration (fixed, not customer-configurable):**

| Setting | Value |
|---------|-------|
| Max retries | 3 |
| Backoff base | 1.0 seconds |
| Backoff max cap | 30.0 seconds |
| Jitter | Full (uniform random from 0 to computed cap) |
| Retryable HTTP codes | 408, 429, 500, 502, 503, 504 |

**Why full jitter?**

Full jitter (AWS-recommended) trades a slightly lower expected delay for much better thundering-herd behavior. When many adapter instances retry simultaneously (e.g., after a brief Beacon outage), full jitter spreads them out across the backoff window instead of all hitting at the same instant.

### What Customers Need to Do

For most single-agent deployments: **nothing**. A single conversation generates ~1-3 requests per turn, far below 250/s.

For high-scale deployments (hundreds of concurrent conversations):

- Monitor `on_retry` callbacks to detect throttling
- Consider staggering conversation starts across a few seconds
- The SDK handles recovery automatically — but sustained 429s mean your org is genuinely over capacity

---

## Transient Server Errors (5xx Handling)

### The Problem

Beacon, like any distributed service, can temporarily return 500/502/503/504 during deployments, load spikes, or upstream issues.

### How the SDK Handles It

Same retry logic as 429 — exponential backoff with full jitter, up to 3 retries.

**Non-retryable errors (fail immediately):**

| Code | Meaning | Why not retried |
|------|---------|-----------------|
| 400 | Bad Request | Payload is malformed — retrying won't fix it |
| 401 | Unauthorized | Bad or expired credentials — with `auth=` the SDK refreshes the token and retries |
| 403 | Forbidden | Permission issue — retrying won't help |
| 404 | Not Found | Wrong endpoint or conversation doesn't exist |
| 409 | Conflict | Already exists — SDK treats as success for lifecycle calls |
| 413 | Payload Too Large | Should not occur with chunking enabled |
| 422 | Unprocessable | Invalid data — retrying won't fix it |

---

## Network Errors

Connection failures (`ConnectionError`) and timeouts (`TimeoutError`) are always retried with the same exponential backoff policy. These cover:

- DNS resolution failures
- TCP connection refused
- TLS handshake failures
- Read timeouts (Beacon didn't respond within 10s)

---

## Observability: Callbacks

The SDK fires callbacks during retry/error scenarios so you can monitor limit-related issues:

```python
from agentic_context import IngestionCallbacks

callbacks = IngestionCallbacks(
    on_success=lambda records, status: print(f"✓ {len(records)} records delivered"),
    on_retry=lambda records, error, attempt, delay: print(
        f"⚠ Retry {attempt}: {error} — waiting {delay:.1f}s"
    ),
    on_error=lambda records, error, attempts: print(
        f"✗ Failed after {attempts} attempts: {error}"
    ),
)
```

**Note:** Callbacks are configured on the `SessionManager` level (internal to adapters). For adapter-level usage, error conditions surface as:

- **Async mode:** `on_error` fires in background; `close()` completes even if some deliveries failed
- **Sync mode:** Exceptions propagate to the caller after retry exhaustion

---

## Summary: What the SDK Handles vs. What Customers Handle

| Concern | SDK Handles | Customer Handles |
|---------|:-----------:|:----------------:|
| Payload too large (200KB) | ✅ Auto-chunks | — |
| Rate limiting (429) | ✅ Auto-retries | — |
| Server errors (5xx) | ✅ Auto-retries | — |
| Network failures | ✅ Auto-retries | — |
| Token expiry (401) | ✅ Auto-refreshes with `auth=` | — |
| Data availability delay (~30s) | — | ✅ Design for eventual consistency |
| Org-level capacity | — | ✅ Monitor & scale |

---

## Enable Debug Logging

To see chunking and retry behavior in action:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("agentic_context").setLevel(logging.DEBUG)
```

This will show:
- When payload chunking activates and how many chunks are created
- Each retry attempt with backoff delay
- Final success or failure for each delivery
