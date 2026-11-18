# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Optional observability callbacks for ingestion.

Customers can implement the IngestionCallbacks protocol to receive
notifications about delivery success, failure, and retries.

Callbacks are optional. If not provided, delivery still runs; the only
default-path signal on failure is ``logger.error`` (e.g. the pending-task
saturation drop path in ``AsyncSessionManager.ingest``). Wire callbacks
to observe success/retry/error events programmatically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import Record


@runtime_checkable
class IngestionCallbacks(Protocol):
    """Optional observability hooks invoked during delivery.

    Rules for implementations:
    - Never raise exceptions (would disrupt delivery)
    - Never block (would delay retries)
    - Keep execution time minimal (< 10ms)
    """

    def on_success(self, records: list[Record], status_code: int) -> None:
        """Called after Beacon accepts entries (HTTP 2xx)."""
        ...

    def on_error(self, records: list[Record], error: Exception, attempts: int) -> None:
        """Called when a batch will not be delivered. Data is lost.

        Two firing paths:

        * ``attempts >= 1`` — invoked from the background delivery task after
          retries were exhausted (or a non-retryable status like 400/401/403
          returned on the first try).
        * ``attempts == 0`` — invoked synchronously from
          :meth:`AsyncSessionManager.ingest` when the pending-task queue is
          saturated (Beacon degraded, in-flight cap reached). The batch was
          never scheduled. Because this fires on the ingest() hot path, the
          "never block" rule matters here: a slow callback stalls the caller.
        """
        ...

    def on_retry(self, records: list[Record], error: Exception, attempt: int, delay: float) -> None:
        """Called before retry sleep. Attempt is 0-indexed."""
        ...
