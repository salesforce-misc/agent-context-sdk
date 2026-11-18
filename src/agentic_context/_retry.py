# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Internal retry policy — not exported, not configurable by customers.

Fixed defaults that work for production Beacon ingestion.
"""
from __future__ import annotations

import random
import time

import httpx

from .client import IngestionError

# Fixed retry configuration
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds
_BACKOFF_MAX = 30.0  # cap
_RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_retryable(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying."""
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(error, IngestionError):
        return error.status_code in _RETRYABLE_CODES
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in _RETRYABLE_CODES
    if isinstance(
        error,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ),
    ):
        return True
    return False


def backoff(attempt: int) -> float:
    """Exponential backoff with full jitter.

    Full jitter (AWS "Exponential Backoff and Jitter"): sample uniformly in
    ``[0, cap]`` where ``cap = min(base * 2**attempt, _BACKOFF_MAX)``. This
    spreads retries across the whole window to avoid thundering-herd
    synchronization, and the result never exceeds the 30s cap.

    attempt=0 → uniform(0, 1.0s)
    attempt=1 → uniform(0, 2.0s)
    attempt=2 → uniform(0, 4.0s)
    (each window capped at 30s)
    """
    cap = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
    return random.uniform(0, cap)
