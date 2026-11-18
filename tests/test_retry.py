# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unit tests for the internal retry policy (`agentic_context._retry`).

`_retry` is internal (not part of the semver-committed surface), but its
classification and backoff shape are the reliability contract every
SessionManager depends on — so the policy itself gets direct coverage here.
"""
from __future__ import annotations

import httpx
import pytest

from agentic_context import _retry
from agentic_context.client import IngestionError


# ── is_retryable: HTTP status classification ──────────────────────────────────


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_retryable_status_codes(code: int) -> None:
    """The full retryable set. 408 (Request Timeout) is
    included — a client-side timeout the server acknowledges is transient.
    """
    assert _retry.is_retryable(IngestionError(code, "transient")) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422])
def test_non_retryable_status_codes(code: int) -> None:
    """4xx client errors (other than 408/429) are permanent — retrying a 401
    or a 400 just burns attempts on a request that can never succeed.
    """
    assert _retry.is_retryable(IngestionError(code, "permanent")) is False


def test_httpx_status_error_uses_response_code() -> None:
    req = httpx.Request("POST", "https://beacon.example.com/x")
    retryable = httpx.HTTPStatusError(
        "boom", request=req, response=httpx.Response(503, request=req)
    )
    permanent = httpx.HTTPStatusError(
        "boom", request=req, response=httpx.Response(400, request=req)
    )
    assert _retry.is_retryable(retryable) is True
    assert _retry.is_retryable(permanent) is False


# ── is_retryable: transport exceptions ────────────────────────────────────────


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.PoolTimeout("pool exhausted"),
    ],
)
def test_httpx_transport_errors_are_retryable(exc: Exception) -> None:
    """Connect/read/write/pool timeouts and connect errors are all transient.

    Regression: ConnectTimeout was previously missing from the tuple, so a
    connection that timed out mid-handshake was treated as permanent.
    """
    assert _retry.is_retryable(exc) is True


@pytest.mark.parametrize("exc", [ConnectionError(), TimeoutError(), OSError()])
def test_stdlib_transport_errors_are_retryable(exc: Exception) -> None:
    assert _retry.is_retryable(exc) is True


@pytest.mark.parametrize("exc", [ValueError("bad"), KeyError("missing"), RuntimeError("x")])
def test_unrelated_exceptions_are_not_retryable(exc: Exception) -> None:
    assert _retry.is_retryable(exc) is False


# ── backoff: full-jitter bounds ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "attempt,cap",
    [(0, 1.0), (1, 2.0), (2, 4.0), (3, 8.0), (4, 16.0)],
)
def test_backoff_within_full_jitter_window(attempt: int, cap: float) -> None:
    """Full jitter samples uniformly in [0, min(base*2**attempt, 30)].

    Sample many times so a bad implementation (e.g. fixed delay, or a lower
    bound above 0) is caught statistically, not just on one lucky draw.
    """
    samples = [_retry.backoff(attempt) for _ in range(2000)]
    assert all(0.0 <= s <= cap for s in samples), "sample escaped [0, cap]"
    # With 2000 draws the min should be well under, and the max well above,
    # the midpoint — confirms it's a distribution, not a constant.
    assert min(samples) < cap * 0.5
    assert max(samples) > cap * 0.5


def test_backoff_capped_at_30s() -> None:
    """Deep attempts saturate at the 30s cap — 2**10 * 1.0 would be 1024s."""
    samples = [_retry.backoff(10) for _ in range(1000)]
    assert all(0.0 <= s <= _retry._BACKOFF_MAX for s in samples)
    assert max(samples) > _retry._BACKOFF_MAX * 0.5  # still spans the window
