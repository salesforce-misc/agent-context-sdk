# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Salesforce Beacon (CDP) authentication — two-leg JWT token exchange.

Handles the full flow from a connected-app private key to a live CDP Bearer
token, with automatic background refresh before the token expires.

Two auth paths, one exchange:

* :class:`ContextAuth` — the **production** path. Pass the four connected-app
  credentials directly. Never touches any external secret server.
* :class:`ApiKeyAuth` — the **demo / onboarding** path. Pass a single
  ``api_key``; the connected-app bundle is downloaded from a secret server on
  first use, an inner :class:`ContextAuth` is built from it, and every call
  delegates to it. Interchangeable with ``ContextAuth`` at the call site.

Requires the ``cryptography`` package, which is a core dependency of
``agent-context-sdk`` (installed automatically by ``pip install agent-context-sdk``).

Usage (production)::

    auth = ContextAuth(
        private_key=Path("server.key").read_text(),
        consumer_key="3MVG9...",
        subject="user@example.com",
        login_url="login.salesforce.com",
    )

    # Sync
    token, base_url = auth.get_token()

    # Async
    token, base_url = await auth.aget_token()

    # Wire directly into a client — token is refreshed automatically
    client = AsyncIngestionClient(base_url=base_url, api_key=token)
    auth.attach(client)   # keeps client.api_key fresh on every refresh

Usage (demo / api-key)::

    # secret_server_url must be supplied explicitly or via
    # AGENTIC_CONTEXT_SECRET_SERVER_URL — no host is baked into the SDK.
    auth = ApiKeyAuth(api_key, secret_server_url="https://secrets.example.com")
    token, base_url = auth.get_token()   # downloads the bundle, then exchanges
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

import httpx

from . import _pkce, _retry

if TYPE_CHECKING:
    from .stores import RefreshTokenStore


logger = logging.getLogger(__name__)

_REFRESH_BUFFER = 300  # seconds before expiry to refresh (5 min)

# Env var that supplies the secret-server base URL for the api-key path when no
# ``secret_server_url`` is passed explicitly. There is deliberately NO default
# host: the SDK fails closed if neither is set, so no demo-/environment-specific
# secret server ever ships baked into the package (a compromised or stale baked
# host would silently receive api keys). The caller (the demo app) supplies it.
_SECRET_SERVER_URL_ENV = "AGENTIC_CONTEXT_SECRET_SERVER_URL"

# Only Salesforce-owned hosts may receive the signed JWT assertion (which
# carries the consumer key + subject). Guards the api-key/bundle path, where
# ``loginUrl`` arrives from an external secret-server response — a compromised
# server could otherwise point leg-1 of the token exchange at an attacker host
# (SSRF / assertion exfil). Anchored + no "_"/leading-dot slack, so
# URL-structure bypasses (``@``, ``/``, ``:``, ``evilsalesforce.com``,
# ``x.salesforce.com.evil.com``) are all rejected. Multi-label subdomains like
# ``acme.my.salesforce.com`` match fine. The direct-constructor
# (production) ``ContextAuth`` path is developer-supplied and intentionally NOT
# constrained by this.
_ALLOWED_LOGIN_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*\.salesforce\.com$")


def _validate_instance_url(url: str, *, field: str) -> str:
    """Validate an ``instance_url`` returned by an upstream OAuth response.

    The two-leg token exchange consumes two ``instance_url`` values from
    upstream JSON responses (leg-1 core, leg-2 CDP) and then POSTs to them
    with a bearer token. A tampered response could redirect either POST at
    an attacker host (SSRF / token exfil) or downgrade the transport to
    plaintext ``http://``. This helper fails closed against both.

    Rules:

    * empty → reject
    * ``http://`` scheme → reject (transport downgrade)
    * bare hostname → normalize to ``https://`` before host extraction
    * host must match :data:`_ALLOWED_LOGIN_HOST` (``*.salesforce.com``)
    * userinfo, non-default port, path, query, fragment → reject

    Returns ``https://{hostname}`` reconstructed from the validated host, not
    the input string. Reconstruction — rather than sanitising in place —
    eliminates parser-differential bypasses where ``urlparse`` and ``httpx``
    could disagree on the effective host (e.g. backslash / userinfo tricks).

    Never echoes ``url`` in the error message — it may be attacker-crafted.
    """
    # isinstance guard before any string operation: upstream JSON can hand us
    # a list/dict/number under this key. Without the type check, [] and {}
    # fail through 'not url' with a misleading "empty" message, and a number
    # or other non-empty non-str would raise AttributeError on .startswith()
    # — fail-closed but not the documented ValueError contract.
    if not isinstance(url, str) or not url:
        raise ValueError(f"OAuth response returned empty {field}")
    if url.startswith("http://"):
        raise ValueError(
            f"OAuth response returned insecure http:// {field} "
            "(transport downgrade — expected https://)"
        )
    if not url.startswith("https://"):
        url = "https://" + url
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not _ALLOWED_LOGIN_HOST.fullmatch(host):
        raise ValueError(
            f"OAuth response returned {field} host outside the allowlist "
            "(expected a *.salesforce.com host)"
        )
    if (
        parsed.username
        or parsed.password
        or parsed.port
        or parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"OAuth response returned {field} with unexpected URL components "
            "(expected scheme + host only)"
        )
    return f"https://{host}"


def _validate_login_host(login_url: str) -> None:
    """Fail closed before POSTing OAuth credentials to ``login_url``.

    ``login_url`` is a developer-supplied bare Salesforce login host (no scheme),
    e.g. ``"login.salesforce.com"``. Both token legs POST to
    ``https://{login_url}/services/oauth2/token`` — the JWT path sends a signed
    assertion, the user-login path sends the refresh token and client secret. A
    tampered or misconfigured value would exfiltrate those credentials to an
    attacker host, so we allowlist the host the same way #56 allowlists the
    ``instance_url`` returned by the response.

    The allowlist regex admits only ``*.salesforce.com`` hosts and — because its
    character class is limited to ``[A-Za-z0-9.-]`` — also rejects any userinfo
    (``@``), path (``/``), port (``:``), query (``?``), fragment (``#``), or
    smuggled scheme (``://``). Mirrors the login-host check in
    :func:`_parse_secret_response`.
    """
    if not _ALLOWED_LOGIN_HOST.fullmatch(login_url):
        raise ValueError(
            "login_url is outside the allowlist (expected a bare "
            "*.salesforce.com host with no scheme, path, port, or userinfo)"
        )


def _key_hint(api_key: str) -> str:
    """Non-reversible fingerprint of an api key for audit logs (never the raw key)."""
    if not api_key:
        return "none"
    return "sha256:" + hashlib.sha256(api_key.encode()).hexdigest()[:12]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign_jwt(claims: dict[str, Any], private_key_pem: str) -> str:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:
        raise ImportError(
            "ContextAuth requires the 'cryptography' package (a core dependency "
            "of agent-context-sdk). Reinstall with: pip install agent-context-sdk"
        ) from exc

    header = {"alg": "RS256", "typ": "JWT"}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(claims, separators=(",", ":")).encode())
    ).encode("ascii")
    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    sig = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return signing_input.decode("ascii") + "." + _b64url(sig)


class ContextAuth:
    """Two-leg JWT → CDP Bearer token exchange with automatic refresh.

    Leg 1: RSA-signed JWT → Salesforce core access token
    Leg 2: core access token → CDP Bearer token

    The token is cached and refreshed automatically 5 minutes before expiry.
    Call ``get_token()`` (sync) or ``aget_token()`` (async) to get a valid
    ``(token, base_url)`` pair at any time.

    This is the production path — pass the four connected-app credentials
    directly. For the demo / api-key onboarding path see :class:`ApiKeyAuth`,
    which resolves these credentials from a secret server and then delegates to
    an instance of this class.

    Args:
        private_key: PEM-encoded RSA private key for the connected app.
        consumer_key: OAuth consumer key (client_id) of the connected app.
        subject: Salesforce username authorised for JWT bearer flow.
        login_url: Salesforce login host, e.g. ``"login.salesforce.com"``
            or ``"acme.my.salesforce.com"``. Do not include scheme.
        token_ttl: Requested JWT lifetime in seconds (default 3600 / 1 hr).
            Salesforce caps this at the org's session timeout setting.
        aid: Registered agent id for the a360 exchange (a360 rejects an unknown
            aid with ``access_denied`` — see :func:`_a360_body`).
        xid: Participant / owner external id for the a360 exchange. One
            instance caches one token, so set it only for a single-owner
            deployment; multi-user services need one instance per owner.
    """

    default_owner_type: ClassVar[str] = "individual"

    def __init__(
        self,
        *,
        private_key: str,
        consumer_key: str,
        subject: str,
        login_url: str,
        token_ttl: int = 3600,
        aid: str | None = None,
        xid: str | None = None,
    ) -> None:
        # The a360 CDP exchange (leg 2) optionally takes two extra params the
        # SDK previously never sent: `aid` (agent id) and `xid` (participant/owner
        # external id). Governance / conversation-participant access is scoped by
        # this pair, and a360 REJECTS an UNKNOWN aid with `access_denied`. When
        # both are None (the default) behaviour is identical to before, so this
        # change is backwards-compatible. `aid` is safe to bake in here (one
        # fixed value per agent service); `xid` is per-owner — a single ContextAuth
        # caches ONE token, so only set `xid` for a single-owner deployment.
        # Multi-user services must use per-owner auth instances instead.
        self._aid = aid
        self._xid = xid
        self._private_key = private_key
        self._consumer_key = consumer_key
        self._subject = subject
        self._login_url = login_url.rstrip("/")
        self._token_ttl = token_ttl

        # Single-tuple state so readers see a consistent (token, base_url, ...)
        # pair atomically. Individual attribute writes could tear (new token +
        # old base_url), which would 401 on Beacon; a tuple assignment is one
        # STORE_ATTR under CPython's GIL, so a concurrent snapshot read gets
        # either the whole old refresh or the whole new one, never a mix.
        # Layout: (token, base_url, core_token, core_instance_url, expiry).
        # Expiry=0.0 means "never fetched — force refresh on first call".
        self._snapshot: tuple[str, str, str, str, float] = ("", "", "", "", 0.0)
        self._lock = Lock()
        self._async_lock: asyncio.Lock | None = None

        # Clients to keep in sync when the token refreshes (any object with api_key attr)
        self._attached: list[Any] = []

    # Read-only views onto the snapshot tuple. Keep the old attribute names
    # available so external inspection (adapters, tests, ops) doesn't break.
    # Writes go through self._snapshot, never these — a setter would create the
    # exact torn-read hazard we introduced the tuple to prevent.
    @property
    def _token(self) -> str:
        return self._snapshot[0]

    @property
    def _base_url(self) -> str:
        return self._snapshot[1]

    @property
    def _core_token(self) -> str:
        return self._snapshot[2]

    @property
    def _core_instance_url(self) -> str:
        return self._snapshot[3]

    @property
    def _expiry(self) -> float:
        return self._snapshot[4]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_token(self) -> tuple[str, str]:
        """Return a valid ``(bearer_token, base_url)`` pair, refreshing if needed.

        Thread-safe. Raises on HTTP or crypto errors.

        Three-state routing so a refresh in flight never stalls a whole fleet
        of readers:

        1. ``now < expiry - _REFRESH_BUFFER`` (fresh): return snapshot, no lock.
        2. ``now < expiry`` (buffer window): try-lock elects one refresher; if
           the try-lock fails, another thread already won the election — return
           the still-valid snapshot rather than queue behind its HTTPS exchange.
           A ``.locked()`` hint would be racy (N threads all see False and all
           queue on ``with self._lock``); ``acquire(blocking=False)`` gives
           exactly one winner per contention wave.
        3. ``now >= expiry`` (expired): must block. Take ``self._lock`` —
           become the refresher, or wait for the one in flight.
        """
        snap = self._snapshot
        now = time.time()
        if now < snap[4] - _REFRESH_BUFFER:
            return snap[0], snap[1]
        if now < snap[4]:
            if not self._lock.acquire(blocking=False):
                return snap[0], snap[1]
            try:
                if time.time() >= self._snapshot[4] - _REFRESH_BUFFER:
                    self._fetch_sync()
                snap = self._snapshot
            finally:
                self._lock.release()
            return snap[0], snap[1]
        with self._lock:
            # Re-check under the lock — another thread may have already
            # refreshed while we were queued for it. Standard
            # double-checked-locking idiom.
            if time.time() >= self._snapshot[4] - _REFRESH_BUFFER:
                self._fetch_sync()
            snap = self._snapshot
        return snap[0], snap[1]

    async def aget_token(self) -> tuple[str, str]:
        """Async version of ``get_token()``. Coroutine-safe.

        Same three-state routing as the sync path. Buffer-window callers see
        a ``locked()`` gate — the coroutine-only scheduling model means the
        race window is only cooperative task-switch boundaries, not thread
        preemption, so a hint is good enough to elect one refresher.
        """
        snap = self._snapshot
        now = time.time()
        if now < snap[4] - _REFRESH_BUFFER:
            return snap[0], snap[1]
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        if now < snap[4] and self._async_lock.locked():
            return snap[0], snap[1]
        async with self._async_lock:
            if time.time() >= self._snapshot[4] - _REFRESH_BUFFER:
                await self._fetch_async()
            snap = self._snapshot
        return snap[0], snap[1]

    def get_core_creds(self) -> tuple[str, str]:
        """Return the ``(core_access_token, instance_url)`` pair from step 1
        of the JWT flow, refreshing the full exchange if needed.

        Beacon's Context Service uses the CDP-exchanged token from
        :meth:`get_token`. The Data Cloud query API
        (``/services/data/vXX.X/ssot/queryv2``) instead wants the original
        Salesforce core access token, and it lives on the org's *instance*
        URL — not Beacon's base_url. Use this method to hydrate live DMO
        data from a skill's ``dataContext`` block.

        Thread-safe. Same refresh semantics as :meth:`get_token`: fast path
        is lock-free, buffer-window callers try-lock elect one refresher
        (losers return the still-valid snapshot without queueing).
        """
        snap = self._snapshot
        now = time.time()
        if now < snap[4] - _REFRESH_BUFFER:
            return snap[2], snap[3]
        if now < snap[4]:
            if not self._lock.acquire(blocking=False):
                return snap[2], snap[3]
            try:
                if time.time() >= self._snapshot[4] - _REFRESH_BUFFER:
                    self._fetch_sync()
                snap = self._snapshot
            finally:
                self._lock.release()
            return snap[2], snap[3]
        with self._lock:
            if time.time() >= self._snapshot[4] - _REFRESH_BUFFER:
                self._fetch_sync()
            snap = self._snapshot
        return snap[2], snap[3]

    async def aget_core_creds(self) -> tuple[str, str]:
        """Async version of :meth:`get_core_creds`. Coroutine-safe."""
        snap = self._snapshot
        now = time.time()
        if now < snap[4] - _REFRESH_BUFFER:
            return snap[2], snap[3]
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        if now < snap[4] and self._async_lock.locked():
            return snap[2], snap[3]
        async with self._async_lock:
            if time.time() >= self._snapshot[4] - _REFRESH_BUFFER:
                await self._fetch_async()
            snap = self._snapshot
        return snap[2], snap[3]

    def invalidate(self) -> None:
        """Immediately invalidate the cached token.

        Call this when an HTTP client receives a 401 response — the next
        get_token() / aget_token() call will re-fetch unconditionally rather
        than waiting for the timer-based refresh window. Thread-safe.
        """
        with self._lock:
            t, b, ct, ci, _ = self._snapshot
            # Replace the whole tuple so the fast-path snapshot read still
            # returns a coherent state — just with expiry=0 forcing refresh.
            self._snapshot = (t, b, ct, ci, 0.0)

    def attach(self, client: Any) -> None:
        """Register a client whose ``api_key`` is kept in sync with refreshes.

        Accepts any object with an ``api_key`` attribute — both
        ``AsyncIngestionClient`` and ``AsyncRetrievalClient`` work.
        After attaching, you never need to update the client's token manually —
        ``ContextAuth`` patches it after every refresh.

        Guarded by ``self._lock`` so ``attach`` cannot interleave with the
        ``_propagate`` iteration inside a token refresh. Without this lock the
        ``if client not in list`` / ``list.append`` pair is a TOCTOU race
        (duplicate registrations) and a client attached mid-refresh can miss
        the just-propagated token — CPython's list iterator has undefined
        behavior when the list grows during iteration.

        Usage::

            token, base_url = await auth.aget_token()
            client = AsyncIngestionClient(base_url=base_url, api_key=token)
            auth.attach(client)
        """
        with self._lock:
            if client not in self._attached:
                self._attached.append(client)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fetch_sync(self) -> None:
        with httpx.Client(timeout=30) as client:
            for attempt in range(_retry._MAX_RETRIES + 1):
                try:
                    token, base_url, expiry, core_token, core_instance_url = _exchange(
                        client, self._login_url, self._consumer_key,
                        self._subject, self._private_key, self._token_ttl,
                        aid=self._aid, xid=self._xid,
                    )
                    break
                except Exception as e:
                    if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                        delay = _retry.backoff(attempt)
                        logger.warning(
                            "[ContextAuth] token exchange failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, _retry._MAX_RETRIES + 1, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        raise
        # Publish atomically — one STORE_ATTR under the GIL so any concurrent
        # snapshot reader (get_token fast path) sees either the whole previous
        # refresh or the whole new one, never a mixed state.
        self._snapshot = (token, base_url, core_token, core_instance_url, expiry)
        self._propagate(token)
        logger.info("[ContextAuth] token refreshed (sync), expiry=%.0f", expiry)

    async def _fetch_async(self) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(_retry._MAX_RETRIES + 1):
                try:
                    token, base_url, expiry, core_token, core_instance_url = await _exchange_async(
                        client, self._login_url, self._consumer_key,
                        self._subject, self._private_key, self._token_ttl,
                        aid=self._aid, xid=self._xid,
                    )
                    break
                except Exception as e:
                    if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                        delay = _retry.backoff(attempt)
                        logger.warning(
                            "[ContextAuth] token exchange failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, _retry._MAX_RETRIES + 1, e, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
        # Same atomic-publish rationale as _fetch_sync. Serialize with
        # ``attach()`` on the sync lock: ``_fetch_async`` runs under
        # ``self._async_lock`` (a coroutine-only lock), so without this the
        # ``_propagate`` iteration can race against an ``attach()`` mutation
        # from another thread.
        with self._lock:
            self._snapshot = (token, base_url, core_token, core_instance_url, expiry)
            self._propagate(token)
        logger.info("[ContextAuth] token refreshed (async), expiry=%.0f", expiry)

    def _propagate(self, token: str) -> None:
        for client in self._attached:
            client.api_key = token


class ApiKeyAuth:
    """Single-``api_key`` auth path — resolves credentials from a secret server.

    This is the "demo" onboarding path: instead of pasting the four raw
    connected-app credentials, the customer supplies **only** an ``api_key``
    minted in the secret-server portal. On first use this class calls
    ``GET /api/secret`` on the secret server, pulls the credential bundle
    (private key, consumer key, subject, login URL), builds an inner
    :class:`ContextAuth`, and thereafter delegates every token / attach /
    invalidate call straight to it. From the client's point of view an
    ``ApiKeyAuth`` is interchangeable with a ``ContextAuth`` — same
    ``get_token`` / ``aget_token`` / ``get_core_creds`` / ``attach`` /
    ``invalidate`` surface, so it drops into ``AsyncIngestionClient(auth=...)``
    and ``adapter.attach(...)`` unchanged.

    Production deployments should keep passing raw credentials to
    :class:`ContextAuth` directly — this path adds a network hop and a
    dependency on the secret server that real deployments don't want. The two
    paths never converge on the network: a ``ContextAuth`` built directly never
    contacts the secret server.

    Credential fetch is lazy and happens exactly once (guarded by a lock on
    the sync path and an asyncio lock on the async path). The very first call
    may be sync (``get_token``) or async (``aget_token``); whichever wins
    builds the inner ``ContextAuth`` and the other reuses it.

    Fails closed: there is **no** default secret-server host. ``secret_server_url``
    must be supplied explicitly or via ``AGENTIC_CONTEXT_SECRET_SERVER_URL``; if
    neither is set the first token fetch raises rather than sending the api key
    to a baked-in host. This keeps any demo-/environment-specific host out of
    the shipped SDK.

    Args:
        api_key: The portal-minted key. Sent as the ``x-api-key`` header
            (never in the URL, so it isn't captured in access logs).
        secret_server_url: Secret-server base URL. Falls back to the
            ``AGENTIC_CONTEXT_SECRET_SERVER_URL`` env var. Required (no default);
            must be ``https://``.
        token_ttl: Forwarded to the inner :class:`ContextAuth`.
    """

    default_owner_type: ClassVar[str] = "individual"

    def __init__(
        self,
        api_key: str,
        *,
        secret_server_url: str | None = None,
        token_ttl: int = 3600,
    ) -> None:
        if not api_key:
            raise ValueError("ApiKeyAuth requires a non-empty api_key.")
        self._api_key = api_key
        # No hardcoded default host — resolve from the arg, then the env var,
        # else leave empty and fail closed on first fetch (see
        # _require_https_secret_url).
        self._secret_server_url = (
            secret_server_url
            or os.environ.get(_SECRET_SERVER_URL_ENV)
            or ""
        ).rstrip("/")
        self._token_ttl = token_ttl

        self._inner: ContextAuth | None = None
        # agent_id the server suggested for this key (empty when unset). Read
        # by adapters as a default agent_id; exposed via the property below.
        self._agent_id: str = ""
        self._lock = Lock()
        self._async_lock: asyncio.Lock | None = None
        # Clients attached before the inner ContextAuth exists — replayed onto
        # it once it's built so no attach() is lost to lazy resolution.
        self._pending_attached: list[Any] = []

    # ── Public API (mirrors ContextAuth) ────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        """The ``agentId`` from the secret bundle, or ``""`` if none / unresolved.

        Adapters may use this as a default ``agent_id`` when the caller didn't
        pass one. It is only populated after the first token fetch resolves the
        bundle, so read it lazily (after a ``get_token``), not at construction.
        """
        return self._agent_id

    def get_token(self) -> tuple[str, str]:
        """Return a valid ``(bearer_token, base_url)`` pair. Thread-safe."""
        return self._ensure_inner_sync().get_token()

    async def aget_token(self) -> tuple[str, str]:
        """Async version of :meth:`get_token`. Coroutine-safe."""
        inner = await self._ensure_inner_async()
        return await inner.aget_token()

    def get_core_creds(self) -> tuple[str, str]:
        """Return the ``(core_access_token, instance_url)`` pair. Thread-safe."""
        return self._ensure_inner_sync().get_core_creds()

    async def aget_core_creds(self) -> tuple[str, str]:
        """Async version of :meth:`get_core_creds`. Coroutine-safe."""
        inner = await self._ensure_inner_async()
        return await inner.aget_core_creds()

    def invalidate(self) -> None:
        """Invalidate the cached token so the next fetch re-exchanges.

        No-op before the inner ``ContextAuth`` exists — there is no token to
        drop yet, and the credential bundle is not re-fetched here (the secret
        server is not the source of 401s).
        """
        if self._inner is not None:
            self._inner.invalidate()

    def attach(self, client: Any) -> None:
        """Register a client whose ``api_key`` follows token refreshes.

        Works before or after the inner ``ContextAuth`` is built: pre-resolution
        attaches are buffered and replayed onto the inner auth the moment it is
        created, so callers can ``attach`` immediately after construction.

        Guarded by ``self._lock`` so it can't interleave with
        ``_build_inner``'s replay-then-clear-then-publish. Without the lock an
        ``attach`` that read ``self._inner is None`` could be preempted by
        ``_build_inner`` (which clears ``_pending_attached`` and publishes the
        inner), then resume and append to the now-abandoned buffer — the client
        would be orphaned from token refresh forever. ``attach`` never calls
        ``_build_inner``, so taking the lock here can't deadlock.
        """
        with self._lock:
            if self._inner is not None:
                self._inner.attach(client)
            elif client not in self._pending_attached:
                self._pending_attached.append(client)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _ensure_inner_sync(self) -> ContextAuth:
        if self._inner is not None:
            return self._inner
        with httpx.Client(timeout=30) as client:
            secret, agent_id = _fetch_secret(
                client, self._secret_server_url, self._api_key
            )
        return self._build_inner(secret, agent_id)

    async def _ensure_inner_async(self) -> ContextAuth:
        if self._inner is not None:
            return self._inner
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        # The asyncio lock dedupes concurrent async first-calls (so the bundle
        # is fetched once on the common all-async path). The fetch runs here,
        # never under self._lock — a threading.Lock held across I/O would stall
        # the event loop. The cross-flavor race (async vs a bare sync call) is
        # made safe by _build_inner's idempotent, self._lock-guarded publish,
        # not by this lock.
        async with self._async_lock:
            if self._inner is not None:
                return self._inner
            async with httpx.AsyncClient(timeout=30) as client:
                secret, agent_id = await _fetch_secret_async(
                    client, self._secret_server_url, self._api_key
                )
            return self._build_inner(secret, agent_id)

    def _build_inner(self, secret: dict[str, Any], agent_id: str) -> ContextAuth:
        """Publish (once) the inner ContextAuth from a resolved secret bundle.

        This is the single publish point, guarded by ``self._lock`` and made
        idempotent so a concurrent sync + async first-call can't both install
        an inner auth. The two ``_ensure_inner_*`` paths use *different* locks
        (``threading.Lock`` vs ``asyncio.Lock``) and so don't mutually exclude
        at the fetch layer — without an idempotent, lock-held publish the
        second builder would create a second inner, replay + ``clear()`` the
        shared ``_pending_attached`` list, and leave the surviving inner with
        no attached clients (token refresh would then stop propagating).
        Whoever wins ``self._lock`` builds, replays the buffered attaches, and
        publishes; a loser (already-built, or the other flavor that fetched
        concurrently) observes ``self._inner`` set, discards its own — harmless,
        idempotent — fetch, and returns the winner. The fetch is intentionally
        left outside this lock so the async path never blocks the event loop on
        it; the only cost is a rare redundant secret-server GET when both
        flavors cold-start at once, and only the winner emits the audit log.
        """
        with self._lock:
            if self._inner is not None:
                return self._inner
            inner = ContextAuth(
                private_key=secret["privateKey"],
                consumer_key=secret["consumerKey"],
                subject=secret["subject"],
                login_url=secret["loginUrl"],
                token_ttl=self._token_ttl,
            )
            self._agent_id = agent_id
            # Replay any attaches buffered before the inner auth existed.
            for client in self._pending_attached:
                inner.attach(client)
            self._pending_attached.clear()
            # Publish last: attach() reads self._inner without the lock, so the
            # buffered clients must be replayed before the inner is visible.
            self._inner = inner
            # AU-12 structured audit record for the credential-download event:
            # actor (key fingerprint, never the raw key), target (secret host),
            # outcome. The resolved agent_id is safe to log (not a secret); the
            # bundle's credential material and the raw api_key are never
            # emitted. Inside the lock so exactly one winner logs per build.
            logger.info(
                "[ApiKeyAuth] credential bundle loaded from secret server",
                extra={
                    "actor": _key_hint(self._api_key),
                    "target": self._secret_server_url,
                    "outcome": "success",
                    "agent_id": agent_id or "<none>",
                },
            )
            return inner


# ── Secret-server bundle helpers (module-level so they're easy to unit-test) ──


def _parse_secret_response(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate a ``GET /api/secret`` body and return ``(secret, agent_id)``.

    Fails closed: a missing/non-dict ``secret`` object, an incomplete credential
    bundle, or a login host outside the Salesforce allowlist each raises a clean
    :class:`ValueError` — never leaking the (possibly attacker-crafted) response
    body. ``agentId`` is optional and returned as ``""`` when unset.
    """
    secret = payload.get("secret") if isinstance(payload, dict) else None
    if not isinstance(secret, dict):
        raise ValueError(
            "secret server returned a malformed credential bundle "
            "(expected a JSON object with a 'secret' field)"
        )
    # The server labels the login/aud host "loginUrl" (falls back to "aud").
    login_url = str(secret.get("loginUrl") or secret.get("aud") or "").rstrip("/")
    missing = [f for f in ("privateKey", "consumerKey", "subject") if not secret.get(f)]
    if missing or not login_url:
        raise ValueError(
            "secret server returned an incomplete credential bundle "
            "(missing privateKey/consumerKey/subject/loginUrl)"
        )
    # SSRF guard (SI-10): the JWT assertion (consumer key + subject) is POSTed
    # to this host. It came from an external server, so allowlist it to
    # Salesforce domains before committing it. Do NOT echo the value — it may
    # be attacker-crafted.
    # fullmatch (not match): `$` in `match` also accepts a single trailing
    # newline, so "login.salesforce.com\n" would slip through. fullmatch
    # requires the whole string to match, rejecting stray-newline hosts.
    if not _ALLOWED_LOGIN_HOST.fullmatch(login_url):
        raise ValueError(
            "secret server returned a login host outside the allowlist "
            "(expected a *.salesforce.com host)"
        )
    # Normalize so the inner ContextAuth consumes the resolved (possibly
    # aud-derived) host regardless of which field carried it.
    secret = {**secret, "loginUrl": login_url}
    agent_id = str(secret.get("agentId") or "")
    return secret, agent_id


def _require_https_secret_url(secret_server_url: str) -> None:
    """Fail closed before sending the api key: require an explicit https URL.

    Two guards, both raising a clean :class:`ValueError`:

    * empty URL → no ``secret_server_url`` was passed and
      ``AGENTIC_CONTEXT_SECRET_SERVER_URL`` is unset. There is no baked-in
      default host, so we refuse rather than guess where to send the key.
    * non-https URL → refuse to send the api key over an unencrypted connection.
    """
    if not secret_server_url:
        raise ValueError(
            "ApiKeyAuth requires an explicit secret_server_url (or the "
            "AGENTIC_CONTEXT_SECRET_SERVER_URL env var). No default secret "
            "server is baked into the SDK — the demo/deployment must supply "
            "the host so no environment-specific server ships in the package."
        )
    if not secret_server_url.startswith("https://"):
        raise ValueError(
            "secret_server_url must be https:// — refusing to send the api key "
            "over an unencrypted connection."
        )


def _log_bundle_failure(
    api_key: str, url: str, exc: httpx.HTTPStatusError
) -> None:
    """AU-2 audit record for a failed credential download, before re-raising.

    Emits the key *fingerprint* (never the raw key), the target host, and the
    HTTP status so a failed credential-acquisition attempt leaves an audit
    trail.
    """
    logger.warning(
        "[ApiKeyAuth] credential bundle download failed",
        extra={
            "actor": _key_hint(api_key),
            "target": url,
            "outcome": "failure",
            "http_status": exc.response.status_code,
        },
    )


def _log_bundle_rejected(api_key: str, url: str) -> None:
    """AU-2 audit record for an application-level bundle rejection.

    Fires when a bundle downloads successfully (HTTP 200) but fails validation
    in :func:`_parse_secret_response` — most importantly the SI-10 SSRF
    allowlist rejection of a non-``*.salesforce.com`` login host, but also a
    malformed/incomplete bundle. ``_log_bundle_failure`` only covers HTTP-level
    errors, so without this a blocked SSRF/exfiltration attempt would leave no
    audit trail. Logs the key *fingerprint* and the secret-server host only —
    never the attacker-supplied ``loginUrl`` (which may be crafted).
    """
    logger.warning(
        "[ApiKeyAuth] credential bundle rejected",
        extra={
            "actor": _key_hint(api_key),
            "target": url,
            "outcome": "rejected",
            "reason": "invalid_bundle",
        },
    )


def _handle_secret_response(
    r: httpx.Response, api_key: str, url: str, is_final: bool
) -> tuple[dict[str, Any], str]:
    """Turn one ``/api/secret`` response into ``(secret, agent_id)`` or raise.

    Shared by the sync and async fetchers (``r.raise_for_status`` and
    ``r.json`` are synchronous on both client flavors), so the two audit-log
    behaviors stay identical:

    * HTTP error → emit the AU-2 *failure* record only on the terminal attempt
      (or a non-retryable status), so a retryable 5xx/429 that recovers doesn't
      leave a trail of ``failure`` records that false-positive a SOC alert;
    * application-level rejection (SSRF allowlist / malformed bundle, a
      non-retryable ``ValueError`` from :func:`_parse_secret_response`) → emit
      the AU-2 *rejected* record before it propagates.
    """
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if is_final or not _retry.is_retryable(exc):
            _log_bundle_failure(api_key, url, exc)
        raise
    try:
        return _parse_secret_response(r.json())
    except ValueError:
        _log_bundle_rejected(api_key, url)
        raise


def _fetch_secret(
    client: httpx.Client, secret_server_url: str, api_key: str
) -> tuple[dict[str, Any], str]:
    """Fetch + validate the credential bundle for ``api_key`` (sync).

    Enforces an explicit ``https://`` URL before sending the key, retries
    transient failures with the shared backoff policy, and emits an AU-2 audit
    record on an HTTP error status. The key rides the ``x-api-key`` header,
    never the URL.
    """
    _require_https_secret_url(secret_server_url)
    url = f"{secret_server_url}/api/secret"
    for attempt in range(_retry._MAX_RETRIES + 1):
        try:
            r = client.get(url, headers={"x-api-key": api_key})
            return _handle_secret_response(
                r, api_key, url, attempt >= _retry._MAX_RETRIES
            )
        except Exception as e:
            if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                delay = _retry.backoff(attempt)
                logger.warning(
                    "[ApiKeyAuth] secret fetch failed (attempt %d/%d): %s — "
                    "retrying in %.1fs",
                    attempt + 1, _retry._MAX_RETRIES + 1, e, delay,
                )
                time.sleep(delay)
            else:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


async def _fetch_secret_async(
    client: httpx.AsyncClient, secret_server_url: str, api_key: str
) -> tuple[dict[str, Any], str]:
    """Async counterpart of :func:`_fetch_secret`."""
    _require_https_secret_url(secret_server_url)
    url = f"{secret_server_url}/api/secret"
    for attempt in range(_retry._MAX_RETRIES + 1):
        try:
            r = await client.get(url, headers={"x-api-key": api_key})
            return _handle_secret_response(
                r, api_key, url, attempt >= _retry._MAX_RETRIES
            )
        except Exception as e:
            if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                delay = _retry.backoff(attempt)
                logger.warning(
                    "[ApiKeyAuth] secret fetch failed (attempt %d/%d): %s — "
                    "retrying in %.1fs",
                    attempt + 1, _retry._MAX_RETRIES + 1, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


# ── Exchange helpers (module-level so they're easy to unit-test) ──────────────

def _build_assertion(
    login_url: str,
    consumer_key: str,
    subject: str,
    private_key: str,
    token_ttl: int,
) -> str:
    claims = {
        "sub": subject,
        "iss": consumer_key,
        "aud": f"https://{login_url}",
        "exp": int(time.time()) + token_ttl,
    }
    return _sign_jwt(claims, private_key)


def _a360_body(core_token: str, aid: str | None, xid: str | None) -> dict[str, str]:
    """Leg-2 (a360 CDP exchange) form body.

    The SDK previously sent only subject_token/type. a360 optionally accepts
    `aid` (agent id) and `xid` (participant external id) and REJECTS an unknown
    aid with `access_denied`. Appended only when set, so the None/None default
    reproduces the original request byte-for-byte.
    """
    body = {
        "grant_type": "urn:salesforce:grant-type:external:cdp",
        "subject_token": core_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
    }
    if aid:
        body["aid"] = aid
    if xid:
        body["xid"] = xid
    return body


# a360 leg-2 hardening. The a360 leg-2 exchange can intermittently return HTTP
# 200 with a body that has NO ``access_token`` — either a transient backend
# glitch, or a persistent refusal (unprovisioned user / unknown aid) surfaced as
# a 200-with-error rather than a 4xx. The SDK previously read
# ``cdp["access_token"]`` blindly and crashed with an opaque ``KeyError`` with no
# diagnosis. We now (a) retry a couple of times to self-heal the transient case,
# and (b) on exhaustion raise a clear error that surfaces a360's OWN non-secret
# diagnostic fields, so a persistent refusal is legible in the logs instead of a
# mystery KeyError.
_A360_EXCHANGE_RETRIES = 2
_A360_EXCHANGE_BACKOFF_S = 0.8

# Non-secret fields a360 puts on an error/refusal body — whitelisted so a token
# or other sensitive value is NEVER echoed into a log line or exception message.
_A360_DIAG_FIELDS = ("error", "error_description", "message", "errorCode", "error_uri")


def _a360_diag(cdp: Any) -> str:
    """A short, NON-SECRET description of an a360 response that lacks a token."""
    if isinstance(cdp, dict):
        parts = [f"{k}={cdp[k]!r}" for k in _A360_DIAG_FIELDS if cdp.get(k)]
        if parts:
            return "; ".join(parts)
        keys = ", ".join(sorted(cdp.keys())) or "<empty>"
        return f"no access_token; response keys: {keys}"
    return f"non-object a360 response ({type(cdp).__name__})"


def _a360_result(cdp: dict[str, Any], token_ttl: int) -> tuple[str, str, float]:
    """Map a token-bearing a360 body to ``(cdp_token, base_url, expiry)``.

    The CDP ``instance_url`` comes from the (upstream) a360 response and becomes
    the base URL for every subsequent Beacon call, so it is host-allowlist
    validated (SSRF / transport-downgrade guard) — not merely https-normalized.
    """
    base_url = _validate_instance_url(cdp.get("instance_url", ""), field="CDP instance_url")
    expiry = time.time() + int(cdp.get("expires_in", token_ttl))
    return cdp["access_token"], base_url, expiry


def _a360_exchange(
    client: httpx.Client,
    core_instance_url: str,
    core_token: str,
    aid: str | None,
    xid: str | None,
    token_ttl: int,
) -> tuple[str, str, float]:
    """Leg-2 a360 CDP exchange (sync) with a bounded retry for a transient
    200-without-``access_token``. Returns ``(cdp_token, base_url, expiry)``."""
    last: Any = None
    for attempt in range(_A360_EXCHANGE_RETRIES + 1):
        r = client.post(
            f"{core_instance_url}/services/a360/token",
            data=_a360_body(core_token, aid, xid),
        )
        r.raise_for_status()
        cdp = r.json()
        if isinstance(cdp, dict) and cdp.get("access_token"):
            return _a360_result(cdp, token_ttl)
        last = cdp
        logger.warning(
            "[auth] a360 exchange returned no access_token (attempt %d/%d): %s",
            attempt + 1,
            _A360_EXCHANGE_RETRIES + 1,
            _a360_diag(cdp),
        )
        if attempt < _A360_EXCHANGE_RETRIES:
            time.sleep(_A360_EXCHANGE_BACKOFF_S)
    raise ValueError(
        f"a360 CDP token exchange returned no access_token after "
        f"{_A360_EXCHANGE_RETRIES + 1} attempts ({_a360_diag(last)})"
    )


async def _a360_exchange_async(
    client: httpx.AsyncClient,
    core_instance_url: str,
    core_token: str,
    aid: str | None,
    xid: str | None,
    token_ttl: int,
) -> tuple[str, str, float]:
    """Async counterpart of :func:`_a360_exchange`."""
    last: Any = None
    for attempt in range(_A360_EXCHANGE_RETRIES + 1):
        r = await client.post(
            f"{core_instance_url}/services/a360/token",
            data=_a360_body(core_token, aid, xid),
        )
        r.raise_for_status()
        cdp = r.json()
        if isinstance(cdp, dict) and cdp.get("access_token"):
            return _a360_result(cdp, token_ttl)
        last = cdp
        logger.warning(
            "[auth] a360 exchange returned no access_token (attempt %d/%d): %s",
            attempt + 1,
            _A360_EXCHANGE_RETRIES + 1,
            _a360_diag(cdp),
        )
        if attempt < _A360_EXCHANGE_RETRIES:
            await asyncio.sleep(_A360_EXCHANGE_BACKOFF_S)
    raise ValueError(
        f"a360 CDP token exchange returned no access_token after "
        f"{_A360_EXCHANGE_RETRIES + 1} attempts ({_a360_diag(last)})"
    )


def _exchange(
    client: httpx.Client,
    login_url: str,
    consumer_key: str,
    subject: str,
    private_key: str,
    token_ttl: int,
    aid: str | None = None,
    xid: str | None = None,
) -> tuple[str, str, float, str, str]:
    """Returns ``(cdp_token, cdp_base_url, expiry, core_token, core_instance_url)``.

    ``core_token`` + ``core_instance_url`` are captured from leg 1 so callers
    can also use them for endpoints that don't take the CDP-exchanged token
    (e.g. ``/services/data/vXX.X/ssot/queryv2`` for Data Cloud queries).
    """
    _validate_login_host(login_url)
    assertion = _build_assertion(login_url, consumer_key, subject, private_key, token_ttl)

    r = client.post(
        f"https://{login_url}/services/oauth2/token",
        data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
    )
    r.raise_for_status()
    core = r.json()
    core_token = core["access_token"]
    core_instance_url = _validate_instance_url(
        core.get("instance_url", ""), field="core instance_url"
    )

    cdp_token, base_url, expiry = _a360_exchange(
        client, core_instance_url, core_token, aid, xid, token_ttl
    )
    return cdp_token, base_url, expiry, core_token, core_instance_url


async def _exchange_async(
    client: httpx.AsyncClient,
    login_url: str,
    consumer_key: str,
    subject: str,
    private_key: str,
    token_ttl: int,
    aid: str | None = None,
    xid: str | None = None,
) -> tuple[str, str, float, str, str]:
    """Async counterpart of :func:`_exchange`."""
    _validate_login_host(login_url)
    assertion = _build_assertion(login_url, consumer_key, subject, private_key, token_ttl)

    r = await client.post(
        f"https://{login_url}/services/oauth2/token",
        data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
    )
    r.raise_for_status()
    core = r.json()
    core_token = core["access_token"]
    core_instance_url = _validate_instance_url(
        core.get("instance_url", ""), field="core instance_url"
    )

    cdp_token, base_url, expiry = await _a360_exchange_async(
        client, core_instance_url, core_token, aid, xid, token_ttl
    )
    return cdp_token, base_url, expiry, core_token, core_instance_url


# ── Authorization-Code / user-login exchange ("Log in with Salesforce") ──
#
# The leg-1 grant here is the logged-in user's OAuth grant (refresh_token) rather
# than a JWT-bearer assertion, so the resulting CDP token acts AS THE USER
# (self-context) instead of a service account. Leg-2 (the a360 CDP exchange) is
# the same `_a360_body` helper as the JWT path, but never with an xid: the core
# token already IS the signed-in employee, so there is no end user to name.


def _user_id_from_id_url(id_url: str) -> str:
    """Extract the Salesforce ``005…`` user id from an OAuth ``id`` URL.

    The token response's ``id`` looks like
    ``https://login.salesforce.com/id/00D…/005…`` — the trailing path segment is
    the user id. Returns ``""`` when the field is absent/malformed.
    """
    return id_url.rsplit("/", 1)[-1] if id_url else ""


def _org_id_from_id_url(id_url: str) -> str:
    """Extract the ``00D…`` org id (second-to-last path segment) from an ``id`` URL."""
    parts = id_url.rstrip("/").rsplit("/", 2) if id_url else []
    return parts[-2] if len(parts) == 3 else ""


class ReauthRequired(Exception):
    """The user's Salesforce grant is gone — they must sign in again.

    Raised by :class:`UserContextAuth` when Salesforce rejects the refresh token
    (``invalid_grant``: expired, revoked, user deactivated, app access removed)
    and by :meth:`UserContextAuth.redeem_code` when the authorization code is
    rejected (expired, already used, verifier mismatch). Retrying cannot help, so
    it is never retried. A backend maps it to "send the browser back through
    login" (e.g. HTTP 401 + ``login_url``) and deletes the stored refresh token.

    ``error`` / ``description`` are Salesforce's non-secret OAuth error fields.
    """

    def __init__(self, error: str, description: str = "") -> None:
        self.error = error
        self.description = description
        super().__init__(
            f"Salesforce sign-in required ({error}{': ' + description if description else ''})"
        )


def _raise_for_grant(r: httpx.Response) -> None:
    """``raise_for_status`` for leg-1 grants, mapping a dead grant to :class:`ReauthRequired`.

    ``invalid_grant`` (400) is the OAuth signal for a refresh token / code that
    is no longer valid. A bare 401 with no OAuth error body is treated the same.
    Other errors — notably ``invalid_client`` (wrong consumer key/secret, an ECA
    misconfiguration a re-login cannot fix) — keep raising ``HTTPStatusError``.
    """
    if r.status_code in (400, 401):
        try:
            body = r.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        if error == "invalid_grant" or (r.status_code == 401 and not error):
            description = body.get("error_description", "") if isinstance(body, dict) else ""
            raise ReauthRequired(error or "unauthorized", str(description or ""))
    r.raise_for_status()


def _validate_redirect_uri(redirect_uri: str) -> None:
    """Require ``https://`` for a backend callback; allow ``http://`` only on loopback."""
    parsed = urlparse(redirect_uri)
    loopback = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (
        not parsed.hostname
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or not (parsed.scheme == "https" or (parsed.scheme == "http" and loopback))
    ):
        raise ValueError(
            "redirect_uri must be an absolute https:// URL (http:// only for "
            "localhost) with no userinfo or fragment, matching a callback URL "
            "on the ECA"
        )


@dataclass(frozen=True)
class AuthorizationRequest:
    """Output of :meth:`UserContextAuth.authorization_request`.

    Redirect the browser to ``url``. Keep ``state`` and ``code_verifier``
    server-side for the callback (e.g. an encrypted, HttpOnly, short-lived
    cookie or a session-store row) — never in a URL or JS-readable storage.
    """

    url: str
    state: str
    code_verifier: str = field(repr=False)
    redirect_uri: str


@dataclass(frozen=True)
class UserLogin:
    """Output of :meth:`UserContextAuth.redeem_code` — one user's completed sign-in.

    Persist ``refresh_token`` (encrypted) keyed by ``(org_id, user_id)``, then
    build a :class:`UserContextAuth` from it per request / per cached user.
    """

    refresh_token: str = field(repr=False)
    user_id: str
    org_id: str
    instance_url: str


def _refresh_body(
    refresh_token: str, consumer_key: str, consumer_secret: str | None
) -> dict[str, str]:
    """Leg-1 refresh_token grant form body.

    ``client_secret`` is sent only for a confidential client. A public client
    (browser PKCE login, consumer key only) omits it.
    """
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": consumer_key,
    }
    if consumer_secret:
        body["client_secret"] = consumer_secret
    return body


def _rotate(core: dict[str, Any], current: str, on_rotate: Any) -> None:
    """Hand a rotated refresh token to ``on_rotate``.

    With refresh-token rotation the previous token is dead the moment the grant
    succeeds, so the new one must be captured before leg 2 can fail.
    """
    new = core.get("refresh_token")
    if new and new != current and on_rotate is not None:
        on_rotate(new)


def _exchange_refresh(
    client: httpx.Client,
    login_url: str,
    refresh_token: str,
    consumer_key: str,
    consumer_secret: str | None,
    token_ttl: int,
    aid: str | None = None,
    on_rotate: Any = None,
) -> tuple[str, str, float, str, str, str]:
    """Leg-1 refresh_token grant + leg-2 a360 exchange (sync).

    Returns ``(cdp_token, cdp_base_url, expiry, core_token, core_instance_url,
    user_id)``. ``user_id`` is the logged-in user's ``005…`` resolved from the
    core token response.

    No ``xid`` is sent: the core token already belongs to the signed-in
    employee, so the CDP token acts as that user. (``xid`` exists for the
    integration-user model, where one service identity stands in for many
    external end users — see :class:`ContextAuth`.)

    Raises :class:`ReauthRequired` when Salesforce rejects the refresh token.
    """
    _validate_login_host(login_url)
    r = client.post(
        f"https://{login_url}/services/oauth2/token",
        data=_refresh_body(refresh_token, consumer_key, consumer_secret),
    )
    _raise_for_grant(r)
    core = r.json()
    _rotate(core, refresh_token, on_rotate)
    core_token = core["access_token"]
    core_instance_url = _validate_instance_url(
        core.get("instance_url", ""), field="core instance_url"
    )
    user_id = _user_id_from_id_url(core.get("id", ""))

    cdp_token, base_url, expiry = _a360_exchange(
        client, core_instance_url, core_token, aid, None, token_ttl
    )
    return cdp_token, base_url, expiry, core_token, core_instance_url, user_id


async def _exchange_refresh_async(
    client: httpx.AsyncClient,
    login_url: str,
    refresh_token: str,
    consumer_key: str,
    consumer_secret: str | None,
    token_ttl: int,
    aid: str | None = None,
    on_rotate: Any = None,
) -> tuple[str, str, float, str, str, str]:
    """Async counterpart of :func:`_exchange_refresh`."""
    _validate_login_host(login_url)
    r = await client.post(
        f"https://{login_url}/services/oauth2/token",
        data=_refresh_body(refresh_token, consumer_key, consumer_secret),
    )
    _raise_for_grant(r)
    core = r.json()
    _rotate(core, refresh_token, on_rotate)
    core_token = core["access_token"]
    core_instance_url = _validate_instance_url(
        core.get("instance_url", ""), field="core instance_url"
    )
    user_id = _user_id_from_id_url(core.get("id", ""))

    cdp_token, base_url, expiry = await _a360_exchange_async(
        client, core_instance_url, core_token, aid, None, token_ttl
    )
    return cdp_token, base_url, expiry, core_token, core_instance_url, user_id


def _check_callback(login_url: str, code: str, state: str, expected_state: str) -> str:
    """Validate the host and the callback's ``state``/``code`` before redeeming."""
    login_url = login_url.rstrip("/")
    _validate_login_host(login_url)
    if not expected_state or not secrets.compare_digest(
        (state or "").encode(), expected_state.encode()
    ):
        raise ValueError("Salesforce login callback state mismatch; aborting")
    if not code:
        raise ValueError("Salesforce login callback carried no authorization code")
    return login_url


def _user_login(r: httpx.Response) -> UserLogin:
    """Parse an authorization_code grant response into a :class:`UserLogin`."""
    _raise_for_grant(r)
    token = _pkce._require_refresh_token(r.json())
    id_url = token.get("id", "")
    return UserLogin(
        refresh_token=token["refresh_token"],
        user_id=_user_id_from_id_url(id_url),
        org_id=_org_id_from_id_url(id_url),
        instance_url=_validate_instance_url(
            token.get("instance_url", ""), field="core instance_url"
        ),
    )


def _audit_login(
    method: str,
    outcome: str,
    *,
    source_ip: str | None,
    org_id: str = "",
    user_id: str = "",
    error: BaseException | None = None,
) -> None:
    """Audit record for a user sign-in result (success or denial).

    Never includes the code, verifier, state or tokens. ``reason`` is the
    exception class (plus the ``ReauthRequired`` error code), not its message.
    ``user_id`` is empty on denial: the user is unknown until Salesforce accepts
    the code. The timestamp comes from the log record.
    """
    reason = ""
    if error is not None:
        reason = type(error).__name__
        if isinstance(error, ReauthRequired):
            reason += f":{error.error}"
    fields = {
        "audit_event": "user_login", "method": method, "outcome": outcome,
        "source_ip": source_ip or "unknown", "org_id": org_id, "user_id": user_id,
        "reason": reason,
    }
    logger.log(
        logging.INFO if outcome == "success" else logging.WARNING,
        "audit: user_login method=%s outcome=%s source_ip=%s org_id=%s user_id=%s reason=%s",
        method, outcome, fields["source_ip"], org_id, user_id, reason,
        extra=fields,
    )


class UserContextAuth:
    """Authorization-Code (user-login) variant of :class:`ContextAuth`.

    Same two-leg CDP token exchange and the **identical public surface**
    (``get_token`` / ``aget_token`` / ``get_core_creds`` / ``aget_core_creds`` /
    ``invalidate`` / ``attach``), so it drops into ``GoogleAdkAdapter(auth=...)``
    and ``AsyncIngestionClient(auth=...)`` unchanged. The only difference from
    ``ContextAuth`` is leg 1:

    * ``ContextAuth`` — leg 1 is a JWT-bearer assertion signed with the connected
      app's private key; the CDP token acts as a **service account** and any
      ``owner_id`` may be looked up.
    * ``UserContextAuth`` — leg 1 is a **refresh_token grant** obtained from the
      user's interactive OAuth login (Authorization Code + PKCE); the CDP token
      acts **as that user** (self-context), so no ``xid`` is sent. The
      resolved ``005…`` is exposed via :attr:`user_id`.

    Because the cached token belongs to one user, one instance represents
    exactly one logged-in user — build a fresh ``UserContextAuth`` (and
    a fresh adapter) per login session rather than sharing one across users.

    Two ways to get the refresh token:

    * **Local / desktop** — :meth:`login` runs the whole browser flow on a
      loopback port (consumer key only).
    * **Backend-hosted callback** (web app) — :meth:`authorization_request` in
      your ``/login`` route, :meth:`redeem_code` (or :meth:`aredeem_code`) in
      your ``/callback`` route. Store the refresh token, then build one
      instance per user on demand::

          auth = UserContextAuth(
              consumer_key=..., consumer_secret=..., login_url=my_domain_host,
              aid="my_agent",
              load_refresh_token=lambda: store.get(org_id, user_id),
              on_refresh_token=lambda t: store.put(org_id, user_id, t),
          )

    When Salesforce rejects the refresh token, every token method raises
    :class:`ReauthRequired` — send the user back through login.

    Args:
        refresh_token: The OAuth refresh token from the user's code redemption.
            May be omitted when ``load_refresh_token`` is given.
        consumer_key: Connected-app / ECA client id (same app as ``ContextAuth``).
        consumer_secret: Connected-app / ECA client secret for a confidential
            client. Omit for a public client (consumer key only), e.g. one
            created by :meth:`login`.
        login_url: Salesforce login host, no scheme.
        aid: Registered agent id for the a360 exchange (a360 rejects an unknown
            aid with ``access_denied`` — see :func:`_a360_body`).
        token_ttl: Fallback lifetime when the a360 response omits ``expires_in``.
        on_refresh_token: Called with the new refresh token whenever Salesforce
            rotates it, so the caller can persist it. Runs before leg 2, and its
            write must be durable before it returns: the old token is already dead.
        load_refresh_token: Returns the latest stored refresh token (or
            ``None``). Called before every refresh grant, and once more if that
            grant fails with ``invalid_grant`` — so when several backend
            replicas share one user's token and another replica rotated it, this
            instance picks up the new value instead of forcing a re-login.
            Synchronous; it also runs on the async path.
    """

    default_owner_type: ClassVar[str] = "employee"

    def __init__(
        self,
        *,
        refresh_token: str | None = None,
        consumer_key: str,
        consumer_secret: str | None = None,
        login_url: str,
        aid: str | None = None,
        token_ttl: int = 3600,
        on_refresh_token: Callable[[str], None] | None = None,
        load_refresh_token: Callable[[], str | None] | None = None,
    ) -> None:
        if not refresh_token and load_refresh_token is None:
            raise ValueError(
                "UserContextAuth requires a non-empty refresh_token or a load_refresh_token."
            )
        self._refresh_token = refresh_token or ""
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._on_refresh_token = on_refresh_token
        self._load_refresh_token = load_refresh_token
        # Tokens Salesforce has already rotated away from. A lagging store may
        # still serve one; adopting it would replace the live token with a dead one.
        self._retired_refresh_tokens: deque[str] = deque(maxlen=16)
        self._on_reauth: Callable[[str], None] | None = None
        self._login_url = login_url.rstrip("/")
        self._aid = aid
        self._token_ttl = token_ttl

        self._token: str = ""
        self._base_url: str = ""
        self._core_token: str = ""
        self._core_instance_url: str = ""
        self._user_id: str = ""
        self._expiry: float = 0.0
        self._lock = Lock()
        self._async_lock: asyncio.Lock | None = None
        self._attached: list[Any] = []

    @classmethod
    def from_store(
        cls,
        store: RefreshTokenStore,
        *,
        org_id: str,
        user_id: str,
        consumer_key: str,
        login_url: str,
        consumer_secret: str | None = None,
        aid: str | None = None,
        token_ttl: int = 3600,
    ) -> UserContextAuth:
        """Build the auth for one user, with their refresh token kept in ``store``.

        Reads the token from the store before every refresh, writes a rotated
        one back (before the CDP exchange), and deletes it (only if still the rejected one) when Salesforce
        rejects it (just before :class:`ReauthRequired` propagates). Save the
        first token yourself after sign-in::

            store.save(login.org_id, login.user_id, login.refresh_token)
        """
        auth = cls(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            login_url=login_url,
            aid=aid,
            token_ttl=token_ttl,
            load_refresh_token=lambda: store.load(org_id, user_id),
            on_refresh_token=lambda token: store.save(org_id, user_id, token),
        )
        auth._on_reauth = lambda rejected: store.delete(org_id, user_id, if_token=rejected)
        return auth

    @classmethod
    def login(
        cls,
        *,
        consumer_key: str,
        login_url: str,
        aid: str | None = None,
        scope: str = _pkce.DEFAULT_SCOPE,
        port: int = _pkce.DEFAULT_PORT,
        timeout: float = _pkce.DEFAULT_TIMEOUT,
        token_ttl: int = 3600,
        on_refresh_token: Callable[[str], None] | None = None,
    ) -> UserContextAuth:
        """Sign the user in through their browser and return a ready auth.

        For **local tools** (CLI, notebook, desktop) that have no web server to
        receive the OAuth redirect: this method runs a short-lived loopback
        listener on ``127.0.0.1:port`` for that one callback, which is why it
        takes a ``port``. A backend that serves users should not use it — host
        your own callback and use :meth:`authorization_request` /
        :meth:`redeem_code`, where you choose the ``redirect_uri`` and the SDK
        opens no port.

        Public-client path: only the ECA ``consumer_key`` is needed — no
        consumer secret, no private key. Opens the Salesforce login page, the
        user authenticates with their own credentials, and the authorization
        code is redeemed with PKCE. The returned instance holds the user's
        refresh token and acts as that user.

        The ECA must register ``http://localhost:<port>/callback`` as a callback
        URL, enable the web-server flow with "Require secret" off, and grant
        every scope requested (``scope`` must be a subset). Salesforce may rotate
        the refresh token on each use; pass ``on_refresh_token`` to persist the
        new value, otherwise it lives only in this process.

        Blocks until the browser callback arrives (``timeout`` seconds).
        """
        login_url = login_url.rstrip("/")
        _validate_login_host(login_url)
        try:
            token = _pkce.pkce_login(
                login_url=login_url,
                consumer_key=consumer_key,
                scope=scope,
                port=port,
                timeout=timeout,
            )
        except Exception as e:
            # The callback arrives on the loopback listener, so the source is local.
            _audit_login("loopback", "denied", source_ip="127.0.0.1", error=e)
            raise
        id_url = token.get("id", "")
        _audit_login(
            "loopback", "success", source_ip="127.0.0.1",
            org_id=_org_id_from_id_url(id_url), user_id=_user_id_from_id_url(id_url),
        )
        if on_refresh_token is not None:
            on_refresh_token(token["refresh_token"])
        auth = cls(
            refresh_token=token["refresh_token"],
            consumer_key=consumer_key,
            login_url=login_url,
            aid=aid,
            token_ttl=token_ttl,
            on_refresh_token=on_refresh_token,
        )
        auth._user_id = _user_id_from_id_url(token.get("id", ""))
        return auth

    # ── Backend-hosted login (web server flow, callback on your server) ────────

    @staticmethod
    def authorization_request(
        *,
        consumer_key: str,
        login_url: str,
        redirect_uri: str,
        scope: str = _pkce.DEFAULT_SCOPE,
    ) -> AuthorizationRequest:
        """Start a sign-in from your backend's ``/login`` route.

        Returns the Salesforce authorize URL (redirect the browser to it with a
        302) plus a fresh ``state`` and PKCE ``code_verifier``. Keep both
        server-side until the callback and hand them to :meth:`redeem_code`.

        ``redirect_uri`` is your backend's callback, e.g.
        ``https://chat.example.com/oauth/callback`` — it must exactly match a
        callback URL registered on the ECA. ``login_url`` should be the org's
        My Domain host so users land on their org's login page (and SSO).
        """
        login_url = login_url.rstrip("/")
        _validate_login_host(login_url)
        _validate_redirect_uri(redirect_uri)
        state = _pkce.new_state()
        verifier = _pkce.new_verifier()
        return AuthorizationRequest(
            url=_pkce.authorize_url(
                login_url=login_url, consumer_key=consumer_key,
                redirect_uri=redirect_uri, scope=scope, state=state, verifier=verifier,
            ),
            state=state,
            code_verifier=verifier,
            redirect_uri=redirect_uri,
        )

    @staticmethod
    def redeem_code(
        *,
        code: str,
        state: str,
        expected_state: str,
        code_verifier: str,
        redirect_uri: str,
        consumer_key: str,
        consumer_secret: str | None = None,
        login_url: str,
        client_ip: str | None = None,
    ) -> UserLogin:
        """Finish a sign-in in your backend's ``/callback`` route.

        ``code`` and ``state`` come from the callback query string;
        ``expected_state``, ``code_verifier`` and ``redirect_uri`` from the
        :class:`AuthorizationRequest` saved at ``/login``. If the callback
        carries ``error`` instead of ``code``, the user declined or the ECA
        refused — don't call this; show the error and offer login again.

        Pass ``consumer_secret`` when the ECA has "Require secret for Web Server
        Flow" on (recommended for a backend). Raises ``ValueError`` on a state
        mismatch (possible CSRF) and :class:`ReauthRequired` when Salesforce
        rejects the code (expired, reused, verifier mismatch).

        Every call writes an ``audit: user_login`` log record (success or
        denial). The SDK never sees the browser's address, so pass ``client_ip``
        (the callback request's source) to have it recorded; otherwise it is
        logged as ``unknown``.
        """
        try:
            login_url = _check_callback(login_url, code, state, expected_state)
            with httpx.Client(timeout=30) as client:
                r = _pkce.redeem_code(
                    client, login_url=login_url, consumer_key=consumer_key,
                    consumer_secret=consumer_secret, redirect_uri=redirect_uri,
                    code=code, verifier=code_verifier,
                )
            login = _user_login(r)
        except Exception as e:
            _audit_login("backend", "denied", source_ip=client_ip, error=e)
            raise
        _audit_login(
            "backend", "success", source_ip=client_ip,
            org_id=login.org_id, user_id=login.user_id,
        )
        return login

    @staticmethod
    async def aredeem_code(
        *,
        code: str,
        state: str,
        expected_state: str,
        code_verifier: str,
        redirect_uri: str,
        consumer_key: str,
        consumer_secret: str | None = None,
        login_url: str,
        client_ip: str | None = None,
    ) -> UserLogin:
        """Async counterpart of :meth:`redeem_code` (same audit record, same ``client_ip``)."""
        try:
            login_url = _check_callback(login_url, code, state, expected_state)
            async with httpx.AsyncClient(timeout=30) as client:
                r = await _pkce.aredeem_code(
                    client, login_url=login_url, consumer_key=consumer_key,
                    consumer_secret=consumer_secret, redirect_uri=redirect_uri,
                    code=code, verifier=code_verifier,
                )
            login = _user_login(r)
        except Exception as e:
            _audit_login("backend", "denied", source_ip=client_ip, error=e)
            raise
        _audit_login(
            "backend", "success", source_ip=client_ip,
            org_id=login.org_id, user_id=login.user_id,
        )
        return login

    # ── Public API (mirrors ContextAuth) ────────────────────────────────────────

    @property
    def user_id(self) -> str:
        """The logged-in user's ``005…`` id, or ``""`` before the first fetch."""
        return self._user_id

    def get_token(self) -> tuple[str, str]:
        with self._lock:
            if time.time() >= self._expiry - _REFRESH_BUFFER:
                self._fetch_sync()
        return self._token, self._base_url

    async def aget_token(self) -> tuple[str, str]:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            if time.time() >= self._expiry - _REFRESH_BUFFER:
                await self._fetch_async()
        return self._token, self._base_url

    def get_core_creds(self) -> tuple[str, str]:
        with self._lock:
            if time.time() >= self._expiry - _REFRESH_BUFFER:
                self._fetch_sync()
        return self._core_token, self._core_instance_url

    async def aget_core_creds(self) -> tuple[str, str]:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            if time.time() >= self._expiry - _REFRESH_BUFFER:
                await self._fetch_async()
        return self._core_token, self._core_instance_url

    def invalidate(self) -> None:
        with self._lock:
            self._expiry = 0.0

    def attach(self, client: Any) -> None:
        if client not in self._attached:
            self._attached.append(client)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fetch_sync(self) -> None:
        with httpx.Client(timeout=30) as client:
            for attempt in range(_retry._MAX_RETRIES + 1):
                try:
                    (
                        token, base_url, expiry, core_token,
                        core_instance_url, user_id,
                    ) = self._grant_sync(client)
                    break
                except ReauthRequired:
                    self._discard_grant()
                    raise
                except Exception as e:
                    if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                        delay = _retry.backoff(attempt)
                        logger.warning(
                            "[UserContextAuth] token exchange failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, _retry._MAX_RETRIES + 1, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        raise
        self._apply(token, base_url, expiry, core_token, core_instance_url, user_id)

    async def _fetch_async(self) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(_retry._MAX_RETRIES + 1):
                try:
                    (
                        token, base_url, expiry, core_token,
                        core_instance_url, user_id,
                    ) = await self._grant_async(client)
                    break
                except ReauthRequired:
                    self._discard_grant()
                    raise
                except Exception as e:
                    if _retry.is_retryable(e) and attempt < _retry._MAX_RETRIES:
                        delay = _retry.backoff(attempt)
                        logger.warning(
                            "[UserContextAuth] token exchange failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, _retry._MAX_RETRIES + 1, e, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
        self._apply(token, base_url, expiry, core_token, core_instance_url, user_id)

    def _discard_grant(self) -> None:
        """The grant we last tried is dead: tell the store (best effort, never masks the error).

        The rejected token is passed along so the store deletes only that one;
        a fresh login saved since must survive.
        """
        if self._on_reauth is None or not self._refresh_token:
            return
        try:
            self._on_reauth(self._refresh_token)
        except Exception:
            logger.warning("[UserContextAuth] failed to delete dead refresh token", exc_info=True)

    def _grant_sync(self, client: httpx.Client) -> tuple[str, str, float, str, str, str]:
        self._reload_refresh_token()
        try:
            return self._exchange_once_sync(client)
        except ReauthRequired:
            # Another replica may have rotated the shared token since our load.
            if not self._reload_refresh_token():
                raise
            return self._exchange_once_sync(client)

    async def _grant_async(
        self, client: httpx.AsyncClient
    ) -> tuple[str, str, float, str, str, str]:
        self._reload_refresh_token()
        try:
            return await self._exchange_once_async(client)
        except ReauthRequired:
            if not self._reload_refresh_token():
                raise
            return await self._exchange_once_async(client)

    def _exchange_once_sync(self, client: httpx.Client) -> tuple[str, str, float, str, str, str]:
        return _exchange_refresh(
            client, self._login_url, self._require_refresh_token(),
            self._consumer_key, self._consumer_secret, self._token_ttl,
            aid=self._aid,
            on_rotate=self._rotate_refresh_token,
        )

    async def _exchange_once_async(
        self, client: httpx.AsyncClient
    ) -> tuple[str, str, float, str, str, str]:
        return await _exchange_refresh_async(
            client, self._login_url, self._require_refresh_token(),
            self._consumer_key, self._consumer_secret, self._token_ttl,
            aid=self._aid,
            on_rotate=self._rotate_refresh_token,
        )

    def _reload_refresh_token(self) -> bool:
        """Adopt the stored refresh token; return True if it replaced ours."""
        if self._load_refresh_token is None:
            return False
        latest = self._load_refresh_token()
        if (
            not latest
            or latest == self._refresh_token
            or latest in self._retired_refresh_tokens
        ):
            return False
        self._refresh_token = latest
        return True

    def _require_refresh_token(self) -> str:
        if not self._refresh_token:
            # Loader returned nothing and none was passed: the user never signed
            # in (or their stored grant was deleted).
            raise ReauthRequired("no_refresh_token", "no stored refresh token for this user")
        return self._refresh_token

    def _rotate_refresh_token(self, new: str) -> None:
        if self._refresh_token and self._refresh_token != new:
            self._retired_refresh_tokens.append(self._refresh_token)
        self._refresh_token = new
        if self._on_refresh_token is not None:
            self._on_refresh_token(new)

    def _apply(
        self, token: str, base_url: str, expiry: float,
        core_token: str, core_instance_url: str, user_id: str,
    ) -> None:
        self._token = token
        self._base_url = base_url
        self._core_token = core_token
        self._core_instance_url = core_instance_url
        if user_id:
            self._user_id = user_id
        self._expiry = expiry
        self._propagate(token)
        logger.info("[UserContextAuth] token refreshed, expiry=%.0f", expiry)

    def _propagate(self, token: str) -> None:
        for client in self._attached:
            client.api_key = token
