# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Pluggable storage for the backend-hosted user-login flow.

Two small interfaces, each with an in-memory and a file implementation:

* :class:`LoginStateStore` — the in-flight login between ``/oauth/login`` and
  ``/oauth/callback``: ``state`` + PKCE ``code_verifier`` + ``redirect_uri``.
  One-time use: :meth:`~LoginStateStore.consume` hands a login back at most once
  and only while it is unexpired, which is the CSRF / replay guard for the
  callback.
* :class:`RefreshTokenStore` — each user's long-lived refresh token, keyed by
  ``(org_id, user_id)``. :meth:`UserContextAuth.from_store` wires one in.

The built-in stores suit tests, demos and a single-process app. A production
backend with several replicas should implement the same few methods over its
own database (DynamoDB, Redis, ...) — the SDK never needs to know which. The
file stores keep secrets as **plaintext** (mode 0600); do not use them for real
refresh tokens in production.

The interfaces are synchronous, also when used from async code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Protocol, runtime_checkable

from .auth import AuthorizationRequest

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_STATE_TTL = 300.0  # seconds a started login stays redeemable


# ── Interfaces ───────────────────────────────────────────────────────────────

@runtime_checkable
class LoginStateStore(Protocol):
    """Holds an :class:`AuthorizationRequest` between ``/login`` and ``/callback``."""

    def save(self, request: AuthorizationRequest) -> None:
        """Remember ``request`` (keyed by ``request.state``) until it expires."""
        ...

    def consume(self, state: str) -> AuthorizationRequest | None:
        """Return and forget the login for ``state``.

        ``None`` when the state is unknown, expired, or already consumed. Must
        succeed for at most one caller per state, even across processes.
        """
        ...


@runtime_checkable
class RefreshTokenStore(Protocol):
    """Holds each user's refresh token, keyed by ``(org_id, user_id)``."""

    def load(self, org_id: str, user_id: str) -> str | None:
        """The stored refresh token, or ``None``."""
        ...

    def save(self, org_id: str, user_id: str, refresh_token: str) -> None:
        """Store (replace) the token. Must be durable when this returns."""
        ...

    def delete(self, org_id: str, user_id: str, *, if_token: str | None = None) -> None:
        """Forget the token. No error if absent.

        With ``if_token``, delete only if the stored value still equals it —
        an atomic compare-and-delete. The SDK passes the token Salesforce
        rejected, so a fresh login saved meanwhile is never wiped. An
        implementation over a shared database must make the compare and the
        delete one atomic operation (e.g. a conditional delete).
        """
        ...


def _audit_token_delete(store: str, org_id: str, user_id: str, outcome: str, conditional: bool) -> None:
    """Audit record for removing a stored credential. Never includes the token itself.

    ``outcome`` is ``deleted``, ``absent`` or ``skipped_token_changed``. The
    store cannot know who asked (the SDK, on a rejected grant, or the app);
    callers who need an actor should log it alongside the call. The timestamp
    comes from the log record.
    """
    logger.info(
        "audit: refresh_token delete store=%s org_id=%s user_id=%s token_type=refresh_token "
        "conditional=%s outcome=%s",
        store, org_id, user_id, conditional, outcome,
        extra={
            "audit_event": "refresh_token_delete", "store": store, "org_id": org_id,
            "user_id": user_id, "token_type": "refresh_token",
            "conditional": conditional, "outcome": outcome,
        },
    )


def _require_key(org_id: str, user_id: str) -> None:
    if not org_id or not user_id:
        raise ValueError("org_id and user_id must both be non-empty")


# ── In-memory ────────────────────────────────────────────────────────────────

class InMemoryLoginStateStore:
    """Process-local :class:`LoginStateStore`. Lost on restart; not shared across replicas."""

    def __init__(self, *, expiration_seconds: float = DEFAULT_STATE_TTL) -> None:
        self._ttl = expiration_seconds
        self._items: dict[str, tuple[AuthorizationRequest, float]] = {}
        self._lock = Lock()

    def save(self, request: AuthorizationRequest) -> None:
        now = time.time()
        with self._lock:
            self._items = {s: v for s, v in self._items.items() if v[1] > now}
            self._items[request.state] = (request, now + self._ttl)

    def consume(self, state: str) -> AuthorizationRequest | None:
        with self._lock:
            item = self._items.pop(state, None)
        if item is None or item[1] <= time.time():
            return None
        return item[0]


class InMemoryRefreshTokenStore:
    """Process-local :class:`RefreshTokenStore`. Lost on restart; not shared across replicas."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], str] = {}
        self._lock = Lock()

    def load(self, org_id: str, user_id: str) -> str | None:
        _require_key(org_id, user_id)
        with self._lock:
            return self._items.get((org_id, user_id))

    def save(self, org_id: str, user_id: str, refresh_token: str) -> None:
        _require_key(org_id, user_id)
        if not refresh_token:
            raise ValueError("refresh_token must be non-empty")
        with self._lock:
            self._items[(org_id, user_id)] = refresh_token

    def delete(self, org_id: str, user_id: str, *, if_token: str | None = None) -> None:
        _require_key(org_id, user_id)
        with self._lock:
            current = self._items.get((org_id, user_id))
            if current is None:
                outcome = "absent"
            elif if_token is None or current == if_token:
                del self._items[(org_id, user_id)]
                outcome = "deleted"
            else:
                outcome = "skipped_token_changed"
        _audit_token_delete("memory", org_id, user_id, outcome, if_token is not None)


# ── File ─────────────────────────────────────────────────────────────────────

def _private_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write JSON to ``path`` (mode 0600) so readers see the old or new file, never a partial one."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:  # mkstemp creates the file 0600
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _hashed_name(*parts: str) -> str:
    """File name for ``parts``: an unsalted SHA-256, there for path safety only.

    Hashing keeps caller-supplied ids out of file names (no path traversal). It
    is **not** a confidentiality boundary: the name is deterministic, so anyone
    who can list the directory can test a known ``(org_id, user_id)`` by hashing
    it. The 0700 directory is what protects the files.
    """
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest() + ".json"


class FileLoginStateStore:
    """:class:`LoginStateStore` backed by one 0600 JSON file per login in ``directory``.

    Works across processes sharing the directory: ``consume`` claims the file with
    an atomic rename, so only one caller wins. Secrets are stored in plaintext.
    """

    def __init__(
        self, directory: str | os.PathLike[str], *, expiration_seconds: float = DEFAULT_STATE_TTL
    ) -> None:
        self._dir = _private_dir(directory)
        self._ttl = expiration_seconds

    def save(self, request: AuthorizationRequest) -> None:
        self._purge_expired()
        _write_atomic(self._dir / _hashed_name(request.state), {
            "url": request.url,
            "state": request.state,
            "code_verifier": request.code_verifier,
            "redirect_uri": request.redirect_uri,
            "expires_at": time.time() + self._ttl,
        })

    def consume(self, state: str) -> AuthorizationRequest | None:
        path = self._dir / _hashed_name(state)
        claimed = path.with_name(f"{path.name}.claimed-{uuid.uuid4().hex}")
        try:
            os.rename(path, claimed)  # atomic: exactly one caller can win
        except FileNotFoundError:
            return None
        try:
            data = json.loads(claimed.read_text())
        except (OSError, ValueError):
            return None
        finally:
            claimed.unlink(missing_ok=True)
        if data.get("expires_at", 0) <= time.time():
            return None
        return AuthorizationRequest(
            url=data["url"], state=data["state"],
            code_verifier=data["code_verifier"], redirect_uri=data["redirect_uri"],
        )

    def _purge_expired(self) -> None:
        now = time.time()
        for path in self._dir.glob("*.json"):
            try:
                if json.loads(path.read_text()).get("expires_at", 0) <= now:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError):
                continue


class FileRefreshTokenStore:
    """:class:`RefreshTokenStore` backed by one 0600 JSON file per user in ``directory``.

    Tokens are stored in **plaintext** — for local development and demos only.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = _private_dir(directory)
        self._thread_lock = Lock()

    def _path(self, org_id: str, user_id: str) -> Path:
        _require_key(org_id, user_id)
        return self._dir / _hashed_name(org_id, user_id)

    @contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        """Serialise save and compare-and-delete across threads and processes."""
        with self._thread_lock:
            if fcntl is None:  # no advisory file locks (Windows): thread lock only
                yield
                return
            fd = os.open(self._dir / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)  # closing releases the flock

    def load(self, org_id: str, user_id: str) -> str | None:
        path = self._path(org_id, user_id)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # Unreadable or corrupt/partial file: treat as "no token" so the
            # caller gets ReauthRequired and the user signs in again, instead
            # of a decode error crashing the refresh. Same policy as
            # FileLoginStateStore.consume.
            logger.warning("refresh-token file unreadable or corrupt; treating as absent: %s", path.name)
            return None
        return (data.get("refresh_token") or None) if isinstance(data, dict) else None

    def save(self, org_id: str, user_id: str, refresh_token: str) -> None:
        if not refresh_token:
            raise ValueError("refresh_token must be non-empty")
        path = self._path(org_id, user_id)
        with self._locked(path):
            _write_atomic(path, {
                "org_id": org_id, "user_id": user_id, "refresh_token": refresh_token,
            })

    def delete(self, org_id: str, user_id: str, *, if_token: str | None = None) -> None:
        path = self._path(org_id, user_id)
        if if_token is None:
            outcome = "deleted" if path.exists() else "absent"
            path.unlink(missing_ok=True)
        else:
            with self._locked(path):
                current = self.load(org_id, user_id)
                if current is None:
                    outcome = "absent"
                elif current == if_token:
                    path.unlink(missing_ok=True)
                    outcome = "deleted"
                else:
                    outcome = "skipped_token_changed"
        _audit_token_delete("file", org_id, user_id, outcome, if_token is not None)
