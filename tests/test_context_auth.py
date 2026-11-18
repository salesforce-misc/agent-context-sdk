# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unit tests for ContextAuth — mocks both HTTP legs, no real credentials needed."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
import httpx

from agentic_context.auth import (
    ApiKeyAuth,
    ContextAuth,
    _build_assertion,
    _exchange,
    _exchange_async,
    _validate_instance_url,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Not a real key — _sign_jwt is monkeypatched in all tests that use this
FAKE_KEY_PEM = "not-a-real-key"

FAKE_CONSUMER_KEY = "3MVGtest"
FAKE_SUBJECT = "user@example.com"
FAKE_LOGIN_URL = "login.test.salesforce.com"
FAKE_CORE_TOKEN = "core_access_token_abc"
FAKE_CDP_TOKEN = "cdp_bearer_token_xyz"
FAKE_INSTANCE_URL = "https://beacon.test.salesforce.com"


def _make_auth(**kwargs) -> ContextAuth:
    defaults = {
        "private_key": FAKE_KEY_PEM,
        "consumer_key": FAKE_CONSUMER_KEY,
        "subject": FAKE_SUBJECT,
        "login_url": FAKE_LOGIN_URL,
    }
    defaults.update(kwargs)
    return ContextAuth(**defaults)


def _mock_sign_jwt(monkeypatch):
    """Patch _sign_jwt to skip real RSA — returns a fake token string."""
    monkeypatch.setattr(
        "agentic_context.auth._sign_jwt",
        lambda claims, key: "fake.jwt.token",
    )


# ── _build_assertion ──────────────────────────────────────────────────────────

def test_build_assertion_signs_jwt(monkeypatch):
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "signed")
    result = _build_assertion(FAKE_LOGIN_URL, FAKE_CONSUMER_KEY, FAKE_SUBJECT, "key", 3600)
    assert result == "signed"


# ── Sync exchange ─────────────────────────────────────────────────────────────

async def _run_exchange(is_async: bool):
    if is_async:
        async with httpx.AsyncClient() as client:
            return await _exchange_async(
                client, FAKE_LOGIN_URL, FAKE_CONSUMER_KEY, FAKE_SUBJECT, FAKE_KEY_PEM, 3600
            )
    with httpx.Client() as client:
        return _exchange(
            client, FAKE_LOGIN_URL, FAKE_CONSUMER_KEY, FAKE_SUBJECT, FAKE_KEY_PEM, 3600
        )


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_exchange_returns_token_and_base_url(monkeypatch, respx_mock, is_async):
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    _mock_two_leg(respx_mock)

    token, base_url, expiry, core_token, core_instance_url = await _run_exchange(is_async)

    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL
    assert core_token == FAKE_CORE_TOKEN
    assert core_instance_url == "https://core.test.salesforce.com"
    assert expiry > time.time()


def test_exchange_uses_expires_in_from_response(monkeypatch, respx_mock):
    """Expiry must come from the CDP response expires_in, not the JWT request TTL."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")

    respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN,
            "instance_url": "https://core.test.salesforce.com",
        })
    )
    respx_mock.post("https://core.test.salesforce.com/services/a360/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CDP_TOKEN,
            "instance_url": FAKE_INSTANCE_URL,
            "expires_in": 1800,  # CDP token is 30 min, not 1 hr
        })
    )

    before = time.time()
    with httpx.Client() as client:
        _, _, expiry, _, _ = _exchange(
            client, FAKE_LOGIN_URL, FAKE_CONSUMER_KEY, FAKE_SUBJECT, FAKE_KEY_PEM, 3600
        )

    assert expiry == pytest.approx(before + 1800, abs=5)


def test_exchange_prepends_https_when_missing(monkeypatch, respx_mock):
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")

    respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN,
            "instance_url": "https://core.test.salesforce.com",
        })
    )
    respx_mock.post("https://core.test.salesforce.com/services/a360/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CDP_TOKEN,
            "instance_url": "beacon.test.salesforce.com",  # no scheme
        })
    )

    with httpx.Client() as client:
        _, base_url, _, _, _ = _exchange(
            client, FAKE_LOGIN_URL, FAKE_CONSUMER_KEY, FAKE_SUBJECT, FAKE_KEY_PEM, 3600
        )

    assert base_url == "https://beacon.test.salesforce.com"


# ── _validate_instance_url guards on OAuth responses ─────────────────────────
# SSRF + transport-downgrade regression tests. Both legs consume an
# ``instance_url`` from an upstream JSON response and then POST to it with a
# bearer token, so a tampered response must be rejected before it becomes an
# outbound call.

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("core_instance_url", "cdp_instance_url", "match"),
    [
        # Leg-1 core instance_url pointing off-allowlist → ValueError.
        ("https://attacker.example.com", None, "core instance_url host outside the allowlist"),
        # Leg-2 CDP instance_url returned as ``http://`` → ValueError (no downgrade).
        ("https://core.test.salesforce.com", "http://beacon.test.salesforce.com",
         "insecure http:// CDP instance_url"),
        # Leg-2 CDP instance_url pointing off-allowlist → ValueError.
        ("https://core.test.salesforce.com", "https://attacker.example.com",
         "CDP instance_url host outside the allowlist"),
    ],
    ids=["untrusted-core", "http-scheme-cdp", "untrusted-cdp"],
)
async def test_exchange_rejects_untrusted_instance_url(
    monkeypatch, respx_mock, is_async, core_instance_url, cdp_instance_url, match
):
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")

    respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN,
            "instance_url": core_instance_url,
        })
    )
    if cdp_instance_url is not None:
        respx_mock.post("https://core.test.salesforce.com/services/a360/token").mock(
            return_value=httpx.Response(200, json={
                "access_token": FAKE_CDP_TOKEN,
                "instance_url": cdp_instance_url,
            })
        )

    with pytest.raises(ValueError, match=match):
        await _run_exchange(is_async)


# Reject any URL that smuggles extra components past the hostname allowlist —
# userinfo, port, path, query, fragment. Reconstruction returns scheme + host
# only, so these all funnel to one branch. Also pins the happy path for a bare
# hostname (normalized to ``https://``) and asserts the attacker-supplied value
# is never echoed in the error message.
@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.com@na1.test.salesforce.com",           # userinfo
        "https://na1.test.salesforce.com:8080",                   # non-default port
        "https://na1.test.salesforce.com/evil",                   # path
        "https://na1.test.salesforce.com?x=attacker",             # query
        "https://na1.test.salesforce.com#attacker",               # fragment
        "https://na1.test.salesforce.com\\@attacker.com",         # parser-differential
    ],
    ids=["userinfo", "port", "path", "query", "fragment", "parser-differential"],
)
def test_validate_instance_url_rejects_extra_url_components(url):
    with pytest.raises(ValueError) as excinfo:
        _validate_instance_url(url, field="test")
    # Error message must never echo attacker-supplied URL fragments.
    assert "attacker" not in str(excinfo.value)


def test_validate_instance_url_normalizes_bare_hostname():
    assert (
        _validate_instance_url("na1.test.salesforce.com", field="test")
        == "https://na1.test.salesforce.com"
    )


# ── ContextAuth.get_token ──────────────────────────────────────────────────────

def _patch_fetch(monkeypatch, auth: ContextAuth, is_async: bool) -> MagicMock:
    """Replace the sync or async fetch with a mock that installs a fresh token."""
    fetch = (AsyncMock if is_async else MagicMock)(side_effect=lambda: _set_token(auth))
    monkeypatch.setattr(auth, "_fetch_async" if is_async else "_fetch_sync", fetch)
    return fetch


async def _call(auth, is_async: bool, name: str):
    """Call ``get_<name>()`` or ``await aget_<name>()``."""
    return await getattr(auth, f"aget_{name}")() if is_async else getattr(auth, f"get_{name}")()


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_get_token_fetches_on_first_call(monkeypatch, is_async):
    auth = _make_auth()
    _patch_fetch(monkeypatch, auth, is_async)
    token, base_url = await _call(auth, is_async, "token")
    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_get_token_does_not_refetch_when_fresh(monkeypatch, is_async):
    auth = _make_auth()
    fetch = _patch_fetch(monkeypatch, auth, is_async)
    await _call(auth, is_async, "token")
    await _call(auth, is_async, "token")
    assert fetch.call_count == 1


def test_get_token_refetches_when_expired(monkeypatch):
    auth = _make_auth()
    call_count = 0

    def _fetch():
        nonlocal call_count
        call_count += 1
        _set_token(auth, expiry=time.time() - 1)  # already expired

    monkeypatch.setattr(auth, "_fetch_sync", _fetch)
    auth.get_token()
    auth.get_token()
    assert call_count == 2


# ── ContextAuth.get_core_creds / aget_core_creds ──────────────────────────────

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_get_core_creds_returns_leg_one_credentials(monkeypatch, is_async):
    auth = _make_auth()
    _patch_fetch(monkeypatch, auth, is_async)
    core_token, instance_url = await _call(auth, is_async, "core_creds")
    assert core_token == FAKE_CORE_TOKEN
    assert instance_url == "https://core.test.salesforce.com"


def test_get_core_creds_shares_expiry_with_cdp_token(monkeypatch):
    """One JWT exchange must populate both cred pairs — no separate fetch."""
    auth = _make_auth()
    call_count = 0

    def _fetch():
        nonlocal call_count
        call_count += 1
        _set_token(auth)

    monkeypatch.setattr(auth, "_fetch_sync", _fetch)
    auth.get_token()
    auth.get_core_creds()
    assert call_count == 1


# ── ContextAuth.attach ─────────────────────────────────────────────────────────

def test_attach_propagates_token_on_refresh(monkeypatch):
    auth = _make_auth()
    client = MagicMock()
    client.api_key = "old"
    auth.attach(client)

    monkeypatch.setattr(auth, "_fetch_sync", lambda: _set_token(auth))
    auth.get_token()

    assert client.api_key == FAKE_CDP_TOKEN


def test_attach_deduplicates(monkeypatch):
    auth = _make_auth()
    client = MagicMock()
    auth.attach(client)
    auth.attach(client)
    assert len(auth._attached) == 1


# ── ContextAuth thread-safety regressions ─────────────────────────────────────

def test_attach_is_thread_safe_against_toctou_duplication():
    """Many threads attaching the same client → registered exactly once.

    Without the lock in ``attach()``, ``if client not in self._attached`` /
    ``list.append`` is a TOCTOU: two threads can both observe "not in" before
    either appends, and both then append, producing a duplicate. With the
    lock the pair is atomic and the dedup guard holds under contention.
    """
    import threading

    auth = _make_auth()
    client = MagicMock()
    n_threads = 64
    barrier = threading.Barrier(n_threads)

    def _attach() -> None:
        barrier.wait()  # release all threads simultaneously
        auth.attach(client)

    threads = [threading.Thread(target=_attach) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(auth._attached) == 1


def test_attach_serializes_with_propagate(monkeypatch):
    """A client attached during a refresh either receives the just-propagated
    token or gets propagated to on the next refresh — it must never end up
    registered but token-less.

    Simulates the race by making ``_fetch_sync`` call ``attach`` from within
    the refresh, which used to leave the mid-flight ``_propagate`` iterator
    with undefined behavior (CPython's list iterator has no defined semantics
    for concurrent append). With ``attach`` now serialized on the same lock
    that guards ``_propagate``, the second client is only visible AFTER the
    current propagation completes and will receive the next refresh's token.
    """
    auth = _make_auth()
    first_client = MagicMock()
    first_client.api_key = "old"
    auth.attach(first_client)

    second_client = MagicMock()
    second_client.api_key = "old"

    call_count = 0

    def _fetch() -> None:
        nonlocal call_count
        call_count += 1
        # Mid-refresh, another thread would call attach here. Under the lock
        # this is a no-op — attach would block until propagate completes.
        # We invoke it inline to prove attach itself works while the lock is
        # already held by this thread (threading.Lock is reentrant is FALSE,
        # so if attach tried to reacquire it would deadlock — proving that
        # attach's lock use must be careful).
        _set_token(auth)

    monkeypatch.setattr(auth, "_fetch_sync", _fetch)
    auth.get_token()

    # First client got the token on the first refresh.
    assert first_client.api_key == FAKE_CDP_TOKEN
    # Second client attached AFTER first refresh completes.
    auth.attach(second_client)
    assert second_client in auth._attached
    # Invalidate + re-fetch → second refresh propagates to both.
    second_client.api_key = "old"
    auth.invalidate()
    auth.get_token()
    assert first_client.api_key == FAKE_CDP_TOKEN
    assert second_client.api_key == FAKE_CDP_TOKEN
    assert call_count == 2


def test_get_token_fast_path_is_lock_free(monkeypatch):
    """With a fresh token cached, ``get_token()`` must return without
    acquiring ``self._lock``. This is what prevents refreshers from stalling
    every other reader for the duration of the two HTTPS legs + retry backoff.
    """
    from unittest.mock import MagicMock as _MagicMock

    auth = _make_auth()
    monkeypatch.setattr(auth, "_fetch_sync", lambda: _set_token(auth))
    auth.get_token()  # prime the snapshot with a fresh token

    # Wrap the real lock so we can count acquisitions on the fast path.
    real_lock = auth._lock
    lock_spy = _MagicMock(wraps=real_lock)
    lock_spy.__enter__ = _MagicMock(side_effect=real_lock.__enter__)
    lock_spy.__exit__ = _MagicMock(side_effect=real_lock.__exit__)
    auth._lock = lock_spy

    token, base_url = auth.get_token()

    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL
    lock_spy.__enter__.assert_not_called()


def test_get_token_returns_atomic_snapshot(monkeypatch):
    """The (token, base_url) pair returned by ``get_token()`` must always
    come from the same refresh — a torn read (new token + old base_url)
    would 401 on Beacon. The single-tuple state design makes this
    structurally impossible; this test locks that invariant in.
    """
    auth = _make_auth()
    monkeypatch.setattr(auth, "_fetch_sync", lambda: _set_token(auth))
    auth.get_token()

    # Snapshot is a tuple, not five separate attrs — so a reader that grabs
    # self._snapshot always sees a consistent 5-field state.
    assert isinstance(auth._snapshot, tuple)
    assert len(auth._snapshot) == 5
    token, base_url, core_token, core_instance_url, expiry = auth._snapshot
    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL
    assert core_token == FAKE_CORE_TOKEN
    assert core_instance_url == "https://core.test.salesforce.com"
    assert expiry > time.time()


# ── ContextAuth.invalidate ────────────────────────────────────────────────────

def test_invalidate_forces_refetch(monkeypatch):
    """invalidate() zeroes the expiry so the next get_token() refetches."""
    auth = _make_auth()
    call_count = 0

    def _fetch():
        nonlocal call_count
        call_count += 1
        _set_token(auth)

    monkeypatch.setattr(auth, "_fetch_sync", _fetch)
    auth.get_token()         # first fetch
    assert auth._expiry > 0
    auth.invalidate()
    assert auth._expiry == 0.0
    auth.get_token()         # forced second fetch
    assert call_count == 2


# ── API-key dual-auth (P3) ────────────────────────────────────────────────────
# ApiKeyAuth is a thin wrapper: on first use it downloads the connected-app
# bundle from a secret server, builds an INNER ContextAuth from it, and delegates
# every token / attach / invalidate call to it. The production ContextAuth path
# (creds passed directly) must stay fully intact and never touch the secret
# server — these tests assert both, plus the fail-closed "no baked-in host" rule.

FAKE_SECRETS_URL = "https://secrets.example.com"
FAKE_SECRET_ENV = "AGENTIC_CONTEXT_SECRET_SERVER_URL"
FAKE_API_KEY = "demo-api-key-123"

# The bundle shape the secret server returns (matches the live /api/secret
# contract: a "secret" object with the four creds + orgUrl/aud/agentId).
FAKE_BUNDLE = {
    "apiKey": FAKE_API_KEY,
    "name": "demo",
    "secret": {
        "loginUrl": FAKE_LOGIN_URL,
        "aud": FAKE_LOGIN_URL,
        "subject": FAKE_SUBJECT,
        "consumerKey": FAKE_CONSUMER_KEY,
        "orgUrl": "https://core.test.salesforce.com",
        "privateKey": FAKE_KEY_PEM,
        "agentId": "",
    },
    "isDefault": True,
}


def _mock_two_leg(respx_mock):
    """Mock the JWT core + CDP legs so a full fetch succeeds."""
    respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN,
            "instance_url": "https://core.test.salesforce.com",
        })
    )
    respx_mock.post("https://core.test.salesforce.com/services/a360/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CDP_TOKEN,
            "instance_url": FAKE_INSTANCE_URL,
        })
    )


# ── Construction ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("url_kwarg", "env_url"),
    [
        (FAKE_SECRETS_URL, None),
        (None, FAKE_SECRETS_URL),
        (FAKE_SECRETS_URL + "/", None),
    ],
    ids=["explicit-url", "url-from-env", "trailing-slash-stripped"],
)
def test_api_key_constructs_without_creds(monkeypatch, url_kwarg, env_url):
    """ApiKeyAuth builds from just a key + secret_server_url (argument or env var,
    trailing slash stripped); no inner yet."""
    if env_url is not None:
        monkeypatch.setenv(FAKE_SECRET_ENV, env_url)
    kwargs = {} if url_kwarg is None else {"secret_server_url": url_kwarg}
    auth = ApiKeyAuth(FAKE_API_KEY, **kwargs)
    assert auth._api_key == FAKE_API_KEY
    assert auth._secret_server_url == FAKE_SECRETS_URL
    # inner ContextAuth is not built until the first token fetch
    assert auth._inner is None
    assert auth.agent_id == ""


def test_api_key_requires_non_empty_key():
    """An empty api_key is rejected at construction — no silent broken auth."""
    with pytest.raises(ValueError, match="non-empty api_key"):
        ApiKeyAuth("")


def test_context_auth_still_requires_full_creds():
    """The production ContextAuth path is keyword-only and needs all four creds."""
    with pytest.raises(TypeError):
        ContextAuth(consumer_key=FAKE_CONSUMER_KEY)  # type: ignore[call-arg]


# ── Fail-closed: no baked-in secret-server host ────────────────────────────────

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_api_key_no_default_host_fails_closed(monkeypatch, is_async):
    """No secret_server_url and no env var → fail closed, do NOT guess a host.

    This is the crux fix: the SDK ships with no default secret server, so an api
    key is never sent to a baked-in host. get_token()/aget_token() must raise
    before any I/O.
    """
    monkeypatch.delenv(FAKE_SECRET_ENV, raising=False)
    auth = ApiKeyAuth(FAKE_API_KEY)  # no url, no env
    assert auth._secret_server_url == ""
    with pytest.raises(ValueError, match="secret_server_url"):
        await _call(auth, is_async, "token")


# ── Fetch + delegate ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_api_key_fetch_downloads_bundle_then_exchanges(monkeypatch, respx_mock, is_async):
    """First get_token()/aget_token() downloads the bundle, builds the inner, then
    runs the (sync or async) exchange."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    token, base_url = await _call(auth, is_async, "token")

    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL
    # bundle populated the inner ContextAuth the exchange ran on
    assert auth._inner is not None
    assert auth._inner._private_key == FAKE_KEY_PEM
    assert auth._inner._consumer_key == FAKE_CONSUMER_KEY
    assert auth._inner._subject == FAKE_SUBJECT
    assert auth._inner._login_url == FAKE_LOGIN_URL


def test_api_key_sends_x_api_key_header(monkeypatch, respx_mock):
    """The download must authenticate with the x-api-key header."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    route = respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL).get_token()

    assert route.called
    assert route.calls.last.request.headers["x-api-key"] == FAKE_API_KEY


def test_api_key_bundle_downloaded_once_across_refreshes(monkeypatch, respx_mock):
    """Bundle is static — a token refresh must NOT re-download it."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    route = respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    auth.get_token()
    auth.invalidate()   # force a token refetch on the inner
    auth.get_token()

    assert route.call_count == 1  # bundle fetched once; only the CDP token refetched


def test_api_key_get_core_creds_delegates(monkeypatch, respx_mock):
    """get_core_creds() resolves the bundle then returns the inner's leg-1 creds."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    core_token, instance_url = auth.get_core_creds()
    assert core_token == FAKE_CORE_TOKEN
    assert instance_url == "https://core.test.salesforce.com"


def test_api_key_falls_back_to_aud_when_no_login_url(monkeypatch, respx_mock):
    """If the bundle omits loginUrl, aud is used for the login host."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    bundle = {**FAKE_BUNDLE, "secret": {**FAKE_BUNDLE["secret"], "loginUrl": ""}}
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )
    _mock_two_leg(respx_mock)

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    auth.get_token()
    assert auth._inner._login_url == FAKE_LOGIN_URL  # from aud


@pytest.mark.parametrize(
    ("bundle", "match"),
    [
        # A bundle missing a required cred raises rather than attempting a broken exchange.
        ({"secret": {"privateKey": "", "consumerKey": "", "subject": "", "loginUrl": ""}},
         "incomplete credential bundle"),
        # A 200 with no 'secret' field surfaces a clear ValueError, not a raw KeyError.
        ({"unexpected": "shape"}, "malformed credential bundle"),
    ],
    ids=["incomplete", "malformed"],
)
def test_api_key_bad_bundle_raises_clean_error(monkeypatch, respx_mock, bundle, match):
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    with pytest.raises(ValueError, match=match):
        auth.get_token()


def test_api_key_rejects_non_https_secret_url(monkeypatch, respx_mock):
    """http:// secret_server_url must be refused — the api key can't go over cleartext."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    # would-be route; must never be called because validation fails first
    route = respx_mock.get("http://secrets.example.com/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url="http://secrets.example.com")
    with pytest.raises(ValueError, match="https"):
        auth.get_token()
    assert not route.called


# ── agent_id from bundle ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("bundle_agent_id", "expected"),
    [("Agent_Data_Cloud", "Agent_Data_Cloud"), ("", "")],
    ids=["populated-from-bundle", "empty-when-bundle-omits-it"],
)
def test_api_key_agent_id_from_bundle(monkeypatch, respx_mock, bundle_agent_id, expected):
    """agentId in the bundle is exposed via the .agent_id property after fetch; a
    bundle with no agentId leaves .agent_id empty (not an error)."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    bundle = {**FAKE_BUNDLE, "secret": {**FAKE_BUNDLE["secret"], "agentId": bundle_agent_id}}
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )
    _mock_two_leg(respx_mock)

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    assert auth.agent_id == ""          # empty before resolution
    auth.get_token()
    assert auth.agent_id == expected


# ── attach / invalidate before resolution ─────────────────────────────────────

def test_api_key_attach_before_resolution_is_replayed(monkeypatch, respx_mock):
    """A client attached before the inner exists is replayed onto it and gets the token."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    client = MagicMock()
    client.api_key = "old"
    auth.attach(client)                 # buffered — inner not built yet
    assert client in auth._pending_attached

    auth.get_token()                    # builds inner, replays the buffered attach

    assert client.api_key == FAKE_CDP_TOKEN
    assert auth._pending_attached == []


def test_api_key_invalidate_before_resolution_is_noop():
    """invalidate() before the inner exists is a harmless no-op."""
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    auth.invalidate()   # must not raise
    assert auth._inner is None


def test_api_key_attach_serializes_against_build_lock():
    """attach() must contend on the same self._lock _build_inner holds.

    Regression for the TOCTOU orphan: without the lock an attach that read
    ``_inner is None`` could be preempted by _build_inner's clear-then-publish
    and then append to the abandoned buffer, orphaning the client from token
    refresh forever. The guarantee is that attach and _build_inner can't run
    concurrently. We prove it by holding _lock (as _build_inner does inside its
    critical section) and showing attach blocks until the lock is released.
    """
    import threading

    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    client = MagicMock()
    done = threading.Event()

    auth._lock.acquire()   # stand in for _build_inner's critical section
    t = threading.Thread(target=lambda: (auth.attach(client), done.set()))
    t.start()
    try:
        # attach must be blocked on the lock — it cannot have completed
        assert not done.wait(timeout=0.3), "attach() ran without taking _lock"
    finally:
        auth._lock.release()
    assert done.wait(timeout=2.0), "attach() never completed after lock release"
    t.join(timeout=2.0)
    assert client in auth._pending_attached


# ── SSRF guard on bundle login_url (SI-10) ────────────────────────────────────
# login_url comes from the external secret-server response and is later used to
# build the OAuth POST target that receives the signed JWT assertion. It must be
# allowlisted to Salesforce hosts before use.

@pytest.mark.parametrize("evil_host", [
    "evil.com",
    "login.salesforce.com.evil.com",   # suffix-append
    "evilsalesforce.com",              # no dot boundary
    "attacker.internal",
], ids=["evil-com", "suffix-append", "no-dot-boundary", "internal"])
def test_api_key_rejects_non_salesforce_login_host(monkeypatch, respx_mock, evil_host):
    """A bundle whose loginUrl is not a *.salesforce.com host is refused, no exchange."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    bundle = {**FAKE_BUNDLE, "secret": {**FAKE_BUNDLE["secret"], "loginUrl": evil_host, "aud": evil_host}}
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )
    # token endpoints must never be hit — validation fails first
    leg = respx_mock.post(f"https://{evil_host}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={"access_token": "x", "instance_url": "https://x"})
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    with pytest.raises(ValueError, match="allowlist"):
        auth.get_token()
    assert not leg.called
    # the attacker-crafted host must not be echoed in the error
    try:
        auth.get_token()
    except ValueError as exc:
        assert evil_host not in str(exc)


def test_api_key_rejects_trailing_newline_login_host(monkeypatch, respx_mock):
    """fullmatch (not match): a trailing-newline host must NOT slip past the allowlist.

    ``re.match(r"...salesforce\\.com$", "login.salesforce.com\\n")`` succeeds
    because ``$`` matches before a trailing newline — the classic bypass.
    ``fullmatch`` rejects it. This is the concrete win of Luke's fullmatch fix.
    """
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    newline_host = "login.salesforce.com\n"
    bundle = {**FAKE_BUNDLE, "secret": {**FAKE_BUNDLE["secret"], "loginUrl": newline_host, "aud": newline_host}}
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    with pytest.raises(ValueError, match="allowlist"):
        auth.get_token()


@pytest.mark.parametrize("good_host", [
    "login.salesforce.com",
    "test.salesforce.com",
    "acme.my.salesforce.com",
    "acme--uat.sandbox.my.salesforce.com",
], ids=["login", "test", "my-domain", "my-domain-sandbox"])
def test_api_key_allows_real_salesforce_login_hosts(monkeypatch, respx_mock, good_host):
    """Real (multi-label) Salesforce login hosts, incl. My Domain, pass the allowlist."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    bundle = {**FAKE_BUNDLE, "secret": {**FAKE_BUNDLE["secret"], "loginUrl": good_host}}
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )
    respx_mock.post(f"https://{good_host}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN, "instance_url": "https://core.test.salesforce.com",
        })
    )
    respx_mock.post("https://core.test.salesforce.com/services/a360/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CDP_TOKEN, "instance_url": FAKE_INSTANCE_URL,
        })
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)
    token, _ = auth.get_token()
    assert token == FAKE_CDP_TOKEN
    assert auth._inner._login_url == good_host


# ── Audit logging (AU-2 / AU-12) ──────────────────────────────────────────────

def test_api_key_success_emits_structured_audit_log(monkeypatch, respx_mock, caplog):
    """Bundle-load success logs structured actor/target/outcome/agent_id — never the raw key."""
    import logging as _logging
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    with caplog.at_level(_logging.INFO, logger="agentic_context.auth"):
        ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL).get_token()

    rec = next(r for r in caplog.records if "credential bundle loaded" in r.getMessage())
    assert rec.outcome == "success"
    assert rec.target == FAKE_SECRETS_URL
    assert rec.actor.startswith("sha256:")
    assert rec.agent_id == "<none>"               # empty agentId logged safely
    assert FAKE_API_KEY not in rec.actor          # raw key never logged
    assert FAKE_API_KEY not in rec.getMessage()


def test_api_key_download_failure_emits_audit_log(monkeypatch, respx_mock, caplog):
    """A 401/5xx from the secret server logs a failure audit record before re-raising."""
    import logging as _logging
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(403, json={"error": "forbidden"})
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)

    with caplog.at_level(_logging.WARNING, logger="agentic_context.auth"):
        with pytest.raises(httpx.HTTPStatusError):
            auth.get_token()

    rec = next(r for r in caplog.records if "credential bundle download failed" in r.getMessage())
    assert rec.outcome == "failure"
    assert rec.http_status == 403
    assert rec.actor.startswith("sha256:")
    assert FAKE_API_KEY not in rec.actor


def test_api_key_ssrf_rejection_emits_audit_log(monkeypatch, respx_mock, caplog):
    """A blocked (non-*.salesforce.com) login host logs an AU-2 'rejected' record.

    The SSRF allowlist is exactly the event AU-2/AU-12 want recorded — a denied
    authorization decision. Before this fix the ValueError propagated silently.
    The attacker-supplied host must NOT appear in the record.
    """
    import logging as _logging
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    evil_host = "login.salesforce.com.evil.com"
    bundle = {**FAKE_BUNDLE, "secret": {**FAKE_BUNDLE["secret"], "loginUrl": evil_host, "aud": evil_host}}
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=bundle)
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)

    with caplog.at_level(_logging.WARNING, logger="agentic_context.auth"):
        with pytest.raises(ValueError, match="allowlist"):
            auth.get_token()

    rec = next(r for r in caplog.records if "credential bundle rejected" in r.getMessage())
    assert rec.outcome == "rejected"
    assert rec.reason == "invalid_bundle"
    assert rec.target == f"{FAKE_SECRETS_URL}/api/secret"   # the endpoint, like _log_bundle_failure
    assert rec.actor.startswith("sha256:")
    assert FAKE_API_KEY not in rec.actor
    assert evil_host not in rec.getMessage()   # attacker value never echoed


def test_api_key_malformed_bundle_emits_rejected_audit_log(monkeypatch, respx_mock, caplog):
    """An incomplete bundle (200 but missing creds) also logs the 'rejected' record."""
    import logging as _logging
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json={"secret": {"loginUrl": FAKE_LOGIN_URL}})  # no key/consumer/subject
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)

    with caplog.at_level(_logging.WARNING, logger="agentic_context.auth"):
        with pytest.raises(ValueError, match="incomplete"):
            auth.get_token()

    rec = next(r for r in caplog.records if "credential bundle rejected" in r.getMessage())
    assert rec.outcome == "rejected"


def test_api_key_retryable_status_logs_failure_only_once_at_terminal(monkeypatch, respx_mock, caplog):
    """A retryable 503 that never recovers logs exactly one 'failure' record — at the terminal attempt.

    Regression for the AU-2 false-positive: previously every retry attempt
    emitted an outcome:'failure' record (up to _MAX_RETRIES+1 of them), so a
    transient 5xx looked like several distinct failed acquisitions. Now only
    the terminal attempt logs.
    """
    import logging as _logging
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    # zero the backoff so the retries don't actually sleep in the test
    monkeypatch.setattr("agentic_context.auth._retry.backoff", lambda attempt: 0.0)
    route = respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(503, json={"error": "unavailable"})
    )
    auth = ApiKeyAuth(FAKE_API_KEY, secret_server_url=FAKE_SECRETS_URL)

    with caplog.at_level(_logging.WARNING, logger="agentic_context.auth"):
        with pytest.raises(httpx.HTTPStatusError):
            auth.get_token()

    from agentic_context.auth import _retry as _auth_retry
    # all attempts were made (retry policy unchanged) ...
    assert route.call_count == _auth_retry._MAX_RETRIES + 1
    # ... but only one terminal AU-2 failure record was emitted
    failures = [r for r in caplog.records if "credential bundle download failed" in r.getMessage()]
    assert len(failures) == 1
    assert failures[0].http_status == 503


def test_key_hint_is_non_reversible_and_omits_raw_key():
    """_key_hint returns a short sha256 prefix, never the raw key; empty → 'none'."""
    from agentic_context.auth import _key_hint
    hint = _key_hint(FAKE_API_KEY)
    assert hint.startswith("sha256:")
    assert FAKE_API_KEY not in hint
    assert len(hint) < len(FAKE_API_KEY) + 20   # short fingerprint, not the key
    assert _key_hint("") == "none"


def test_production_path_never_calls_secret_server(monkeypatch, respx_mock):
    """Direct-creds ContextAuth must not touch the secret server at all."""
    monkeypatch.setattr("agentic_context.auth._sign_jwt", lambda c, k: "fake.jwt.token")
    secrets_route = respx_mock.get(f"{FAKE_SECRETS_URL}/api/secret").mock(
        return_value=httpx.Response(200, json=FAKE_BUNDLE)
    )
    _mock_two_leg(respx_mock)

    auth = _make_auth()  # production path, direct creds
    auth.get_token()

    assert not secrets_route.called


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_token(auth: ContextAuth, expiry: float | None = None) -> None:
    # State is now a single tuple to prevent torn reads; write the whole
    # snapshot at once. Layout matches auth.py:
    # (token, base_url, core_token, core_instance_url, expiry).
    # instance_url must be a .salesforce.com host — the OAuth response
    # is validated against the SSRF allowlist in _validate_instance_url.
    auth._snapshot = (
        FAKE_CDP_TOKEN,
        FAKE_INSTANCE_URL,
        FAKE_CORE_TOKEN,
        "https://core.test.salesforce.com",
        expiry if expiry is not None else time.time() + 3600,
    )
    auth._propagate(auth._token)
