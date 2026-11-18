# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for the login-state / refresh-token stores and ``UserContextAuth.from_store``."""
from __future__ import annotations

import os
import stat
import threading
import time

import httpx
import pytest

from agentic_context import (
    AuthorizationRequest,
    FileLoginStateStore,
    FileRefreshTokenStore,
    InMemoryLoginStateStore,
    InMemoryRefreshTokenStore,
    LoginStateStore,
    ReauthRequired,
    RefreshTokenStore,
    UserContextAuth,
)

LOGIN_URL = "login.test.salesforce.com"
TOKEN_ENDPOINT = f"https://{LOGIN_URL}/services/oauth2/token"
CORE_INSTANCE = "https://core.test.salesforce.com"
ID_URL = f"https://{LOGIN_URL}/id/00Dxx0000001gPFEAY/005xx000001Sv6DAAS"


def _request(state: str = "st1") -> AuthorizationRequest:
    return AuthorizationRequest(
        url=f"https://{LOGIN_URL}/services/oauth2/authorize?state={state}",
        state=state, code_verifier="verifier-secret",
        redirect_uri="https://chat.example.com/oauth/callback",
    )


@pytest.fixture(params=["memory", "file"])
def state_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryLoginStateStore(expiration_seconds=60)
    return FileLoginStateStore(tmp_path / "state", expiration_seconds=60)


@pytest.fixture(params=["memory", "file"])
def token_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryRefreshTokenStore()
    return FileRefreshTokenStore(tmp_path / "tokens")


# ── LoginStateStore ──────────────────────────────────────────────────────────

def test_state_store_satisfies_protocol(state_store):
    assert isinstance(state_store, LoginStateStore)


def test_state_roundtrip_keeps_verifier_and_redirect(state_store):
    req = _request()
    state_store.save(req)
    assert state_store.consume("st1") == req


def test_state_is_one_time_use(state_store):
    state_store.save(_request())
    assert state_store.consume("st1") is not None
    assert state_store.consume("st1") is None


def test_unknown_state_returns_none(state_store):
    assert state_store.consume("never-issued") is None


def test_expired_state_returns_none(tmp_path):
    for store in (InMemoryLoginStateStore(expiration_seconds=0.05),
                  FileLoginStateStore(tmp_path, expiration_seconds=0.05)):
        store.save(_request())
        time.sleep(0.1)
        assert store.consume("st1") is None


def test_state_ids_are_independent(state_store):
    state_store.save(_request("a"))
    state_store.save(_request("b"))
    assert state_store.consume("b").state == "b"
    assert state_store.consume("a").state == "a"


def test_concurrent_consume_has_exactly_one_winner(state_store):
    state_store.save(_request())
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(state_store.consume("st1"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r is not None for r in results) == 1


def test_file_state_store_purges_expired_on_save(tmp_path):
    store = FileLoginStateStore(tmp_path, expiration_seconds=0.05)
    store.save(_request("old"))
    time.sleep(0.1)
    store.save(_request("new"))
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_file_state_store_does_not_use_state_in_filename(tmp_path):
    store = FileLoginStateStore(tmp_path)
    store.save(_request("../../etc/passwd"))
    assert all("passwd" not in p.name for p in tmp_path.iterdir())
    assert store.consume("../../etc/passwd") is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_file_stores_use_private_permissions(tmp_path):
    states = FileLoginStateStore(tmp_path / "s")
    states.save(_request())
    tokens = FileRefreshTokenStore(tmp_path / "t")
    tokens.save("00D1", "005A", "rt")
    for d in (tmp_path / "s", tmp_path / "t"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
        for f in d.iterdir():
            assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_state_repr_hides_verifier():
    assert "verifier-secret" not in repr(_request())


# ── RefreshTokenStore ────────────────────────────────────────────────────────

def test_token_store_satisfies_protocol(token_store):
    assert isinstance(token_store, RefreshTokenStore)


def test_token_roundtrip_replace_delete(token_store):
    assert token_store.load("00D1", "005A") is None
    token_store.save("00D1", "005A", "rt1")
    assert token_store.load("00D1", "005A") == "rt1"
    token_store.save("00D1", "005A", "rt2")
    assert token_store.load("00D1", "005A") == "rt2"
    token_store.delete("00D1", "005A")
    assert token_store.load("00D1", "005A") is None
    token_store.delete("00D1", "005A")  # absent: no error


def test_tokens_are_keyed_by_org_and_user(token_store):
    token_store.save("00D1", "005A", "a")
    token_store.save("00D1", "005B", "b")
    token_store.save("00D2", "005A", "c")
    assert [token_store.load("00D1", "005A"), token_store.load("00D1", "005B"),
            token_store.load("00D2", "005A")] == ["a", "b", "c"]


@pytest.mark.parametrize("org, user", [("", "005A"), ("00D1", "")])
def test_token_store_rejects_empty_keys(token_store, org, user):
    with pytest.raises(ValueError):
        token_store.save(org, user, "rt")
    with pytest.raises(ValueError):
        token_store.load(org, user)


def test_token_store_rejects_empty_token(token_store):
    with pytest.raises(ValueError):
        token_store.save("00D1", "005A", "")


def test_file_token_store_survives_new_instance(tmp_path):
    FileRefreshTokenStore(tmp_path).save("00D1", "005A", "rt")
    assert FileRefreshTokenStore(tmp_path).load("00D1", "005A") == "rt"


@pytest.mark.parametrize("content", [
    "",                       # empty / truncated write
    '{"refresh_token": "rt',  # partial JSON
    "not json at all",
    "[1, 2, 3]",              # valid JSON, wrong shape
    "null",
])
def test_file_token_store_corrupt_file_loads_as_absent(tmp_path, content):
    """A corrupt file means 'no token' (caller re-auths), not a crash mid-refresh."""
    store = FileRefreshTokenStore(tmp_path)
    store.save("00D1", "005A", "rt")
    (path,) = tmp_path.glob("*.json")
    path.write_text(content)

    assert store.load("00D1", "005A") is None

    store.save("00D1", "005A", "fresh")          # and the slot is recoverable
    assert store.load("00D1", "005A") == "fresh"


def test_file_token_store_unreadable_file_loads_as_absent(tmp_path):
    store = FileRefreshTokenStore(tmp_path)
    store.save("00D1", "005A", "rt")
    (path,) = tmp_path.glob("*.json")
    path.write_bytes(b"\xff\xfe\x00bad")        # not valid UTF-8

    assert store.load("00D1", "005A") is None


def test_file_token_store_ids_cannot_escape_directory(tmp_path):
    store = FileRefreshTokenStore(tmp_path / "t")
    store.save("../../x", "../y", "rt")
    assert {p.parent for p in tmp_path.rglob("*.json")} == {tmp_path / "t"}


# ── UserContextAuth.from_store ───────────────────────────────────────────────

def _auth(store, **kw):
    return UserContextAuth.from_store(
        store, org_id="00D1", user_id="005A", consumer_key="3MVGtest",
        consumer_secret="s", login_url=LOGIN_URL, **kw,
    )


def test_from_store_refreshes_with_stored_token(token_store, respx_mock):
    token_store.save("00D1", "005A", "stored_rt")
    leg1 = respx_mock.post(TOKEN_ENDPOINT).mock(return_value=httpx.Response(200, json={
        "access_token": "core", "instance_url": CORE_INSTANCE, "id": ID_URL}))
    respx_mock.post(f"{CORE_INSTANCE}/services/a360/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "cdp", "instance_url": "https://beacon.test.salesforce.com"}))

    assert _auth(token_store).get_token()[0] == "cdp"
    assert b"refresh_token=stored_rt" in leg1.calls.last.request.content


def test_from_store_persists_rotated_token(token_store, respx_mock):
    token_store.save("00D1", "005A", "old_rt")
    respx_mock.post(TOKEN_ENDPOINT).mock(return_value=httpx.Response(200, json={
        "access_token": "core", "instance_url": CORE_INSTANCE, "id": ID_URL,
        "refresh_token": "new_rt"}))
    respx_mock.post(f"{CORE_INSTANCE}/services/a360/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "cdp", "instance_url": "https://beacon.test.salesforce.com"}))

    _auth(token_store).get_token()
    assert token_store.load("00D1", "005A") == "new_rt"


def test_from_store_deletes_dead_token_on_reauth(token_store, respx_mock):
    token_store.save("00D1", "005A", "revoked_rt")
    respx_mock.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"}))

    with pytest.raises(ReauthRequired):
        _auth(token_store).get_token()
    assert token_store.load("00D1", "005A") is None


async def test_from_store_async_deletes_dead_token_on_reauth(token_store, respx_mock):
    token_store.save("00D1", "005A", "revoked_rt")
    respx_mock.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"}))

    with pytest.raises(ReauthRequired):
        await _auth(token_store).aget_token()
    assert token_store.load("00D1", "005A") is None


def test_delete_if_token_only_removes_matching_token(token_store):
    token_store.save("00D1", "005A", "fresh")
    token_store.delete("00D1", "005A", if_token="stale")
    assert token_store.load("00D1", "005A") == "fresh"
    token_store.delete("00D1", "005A", if_token="fresh")
    assert token_store.load("00D1", "005A") is None


def test_delete_is_audited_without_the_token(token_store, caplog):
    token_store.save("00D1", "005A", "secret_rt_value")
    with caplog.at_level("INFO", logger="agentic_context.stores"):
        token_store.delete("00D1", "005A", if_token="other")
        token_store.delete("00D1", "005A", if_token="secret_rt_value")
        token_store.delete("00D1", "005A")
    outcomes = [r.outcome for r in caplog.records if getattr(r, "audit_event", "") == "refresh_token_delete"]
    assert outcomes == ["skipped_token_changed", "deleted", "absent"]
    assert "secret_rt_value" not in caplog.text and "other" not in caplog.text
    assert "00D1" in caplog.text and "005A" in caplog.text


def test_from_store_keeps_fresh_login_saved_after_rejection(token_store, respx_mock):
    # Final reload sees the rejected token; a new login lands before cleanup.
    token_store.save("00D1", "005A", "revoked_rt")
    respx_mock.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"}))
    auth = _auth(token_store)
    real_discard = auth._discard_grant

    def discard_after_new_login():
        token_store.save("00D1", "005A", "fresh_login_rt")
        real_discard()

    auth._discard_grant = discard_after_new_login
    with pytest.raises(ReauthRequired):
        auth.get_token()
    assert token_store.load("00D1", "005A") == "fresh_login_rt"


def test_from_store_with_nothing_stored_raises_reauth(token_store, respx_mock):
    route = respx_mock.post(TOKEN_ENDPOINT)
    with pytest.raises(ReauthRequired):
        _auth(token_store).get_token()
    assert not route.called


def test_failing_delete_does_not_mask_reauth(respx_mock):
    class Broken(InMemoryRefreshTokenStore):
        def delete(self, org_id, user_id):
            raise RuntimeError("db down")

    store = Broken()
    store.save("00D1", "005A", "rt")
    respx_mock.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"}))
    with pytest.raises(ReauthRequired):
        _auth(store).get_token()
