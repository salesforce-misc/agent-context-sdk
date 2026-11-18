# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Unit tests for the user-login (refresh_token) auth path — UserContextAuth
and the module-level ``_exchange_refresh`` / ``_exchange_refresh_async`` helpers.

Mocks both HTTP legs with respx; no real credentials. Mirrors the style of
``test_context_auth.py`` (which covers the JWT-bearer ``ContextAuth`` path).

The behaviors pinned here are the ones flagged as ship blockers in review:

* no ``xid`` is ever sent — the core token already belongs to the signed-in
  employee (xid is only for the integration-user model of ``ContextAuth``);
* the resolved ``005…`` is still exposed as ``user_id``;
* ``aid`` is sent when set and omitted when unset;
* ``login_url`` and the core ``instance_url`` are host-allowlisted BEFORE any
  credential-bearing POST (SSRF / token-exfil guard, same as #56);
* a 200-without-``access_token`` from a360 retries, then raises a diagnostic
  that surfaces only a360's non-secret fields.
"""
from __future__ import annotations

import time
from urllib.parse import parse_qs

import httpx
import pytest

from agentic_context.auth import (
    _A360_EXCHANGE_RETRIES,
    ReauthRequired,
    UserContextAuth,
    UserLogin,
    _exchange_refresh,
    _exchange_refresh_async,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_LOGIN_URL = "login.test.salesforce.com"
FAKE_CONSUMER_KEY = "3MVGtest"
FAKE_CONSUMER_SECRET = "consumer_secret_shhh"
FAKE_REFRESH_TOKEN = "refresh_token_abc"
FAKE_CORE_TOKEN = "core_access_token_abc"
FAKE_CDP_TOKEN = "cdp_bearer_token_xyz"
FAKE_CORE_INSTANCE = "https://core.test.salesforce.com"
FAKE_INSTANCE_URL = "https://beacon.test.salesforce.com"
FAKE_USER_ID = "005xx000001Sv6DAAS"
FAKE_ID_URL = f"https://{FAKE_LOGIN_URL}/id/00Dxx0000001gPFEAY/{FAKE_USER_ID}"

_A360_ENDPOINT = f"{FAKE_CORE_INSTANCE}/services/a360/token"


def _mock_refresh_leg(respx_mock, *, id_url=FAKE_ID_URL, instance_url=FAKE_CORE_INSTANCE):
    """Mock leg-1 (refresh_token grant)."""
    body = {"access_token": FAKE_CORE_TOKEN, "instance_url": instance_url}
    if id_url is not None:
        body["id"] = id_url
    return respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json=body)
    )


def _mock_a360_leg(respx_mock, **override):
    """Mock leg-2 (a360 CDP exchange), success by default."""
    body = {"access_token": FAKE_CDP_TOKEN, "instance_url": FAKE_INSTANCE_URL}
    body.update(override)
    return respx_mock.post(_A360_ENDPOINT).mock(
        return_value=httpx.Response(200, json=body)
    )


def _a360_form(route) -> dict[str, str]:
    """Flatten the last a360 request's urlencoded form body to {key: value}."""
    parsed = parse_qs(route.calls.last.request.content.decode())
    return {k: v[0] for k, v in parsed.items()}


async def _run_refresh(is_async: bool, login_url: str = FAKE_LOGIN_URL, **kwargs):
    """Run ``_exchange_refresh`` or ``_exchange_refresh_async`` with the fake creds."""
    args = (login_url, FAKE_REFRESH_TOKEN, FAKE_CONSUMER_KEY, FAKE_CONSUMER_SECRET, 3600)
    if is_async:
        async with httpx.AsyncClient() as client:
            return await _exchange_refresh_async(client, *args, **kwargs)
    with httpx.Client() as client:
        return _exchange_refresh(client, *args, **kwargs)


# ── Happy path ─────────────────────────────────────────────────────────────────

def test_refresh_exchange_returns_tokens_and_user_id(respx_mock):
    _mock_refresh_leg(respx_mock)
    _mock_a360_leg(respx_mock)

    with httpx.Client() as client:
        (token, base_url, expiry, core_token, core_instance_url, user_id) = _exchange_refresh(
            client, FAKE_LOGIN_URL, FAKE_REFRESH_TOKEN,
            FAKE_CONSUMER_KEY, FAKE_CONSUMER_SECRET, 3600,
        )

    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL
    assert core_token == FAKE_CORE_TOKEN
    assert core_instance_url == FAKE_CORE_INSTANCE
    assert user_id == FAKE_USER_ID
    assert expiry > time.time()


def test_refresh_leg_one_sends_refresh_grant_and_secret(respx_mock):
    leg1 = _mock_refresh_leg(respx_mock)
    _mock_a360_leg(respx_mock)

    with httpx.Client() as client:
        _exchange_refresh(
            client, FAKE_LOGIN_URL, FAKE_REFRESH_TOKEN,
            FAKE_CONSUMER_KEY, FAKE_CONSUMER_SECRET, 3600,
        )

    form = {k: v[0] for k, v in parse_qs(leg1.calls.last.request.content.decode()).items()}
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == FAKE_REFRESH_TOKEN
    assert form["client_id"] == FAKE_CONSUMER_KEY
    assert form["client_secret"] == FAKE_CONSUMER_SECRET


# ── No xid: the token is the employee's own ─────────────────────────────────

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("aid", ["agent_1", None], ids=["aid-set", "aid-unset"])
async def test_refresh_never_sends_xid_and_sends_aid_only_when_set(respx_mock, is_async, aid):
    """The core token response carries the user's 005, but it is NOT sent as
    xid — the employee's own token needs no external-id stand-in. ``aid`` is
    sent when set; no aid → neither aid nor xid is sent."""
    _mock_refresh_leg(respx_mock)
    a360 = _mock_a360_leg(respx_mock)

    kwargs = {} if aid is None else {"aid": aid}
    *_, user_id = await _run_refresh(is_async, **kwargs)

    assert user_id == FAKE_USER_ID
    form = _a360_form(a360)
    assert "xid" not in form
    assert form.get("aid") == aid
    assert form["grant_type"] == "urn:salesforce:grant-type:external:cdp"


def test_user_context_auth_has_no_xid_parameter():
    with pytest.raises(TypeError):
        UserContextAuth(
            refresh_token=FAKE_REFRESH_TOKEN, consumer_key=FAKE_CONSUMER_KEY,
            login_url=FAKE_LOGIN_URL, xid="005EXPLICITxxxxxAAA",
        )


# ── URL allowlisting BEFORE the credential POST (ship blocker) ────────────────

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("login_url", "core_instance_url", "match"),
    [
        # A non-*.salesforce.com login host is refused before the refresh token +
        # client secret are ever POSTed.
        ("attacker.example.com", None, "allowlist"),
        # Leg-1 core instance_url off-allowlist → ValueError before the a360 POST.
        (FAKE_LOGIN_URL, "https://attacker.example.com",
         "core instance_url host outside the allowlist"),
    ],
    ids=["untrusted-login-host", "untrusted-core-instance-url"],
)
async def test_refresh_rejects_untrusted_url_before_credential_post(
    respx_mock, is_async, login_url, core_instance_url, match
):
    if core_instance_url is None:
        leg = respx_mock.post(f"https://{login_url}/services/oauth2/token").mock(
            return_value=httpx.Response(200, json={"access_token": "x", "instance_url": "https://x"})
        )
    else:
        _mock_refresh_leg(respx_mock, instance_url=core_instance_url)
        leg = respx_mock.post(_A360_ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(ValueError, match=match):
        await _run_refresh(is_async, login_url=login_url)
    assert not leg.called


# ── a360 200-without-access_token (retry + non-secret diagnostic) ─────────────

def test_refresh_200_without_access_token_retries_then_raises_diagnostic(monkeypatch, respx_mock):
    """A persistent 200-with-error surfaces a360's non-secret fields, retries the
    bounded number of times, and never echoes a secret value."""
    monkeypatch.setattr("agentic_context.auth._A360_EXCHANGE_BACKOFF_S", 0.0)
    _mock_refresh_leg(respx_mock)
    a360 = respx_mock.post(_A360_ENDPOINT).mock(
        return_value=httpx.Response(200, json={
            "error": "access_denied",
            "error_description": "unknown aid",
            "refresh_token": "SECRET_SHOULD_NOT_LEAK",  # non-whitelisted → must not appear
        })
    )

    with httpx.Client() as client, pytest.raises(ValueError, match="no access_token") as excinfo:
        _exchange_refresh(
            client, FAKE_LOGIN_URL, FAKE_REFRESH_TOKEN,
            FAKE_CONSUMER_KEY, FAKE_CONSUMER_SECRET, 3600,
        )

    msg = str(excinfo.value)
    assert "access_denied" in msg
    assert "unknown aid" in msg
    assert "SECRET_SHOULD_NOT_LEAK" not in msg
    assert a360.call_count == _A360_EXCHANGE_RETRIES + 1


# ── UserContextAuth (class-level integration: caller + helper) ────────────────

def test_user_context_auth_get_token_exposes_user_id_without_xid(respx_mock):
    """End-to-end through the class: get_token() fetches and exposes user_id;
    the a360 body carries no xid."""
    _mock_refresh_leg(respx_mock)
    a360 = _mock_a360_leg(respx_mock)

    auth = UserContextAuth(
        refresh_token=FAKE_REFRESH_TOKEN,
        consumer_key=FAKE_CONSUMER_KEY,
        consumer_secret=FAKE_CONSUMER_SECRET,
        login_url=FAKE_LOGIN_URL,
    )
    token, base_url = auth.get_token()

    assert token == FAKE_CDP_TOKEN
    assert base_url == FAKE_INSTANCE_URL
    assert auth.user_id == FAKE_USER_ID
    assert "xid" not in _a360_form(a360)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"refresh_token": "", "consumer_secret": FAKE_CONSUMER_SECRET}, "non-empty refresh_token"),
        ({}, "load_refresh_token"),
    ],
    ids=["empty-refresh-token", "no-refresh-token-or-loader"],
)
def test_user_context_auth_requires_refresh_token_or_loader(kwargs, match):
    with pytest.raises(ValueError, match=match):
        UserContextAuth(consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL, **kwargs)


# ── Public client (consumer key only) + refresh-token rotation ───────────────

def _leg1_form(route) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(route.calls.last.request.content.decode()).items()}


def test_public_client_refresh_omits_client_secret(respx_mock):
    leg1 = _mock_refresh_leg(respx_mock)
    _mock_a360_leg(respx_mock)

    with httpx.Client() as client:
        _exchange_refresh(
            client, FAKE_LOGIN_URL, FAKE_REFRESH_TOKEN,
            FAKE_CONSUMER_KEY, None, 3600,
        )

    form = _leg1_form(leg1)
    assert form["grant_type"] == "refresh_token"
    assert form["client_id"] == FAKE_CONSUMER_KEY
    assert "client_secret" not in form


@pytest.mark.parametrize(
    ("returned_refresh_token", "expected_reported"),
    [("rotated_refresh_token", ["rotated_refresh_token"]), (FAKE_REFRESH_TOKEN, [])],
    ids=["rotated-reported-and-used-next", "unchanged-not-reported"],
)
def test_refresh_token_rotation_is_reported_and_used_next(
    respx_mock, returned_refresh_token, expected_reported
):
    rotated: list[str] = []
    leg1 = respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN,
            "instance_url": FAKE_CORE_INSTANCE,
            "id": FAKE_ID_URL,
            "refresh_token": returned_refresh_token,
        })
    )
    _mock_a360_leg(respx_mock)

    auth = UserContextAuth(
        refresh_token=FAKE_REFRESH_TOKEN,
        consumer_key=FAKE_CONSUMER_KEY,
        login_url=FAKE_LOGIN_URL,
        on_refresh_token=rotated.append,
    )
    auth.get_token()
    assert rotated == expected_reported

    auth.invalidate()
    auth.get_token()
    assert _leg1_form(leg1)["refresh_token"] == returned_refresh_token


# ── UserContextAuth.login (PKCE browser login) ───────────────────────────────

def test_login_builds_public_client_auth(monkeypatch, respx_mock):
    seen: dict = {}
    saved: list[str] = []

    def fake_pkce_login(**kwargs):
        seen.update(kwargs)
        return {"refresh_token": "login_refresh", "access_token": "x", "id": FAKE_ID_URL}

    monkeypatch.setattr("agentic_context._pkce.pkce_login", fake_pkce_login)
    a360 = _mock_a360_leg(respx_mock)
    respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN,
            "instance_url": FAKE_CORE_INSTANCE,
            "id": FAKE_ID_URL,
        })
    )

    auth = UserContextAuth.login(
        consumer_key=FAKE_CONSUMER_KEY,
        login_url=FAKE_LOGIN_URL,
        aid="gov_agent_1",
        on_refresh_token=saved.append,
    )

    assert seen["consumer_key"] == FAKE_CONSUMER_KEY
    assert seen["login_url"] == FAKE_LOGIN_URL
    assert "cdp_ingest_api" in seen["scope"]
    assert saved == ["login_refresh"]
    assert auth.user_id == FAKE_USER_ID

    token, base_url = auth.get_token()
    assert (token, base_url) == (FAKE_CDP_TOKEN, FAKE_INSTANCE_URL)
    form = _a360_form(a360)
    assert form["aid"] == "gov_agent_1"
    assert "xid" not in form


def test_login_rejects_non_salesforce_host_before_opening_browser(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("browser flow must not start")

    monkeypatch.setattr("agentic_context._pkce.pkce_login", boom)
    with pytest.raises(ValueError, match="allowlist"):
        UserContextAuth.login(consumer_key=FAKE_CONSUMER_KEY, login_url="evil.example.com")


def _drive_callback(query_for):
    """open_browser stand-in: parse the authorize URL, then hit the loopback."""
    import threading
    import urllib.request
    from urllib.parse import parse_qs as _pq, urlparse

    captured: dict = {}

    def open_browser(url: str) -> None:
        q = {k: v[0] for k, v in _pq(urlparse(url).query).items()}
        captured.update(q)

        def hit() -> None:
            # query_for returns one query string, or a list sent in order
            # (e.g. a stray request before / a duplicate after the real one).
            queries = query_for(q)
            for query in [queries] if isinstance(queries, str) else queries:
                try:
                    urllib.request.urlopen(
                        f"{q['redirect_uri']}?{query}", timeout=5
                    ).read()
                except OSError:
                    # HTTPError (400 for a stray request), connection refused,
                    # or connection reset (listener shut down after the first
                    # valid one, possibly mid-request): all expected outcomes
                    # for the extra requests.
                    pass

        threading.Thread(target=hit, daemon=True).start()

    return open_browser, captured


def test_pkce_login_redeems_code_with_verifier(respx_mock):
    from agentic_context import _pkce

    token_route = respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "a", "refresh_token": "r", "id": FAKE_ID_URL,
        })
    )
    open_browser, captured = _drive_callback(lambda q: f"code=abc&state={q['state']}")

    out = _pkce.pkce_login(
        login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
        port=18931, timeout=10, open_browser=open_browser,
    )

    assert out["refresh_token"] == "r"
    assert captured["response_type"] == "code"
    assert captured["code_challenge_method"] == "S256"
    assert captured["redirect_uri"] == "http://localhost:18931/callback"
    form = _leg1_form(token_route)
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "abc"
    assert "client_secret" not in form
    # The verifier sent to /token must hash to the challenge sent to /authorize.
    assert _pkce._challenge(form["code_verifier"]) == captured["code_challenge"]


def test_pkce_login_ignores_callback_without_matching_state(respx_mock):
    """A forged-state request is refused, not treated as the answer: the login
    keeps waiting and (with nothing valid arriving) times out."""
    from agentic_context import _pkce

    open_browser, _ = _drive_callback(lambda q: "code=abc&state=forged")
    with pytest.raises(TimeoutError):
        _pkce.pkce_login(
            login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
            port=18932, timeout=1, open_browser=open_browser,
        )


def test_pkce_login_stray_request_before_real_one_does_not_break_login(respx_mock):
    from agentic_context import _pkce

    token_route = respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "a", "refresh_token": "r", "id": FAKE_ID_URL,
        })
    )
    open_browser, _ = _drive_callback(lambda q: [
        "error=access_denied&state=forged",     # stray, wrong state
        "code=real&state=" + q["state"],        # the genuine redirect
    ])

    out = _pkce.pkce_login(
        login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
        port=18934, timeout=10, open_browser=open_browser,
    )

    assert out["refresh_token"] == "r"
    assert _leg1_form(token_route)["code"] == "real"


def test_pkce_login_duplicate_callback_cannot_overwrite_captured_code(respx_mock):
    from agentic_context import _pkce

    token_route = respx_mock.post(f"https://{FAKE_LOGIN_URL}/services/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "a", "refresh_token": "r", "id": FAKE_ID_URL,
        })
    )
    # Even a duplicate carrying the CORRECT state must not replace the first code.
    open_browser, _ = _drive_callback(lambda q: [
        "code=first&state=" + q["state"],
        "code=second&state=" + q["state"],
    ])

    _pkce.pkce_login(
        login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
        port=18935, timeout=10, open_browser=open_browser,
    )

    assert _leg1_form(token_route)["code"] == "first"


def test_pkce_login_surfaces_authorize_error(respx_mock):
    from agentic_context import _pkce

    open_browser, _ = _drive_callback(
        lambda q: f"error=access_denied&error_description=nope&state={q['state']}"
    )
    with pytest.raises(ValueError, match="access_denied"):
        _pkce.pkce_login(
            login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
            port=18933, timeout=10, open_browser=open_browser,
        )


# ── Backend-hosted login: authorization_request + redeem_code ────────────────

FAKE_REDIRECT = "https://chat.example.com/oauth/callback"
FAKE_ORG_ID = "00Dxx0000001gPFEAY"
_TOKEN_ENDPOINT = f"https://{FAKE_LOGIN_URL}/services/oauth2/token"


def _code_grant_response(**override):
    body = {
        "access_token": FAKE_CORE_TOKEN,
        "refresh_token": "server_refresh",
        "instance_url": FAKE_CORE_INSTANCE,
        "id": FAKE_ID_URL,
    }
    body.update(override)
    return httpx.Response(200, json=body)


def test_authorization_request_builds_pkce_authorize_url():
    from urllib.parse import urlparse

    from agentic_context import _pkce

    req = UserContextAuth.authorization_request(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL,
        redirect_uri=FAKE_REDIRECT,
    )
    parsed = urlparse(req.url)
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert parsed.netloc == FAKE_LOGIN_URL
    assert parsed.path == "/services/oauth2/authorize"
    assert q["response_type"] == "code"
    assert q["client_id"] == FAKE_CONSUMER_KEY
    assert q["redirect_uri"] == FAKE_REDIRECT
    assert q["state"] == req.state
    assert q["code_challenge_method"] == "S256"
    assert q["code_challenge"] == _pkce._challenge(req.code_verifier)
    assert "cdp_ingest_api" in q["scope"]
    # The verifier is a secret: never in the URL, never in repr.
    assert req.code_verifier not in req.url
    assert req.code_verifier not in repr(req)


def test_authorization_request_is_fresh_each_call():
    a = UserContextAuth.authorization_request(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL, redirect_uri=FAKE_REDIRECT,
    )
    b = UserContextAuth.authorization_request(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL, redirect_uri=FAKE_REDIRECT,
    )
    assert a.state != b.state
    assert a.code_verifier != b.code_verifier


@pytest.mark.parametrize("redirect_uri", [
    "http://chat.example.com/oauth/callback",   # plaintext off-loopback
    "/oauth/callback",                          # relative
    "https://chat.example.com/cb#frag",         # fragment
    "https://a:b@evil.com/cb",                  # userinfo
    "https://@evil.com/cb",                     # empty userinfo
])
def test_authorization_request_rejects_bad_redirect_uri(redirect_uri):
    with pytest.raises(ValueError, match="redirect_uri"):
        UserContextAuth.authorization_request(
            consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL, redirect_uri=redirect_uri,
        )


def test_authorization_request_allows_http_loopback():
    req = UserContextAuth.authorization_request(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL,
        redirect_uri="http://localhost:8000/oauth/callback",
    )
    assert req.redirect_uri == "http://localhost:8000/oauth/callback"


def test_authorization_request_rejects_untrusted_login_host():
    with pytest.raises(ValueError, match="allowlist"):
        UserContextAuth.authorization_request(
            consumer_key=FAKE_CONSUMER_KEY, login_url="evil.example.com",
            redirect_uri=FAKE_REDIRECT,
        )


def _redeem_kwargs(**override):
    kw = dict(
        code="auth_code", state="st", expected_state="st", code_verifier="ver",
        redirect_uri=FAKE_REDIRECT, consumer_key=FAKE_CONSUMER_KEY,
        consumer_secret=FAKE_CONSUMER_SECRET, login_url=FAKE_LOGIN_URL,
    )
    kw.update(override)
    return kw


async def _redeem(is_async: bool, **override):
    """Call ``redeem_code`` or ``await aredeem_code`` with ``_redeem_kwargs``."""
    kw = _redeem_kwargs(**override)
    if is_async:
        return await UserContextAuth.aredeem_code(**kw)
    return UserContextAuth.redeem_code(**kw)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_redeem_code_sends_secret_and_verifier_and_returns_login(respx_mock, is_async):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())

    login = await _redeem(is_async)

    assert login == UserLogin(
        refresh_token="server_refresh", user_id=FAKE_USER_ID,
        org_id=FAKE_ORG_ID, instance_url=FAKE_CORE_INSTANCE,
    )
    assert "server_refresh" not in repr(login)
    form = _leg1_form(route)
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "auth_code"
    assert form["code_verifier"] == "ver"
    assert form["redirect_uri"] == FAKE_REDIRECT
    assert form["client_secret"] == FAKE_CONSUMER_SECRET


def test_redeem_code_public_client_omits_secret(respx_mock):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())
    UserContextAuth.redeem_code(**_redeem_kwargs(consumer_secret=None))
    assert "client_secret" not in _leg1_form(route)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "state, expected",
    [("forged", "st"), ("", "st"), ("st", "")],
    ids=["forged", "empty-state", "empty-expected"],
)
async def test_redeem_code_rejects_state_mismatch_before_any_post(respx_mock, is_async, state, expected):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())
    with pytest.raises(ValueError, match="state mismatch"):
        await _redeem(is_async, state=state, expected_state=expected)
    assert not route.called


def test_redeem_code_rejects_missing_code(respx_mock):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())
    with pytest.raises(ValueError, match="no authorization code"):
        UserContextAuth.redeem_code(**_redeem_kwargs(code=""))
    assert not route.called


def test_redeem_code_rejects_untrusted_login_host(respx_mock):
    with pytest.raises(ValueError, match="allowlist"):
        UserContextAuth.redeem_code(**_redeem_kwargs(login_url="evil.example.com"))


def test_redeem_code_invalid_grant_raises_reauth_required(respx_mock):
    respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=httpx.Response(
        400, json={"error": "invalid_grant", "error_description": "invalid authorization code"},
    ))
    with pytest.raises(ReauthRequired) as exc:
        UserContextAuth.redeem_code(**_redeem_kwargs())
    assert exc.value.error == "invalid_grant"
    assert exc.value.description == "invalid authorization code"


def test_redeem_code_without_refresh_token_raises(respx_mock):
    resp = _code_grant_response()
    body = resp.json()
    del body["refresh_token"]
    respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ValueError, match="refresh_token"):
        UserContextAuth.redeem_code(**_redeem_kwargs())


def test_redeem_code_rejects_untrusted_instance_url(respx_mock):
    respx_mock.post(_TOKEN_ENDPOINT).mock(
        return_value=_code_grant_response(instance_url="https://evil.example.com"),
    )
    with pytest.raises(ValueError, match="allowlist"):
        UserContextAuth.redeem_code(**_redeem_kwargs())


# ── ReauthRequired on refresh ────────────────────────────────────────────────

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_refresh_invalid_grant_raises_reauth_required_without_retry(respx_mock, is_async):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=httpx.Response(
        400, json={"error": "invalid_grant", "error_description": "expired access/refresh token"},
    ))
    auth = UserContextAuth(
        refresh_token=FAKE_REFRESH_TOKEN, consumer_key=FAKE_CONSUMER_KEY,
        login_url=FAKE_LOGIN_URL,
    )
    with pytest.raises(ReauthRequired, match="expired access/refresh token") as exc:
        await auth.aget_token() if is_async else auth.get_token()
    assert exc.value.error == "invalid_grant"
    assert route.call_count == 1  # never retried
    assert FAKE_REFRESH_TOKEN not in str(exc.value)


@pytest.mark.parametrize(
    ("response", "secret", "expected_exc"),
    [
        (httpx.Response(401, text="nope"), None, ReauthRequired),
        # A wrong consumer key/secret is an ECA misconfiguration, not a dead grant:
        # sending the user through login again would not fix it.
        (httpx.Response(400, json={"error": "invalid_client",
                                   "error_description": "invalid client credentials"}),
         "bad", httpx.HTTPStatusError),
    ],
    ids=["bare-401-reauth-required", "invalid-client-stays-http-status-error"],
)
def test_refresh_leg_one_error_classification(respx_mock, response, secret, expected_exc):
    respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=response)
    with httpx.Client() as client, pytest.raises(expected_exc):
        _exchange_refresh(
            client, FAKE_LOGIN_URL, FAKE_REFRESH_TOKEN, FAKE_CONSUMER_KEY, secret, 3600,
        )


# ── load_refresh_token (shared store across replicas) ────────────────────────

@pytest.mark.parametrize(
    ("constructor_token", "stored", "expected_sent"),
    [(None, "stored_token", "stored_token"), (FAKE_REFRESH_TOKEN, None, FAKE_REFRESH_TOKEN)],
    ids=["loader-replaces-constructor-token", "loader-none-falls-back-to-constructor"],
)
def test_load_refresh_token_precedence(respx_mock, constructor_token, stored, expected_sent):
    leg1 = _mock_refresh_leg(respx_mock)
    _mock_a360_leg(respx_mock)
    kwargs = {} if constructor_token is None else {"refresh_token": constructor_token}
    UserContextAuth(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL,
        load_refresh_token=lambda: stored, **kwargs,
    ).get_token()
    assert _leg1_form(leg1)["refresh_token"] == expected_sent


def test_no_stored_token_raises_reauth_required_without_post(respx_mock):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())
    auth = UserContextAuth(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL,
        load_refresh_token=lambda: None,
    )
    with pytest.raises(ReauthRequired) as exc:
        auth.get_token()
    assert exc.value.error == "no_refresh_token"
    assert not route.called


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_invalid_grant_retries_once_with_token_rotated_by_another_replica(respx_mock, is_async):
    # Replica A loaded "old"; replica B rotated it to "new" before A's grant.
    store = {"rt": "old"}
    loads: list[str] = []

    def load() -> str:
        loads.append(store["rt"])
        value = store["rt"]
        store["rt"] = "new"  # B's rotation lands right after A's first read
        return value

    def leg1(request):
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if form["refresh_token"] == "old":
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(200, json={
            "access_token": FAKE_CORE_TOKEN, "instance_url": FAKE_CORE_INSTANCE,
            "id": FAKE_ID_URL,
        })

    route = respx_mock.post(_TOKEN_ENDPOINT).mock(side_effect=leg1)
    _mock_a360_leg(respx_mock)

    auth = UserContextAuth(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL, load_refresh_token=load,
    )
    token = await auth.aget_token() if is_async else auth.get_token()
    assert token == (FAKE_CDP_TOKEN, FAKE_INSTANCE_URL)
    assert loads == ["old", "new"]
    assert route.call_count == 2


def test_retry_after_leg2_failure_keeps_rotated_token_when_store_lags(respx_mock):
    # Leg 1 rotates "old" -> "new" (old is now dead). The store never received
    # "new" and still serves "old"; the retry must not fall back to it.
    def leg1(request):
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if form["refresh_token"] == "old":
            return httpx.Response(200, json={
                "access_token": FAKE_CORE_TOKEN, "instance_url": FAKE_CORE_INSTANCE,
                "id": FAKE_ID_URL, "refresh_token": "new",
            })
        if form["refresh_token"] == "new":
            return httpx.Response(200, json={
                "access_token": FAKE_CORE_TOKEN, "instance_url": FAKE_CORE_INSTANCE,
                "id": FAKE_ID_URL,
            })
        return httpx.Response(400, json={"error": "invalid_grant"})

    leg1_route = respx_mock.post(_TOKEN_ENDPOINT).mock(side_effect=leg1)
    respx_mock.post(_A360_ENDPOINT).mock(side_effect=[
        httpx.Response(503),
        httpx.Response(200, json={
            "access_token": FAKE_CDP_TOKEN, "instance_url": FAKE_INSTANCE_URL,
            "expires_in": 3600,
        }),
    ])
    auth = UserContextAuth(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL,
        load_refresh_token=lambda: "old",
    )
    assert auth.get_token() == (FAKE_CDP_TOKEN, FAKE_INSTANCE_URL)
    sent = [parse_qs(c.request.content.decode())["refresh_token"][0] for c in leg1_route.calls]
    assert sent == ["old", "new"]


def test_invalid_grant_with_unchanged_stored_token_raises(respx_mock):
    route = respx_mock.post(_TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"}),
    )
    auth = UserContextAuth(
        consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL,
        load_refresh_token=lambda: "revoked",
    )
    with pytest.raises(ReauthRequired):
        auth.get_token()
    assert route.call_count == 1


def test_new_names_exported_from_package():
    import agentic_context as ac

    for name in ("AuthorizationRequest", "UserLogin", "ReauthRequired"):
        assert name in ac.__all__
        assert getattr(ac, name) is getattr(__import__("agentic_context.auth", fromlist=[name]), name)


def test_pkce_login_port_in_use_gives_actionable_error():
    import socket

    from agentic_context import _pkce

    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        port = blocker.getsockname()[1]
        with pytest.raises(OSError, match=rf"127\.0\.0\.1:{port}.*port="):
            _pkce.pkce_login(
                login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
                port=port, timeout=1, open_browser=lambda url: pytest.fail("must not open"),
            )


def test_pkce_login_closes_socket_when_listener_thread_fails_to_start(monkeypatch):
    import socket

    from agentic_context import _pkce

    class BoomThread:
        def __init__(self, *a, **k): ...
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(_pkce.threading, "Thread", BoomThread)
    port = 53999
    with pytest.raises(RuntimeError):
        _pkce.pkce_login(
            login_url=FAKE_LOGIN_URL, consumer_key=FAKE_CONSUMER_KEY,
            port=port, open_browser=lambda url: None,
        )
    # Port is free again: the bound socket was closed.
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def _audit_records(caplog):
    return [r for r in caplog.records if getattr(r, "audit_event", "") == "user_login"]


def test_redeem_code_success_is_audited(respx_mock, caplog):
    respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())
    with caplog.at_level("INFO", logger="agentic_context.auth"):
        UserContextAuth.redeem_code(**_redeem_kwargs(client_ip="203.0.113.7"))
    (rec,) = _audit_records(caplog)
    assert (rec.outcome, rec.method, rec.source_ip) == ("success", "backend", "203.0.113.7")
    assert (rec.user_id, rec.org_id) == (FAKE_USER_ID, FAKE_ORG_ID)
    for secret in ("server_refresh", "auth_code", "ver", FAKE_CONSUMER_SECRET):
        assert secret not in caplog.text


def test_redeem_code_denials_are_audited(respx_mock, caplog):
    respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=httpx.Response(
        400, json={"error": "invalid_grant"}))
    with caplog.at_level("INFO", logger="agentic_context.auth"):
        with pytest.raises(ReauthRequired):
            UserContextAuth.redeem_code(**_redeem_kwargs())
        with pytest.raises(ValueError):
            UserContextAuth.redeem_code(**_redeem_kwargs(state="forged"))
    recs = _audit_records(caplog)
    assert [r.outcome for r in recs] == ["denied", "denied"]
    assert recs[0].reason == "ReauthRequired:invalid_grant"
    assert recs[1].reason == "ValueError"
    assert recs[0].source_ip == "unknown"
    assert "forged" not in caplog.text


async def test_aredeem_code_is_audited(respx_mock, caplog):
    respx_mock.post(_TOKEN_ENDPOINT).mock(return_value=_code_grant_response())
    with caplog.at_level("INFO", logger="agentic_context.auth"):
        await UserContextAuth.aredeem_code(**_redeem_kwargs(client_ip="198.51.100.2"))
    (rec,) = _audit_records(caplog)
    assert (rec.outcome, rec.source_ip, rec.user_id) == ("success", "198.51.100.2", FAKE_USER_ID)


def test_login_success_and_denial_are_audited(monkeypatch, caplog):
    token = {"refresh_token": "r", "id": FAKE_ID_URL}
    monkeypatch.setattr(_pkce_module(), "pkce_login", lambda **kw: token)
    with caplog.at_level("INFO", logger="agentic_context.auth"):
        UserContextAuth.login(consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL)

        def boom(**kw):
            raise TimeoutError("Timed out")
        monkeypatch.setattr(_pkce_module(), "pkce_login", boom)
        with pytest.raises(TimeoutError):
            UserContextAuth.login(consumer_key=FAKE_CONSUMER_KEY, login_url=FAKE_LOGIN_URL)
    ok, denied = _audit_records(caplog)
    assert (ok.outcome, ok.method, ok.source_ip, ok.user_id) == (
        "success", "loopback", "127.0.0.1", FAKE_USER_ID)
    assert (denied.outcome, denied.reason) == ("denied", "TimeoutError")


def _pkce_module():
    from agentic_context import _pkce
    return _pkce
