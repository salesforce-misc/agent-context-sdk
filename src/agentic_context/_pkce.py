# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""OAuth web-server flow with PKCE — authorize URL, code redemption, loopback login.

Internal helpers behind :class:`UserContextAuth`:

* :func:`authorize_url` / :func:`redeem_code` / :func:`aredeem_code` — the two
  halves of the flow, split so a **backend** can host the callback itself:
  build the URL in ``/login``, redeem the code in ``/callback``. Used by
  :meth:`UserContextAuth.authorization_request` and
  :meth:`UserContextAuth.redeem_code`. ``client_secret`` is sent when given
  (confidential client — the ECA has "Require secret" on).
* :func:`pkce_login` — the **local** public-client path behind
  :meth:`UserContextAuth.login`: a loopback listener on ``127.0.0.1`` captures
  the code. Consumer key only; the ECA must register
  ``http://localhost:<port>/callback`` and have "Require secret" off.

Callers must host-validate ``login_url`` (a bare host, no scheme) first.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import secrets
import threading
import webbrowser
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

DEFAULT_SCOPE = "api refresh_token cdp_ingest_api"
# Rarely occupied (the AWS CLI uses it too), unlike 8080 which dev servers and
# Docker port-forwards commonly hold. Must match a callback URL on the ECA.
DEFAULT_PORT = 53682
DEFAULT_TIMEOUT = 300.0


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def new_verifier() -> str:
    """A fresh PKCE code verifier (RFC 7636: 43-128 unreserved characters)."""
    return secrets.token_urlsafe(64)[:128]


def new_state() -> str:
    """A fresh, unguessable OAuth ``state`` value (CSRF guard for the callback)."""
    return secrets.token_urlsafe(32)


def authorize_url(
    *,
    login_url: str,
    consumer_key: str,
    redirect_uri: str,
    scope: str,
    state: str,
    verifier: str,
) -> str:
    """The ``/services/oauth2/authorize`` URL to send the user's browser to."""
    return f"https://{login_url}/services/oauth2/authorize?" + urlencode({
        "response_type": "code",
        "client_id": consumer_key,
        "redirect_uri": redirect_uri,
        "code_challenge": _challenge(verifier),
        "code_challenge_method": "S256",
        "scope": scope,
        "state": state,
    })


def _code_body(
    consumer_key: str,
    consumer_secret: str | None,
    redirect_uri: str,
    code: str,
    verifier: str,
) -> dict[str, str]:
    body = {
        "grant_type": "authorization_code",
        "client_id": consumer_key,
        "redirect_uri": redirect_uri,
        "code": code,
        "code_verifier": verifier,
    }
    if consumer_secret:
        body["client_secret"] = consumer_secret
    return body


def _require_refresh_token(token: dict[str, Any]) -> dict[str, Any]:
    if not token.get("refresh_token"):
        raise ValueError(
            "Token response has no refresh_token — the ECA must allow the "
            "'refresh_token' scope and the login must request it"
        )
    return token


def redeem_code(
    client: httpx.Client,
    *,
    login_url: str,
    consumer_key: str,
    consumer_secret: str | None,
    redirect_uri: str,
    code: str,
    verifier: str,
) -> httpx.Response:
    """POST the authorization_code grant and return the raw response."""
    return client.post(
        f"https://{login_url}/services/oauth2/token",
        data=_code_body(consumer_key, consumer_secret, redirect_uri, code, verifier),
    )


async def aredeem_code(
    client: httpx.AsyncClient,
    *,
    login_url: str,
    consumer_key: str,
    consumer_secret: str | None,
    redirect_uri: str,
    code: str,
    verifier: str,
) -> httpx.Response:
    """Async counterpart of :func:`redeem_code`."""
    return await client.post(
        f"https://{login_url}/services/oauth2/token",
        data=_code_body(consumer_key, consumer_secret, redirect_uri, code, verifier),
    )


def pkce_login(
    *,
    login_url: str,
    consumer_key: str,
    scope: str = DEFAULT_SCOPE,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    open_browser: Callable[[str], Any] = webbrowser.open,
) -> dict[str, Any]:
    """Run the interactive login and return the raw token response.

    The response carries ``access_token``, ``refresh_token`` and ``id`` (the
    ``…/id/<org>/<user>`` URL). Blocks until the browser callback arrives or
    ``timeout`` seconds pass (``TimeoutError``). ``login_url`` must already be
    host-validated by the caller; it is a bare host with no scheme.
    """
    redirect_uri = f"http://localhost:{port}/callback"
    verifier = new_verifier()
    state = new_state()

    params: dict[str, str] = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # stdlib hook name
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            # Only the redirect carrying OUR state counts (Salesforce echoes it
            # on error redirects too). A stray request must neither end the
            # wait nor overwrite what a valid one captured, so anything else is
            # refused and the listener keeps waiting.
            if done.is_set() or not secrets.compare_digest(
                query.get("state", ""), state
            ):
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Unexpected request.")
                return
            params.update(query)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Authentication complete. You can close this window.")
            done.set()

        def log_message(self, *args: Any) -> None:  # silence request logging
            pass

    # No fallback to a neighbouring port: the redirect_uri must exactly match a
    # callback URL registered on the ECA, so only the configured port can work.
    try:
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        raise OSError(
            f"Cannot listen on 127.0.0.1:{port} for the Salesforce login callback "
            f"({e.strerror or e}). Free the port, or pass a different port= and "
            f"register http://localhost:<port>/callback as a callback URL on the ECA."
        ) from e
    serving = False
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        serving = True
        open_browser(authorize_url(
            login_url=login_url, consumer_key=consumer_key,
            redirect_uri=redirect_uri, scope=scope, state=state, verifier=verifier,
        ))
        if not done.wait(timeout):
            raise TimeoutError("Timed out waiting for the Salesforce login callback")
    finally:
        # shutdown() waits for serve_forever() to exit, so it would hang if the
        # thread never started; the socket is bound either way and must be closed.
        if serving:
            server.shutdown()
        server.server_close()

    if "error" in params:
        raise ValueError(
            f"Salesforce login failed: {params.get('error')}: "
            f"{params.get('error_description', '')}"
        )
    if not secrets.compare_digest(params.get("state", ""), state):
        raise ValueError("Salesforce login callback state mismatch; aborting")
    if not params.get("code"):
        raise ValueError("Salesforce login callback carried no authorization code")

    with httpx.Client(timeout=30) as client:
        r = redeem_code(
            client, login_url=login_url, consumer_key=consumer_key,
            consumer_secret=None, redirect_uri=redirect_uri,
            code=params["code"], verifier=verifier,
        )
    r.raise_for_status()
    return _require_refresh_token(r.json())
