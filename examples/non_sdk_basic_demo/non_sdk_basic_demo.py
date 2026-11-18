"""Zero-SDK demo — proves Salesforce auth + Beacon API work end-to-end.

Does the same thing ``ace_basic_demo`` does, but WITHOUT importing
``agentic_context``. Every HTTP call is raw ``httpx`` so you can inspect
exactly what the SDK is doing on your behalf.

Flow
----
1. OAuth 2.0 Authorization Code + PKCE loopback against Salesforce
     → SF access_token  + instance_url  (leg 1)
2. a360 CDP data-cloud exchange
     → CDP access_token + Tenant-Specific Endpoint (TSE)  (leg 2)
3. ``create`` — POST   {TSE}/api/v1/agentic/conversations
   ``list``   — GET    {TSE}/api/v1/agentic/conversations

Env vars (same file ``examples/ace_basic_demo`` uses: ``examples/ace_agent.env``)
    SF_CONSUMER_KEY   Connected-app consumer key (client_id).
    SF_LOGIN_URL      Salesforce login host, no scheme
                      (e.g. ``acme.my.salesforce.com``).
    BEACON_AGENT_ID   Registered agent id (e.g. ``TestAgent``).

ECA setup (same requirements as the basic demo)
    - callback URL  ``http://localhost:53682/callback``
    - web-server flow enabled, "Require secret for web server flow" → off
    - scopes:  ``api refresh_token cdp_ingest_api``

Install
    pip install httpx

Run
    python examples/non_sdk_basic_demo/non_sdk_basic_demo.py create \\
        --conversation-id raw-demo-1 --name "raw-api-check"
    python examples/non_sdk_basic_demo/non_sdk_basic_demo.py list --status ACTIVE
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import urllib.parse
import uuid
import webbrowser
from pathlib import Path

import httpx


# ─── Config: load examples/ace_agent.env (shell env wins) ─────────────
def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(Path(__file__).resolve().parent.parent / "ace_agent.env")


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"missing env var: {name} (set in examples/ace_agent.env)")
    return val


# ─── Leg 1: Authorization Code + PKCE loopback ────────────────────────
PORT = 53682
REDIRECT_URI = f"http://localhost:{PORT}/callback"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


class _CodeCatcher(http.server.BaseHTTPRequestHandler):
    """One-shot loopback handler. Captures ?code=... from the redirect."""

    code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            _CodeCatcher.code = params["code"][0]
            body = b"<h2>Login complete.</h2> You can close this tab."
        else:
            _CodeCatcher.error = params.get("error_description", ["unknown"])[0]
            body = f"<h2>Login failed:</h2> {_CodeCatcher.error}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:  # silence stderr noise
        pass


def _loopback_authorize(login_url: str, consumer_key: str) -> tuple[str, str]:
    """Browser popup → authorize → captures ``code``. Returns ``(code, verifier)``."""
    verifier, challenge = _make_pkce()
    state = secrets.token_urlsafe(16)
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": consumer_key,
        "redirect_uri": REDIRECT_URI,
        "scope": "api refresh_token cdp_ingest_api",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "prompt": "login",
    })
    auth_url = f"https://{login_url}/services/oauth2/authorize?{params}"

    # Reserve the port BEFORE opening the browser so a stale listener fails fast.
    try:
        httpd = http.server.HTTPServer(("127.0.0.1", PORT), _CodeCatcher)
    except OSError as e:
        sys.exit(f"port {PORT} is in use ({e}). Kill whatever is listening and retry.")

    print(f"[auth] opening browser → {auth_url[:80]}...")
    webbrowser.open(auth_url)
    # handle_request() blocks until exactly one request (the OAuth callback)
    # is served, then returns. The one-shot handler stashes the code/error on
    # the class for us to read below.
    httpd.handle_request()
    if _CodeCatcher.error:
        sys.exit(f"[auth] Salesforce returned: {_CodeCatcher.error}")
    assert _CodeCatcher.code is not None
    return _CodeCatcher.code, verifier


def _leg1_code_exchange(
    login_url: str, consumer_key: str, code: str, verifier: str
) -> tuple[str, str]:
    """POST /services/oauth2/token with the auth code. Returns ``(sf_token, instance_url)``."""
    r = httpx.post(
        f"https://{login_url}/services/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": consumer_key,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        },
        timeout=30.0,
    )
    print(f"[leg1] POST https://{login_url}/services/oauth2/token  → {r.status_code}")
    r.raise_for_status()
    body = r.json()
    return body["access_token"], body["instance_url"]


# ─── Leg 2: a360 CDP data-cloud exchange ───────────────────────────────
def _leg2_a360_exchange(
    core_instance_url: str, sf_token: str, agent_id: str
) -> tuple[str, str]:
    """POST {core_instance_url}/services/a360/token. Returns ``(cdp_token, tse_base_url)``."""
    r = httpx.post(
        f"{core_instance_url}/services/a360/token",
        data={
            "grant_type": "urn:salesforce:grant-type:external:cdp",
            "subject_token": sf_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "aid": agent_id,
            "dataspace": "default",
        },
        timeout=30.0,
    )
    print(f"[leg2] POST {core_instance_url}/services/a360/token  → {r.status_code}")
    r.raise_for_status()
    body = r.json()
    if "access_token" not in body:
        sys.exit(f"[leg2] a360 returned no access_token: {body}")
    cdp_token = body["access_token"]
    # a360 returns the TSE in `instance_url`; it may be bare host or scheme+host.
    tse = body["instance_url"]
    if not tse.startswith("http"):
        tse = f"https://{tse}"
    return cdp_token, tse


# ─── Beacon lifecycle calls (raw httpx) ────────────────────────────────
def _beacon_headers(cdp_token: str, owner_id: str, agent_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {cdp_token}",
        "Content-Type": "application/json",
        "dataspace": "default",
        "owner_id": owner_id,
        "agent_id": agent_id,
    }


def create_conversation(
    tse: str, cdp_token: str, owner_id: str, agent_id: str,
    conversation_id: str, name: str,
) -> None:
    payload = {
        "conversationId": conversation_id,
        "conversationOwnerId": owner_id,
        "ownerType": "employee",
        "name": name,
        "channel": {"type": "web", "id": "non-sdk-basic-demo"},
        "variable": {
            "Name": name,
            "ConversationState": "ACTIVE",
            "ExternalRecordId": f"ext-{conversation_id}",
            "MessageDeliveryShortCodeText": "raw-demo",
            "ConversationExpirationDateTime": "2099-01-01T00:00:00Z",
        },
    }
    url = f"{tse}/api/v1/agentic/conversations"
    r = httpx.post(
        url,
        headers=_beacon_headers(cdp_token, owner_id, agent_id),
        json=payload,
        timeout=30.0,
    )
    print(f"[create] POST {url}  → {r.status_code}")
    print(f"[create] conversationId: {conversation_id}")
    print(f"[create] body: {r.text}")


def list_conversations(
    tse: str, cdp_token: str, owner_id: str, agent_id: str,
    status: str | None, page_size: int,
) -> None:
    params: dict[str, str | int] = {
        "owner_id": owner_id,
        "owner_type": "employee",
        "page_size": page_size,
    }
    if status:
        params["status"] = status
    url = f"{tse}/api/v1/agentic/conversations"
    r = httpx.get(
        url,
        headers=_beacon_headers(cdp_token, owner_id, agent_id),
        params=params,
        timeout=30.0,
    )
    print(f"[list] GET {url}")
    print(f"[list] params: {params}")
    print(f"[list] status: {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2))
    except ValueError:
        print(f"[list] body: {r.text}")


# ─── Owner id: pull from the SF access-token introspection /userinfo ──
def _signed_in_user_id(instance_url: str, sf_token: str) -> str:
    """GET /services/oauth2/userinfo → ``user_id`` is the signed-in 005…."""
    r = httpx.get(
        f"{instance_url}/services/oauth2/userinfo",
        headers={"Authorization": f"Bearer {sf_token}"},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["user_id"]


# ─── Top-level driver ──────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Zero-SDK Salesforce auth + Beacon demo.")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_create = sub.add_parser("create", help="Create a new conversation.")
    p_create.add_argument("--conversation-id", default=None)
    p_create.add_argument("--name", default="raw-api-check")
    p_list = sub.add_parser("list", help="List conversations for the signed-in user.")
    p_list.add_argument("--status", default=None, help="e.g. ACTIVE")
    p_list.add_argument("--page-size", type=int, default=20)
    args = p.parse_args()

    login_url = _require("SF_LOGIN_URL")
    consumer_key = _require("SF_CONSUMER_KEY")
    agent_id = _require("BEACON_AGENT_ID")

    print("─── Leg 1: Salesforce OAuth (PKCE loopback) ───")
    code, verifier = _loopback_authorize(login_url, consumer_key)
    sf_token, instance_url = _leg1_code_exchange(login_url, consumer_key, code, verifier)
    owner_id = _signed_in_user_id(instance_url, sf_token)
    print(f"[auth] signed in as {owner_id}")
    print(f"[auth] core instance_url: {instance_url}")

    print("\n─── Leg 2: a360 CDP exchange ───")
    cdp_token, tse = _leg2_a360_exchange(instance_url, sf_token, agent_id)
    print(f"[auth] TSE: {tse}")

    print(f"\n─── Beacon call ({args.cmd}) ───")
    if args.cmd == "create":
        conversation_id = args.conversation_id or f"raw-{uuid.uuid4().hex[:12]}"
        create_conversation(tse, cdp_token, owner_id, agent_id, conversation_id, args.name)
    elif args.cmd == "list":
        list_conversations(tse, cdp_token, owner_id, agent_id, args.status, args.page_size)


if __name__ == "__main__":
    main()
