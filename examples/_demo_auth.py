"""Shared auth helper for the example demos.

Used by ``ace_basic_demo``, ``interactive_agent``, and ``quickstart_demo``
so one browser login works for all three. Two auth paths are defined here:

- ``build_auth`` (ACTIVE) — ``UserContextAuth.login()`` OAuth + PKCE loopback,
  with silent reuse of a cached refresh token.
- ``build_auth_service_account`` — ``ContextAuth`` service-account JWT bearer
  flow (no browser). ``interactive_agent`` selects it with ``--auth ent``; the
  other demos can swap ``from _demo_auth import build_auth`` for it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from agentic_context import ContextAuth, ReauthRequired, UserContextAuth


# Shared across all demos so a single browser login persists.
TOKEN_CACHE = Path.home() / ".cache" / "agentic_context_demos" / "refresh_token.json"


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"missing env var: {name}\n"
            f"Set it in examples/ace_agent.env (copy from ace_agent.env.example) "
            f"or export it in the shell."
        )
    return val


# Values that mean "not filled in yet": the template default, and the
# `placeholder` people paste to get past the missing-var check.
_API_KEY_PLACEHOLDERS = {"", "placeholder", "paste-your-anthropic-api-key-here"}
_DEVBAR_PATHS = ("devbar", "/Applications/devbar.app/Contents/MacOS/devbar")


def anthropic_api_key() -> str:
    """Return the Anthropic/gateway API key for the demos.

    Uses ``$ANTHROPIC_API_KEY`` when it holds a real value. Otherwise, if the
    optional DevBar app is installed (Salesforce-internal), asks it for the
    current gateway key via ``devbar auth claude``, the same helper Claude
    Code uses. The result is written back to ``os.environ`` so subprocesses
    (e.g. the Claude CLI) inherit it.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key not in _API_KEY_PLACEHOLDERS:
        return key
    for exe in _DEVBAR_PATHS:
        if not shutil.which(exe):
            continue
        try:
            out = subprocess.run(
                [exe, "auth", "claude"],
                capture_output=True, text=True, check=True, timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            detail = (getattr(e, "stderr", "") or getattr(e, "stdout", "") or str(e)).strip()
            sys.exit(f"`devbar auth claude` failed: {detail}\n"
                     f"Try `devbar auth login`, or set ANTHROPIC_API_KEY.")
        key = out.stdout.strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            return key
    sys.exit(
        "missing env var: ANTHROPIC_API_KEY\n"
        "Set it in examples/ace_agent.env (copy from ace_agent.env.example) "
        "or export it in the shell."
    )


def _save_refresh_token(token: str) -> None:
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(json.dumps({"refresh_token": token}))
    try:
        TOKEN_CACHE.chmod(0o600)
    except OSError:
        pass


def _load_refresh_token() -> str | None:
    try:
        data = json.loads(TOKEN_CACHE.read_text())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return data.get("refresh_token") if isinstance(data, dict) else None


# ─── Auth: choose one of two paths ────────────────────────────────────
#
# PRIMARY (default):    UserContextAuth.login() — OAuth + PKCE loopback.
#     Browser opens; the signed-in user's 005… becomes owner_id.
#     Needs: SF_CONSUMER_KEY, SF_LOGIN_URL, BEACON_AGENT_ID.
#
# ALTERNATIVE:          ContextAuth — service-account JWT bearer.
#     No browser; pre-provisioned private key.
#     Needs: SF_PRIVATE_KEY_PATH, SF_CONSUMER_KEY, SF_SUBJECT, SF_LOGIN_URL,
#            BEACON_AGENT_ID, and an owner id (BEACON_OWNER_ID or passed in).
# ──────────────────────────────────────────────────────────────────────


def build_auth(agent_id: str) -> UserContextAuth:
    """UserContextAuth path (default) — OAuth + PKCE loopback with cache.

    Silent-reuse: if ``~/.cache/agentic_context_demos/refresh_token.json`` holds a
    refresh token from a prior run, UserContextAuth is constructed directly
    (no browser). First run — or when Salesforce has revoked the token
    (:class:`ReauthRequired`) — opens the loopback OAuth login. The signed-in
    user's ``005…`` is available on ``auth.user_id``.
    """
    cached = _load_refresh_token()
    if cached:
        try:
            auth = UserContextAuth(
                refresh_token=cached,
                consumer_key=_require("SF_CONSUMER_KEY"),
                login_url=_require("SF_LOGIN_URL"),
                aid=agent_id,
                on_refresh_token=_save_refresh_token,
            )
            auth.get_token()  # probe so ReauthRequired surfaces here, not deep in a call
            print(f"[auth] cached refresh token reused ({TOKEN_CACHE})")
            return auth
        except ReauthRequired:
            print("[auth] cached refresh token revoked; falling back to browser login")

    print("[auth] user login via UserContextAuth (OAuth + PKCE loopback)")
    return UserContextAuth.login(
        consumer_key=_require("SF_CONSUMER_KEY"),
        login_url=_require("SF_LOGIN_URL"),
        aid=agent_id,
        on_refresh_token=_save_refresh_token,
    )


def build_auth_service_account(agent_id: str, owner_id: str | None = None) -> ContextAuth:
    """ContextAuth path — service-account JWT bearer. No browser login.

    Needs: SF_PRIVATE_KEY_PATH, SF_CONSUMER_KEY, SF_SUBJECT, SF_LOGIN_URL.
    ``owner_id`` defaults to ``$BEACON_OWNER_ID`` and is sent as ``xid``.

    Note: ContextAuth has no ``user_id`` attribute, so the owner id must be
    supplied explicitly rather than read off the signed-in user.
    """
    private_key = Path(_require("SF_PRIVATE_KEY_PATH")).expanduser().read_text()
    print("[auth] service-account login via ContextAuth (JWT bearer)")
    return ContextAuth(
        private_key=private_key,
        consumer_key=_require("SF_CONSUMER_KEY"),
        subject=_require("SF_SUBJECT"),
        login_url=_require("SF_LOGIN_URL"),
        aid=agent_id,
        xid=owner_id or _require("BEACON_OWNER_ID"),
    )
