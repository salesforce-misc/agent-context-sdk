"""Thin CLI that walks the four ACE Agentic lifecycle calls end-to-end:
create a conversation, ingest a user+agent exchange, read it back, list
conversations.

Auth is built in: the script uses ``agentic_context.UserContextAuth.login()``
to run the OAuth + PKCE loopback flow on startup, so you never paste a bearer
token. The signed-in user's ``005…`` becomes the owner for every call.

Config via a param file (recommended for new users):

    1. Copy ``examples/ace_agent.env.example`` to ``examples/ace_agent.env``.
    2. Fill in the values.
    3. ``python examples/ace_basic_demo.py create``

The script auto-loads ``examples/ace_agent.env`` (or ``.env`` in the current
directory) on startup — no shell exports required. Variables already set in
the environment WIN over the file, so you can override one on the command
line without editing the file.

Needs: SF_CONSUMER_KEY, SF_LOGIN_URL, BEACON_AGENT_ID.
Opens a browser popup; the signed-in user's 005… becomes owner_id. The ECA
must register http://localhost:53682/callback and have "Require secret"
turned off.

Env vars (set in the file or export in the shell):
    SF_CONSUMER_KEY       connected-app consumer key (client_id)
    SF_LOGIN_URL          Salesforce login host, no scheme
                          e.g. acme.my.salesforce.com
    BEACON_AGENT_ID       e.g. default_agent  (goes in the `agent_id` header,
                          and into the a360 exchange as `aid`)

Install:
    pip install httpx anyio

Run:
    python examples/ace_basic_demo.py create [--name NAME] [--conversation-id CID]
    python examples/ace_basic_demo.py ingest --conversation-id CID
                                       [--user-message MSG] [--agent-response MSG]
    python examples/ace_basic_demo.py read   --conversation-id CID
    python examples/ace_basic_demo.py list   [--status ACTIVE] [--page-size 20]
    python examples/ace_basic_demo.py --env-file path/to/custom.env list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import anyio
import httpx

from datetime import datetime, timezone

from agentic_context import (
    AsyncIngestionClient,
    AsyncRetrievalClient,
    AsyncSessionManager,
    IngestionError,
    Record,
    UserContextAuth,
)

# Shared auth helper (see examples/_demo_auth.py). To swap to the
# service-account flow, change this import to ``build_auth_service_account``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _demo_auth import build_auth  # noqa: E402


def _load_env_file(path: Path) -> int:
    """Read a KEY=value / `export KEY=value` file into os.environ.

    Already-set env vars WIN — the file never overwrites an existing value,
    so an ad-hoc ``BEACON_AGENT_ID=foo python ace_basic_demo.py …`` still works.
    Lines starting with ``#`` or blank lines are skipped. Values may be
    wrapped in single or double quotes; both get stripped. Returns the
    number of variables loaded.
    """
    if not path.is_file():
        return 0
    loaded = 0
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
        if value and value[0] not in ("'", '"') and "#" in value:
            value = value.split("#", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _apply_env_file(explicit: str | None) -> None:
    """Load env from an explicit file, then fall back to defaults.

    Default search order (first hit wins):
      1. ``examples/ace_agent.env`` — one dir up (shared with interactive_agent)
      2. ``.env`` in the current working directory
    """
    here = Path(__file__).resolve().parent
    candidates = (
        [Path(explicit)] if explicit else [here.parent / "ace_agent.env", Path.cwd() / ".env"]
    )
    for p in candidates:
        if p.is_file():
            n = _load_env_file(p)
            print(f"[env] loaded {n} var(s) from {p}", file=sys.stderr)
            return
    if explicit:
        sys.exit(f"--env-file not found: {explicit}")


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--env-file", default=None)
_pre_args, _ = _pre.parse_known_args()
_apply_env_file(_pre_args.env_file)


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"missing env var: {name}\n"
            f"Set it in examples/ace_agent.env (copy from ace_agent.env.example) "
            f"or export it in the shell."
        )
    return val


def _build_auth(agent_id: str, owner_type_override: str | None) -> tuple[UserContextAuth, str, str]:
    """Thin demo-side wrapper: delegate to the shared ``build_auth`` and derive
    the ``(owner_id, owner_type)`` tuple ace_basic_demo callers expect.

    See ``examples/_demo_auth.py`` for the two auth paths (UserContextAuth
    active; ContextAuth commented out).
    """
    auth = build_auth(agent_id)
    owner_type = owner_type_override or auth.default_owner_type
    source = "--owner-type" if owner_type_override else "auth default"
    print(f"[auth] signed in as {auth.user_id} (owner_type={owner_type} from {source})")
    return auth, auth.user_id, owner_type


async def _auth_bits(auth: UserContextAuth, agent_id: str) -> tuple[dict[str, str], str]:
    token, base_url = await auth.aget_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "agent_id": agent_id,
    }
    return headers, base_url.rstrip("/")


async def create_conversation(
    auth: UserContextAuth,
    agent_id: str,
    owner_id: str,
    owner_type: str,
    name: str,
    conversation_id: str | None,
) -> None:
    """Create a conversation via the SDK's AsyncIngestionClient.

    Goes through the public SDK method so this demo exercises the same code
    path the adapters use in production — token refresh, header shape, and
    error mapping all handled by the client.
    """
    conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    token, base_url = await auth.aget_token()
    async with AsyncIngestionClient(base_url=base_url, api_key=token, auth=auth) as client:
        try:
            resp = await client.create_conversation(
                conversation_id, owner_id, agent_id,
                name=name, owner_type=owner_type,
            )
            print(f"POST {base_url}/api/v1/agentic/conversations")
            print(f"conversationId: {conversation_id}")
            print(f"status: 200")
            print(f"body: {json.dumps(resp)}")
        except IngestionError as e:
            print(f"POST {base_url}/api/v1/agentic/conversations")
            print(f"conversationId: {conversation_id}")
            print(f"status: {e.status_code}")
            print(f"body: {e.body}")


async def ingest_turn(
    auth: UserContextAuth,
    agent_id: str,
    owner_id: str,
    owner_type: str,
    conversation_id: str,
    session_id: str | None,
    user_message: str,
    agent_response: str,
) -> None:
    """Ingest one user+agent exchange via AsyncSessionManager.

    SessionManager owns the full Beacon lifecycle (create_conversation →
    create_session → add_participants → add_entries) lazily on the first
    ingest(). Re-running against the same conversation_id/session_id reuses
    the conversation (409 on re-create is success) but writes the entries
    again — event_id is not sent, so nothing is deduped.
    """
    session_id = session_id or conversation_id
    interaction_id = f"turn-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    records = [
        Record(
            event_id=f"user-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            agent_id=agent_id,
            event_type="user_message",
            participant_id=owner_id,
            content=user_message,
            interaction_id=interaction_id,
            timestamp=now,
        ),
        Record(
            event_id=f"agent-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            agent_id=agent_id,
            event_type="agent_response",
            participant_id=agent_id,
            content=agent_response,
            interaction_id=interaction_id,
            timestamp=now,
        ),
    ]
    token, base_url = await auth.aget_token()
    async with AsyncIngestionClient(base_url=base_url, api_key=token, auth=auth) as client:
        mgr = AsyncSessionManager(
            client,
            agent_id=agent_id,
            owner_id=owner_id,
            owner_type=owner_type,
            session_id=session_id,
        )
        print(f"ingest conversationId={conversation_id} sessionId={session_id}")
        print(f"  user_message: {user_message}")
        print(f"  agent_response: {agent_response}")
        await mgr.ingest(records)
        await mgr.close()
        print("ingest ok (lifecycle + entries delivered)")


async def read_conversation(
    auth: UserContextAuth,
    agent_id: str,
    owner_id: str,
    owner_type: str,
    conversation_id: str,
    scope: str,
) -> None:
    """Read a conversation's full history via AsyncRetrievalClient."""
    token, base_url = await auth.aget_token()
    async with AsyncRetrievalClient(
        base_url=base_url, api_key=token, auth=auth, dataspace="default",
    ) as rc:
        result = await rc.get_conversation(
            conversation_id, owner_id, agent_id,
            scope=scope, owner_type=owner_type,
        )
        print(f"GET {base_url}/api/v1/agentic/conversations/{conversation_id}")
        print(f"params: owner_id={owner_id} owner_type={owner_type} scope={scope}")
        print(json.dumps(result, indent=2))


async def list_conversations(
    auth: UserContextAuth,
    agent_id: str,
    owner_id: str,
    owner_type: str,
    status: str | None,
    page_size: int,
) -> None:
    """List conversations with raw httpx.

    AsyncIngestionClient wraps only the lifecycle endpoints (create/close);
    listing isn't in the SDK's surface today, so this call stays raw.
    """
    params: dict[str, str | int] = {
        "owner_id": owner_id,
        "owner_type": owner_type,
        "page_size": page_size,
    }
    if status:
        params["status"] = status
    headers, base = await _auth_bits(auth, agent_id)
    url = f"{base}/api/v1/agentic/conversations"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers, params=params)
    print(f"GET {url}")
    print(f"params: {params}")
    print(f"status: {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(f"body: {resp.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin ACE Agentic API demo.")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a KEY=value env file. Defaults to the shared "
             "examples/ace_agent.env (one dir up), then .env in CWD.",
    )
    parser.add_argument(
        "--owner-type",
        choices=["individual", "employee"],
        default=None,
        help="Override auth.default_owner_type. UserContextAuth defaults to "
             "'employee'; pass 'individual' to read/write conversations "
             "written under a service-account (ContextAuth) flow.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new conversation.")
    p_create.add_argument("--name", default="ace-conv")
    p_create.add_argument("--conversation-id", default=None)

    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest one user+agent exchange via AsyncSessionManager.",
    )
    p_ingest.add_argument("--conversation-id", required=True)
    p_ingest.add_argument(
        "--session-id", default=None,
        help="Session id. Defaults to --conversation-id (SDK rule: a session "
             "shares its id with its conversation).",
    )
    p_ingest.add_argument("--user-message", default="What's my plan?")
    p_ingest.add_argument(
        "--agent-response",
        default="Your plan is Gold PPO; deductible is $500.",
    )

    p_read = sub.add_parser(
        "read",
        help="Read the full message history for a conversation.",
    )
    p_read.add_argument("--conversation-id", required=True)
    p_read.add_argument(
        "--scope", default="all", choices=["all", "TRACES"],
        help="all = messages + traces across every session (default); "
             "TRACES = tool_call/result/reasoning only (server-dependent).",
    )

    p_list = sub.add_parser("list", help="List conversations for this owner.")
    p_list.add_argument("--status", default=None, help="e.g. ACTIVE")
    p_list.add_argument("--page-size", type=int, default=20)

    args = parser.parse_args()
    agent_id = _require("BEACON_AGENT_ID")
    auth, owner_id, owner_type = _build_auth(agent_id, args.owner_type)
    if args.cmd == "create":
        anyio.run(
            create_conversation,
            auth, agent_id, owner_id, owner_type, args.name, args.conversation_id,
        )
    elif args.cmd == "ingest":
        anyio.run(
            ingest_turn,
            auth, agent_id, owner_id, owner_type,
            args.conversation_id, args.session_id,
            args.user_message, args.agent_response,
        )
    elif args.cmd == "read":
        anyio.run(
            read_conversation,
            auth, agent_id, owner_id, owner_type, args.conversation_id, args.scope,
        )
    elif args.cmd == "list":
        anyio.run(
            list_conversations,
            auth, agent_id, owner_id, owner_type, args.status, args.page_size,
        )


if __name__ == "__main__":
    main()
