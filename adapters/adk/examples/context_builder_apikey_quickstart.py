# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Context Builder api-key quickstart — the whole SDK wiring in one file.

This is the drop-in an app hosted on the **Context Builder / Secret Manager**
platform needs. The premise of that platform is *"the customer only gives you an
API key"* — everything else (the Connected App private key, consumer key,
subject, login URL, and the agent id) is vended by the Secret Manager off that
one key. This file shows the minimum wiring that turns a bare ADK ``LlmAgent``
into a memory-backed agent given nothing but the key.

Pipeline the key travels through::

    Context Builder ──register agent──▶ issues 1 API key per agent
    (genai-app-...contextbuilder)         │
                                          ▼
    Secret Manager ◀── GET /api/secret ── ApiKeyAuth  (x-api-key: <key>)
    (context-builder-secrets-...)         returns bundle: privateKey /
                                          consumerKey / subject / loginUrl /
                                          agentId
                                          │  ApiKeyAuth builds an inner
                                          ▼  ContextAuth, runs JWT → CDP exchange
    Demo App ── "Enter User ID" ──▶ owner_id (per conversation, per turn)
    (context-builder-demo-...)      imports THIS wiring for memory

Two facts drive the shape of the code below:

  * **owner_id is per-conversation, not per-adapter.** The demo front-end has an
    "Enter User ID" field — that value is the ``owner_id`` and it changes every
    conversation. So we never bake it into the adapter; the ADK session carries
    it as ``session.user_id`` and ``wrap_before_model`` auto-binds it on the
    first turn. (Nothing to do here beyond creating ADK sessions with the real
    user id.)

  * **one API key ↔ one agent.** The Secret Manager maps each key to exactly one
    agent, and the bundle carries that agent's ``agentId``. A multi-agent app
    (the demo's handoff story) holds one key — and one adapter — per agent. See
    ``build_agent`` below; call it once per key.

Run (live, against the real Secret Manager + Beacon)::

    export BEACON_API_KEY=...                        # the per-agent key
    export AGENTIC_CONTEXT_SECRET_SERVER_URL=https://<secret-manager-host>
    export GOOGLE_API_KEY=...                         # for the Gemini model
    cd adapters/adk && python examples/context_builder_apikey_quickstart.py

Run (dry): with ``BEACON_API_KEY`` / ``AGENTIC_CONTEXT_SECRET_SERVER_URL`` unset
the script prints what it *would* build and exits 0 — an import-and-wiring smoke
test that needs no network and no secrets.
"""
from __future__ import annotations

import asyncio
import os
import sys

from agentic_context import ApiKeyAuth
from agentic_context_adk import GoogleAdkAdapter


# ---------------------------------------------------------------------------
# The entire wiring: API key in → memory-backed adapter out.
# ---------------------------------------------------------------------------


async def build_adapter(
    api_key: str,
    *,
    secret_server_url: str | None = None,
    context_mode: str = "auto",
) -> GoogleAdkAdapter:
    """Construct a memory adapter from nothing but a Context Builder API key.

    The customer's app calls this once per agent at startup. No private key, no
    ``agent_id``, no Beacon URL — all of it is resolved from the key via the
    Secret Manager.

    Args:
        api_key: the per-agent key issued by Context Builder.
        secret_server_url: the Secret Manager base URL. Optional — if omitted,
            ``ApiKeyAuth`` reads ``AGENTIC_CONTEXT_SECRET_SERVER_URL``. There is
            no baked-in default; one of the two MUST be set or the first token
            fetch raises ``ValueError`` (fail-closed by design).
        context_mode: context behaviour — ``"auto"`` (default; fast context
            injected once at conversation open, then the agent drives deep
            recall via the tool) or ``"manual"`` (no auto-injection; you own
            retrieval). Leave ``"auto"`` for the demo.

    Returns:
        A ready ``GoogleAdkAdapter``. Its Beacon clients are built lazily on the
        first turn (that's also when the ``/api/secret`` fetch happens), so this
        function itself does no network I/O beyond nothing.

    Note:
        If you already know the ``agent_id`` (you registered the agent, so you
        do), the whole thing collapses to a **one-liner** — pass the key
        straight to the adapter and skip the manual ``ApiKeyAuth`` +
        ``aget_token`` dance::

            adapter = GoogleAdkAdapter(agent_id="agent1", agent_api_key=api_key)

        This function takes the longer route only to *resolve* ``agent_id`` from
        the bundle (useful when the caller doesn't know it up front). Both paths
        build the same ``ApiKeyAuth`` under the hood.
    """
    # 1. The key → bundle handshake. Lazy: no fetch happens here.
    auth = ApiKeyAuth(api_key, secret_server_url=secret_server_url)

    # 2. Resolve agent_id from the bundle. The adapter does NOT auto-read
    #    auth.agent_id today, so we trigger the fetch and read it ourselves.
    #    aget_token() runs the /api/secret download + the JWT → CDP exchange and
    #    populates auth.agent_id. (If the key's bundle leaves agentId empty — as
    #    the platform default key does — fall back to the env override, then to
    #    a placeholder default so the demo still runs.)
    await auth.aget_token()
    agent_id = auth.agent_id or os.environ.get("BEACON_AGENT_ID") or "Agent_Data_Cloud"

    # 3. The adapter. owner_id is intentionally omitted — it is per-conversation
    #    and auto-detected from session.user_id (the demo's "Enter User ID").
    return GoogleAdkAdapter(
        agent_id=agent_id,
        auth=auth,
        context_mode=context_mode,
    )


def build_agent(adapter: GoogleAdkAdapter, *, name: str, instruction: str):
    """Wrap an ADK ``LlmAgent`` with the adapter's tools + callbacks.

    This is the only place the customer's own agent definition meets the SDK:
    spread ``default_tools()`` into ``tools`` and unpack ``callback_kwargs()``.
    Everything memory-related (context injection, ingestion, the deep-recall
    tool) rides those two calls — the agent's own logic never mentions memory.
    """
    from google.adk.agents import LlmAgent  # imported here so dry-run needs no ADK

    return LlmAgent(
        name=name,
        model="gemini-3-flash-preview",
        instruction=instruction,
        tools=[*adapter.default_tools()],
        **adapter.callback_kwargs(),
    )


# ---------------------------------------------------------------------------
# Demo entrypoint — live if creds are present, dry otherwise.
# ---------------------------------------------------------------------------


async def _main() -> int:
    api_key = os.environ.get("BEACON_API_KEY", "")
    secret_url = os.environ.get("AGENTIC_CONTEXT_SECRET_SERVER_URL", "")

    if not api_key or not secret_url:
        print(
            "DRY RUN — set BEACON_API_KEY + AGENTIC_CONTEXT_SECRET_SERVER_URL to "
            "run live.\n"
            "This example would, given only the key:\n"
            "  1. ApiKeyAuth(key) → GET {secret_url}/api/secret (x-api-key) → bundle\n"
            "  2. await auth.aget_token() → JWT→CDP exchange, resolves agent_id\n"
            "  3. GoogleAdkAdapter(agent_id=<from bundle>, auth=auth) — no owner_id\n"
            "  4. LlmAgent(tools=[*adapter.default_tools()], **adapter.callback_kwargs())\n"
            "  5. owner_id arrives per conversation via session.user_id "
            "(the demo's 'Enter User ID').\n"
            "Import + wiring OK."
        )
        return 0

    adapter = await build_adapter(api_key, secret_server_url=secret_url)
    agent = build_agent(
        adapter,
        name="shopping_assistant",
        instruction="You are a helpful shopping assistant.",
    )
    print(f"Live: adapter bound to agent_id={adapter.agent_id!r}; "
          f"agent={agent.name!r} wired with memory tools + callbacks.")
    print("Create ADK sessions with the real user id as session.user_id — "
          "owner binding is automatic on the first turn.")
    await adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
