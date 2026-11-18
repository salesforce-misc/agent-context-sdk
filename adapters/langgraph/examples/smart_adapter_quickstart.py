# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Smart LangGraph adapter quickstart - three worked scenarios.

Demonstrates the self-driving construction path of ``LangGraphAdapter``:
the customer brings only a vanilla compiled ``StateGraph`` plus
connection params; the adapter handles GET-context, SystemMessage
injection, graph drive, and per-turn POST automatically.

Scenarios:
  1. Multi-session memory carryover - session 2 rehydrates from session 1.
  2. Day-zero CRM personalization - cold start uses CRM facts.
  3. Cross-topic preference application - opts in to ``context_mode=
     "per-turn"``. Like ``auto``, this installs the ``recall_customer_context``
     tool and injects an intent instruction; on top of that it deterministically
     seeds fast context every turn (the fallback) for highest fidelity.

The graph is intentionally trivial: one LLM node backed by
``FakeListChatModel`` with canned responses, so the script runs offline
without any model API key. The point is that the customer's graph never
has to know about agentic memory.

Run (live, against a real Beacon)::

    export CDP_TOKEN=...
    export CDP_BASE_URL=https://beacon.example.com
    export CDP_DATASPACE=default
    cd adapters/langgraph && uv run python examples/smart_adapter_quickstart.py

Run (dry): with env vars unset the script prints a friendly notice and
exits 0 - useful as an import-and-wiring smoke test.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentic_context_langgraph import LangGraphAdapter


# ---------------------------------------------------------------------------
# 1. The customer's graph - vanilla StateGraph, zero memory plumbing.
# ---------------------------------------------------------------------------


class ChatState(TypedDict):
    """Minimal LangGraph state.

    ``add_messages`` is the standard reducer: it dedupes injected
    SystemMessages by stable id, which is exactly the contract the smart
    adapter relies on for context injection.
    """

    messages: Annotated[list[BaseMessage], add_messages]


def build_demo_graph():
    """Compiled StateGraph with a single LLM node backed by a fake model."""
    fake_llm = FakeListChatModel(
        responses=[
            "Got it - 16-inch, 32GB RAM, ~$2000 budget. Noted.",
            "Welcome back. Picking up the laptop hunt where we left off.",
            "Hello Sarah - I see you're on the Premier tier. How can I help?",
            "Booking a vegetarian-friendly spot for two on Friday at 7pm.",
        ]
    )

    async def llm_node(state: ChatState) -> dict[str, list[BaseMessage]]:
        # The adapter has already injected the context SystemMessage (if any)
        # into state["messages"] before this node runs.
        reply = await fake_llm.ainvoke(state["messages"])
        return {"messages": [reply]}

    builder: StateGraph = StateGraph(ChatState)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 2. Pretty-printer for streamed records.
# ---------------------------------------------------------------------------


def format_record(record: object) -> str:
    """One-line render: ``[event_type] participant_id: content[:80]``."""
    event_type = getattr(record, "event_type", "?")
    participant = getattr(record, "participant_id", None) or "-"
    content = getattr(record, "content", "") or ""
    snippet = content.replace("\n", " ").strip()[:80]
    return f"  [{event_type}] {participant}: {snippet}"


# ---------------------------------------------------------------------------
# 3. Scenario driver.
# ---------------------------------------------------------------------------


async def run_scenarios(api_key: str, base_url: str, dataspace: str) -> None:
    graph = build_demo_graph()

    # Default policy is "session_first": one GET on the first turn of each
    # new session_id. ``add_messages`` keeps the injected SystemMessage in
    # state across subsequent turns of the same session - so we pay 1 GET
    # per session, not per turn. Covers scenarios 1 and 2.
    adapter = LangGraphAdapter(
        graph=graph,
        agent_id="support-bot",
        owner_id="sarah@corp.com",
        api_key=api_key,
        base_url=base_url,
        dataspace=dataspace,
    )

    # ``async with`` opens the internal AsyncIngestionClient +
    # AsyncRetrievalClient (an AsyncSessionManager is created lazily per
    # session during stream()) and guarantees all owned resources are
    # closed (with the 30s drain budget) on exit.
    async with adapter:
        # ----- Scenario 1: multi-session memory carryover ------------------
        # ses-1 establishes preferences. Turn 1 GETs context (likely empty
        # on day zero), drives the graph, POSTs the user turn + reply.
        print("\n=== Scenario 1: multi-session memory carryover ===")
        print("Session 1 - establishing preferences")
        async for record in adapter.stream(
            "I'm laptop shopping. 16-inch, 32GB RAM, ~$2000 budget.",
            session_id="ses-1",
        ):
            print(format_record(record))

        # ses-2 has the same owner_id but a new session_id. Turn 1 of ses-2
        # triggers a fresh GET; Beacon returns the budget + spec preferences
        # learned in ses-1 as ``userContext`` facts. The adapter injects
        # them as SystemMessage(id="agentic-memory-context") *before* the
        # LLM runs - so the agent answers "Hey, I'm back" with full context
        # and never asks the user to repeat.
        print("Session 2 - adapter auto-rehydrates from session 1")
        async for record in adapter.stream(
            "Hey, I'm back.",
            session_id="ses-2",
        ):
            print(format_record(record))

        # ----- Scenario 2: day-zero CRM personalization --------------------
        # No prior chat history, but Beacon's GET /agentic/context still
        # surfaces CRM-derived facts about owner_id (tier, region, account
        # flags). The agent's first reply addresses the customer by name
        # and acknowledges tier - powered entirely by injected context.
        print("\n=== Scenario 2: day-zero CRM personalization ===")
        async for record in adapter.stream(
            "Hi, I need help with my account.",
            session_id="ses-3-crm",
        ):
            print(format_record(record))

        # ----- Scenario 3: cross-topic preference application --------------
        # A preference learned in one topic (vegetarian) must apply to a
        # query in an unrelated topic (dinner reservation). With the default
        # ``auto`` mode the SystemMessage injected at turn 1 may not yet
        # contain the dietary fact if the opening turn was elsewhere.
        # ``context_mode="per-turn"`` installs the recall tool + a force-deep
        # instruction (like ``auto``) and, on top of that, deterministically
        # seeds fresh fast context before each user turn for highest fidelity:
        #   + always fresh - covers topic shifts within a session.
        #   - +1 GET per turn (latency and quota cost vs. auto).
        # The stable SystemMessage id makes ``add_messages`` *replace*
        # rather than *append*, so prompts stay flat across turns.
        print("\n=== Scenario 3: cross-topic preference application ===")
        every_turn_adapter = LangGraphAdapter(
            graph=graph,
            agent_id="support-bot",
            owner_id="sarah@corp.com",
            api_key=api_key,
            base_url=base_url,
            dataspace=dataspace,
            context_mode="per-turn",  # opt-in - see comment above.
        )
        async with every_turn_adapter:
            async for record in every_turn_adapter.stream(
                "Book me a dinner reservation for two on Friday at 7pm.",
                session_id="ses-4-xtopic",
            ):
                print(format_record(record))


# ---------------------------------------------------------------------------
# 4. Entrypoint.
# ---------------------------------------------------------------------------


def _dry_run_notice() -> None:
    print(
        "[dry-run] CDP_TOKEN / CDP_BASE_URL not set - skipping live calls.\n"
        "          Set the following to run end-to-end against Beacon:\n"
        "              CDP_TOKEN=<beacon API token>\n"
        "              CDP_BASE_URL=<https://beacon-host>\n"
        "              CDP_DATASPACE=<dataspace, default 'default'>\n"
    )


def main() -> int:
    api_key = os.environ.get("CDP_TOKEN")
    base_url = os.environ.get("CDP_BASE_URL")
    dataspace = os.environ.get("CDP_DATASPACE", "default")

    # Build the graph eagerly so import / wiring errors surface even in
    # dry-run mode - the most useful smoke test without network access.
    _ = build_demo_graph()

    if not api_key or not base_url:
        _dry_run_notice()
        return 0

    try:
        asyncio.run(run_scenarios(api_key=api_key, base_url=base_url, dataspace=dataspace))
    except Exception as exc:  # pragma: no cover - example script
        print(f"[error] scenario run failed: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
