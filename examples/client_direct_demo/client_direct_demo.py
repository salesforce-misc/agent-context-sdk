#!/usr/bin/env python3
"""AgentContextClient via direct method calls (Design.md §8).

One of two AgentContextClient demos: call start_conversation /
ingest_customer_message / ingest_agent_message / call_tool from the host.
This demo uses LangGraph nodes + ToolNode; see client_hooks_demo for the
hooks + MCP path.

Env: examples/ace_agent.env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_context import AgentContextClient, CONTEXT_PACK_INSTRUCTION

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _demo_auth import anthropic_api_key, build_auth  # noqa: E402


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
        if value and value[0] not in ("'", '"') and "#" in value:
            value = value.split("#", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(Path(__file__).resolve().parent.parent / "ace_agent.env")


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"missing env var: {name}")
    return val


def _pretty(obj: object) -> str:
    try:
        return json.dumps(obj, indent=2, default=str, sort_keys=False)
    except (TypeError, ValueError):
        return repr(obj)


def _install_pretty_wire_hooks() -> None:
    """Pretty-print Beacon HTTP like interactive_agent (>>> GET/POST … <<< body)."""
    from agentic_context.client import IngestionClient
    from agentic_context.retrieval import RetrievalClient

    original_post = IngestionClient._post

    def logged_post(self, path, payload, *, owner_id, agent_id):
        print(f"\n>>> POST {self.base_url}{path}")
        print(f">>> headers: agent_id={agent_id}  owner_id={owner_id}")
        if payload is not None:
            print(">>> body:")
            print(_pretty(payload))
        try:
            resp = original_post(
                self, path, payload, owner_id=owner_id, agent_id=agent_id
            )
            print("<<< 2xx response:")
            print(_pretty(resp))
            return resp
        except Exception as e:
            print(f"<<< error: {e!r}")
            raise

    IngestionClient._post = logged_post  # type: ignore[method-assign]

    original_get = RetrievalClient._get

    def logged_get(self, path, *, params, headers):
        print(f"\n>>> GET {self.base_url}{path}")
        safe_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        print(f">>> headers: {safe_headers}")
        print(f">>> params: {params}")
        try:
            resp = original_get(self, path, params=params, headers=headers)
            print("<<< 2xx response:")
            print(_pretty(resp))
            return resp
        except Exception as e:
            print(f"<<< error: {e!r}")
            raise

    RetrievalClient._get = logged_get  # type: ignore[method-assign]

    original_send = RetrievalClient._send

    def logged_send(
        self, method, path, *, headers, params=None, json=None, timeout=None
    ):
        print(f"\n>>> {method} {self.base_url}{path}")
        safe_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        print(f">>> headers: {safe_headers}")
        if params is not None:
            print(f">>> params: {params}")
        if json is not None:
            print(">>> body:")
            print(_pretty(json))
        try:
            resp = original_send(
                self,
                method,
                path,
                headers=headers,
                params=params,
                json=json,
                timeout=timeout,
            )
            print("<<< 2xx response:")
            print(_pretty(resp))
            return resp
        except Exception as e:
            print(f"<<< error: {e!r}")
            raise

    RetrievalClient._send = logged_send  # type: ignore[method-assign]


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        if getattr(m, "tool_calls", None):
            continue
        if m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _build_graph(
    *,
    ace: AgentContextClient,
    session_id: str,
    auth,
    agent_id: str,
    model_with_tools,
    tools,
    lifecycle: dict[str, Any],
    owner_id: str | None = None,
    owner_type: str | None = None,
):
    """Graph: pre_turn → agent/tools → post_turn."""

    async def pre_turn(state: ChatState) -> dict:
        if not lifecycle["opened"]:
            print(f"[node pre_turn] ace_start_conversation session={session_id}")
            open_kwargs: dict[str, Any] = {
                "conversation_id": session_id,
                "title": "client-direct-demo",
                "auth": auth,
                "agent_id": agent_id,
            }
            if owner_id:
                open_kwargs["owner_id"] = owner_id
            if owner_type:
                open_kwargs["owner_type"] = owner_type
            opened = await asyncio.to_thread(ace.start_conversation, **open_kwargs)
            lifecycle["opened"] = True
            lifecycle["pack_ids"] = [p for p in (opened.get("pack_ids") or []) if p]
            suffix = opened.get("system_prompt_suffix") or (
                f"{CONTEXT_PACK_INSTRUCTION}\n\n{opened.get('pack_catalog_text') or ''}"
            )
            ctx_snippet = json.dumps(opened.get("context") or {}, default=str)[:2500]
            system = (
                "You are a helpful assistant with access to Salesforce Data 360 Context.\n"
                f"conversation_id={session_id}\n"
                "Use ace_get_context for customer memory mid-turn when needed.\n"
                "Use ace_get_context_pack for structured pack data (ids from the "
                "available-packs block). Do not assume a business domain.\n\n"
                f"{suffix}\n\n"
                f"customer_context (seeded at session open):\n{ctx_snippet}\n"
            )
            # Kept out of checkpointed state; agent_node prepends it per call.
            lifecycle["system"] = system
            print(
                f"[node pre_turn] opened pack_ids={lifecycle['pack_ids']} "
                f"context_keys={list((opened.get('context') or {}).keys())}"
            )

        user_text = _last_human_text(state["messages"])
        if user_text:
            print(f"[node pre_turn] ace_ingest_customer_message {user_text!r}")
            await asyncio.to_thread(ace.ingest_customer_message, user_text, session_id)
        return {}

    async def agent_node(state: ChatState) -> dict[str, list[BaseMessage]]:
        system = SystemMessage(content=lifecycle["system"])
        reply = await model_with_tools.ainvoke([system, *state["messages"]])
        return {"messages": [reply]}

    async def post_turn(state: ChatState) -> dict:
        reply = _last_ai_text(state["messages"])
        if reply and lifecycle["opened"]:
            print(f"[node post_turn] ace_ingest_agent_message {reply[:120]!r}...")
            await asyncio.to_thread(ace.ingest_agent_message, reply, session_id)
        return {}

    def _route_after_agent(state: ChatState) -> str:
        return "tools" if tools_condition(state) == "tools" else "post_turn"

    builder: StateGraph = StateGraph(ChatState)
    builder.add_node("pre_turn", pre_turn)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("post_turn", post_turn)
    builder.add_edge(START, "pre_turn")
    builder.add_edge("pre_turn", "agent")
    builder.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "post_turn": "post_turn"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("post_turn", END)
    return builder.compile(checkpointer=InMemorySaver())


def _scripted_turns(pack_ids: list[str]) -> list[str]:
    turns = ["hi"]
    if pack_ids:
        turns.append(
            f"Please use context pack {pack_ids[0]} if helpful, "
            "otherwise say what you know from context."
        )
    else:
        turns.append("What do you already know about me from context?")
    return turns


async def main_async(
    *,
    scripted: bool,
    owner_id: str | None = None,
    owner_type: str | None = None,
) -> None:
    _install_pretty_wire_hooks()
    agent_id = _require("BEACON_AGENT_ID")
    # Optional: unset means api.anthropic.com.
    gateway_base = os.environ.get("ANTHROPIC_BASE_URL") or None
    auth = build_auth(agent_id)
    ace = AgentContextClient()
    session_id = f"client-direct-{uuid.uuid4().hex[:8]}"
    lifecycle: dict[str, Any] = {"opened": False, "pack_ids": [], "system": ""}

    if owner_id and not owner_type:
        owner_type = "individual" if owner_id.startswith("003") else None

    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    mid_specs = list(ace.tool_specs())
    assert [s["name"] for s in mid_specs] == [
        "ace_get_context",
        "ace_get_context_pack",
    ], mid_specs
    _JSON_TO_PY = {
        "string": str,
        "object": dict,
        "number": float,
        "integer": int,
        "boolean": bool,
    }

    # ToolNode runs one reply's tool calls in parallel threads; the client
    # isn't safe for concurrent calls, so serialize them.
    tool_lock = threading.Lock()

    def _make_tool(spec: dict) -> StructuredTool:
        name = spec["name"]
        props = (spec.get("parameters") or {}).get("properties") or {}
        required = set((spec.get("parameters") or {}).get("required") or [])
        fields: dict[str, Any] = {}
        for key, sch in props.items():
            py = _JSON_TO_PY.get((sch or {}).get("type"), Any)
            desc = (sch or {}).get("description") or ""
            if key in required:
                fields[key] = (py, Field(description=desc))
            else:
                fields[key] = (Optional[py], Field(default=None, description=desc))
        ArgsModel = create_model(f"{name}_Args", **fields)

        def _fn(**kwargs: object) -> dict:
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            print(f"[tool] {name} args={cleaned}")
            try:
                with tool_lock:
                    return ace.call_tool(name, cleaned)
            except (ValueError, RuntimeError) as e:
                # Bad tool call: tell the model instead of aborting the graph run
                return {"status": "error", "message": str(e)}

        _fn.__name__ = name
        return StructuredTool.from_function(
            func=_fn,
            name=name,
            description=spec["description"],
            args_schema=ArgsModel,
        )

    tools = [_make_tool(s) for s in mid_specs]

    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    model = ChatAnthropic(
        model=model_name,
        api_key=anthropic_api_key(),
        base_url=gateway_base,
        max_tokens=2048,
    )
    model_with_tools = model.bind_tools(tools)
    graph = _build_graph(
        ace=ace,
        session_id=session_id,
        auth=auth,
        agent_id=agent_id,
        model_with_tools=model_with_tools,
        tools=tools,
        lifecycle=lifecycle,
        owner_id=owner_id,
        owner_type=owner_type,
    )

    print("\n=== AgentContextClient — direct method calls ===")
    print(f"session={session_id} (open on first pre_turn)")
    if owner_id:
        print(
            f"owner_id={owner_id} owner_type={owner_type or '(auth default)'} "
            f"(auth user={getattr(auth, 'user_id', None)!r})"
        )
    else:
        print(
            f"owner_id=(auth user {getattr(auth, 'user_id', None)!r}) "
            "— pass --owner-id for a customer id"
        )
    print("Type a message, or /quit to exit.\n")

    turns = _scripted_turns([]) if scripted else None
    turn_i = 0
    while True:
        if turns is not None:
            if turn_i >= len(turns):
                break
            if turn_i == 1 and lifecycle["pack_ids"]:
                turns = _scripted_turns(lifecycle["pack_ids"])
            text = turns[turn_i]
            turn_i += 1
            print(f"you> {text}")
        else:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text or text in ("/quit", "/exit", "quit"):
                break

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": session_id}},
        )
        reply = _last_ai_text(result["messages"])
        tool_names = [
            tc.get("name")
            for m in result["messages"]
            for tc in (getattr(m, "tool_calls", None) or [])
        ]
        if tool_names:
            print(f"[llm tools called] {tool_names}")
        print(f"agent> {reply}\n")

    if scripted and lifecycle["opened"]:
        print("[verify] ace_get_context")
        ctx = ace.get_context()
        print(
            f"  status={ctx.get('status')} "
            f"has_userContext={bool((ctx.get('context') or {}).get('userContext'))}"
        )
        packs = lifecycle["pack_ids"]
        if packs:
            print(f"[verify] ace_get_context_pack {packs[0]!r}")
            pack = ace.get_context_pack(context_pack_id=packs[0], params={})
            print(
                f"  status={pack.get('status')} "
                f"keys={list((pack.get('context') or {}).keys())}"
            )

    print(f"Done. session={session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Non-interactive: hi + pack follow-up then exit (CI / recording)",
    )
    parser.add_argument(
        "--owner-id",
        default=os.environ.get("BEACON_OWNER_ID"),
        help="Customer owner_id for AgentContextClient (defaults to $BEACON_OWNER_ID, "
        "else the signed-in UserContextAuth user).",
    )
    parser.add_argument(
        "--owner-type",
        default=None,
        help="Owner type (e.g. individual, employee). "
        "Defaults to individual when --owner-id starts with 003.",
    )
    args = parser.parse_args()
    asyncio.run(
        main_async(
            scripted=args.scripted,
            owner_id=args.owner_id,
            owner_type=args.owner_type,
        )
    )


if __name__ == "__main__":
    main()
