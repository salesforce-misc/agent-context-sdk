#!/usr/bin/env python
# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Run adapter smoke tests — one adapter, several, or all; one tier or both.

    python smoke/run_smoke.py                          # every adapter, offline
    python smoke/run_smoke.py --adapter adk openai     # just these, offline
    python smoke/run_smoke.py --tier live --adapter claude
    python smoke/run_smoke.py --tier all               # offline + live, everything
    python smoke/run_smoke.py --list
    python smoke/run_smoke.py --adapter foundry -- -x -vv   # extra pytest args

Exit code is pytest's, so this drops straight into CI.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SMOKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SMOKE_DIR.parent

ADAPTERS = {
    "langgraph": "LangGraph (agentic_context_langgraph)",
    "claude": "Claude Agent SDK (agentic_context_claude)",
    "adk": "Google ADK (agentic_context_adk)",
    "openai": "OpenAI Agents SDK (agentic_context_openai)",
    "foundry": "Microsoft Agent Framework / Foundry (agentic_context_foundry)",
}
TIERS = {
    "offline": "real framework + scripted LLM + mocked Beacon; no credentials",
    "live": "real framework + scripted LLM + real dev Beacon; needs credentials in examples/ace_agent.env or the shell",
    "all": "both tiers",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--adapter", "-a", nargs="+", default=["all"], choices=[*ADAPTERS, "all"],
        help="adapter(s) to run (default: all)",
    )
    parser.add_argument("--tier", "-t", default="offline", choices=list(TIERS), help="default: offline")
    parser.add_argument("--list", action="store_true", help="list adapters and tiers, then exit")
    args = parser.parse_args(argv)

    if args.list:
        print("adapters:")
        for name, desc in ADAPTERS.items():
            print(f"  {name:<10} {desc}")
        print("tiers:")
        for name, desc in TIERS.items():
            print(f"  {name:<10} {desc}")
        return 0

    names = list(ADAPTERS) if "all" in args.adapter else list(dict.fromkeys(args.adapter))
    files = [str(SMOKE_DIR / f"test_{name}.py") for name in names]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        print(f"missing smoke modules: {missing}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    cmd = [sys.executable, "-m", "pytest", "-c", str(SMOKE_DIR / "pytest.ini"), *files]
    if args.tier == "all":
        env["SMOKE_LIVE"] = "1"
    elif args.tier == "live":
        env["SMOKE_LIVE"] = "1"
        cmd += ["-m", "live"]
    else:
        env.pop("SMOKE_LIVE", None)
        cmd += ["-m", "offline"]
    cmd += extra

    print(f"smoke: adapters={','.join(names)} tier={args.tier}", flush=True)
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
