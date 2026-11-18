#!/usr/bin/env bash
# One-shot setup shared by every demo under examples/.
#
#   ./examples/setup.sh
#   source .venv/bin/activate
#   python examples/<demo>/<demo>.py
#
# Safe to re-run: reuses an existing .venv and only (re)installs deps.
# Override the interpreter with PYTHON=/path/to/python3.11 if needed.
set -euo pipefail

PYENV_VERSION_WANTED="3.11.14"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
cd "$REPO_ROOT"

# 1. Create .venv if it doesn't exist yet.
if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ -z "${PYTHON:-}" ]]; then
    # Equivalent to `pyenv shell $PYENV_VERSION_WANTED && python3.11`, but
    # resolves the real interpreter instead of going through the shim (which
    # fails when no 3.11 is selected). Avoid `pyenv versions | grep -q`:
    # under pipefail, grep closing early SIGPIPEs pyenv and the test fails.
    PYENV_PREFIX=""
    if command -v pyenv >/dev/null 2>&1; then
      PYENV_PREFIX="$(pyenv prefix "$PYENV_VERSION_WANTED" 2>/dev/null || true)"
    fi
    if [[ -n "$PYENV_PREFIX" && -x "$PYENV_PREFIX/bin/python3.11" ]]; then
      PYTHON="$PYENV_PREFIX/bin/python3.11"
    elif python3.11 -c '' >/dev/null 2>&1; then
      PYTHON="$(command -v python3.11)"
    else
      echo "error: Python 3.11 not found." >&2
      echo "  pyenv install $PYENV_VERSION_WANTED   (or set PYTHON=/path/to/python3.11)" >&2
      exit 1
    fi
  fi
  echo "[setup] creating .venv with $PYTHON"
  "$PYTHON" -m venv "$VENV"
else
  echo "[setup] reusing existing .venv ($("$VENV/bin/python" --version))"
fi

# uv-created venvs ship without pip: use uv when it's available, otherwise
# bootstrap pip into the venv.
if "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  PIP=("$VENV/bin/python" -m pip)
  "${PIP[@]}" install --quiet --upgrade pip
elif command -v uv >/dev/null 2>&1; then
  export VIRTUAL_ENV="$VENV"  # uv pip installs into the active venv
  PIP=(uv pip)
else
  "$VENV/bin/python" -m ensurepip --upgrade >/dev/null
  PIP=("$VENV/bin/python" -m pip)
fi

# 2. Install deps — the union of what every demo needs.
echo "[setup] installing dependencies"
"${PIP[@]}" install --quiet -e .                                        # the SDK itself
"${PIP[@]}" install --quiet langgraph langchain-anthropic langchain-core  # interactive_agent
"${PIP[@]}" install --quiet claude-agent-sdk                              # quickstart_demo
"${PIP[@]}" install --quiet anyio httpx                                   # ace_basic_demo, non_sdk_basic_demo

# 3. Sanity-check the imports the demos need.
"$VENV/bin/python" -c "import agentic_context, agentic_context_langgraph, langgraph, langchain_anthropic, claude_agent_sdk, anyio, httpx" \
  && echo "[setup] imports OK"

# 4. Param file reminder (never overwrite an existing one).
if [[ ! -f examples/ace_agent.env ]]; then
  cp examples/ace_agent.env.example examples/ace_agent.env
  echo "[setup] created examples/ace_agent.env from the template — edit it before running"
fi

cat <<EOF

[setup] done. Next:
  source .venv/bin/activate
  python examples/interactive_agent/interactive_agent.py
  python examples/quickstart_demo/quickstart_demo.py "hello"
  python examples/ace_basic_demo/ace_basic_demo.py list
  python examples/non_sdk_basic_demo/non_sdk_basic_demo.py list
EOF
