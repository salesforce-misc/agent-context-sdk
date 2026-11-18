# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Offline, re-runnable gate for the unified single-wheel packaging.

The six import packages (``agentic_context`` core plus the ``_langgraph`` /
``_adk`` / ``_claude`` / ``_foundry`` / ``_openai`` adapters) ship inside one
``agent-context-sdk`` wheel even though the adapter sources live out-of-tree
under ``adapters/<fw>/src/``. This test builds the real wheel and inspects the
artifact (never the source tree) to prove:

* all six packages land at the wheel root with NO ``adapters/`` prefix,
* the distribution advertises NO framework extras (single `pip install
  agent-context-sdk` is the only install form — the framework SDKs are expected
  to already be present because the user runs that framework),
* the framework SDKs (langgraph / google-adk / claude-agent-sdk /
  agent-framework-foundry / openai-agents) are NOT declared as runtime
  dependencies, and
* the branding strip held (neutral author, no salesforce URLs/keywords).

It is deliberately offline: it builds with ``--no-isolation`` so it uses the
already-installed backend rather than downloading one, and it SKIPS (never
fails) when the ``build``/``hatchling`` backend is absent, so the core suite
baseline never regresses in environments that lack the build tooling.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The six import packages that MUST be present at the wheel root.
EXPECTED_PACKAGES = {
    "agentic_context",
    "agentic_context_langgraph",
    "agentic_context_adk",
    "agentic_context_claude",
    "agentic_context_foundry",
    "agentic_context_openai",
}

# Framework extras that must NOT be advertised: the framework SDKs are neither
# core deps nor opt-in extras — a single `pip install agent-context-sdk` is it.
FORBIDDEN_EXTRAS = {"langgraph", "adk", "claude", "foundry", "openai", "all"}

# The framework SDK distributions that must NOT appear as runtime dependencies.
FRAMEWORK_SDKS = (
    "langgraph",
    "langchain-core",
    "google-adk",
    "claude-agent-sdk",
    "agent-framework-foundry",
    "openai-agents",
)


def _build_backend_available() -> bool:
    """True only if we can build the wheel offline (build + hatchling present)."""
    try:
        import build  # noqa: F401
        import hatchling  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the real wheel once and return its path, or skip if unbuildable."""
    if not _build_backend_available():
        pytest.skip("build/hatchling not installed; wheel-structure gate skipped")

    out_dir = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(out_dir),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"wheel build unavailable in this environment:\n{proc.stderr}")

    wheels = list(out_dir.glob("agent_context_sdk-*.whl"))
    assert wheels, f"no wheel produced in {out_dir}: {proc.stdout}\n{proc.stderr}"
    return wheels[0]


def _top_level_dirs(whl: Path) -> set[str]:
    with zipfile.ZipFile(whl) as zf:
        return {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}


def _metadata(whl: Path) -> str:
    with zipfile.ZipFile(whl) as zf:
        meta_name = next(
            n for n in zf.namelist() if n.endswith(".dist-info/METADATA")
        )
        return zf.read(meta_name).decode("utf-8")


def test_wheel_contains_all_packages(built_wheel: Path) -> None:
    top = _top_level_dirs(built_wheel)
    missing = EXPECTED_PACKAGES - top
    assert not missing, f"wheel is missing top-level packages: {sorted(missing)}"


def test_wheel_has_no_adapters_prefix(built_wheel: Path) -> None:
    # Regression guard: the adapters must NOT be nested under an `adapters/`
    # namespace inside the wheel — hatchling must strip the source prefix so
    # each adapter imports as its own top-level package.
    with zipfile.ZipFile(built_wheel) as zf:
        nested = [n for n in zf.namelist() if n.startswith("adapters/")]
    assert not nested, f"adapter sources leaked under adapters/ in wheel: {nested[:5]}"


def test_wheel_ships_license(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as zf:
        licenses = [n for n in zf.namelist() if n.endswith(".dist-info/licenses/LICENSE.txt")]
    assert licenses, "LICENSE.txt missing from wheel dist-info/licenses/"


def test_license_metadata_is_pep639_expression(built_wheel: Path) -> None:
    # PEP 639: an SPDX expression plus License-File, not the full text pasted
    # into a `License:` header (which PyPI renders verbatim in the sidebar).
    msg = Parser().parsestr(_metadata(built_wheel))
    assert msg.get("License-Expression") == "LicenseRef-Salesforce-Public-Code-License"
    assert msg.get("License") is None
    assert "LICENSE.txt" in (msg.get_all("License-File") or [])
    assert not [c for c in msg.get_all("Classifier") or [] if c.startswith("License ::")]


def test_wheel_contains_ace_mid_turn_tools_skill(built_wheel: Path) -> None:
    # AgentContextClient.tool_skill_dir() must resolve after `pip install` — the
    # packaged skill is force-included under agentic_context/skills/.
    expected = "agentic_context/skills/ace-mid-turn-tools/SKILL.md"
    with zipfile.ZipFile(built_wheel) as zf:
        names = set(zf.namelist())
    assert expected in names, (
        f"wheel missing AgentContextClient skill {expected!r}; "
        f"skills present={[n for n in sorted(names) if 'skills/' in n][:20]}"
    )


def test_wheel_advertises_no_framework_extras(built_wheel: Path) -> None:
    # Single `pip install agent-context-sdk` is the only install form: the
    # framework SDKs are neither core deps nor opt-in extras. So none of the
    # per-framework extras (nor the fan-out `all`) may be advertised.
    msg = Parser().parsestr(_metadata(built_wheel))
    provided = set(msg.get_all("Provides-Extra") or [])
    leaked = FORBIDDEN_EXTRAS & provided
    assert not leaked, f"wheel unexpectedly advertises framework extras: {sorted(leaked)}"


def test_no_self_reference_to_agentic_context(built_wheel: Path) -> None:
    # With the `all` fan-out extra gone, NOTHING may pull `agent-context-sdk` back
    # in as a self-dependency.
    msg = Parser().parsestr(_metadata(built_wheel))
    offenders = [
        req
        for req in (msg.get_all("Requires-Dist") or [])
        if req.lower().startswith(("agentic-context", "agent-context-sdk"))
    ]
    assert not offenders, f"unexpected agent-context-sdk self-reference(s): {offenders}"


def _requires_dist(whl: Path) -> list[str]:
    """The wheel's ``Requires-Dist`` lines (one per dependency)."""
    msg = Parser().parsestr(_metadata(whl))
    return list(msg.get_all("Requires-Dist") or [])


def test_framework_sdks_are_not_runtime_deps(built_wheel: Path) -> None:
    # The framework SDKs must never be imposed on end users: not as core deps and
    # (since extras are gone) not via any user-facing extra. They MAY appear
    # under the `dev` extra — that only affects contributors running the test
    # suite, never a `pip install agent-context-sdk`. Match the distribution name
    # at the START of a Requires-Dist line.
    reqs = _requires_dist(built_wheel)
    for dep in FRAMEWORK_SDKS:
        offenders = [
            r
            for r in reqs
            if r.lower().startswith(dep) and "extra == 'dev'" not in r
        ]
        assert not offenders, (
            f"{dep} must not be a user-facing runtime dependency "
            f"(single-install policy), got: {offenders}"
        )


def test_cryptography_is_an_unconditional_core_dep(built_wheel: Path) -> None:
    # ContextAuth (core auth.py) signs the JWT with cryptography, and every
    # adapter's __init__ mandates auth credentials, so cryptography is a hard
    # CORE requirement — it must be an unconditional Requires-Dist (no
    # `extra == ...` marker), guaranteeing the ContextAuth quickstart works on a
    # bare `pip install agent-context-sdk` regardless of transitive resolution.
    reqs = _requires_dist(built_wheel)
    crypto = [r for r in reqs if r.lower().startswith("cryptography")]
    assert crypto, f"cryptography is not declared as a Requires-Dist: {reqs}"
    unconditional = [r for r in crypto if "extra ==" not in r]
    assert unconditional, (
        f"cryptography must be a core dep with no extra marker, got: {crypto}"
    )


def test_wheel_version_matches_package(built_wheel: Path) -> None:
    import agentic_context

    msg = Parser().parsestr(_metadata(built_wheel))
    assert msg.get("Version") == agentic_context.__version__


def test_branding_stripped_from_metadata(built_wheel: Path) -> None:
    # US-12: no Salesforce-identifying author/keyword survives into the
    # publishable metadata. `name` is agent-context-sdk. Project URLs are the
    # exception: they must point at the public source repo, and nowhere else.
    meta = _metadata(built_wheel)
    msg = Parser().parsestr(meta)
    assert "salesforce" not in (msg.get("Author") or "").lower()
    assert "salesforce" not in (msg.get("Keywords") or "").lower()
    urls = msg.get_all("Project-URL") or []
    assert urls, "Project-URL metadata missing"
    for url in urls:
        assert "https://github.com/salesforce-misc/agent-context-sdk" in url, url


# ─── sdist content gate ────────────────────────────────────────────────────
#
# The wheel tests above verify the runtime install artifact. The sdist is a
# separate publishable artifact: PyPI serves it to any `pip install` that runs
# with `--no-binary :all:` (and Linux distros/build farms rebuild from it), so
# a tarball that leaks internal E2E harness code reaches customers just as
# surely as a wheel that does.
#
# The bug this guards against: hatchling's plain `include` list under
# `[tool.hatch.build.targets.sdist]` is ADDITIVE against the target's default
# file selection, not a whitelist. Before pyproject.toml switched to
# `only-include`, the published sdist carried internal test harnesses, all
# `adapters/*/tests/`, per-adapter `pyproject.toml`s, and per-adapter READMEs
# — none of which a public customer should ever receive.


# Paths that MUST NOT appear in the sdist. Substrings are matched against the
# tarball's member names (which are prefixed with `agent_context_sdk-X.Y.Z/`), so
# a leading `/` anchors the match at the tarball root.
FORBIDDEN_SDIST_SUBSTRINGS = (
    "/adapters/adk/tests/",
    "/adapters/claude/tests/",
    "/adapters/langgraph/tests/",
    "/adapters/foundry/tests/",
    "/adapters/openai/tests/",
    "/adapters/adk/pyproject.toml",
    "/adapters/claude/pyproject.toml",
    "/adapters/langgraph/pyproject.toml",
    "/adapters/foundry/pyproject.toml",
    "/adapters/openai/pyproject.toml",
    "/adapters/adk/README.md",
    "/adapters/claude/README.md",
    "/adapters/langgraph/README.md",
    "/adapters/foundry/README.md",
    "/adapters/openai/README.md",
    "/adapters/README.md",
    # Dev-only smoke suite; never part of the published sdist.
    "/smoke/",
)

# Paths that MUST appear in the sdist. Without any one of these, an sdist-only
# install (`pip install --no-binary :all: agent-context-sdk`) cannot rebuild the
# wheel — the whitelist would have over-stripped.
EXPECTED_SDIST_SUBSTRINGS = (
    "/src/agentic_context/__init__.py",
    "/adapters/adk/src/agentic_context_adk/__init__.py",
    "/adapters/claude/src/agentic_context_claude/__init__.py",
    "/adapters/langgraph/src/agentic_context_langgraph/__init__.py",
    "/adapters/foundry/src/agentic_context_foundry/__init__.py",
    "/adapters/openai/src/agentic_context_openai/__init__.py",
    "/tests/",
    "/README.md",
    "/pyproject.toml",
    "/LICENSE.txt",
)


@pytest.fixture(scope="module")
def built_sdist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the real sdist once and return its path, or skip if unbuildable."""
    if not _build_backend_available():
        pytest.skip("build/hatchling not installed; sdist-content gate skipped")

    out_dir = tmp_path_factory.mktemp("dist-sdist")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(out_dir),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # hatchling/build ARE installed (checked above) — a nonzero build here is
        # a real packaging regression (e.g. invalid only-include entry pointing at
        # a path that no longer exists), not an environment issue. Fail loudly with
        # the build output so CI catches it instead of silently green-lighting.
        pytest.fail(
            "sdist build failed with hatchling installed — packaging regression.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    sdists = list(out_dir.glob("agent_context_sdk-*.tar.gz"))
    assert sdists, f"no sdist produced in {out_dir}: {proc.stdout}\n{proc.stderr}"
    return sdists[0]


def _sdist_names(tarball: Path) -> list[str]:
    with tarfile.open(tarball) as tf:
        return tf.getnames()


def test_sdist_omits_internal_paths(built_sdist: Path) -> None:
    # Regression guard for the sdist `include` → `only-include` switch in
    # pyproject.toml. Internal E2E harnesses, per-framework adapter
    # tests, per-adapter dev configs, and adapter READMEs must never leak to
    # the published tarball.
    names = _sdist_names(built_sdist)
    leaks = sorted(
        {n for n in names for pat in FORBIDDEN_SDIST_SUBSTRINGS if pat in n}
    )
    assert not leaks, f"sdist leaks internal paths: {leaks}"


def test_sdist_contains_expected_paths(built_sdist: Path) -> None:
    # Positive sanity check: the strict whitelist didn't over-strip. The five
    # package sources plus root tests and metadata files MUST ship so an
    # sdist-only install can rebuild the wheel.
    names = _sdist_names(built_sdist)
    missing = [
        pat for pat in EXPECTED_SDIST_SUBSTRINGS if not any(pat in n for n in names)
    ]
    assert not missing, f"sdist is missing expected paths: {missing}"
