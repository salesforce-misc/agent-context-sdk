from __future__ import annotations

import warnings

import pytest

import agentic_context_foundry.adapter as adapter_mod
from agentic_context_foundry.adapter import _SDK_INSTALL_HINT, _SDK_MODULE, _require_sdk


class _Auth:
    def get_token(self) -> tuple[str, str]:
        return "token", "http://beacon.test"

    def attach(self, _client: object) -> None:
        return None


def _block_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real = adapter_mod.importlib.import_module

    def _fake(name: str, *args: object, **kwargs: object) -> object:
        if name == _SDK_MODULE:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(adapter_mod.importlib, "import_module", _fake)


def test_require_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    # SDK installed in the dev venv → no-op.
    _require_sdk("run")

    _block_import(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        _require_sdk("run")

    message = str(excinfo.value)
    assert "FoundryAdapter.run()" in message
    assert _SDK_INSTALL_HINT in message
    assert excinfo.value.__cause__ is not None


def test_constructor_warns_only_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        adapter = adapter_mod.FoundryAdapter(agent_id="bot", owner_id="user@x.com", auth=_Auth())

    assert not [w for w in record if _SDK_INSTALL_HINT in str(w.message)]
    assert adapter.capabilities.working is True
    assert not hasattr(adapter, "context_mode")

    monkeypatch.setattr(adapter_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.warns(UserWarning) as record:
        adapter_mod.FoundryAdapter(agent_id="bot", owner_id="user@x.com", auth=_Auth())

    text = str(record[0].message)
    assert _SDK_MODULE in text
    assert _SDK_INSTALL_HINT in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"owner_id": "user@x.com"}, "credentials"),
        ({"auth": _Auth()}, "requires owner_id"),
    ],
    ids=["credentials-missing", "owner-id-missing"],
)
def test_constructor_validation(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, match: str,
) -> None:
    monkeypatch.setattr(adapter_mod.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(ValueError, match=match):
        adapter_mod.FoundryAdapter(agent_id="bot", **kwargs)
