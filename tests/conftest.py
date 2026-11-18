# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

from __future__ import annotations

import contextlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
import responses as responses_lib
import respx

from agentic_context import Capabilities, Record

# Shared across retrieval tests so the ``/api/v1/agentic/context`` path is
# defined once. Matches the ``BASE_URL`` defined in those tests.
CONTEXT_URL = "https://beacon.example.com/api/v1/agentic/context"


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def record(now: datetime) -> Record:
    return Record(
        event_id="evt-1",
        session_id="ses-1",
        agent_id="agent-1",
        event_type="user_message",
        participant_id="user-001",
        content="hello",
        category="Message",
        interaction_id="turn-001",
        timestamp=now,
    )


@pytest.fixture
def trace_record(now: datetime) -> Record:
    return Record(
        event_id="step-1",
        session_id="ses-1",
        agent_id="agent-1",
        event_type="tool_call",
        participant_id="agent-1",
        content=None,
        category="Trace",
        interaction_id="turn-001",
        variable={"Name": "OrderLookup", "InputValueText": '{"orderId":"123"}'},
        timestamp=now,
    )


@pytest.fixture
def working_caps() -> Capabilities:
    return Capabilities(working=True, persistent=False)


# ─── Sync/async HTTP harness ──────────────────────────────────────────

@dataclass
class _Req:
    headers: Any
    url: str
    params: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


class _Http:
    """One mock HTTP backend + client factory for a sync/async client pair."""

    def __init__(self, is_async: bool, mock: Any, module: Any) -> None:
        self.is_async = is_async
        self._mock = mock
        self._module = module
        self._queued: dict[tuple[str, str], list] = {}
        self._routes: dict[tuple[str, str], Any] = {}

    def add(self, method: str, url: str, status: int = 200, *, json: Any = None,
            body: bytes | None = None, content_type: str | None = None,
            times: int = 1) -> None:
        """Queue ``times`` identical responses for ``method url``; responses for
        the same route are served in the order they were added."""
        for _ in range(times):
            if self.is_async:
                headers = {"content-type": content_type} if content_type else None
                resp = (httpx.Response(status, content=body, headers=headers)
                        if body is not None
                        else httpx.Response(status, json=json))
                key = (method, url)
                self._queued.setdefault(key, []).append(resp)
                self._routes[key] = self._mock.route(method=method, url=url).mock(
                    side_effect=list(self._queued[key])
                )
            elif body is not None:
                self._mock.add(method, url, body=body, status=status,
                               content_type=content_type or "text/plain")
            else:
                self._mock.add(method, url, json=json, status=status)

    @property
    def calls(self) -> list[_Req]:
        out = []
        for call in self._mock.calls:
            r = call.request
            if self.is_async:
                out.append(_Req(r.headers, str(r.url), dict(r.url.params), r.content))
            else:
                out.append(_Req(r.headers, r.url, dict(r.params), r.body or b""))
        return out

    @contextlib.asynccontextmanager
    async def client(self, **kwargs: Any):
        """Open the module's sync or async client (``HTTP_CLIENTS``) on its
        ``BASE_URL``/``API_KEY``."""
        kwargs = {"base_url": self._module.BASE_URL, "api_key": self._module.API_KEY, **kwargs}
        sync_cls, async_cls = self._module.HTTP_CLIENTS
        if self.is_async:
            async with async_cls(**kwargs) as c:
                yield c
        else:
            with sync_cls(**kwargs) as c:
                yield c

    async def run(self, value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value

    def assert_all_consumed(self) -> None:
        # respx's assert_all_called only needs one hit per route; check every
        # queued response was served, matching responses' assert_all_requests_are_fired.
        for key, route in self._routes.items():
            assert route.call_count == len(self._queued[key]), (
                f"{key}: {len(self._queued[key])} responses queued, {route.call_count} served"
            )


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def http(request):
    """Runs a test once against the sync client (``responses``) and once against
    the async client (``respx``). Fails the test on any unmocked request and on
    any registered response that is never consumed, so a harness mistake cannot
    silently pass both clients."""
    if request.param:
        with respx.mock(assert_all_called=True, assert_all_mocked=True) as router:
            h = _Http(True, router, request.module)
            yield h
            h.assert_all_consumed()
    else:
        with responses_lib.RequestsMock(assert_all_requests_are_fired=True) as rsps:
            yield _Http(False, rsps, request.module)
