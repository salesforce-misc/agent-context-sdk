# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Async HTTP transport for the Context Service (Beacon) REST API.

Same surface as IngestionClient — only the I/O is non-blocking (httpx).
Use inside an event loop. Pairs with AsyncSessionManager.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from .auth import ContextAuth
from .client import IngestionError, _json_or_empty, _sanitize_header
from .models import to_ace_timestamp

logger = logging.getLogger(__name__)


class AsyncIngestionClient:
    """Async, non-blocking HTTP client for the Context Service (Beacon) REST API.

    Same four lifecycle methods as IngestionClient, same endpoint paths,
    same auth, same error handling — only the I/O backend differs.

    Use as an async context manager so the connection pool closes cleanly.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        dataspace: str = "default",
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
        auth: ContextAuth | None = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.startswith(("https://", "http://")):
            raise ValueError(
                f"base_url must start with 'https://' or 'http://'; got {base_url!r}"
            )
        if normalized.startswith("http://"):
            logger.warning(
                "AsyncIngestionClient base_url uses http:// (not TLS); "
                "credentials will be sent in clear text"
            )
        self.base_url = normalized
        self.api_key = api_key
        self.dataspace = dataspace
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._auth = auth
        if auth is None:
            # Async client is used by streaming/long-running workloads where
            # a mid-session token expiry yields one 401 → non-retryable →
            # silent record drop in _deliver. With auth=ContextAuth(...) the
            # client refreshes and retries on 401 automatically.
            logger.warning(
                "AsyncIngestionClient constructed with static api_key and no auth=. "
                "Static tokens cannot be refreshed; a mid-session expiry will yield "
                "one 401 and drop the record. For long-running or streaming "
                "workloads, pass auth=ContextAuth(...) so 401s auto-recover."
            )

    async def __aenter__(self) -> AsyncIngestionClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the httpx client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    def _headers(self, owner_id: str, agent_id: str) -> dict[str, str]:
        """Standard headers for all Context Service requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "dataspace": _sanitize_header(self.dataspace),
            "owner_id": _sanitize_header(owner_id),
            "agent_id": _sanitize_header(agent_id),
        }

    # ─── Lifecycle Endpoints ─────────────────────────────────────

    async def create_conversation(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
        *,
        name: str | None = None,
        owner_type: str = "individual",
        channel: dict[str, Any] | None = None,
        variable: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations

        ``channel`` and ``variable`` describe the agent's deployment surface
        and Beacon-side conversation state. The SDK can't derive them —
        callers (adapters, demos) pass them, or get the defaults below.
        Full-override on both: pass a dict to replace the default entirely.
        """
        resolved_name = name or f"Conversation {conversation_id[:12]}"
        payload = {
            "conversationId": conversation_id,
            "conversationOwnerId": owner_id,
            "ownerType": owner_type,
            "name": resolved_name,
            "channel": channel if channel is not None else {"type": "web", "id": "agent-sdk"},
            "variable": variable if variable is not None else {
                "Name": resolved_name,
                "ConversationState": "ACTIVE",
                "ExternalRecordId": f"ext-{conversation_id}",
                "MessageDeliveryShortCodeText": "sdk-default",
                "ConversationExpirationDateTime": "2099-01-01T00:00:00Z",
            },
        }
        return await self._post(
            "/api/v1/agentic/conversations",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    async def create_session(
        self,
        conversation_id: str,
        session_id: str,
        owner_id: str,
        agent_id: str,
        *,
        owner_type: str = "individual",
        variable: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/sessions

        Only the owner participant is included here. The agent participant is
        added in a separate add_participants call so get_context can scope
        correctly by participant ID. ``variable`` is the session-level
        metadata block (AiAgentChannelType, RelatedVoiceCallId, etc.).
        """
        now_iso = to_ace_timestamp()
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "conversationOwnerId": owner_id,
            "ownerType": owner_type,
            "participants": [
                {
                    "id": owner_id,
                    "type": owner_type,
                    "role": "owner",
                    "startTime": now_iso,
                    "endTime": None,
                    "orgId": "default",
                    "variable": {},
                },
            ],
        }
        if variable is not None:
            payload["variable"] = variable
        return await self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/sessions",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    async def add_participants(
        self,
        conversation_id: str,
        session_id: str,
        owner_id: str,
        agent_id: str,
        participants: list[dict[str, Any]],
        *,
        variable: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/sessions/{id}/participants

        ``variable`` is an optional top-level metadata block on the request
        (e.g., ``{"ParticipantAttributeText": ...}``).
        """
        payload: dict[str, Any] = {"participants": participants}
        if variable is not None:
            payload["variable"] = variable
        return await self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/sessions/{quote(session_id, safe='')}/participants",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    async def add_entries(
        self,
        conversation_id: str,
        session_id: str,
        entries: list[dict[str, Any]],
        owner_id: str,
        agent_id: str,
        *,
        variable: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/entries

        Body shape verified against the live Beacon API.
        ``variable`` is an optional top-level metadata block on the request
        (e.g., ``{"VariableText": ...}``).
        """
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "conversationOwnerId": owner_id,
            "entries": entries,
        }
        if variable is not None:
            payload["variable"] = variable
        return await self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/entries",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    async def close_session(
        self,
        conversation_id: str,
        session_id: str,
        owner_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/sessions/{id}/close"""
        return await self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/sessions/{quote(session_id, safe='')}/close",
            None,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    async def close_conversation(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/close"""
        return await self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/close",
            None,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    # ─── Internal ────────────────────────────────────────────────

    async def _post(
        self, path: str, payload: dict[str, Any] | None, *, owner_id: str, agent_id: str
    ) -> dict[str, Any]:
        """Send async POST request with standard headers."""
        resp = await self._client.post(
            self.base_url + path,
            json=payload,
            headers=self._headers(owner_id, agent_id),
        )
        if resp.status_code == 401 and self._auth is not None:
            self._auth.invalidate()
            self.api_key, _ = await self._auth.aget_token()
            resp = await self._client.post(
                self.base_url + path,
                json=payload,
                headers=self._headers(owner_id, agent_id),
            )
        if resp.status_code >= 400:
            raise IngestionError(resp.status_code, resp.text)
        return _json_or_empty(
            resp.status_code, resp.text, resp.json, resp.headers.get("content-type", "")
        )
