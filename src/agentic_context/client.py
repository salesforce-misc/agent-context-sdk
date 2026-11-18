# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Sync HTTP transport for the Context Service (Beacon) REST API.

Handles all lifecycle endpoints:
- Create/close conversations
- Create/close sessions
- Add entries

Single-purpose. No retry, no queue, no rate-limit handling — those are the caller's concern.
Use as a context manager so the connection pool closes cleanly.
"""
from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict

from .models import to_ace_timestamp

if TYPE_CHECKING:
    from .auth import ContextAuth

logger = logging.getLogger(__name__)


def _json_or_empty(
    status_code: int, text: str, parse: Callable[[], Any], content_type: str = ""
) -> dict[str, Any]:
    """Decode an ingestion 2xx body; a non-JSON or non-object body maps to ``{}``.

    The write already landed, so raising here would make a retry re-send it
    (requests' JSONDecodeError subclasses OSError) and repeat the write. Log
    and return ``{}`` instead.
    """
    if not text:
        return {}
    try:
        data = parse()
    except ValueError:
        logger.warning(
            "Context Service returned %s with a non-JSON body "
            "(content-type %r, body %r); treating the write as accepted",
            status_code, content_type, text[:200],
        )
        return {}
    return data if isinstance(data, dict) else {}


def _json_or_raise(
    status_code: int, text: str, parse: Callable[[], Any], content_type: str = ""
) -> Any:
    """Decode a retrieval 2xx body. Objects and lists pass through; other JSON
    values map to ``{}``.

    A non-JSON body (e.g. a proxy or a wrong ``base_url`` serving HTML) raises
    :class:`IngestionError` instead of reading as "no context". A 2xx status is
    not retryable, so nothing is re-sent.
    """
    if not text:
        return {}
    try:
        data = parse()
    except ValueError:
        raise IngestionError(
            status_code,
            f"non-JSON response body (content-type {content_type!r}): {text[:200]}",
        ) from None
    return data if isinstance(data, (dict, list)) else {}


def _sanitize_header(value: str) -> str:
    """Strip CR/LF/null bytes from header values to neutralize injection."""
    return value.replace("\r", "").replace("\n", "").replace("\x00", "")


class IngestionResponse(BaseModel):
    """Server response from a successful POST."""

    model_config = ConfigDict(extra="allow", frozen=True)

    accepted: int
    rejected: int = 0
    errors: list[dict[str, Any]] = []


class IngestionError(RuntimeError):
    """Raised when the server returns a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"ingestion failed: HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class IngestionClient:
    """Sync, blocking HTTP client for the Context Service (Beacon) REST API.

    Handles all lifecycle endpoints:
    - Create/close conversations
    - Create/close sessions
    - Add entries

    Use as a context manager so the connection pool closes cleanly.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        dataspace: str = "default",
        timeout: float = 10.0,
        session: requests.Session | None = None,
        auth: "ContextAuth | None" = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.startswith(("https://", "http://")):
            raise ValueError(
                f"base_url must start with 'https://' or 'http://'; got {base_url!r}"
            )
        if normalized.startswith("http://"):
            logger.warning(
                "IngestionClient base_url uses http:// (not TLS); "
                "credentials will be sent in clear text"
            )
        self.base_url = normalized
        self.api_key = api_key
        self.dataspace = dataspace
        self.timeout = timeout
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._auth = auth

    def __enter__(self) -> IngestionClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session:
            self._session.close()

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

    def create_conversation(
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
        return self._post(
            "/api/v1/agentic/conversations",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    def create_session(
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
        return self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/sessions",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    def add_participants(
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
        return self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/sessions/{quote(session_id, safe='')}/participants",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    def add_entries(
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

        Body shape verified against the live Beacon API:
        sessionId + conversationOwnerId + entries. ``variable`` is an
        optional top-level metadata block on the request.
        """
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "conversationOwnerId": owner_id,
            "entries": entries,
        }
        if variable is not None:
            payload["variable"] = variable
        return self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/entries",
            payload,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    def close_session(
        self,
        conversation_id: str,
        session_id: str,
        owner_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/sessions/{id}/close"""
        return self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/sessions/{quote(session_id, safe='')}/close",
            None,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    def close_conversation(
        self,
        conversation_id: str,
        owner_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """POST /api/v1/agentic/conversations/{id}/close"""
        return self._post(
            f"/api/v1/agentic/conversations/{quote(conversation_id, safe='')}/close",
            None,
            owner_id=owner_id,
            agent_id=agent_id,
        )

    # ─── Internal ────────────────────────────────────────────────

    def _post(
        self, path: str, payload: dict[str, Any] | None, *, owner_id: str, agent_id: str
    ) -> dict[str, Any]:
        """Send POST request with standard headers."""
        resp = self._session.post(
            self.base_url + path,
            json=payload,
            headers=self._headers(owner_id, agent_id),
            timeout=self.timeout,
        )
        if resp.status_code == 401 and self._auth is not None:
            self._auth.invalidate()
            self.api_key, _ = self._auth.get_token()
            resp = self._session.post(
                self.base_url + path,
                json=payload,
                headers=self._headers(owner_id, agent_id),
                timeout=self.timeout,
            )
        if not resp.ok:
            raise IngestionError(resp.status_code, resp.text)
        return _json_or_empty(
            resp.status_code, resp.text, resp.json, resp.headers.get("content-type", "")
        )
