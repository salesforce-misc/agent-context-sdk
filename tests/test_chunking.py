# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for payload chunking logic (_chunk_entries).

Verifies that entries are correctly split when exceeding the Beacon
200KB per-request payload limit.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_context import Record
from agentic_context.session import AsyncSessionManager, SessionManager, _chunk_entries

# ─── Unit Tests: _chunk_entries ──────────────────────────────────────────────


def _entry(i, content, **extra):
    return {"id": f"msg-{i}", "type": "agent_response", "content": content,
            "timestamp": "2024-01-01T00:00:00Z", "category": "Message", "participantId": "agent-1", **extra}


_SMALL = [_entry(i, "Hello!", type="user_message", participantId="user-1") for i in range(5)]
_TINY = [{"id": "x0", "content": "a"}, {"id": "x1", "content": "a"}, {"id": "x2", "content": "a"}]


class TestChunkEntries:
    @pytest.mark.parametrize(
        ("entries", "max_bytes", "expected"),
        [
            ([], None, []),
            # Typical agent turn: a few small entries → single chunk.
            (_SMALL, None, [_SMALL]),
            # Entries fitting exactly at the boundary stay in one chunk.
            (_TINY, sum(len(json.dumps(e, separators=(",", ":"))) for e in _TINY), [_TINY]),
        ],
        ids=["empty", "small-single-chunk", "exact-boundary"],
    )
    def test_fits_in_one_chunk(self, entries, max_bytes, expected):
        kwargs = {} if max_bytes is None else {"max_bytes": max_bytes}
        assert _chunk_entries(entries, **kwargs) == expected

    @pytest.mark.parametrize(
        ("entries", "max_bytes"),
        [
            # ~50KB each → 4 entries = ~200KB → should split.
            ([_entry(i, "x" * 50_000) for i in range(4)], None),
            # ~60KB each → 3 fit per chunk; order must survive across chunks.
            ([_entry(i, "z" * 60_000, type="test", participantId="x") for i in range(7)], None),
            # Respects custom max_bytes: a very small limit → many chunks.
            ([{"id": f"msg-{i}", "type": "test", "content": "a" * 100} for i in range(10)], 200),
        ],
        ids=["50kb-x4", "60kb-x7", "custom-max-bytes"],
    )
    def test_large_entries_split_in_order(self, entries, max_bytes):
        """Entries exceeding the limit are split into multiple chunks; every entry
        is present exactly once and entry order is maintained across chunks."""
        kwargs = {} if max_bytes is None else {"max_bytes": max_bytes}
        chunks = _chunk_entries(entries, **kwargs)
        assert len(chunks) > 1
        flattened = [entry for chunk in chunks for entry in chunk]
        assert [e["id"] for e in flattened] == [e["id"] for e in entries]

    def test_single_entry_exceeding_limit_is_own_chunk(self):
        """A single entry that exceeds the limit still gets its own chunk."""
        huge_content = "y" * 200_000
        entries = [
            {"id": "msg-huge", "type": "agent_response", "content": huge_content, "timestamp": "2024-01-01T00:00:00Z", "category": "Message", "participantId": "agent-1"},
            {"id": "msg-small", "type": "user_message", "content": "hi", "timestamp": "2024-01-01T00:00:00Z", "category": "Message", "participantId": "user-1"},
        ]
        chunks = _chunk_entries(entries)
        # The huge entry gets its own chunk, the small one gets another
        assert len(chunks) == 2
        assert chunks[0][0]["id"] == "msg-huge"
        assert chunks[1][0]["id"] == "msg-small"


# ─── Integration: Sync/Async SessionManager with chunking ───────────────────


def _records(content: str, event_type: str, participant: str, n: int) -> list[Record]:
    return [
        Record(
            event_id=f"msg-{i}",
            session_id="conv-1",
            agent_id="agent-1",
            event_type=event_type,
            participant_id=participant,
            content=content,
            category="Message",
            timestamp=datetime.now(timezone.utc),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("records", "expect_split"),
    [
        # ~300KB total when serialized → exceeds the payload limit → should split.
        (_records("x" * 60_000, "agent_response", "agent-1", 5), True),
        (_records("Hello!", "user_message", "user-1", 3), False),
    ],
    ids=["large-splits", "small-single-post"],
)
async def test_ingest_chunks_into_add_entries_calls(is_async, records, expect_split):
    """Ingest splits large payloads into multiple add_entries calls and sends a
    small payload as a single call; all entries are delivered either way."""
    client = MagicMock()
    for name in ("create_conversation", "create_session", "add_participants",
                 "add_entries", "close_session"):
        setattr(client, name, AsyncMock(return_value={}) if is_async else MagicMock(return_value={}))
    cls = AsyncSessionManager if is_async else SessionManager
    mgr = cls(client, agent_id="agent-1", owner_id="user-1", session_id="conv-1")

    if is_async:
        await mgr.ingest(records)
        # Let the background task complete
        await asyncio.sleep(0.1)
    else:
        mgr.ingest(records)

    if expect_split:
        assert client.add_entries.call_count > 1
    else:
        assert client.add_entries.call_count == 1
    total_entries = sum(len(call.args[2]) for call in client.add_entries.call_args_list)
    assert total_entries == len(records)

    if is_async:
        await mgr.close()
