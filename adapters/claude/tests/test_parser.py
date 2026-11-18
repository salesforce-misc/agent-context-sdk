# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Tests for TranscriptParser."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_context_claude.parser import TranscriptParser

FIXTURES = Path(__file__).parent / "fixtures"


class TestReadAll:
    def test_reads_all_entries(self) -> None:
        parser = TranscriptParser()
        entries = parser.read_all(str(FIXTURES / "sample_transcript.jsonl"))

        assert len(entries) == 4
        assert entries[0]["type"] == "user"
        assert entries[1]["type"] == "assistant"
        assert entries[2]["type"] == "user"
        assert entries[3]["type"] == "assistant"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        transcript = tmp_path / "bad.jsonl"
        transcript.write_text(
            '{"type":"user","uuid":"u1","message":{"role":"user","content":"hi"}}\n'
            'this is not json\n'
            '{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":"hello"}}\n'
        )
        parser = TranscriptParser()
        entries = parser.read_all(str(transcript))

        assert len(entries) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")
        parser = TranscriptParser()
        entries = parser.read_all(str(transcript))

        assert entries == []


class TestReadIncremental:
    def test_reads_only_new_lines(self, tmp_path: Path) -> None:
        transcript = tmp_path / "session.jsonl"
        # Write initial content
        transcript.write_text(
            '{"type":"user","uuid":"u1","message":{"role":"user","content":"turn 1"}}\n'
        )
        parser = TranscriptParser()

        # First read
        entries1 = parser.read_incremental(str(transcript))
        assert len(entries1) == 1
        assert entries1[0]["uuid"] == "u1"

        # Append new content
        with open(str(transcript), "a") as f:
            f.write('{"type":"assistant","uuid":"a1","message":{"role":"assistant","content":"resp 1"}}\n')

        # Second read — only gets the new line
        entries2 = parser.read_incremental(str(transcript))
        assert len(entries2) == 1
        assert entries2[0]["uuid"] == "a1"

    def test_resets_on_new_path(self, tmp_path: Path) -> None:
        t1 = tmp_path / "session1.jsonl"
        t2 = tmp_path / "session2.jsonl"
        t1.write_text('{"type":"user","uuid":"u1","message":{"role":"user","content":"s1"}}\n')
        t2.write_text('{"type":"user","uuid":"u2","message":{"role":"user","content":"s2"}}\n')

        parser = TranscriptParser()

        entries1 = parser.read_incremental(str(t1))
        assert len(entries1) == 1

        # Switch path — should reset and read from start
        entries2 = parser.read_incremental(str(t2))
        assert len(entries2) == 1
        assert entries2[0]["uuid"] == "u2"

    def test_no_new_data_returns_empty(self, tmp_path: Path) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_text('{"type":"user","uuid":"u1","message":{"role":"user","content":"hi"}}\n')

        parser = TranscriptParser()
        parser.read_incremental(str(transcript))  # consume all

        entries = parser.read_incremental(str(transcript))
        assert entries == []


class TestReset:
    def test_reset_allows_re_read(self, tmp_path: Path) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_text('{"type":"user","uuid":"u1","message":{"role":"user","content":"hi"}}\n')

        parser = TranscriptParser()
        parser.read_incremental(str(transcript))

        parser.reset()
        entries = parser.read_incremental(str(transcript))
        assert len(entries) == 1
