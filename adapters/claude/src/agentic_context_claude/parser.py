# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Incremental reader for Claude Agent SDK transcript JSONL files.

The transcript is append-only. By tracking byte offset, we read only
new lines since the last call — O(new_data) per turn regardless of
total transcript size.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class TranscriptParser:
    """Incremental reader for Claude Agent SDK transcript JSONL files."""

    def __init__(self) -> None:
        self._offset: int = 0
        self._current_path: str | None = None

    def read_incremental(self, path: str) -> list[dict]:
        """Read lines added since last call.

        If path changes (new session), resets offset to 0.
        Skips malformed JSON lines (logs warning, continues).
        """
        if path != self._current_path:
            self._offset = 0
            self._current_path = path

        entries: list[dict] = []
        with open(path, "r") as f:
            f.seek(self._offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line at offset {self._offset}")
            self._offset = f.tell()

        return entries

    def read_all(self, path: str) -> list[dict]:
        """Read entire transcript. For end_session mode."""
        entries: list[dict] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in transcript")
        return entries

    def reset(self) -> None:
        """Reset parser state."""
        self._offset = 0
        self._current_path = None
