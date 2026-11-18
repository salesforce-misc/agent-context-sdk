# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Regression test for TopicShiftPolicy.should_fetch's embedder-failure site.

Previously an embedder exception was swallowed with a bare
``except Exception: return False`` — indistinguishable from "no topic shift
detected." The fix logs a WARNING naming the function before falling back to
the same no-shift default, so a broken/misconfigured embedder is observable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from agentic_context_langgraph import TopicShiftPolicy

LOGGER_NAME = "agentic_context_langgraph.policies"


def test_topic_shift_should_fetch_survives_attribute_error(caplog) -> None:
    def broken_embedder(text: str) -> list[float]:
        raise AttributeError("no attribute 'embed'")

    policy = TopicShiftPolicy(embedder=broken_embedder, threshold=0.7)
    once = datetime.now(timezone.utc)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = policy.should_fetch(
            session_id="ses-1",
            turn_idx=1,
            last_user_text="something new",
            last_payload={"summary": "prior topic"},
            last_fetched_at=once,
        )

    assert result is False
    assert "should_fetch" in caplog.text
    assert "AttributeError" in caplog.text
