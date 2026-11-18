# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""CRM-shaped Beacon payloads for the Day-Zero CRM Context tests.

These fixtures exist because the rest of the test suite uses generic
``{"text": "...", "type": "preference", "confidence": 0.9}`` shapes that
don't exercise the day-zero use case where context is preloaded from a
real CRM (Salesforce / etc.) before the user has said anything.

The reference profile is "Sarah Chen": Enterprise Gold tier, open P1
case, recent purchase, churn risk Medium, NPS 7, account manager,
Singapore.

Two flavors are provided — a compact "fast" payload (flagship signals
only) and a rich "deep" payload (every day-zero signal) — so callers can
assert the SDK formatter handles both shapes without losing urgent
signals. Both ride over the same POST `get_context` wire format; the
distinction is purely payload shape, not an SDK knob.
"""
from __future__ import annotations

# Compact "fast mode" — flagship signals only, suited for first-token
# latency. The agent must still be able to greet by name and surface the
# urgent open case from this alone.
SARAH_CHEN_FAST = {
    "summary": (
        "Sarah Chen, Enterprise Gold tier (Singapore). Open P1 case "
        "#4521 (API latency, opened 3d ago). Churn risk: Medium."
    ),
    "userContext": [
        {
            "text": "Open P1 case #4521: API latency spike (3 days unresolved)",
            "type": "open_case",
            "confidence": 0.99,
        },
        {
            "text": "Account tier: Enterprise Gold",
            "type": "account_tier",
            "confidence": 0.99,
        },
        {
            "text": "Churn risk: Medium",
            "type": "churn_risk",
            "confidence": 0.87,
        },
        {
            "text": "Customer name: Sarah Chen",
            "type": "identity",
            "confidence": 0.99,
        },
    ],
}


# Deep "high-context" payload — every signal the day-zero spec calls
# out. Order is intentionally NOT priority-sorted — the formatter is
# what's expected to pull urgent signals to the top.
SARAH_CHEN_DEEP = {
    "summary": (
        "Sarah Chen — Enterprise Gold customer in Singapore. Renewed "
        "Data 360 Enterprise license 2 months ago. Currently has an "
        "unresolved P1 case (API latency, 3 days). NPS 7, churn risk "
        "Medium. Account Manager: James Rodriguez. Prefers email."
    ),
    "userContext": [
        {
            "text": "Customer name: Sarah Chen",
            "type": "identity",
            "confidence": 0.99,
        },
        {
            "text": "Location: Singapore",
            "type": "location",
            "confidence": 0.99,
        },
        {
            "text": "Preferred contact: email",
            "type": "preference",
            "confidence": 0.93,
        },
        {
            "text": "Last purchase: Data 360 Enterprise License (renewed 2026-04-15)",
            "type": "purchase_history",
            "confidence": 0.94,
        },
        {
            "text": "Open P1 case #4521: API latency spike (opened 3 days ago, unresolved)",
            "type": "open_case",
            "confidence": 0.99,
        },
        {
            "text": "Churn risk: Medium",
            "type": "churn_risk",
            "confidence": 0.87,
        },
        {
            "text": "NPS: 7",
            "type": "nps",
            "confidence": 0.92,
        },
        {
            "text": "Account Manager: James Rodriguez",
            "type": "account_manager",
            "confidence": 0.99,
        },
        {
            "text": "Account tier: Enterprise Gold",
            "type": "account_tier",
            "confidence": 0.99,
        },
    ],
}


# Negative-control payload — same owner, no urgent signals. Lets tests
# assert the formatter doesn't invent ranking when nothing demands it.
SARAH_CHEN_QUIET = {
    "summary": "Sarah Chen — Enterprise Gold customer in Singapore.",
    "userContext": [
        {"text": "Customer name: Sarah Chen", "type": "identity", "confidence": 0.99},
        {"text": "Location: Singapore", "type": "location", "confidence": 0.99},
        {
            "text": "Account tier: Enterprise Gold",
            "type": "account_tier",
            "confidence": 0.99,
        },
    ],
}


URGENT_FACT_TYPES = ("open_case", "sla_breach", "churn_risk", "escalation")


__all__ = [
    "SARAH_CHEN_FAST",
    "SARAH_CHEN_DEEP",
    "SARAH_CHEN_QUIET",
    "URGENT_FACT_TYPES",
]
