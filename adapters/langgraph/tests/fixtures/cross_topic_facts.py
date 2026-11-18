# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

"""Cross-topic fact fixtures for the cross-topic application tests.

These verify that user facts extracted in one topical conversation
(travel planning) generalize to retrievals scoped to other conversations
(restaurant pick, gift ideas) for the same ``owner_id``. The Beacon
contract is the load-bearing layer — every ``GET /context`` for the same
owner should return the same generalized fact set regardless of which
``conversation_id`` is passed.

These fixtures simulate the **same** Beacon response across three
conversation scopes; tests assert that the formatter renders cross-cutting
preferences (vegetarian, bad knee, has wife) in all three without
duplicating them.

Fact ``type`` values are intentionally **neutral** (preference, medical,
family, interest, history) — not topic-tagged (no
``travel_preference``, no ``restaurant_preference``). Topic-locked
taxonomies would defeat generalization at the schema layer.
"""
from __future__ import annotations


# Single canonical fact set extracted from S1 (travel planning).
# Every cross-topic test reuses this verbatim — proving the SDK does NOT
# scope facts by ``conversation_id``.
GENERALIZED_USER_FACTS = [
    {"text": "Vegetarian — needs strong veggie options", "type": "preference", "confidence": 0.92},
    {"text": "Avoids crowded tourist spots", "type": "preference", "confidence": 0.88},
    {"text": "Bad knee — limit stairs and heavy hiking", "type": "medical", "confidence": 0.91},
    {"text": "Travels with wife", "type": "family", "confidence": 0.99},
    {"text": "Photographer — values golden-hour light", "type": "interest", "confidence": 0.85},
    {"text": "Previously visited Tokyo", "type": "history", "confidence": 0.95},
]


# S1 — travel planning conversation. Summary is topic-specific, facts are
# generalized.
TRAVEL_CONTEXT = {
    "summary": (
        "User is planning a March trip to Japan with their wife. Budget "
        "and itinerary still TBD. Will follow up with more questions."
    ),
    "userContext": list(GENERALIZED_USER_FACTS),
}


# S2 — restaurant pick. Same owner, different conversation. Beacon
# returns the same generalized facts.
RESTAURANT_CONTEXT = {
    "summary": (
        "User is choosing a restaurant for a date-night dinner with "
        "their wife this weekend (local — not travel)."
    ),
    "userContext": list(GENERALIZED_USER_FACTS),
}


# S3 — gift ideas. Same owner, different conversation again.
GIFT_CONTEXT = {
    "summary": (
        "User is looking for a birthday gift for their wife — possibly "
        "an experience gift, mentions she has been stressed."
    ),
    "userContext": list(GENERALIZED_USER_FACTS),
}


# Duplicate-laden payload: simulates Beacon returning the same fact
# multiple times because two ingest paths (working memory + persistent
# memory) both surfaced it. Used to assert formatter-side dedup.
DUPLICATED_FACTS = {
    "summary": "User profile.",
    "userContext": [
        {"text": "Vegetarian — needs strong veggie options", "type": "preference", "confidence": 0.92},
        {"text": "Vegetarian — needs strong veggie options", "type": "preference", "confidence": 0.85},
        {"text": "Bad knee — limit stairs and heavy hiking", "type": "medical", "confidence": 0.91},
        {"text": "vegetarian — needs strong veggie options", "type": "preference", "confidence": 0.7},
        {"text": "Travels with wife", "type": "family", "confidence": 0.99},
    ],
}


__all__ = [
    "GENERALIZED_USER_FACTS",
    "TRAVEL_CONTEXT",
    "RESTAURANT_CONTEXT",
    "GIFT_CONTEXT",
    "DUPLICATED_FACTS",
]
