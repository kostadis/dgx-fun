"""Shared retry *policy* (not the predicate).

The retryable *predicate* is transport-specific — dgxlib's plain client matches
``urllib`` errors, CampaignGenerator matches ``openai``/``httpx`` ones — so it
cannot live here. What is shared is the *policy*: which HTTP status codes are
worth retrying against a vLLM slot (overload / transient backend errors). Each
transport maps its own exceptions onto this set.
"""

from __future__ import annotations

# 500/502/503: transient vLLM/backend errors; 529: overloaded.
RETRYABLE_STATUS = frozenset({500, 502, 503, 529})


def is_retryable_status(code: int | None) -> bool:
    """True if an HTTP status code is worth retrying."""
    return code in RETRYABLE_STATUS
