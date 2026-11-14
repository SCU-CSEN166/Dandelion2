"""Flag messages that may report the same field activity.

This is a review aid, not an automatic merge/delete step. It compares anonymized
message text and structured extraction fields when available. Thresholds are
initial heuristics and should be tuned against labeled duplicate pairs.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from .ingest import Message
from .utils import require_anonymized

DEFAULT_THRESHOLD = 0.62

_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "been", "but", "for", "from",
    "had", "has", "have", "he", "her", "here", "i", "in", "is", "it",
    "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "there", "they", "this", "to", "today", "was", "we", "were", "will",
    "with", "you", "your",
}

_EVENT_FIELDS = (
    ("activity_type", 0.35),
    ("location", 0.30),
    ("participant_count", 0.15),
    ("referrals", 0.10),
    ("status", 0.10),
)


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _text_similarity(left: str, right: str) -> float:
    """Dice overlap over content words; works without an added ML dependency."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _field_similarity(field: str, left: Any, right: Any) -> float:
    if not (_present(left) and _present(right)):
        return 0.0
    if field in {"participant_count", "referrals"}:
        try:
            a, b = int(left), int(right)
        except (TypeError, ValueError):
            return float(str(left).casefold() == str(right).casefold())
        if a == b:
            return 1.0
        # Small count differences may be reporting variation for the same event.
        return max(0.0, 1.0 - abs(a - b) / max(abs(a), abs(b), 1))
    return float(str(left).strip().casefold() == str(right).strip().casefold())


def _event_similarity(left: dict[str, Any] | None,
                      right: dict[str, Any] | None) -> tuple[float, list[str]]:
    if not left or not right:
        return 0.0, []

    available: list[tuple[float, float, str]] = []
    matched: list[str] = []
    for field, weight in _EVENT_FIELDS:
        a, b = left.get(field), right.get(field)
        if _present(a) and _present(b):
            similarity = _field_similarity(field, a, b)
            available.append((weight, similarity, field))
            if similarity >= 0.8:
                matched.append(field)

    if not available:
        return 0.0, matched
    # Renormalize when one or more extraction fields are missing.
    total_weight = sum(weight for weight, _, _ in available)
    score = sum(weight * similarity for weight, similarity, _ in available) / total_weight
    return score, matched


def _text(message: Message) -> str:
    # Never use raw text in this post-anonymization step.
    return message.anonymized_text or ""


def find_duplicate_candidates(
    messages: list[Message],
    threshold: float = DEFAULT_THRESHOLD,
    relevant_only: bool = True,
) -> list[dict[str, Any]]:
    """Return scored message pairs that merit human duplicate review.

    Each result contains the two message IDs, a score in [0, 1], and matching
    evidence. The initial threshold is heuristic; calibrate it using labeled
    examples from this project's evaluation set before claiming precision/recall.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    require_anonymized(messages)

    candidates: list[dict[str, Any]] = []
    eligible = [
        message for message in messages
        if _text(message).strip()
        and (
            not relevant_only
            or (message.relevance is not None and message.relevance.get("label") == "relevant")
        )
    ]

    for left, right in combinations(eligible, 2):
        text_score = _text_similarity(_text(left), _text(right))
        event_score, matched_fields = _event_similarity(left.extraction, right.extraction)

        # Use event fields where available; fall back to text similarity otherwise.
        if event_score:
            score = 0.55 * event_score + 0.45 * text_score
        else:
            score = text_score

        # Keep the rule conservative: strong event-field agreement can surface
        # a paraphrase, while text-only matches need substantial word overlap.
        likely_duplicate = (
            score >= threshold
            and (text_score >= 0.25 or (event_score >= 0.8 and len(matched_fields) >= 2))
        )
        if not likely_duplicate:
            continue

        evidence = []
        if text_score >= 0.25:
            evidence.append("overlapping content words")
        evidence.extend(f"matching {field.replace('_', ' ')}" for field in matched_fields)

        candidates.append({
            "message_a": left.id,
            "message_b": right.id,
            "score": round(score, 3),
            "text_similarity": round(text_score, 3),
            "event_similarity": round(event_score, 3),
            "evidence": evidence,
            "needs_human_review": True,
        })

    return sorted(candidates, key=lambda item: item["score"], reverse=True)
