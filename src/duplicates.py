"""Step 6: flag messages that may report the same field activity as an earlier one.

    uv run main.py --input data/dandelion_audio
    uv run -m src.evaluation.hidden_test --pipeline-output output/reports.csv

Flags are for a person to review: nothing is merged or deleted. Pairs are scored on shared content words and matching
extracted fields. The thresholds are first guesses; tune them on labeled duplicate pairs.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from .ingest import Message
from .relevance import is_relevant
from .utils import require_anonymized

DEFAULT_THRESHOLD = 0.62

# How much each extracted field counts toward event similarity.
EVENT_FIELD_WEIGHTS = {
    "activity_type": 0.35,
    "location": 0.30,
    "participant_count": 0.15,
    "referrals": 0.10,
    "status": 0.10,
}
COUNT_FIELDS = {"participant_count", "referrals"}
MASK = re.compile(r"\[[A-Z_]+\]")  # what anonymize() puts in place of private text
MATCHING_FIELD_SIMILARITY = 0.8  # a field this similar counts as matching

# Final score = EVENT_WEIGHT * event similarity + TEXT_WEIGHT * text similarity, when fields were extracted.
EVENT_WEIGHT = 0.55
TEXT_WEIGHT = 0.45

# A pair also needs real word overlap, or strong agreement on several fields (a paraphrase).
MIN_TEXT_SIMILARITY = 0.25
STRONG_EVENT_SIMILARITY = 0.8
MIN_MATCHING_FIELDS = 2

SCORE_DECIMALS = 3

STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "been", "but", "for", "from",
    "had", "has", "have", "he", "her", "here", "i", "in", "is", "it",
    "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "there", "they", "this", "to", "today", "was", "we", "were", "will",
    "with", "you", "your",
}


def mark_duplicates(messages: list[Message], threshold: float = DEFAULT_THRESHOLD) -> None:
    """Set `duplicate_of` on the later message of each likely pair to the earlier message's id.

    A message keeps only its best-scoring match. Runs after extract(), whose fields it compares.
    """
    position = {message.id: index for index, message in enumerate(messages)}
    by_id = {message.id: message for message in messages}
    for candidate in find_duplicate_candidates(messages, threshold):  # best first
        earlier, later = sorted((candidate["message_a"], candidate["message_b"]), key=position.__getitem__)
        if by_id[later].duplicate_of is None:
            by_id[later].duplicate_of = earlier


def find_duplicate_candidates(messages: list[Message], threshold: float = DEFAULT_THRESHOLD,
                              relevant_only: bool = True) -> list[dict[str, Any]]:
    """Message pairs that may be duplicates, highest score first.

    Each has both message ids, a score from 0 to 1, and the evidence for it.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    require_anonymized(messages)
    eligible = [message for message in messages
                if message.anonymized_text and message.anonymized_text.strip()
                and (is_relevant(message) or not relevant_only)]
    pairs = (_compare(left, right, threshold) for left, right in combinations(eligible, 2))
    candidates = [candidate for candidate in pairs if candidate is not None]
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def _compare(left: Message, right: Message, threshold: float) -> dict[str, Any] | None:
    """The pair as a candidate, or None if it doesn't look like a duplicate."""
    text_score = _text_similarity(left.anonymized_text, right.anonymized_text)
    event_score, matching_fields = _event_similarity(left.extraction, right.extraction)
    score = EVENT_WEIGHT * event_score + TEXT_WEIGHT * text_score if event_score else text_score

    shares_words = text_score >= MIN_TEXT_SIMILARITY
    strong_event_match = event_score >= STRONG_EVENT_SIMILARITY and len(matching_fields) >= MIN_MATCHING_FIELDS
    if score < threshold or not (shares_words or strong_event_match):
        return None

    evidence = ["overlapping content words"] if shares_words else []
    evidence += [f"matching {field.replace('_', ' ')}" for field in matching_fields]
    return {
        "message_a": left.id,
        "message_b": right.id,
        "score": round(score, SCORE_DECIMALS),
        "text_similarity": round(text_score, SCORE_DECIMALS),
        "event_similarity": round(event_score, SCORE_DECIMALS),
        "evidence": evidence,
        "needs_human_review": True,
    }


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in STOP_WORDS and len(word) > 1}


def _text_similarity(left: str, right: str) -> float:
    """Dice overlap of content words."""
    left_words, right_words = _content_words(left), _content_words(right)
    if not left_words or not right_words:
        return 0.0
    return 2 * len(left_words & right_words) / (len(left_words) + len(right_words))


def _event_similarity(left: dict | None, right: dict | None) -> tuple[float, list[str]]:
    """Weighted similarity over the fields both messages have, and which fields match."""
    if not left or not right:
        return 0.0, []
    similarities = {field: _field_similarity(field, left.get(field), right.get(field))
                    for field in EVENT_FIELD_WEIGHTS if _present(left.get(field)) and _present(right.get(field))}
    if not similarities:
        return 0.0, []
    total_weight = sum(EVENT_FIELD_WEIGHTS[field] for field in similarities)
    score = sum(EVENT_FIELD_WEIGHTS[field] * similarity for field, similarity in similarities.items()) / total_weight
    matching = [field for field, similarity in similarities.items() if similarity >= MATCHING_FIELD_SIMILARITY]
    return score, matching


def _field_similarity(field: str, left: Any, right: Any) -> float:
    if field in COUNT_FIELDS:
        try:
            return _count_similarity(int(left), int(right))
        except (TypeError, ValueError):
            return float(str(left).casefold() == str(right).casefold())
    return float(str(left).strip().casefold() == str(right).strip().casefold())


def _count_similarity(left: int, right: int) -> float:
    """1 for equal counts, less the further apart; two reports of one event often differ slightly."""
    if left == right:
        return 1.0
    return max(0.0, 1.0 - abs(left - right) / max(abs(left), abs(right), 1))


def _present(value: Any) -> bool:
    """Whether a field says something. A masked value such as "[LOCATION]" doesn't: any two
    masked places would otherwise match, and a masked place would never match a named one."""
    text = "" if value is None else str(value).strip()
    return text != "" and not MASK.fullmatch(text)
