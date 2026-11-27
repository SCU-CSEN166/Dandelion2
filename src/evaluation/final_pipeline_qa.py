"""Final integrated QA smoke test for the Dandelion pipeline.

Runs a small set of regression cases through anonymization, relevance,
extraction, and duplicate detection, then writes a JSON report.

Usage:
    uv run -m src.evaluation.final_pipeline_qa
"""

import json
from pathlib import Path

from main import review_reasons
from src.anonymize import Anonymizer, CombinedNER, GlinerNER, anonymize, bert_ner
from src.config import (
    EXTRACTION_BACKEND,
    PUBLIC_ENTITIES,
    RELEVANCE_COUNT_PLANS,
    RELEVANCE_MIN_WORDS,
    RELEVANCE_SKIP_QUESTIONS,
    USE_GLINER,
)
from src.duplicates import mark_duplicates
from src.extract import extract, make_extractor
from src.ingest import Message
from src.relevance import RelevanceClassifier, classify
from src.utils import write_json

CASES = Path("data/final_pipeline_qa_cases.json")
OUTPUT = Path("output/evaluations/final_pipeline_qa.json")
EXTRACTION_CONTEXT = 5


def _messages(cases: list[dict]) -> list[Message]:
    return [
        Message(id=case["id"], source="final_pipeline_qa", text=case["text"])
        for case in cases
    ]


def _checks(case: dict, message: Message) -> dict[str, bool]:
    expected = case.get("expected", {})
    checks: dict[str, bool] = {}

    if "relevance" in expected:
        checks["relevance"] = (
            message.relevance is not None
            and message.relevance.get("label") == expected["relevance"]
        )

    extraction = message.extraction or {}

    if "activity_type" in expected:
        checks["activity_type"] = extraction.get("activity_type") == expected["activity_type"]

    if "participant_count" in expected:
        checks["participant_count"] = extraction.get("participant_count") == expected["participant_count"]

    if "contains_masks" in expected:
        text = message.anonymized_text or ""
        checks["contains_masks"] = all(mask in text for mask in expected["contains_masks"])

    if "activity_phrase_should_remain" in expected:
        phrase = expected["activity_phrase_should_remain"].casefold()
        checks["activity_phrase_should_remain"] = phrase in (message.anonymized_text or "").casefold()

    if "duplicate_of" in expected:
        checks["duplicate_of"] = message.duplicate_of == expected["duplicate_of"]

    return checks


def _record(case: dict, message: Message) -> dict:
    checks = _checks(case, message)
    reasons = review_reasons(message)
    return {
        "id": case["id"],
        "text": case["text"],
        "anonymized_text": message.anonymized_text,
        "relevance": message.relevance,
        "extraction": message.extraction,
        "duplicate_of": message.duplicate_of,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    messages = _messages(cases)

    ner = CombinedNER(bert_ner(), GlinerNER()) if USE_GLINER else None
    anonymize(messages, Anonymizer(ner=ner, public_entities=PUBLIC_ENTITIES))

    classify(
        messages,
        RelevanceClassifier(
            count_plans=RELEVANCE_COUNT_PLANS,
            min_words=RELEVANCE_MIN_WORDS,
            skip_questions=RELEVANCE_SKIP_QUESTIONS,
        ),
    )

    extract(
        messages,
        make_extractor(EXTRACTION_BACKEND),
        context_size=EXTRACTION_CONTEXT,
    )

    mark_duplicates(messages)

    records = [_record(case, message) for case, message in zip(cases, messages)]
    passed = sum(record["passed"] for record in records)
    report = {
        "case_count": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "cases": records,
    }

    write_json(OUTPUT, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
