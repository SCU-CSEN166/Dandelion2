"""Evaluate duplicate detection on labeled development message pairs.

    uv run -m src.evaluation.duplicates
    uv run -m src.evaluation.duplicates --threshold 0.70
    uv run -m src.evaluation.duplicates --sweep

This intentionally uses a development set that is separate from the TA hidden test.
Each case contains two already-anonymized, already-extracted Event messages so this
script evaluates only the duplicate-detection logic in src/duplicates.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.duplicates import DEFAULT_THRESHOLD, find_duplicate_candidates
from src.evaluation.metrics import binary_counts, scores
from src.ingest import Message
from src.relevance import RELEVANT
from src.utils import write_json

DEFAULT_INPUT = Path("data/duplicate_dev_cases.json")
DEFAULT_OUTPUT = Path("output/evaluations/duplicates.json")


def _message(case_id: str, suffix: str, record: dict) -> Message:
    message = Message(id=f"{case_id}:{suffix}", source="duplicate_dev", text=record["text"])
    # The duplicate step runs after anonymization, relevance, and extraction in main.py.
    # These labeled development cases therefore provide those upstream outputs directly.
    message.anonymized_text = record["text"]
    message.relevance = {"label": RELEVANT, "score": 1.0}
    message.extraction = record["extraction"]
    return message


def evaluate(cases: list[dict], threshold: float) -> dict:
    decisions: list[tuple[bool, bool]] = []
    case_results = []

    for case in cases:
        left = _message(case["id"], "a", case["message_a"])
        right = _message(case["id"], "b", case["message_b"])
        candidates = find_duplicate_candidates([left, right], threshold=threshold)
        predicted = bool(candidates)
        expected = bool(case["expected_duplicate"])
        decisions.append((predicted, expected))

        candidate = candidates[0] if candidates else None
        case_results.append({
            "id": case["id"],
            "expected_duplicate": expected,
            "predicted_duplicate": predicted,
            "correct": predicted == expected,
            "reason": case.get("reason"),
            "score": candidate["score"] if candidate else None,
            "text_similarity": candidate["text_similarity"] if candidate else None,
            "event_similarity": candidate["event_similarity"] if candidate else None,
            "evidence": candidate["evidence"] if candidate else [],
        })

    metrics = scores(*binary_counts(decisions))
    return {
        "threshold": threshold,
        "case_count": len(cases),
        "positive_cases": sum(case["expected_duplicate"] for case in cases),
        "negative_cases": sum(not case["expected_duplicate"] for case in cases),
        "metrics": metrics,
        "cases": case_results,
    }


def sweep(cases: list[dict]) -> list[dict]:
    results = []
    for step in range(40, 81, 2):
        threshold = step / 100
        result = evaluate(cases, threshold)
        results.append({"threshold": threshold, **result["metrics"]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sweep", action="store_true", help="also evaluate thresholds from 0.40 to 0.80")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    result = evaluate(cases, args.threshold)
    report = {
        "dataset": str(args.input),
        "note": "Development-only duplicate evaluation; do not tune on the TA hidden test.",
        **result,
    }
    if args.sweep:
        report["threshold_sweep"] = sweep(cases)

    write_json(args.output, report)
    print(json.dumps({
        "threshold": report["threshold"],
        "case_count": report["case_count"],
        "metrics": report["metrics"],
    }, indent=2))

    failures = [case for case in report["cases"] if not case["correct"]]
    if failures:
        print("\nMisclassified cases:")
        for case in failures:
            print(f"- {case['id']}: expected={case['expected_duplicate']} predicted={case['predicted_duplicate']} "
                  f"score={case['score']} ({case['reason']})")


if __name__ == "__main__":
    main()
