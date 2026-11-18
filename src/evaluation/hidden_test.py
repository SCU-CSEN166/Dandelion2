"""Score the pipeline on the hidden test (data/dandelion_audio/hidden_test_outputs.csv).

    uv run -m src.evaluation.hidden_test                               # score output/HIDDEN_TEST_OUTPUT.csv
    uv run -m src.evaluation.hidden_test --models nuextract flan-t5   # also score the extractors alone
    uv run -m src.evaluation.hidden_test --pipeline-output output/reports.csv

The pipeline CSV comes from `uv run main.py --input data/dandelion_audio` (written to output/reports.csv).

Writes a short JSON of TP/FP/FN/TN and metrics per check.
- relevance: "Event" is the positive class.
- activity_type, participants, duplicate_of: slot filling (see metrics.py).
- pii: one decision per PII item in the solution. TP = hidden, FN = still in the text,
  including near spellings, since speech-to-text changes names ("faith aching").
- extractors: each model reads the solution's anonymized text, so speech-to-text and
  anonymization errors don't count against it.

The solution's activity labels are free text, mapped by hand to our categories (ACTIVITY_LABELS).
"""

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

from src.evaluation.metrics import binary_counts, scores, slot_scores
from src.extract import EXTRACTORS, Extractor, make_extractor
from src.utils import write_json

SOLUTION = Path("data/dandelion_audio/hidden_test_outputs.csv")
PIPELINE_OUTPUT = Path("output/HIDDEN_TEST_OUTPUT.csv")
OUTPUT = Path("output/evaluations/hidden_test.json")

EVENT = "Event"
NONE = "none"
ACTIVITY_LABELS = {
    "Family planning outreach": "family planning outreach",
    "Women's group table banking (savings) meeting": "savings group meeting",
    "Menstrual hygiene session and sanitary pad distribution": "menstrual hygiene session",
    "Gender-based violence awareness dialogue": "gender-based violence dialogue",
    "Cervical cancer screening camp": "cancer screening",
    "Kitchen garden training": "agriculture training",
}
LEAK_SIMILARITY = 0.8  # a span of text at least this close to a PII item counts as a leak
DECIMALS = 3


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def cell_value(cell: str | None):
    """'' and 'none' -> None, '42' -> 42, anything else unchanged."""
    cell = (cell or "").strip()
    if not cell or cell.casefold() == NONE:
        return None
    return int(cell) if cell.isdigit() else cell


def activity(cell: str | None) -> str | None:
    label = cell_value(cell)
    return None if label is None else ACTIVITY_LABELS.get(label, label.casefold())


def pii_items(cell: str) -> list[str]:
    """"Lucy Kiprop (also 'Lucy'); Tumaini (women's group name)" -> ["Lucy", "Lucy Kiprop", "Tumaini"]."""
    items = []
    for item in cell.split(";"):
        items += re.findall(r"also '([^']+)'", item)
        item = re.sub(r"\(.*?\)", "", item).strip()
        if item:
            items.append(item)
    return items


def leaked(item: str, text: str) -> bool:
    """Whether `item`, or a near spelling of it, is still in `text`. Phone numbers must match exactly."""
    target_words = _words(item)
    target = "".join(target_words)
    words = _words(text)
    for length in range(max(1, len(target_words) - 1), len(target_words) + 2):
        for start in range(len(words) - length + 1):
            window = "".join(words[start:start + length])
            if window == target or (not target.isdigit() and _similarity(window, target) >= LEAK_SIMILARITY):
                return True
    return False


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _similarity(first: str, second: str) -> float:
    return difflib.SequenceMatcher(None, first, second).ratio()


def evaluate_pipeline(solution: list[dict], rows: list[dict]) -> dict:
    by_id = {row["Message_id"]: row for row in rows}
    pairs = [(by_id[record["message_id"]], record) for record in solution]
    relevance = [(row["relevance"] == EVENT, record["relevance"] == EVENT) for row, record in pairs]
    fields = {
        "activity_type": [(activity(row["activity_type"]), activity(record["activity_type"])) for row, record in pairs],
        "participants": [(cell_value(row["participants"]), cell_value(record["participants"])) for row, record in pairs],
        "duplicate_of": [(cell_value(row["duplicate_of"]), cell_value(record["duplicate_of"])) for row, record in pairs],
    }
    leaks = [leaked(item, row["anonymized_text"]) for row, record in pairs for item in pii_items(record["pii_items"])]
    return {"relevance": _rounded(scores(*binary_counts(relevance))),
            **{field: _rounded(slot_scores(field_pairs)) for field, field_pairs in fields.items()},
            "pii": _rounded(scores(tp=leaks.count(False), fp=0, fn=leaks.count(True), tn=0))}


def evaluate_extractor(solution: list[dict], extractor: Extractor) -> dict:
    predictions = [extractor.extract(record["expected_anonymized_text"]) for record in solution]
    activity_pairs = [(predicted["activity_type"], activity(record["activity_type"]))
                      for predicted, record in zip(predictions, solution)]
    participant_pairs = [(predicted["participant_count"], cell_value(record["participants"]))
                         for predicted, record in zip(predictions, solution)]
    return {"activity_type": _rounded(slot_scores(activity_pairs)),
            "participants": _rounded(slot_scores(participant_pairs))}


def _rounded(metrics: dict) -> dict:
    return {key: round(value, DECIMALS) if isinstance(value, float) else value for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solution", type=Path, default=SOLUTION)
    parser.add_argument("--pipeline-output", type=Path, default=PIPELINE_OUTPUT)
    parser.add_argument("--models", nargs="*", choices=list(EXTRACTORS), default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    solution = load_csv(args.solution)
    report = {"messages": len(solution),
              "pipeline": evaluate_pipeline(solution, load_csv(args.pipeline_output))}
    if args.models:
        report["extractors"] = {name: evaluate_extractor(solution, make_extractor(name)) for name in args.models}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
