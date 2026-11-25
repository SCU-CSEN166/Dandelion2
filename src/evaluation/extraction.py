"""Evaluate activity-field extraction against labeled messages.

    uv run -m src.evaluation.extraction data/whatsapp_labels.json                  # NuExtract
    uv run -m src.evaluation.extraction data/whatsapp_labels.json --model flan-t5  # Flan-T5

Each message is extracted on its own, without context or anonymization, so this measures
the model alone. A value is correct only when it equals the label exactly. Fields are
scored as slot filling (see metrics.py); choice fields also get a confusion matrix.
"""

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import confusion_matrix, slot_scores
from src.extract import EXTRACTORS, Extractor, make_extractor
from src.utils import write_json

DEFAULT_MODEL = "nuextract"
OUTPUT_DIR = Path("output/evaluations")


def evaluate(records: list[dict], extractor: Extractor) -> dict:
    fields = list(extractor.template)
    outputs = []
    pairs = {field: [] for field in fields}
    for index, record in enumerate(records):
        predicted = extractor.extract(record["text"])
        # New output fields may not have labels in older evaluation sets. Keep the
        # prediction for inspection, but only score fields with a ground-truth label.
        expected = {field: record[field] for field in fields if field in record}
        outputs.append({"id": record.get("message_id", index), "predicted": predicted, "expected": expected})
        for field in fields:
            if field in expected:
                pairs[field].append((predicted[field], expected[field]))
    choice_fields = [field for field in fields if isinstance(extractor.template[field], list)]
    return {"records": outputs, "metrics": {
        "records": len(records),
        "fields": {field: slot_scores(pairs[field]) if pairs[field] else None for field in fields},
        "confusion_matrices": {field: confusion_matrix(pairs[field]) for field in choice_fields if pairs[field]}}}


def default_output(model: str) -> Path:
    """extraction.json for NuExtract, extraction_<model>.json for the others."""
    suffix = "" if model == DEFAULT_MODEL else f"_{model}"
    return OUTPUT_DIR / f"extraction{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", choices=list(EXTRACTORS), default=DEFAULT_MODEL, help="extraction model")
    parser.add_argument("--output", type=Path, help="default: output/evaluations/extraction[_<model>].json")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    extractor = make_extractor(args.model)
    result = evaluate(records, extractor)
    report = {"models": {"extraction": extractor.model_name}, "dataset": str(args.input),
              "metrics": result["metrics"], "records": result["records"]}
    write_json(args.output or default_output(args.model), report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
