"""Evaluate anonymization against annotated PII.

    uv run -m src.evaluation.anonymization data/synthetic_unanonymized.json data/whatsapp_labels.json
    uv run -m src.evaluation.anonymization data/whatsapp_labels.json --ner both --public-entities data/public_entities.json

Each annotated entity is one decision:
    TP = private entity fully masked, FN = private entity not fully masked (partly masked is a miss),
    FP = public entity touched by a mask, TN = public entity untouched.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.anonymize import (
    GLINER_MODEL,
    NER_MODEL,
    Anonymizer,
    CombinedNER,
    GlinerNER,
    bert_ner,
)
from src.evaluation.metrics import scores
from src.utils import write_json

DEFAULT_OUTPUT = Path("output/evaluations/anonymization.json")
NER_CHOICES = ("bert", "gliner", "both")
NER_MODEL_NAMES = {"bert": NER_MODEL, "gliner": GLINER_MODEL, "both": f"{NER_MODEL} + {GLINER_MODEL}"}


def evaluate(records: list[dict], anonymizer: Anonymizer) -> dict:
    outputs = []
    outcomes = Counter()
    unannotated_masks = 0
    for index, record in enumerate(records):
        record_id = record.get("id", index)
        entities = record.get("entities", [])
        _check_offsets(record["text"], entities, record_id)
        output = anonymizer.anonymize(record["text"])
        outputs.append({"id": record_id, **output})
        masks = output["redactions"]
        outcomes.update(_outcome(entity, masks) for entity in entities)
        unannotated_masks += sum(not any(_overlaps(mask, entity) for entity in entities) for mask in masks)
    return {"records": outputs, "metrics": {
        "anonymization_annotated_entities": scores(outcomes["TP"], outcomes["FP"], outcomes["FN"], outcomes["TN"]),
        "fully_redacted_pii": outcomes["TP"],
        "total_pii": outcomes["TP"] + outcomes["FN"],
        "preserved_safe_entities": outcomes["TN"],
        "total_safe_entities": outcomes["FP"] + outcomes["TN"],
        "redactions_outside_annotations": unannotated_masks}}


def _outcome(entity: dict, masks: list[dict]) -> str:
    if entity["should_redact"]:
        fully_masked = all(any(mask["start"] <= position < mask["end"] for mask in masks)
                           for position in range(entity["start"], entity["end"]))
        return "TP" if fully_masked else "FN"
    return "FP" if any(_overlaps(mask, entity) for mask in masks) else "TN"


def _overlaps(first: dict, second: dict) -> bool:
    return first["start"] < second["end"] and second["start"] < first["end"]


def _check_offsets(text: str, entities: list[dict], record_id) -> None:
    for entity in entities:
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= len(text) or text[start:end] != entity["text"]:
            raise ValueError(f"Invalid gold offset in record {record_id}")


def load_records(path: Path) -> list[dict]:
    """Records with "text" and "entities"; data/whatsapp_labels.json is converted to that format."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    return [from_whatsapp_label(record) if "message_id" in record else record for record in records]


def from_whatsapp_label(label: dict) -> dict:
    """Find each labeled PII value in the text. A labeled location that isn't PII (e.g. "Nakuru")
    becomes a must-not-mask entity."""
    text = label["text"]
    pii_values = {pii["value"] for pii in label["pii"]}
    entities = []
    for value in pii_values:
        found = _occurrences(text, value, should_redact=True)
        if not found:
            raise ValueError(f"labeled value {value!r} not found in {text!r}")
        entities += found
    location = label.get("location")
    if location and location not in pii_values:
        entities += _occurrences(text, location, should_redact=False)
    return {"id": label["message_id"], "text": text, "entities": entities, "relevance": label["relevance"]}


def _occurrences(text: str, value: str, should_redact: bool) -> list[dict]:
    found, start = [], text.find(value)
    while start != -1:
        found.append({"text": value, "start": start, "end": start + len(value), "should_redact": should_redact})
        start = text.find(value, start + 1)
    return found


def _make_ner(choice: str):
    """None lets the Anonymizer load BERT itself."""
    if choice == "gliner":
        return GlinerNER()
    if choice == "both":
        return CombinedNER(bert_ner(), GlinerNER())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public-entities", type=Path, help="reviewed JSON list of public places to leave visible")
    parser.add_argument("--ner", choices=NER_CHOICES, default="bert", help="which NER model(s) to use")
    args = parser.parse_args()

    public = json.loads(args.public_entities.read_text()) if args.public_entities else []
    anonymizer = Anonymizer(ner=_make_ner(args.ner), public_entities=public)
    results = {str(path): evaluate(load_records(path), anonymizer) for path in args.inputs}
    report = {"models": {"ner": NER_MODEL_NAMES[args.ner]},
              "metrics": {dataset: result["metrics"] for dataset, result in results.items()},
              "records": {dataset: result["records"] for dataset, result in results.items()}}
    write_json(args.output, report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
