"""Score anonymization and relevance against labeled data.

    uv run -m src.evaluation.baseline

From Prableen's Week 6 evaluate.py. Every dataset is converted to one gold format,
    {"id", "text", "entities": [{"text", "start", "end", "should_redact"}], "relevance"?}
then scored. Sanitized outputs (IDs, masked text, redaction offsets, classifications)
and metrics are written to OUTPUT_DIR/baseline_<dataset>.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src.anonymize import NER_MODEL, Anonymizer
from src.config import OUTPUT_DIR, PUBLIC_ENTITIES, WHATSAPP_LABELS
from src.relevance import CLASSIFIER_MODEL, IRRELEVANT, RELEVANT, RelevanceClassifier
from src.utils import JSON_INDENT, write_json

DATASETS = {
    # The WhatsApp group chat in WhatsAppExports/export2: PII and relevance labels.
    "whatsapp_labels": WHATSAPP_LABELS,
    # 20 synthetic records with dense PII and public-entity distractors; no relevance labels.
    "synthetic": Path("Archive/synthetic_unanonymized.json"),
}


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    return [from_whatsapp_label(record) if "message_id" in record else record for record in records]


def from_whatsapp_label(label: dict) -> dict:
    """data/whatsapp_labels.json lists PII by value, so locate each occurrence in the
    text. A labeled location that isn't PII (e.g. "Nakuru") is a should-not-redact entity.
    """
    text = label["text"]
    pii_values = {pii["value"] for pii in label["pii"]}
    entities = [entity for value in pii_values for entity in _occurrences(text, value, should_redact=True)]
    location = label.get("location")
    if location and location not in pii_values:
        entities += _occurrences(text, location, should_redact=False, required=False)
    return {"id": label["message_id"], "text": text, "entities": entities, "relevance": label["relevance"]}


def _occurrences(text: str, value: str, should_redact: bool, required: bool = True) -> list[dict]:
    found, start = [], text.find(value)
    while start != -1:
        found.append({"text": value, "start": start, "end": start + len(value), "should_redact": should_redact})
        start = text.find(value, start + 1)
    if required and not found:
        raise ValueError(f"labeled value {value!r} not found in {text!r}")
    return found


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass
class ConfusionMatrix:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if actual:
            if predicted:
                self.true_positives += 1
            else:
                self.false_negatives += 1
        elif predicted:
            self.false_positives += 1
        else:
            self.true_negatives += 1

    @property
    def total(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives + self.true_negatives

    def metrics(self) -> dict:
        tp, fp, fn, tn = self.true_positives, self.false_positives, self.false_negatives, self.true_negatives
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "accuracy": _ratio(tp + tn, self.total),
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
                "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def _overlaps(start: int, end: int, other: dict) -> bool:
    return other["start"] < end and start < other["end"]


def _validate_entity(text: str, entity: dict, record_id) -> None:
    start, end = entity["start"], entity["end"]
    if not 0 <= start < end <= len(text) or text[start:end] != entity["text"]:
        raise ValueError(f"Invalid gold offset in record {record_id}")


def _score_entities(text: str, entities: list[dict], redactions: list[dict],
                    pii: ConfusionMatrix, record_id) -> None:
    """Anonymization decision units are annotated entities:
    TP = private entity fully covered, FN = private entity not fully covered,
    FP = safe entity touched by a redaction, TN = safe entity untouched.
    """
    for entity in entities:
        _validate_entity(text, entity, record_id)
        start, end = entity["start"], entity["end"]
        touched = any(_overlaps(start, end, span) for span in redactions)
        fully_covered = all(any(span["start"] <= position < span["end"] for span in redactions)
                            for position in range(start, end))
        # A private entity only counts as redacted when fully covered; partial masking is a miss.
        redacted = fully_covered if entity["should_redact"] else touched
        pii.add(predicted=redacted, actual=entity["should_redact"])


def _redactions_outside(entities: list[dict], redactions: list[dict]) -> int:
    """Redactions that touch no annotated entity (unlabeled PII, or over-masking)."""
    return sum(not any(_overlaps(entity["start"], entity["end"], span) for entity in entities)
               for span in redactions)


def evaluate(records: list[dict], anonymizer: Anonymizer, classifier: RelevanceClassifier) -> dict:
    outputs = []
    pii = ConfusionMatrix()
    relevance = ConfusionMatrix()
    redactions_outside_annotations = 0
    for index, record in enumerate(records):
        record_id = record.get("id", index)
        text = record["text"]  # Gold offsets refer to text, never translation_en.
        entities = record.get("entities", [])
        anonymized = anonymizer.anonymize(text)
        classification = classifier.classify(anonymized["text"])
        outputs.append({"id": record_id, **anonymized, "classification": classification})

        _score_entities(text, entities, anonymized["redactions"], pii, record_id)
        redactions_outside_annotations += _redactions_outside(entities, anonymized["redactions"])

        gold = record.get("relevance")
        if gold is not None:
            if gold not in (RELEVANT, IRRELEVANT):
                raise ValueError(f"relevance must be {RELEVANT} or {IRRELEVANT}")
            relevance.add(predicted=classification["label"] == RELEVANT, actual=gold == RELEVANT)

    return {"records": outputs, "metrics": {
        "anonymization_annotated_entities": pii.metrics(),
        "fully_redacted_pii": pii.true_positives,
        "total_pii": pii.true_positives + pii.false_negatives,
        "preserved_safe_entities": pii.true_negatives,
        "total_safe_entities": pii.false_positives + pii.true_negatives,
        "redactions_outside_annotations": redactions_outside_annotations,
        "classification": relevance.metrics() if relevance.total else None,
        "classification_labeled_records": relevance.total}}


def main() -> None:
    anonymizer = Anonymizer(public_entities=PUBLIC_ENTITIES)
    classifier = RelevanceClassifier()
    for name, path in DATASETS.items():
        report = evaluate(load_records(path), anonymizer, classifier)
        report["dataset"] = str(path)
        report["models"] = {"ner": NER_MODEL, "classification": CLASSIFIER_MODEL}
        output_path = OUTPUT_DIR / f"baseline_{name}.json"
        write_json(output_path, report)
        print(f"== {name} ({path}) -> {output_path}")
        print(json.dumps(report["metrics"], indent=JSON_INDENT))


if __name__ == "__main__":
    main()
