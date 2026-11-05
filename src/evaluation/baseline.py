"""Score anonymization and relevance against labeled data.

    uv run -m src.evaluation.baseline

From Prableen's Week 6 evaluate.py. Every dataset is converted to one gold format,
    {"id", "text", "entities": [{"text", "start", "end", "should_redact"}], "relevance"?}
then scored. Sanitized outputs (IDs, masked text, redaction offsets, classifications)
and metrics are written to OUTPUT_DIR/baseline_<dataset>.json.
"""

import json
from pathlib import Path

from src.anonymize import NER_MODEL, Anonymizer
from src.relevance import CLASSIFIER_MODEL, RelevanceClassifier

DATASETS = {
    # The WhatsApp group chat in WhatsAppExports/export2: PII and relevance labels.
    "whatsapp_labels": Path("data/whatsapp_labels.json"),
    # 20 synthetic records with dense PII and public-entity distractors; no relevance labels.
    "synthetic": Path("Archive/synthetic_unanonymized.json"),
}
OUTPUT_DIR = Path("output")
PUBLIC_ENTITIES: list[str] = []  # reviewed list, independent of test annotations


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    return [from_whatsapp_label(r) if "message_id" in r else r for r in records]


def from_whatsapp_label(label: dict) -> dict:
    """data/whatsapp_labels.json lists PII by value, so locate each occurrence in the
    text. A labeled location that isn't PII (e.g. "Nakuru") is a should-not-redact entity.
    """
    text = label["text"]
    pii_values = {p["value"] for p in label["pii"]}
    entities = [e for value in pii_values for e in _occurrences(text, value, should_redact=True)]
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


def metrics(tp, fp, fn, tn):
    def div(a, b):
        return a / b if b else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": div(tp + tn, tp + fp + fn + tn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn)}


def evaluate(records: list[dict], anonymizer: Anonymizer, classifier: RelevanceClassifier) -> dict:
    """Anonymization decision units are annotated entities:
    TP = private entity fully covered, FN = private entity not fully covered,
    FP = safe entity touched by a redaction, TN = safe entity untouched.
    """
    outputs = []
    pii = [0, 0, 0, 0]
    relevance = [0, 0, 0, 0]
    full = total = preserved = safe = extra = 0
    labeled = 0
    for i, record in enumerate(records):
        text = record["text"]  # Gold offsets refer to text, never translation_en.
        anonymized = anonymizer.anonymize(text)
        output = {**anonymized, "classification": classifier.classify(anonymized["text"])}
        outputs.append({"id": record.get("id", i), **output})
        spans = output["redactions"]
        for entity in record.get("entities", []):
            a, b = entity["start"], entity["end"]
            if not 0 <= a < b <= len(text) or text[a:b] != entity["text"]:
                raise ValueError("Invalid gold offset in record %s" % record.get("id", i))
            overlap = any(s["start"] < b and a < s["end"] for s in spans)
            covered = all(any(s["start"] <= pos < s["end"] for s in spans) for pos in range(a, b))
            redact = entity["should_redact"]
            # Positive requires complete coverage; partial PII masking is a miss.
            index = (0 if covered else 2) if redact else (1 if overlap else 3)
            pii[index] += 1
            if redact:
                total += 1
                full += covered
            else:
                safe += 1
                preserved += not overlap
        extra += sum(not any(s["start"] < e["end"] and e["start"] < s["end"]
                             for e in record.get("entities", [])) for s in spans)
        gold = record.get("relevance")
        if gold is not None:
            if gold not in ("relevant", "irrelevant"):
                raise ValueError("relevance must be relevant or irrelevant")
            labeled += 1
            predicted = output["classification"]["label"] == "relevant"
            actual = gold == "relevant"
            relevance[(0 if predicted else 2) if actual else (1 if predicted else 3)] += 1
    return {"records": outputs, "metrics": {
        "anonymization_annotated_entities": metrics(*pii),
        "fully_redacted_pii": full, "total_pii": total,
        "preserved_safe_entities": preserved, "total_safe_entities": safe,
        "redactions_outside_annotations": extra,
        "classification": metrics(*relevance) if labeled else None,
        "classification_labeled_records": labeled}}


if __name__ == "__main__":
    anonymizer = Anonymizer(public_entities=PUBLIC_ENTITIES)
    classifier = RelevanceClassifier()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in DATASETS.items():
        report = evaluate(load_records(path), anonymizer, classifier)
        report["dataset"] = str(path)
        report["models"] = {"ner": NER_MODEL, "classification": CLASSIFIER_MODEL}
        out = OUTPUT_DIR / f"baseline_{name}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"== {name} ({path}) -> {out}")
        print(json.dumps(report["metrics"], indent=2))
