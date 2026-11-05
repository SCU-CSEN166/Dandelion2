"""Score extracted activity fields against data/whatsapp_labels.json.

    uv run -m src.evaluation.extraction

The labels annotate the WhatsAppExports/export2 chat, so that chat is parsed and
anonymized exactly as in the pipeline, and each labeled message (matched by line
number) is extracted with its preceding messages as context. The relevance step is
skipped so extraction is scored on its own. Per-record outputs and per-field
accuracy are written to OUTPUT.

Scoring per field (null must match null):
    status             exact; a null prediction counts as "neither"
    activity_type      lenient: every word of the predicted category appears in the
                       gold label ("savings group meeting" ~ "women's savings group meeting")
    location           exact, case-insensitive. Anonymization masks private places, so
                       location_detected also reports whether any location was found.
    participant_count, referrals   exact
"""

import json
from pathlib import Path

from src.anonymize import Anonymizer, anonymize
from src.extract import CONTEXT_MESSAGES, EXTRACTION_MODEL, Extractor, context_for
from src.ingest import Message, load

DATASET = Path("data/whatsapp_labels.json")
CHAT = Path("WhatsAppExports/export2")
OUTPUT = Path("output/extraction_whatsapp_labels.json")
PUBLIC_ENTITIES: list[str] = []  # keep in sync with main.py


def field_correct(field: str, predicted, gold) -> bool:
    if field == "status":
        return (predicted or "neither") == gold
    if predicted is None or gold is None:
        return predicted is None and gold is None
    if field == "activity_type":
        return set(predicted.casefold().split()) <= set(gold.casefold().replace("'s", "").split())
    if field == "location":
        return predicted.casefold() == gold.casefold()
    return predicted == gold


def evaluate(labels: list[dict], messages: list[Message], extractor: Extractor,
             context_size: int = CONTEXT_MESSAGES) -> dict:
    """messages: the anonymized chat the labels refer to, in order."""
    index_by_line = {m.metadata["line"]: i for i, m in enumerate(messages)}
    fields = list(extractor.template)
    correct = dict.fromkeys(fields + ["location_detected"], 0)
    outputs = []
    for label in labels:
        i = index_by_line[label["source_line"]]
        text = messages[i].anonymized_text
        predicted = extractor.extract(text, context_for(messages, i, context_size) if context_size else "")
        outputs.append({"id": label["message_id"], "text": text, "predicted": predicted,
                        "gold": {f: label[f] for f in fields}})
        for field in fields:
            correct[field] += field_correct(field, predicted[field], label[field])
        correct["location_detected"] += (predicted["location"] is None) == (label["location"] is None)
    n = len(labels)
    return {"records": outputs, "metrics": {
        "records": n,
        "context_messages": context_size,
        "accuracy": {field: count / n if n else None for field, count in correct.items()},
        "needs_review": sum(bool(o["predicted"]["needs_review"]) for o in outputs)}}


if __name__ == "__main__":
    labels = json.loads(DATASET.read_text(encoding="utf-8"))
    messages = [m for m in load(CHAT) if not m.is_system]
    anonymize(messages, Anonymizer(public_entities=PUBLIC_ENTITIES))
    report = evaluate(labels, messages, Extractor())
    report["dataset"] = str(DATASET)
    report["model"] = EXTRACTION_MODEL
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{EXTRACTION_MODEL} on {DATASET} -> {OUTPUT}")
    print(json.dumps(report["metrics"], indent=2))
