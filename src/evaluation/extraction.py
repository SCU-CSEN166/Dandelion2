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

from src.anonymize import Anonymizer, anonymize
from src.config import LABELED_CHAT, OUTPUT_DIR, PUBLIC_ENTITIES, WHATSAPP_LABELS
from src.extract import CONTEXT_MESSAGES, EXTRACTION_MODEL, NEEDS_REVIEW, Extractor, context_for
from src.ingest import Message, load
from src.utils import JSON_INDENT, write_json

OUTPUT = OUTPUT_DIR / "extraction_whatsapp_labels.json"
NO_STATUS = "neither"  # gold label for messages that are neither completed nor planned
LOCATION_DETECTED = "location_detected"


def field_correct(field: str, predicted, gold) -> bool:
    if field == "status":
        return (predicted or NO_STATUS) == gold
    if predicted is None or gold is None:
        return predicted is None and gold is None
    if field == "activity_type":
        predicted_words = set(predicted.casefold().split())
        gold_words = set(gold.casefold().replace("'s", "").split())  # "women's" -> "women"
        return predicted_words <= gold_words
    if field == "location":
        return predicted.casefold() == gold.casefold()
    return predicted == gold


def evaluate(labels: list[dict], messages: list[Message], extractor: Extractor,
             context_size: int = CONTEXT_MESSAGES) -> dict:
    """messages: the anonymized chat the labels refer to, in order."""
    index_by_line = {message.metadata["line"]: index for index, message in enumerate(messages)}
    fields = list(extractor.template)
    correct_counts = dict.fromkeys(fields + [LOCATION_DETECTED], 0)
    outputs = []
    for label in labels:
        index = index_by_line[label["source_line"]]
        text = messages[index].anonymized_text
        predicted = extractor.extract(text, context_for(messages, index, context_size))
        outputs.append({"id": label["message_id"], "text": text, "predicted": predicted,
                        "gold": {field: label[field] for field in fields}})
        for field in fields:
            correct_counts[field] += field_correct(field, predicted[field], label[field])
        correct_counts[LOCATION_DETECTED] += (predicted["location"] is None) == (label["location"] is None)

    total = len(labels)
    return {"records": outputs, "metrics": {
        "records": total,
        "context_messages": context_size,
        "accuracy": {field: count / total if total else None for field, count in correct_counts.items()},
        NEEDS_REVIEW: sum(bool(output["predicted"][NEEDS_REVIEW]) for output in outputs)}}


def main() -> None:
    labels = json.loads(WHATSAPP_LABELS.read_text(encoding="utf-8"))
    messages = [message for message in load(LABELED_CHAT) if not message.is_system]
    anonymize(messages, Anonymizer(public_entities=PUBLIC_ENTITIES))
    report = evaluate(labels, messages, Extractor())
    report["dataset"] = str(WHATSAPP_LABELS)
    report["model"] = EXTRACTION_MODEL
    write_json(OUTPUT, report)
    print(f"{EXTRACTION_MODEL} on {WHATSAPP_LABELS} -> {OUTPUT}")
    print(json.dumps(report["metrics"], indent=JSON_INDENT))


if __name__ == "__main__":
    main()
