"""Dandelion pipeline entry point. Edit the settings below, then:

    uv run main.py

Steps:
    1) Parse input (WhatsApp export, JSON/JSONL, plain text)  -> src.ingest
    2) Transcribe audio attachments                           -> src.transcribe
    3) Anonymize                                              -> src.anonymize
    4) Determine relevance                                    -> src.relevance
    5) Extract to JSON                                        -> src.extract
"""

import json
from pathlib import Path

from src.anonymize import Anonymizer, anonymize
from src.extract import extract
from src.ingest import Message, load
from src.relevance import classify
from src.transcribe import transcribe

# Any mix of WhatsApp exports (.zip/.txt/folder), .json/.jsonl, or plain .txt
INPUTS = [
    "WhatsAppExports/export2",
]
MESSAGES_OUTPUT = Path("output/messages.json")  # every message, processed and anonymized
REPORTS_OUTPUT = Path("output/reports.json")    # only relevant messages, with extracted fields

# Reviewed public places/organizations the anonymizer may leave unmasked, e.g. "Nakuru"
PUBLIC_ENTITIES: list[str] = []

# Speed knobs for test runs
LIMIT: int | None = 20   # only the first N messages of each input; None = all
EXTRACTION_CONTEXT = 5   # earlier messages used to read status; 0 = faster, less accurate


def summarize(m: Message) -> str:
    when = m.timestamp.strftime("%Y-%m-%d %H:%M") if m.timestamp else "-"
    extras = "".join(f" [{a.kind}: {a.filename or 'omitted'}]" for a in m.attachments)
    rel = f"{m.relevance['label']} {m.relevance['score']:.2f}" if m.relevance else "-"
    line = f"{m.id:<22} {when}  ({rel})  {m.anonymized_sender or '-'}: {m.anonymized_text}{extras}"
    if m.extraction:
        fields = {k: v for k, v in m.extraction.items() if v}
        line += f"\n{'':<24}-> {fields}"
    return line


def anonymized_record(m: Message) -> dict:
    """m.to_dict() without anything raw: original text, sender, transcripts, source metadata."""
    record = m.to_dict()
    for key in ("text", "sender", "metadata"):
        del record[key]
    for attachment in record["attachments"]:
        del attachment["transcript"]  # transcripts are already in anonymized_text
    return record


def report(m: Message) -> dict:
    """One flat activity record for staff to review."""
    return {
        "id": m.id,
        "conversation": m.conversation,
        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
        "sender": m.anonymized_sender,
        "text": m.anonymized_text,
        "relevance_score": round(m.relevance["score"], 3),
        **m.extraction,
    }


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    # 1) Parse
    # Drop notices like "Messages are end-to-end encrypted", then apply LIMIT per input
    messages = []
    for path in INPUTS:
        messages += [m for m in load(path) if not m.is_system][:LIMIT]
    print(f"1) Loaded {len(messages)} messages from {len(INPUTS)} input(s)", flush=True)

    # 2) Transcribe (audio transcripts then show up in m.full_text)
    print(f"2) Transcribing {sum(len(m.audio) for m in messages)} audio attachment(s)", flush=True)
    transcribe(messages)

    # 3) Anonymize (fills m.anonymized_text, m.redactions, m.anonymized_sender)
    print("3) Anonymizing", flush=True)
    anonymize(messages, Anonymizer(public_entities=PUBLIC_ENTITIES))

    # 4) Determine relevance on the anonymized text (fills m.relevance)
    print("4) Classifying relevance", flush=True)
    classify(messages)

    # 5) Extract activity fields from relevant messages (fills m.extraction)
    print("5) Extracting activity fields", flush=True)
    extract(messages, context_size=EXTRACTION_CONTEXT)

    for m in messages:
        print(summarize(m))
    reports = [report(m) for m in messages if m.extraction]
    print(f"{len(reports)} of {len(messages)} messages kept as activity reports")
    write_json(MESSAGES_OUTPUT, [anonymized_record(m) for m in messages])
    write_json(REPORTS_OUTPUT, reports)


if __name__ == "__main__":
    main()
