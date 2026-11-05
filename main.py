"""Dandelion pipeline entry point. Edit the settings below, then:

    uv run main.py

Steps:
    1) Parse input (WhatsApp export, JSON/JSONL, plain text)  -> src.ingest
    2) Transcribe audio attachments                           -> src.transcribe
    3) Anonymize                                              -> src.anonymize
    4) Determine relevance                                    -> src.relevance
    5) Extract to JSON                                        -> src.extract

Settings shared with the evaluation scripts (e.g. PUBLIC_ENTITIES) live in src/config.py.
"""

from src.anonymize import Anonymizer, anonymize
from src.config import OUTPUT_DIR, PUBLIC_ENTITIES
from src.extract import extract
from src.ingest import Message, load
from src.relevance import classify
from src.transcribe import transcribe
from src.utils import write_json

# Any mix of WhatsApp exports (.zip/.txt/folder), .json/.jsonl, or plain .txt
INPUTS = [
    "WhatsAppExports/export2",
]
MESSAGES_OUTPUT = OUTPUT_DIR / "messages.json"  # every message, processed and anonymized
REPORTS_OUTPUT = OUTPUT_DIR / "reports.json"    # only relevant messages, with extracted fields

# Speed knobs for test runs
LIMIT: int | None = 20   # only the first N messages of each input; None = all
EXTRACTION_CONTEXT = 5   # earlier messages used to read status; 0 = faster, less accurate

# Console summary layout
ID_COLUMN_WIDTH = 22
EXTRACTION_INDENT = ID_COLUMN_WIDTH + 2  # lines extracted fields up under the timestamp
SCORE_DECIMALS = 3
MISSING = "-"

# Raw (non-anonymized) fields removed before anything is written to disk.
RAW_MESSAGE_FIELDS = ("text", "sender", "metadata")


def summarize(message: Message) -> str:
    """One console line per message (plus extracted fields, if any)."""
    when = message.timestamp.strftime("%Y-%m-%d %H:%M") if message.timestamp else MISSING
    attachments = "".join(f" [{attachment.kind}: {attachment.filename or 'omitted'}]"
                          for attachment in message.attachments)
    relevance = (f"{message.relevance['label']} {message.relevance['score']:.2f}"
                 if message.relevance else MISSING)
    sender = message.anonymized_sender or MISSING
    line = f"{message.id:<{ID_COLUMN_WIDTH}} {when}  ({relevance})  {sender}: {message.anonymized_text}{attachments}"
    if message.extraction:
        found_fields = {field: value for field, value in message.extraction.items() if value}
        line += f"\n{'':<{EXTRACTION_INDENT}}-> {found_fields}"
    return line


def anonymized_record(message: Message) -> dict:
    """message.to_dict() without anything raw: original text, sender, transcripts, source metadata."""
    record = message.to_dict()
    for key in RAW_MESSAGE_FIELDS:
        del record[key]
    for attachment in record["attachments"]:
        del attachment["transcript"]  # transcripts are already in anonymized_text
    return record


def report(message: Message) -> dict:
    """One flat activity record for staff to review."""
    return {
        "id": message.id,
        "conversation": message.conversation,
        "timestamp": message.iso_timestamp,
        "sender": message.anonymized_sender,
        "text": message.anonymized_text,
        "relevance_score": round(message.relevance["score"], SCORE_DECIMALS),
        **message.extraction,
    }


def save(path, data) -> None:
    write_json(path, data)
    print(f"Wrote {path}")


def main() -> None:
    # 1) Parse
    # Drop notices like "Messages are end-to-end encrypted", then apply LIMIT per input
    messages = []
    for path in INPUTS:
        messages += [message for message in load(path) if not message.is_system][:LIMIT]
    print(f"1) Loaded {len(messages)} messages from {len(INPUTS)} input(s)", flush=True)

    # 2) Transcribe (audio transcripts then show up in message.full_text)
    audio_count = sum(len(message.audio) for message in messages)
    print(f"2) Transcribing {audio_count} audio attachment(s)", flush=True)
    transcribe(messages)

    # 3) Anonymize (fills anonymized_text, redactions, anonymized_sender)
    print("3) Anonymizing", flush=True)
    anonymize(messages, Anonymizer(public_entities=PUBLIC_ENTITIES))

    # 4) Determine relevance on the anonymized text (fills relevance)
    print("4) Classifying relevance", flush=True)
    classify(messages)

    # 5) Extract activity fields from relevant messages (fills extraction)
    print("5) Extracting activity fields", flush=True)
    extract(messages, context_size=EXTRACTION_CONTEXT)

    for message in messages:
        print(summarize(message))
    reports = [report(message) for message in messages if message.extraction]
    print(f"{len(reports)} of {len(messages)} messages kept as activity reports")
    save(MESSAGES_OUTPUT, [anonymized_record(message) for message in messages])
    save(REPORTS_OUTPUT, reports)


if __name__ == "__main__":
    main()
