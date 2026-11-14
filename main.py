"""Dandelion pipeline entry point. Edit the settings below, then:

    uv run main.py

Or run the whole pipeline on one message instead of INPUTS (prints the result and
writes no files):

    uv run main.py --message "Today we held a youth health session with 20 girls"
    uv run main.py --audio data/WhatsAppExports/export2/00003084-AUDIO-2026-11-04-16-05-11.opus

Or run a folder of voice notes / recordings (.mp4 etc. are converted to audio-only .mp3 first):

    uv run main.py --folder data/dandelion_audio

Steps:
    1) Parse input (WhatsApp export, JSON/JSONL, plain text)  -> src.ingest
    2) Transcribe audio attachments                           -> src.transcribe
    3) Anonymize                                              -> src.anonymize
    4) Determine relevance                                    -> src.relevance
    5) Extract to JSON                                        -> src.extract

Settings shared with the evaluation scripts (e.g. PUBLIC_ENTITIES) live in src/config.py.
"""

import argparse
import csv
import json
from pathlib import Path

from src.anonymize import Anonymizer, CombinedNER, GlinerNER, anonymize
from src.config import (EXTRACTION_BACKEND, OUTPUT_DIR, PUBLIC_ENTITIES, RELEVANCE_COUNT_PLANS, RELEVANCE_MIN_WORDS,
                        RELEVANCE_SKIP_QUESTIONS, USE_GLINER)
from src.extract import extract, make_extractor
from src.ingest import Attachment, Message, load
from src.relevance import RelevanceClassifier, classify
from src.transcribe import transcribe
from src.utils import write_json

# Any mix of WhatsApp exports (.zip/.txt/folder), .json/.jsonl, or plain .txt
INPUTS = [
    "data/WhatsAppExports/export2",
]
MESSAGES_OUTPUT = OUTPUT_DIR / "messages.json"  # every message, processed and anonymized
REPORTS_OUTPUT = OUTPUT_DIR / "reports.json"    # only relevant messages, with extracted fields
CSV_OUTPUT = OUTPUT_DIR / "reports.csv"         # every message, one row each, in Dandelion's format

# CSV export format
CSV_COLUMNS = ["Message_id", "anonymized_text", "relevance", "activity_type", "participants", "duplicate_of"]
CSV_RELEVANCE_LABELS = {"relevant": "Event", "irrelevant": "Not relevant"}
CSV_NONE = "none"

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


def csv_rows(messages: list[Message]) -> list[dict]:
    """One row per message in Dandelion's CSV format; Message_id is the 1-based row number."""
    row_ids = {message.id: row_id for row_id, message in enumerate(messages, start=1)}
    rows = []
    for message in messages:
        fields = message.extraction or {}
        activity = fields.get("activity_type")
        participants = fields.get("participant_count")
        rows.append({
            "Message_id": row_ids[message.id],
            "anonymized_text": " ".join(message.anonymized_text.split()),  # one line per row
            "relevance": CSV_RELEVANCE_LABELS[message.relevance["label"]] if message.relevance else CSV_NONE,
            "activity_type": activity.capitalize() if activity else CSV_NONE,
            "participants": participants if participants is not None else CSV_NONE,
            "duplicate_of": CSV_NONE,  # duplicate detection not implemented yet
        })
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def single_message(text: str, audio_path: str | None) -> Message:
    """A Message built from the command line, as if it came from a chat."""
    attachments = [Attachment(kind="audio", filename=Path(audio_path).name, path=audio_path)] if audio_path else []
    return Message(id="cli:0", source="cli", text=text, attachments=attachments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dandelion pipeline on INPUTS, or on one message.")
    parser.add_argument("--message", help="run only this message text (no files are written)")
    parser.add_argument("--audio", help="run only this audio file as a voice note (with --message: text + audio)")
    parser.add_argument("--folder", help="run this folder of voice notes / recordings instead of INPUTS")
    args = parser.parse_args()
    if args.folder and not Path(args.folder).is_dir():
        parser.error(f"folder not found: {args.folder}")
    if args.audio and not Path(args.audio).is_file():
        parser.error(f"audio file not found: {args.audio}")
    return args


def save(path, data) -> None:
    write_json(path, data)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    one_message = args.message is not None or args.audio is not None

    # 1) Parse
    if one_message:
        messages = [single_message(args.message or "", args.audio)]
        print("1) Loaded 1 message from the command line", flush=True)
    else:
        # Drop notices like "Messages are end-to-end encrypted", then apply LIMIT per input
        inputs = [args.folder] if args.folder else INPUTS
        messages = []
        for path in inputs:
            messages += [message for message in load(path) if not message.is_system][:LIMIT]
        print(f"1) Loaded {len(messages)} messages from {len(inputs)} input(s)", flush=True)

    # 2) Transcribe (audio transcripts then show up in message.full_text)
    audio_count = sum(len(message.audio) for message in messages)
    print(f"2) Transcribing {audio_count} audio attachment(s)", flush=True)
    transcribe(messages)

    # 3) Anonymize (fills anonymized_text, redactions, anonymized_sender)
    print("3) Anonymizing", flush=True)
    # BERT + GLiNER together when USE_GLINER (Anonymizer().ner is the default BERT pipeline).
    ner = CombinedNER(Anonymizer().ner, GlinerNER()) if USE_GLINER else None
    anonymize(messages, Anonymizer(ner=ner, public_entities=PUBLIC_ENTITIES))

    # 4) Determine relevance on the anonymized text (fills relevance)
    print("4) Classifying relevance", flush=True)
    classify(messages, RelevanceClassifier(count_plans=RELEVANCE_COUNT_PLANS, min_words=RELEVANCE_MIN_WORDS,
                                           skip_questions=RELEVANCE_SKIP_QUESTIONS))

    # 5) Extract activity fields from relevant messages (fills extraction)
    print("5) Extracting activity fields", flush=True)
    extract(messages, make_extractor(EXTRACTION_BACKEND), context_size=EXTRACTION_CONTEXT)

    for message in messages:
        print(summarize(message))
    if one_message:
        # Print instead of saving, so a quick test never overwrites a full run's output files.
        message = messages[0]
        result = report(message) if message.extraction else anonymized_record(message)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    reports = [report(message) for message in messages if message.extraction]
    print(f"{len(reports)} of {len(messages)} messages kept as activity reports")
    save(MESSAGES_OUTPUT, [anonymized_record(message) for message in messages])
    save(REPORTS_OUTPUT, reports)
    save_csv(CSV_OUTPUT, csv_rows(messages))


if __name__ == "__main__":
    main()

