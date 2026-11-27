"""Dandelion pipeline entry point.

Run the inputs listed in INPUTS below:

    uv run main.py

Run other inputs instead. Any mix of files and folders works; the format is detected:

    # a folder of voice notes or videos
    uv run main.py --input data/dandelion_audio
    # a WhatsApp export: unzipped folder, .zip, or just the chat .txt (no media)
    uv run main.py --input data/WhatsAppExports/export2
    uv run main.py --input "data/WhatsAppExports/export/WhatsApp Chat - dandelion africa.zip"
    uv run main.py --input data/WhatsAppExports/export2/_chat.txt
    # JSON / JSONL records: one file, several files, or a whole folder of them
    uv run main.py --input data/synthetic_unanonymized.json
    uv run main.py --input data/synthetic_unanonymized.json data/whatsapp_labels.json
    uv run main.py --input my_reports/*.json
    # plain text, one message per line
    uv run main.py --input notes.txt

Each input is cut to its first LIMIT (20) messages; --limit changes that, and 0 means all:

    uv run main.py --input data/WhatsAppExports/export2 --limit 0

Run one message (prints the result and writes no files):

    uv run main.py --message "Today we held a youth health session with 20 girls"
    uv run main.py --audio data/WhatsAppExports/export2/00003084-AUDIO-2026-11-04-16-05-11.opus

Other options:

    # reuse transcripts from earlier runs (see below)
    uv run main.py --input data/dandelion_audio --transcript-cache
    # a different speech-to-text model (the default is openai/whisper-small.en)
    uv run main.py --input data/dandelion_audio --asr-model distil-whisper/distil-small.en

--transcript-cache reuses audio already transcribed with the same model, from
output/transcript_cache.json or the path given after it (see src/transcribe.py).

Steps:
    1) Parse input (WhatsApp, audio folder, JSON, plain text) -> src.ingest
    2) Transcribe audio attachments                           -> src.transcribe
    3) Anonymize                                              -> src.anonymize
    4) Determine relevance                                    -> src.relevance
    5) Extract to JSON                                        -> src.extract
    6) Flag possible duplicates                               -> src.duplicates

Settings shared with the evaluation scripts (e.g. PUBLIC_ENTITIES) live in src/config.py.
"""

import argparse
import csv
import json
from pathlib import Path

from src.anonymize import Anonymizer, CombinedNER, GlinerNER, anonymize, bert_ner
from src.config import (
    EXTRACTION_BACKEND,
    OUTPUT_DIR,
    PUBLIC_ENTITIES,
    RELEVANCE_COUNT_PLANS,
    RELEVANCE_MIN_WORDS,
    RELEVANCE_SKIP_QUESTIONS,
    TRANSCRIPT_CACHE,
    USE_GLINER,
)
from src.duplicates import mark_duplicates
from src.extract import extract, make_extractor
from src.ingest import Attachment, Message, load
from src.relevance import RelevanceClassifier, classify
from src.transcribe import DEFAULT_MODEL as DEFAULT_ASR_MODEL
from src.transcribe import Transcriber, TranscriptCache, transcribe
from src.utils import write_json

# Run when no --input is given. Any mix of WhatsApp exports (.zip/.txt/folder), folders of
# audio or video, .json/.jsonl, or plain .txt.
INPUTS = [
    "data/WhatsAppExports/export2",
]
MESSAGES_OUTPUT = OUTPUT_DIR / "messages.json"  # every message, processed and anonymized
REPORTS_OUTPUT = OUTPUT_DIR / "reports.json"    # only relevant messages, with extracted fields
CSV_OUTPUT = OUTPUT_DIR / "reports.csv"         # every message, one row each, in Dandelion's format

# CSV export format
CSV_COLUMNS = ["Message_id", "anonymized_text", "relevance", "activity_type", "participants", "duplicate_of",
               "activity_description", "needs_review", "review_reasons"]
CSV_RELEVANCE_LABELS = {"relevant": "Event", "irrelevant": "Not relevant"}
CSV_NONE = "none"

# Speed knobs for test runs
LIMIT = 20               # only the first N messages of each input; 0 = all (or use --limit)
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
    if message.duplicate_of:
        line += f"\n{'':<{EXTRACTION_INDENT}}-> possible duplicate of {message.duplicate_of}"
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
    fields = message.extraction or {}
    reasons = review_reasons(message)
    return {
        "id": message.id,
        "conversation": message.conversation,
        "timestamp": message.iso_timestamp,
        "sender": message.anonymized_sender,
        "text": message.anonymized_text,
        "relevance_score": round(message.relevance["score"], SCORE_DECIMALS),
        **{key: value for key, value in fields.items() if key != "needs_review"},
        "extraction_review_fields": fields.get("needs_review", []),
        "duplicate_of": message.duplicate_of,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }


def review_reasons(message: Message) -> list[str]:
    """Explain why a person should inspect this record before relying on it."""
    reasons = []
    if message.audio:
        reasons.append("Check the audio transcript, especially names, locations, and phone numbers.")
    if message.redactions:
        reasons.append("Verify that personal information was fully redacted.")
    if message.relevance and message.relevance.get("uncertain"):
        reasons.append("Review the relevance decision; its score is near the classification cutoff.")
    fields = message.extraction or {}
    for field in fields.get("needs_review", []):
        reasons.append(f"Verify the extracted {field.replace('_', ' ')}.")
    if fields.get("activity_type") == "other":
        reasons.append("Review the activity category; the model returned 'other'.")
    if message.duplicate_of:
        reasons.append(f"Check whether this is a duplicate of {message.duplicate_of}.")
    return reasons


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
            "duplicate_of": row_ids.get(message.duplicate_of, CSV_NONE),
            "activity_description": fields.get("activity_description") or CSV_NONE,
            "needs_review": "TRUE" if review_reasons(message) else "FALSE",
            "review_reasons": "; ".join(review_reasons(message)) or CSV_NONE,
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
    parser.add_argument("--input", "--folder", dest="inputs", nargs="+", metavar="PATH",
                        help="run these files or folders instead of INPUTS (any supported format)")
    parser.add_argument("--limit", type=int, default=LIMIT,
                        help="only the first N messages of each input; 0 = all (default: %(default)s)")
    parser.add_argument("--message", help="run only this message text (no files are written)")
    parser.add_argument("--audio", help="run only this audio file as a voice note (with --message: text + audio)")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="speech-to-text model (default: %(default)s)")
    parser.add_argument("--transcript-cache", nargs="?", const=TRANSCRIPT_CACHE, type=Path, metavar="PATH",
                        help=f"reuse and save transcripts in this JSON file (default path: {TRANSCRIPT_CACHE})")
    args = parser.parse_args()
    missing = [path for path in args.inputs or [] if not Path(path).exists()]
    if missing:
        parser.error(f"input not found: {', '.join(missing)}")
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
        # Drop notices like "Messages are end-to-end encrypted", then apply the limit per input
        inputs = args.inputs or INPUTS
        limit = args.limit or None  # 0 means no limit
        messages = []
        for path in inputs:
            messages += [message for message in load(path) if not message.is_system][:limit]
        print(f"1) Loaded {len(messages)} messages from {len(inputs)} input(s)", flush=True)

    # 2) Transcribe (audio transcripts then show up in message.full_text)
    audio_count = sum(len(message.audio) for message in messages)
    print(f"2) Transcribing {audio_count} audio attachment(s) with {args.asr_model}", flush=True)
    cache = TranscriptCache(args.transcript_cache) if args.transcript_cache else None
    reused = transcribe(messages, Transcriber(model=args.asr_model), cache)
    if cache:
        print(f"   {reused} reused from {args.transcript_cache}", flush=True)

    # 3) Anonymize (fills anonymized_text, redactions, anonymized_sender)
    print("3) Anonymizing", flush=True)
    ner = CombinedNER(bert_ner(), GlinerNER()) if USE_GLINER else None  # None: BERT alone
    anonymize(messages, Anonymizer(ner=ner, public_entities=PUBLIC_ENTITIES))

    # 4) Determine relevance on the anonymized text (fills relevance)
    print("4) Classifying relevance", flush=True)
    classify(messages, RelevanceClassifier(count_plans=RELEVANCE_COUNT_PLANS, min_words=RELEVANCE_MIN_WORDS,
                                           skip_questions=RELEVANCE_SKIP_QUESTIONS))

    # 5) Extract activity fields from relevant messages (fills extraction)
    print("5) Extracting activity fields", flush=True)
    extract(messages, make_extractor(EXTRACTION_BACKEND), context_size=EXTRACTION_CONTEXT)

    # 6) Flag messages that may repeat an earlier report (fills duplicate_of)
    print("6) Flagging possible duplicates", flush=True)
    mark_duplicates(messages)

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

