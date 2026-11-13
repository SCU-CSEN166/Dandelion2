"""Step 2: transcribe audio attachments with a Hugging Face ASR model.

Transcripts can also be loaded from a saved file so later pipeline stages can be
re-run without paying the ASR cost again. Supported transcript files:
- CSV with filename,transcript columns
- JSON mapping filename -> transcript, or a list of objects with those fields
- tab-separated text: filename<TAB>transcript
"""

import csv
import json
from pathlib import Path

from .ingest import Message
from .utils import torch_device

DEFAULT_MODEL = "distil-whisper/distil-small.en"
DEFAULT_CHUNK_SECONDS = 15  # Whisper-family models handle at most 30 s per window


class Transcriber:
    def __init__(self, model: str = DEFAULT_MODEL, chunk_length_s: int = DEFAULT_CHUNK_SECONDS):
        """Loads the model immediately; construct one only when there's audio to transcribe."""
        from transformers import pipeline

        self.model = model
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=chunk_length_s,
            device=torch_device(allow_mps=False),
        )

    def __call__(self, audio: str | bytes) -> str:
        """audio: a file path or the raw bytes of an audio file (anything ffmpeg can read)."""
        return self.asr(audio)["text"].strip()


def _transcript_keys(filename: str) -> tuple[str, str]:
    """Match saved transcripts to either the full filename or just its stem."""
    path = Path(filename)
    return path.name, path.stem


def load_transcripts(path: str | Path) -> dict[str, str]:
    """Load previously generated transcripts, keyed by both filename and file stem."""
    path = Path(path)
    mapping: dict[str, str] = {}

    def add(filename: str, transcript: str) -> None:
        if not filename or not transcript:
            return
        for key in _transcript_keys(filename):
            mapping[key] = transcript.strip()

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for filename, transcript in data.items():
                add(str(filename), str(transcript))
        else:
            for row in data:
                add(str(row.get("filename", row.get("file", ""))), str(row.get("transcript", "")))
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                add(row.get("filename", row.get("file", "")), row.get("transcript", ""))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            filename, separator, transcript = line.partition("\t")
            if separator:
                add(filename, transcript)

    return mapping


def apply_transcripts(messages: list[Message], transcripts: dict[str, str]) -> int:
    """Fill audio transcripts from a saved mapping. Returns the number reused."""
    reused = 0
    for message in messages:
        for attachment in message.audio:
            candidates = []
            if attachment.filename:
                candidates.extend(_transcript_keys(attachment.filename))
            if attachment.path:
                candidates.extend(_transcript_keys(attachment.path))
            transcript = next((transcripts[key] for key in candidates if key in transcripts), None)
            if transcript is not None:
                attachment.transcript = transcript
                reused += 1
    return reused


def save_transcripts(messages: list[Message], path: str | Path) -> None:
    """Save current audio transcripts to a reusable CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "transcript"])
        writer.writeheader()
        for message in messages:
            for attachment in message.audio:
                if attachment.transcript:
                    writer.writerow({
                        "filename": attachment.filename or Path(attachment.path or "audio").name,
                        "transcript": attachment.transcript,
                    })


def transcribe(messages: list[Message], transcriber: Transcriber | None = None) -> None:
    """Fill in `transcript` on every audio attachment that has a local file."""
    pending = [attachment for message in messages for attachment in message.audio
               if attachment.path and attachment.transcript is None]
    if pending and transcriber is None:
        transcriber = Transcriber()  # only load the model when there's audio to transcribe
    for attachment in pending:
        attachment.transcript = transcriber(attachment.path)
