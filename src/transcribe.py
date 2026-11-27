"""Step 2: transcribe audio attachments with a Hugging Face speech-to-text model.

    uv run main.py --input data/dandelion_audio                     # transcribe everything
    uv run main.py --input data/dandelion_audio --transcript-cache  # reuse output/transcript_cache.json
    uv run -m src.evaluation.transcription

Transcription is the slowest step, so transcripts can be cached between runs (opt-in). A cached
transcript is reused only for the same audio content (a SHA-256 hash of the file) and
the same model settings. Renaming a file still reuses it; editing a file, or changing
the model, transcribes it again.
"""

import hashlib
import json
from functools import cached_property
from pathlib import Path

from .ingest import Message
from .utils import torch_device, write_json

# Chosen on the development set (output/evaluations/transcription_dev.json): WER 0.363 vs 0.426 for
# distil-whisper/distil-small.en, at about half its speed. Whisper Large was no more accurate and ~2.5x slower.
DEFAULT_MODEL = "openai/whisper-small.en"
DEFAULT_CHUNK_SECONDS = 15  # Whisper models read at most 30 s at a time


class Transcriber:
    """Speech-to-text. The model loads on first use, so a fully cached run never loads it."""

    def __init__(self, model: str = DEFAULT_MODEL, chunk_length_s: int = DEFAULT_CHUNK_SECONDS):
        self.model = model
        self.chunk_length_s = chunk_length_s

    @property
    def settings(self) -> str:
        """Everything that changes the output; part of each cache key."""
        return f"{self.model}|chunk={self.chunk_length_s}s"

    @cached_property
    def asr(self):
        from transformers import pipeline
        return pipeline("automatic-speech-recognition", model=self.model, chunk_length_s=self.chunk_length_s,
                        device=torch_device(allow_mps=False))

    def __call__(self, audio: str | bytes) -> str:
        """audio: a file path or the bytes of any audio file ffmpeg can read."""
        return self.asr(audio)["text"].strip()


class TranscriptCache:
    """Transcripts saved in a JSON file, keyed by audio content and model settings.

    The transcripts are raw, un-anonymized speech: never commit a cache of real messages.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.transcripts: dict[str, str] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {})

    def get(self, audio_path: str, settings: str) -> str | None:
        return self.transcripts.get(_cache_key(audio_path, settings))

    def put(self, audio_path: str, settings: str, transcript: str) -> None:
        self.transcripts[_cache_key(audio_path, settings)] = transcript

    def save(self) -> None:
        write_json(self.path, self.transcripts)


def _cache_key(audio_path: str, settings: str) -> str:
    return f"{settings}|sha256={hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()}"


def transcribe(messages: list[Message], transcriber: Transcriber | None = None,
               cache: TranscriptCache | None = None) -> int:
    """Fill in `transcript` on every audio attachment that has a local file and none yet.

    Returns how many transcripts came from `cache`.
    """
    transcriber = transcriber or Transcriber()
    pending = [attachment for message in messages for attachment in message.audio
               if attachment.path and attachment.transcript is None]
    reused = 0
    for attachment in pending:
        cached = cache.get(attachment.path, transcriber.settings) if cache else None
        if cached is not None:
            attachment.transcript = cached
            reused += 1
            continue
        attachment.transcript = transcriber(attachment.path)
        if cache:
            cache.put(attachment.path, transcriber.settings, attachment.transcript)
    if cache:
        cache.save()
    return reused
