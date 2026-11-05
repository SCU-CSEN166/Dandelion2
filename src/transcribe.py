"""Step 2: transcribe audio attachments with a Hugging Face ASR model."""

import torch
from transformers import pipeline

from .ingest import Message

DEFAULT_MODEL = "distil-whisper/distil-small.en"


class Transcriber:
    def __init__(self, model: str = DEFAULT_MODEL, chunk_length_s: int = 15):
        self.model = model
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=chunk_length_s,
            device=0 if torch.cuda.is_available() else -1,
        )

    def __call__(self, audio: str | bytes) -> str:
        """audio: a file path or the raw bytes of an audio file (anything ffmpeg can read)."""
        return self.asr(audio)["text"].strip()


def transcribe(messages: list[Message], transcriber: Transcriber | None = None) -> None:
    """Fill in `transcript` on every audio attachment that has a local file."""
    pending = [a for m in messages for a in m.audio if a.path and a.transcript is None]
    if pending and transcriber is None:
        transcriber = Transcriber()  # only load the model when there's audio to transcribe
    for a in pending:
        a.transcript = transcriber(a.path)
