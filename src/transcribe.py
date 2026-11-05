"""Step 2: transcribe audio attachments with a Hugging Face ASR model."""

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


def transcribe(messages: list[Message], transcriber: Transcriber | None = None) -> None:
    """Fill in `transcript` on every audio attachment that has a local file."""
    pending = [attachment for message in messages for attachment in message.audio
               if attachment.path and attachment.transcript is None]
    if pending and transcriber is None:
        transcriber = Transcriber()  # only load the model when there's audio to transcribe
    for attachment in pending:
        attachment.transcript = transcriber(attachment.path)
