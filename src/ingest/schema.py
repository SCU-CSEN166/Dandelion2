"""Common record format that every input source is normalized into.

Downstream steps (transcription, anonymization, relevance, extraction) only
see `Message` objects, so they don't care whether the data came from a
WhatsApp export, the synthetic dataset, or a speech corpus.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

AttachmentKind = Literal["audio", "image", "video", "document", "other"]

_EXTENSIONS: dict[AttachmentKind, set[str]] = {
    "audio": {"opus", "ogg", "m4a", "mp3", "wav", "aac", "amr", "flac"},
    "image": {"jpg", "jpeg", "png", "webp", "gif", "heic"},
    "video": {"mp4", "mov", "3gp", "avi", "mkv"},
    "document": {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv"},
}


def kind_from_filename(filename: str) -> AttachmentKind:
    extension = filename.rsplit(".", 1)[-1].lower()
    return next((kind for kind, extensions in _EXTENSIONS.items() if extension in extensions), "other")


@dataclass
class Attachment:
    kind: AttachmentKind
    filename: str | None = None  # None when the export only says "<Media omitted>"
    path: str | None = None  # local file, or None when the media wasn't included
    transcript: str | None = None  # filled in by the transcription step


@dataclass
class Message:
    id: str  # unique within a run, e.g. "whatsapp:export:3"
    source: str  # which Source produced it, e.g. "whatsapp"
    text: str = ""
    sender: str | None = None
    timestamp: datetime | None = None
    conversation: str | None = None  # chat / file / dataset the message belongs to
    attachments: list[Attachment] = field(default_factory=list)
    is_system: bool = False  # WhatsApp notices like "Messages are end-to-end encrypted"
    metadata: dict[str, Any] = field(default_factory=dict)  # source-specific extras

    # Filled in by later pipeline steps
    anonymized_text: str | None = None
    anonymized_sender: str | None = None  # stable pseudonym, e.g. "Sender 1"
    redactions: list[dict[str, Any]] = field(default_factory=list)  # spans into full_text
    relevance: dict[str, Any] | None = None  # {"label": "relevant" | "irrelevant", "score": ...}
    extraction: dict[str, Any] | None = None  # activity fields; only for relevant messages

    @property
    def audio(self) -> list[Attachment]:
        return [attachment for attachment in self.attachments if attachment.kind == "audio"]

    @property
    def full_text(self) -> str:
        """Typed text plus any audio transcripts: what later steps should read."""
        parts = [self.text] + [attachment.transcript for attachment in self.audio]
        return "\n".join(part for part in parts if part)

    @property
    def iso_timestamp(self) -> str | None:
        return self.timestamp.isoformat() if self.timestamp else None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["timestamp"] = self.iso_timestamp
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Message":
        record = dict(record)
        record["timestamp"] = datetime.fromisoformat(record["timestamp"]) if record.get("timestamp") else None
        record["attachments"] = [Attachment(**attachment) for attachment in record.get("attachments", [])]
        return cls(**record)
