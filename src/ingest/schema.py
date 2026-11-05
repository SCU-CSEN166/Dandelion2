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
    ext = filename.rsplit(".", 1)[-1].lower()
    return next((kind for kind, exts in _EXTENSIONS.items() if ext in exts), "other")


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
        return [a for a in self.attachments if a.kind == "audio"]

    @property
    def full_text(self) -> str:
        """Typed text plus any audio transcripts: what later steps should read."""
        parts = [self.text] + [a.transcript for a in self.audio if a.transcript]
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        d = dict(d)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else None
        d["attachments"] = [Attachment(**a) for a in d.get("attachments", [])]
        return cls(**d)
