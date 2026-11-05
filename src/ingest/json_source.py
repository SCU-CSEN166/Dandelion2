"""JSON / JSONL datasets: the synthetic test set, Hugging Face exports, etc.

Accepted shapes:
    [ {...}, {...} ]                      a list of records
    { "records": [ {...}, ... ], ... }    any top-level key holding a list of records
    one {...} per line                    .jsonl

Each record's fields are matched to `Message` fields by common names (see
`ALIASES`), or explicitly via `fields=`. Every field that isn't used is kept in
`metadata`, so labels such as the synthetic set's "entities" survive for evaluation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Source, register
from .schema import Attachment, Message, kind_from_filename

ALIASES = {
    "id": ["id", "message_id", "uid"],
    "text": ["text", "message", "body", "content", "transcript", "transcription", "sentence"],
    "sender": ["sender", "author", "from", "speaker", "user", "username"],
    "timestamp": ["timestamp", "datetime", "date", "time", "created_at"],
    "conversation": ["conversation", "chat", "group", "thread"],
    "audio": ["audio", "audio_path", "audio_file", "path", "file"],
}
_RECORD_KEYS = ("records", "messages", "data", "items", "rows")


@register
class JsonSource(Source):
    name = "json"

    def __init__(self, fields: dict[str, str] | None = None):
        """fields: override the alias lookup, e.g. {"text": "transcription_en"}."""
        self.fields = fields or {}

    @classmethod
    def can_load(cls, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in (".json", ".jsonl")

    def load(self, path: Path) -> list[Message]:
        path = Path(path)
        records = _read_records(path)
        return [self._to_message(record, index, path) for index, record in enumerate(records)]

    def _to_message(self, record: dict[str, Any], index: int, path: Path) -> Message:
        rest = dict(record)

        def take(field: str) -> Any:
            keys = [self.fields[field]] if field in self.fields else ALIASES[field]
            for key in keys:
                if key in rest:
                    return rest.pop(key)
            return None

        record_id = take("id")
        text = take("text")
        timestamp = _parse_timestamp(take("timestamp"), rest)
        audio = take("audio")
        # Hugging Face audio columns look like {"path": ..., "array": [...], "sampling_rate": ...}
        if isinstance(audio, dict):
            audio = audio.get("path")
        attachments = [_audio_attachment(audio, path.parent)] if isinstance(audio, str) else []

        return Message(
            id=f"{self.name}:{path.stem}:{record_id if record_id is not None else index}",
            source=self.name,
            text=str(text or ""),
            sender=take("sender"),
            timestamp=timestamp,
            conversation=take("conversation") or path.stem,
            attachments=attachments,
            metadata=rest,
        )


def _audio_attachment(audio_path: str, base_dir: Path) -> Attachment:
    """An attachment for `audio_path` (relative paths are relative to the dataset file)."""
    file = Path(audio_path) if Path(audio_path).is_absolute() else base_dir / audio_path
    local_path = str(file) if file.is_file() else None
    return Attachment(kind_from_filename(audio_path), Path(audio_path).name, local_path)


def _read_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in file if line.strip()]
        data = json.load(file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _RECORD_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError(f"{path}: expected a JSON list or object")


def _parse_timestamp(value: Any, rest: dict[str, Any]) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        rest["raw_timestamp"] = value  # keep it so nothing is silently lost
        return None
