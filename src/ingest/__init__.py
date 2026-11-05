"""Turn any supported input into a list of `Message` records.

    from src.ingest import load
    messages = load("WhatsAppExports/export.zip")
    records = [message.to_dict() for message in messages]
"""

import json
from pathlib import Path

from ..utils import write_json
from .base import SOURCES, Source, register
from .schema import Attachment, Message

# Importing a module registers its Source. Auto-detection tries them in this
# order, so the generic plain-text fallback must come last.
from . import whatsapp, json_source, text_source  # noqa: F401, E402

__all__ = ["Attachment", "Message", "Source", "register", "load", "save_json", "load_json"]


def load(path: str | Path, source: str | None = None, **options) -> list[Message]:
    """Load `path` with the named source, or auto-detect it if `source` is None.

    Extra keyword arguments are passed to the source's constructor.
    """
    path = Path(path)
    if source is not None:
        if source not in SOURCES:
            raise ValueError(f"Unknown source {source!r}; choose from {sorted(SOURCES)}")
        source_class = SOURCES[source]
    else:
        source_class = next((candidate for candidate in SOURCES.values() if candidate.can_load(path)), None)
        if source_class is None:
            raise ValueError(f"No source recognizes {path}; pass source= explicitly")
    return source_class(**options).load(path)


def save_json(messages: list[Message], path: str | Path) -> None:
    write_json(path, [message.to_dict() for message in messages])


def load_json(path: str | Path) -> list[Message]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Message.from_dict(record) for record in records]
