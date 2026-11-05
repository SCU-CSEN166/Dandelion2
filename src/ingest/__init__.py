"""Turn any supported input into a list of `Message` records.

    from src.ingest import load
    messages = load("WhatsAppExports/export.zip")
    records = [m.to_dict() for m in messages]
"""

import json
from pathlib import Path

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
        cls = SOURCES[source]
    else:
        cls = next((c for c in SOURCES.values() if c.can_load(path)), None)
        if cls is None:
            raise ValueError(f"No source recognizes {path}; pass source= explicitly")
    return cls(**options).load(path)


def save_json(messages: list[Message], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in messages], f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> list[Message]:
    with open(path, encoding="utf-8") as f:
        return [Message.from_dict(d) for d in json.load(f)]
