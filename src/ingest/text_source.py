"""Plain text: one message per line (or per blank-line-separated block).

This is the fallback for .txt files that aren't WhatsApp exports, so it must be
registered after `WhatsAppSource`.
"""

from pathlib import Path

from .base import Source, register
from .schema import Message


@register
class TextSource(Source):
    name = "text"

    def __init__(self, separator: str = "\n"):
        """separator: "\\n" for one message per line, "\\n\\n" for paragraphs."""
        self.separator = separator

    @classmethod
    def can_load(cls, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".txt"

    def load(self, path: Path) -> list[Message]:
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        chunks = [c.strip() for c in text.split(self.separator)]
        return [
            Message(id=f"{self.name}:{path.stem}:{i}", source=self.name, text=chunk, conversation=path.stem)
            for i, chunk in enumerate(c for c in chunks if c)
        ]
