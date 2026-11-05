"""Interface every input source implements.

To add a new input format (e.g. the synthetic JSON dataset or a Hugging Face
speech corpus), subclass `Source`, implement `can_load` and `load`, and
decorate the class with `@register`. `ingest.load(path)` will then pick it up.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .schema import Message


class Source(ABC):
    name: str

    @classmethod
    @abstractmethod
    def can_load(cls, path: Path) -> bool:
        """Cheap check for whether `path` looks like this source's format."""

    @abstractmethod
    def load(self, path: Path) -> list[Message]:
        """Read `path` and return its contents as normalized messages."""


SOURCES: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    SOURCES[cls.name] = cls
    return cls
