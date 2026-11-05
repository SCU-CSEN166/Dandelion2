"""Small helpers shared by the pipeline steps and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # src.ingest imports this module, so only import it for type hints
    from .ingest import Message

JSON_INDENT = 2


def write_json(path: str | Path, data: Any) -> None:
    """Write `data` as pretty-printed UTF-8 JSON, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=JSON_INDENT, ensure_ascii=False) + "\n", encoding="utf-8")


def require_anonymized(messages: list[Message]) -> None:
    """Raise if any message skipped the anonymize step; later steps must never see raw text."""
    for message in messages:
        if message.anonymized_text is None:
            raise ValueError(f"{message.id} has not been anonymized; run anonymize() first")


def torch_device(allow_mps: bool = True) -> str:
    """The fastest available torch device: "cuda", then "mps" (Apple silicon) if allowed, else "cpu"."""
    import torch  # imported lazily so loading data never pays for torch

    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
