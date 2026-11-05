"""Step 3: mask personal information with BERT NER plus regex.

From Prableen's Week 6 baseline. Constructing an Anonymizer never downloads the
model; it loads on first use.
"""

import re
from functools import cached_property

from .ingest import Message

NER_MODEL = "dslim/bert-base-NER"

PATTERNS = {
    "EMAIL": r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w-])",
    "PHONE": r"(?<!\w)(?:\+?254[\s.-]?[17]|0[17])(?:[\s.-]?\d){8}(?!\w)",
    "NATIONAL_ID": r"\b(?:ID\s*(?:No\.?|number)?\s*[:#.-]?\s*)(\d{7,8})\b",
}
_NER_LABELS = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORG"}


class Anonymizer:
    def __init__(self, ner=None, public_entities=()):
        """
        ner: a token-classification pipeline (tests inject a fake one).
        public_entities: reviewed public places/organizations to leave unmasked.
            Never derive this list from evaluation labels.
        """
        self._ner = ner
        self.public_entities = {s.casefold() for s in public_entities}

    @cached_property
    def ner(self):
        if self._ner is not None:
            return self._ner
        from transformers import pipeline
        return pipeline("token-classification", model=NER_MODEL,
                        aggregation_strategy="simple", stride=64, device=-1)

    def detect(self, text: str, known_names=()) -> list[dict]:
        """Return original character spans, without copying private surface text.

        Unknown locations/organizations are conservatively masked. PER spans are
        always masked, as is any of `known_names` (e.g. chat senders) found in the text.
        """
        if not text.strip():
            return []
        spans = []
        for entity in self.ner(text):
            kind = entity["entity_group"]
            start, end = int(entity["start"]), int(entity["end"])
            # BERT can return only a subword; mask the whole boundary word.
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            if kind not in _NER_LABELS:
                continue
            if kind != "PER" and text[start:end].casefold() in self.public_entities:
                continue
            spans.append({"start": start, "end": end, "label": _NER_LABELS[kind]})
        for label, pattern in PATTERNS.items():
            for match in re.finditer(pattern, text, re.I):
                start, end = match.span(1 if label == "NATIONAL_ID" else 0)
                if text[start:end].casefold() not in self.public_entities:
                    spans.append({"start": start, "end": end, "label": label})
        for name in known_names:
            for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.I):
                spans.append({"start": match.start(), "end": match.end(), "label": "PERSON"})
        # Union overlapping spans so partial model predictions cannot expose suffixes.
        merged = []
        for span in sorted(spans, key=lambda s: (s["start"], -s["end"])):
            if merged and span["start"] < merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            else:
                merged.append(dict(span))
        return merged

    def anonymize(self, text: str, known_names=()) -> dict:
        spans = self.detect(text, known_names)
        result = text
        for span in reversed(spans):
            result = result[:span["start"]] + "[" + span["label"] + "]" + result[span["end"]:]
        return {"text": result, "redactions": spans}


def anonymize(messages: list[Message], anonymizer: Anonymizer | None = None) -> None:
    """Fill in `anonymized_text`, `redactions` and `anonymized_sender`.

    Senders get stable pseudonyms ("Sender 1", ...) in order of first appearance,
    and their names (full or any part of 3+ letters) are masked wherever they
    appear in message text, in case NER misses them.
    """
    anonymizer = anonymizer or Anonymizer()
    pseudonyms: dict[str, str] = {}
    for m in messages:
        if m.sender and not m.is_system:
            pseudonyms.setdefault(m.sender, f"Sender {len(pseudonyms) + 1}")
    names = set(pseudonyms) | {part for name in pseudonyms for part in name.split() if len(part) >= 3}

    for m in messages:
        result = anonymizer.anonymize(m.full_text, known_names=names)
        m.anonymized_text = result["text"]
        m.redactions = result["redactions"]
        m.anonymized_sender = pseudonyms.get(m.sender)
