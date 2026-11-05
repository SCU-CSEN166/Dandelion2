"""Step 3: mask personal information with BERT NER plus regex.

From Prableen's Week 6 baseline. Constructing an Anonymizer never downloads the
model; it loads on first use.
"""

import re
from functools import cached_property

from .ingest import Message
from .utils import torch_device

NER_MODEL = "dslim/bert-base-NER"
NER_STRIDE_TOKENS = 64  # overlap between windows when a message exceeds BERT's input length

# Each pattern's `value` group is the part that gets masked.
PII_PATTERNS = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in {
        "EMAIL": r"(?<![\w.+-])(?P<value>[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})(?![\w-])",
        # Kenyan mobile numbers: +254 7xx / 01xx ... or the local 07xx / 01xx form.
        "PHONE": r"(?<!\w)(?P<value>(?:\+?254[\s.-]?[17]|0[17])(?:[\s.-]?\d){8})(?!\w)",
        # Only the digits after an "ID" / "ID No." / "ID number" prefix.
        "NATIONAL_ID": r"\b(?:ID\s*(?:No\.?|number)?\s*[:#.-]?\s*)(?P<value>\d{7,8})\b",
    }.items()
}

# NER entity group -> redaction label. Other groups (e.g. MISC) are left alone.
NER_LABELS = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORG"}
PERSON_NER_GROUP = "PER"
PERSON_LABEL = NER_LABELS[PERSON_NER_GROUP]

# Parts of a sender's name shorter than this ("Al", "Jo") are too likely to match ordinary words.
MIN_NAME_PART_LENGTH = 3


def _span(start: int, end: int, label: str) -> dict:
    return {"start": start, "end": end, "label": label}


def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) to whole-word boundaries; BERT can return only a subword."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


def _merge_overlapping(spans: list[dict]) -> list[dict]:
    """Union overlapping spans so partial model predictions cannot expose suffixes."""
    merged: list[dict] = []
    for span in sorted(spans, key=lambda span: (span["start"], -span["end"])):
        if merged and span["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


class Anonymizer:
    def __init__(self, ner=None, public_entities=()):
        """
        ner: a token-classification pipeline (tests inject a fake one).
        public_entities: reviewed public places/organizations to leave unmasked.
            Never derive this list from evaluation labels.
        """
        self._ner = ner
        self.public_entities = {entity.casefold() for entity in public_entities}

    @cached_property
    def ner(self):
        if self._ner is not None:
            return self._ner
        from transformers import pipeline
        return pipeline("token-classification", model=NER_MODEL,
                        aggregation_strategy="simple", stride=NER_STRIDE_TOKENS, device=torch_device(allow_mps=False))

    def _is_public(self, surface_text: str) -> bool:
        return surface_text.casefold() in self.public_entities

    def _ner_spans(self, text: str) -> list[dict]:
        """PERSON spans are always masked; unknown locations/organizations are masked conservatively."""
        spans = []
        for entity in self.ner(text):
            group = entity["entity_group"]
            if group not in NER_LABELS:
                continue
            start, end = _expand_to_word(text, int(entity["start"]), int(entity["end"]))
            if group != PERSON_NER_GROUP and self._is_public(text[start:end]):
                continue
            spans.append(_span(start, end, NER_LABELS[group]))
        return spans

    def _pattern_spans(self, text: str) -> list[dict]:
        spans = []
        for label, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                start, end = match.span("value")
                if not self._is_public(text[start:end]):
                    spans.append(_span(start, end, label))
        return spans

    @staticmethod
    def _known_name_spans(text: str, known_names) -> list[dict]:
        return [
            _span(match.start(), match.end(), PERSON_LABEL)
            for name in known_names
            for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)
        ]

    def detect(self, text: str, known_names=()) -> list[dict]:
        """Return original character spans, without copying private surface text.

        known_names (e.g. chat senders) are always masked wherever they appear.
        """
        if not text.strip():
            return []
        spans = self._ner_spans(text) + self._pattern_spans(text) + self._known_name_spans(text, known_names)
        return _merge_overlapping(spans)

    def anonymize(self, text: str, known_names=()) -> dict:
        spans = self.detect(text, known_names)
        masked = text
        for span in reversed(spans):  # right to left, so earlier offsets stay valid
            masked = masked[:span["start"]] + f"[{span['label']}]" + masked[span["end"]:]
        return {"text": masked, "redactions": spans}


def sender_pseudonyms(messages: list[Message]) -> dict[str, str]:
    """Map each real sender to "Sender 1", "Sender 2", ... in order of first appearance."""
    pseudonyms: dict[str, str] = {}
    for message in messages:
        if message.sender and not message.is_system:
            pseudonyms.setdefault(message.sender, f"Sender {len(pseudonyms) + 1}")
    return pseudonyms


def names_to_mask(senders) -> set[str]:
    """Each full sender name plus every part of it long enough to be distinctive."""
    parts = {part for name in senders for part in name.split() if len(part) >= MIN_NAME_PART_LENGTH}
    return set(senders) | parts


def anonymize(messages: list[Message], anonymizer: Anonymizer | None = None) -> None:
    """Fill in `anonymized_text`, `redactions` and `anonymized_sender`.

    Senders get stable pseudonyms, and their names are masked wherever they
    appear in message text, in case NER misses them.
    """
    anonymizer = anonymizer or Anonymizer()
    pseudonyms = sender_pseudonyms(messages)
    known_names = names_to_mask(pseudonyms)

    for message in messages:
        result = anonymizer.anonymize(message.full_text, known_names=known_names)
        message.anonymized_text = result["text"]
        message.redactions = result["redactions"]
        message.anonymized_sender = pseudonyms.get(message.sender)
