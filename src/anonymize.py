"""Step 3: mask personal information with NER models (BERT, GLiNER) plus regex patterns.

    uv run main.py --message "Grace Wanjiru's number is 0712 345 678"
    uv run -m src.evaluation.anonymization data/synthetic_unanonymized.json data/whatsapp_labels.json

Models load on first use, so creating an Anonymizer is free.
"""

import re
from functools import cached_property

from .ingest import Message
from .utils import torch_device

NER_MODEL = "dslim/bert-base-NER"
NER_STRIDE_TOKENS = 64  # overlap between windows when a message is longer than BERT's input

GLINER_MODEL = "knowledgator/gliner-pii-base-v1.0"
GLINER_THRESHOLD = 0.5
# What GLiNER is asked to find -> the entity group BERT would report.
GLINER_LABELS = {"person": "PER", "location": "LOC", "organization": "ORG", "date of birth": "DOB"}

# Entity group -> mask label. Other groups (e.g. MISC) are left alone. Only GLiNER finds DOB.
NER_LABELS = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORG", "DOB": "DATE_OF_BIRTH"}
PERSON_NER_GROUP = "PER"
PERSON_LABEL = NER_LABELS[PERSON_NER_GROUP]

# Sender name parts shorter than this ("Al", "Jo") would match ordinary words.
MIN_NAME_PART_LENGTH = 3

# Digit words Whisper writes when a number is read aloud ("0711100 for 82").
_SPOKEN_DIGIT_WORDS = r"(?:oh|zero|one|two|to|too|three|four|for|five|six|seven|eight|nine|double|triple)"

# (label, pattern) pairs. Only each pattern's `value` group is masked. (?-i:...) marks
# case-sensitive parts, so ordinary lowercase words never match the ID-code patterns.
PII_PATTERNS = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in [
        ("EMAIL", r"(?<![\w.+-])(?P<value>[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})(?![\w-])"),
        # Kenyan mobiles: +254 7xx / +254 1xx, or 07xx / 01xx.
        ("PHONE", r"(?<!\w)(?P<value>(?:\+?254[\s.-]?[17]|0[17])(?:[\s.-]?\d){8})(?!\w)"),
        # The same as speech-to-text writes them: stray separators ("079--0-100-274"), maybe a digit short.
        # Commas too: Whisper writes slowly read digits as "0722, 000, 917", which left "000, 917" visible.
        ("PHONE", r"(?<![\w.])(?P<value>0[\s.,-]*[17](?:[\s.,-]*\d){7,8})(?!\d)"),
        # Anything number-like after "my number is" / "call me on", spoken digit words included.
        ("PHONE", (rf"\b(?:number\s+is|number\s*:|(?:call|reach|contact|text)\s+(?:me|her|him|us|them)\s+on)\s*"
                   rf"(?P<value>\d(?:[\d.\-]|\s(?=[\d.\-]|{_SPOKEN_DIGIT_WORDS}\b)|{_SPOKEN_DIGIT_WORDS}\b)*\d)")),
        # The digits after "ID" / "ID No." / "ID number".
        ("NATIONAL_ID", r"\b(?:ID\s*(?:No\.?|number)?\s*[:#.-]?\s*)(?P<value>\d{7,8})\b"),
        # M-Pesa transaction codes: 10 capitals and digits, mixing both ("SCK7H2LQ9D").
        ("MPESA_CODE", r"(?<![\w-])(?P<value>(?-i:(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}))(?![\w-])"),
        # Health record numbers: capital-letter groups, then digits ("NKR-MCH-004512").
        ("HEALTH_RECORD_NO", r"(?<![\w-])(?P<value>(?-i:[A-Z]{2,5}(?:-[A-Z]{2,5})*-\d{3,}))(?![\w-])"),
        # Police Occurrence Book numbers ("OB 37/09/2026", "OB No. 12/3/2025").
        ("CASE_NUMBER", r"(?<!\w)(?P<value>(?-i:OB)\s*(?:No\.?\s*)?\d{1,4}/\d{1,2}/\d{2,4})(?!\d)"),
    ]
]


def bert_ner():
    """The BERT NER pipeline, as {"entity_group", "start", "end"} dicts. Loads the model."""
    from transformers import pipeline
    return pipeline("token-classification", model=NER_MODEL, aggregation_strategy="simple",
                    stride=NER_STRIDE_TOKENS, device=torch_device(allow_mps=False))


class GlinerNER:
    """GLiNER with the same output format as bert_ner(), e.g. Anonymizer(ner=GlinerNER())."""

    def __init__(self, model_name: str = GLINER_MODEL, threshold: float = GLINER_THRESHOLD):
        self.model_name = model_name
        self.threshold = threshold

    @cached_property
    def model(self):
        from gliner import GLiNER
        return GLiNER.from_pretrained(self.model_name)

    def __call__(self, text: str) -> list[dict]:
        entities = self.model.predict_entities(text, list(GLINER_LABELS), threshold=self.threshold)
        return [{"entity_group": GLINER_LABELS[entity["label"]], "start": entity["start"], "end": entity["end"]}
                for entity in entities]


class CombinedNER:
    """Every entity any of `models` finds. BERT and GLiNER miss different names, so together they hide more."""

    def __init__(self, *models):
        self.models = models

    def __call__(self, text: str) -> list[dict]:
        return [entity for model in self.models for entity in model(text)]


class Anonymizer:
    def __init__(self, ner=None, public_entities=()):
        """
        ner: an NER function such as bert_ner() (the default) or GlinerNER(); tests can pass a fake.
        public_entities: places/organizations to leave visible. Never build it from evaluation labels.
        """
        self._ner = ner
        self.public_entities = {entity.casefold() for entity in public_entities}

    @cached_property
    def ner(self):
        return self._ner if self._ner is not None else bert_ner()

    def anonymize(self, text: str, known_names=()) -> dict:
        """The text with every span replaced by "[LABEL]", plus the spans themselves."""
        spans = self.detect(text, known_names)
        masked = text
        for span in reversed(spans):  # right to left, so earlier offsets stay valid
            masked = masked[:span["start"]] + f"[{span['label']}]" + masked[span["end"]:]
        return {"text": masked, "redactions": spans}

    def detect(self, text: str, known_names=()) -> list[dict]:
        """Spans to mask as {"start", "end", "label"}. `known_names` (e.g. senders) are always masked."""
        if not text.strip():
            return []
        spans = self._ner_spans(text) + self._pattern_spans(text) + _known_name_spans(text, known_names)
        return _merge_overlapping(spans)

    def _is_public(self, surface_text: str) -> bool:
        return surface_text.casefold() in self.public_entities

    def _ner_spans(self, text: str) -> list[dict]:
        """People are always masked; places and organizations unless they are public."""
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
        for label, pattern in PII_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span("value")
                if not self._is_public(text[start:end]):
                    spans.append(_span(start, end, label))
        return spans


def _span(start: int, end: int, label: str) -> dict:
    return {"start": start, "end": end, "label": label}


def _known_name_spans(text: str, known_names) -> list[dict]:
    return [
        _span(match.start(), match.end(), PERSON_LABEL)
        for name in known_names
        for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)
    ]


def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a span to whole words; BERT can return part of a word."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


def _merge_overlapping(spans: list[dict]) -> list[dict]:
    """Join overlapping spans, so a partly masked name can't leave its end visible."""
    merged: list[dict] = []
    for span in sorted(spans, key=lambda span: (span["start"], -span["end"])):
        if merged and span["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def sender_pseudonyms(messages: list[Message]) -> dict[str, str]:
    """Each real sender -> "Sender 1", "Sender 2", ... in order of first message."""
    pseudonyms: dict[str, str] = {}
    for message in messages:
        if message.sender and not message.is_system:
            pseudonyms.setdefault(message.sender, f"Sender {len(pseudonyms) + 1}")
    return pseudonyms


def names_to_mask(senders) -> set[str]:
    """Each full sender name, plus every part of it long enough to be distinctive."""
    parts = {part for name in senders for part in name.split() if len(part) >= MIN_NAME_PART_LENGTH}
    return set(senders) | parts


def anonymize(messages: list[Message], anonymizer: Anonymizer | None = None) -> None:
    """Fill in `anonymized_text`, `redactions` and `anonymized_sender`.

    Sender names are masked wherever they appear in the text, in case NER misses them.
    """
    anonymizer = anonymizer or Anonymizer()
    pseudonyms = sender_pseudonyms(messages)
    known_names = names_to_mask(pseudonyms)
    for message in messages:
        result = anonymizer.anonymize(message.full_text, known_names=known_names)
        message.anonymized_text = result["text"]
        message.redactions = result["redactions"]
        message.anonymized_sender = pseudonyms.get(message.sender)
