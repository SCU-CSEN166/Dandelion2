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
        # "+254, 733, 905, 217" too: the stress test (src/evaluation/asr_noise.py) found it leaking.
        ("PHONE", r"(?<![\w.])(?P<value>(?:\+?254|0)[\s.,-]*[17](?:[\s.,-]*\d){7,8})(?!\d)"),
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


# A transcript counts as lowercased when under this share of its words start with a capital.
# Only voice-note-length text: short lowercase chats ("lemme double check") are typed, not transcribed.
LOWERCASE_SHARE = 0.05
MIN_WORDS_FOR_CASE_CHECK = 20

# Pattern spans outrank NER spans when they overlap (see Anonymizer.detect).
PATTERN_PRIORITY = 1

# Phone and ID numbers read out as words, with no lead-in phrase (see _spoken_number_spans).
_DIGIT_WORDS = {"oh": "0", "zero": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
                "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_REPEAT_WORDS = {"double": 2, "triple": 3}
_AMBIGUOUS_DIGIT_WORDS = {"oh", "to", "too", "for"}  # also ordinary words, so never the last digit
_SPOKEN_WORD = "|".join(sorted([*_DIGIT_WORDS, *_REPEAT_WORDS], key=len, reverse=True))
_SPOKEN_TOKEN = re.compile(rf"\+?\d+|\b(?:{_SPOKEN_WORD})\b", re.IGNORECASE)
# A run of digit groups and digit words; _spoken_number_spans decides which runs are numbers.
_SPOKEN_RUN = re.compile(
    rf"(?<![\w.+])(?:\+?\d+|(?:{_SPOKEN_WORD})\b)(?:[\s.,-]+(?:\d+|(?:{_SPOKEN_WORD})\b))+", re.IGNORECASE)
_ID_BEFORE = re.compile(r"\bID\s*(?:No\.?|number)?\s*[:#.-]?\s*$", re.IGNORECASE)


# Self-introductions ("it's faith aching", "Beatrice Artino here"). Speech-to-text misspells and
# lowercases local names so NER misses them, but voice notes almost always open this way.
INTRO_WINDOW_CHARS = 80  # only near the start of a message, where people say who they are
MAX_INTRO_NAME_WORDS = 3
_INTRO_BEFORE_NAME = re.compile(r"\b(?:it'?s|it is|this is|my name is)\s+", re.IGNORECASE)
_INTRO_BEFORE_HERE = re.compile(r"(?:^|[,.!:;]\s*)((?:[A-Za-z'-]+\s+){1,4})here\b", re.IGNORECASE)
_TITLES = {"sister", "mama", "mr", "mrs", "ms", "dr", "nurse", "teacher", "madam", "auntie"}
# Words that end an introduction or show it isn't one ("this is the report", "it's really bad").
_NOT_NAME_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "to", "from", "with", "in", "on", "at", "of", "for", "by",
    "about", "is", "was", "are", "were", "be", "been", "not", "no", "yes", "just", "also", "very", "really",
    "too", "now", "still", "again", "already", "here", "there", "today", "tomorrow", "yesterday", "tonight",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "her", "his", "they", "them", "their",
    "it", "its", "this", "that", "these", "those", "what", "who", "how", "why", "when", "where", "which",
    "all", "everyone", "everybody", "team", "ladies", "guys", "group", "good", "great", "fine", "okay", "ok",
    "well", "bad", "sad", "nice", "true", "right", "sure", "done", "late", "back", "important", "going",
    "coming", "doing", "only", "one", "two", "time", "same", "another", "other", "sorry", "happy",
    "reporting", "speaking", "calling", "writing", "sharing", "checking",
}


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
        return {"text": _mask(text, spans), "redactions": spans}

    def detect(self, text: str, known_names=()) -> list[dict]:
        """Spans to mask as {"start", "end", "label"}. `known_names` (e.g. senders) are always masked.

        Where spans overlap, a pattern's label wins over an NER label: GLiNER once called the
        phone number in "call me on 0722 000 917" a DATE_OF_BIRTH.
        """
        if not text.strip():
            return []
        spans = (self._ner_spans(text) + _known_name_spans(text, known_names) + _intro_name_spans(text)
                 + [{**span, "priority": PATTERN_PRIORITY} for span in self._pattern_spans(text)])
        return _merge_overlapping(spans)

    def _is_public(self, surface_text: str) -> bool:
        return surface_text.casefold() in self.public_entities

    @cached_property
    def cased_vocabulary(self) -> set[str]:
        """Words BERT's cased tokenizer knows whole, e.g. "visited" and "Esther" but not "esther"."""
        from transformers import AutoTokenizer
        return set(AutoTokenizer.from_pretrained(NER_MODEL).vocab)

    def _ner_spans(self, text: str) -> list[dict]:
        """People are always masked; places and organizations unless they are public.

        A lowercased transcript is also read a second time with its names recapitalized (see
        truecase), because BERT is case-sensitive: in the speech-to-text stress test, all-lowercase
        text dropped name recall from 1.00 to 0.57.
        """
        entities = list(self.ner(text))
        if looks_lowercased(text):
            # Only names: recapitalized slang and Swahili ("Matatu", "Lemme") looked like places and orgs.
            entities += [entity for entity in self.ner(truecase(text, self.cased_vocabulary))
                         if entity["entity_group"] == PERSON_NER_GROUP]
        spans = []
        for entity in entities:
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
        return spans + _spoken_number_spans(text)


def looks_lowercased(text: str) -> bool:
    """True for transcripts with (almost) no capital letters, like "hello ladies it's rose in jerry"."""
    words = re.findall(r"[A-Za-z]+", text)
    capitalized = sum(word[0].isupper() for word in words)
    return len(words) >= MIN_WORDS_FOR_CASE_CHECK and capitalized / len(words) < LOWERCASE_SHARE


def truecase(text: str, cased_vocabulary: set[str]) -> str:
    """Capitalize each lowercase word BERT doesn't know in lowercase: "esther" -> "Esther", "wanjiku" ->
    "Wanjiku", but "visited" stays. Same length as `text`, so NER offsets still line up."""
    def fix(match: re.Match) -> str:
        word = match.group(0)
        return word if word in cased_vocabulary or len(word) < MIN_NAME_PART_LENGTH else word.capitalize()
    return re.sub(r"\b[a-z][a-z'-]*\b", fix, text)


def _mask(text: str, spans: list[dict]) -> str:
    for span in reversed(spans):  # right to left, so earlier offsets stay valid
        text = text[:span["start"]] + f"[{span['label']}]" + text[span["end"]:]
    return text


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


def _intro_name_spans(text: str) -> list[dict]:
    """The name in a self-introduction near the start: "it's [faith aching]", "[Beatrice Artino] here"."""
    spans = []
    for match in _INTRO_BEFORE_NAME.finditer(text, 0, INTRO_WINDOW_CHARS):
        words = list(re.finditer(r"[A-Za-z'-]+", text[match.end():match.end() + 60]))
        name = _leading_name_words(words, text[match.end():])
        if name:
            spans.append(_span(match.end() + name[0].start(), match.end() + name[-1].end(), PERSON_LABEL))
    for match in _INTRO_BEFORE_HERE.finditer(text[:INTRO_WINDOW_CHARS]):
        start = match.start(1)
        words = _without_titles(list(re.finditer(r"[A-Za-z'-]+", match.group(1))))
        if 0 < len(words) <= MAX_INTRO_NAME_WORDS and all(_could_be_name(word.group(0)) for word in words):
            spans.append(_span(start + words[0].start(), start + words[-1].end(), PERSON_LABEL))
    return spans


def _leading_name_words(words: list[re.Match], following: str) -> list[re.Match]:
    """The first 1-3 words that could be a name, stopping at punctuation or an ordinary word."""
    name = []
    for word in words:
        if name and re.search(r"[,.!?;:]", following[name[-1].end():word.start()]):
            break  # "it's faith aching. I was ..." stops at the full stop
        if len(name) == MAX_INTRO_NAME_WORDS or not _could_be_name(word.group(0)):
            break
        name.append(word)
    return _without_titles(name)


def _without_titles(words: list[re.Match]) -> list[re.Match]:
    """"Sister Nancy": mask the name, keep the title."""
    while words and words[0].group(0).casefold().rstrip(".") in _TITLES:
        words = words[1:]
    return words


def _could_be_name(word: str) -> bool:
    return len(word) >= 2 and word.casefold() not in _NOT_NAME_WORDS


def _spoken_number_spans(text: str) -> list[dict]:
    """Phone and ID numbers read out as words, with no "my number is" before them.

    "zero seven one two three four five six seven eight", "two five four seven one two ...", or a mix
    like "0722 three four five ...". A run is a PHONE only if it spells a Kenyan mobile (07/01 + 7-8
    digits, or 254 + 9), and a NATIONAL_ID only if it has 7-8 digits right after "ID"; so ordinary
    counting ("oh one or two", "one two three four") stays visible.
    """
    spans = []
    for match in _SPOKEN_RUN.finditer(text):
        tokens = list(_SPOKEN_TOKEN.finditer(match.group(0)))
        while tokens and tokens[-1].group(0).casefold() in _AMBIGUOUS_DIGIT_WORDS:
            tokens.pop()  # "... eight to the clinic": the trailing "to" is a word, not a two
        if not tokens:
            continue
        digits = _spelled_digits(token.group(0) for token in tokens)
        start, end = match.start() + tokens[0].start(), match.start() + tokens[-1].end()
        if ((9 <= len(digits) <= 10 and digits[:2] in ("07", "01"))
                or (len(digits) == 12 and digits[:4] in ("2547", "2541"))):
            spans.append(_span(start, end, "PHONE"))
        elif 7 <= len(digits) <= 8 and _ID_BEFORE.search(text[max(0, start - 20):start]):
            spans.append(_span(start, end, "NATIONAL_ID"))
    return spans


def _spelled_digits(tokens) -> str:
    """"zero seven double one" -> "0711"; digit groups are kept as they are ("+254" -> "254")."""
    digits, repeat = "", 1
    for token in (token.casefold().lstrip("+") for token in tokens):
        if token in _REPEAT_WORDS:
            repeat = _REPEAT_WORDS[token]
            continue
        digits += (token if token.isdigit() else _DIGIT_WORDS[token]) * repeat
        repeat = 1
    return digits


def _merge_overlapping(spans: list[dict]) -> list[dict]:
    """Join overlapping spans, so a partly masked name can't leave its end visible.

    The merged span keeps the label of its highest-priority part (patterns beat NER).
    """
    merged: list[dict] = []
    for span in sorted(spans, key=lambda span: (span["start"], -span["end"])):
        if merged and span["start"] < merged[-1]["end"]:
            last = merged[-1]
            last["end"] = max(last["end"], span["end"])
            if span.get("priority", 0) > last.get("priority", 0):
                last["label"], last["priority"] = span["label"], span["priority"]
        else:
            merged.append(dict(span))
    return [_span(span["start"], span["end"], span["label"]) for span in merged]


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


def _reusable_name_parts(masked_names) -> set[str]:
    """Capitalized parts of masked names, to mask again in every message ("Grace Wanjiru" -> "Grace").

    Only capitalized words, so a misheard "Nancy share ono" doesn't hide every "share" in the chat,
    and no titles or ordinary words ("Sister", "Hello").
    """
    return {part for name in masked_names for part in name.split()
            if part[:1].isupper() and len(part) >= MIN_NAME_PART_LENGTH
            and part.casefold() not in _TITLES | _NOT_NAME_WORDS}


def anonymize(messages: list[Message], anonymizer: Anonymizer | None = None) -> None:
    """Fill in `anonymized_text`, `redactions` and `anonymized_sender`.

    Sender names are masked wherever they appear in the text, in case NER misses them. So is every
    name found in any message: "Grace Wanjiru" caught in one voice note hides a lowercase "grace"
    in the next, which NER alone missed on the hidden test.
    """
    anonymizer = anonymizer or Anonymizer()
    pseudonyms = sender_pseudonyms(messages)
    sender_names = names_to_mask(pseudonyms)
    spans_by_message = [anonymizer.detect(message.full_text, sender_names) for message in messages]
    found_names = _reusable_name_parts(message.full_text[span["start"]:span["end"]]
                                       for message, spans in zip(messages, spans_by_message)
                                       for span in spans if span["label"] == PERSON_LABEL)
    for message, spans in zip(messages, spans_by_message):
        spans = _merge_overlapping(spans + _known_name_spans(message.full_text, found_names))
        message.anonymized_text = _mask(message.full_text, spans)
        message.redactions = spans
        message.anonymized_sender = pseudonyms.get(message.sender)
