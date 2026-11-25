"""Step 5: extract activity fields (status, type, place, counts) from relevant messages.

    uv run main.py --message "Today we held a youth health session with 20 girls"
    uv run -m src.evaluation.extraction data/whatsapp_labels.json --model flan-t5

Two models fill the same TEMPLATE: NuExtract (the default) and Flan-T5. Pick one with
EXTRACTION_BACKEND in src/config.py. Models load on first use.

Status is read with the previous few messages as context, because chat replies are
often fragments ("I think 9am but I'll double check"). Other fields use the message
alone: on data/whatsapp_labels.json, context dropped activity_type accuracy from 0.74
to 0.35, and counts from earlier messages leaked in.
"""

import difflib
import json
import re
from functools import cached_property
from typing import Any

from word2number import w2n

from .ingest import Message
from .relevance import is_relevant
from .utils import require_anonymized, torch_device

NUEXTRACT_MODEL = "numind/NuExtract-2.0-2B"
NUEXTRACT_MAX_NEW_TOKENS = 128  # the filled-in template is a short JSON object
TEMPLATE_INDENT = 4             # NuExtract's own examples use 4-space JSON

# Template field types. A list means "one of these".
INTEGER = "integer"
VERBATIM_STRING = "verbatim-string"  # must be copied from the text

TEMPLATE = {
    "status": ["completed", "planned", "neither"],
    # The last five types were added for the voice notes in data/dandelion_audio.
    "activity_type": ["family planning outreach", "youth health session", "savings group meeting",
                      "business training", "community health visit", "nutrition talk",
                      "menstrual hygiene session", "gender-based violence dialogue",
                      "cancer screening", "agriculture training", "maternal health session",
                      "health education session", "community outreach", "other"],
    # Keep a useful, source-grounded description even when the category is broad or "other".
    "activity_description": VERBATIM_STRING,
    "location": VERBATIM_STRING,
    "participant_count": INTEGER,
    "referrals": INTEGER,
}
CONTEXT_FIELDS = ("status",)
CONTEXT_MESSAGES = 5

# Each extraction lists under NEEDS_REVIEW the fields a person should check.
NEEDS_REVIEW = "needs_review"
UNPARSEABLE_OUTPUT = "unparseable_output"

FLAN_MODEL = "google/flan-t5-base"
FLAN_MAX_NEW_TOKENS = 16  # answers are one option, a number or a short place name
FLAN_NONE = "none"        # what Flan is told to answer when the message doesn't say
FLAN_NO_ANSWERS = {FLAN_NONE, "unknown", "no", "n/a", "not mentioned"}
FLAN_NULLABLE_CHOICES = ("activity_type",)  # choice fields that also offer "none"; status has "neither"
FLAN_INSTRUCTION = "Read the WhatsApp message and answer the question."
FLAN_QUESTIONS = {
    "status": "Has the activity in this message already happened (completed), is it planned for "
              "the future (planned), or is there no activity (neither)?",
    "activity_type": "Does this message report a community activity? If so, which type? "
                     "Small talk, logistics and follow-up details are \"none\".",
    "activity_description": "What specific community activity is described? Copy a short phrase from "
                           "the message. Answer none if no activity is described.",
    "location": "Where did the activity take place? Copy the place name from the message.",
    "participant_count": "How many people attended or were reached? Answer with a number.",
    "referrals": "How many people were referred to a clinic or need follow-up? This is not the number "
                 "who attended. Answer with a number.",
}
IGNORED_LABEL = -100  # label id PyTorch's cross-entropy skips


class Extractor:
    """Fills TEMPLATE using NuExtract. Subclasses change the model by overriding _load_generator."""

    model_name = NUEXTRACT_MODEL

    def __init__(self, generate=None, template: dict = TEMPLATE):
        """generate: a function from text to the model's raw JSON answer; tests can pass a fake."""
        self._generate = generate
        self.template = template

    @cached_property
    def generate(self):
        return self._generate or self._load_generator()

    def _load_generator(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = torch_device()
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        processor = AutoProcessor.from_pretrained(NUEXTRACT_MODEL, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(NUEXTRACT_MODEL, dtype=dtype).to(device)
        template = json.dumps(self.template, indent=TEMPLATE_INDENT)

        def generate(text: str) -> str:
            prompt = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], template=template, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=NUEXTRACT_MAX_NEW_TOKENS, do_sample=False)
            new_tokens = output[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return generate

    def extract(self, text: str, context: str = "") -> dict:
        """One value (or None) per template field, plus NEEDS_REVIEW: fields whose value isn't
        in the text, and UNPARSEABLE_OUTPUT if the model's answer wasn't JSON.

        context: earlier messages, used only for CONTEXT_FIELDS.
        """
        result = self._fill_template(text)
        if context and text.strip():
            with_context = self._fill_template(f"Earlier messages:\n{context}\n\nMessage:\n{text}")
            for field in CONTEXT_FIELDS:
                result[field] = with_context[field]
            if UNPARSEABLE_OUTPUT in with_context[NEEDS_REVIEW]:
                result[NEEDS_REVIEW].append(UNPARSEABLE_OUTPUT)
        return result

    def _fill_template(self, text: str) -> dict:
        result: dict[str, Any] = {**dict.fromkeys(self.template), NEEDS_REVIEW: []}
        if not text.strip():
            return result
        try:
            answer = json.loads(self.generate(text))
        except json.JSONDecodeError:
            result[NEEDS_REVIEW].append(UNPARSEABLE_OUTPUT)
            return result
        if not isinstance(answer, dict):
            return result
        for field, field_type in self.template.items():
            result[field], needs_review = _validate(answer.get(field), field_type, text)
            if needs_review:
                result[NEEDS_REVIEW].append(field)
        return result


class FlanT5Extractor(Extractor):
    """Fills TEMPLATE using Flan-T5, one question per field.

    For choice fields Flan scores every option and the likeliest wins (free generation
    nearly always picked the same option). The answers are turned into the JSON NuExtract
    would return, so validation and NEEDS_REVIEW work the same.
    """

    model_name = FLAN_MODEL

    def _load_generator(self):
        flan = _FlanModel()

        def generate(text: str) -> str:
            return json.dumps(self._answer_fields(flan, text))

        return generate

    def _answer_fields(self, flan: "_FlanModel", text: str) -> dict:
        fields = [field for field in self.template if field in FLAN_QUESTIONS]
        open_fields = [field for field in fields if not isinstance(self.template[field], list)]
        answers = {}
        for field in fields:
            if field not in open_fields:
                options = _flan_options(field, self.template[field])
                answers[field] = flan.choose(_flan_choice_prompt(text, field, options), options)
        if open_fields:
            answers.update(zip(open_fields, flan.answer([_flan_open_prompt(text, field) for field in open_fields])))
        return {field: _flan_answer_to_value(answers[field], self.template[field], text) for field in fields}


class _FlanModel:
    """Flan-T5, loaded once, either picking one of several options or answering freely."""

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL).to(self.device).eval()

    def choose(self, prompt: str, options: list[str]) -> str:
        """The option Flan finds likeliest as the answer: lowest mean loss per answer token."""
        import torch

        inputs = self.tokenizer([prompt] * len(options), return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
        labels = self.tokenizer(options, return_tensors="pt", padding=True).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = IGNORED_LABEL  # padding isn't part of the answer
        with torch.no_grad():
            logits = self.model(**inputs, labels=labels).logits
        token_losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=IGNORED_LABEL, reduction="none")
        mean_losses = token_losses.sum(dim=1) / (labels != IGNORED_LABEL).sum(dim=1)
        return options[int(mean_losses.argmin())]

    def answer(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=FLAN_MAX_NEW_TOKENS, do_sample=False)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def _flan_options(field: str, choices: list[str]) -> list[str]:
    return ([FLAN_NONE] if field in FLAN_NULLABLE_CHOICES else []) + list(choices)


def _flan_choice_prompt(text: str, field: str, options: list[str]) -> str:
    option_lines = "\n".join(f"- {option}" for option in options)
    return f"{FLAN_INSTRUCTION}\n\nMessage: {text}\n\nQuestion: {FLAN_QUESTIONS[field]}\n\nOPTIONS:\n{option_lines}"


def _flan_open_prompt(text: str, field: str) -> str:
    instruction = f'{FLAN_INSTRUCTION} If the message does not say, answer "{FLAN_NONE}".'
    return f"{instruction}\n\nMessage: {text}\n\nQuestion: {FLAN_QUESTIONS[field]}"


def _flan_answer_to_value(answer: str, field_type: Any, text: str) -> Any:
    """The value NuExtract would have put in its JSON for this answer (None if unsaid)."""
    answer = answer.strip().strip(".").strip()
    if not answer or answer.casefold() in FLAN_NO_ANSWERS:
        return None
    if isinstance(field_type, list):
        return _closest_option(answer, field_type)
    if field_type == INTEGER:
        return _parse_count(answer)
    return answer if answer.casefold() in text.casefold() else None


def _closest_option(answer: str, options: list[str]) -> str | None:
    by_casefold = {option.casefold(): option for option in options}
    if answer.casefold() in by_casefold:
        return by_casefold[answer.casefold()]
    close = difflib.get_close_matches(answer.casefold(), by_casefold, n=1)
    return by_casefold[close[0]] if close else None


def _parse_count(answer: str) -> int | None:
    """"31 women" -> 31, "twenty five" -> 25."""
    digits = re.search(r"\d+", answer)
    if digits:
        return int(digits.group())
    try:
        return w2n.word_to_num(answer)
    except ValueError:
        return None


def _validate(value: Any, field_type: Any, text: str) -> tuple[Any, bool]:
    """(the value if it fits its type, else None; whether a person should check it).

    Numbers and verbatim strings that don't appear in `text` are flagged for review.
    """
    if value is None:
        return None, False
    if isinstance(field_type, list):
        return (value if value in field_type else None), False
    if field_type == INTEGER:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None, False
        return number, not re.search(rf"(?<!\d){number}(?!\d)", text)
    string = str(value)
    return string, field_type == VERBATIM_STRING and string.casefold() not in text.casefold()


EXTRACTORS = {"nuextract": Extractor, "flan-t5": FlanT5Extractor}


def make_extractor(backend: str = "nuextract") -> Extractor:
    """An extractor by name ("nuextract" or "flan-t5")."""
    return EXTRACTORS[backend]()


def context_for(messages: list[Message], index: int, size: int = CONTEXT_MESSAGES) -> str:
    """The anonymized `size` messages before messages[index] in the same conversation."""
    current = messages[index]
    earlier = [message for message in messages[max(0, index - size):index]
               if message.conversation == current.conversation]
    return "\n".join(f"{message.anonymized_sender or 'Unknown'}: {message.anonymized_text}" for message in earlier)


def extract(messages: list[Message], extractor: Extractor | None = None,
            context_size: int = CONTEXT_MESSAGES) -> None:
    """Fill in `extraction` for relevant messages; others stay None.

    context_size: earlier messages shown when reading status; 0 halves the model calls.
    """
    require_anonymized(messages)
    extractor = extractor or make_extractor()
    relevant_indices = [index for index, message in enumerate(messages) if is_relevant(message)]
    for progress, index in enumerate(relevant_indices, start=1):
        print(f"\r  extracting {progress}/{len(relevant_indices)}", end="", flush=True)
        context = context_for(messages, index, context_size)
        messages[index].extraction = extractor.extract(messages[index].anonymized_text, context)
    if relevant_indices:
        print()
