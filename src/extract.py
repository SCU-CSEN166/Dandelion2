"""Step 5: extract structured activity fields with NuExtract (template-based JSON extraction).

FlanT5Extractor is an alternative backed by google/flan-t5-base (one question per field);
pick it with EXTRACTION_BACKEND in src/config.py, or compare both with src.evaluation.extraction.

Runs on anonymized text, so private locations come out as "[LOCATION]". Chat messages
are often fragments ("I think 9am but I'll double check"), so status is read with the
previous few messages as context. Everything else comes from the message alone: on
data/whatsapp_labels.json, context made activity_type worse (0.74 -> 0.35 accuracy
with 5 messages) while improving status (0.45 -> 0.58), and it would let numbers from
earlier messages leak into counts. Constructing an Extractor never downloads the
model; it loads on first use.
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

EXTRACTION_MODEL = "numind/NuExtract-2.0-2B"
MAX_NEW_TOKENS = 128  # the filled-in template is a short JSON object
TEMPLATE_INDENT = 4   # NuExtract's examples use 4-space template JSON

# NuExtract field types. A list in the template means "one of these".
INTEGER = "integer"
VERBATIM_STRING = "verbatim-string"  # must be copied from the text

TEMPLATE = {
    "status": ["completed", "planned", "neither"],
    # Dandelion Africa's program areas. The last five were added for the real voice notes
    # (data/dandelion_audio), which were otherwise mostly "other".
    "activity_type": ["family planning outreach", "youth health session", "savings group meeting",
                      "business training", "community health visit", "nutrition talk",
                      "menstrual hygiene session", "gender-based violence dialogue",
                      "cancer screening", "agriculture training", "other"],
    "location": VERBATIM_STRING,
    "participant_count": INTEGER,
    "referrals": INTEGER,
}
CONTEXT_FIELDS = ("status",)  # read with earlier messages as context
CONTEXT_MESSAGES = 5

# Key in each extraction listing what a human should check, and the flag for bad model output.
NEEDS_REVIEW = "needs_review"
UNPARSEABLE_OUTPUT = "unparseable_output"


class Extractor:
    model_name = EXTRACTION_MODEL

    def __init__(self, generate=None, template: dict = TEMPLATE):
        """generate: a function from text to the model's raw JSON output (tests inject a fake one)."""
        self._generate = generate
        self.template = template

    @cached_property
    def generate(self):
        if self._generate is not None:
            return self._generate
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = torch_device()
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        processor = AutoProcessor.from_pretrained(EXTRACTION_MODEL, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(EXTRACTION_MODEL, dtype=dtype).to(device)
        template = json.dumps(self.template, indent=TEMPLATE_INDENT)

        def generate(text: str) -> str:
            prompt = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], template=template, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return generate

    def extract(self, text: str, context: str = "") -> dict:
        """Return one value (or None) per template field, plus `needs_review`: fields whose
        value doesn't appear in the text, or a model error, for a human to check.

        context: earlier messages, used only for CONTEXT_FIELDS.
        """
        result = self._extract(text)
        if context and text.strip():
            with_context = self._extract(f"Earlier messages:\n{context}\n\nMessage:\n{text}")
            for field in CONTEXT_FIELDS:
                result[field] = with_context[field]
            if UNPARSEABLE_OUTPUT in with_context[NEEDS_REVIEW]:
                result[NEEDS_REVIEW].append(UNPARSEABLE_OUTPUT)
        return result

    def _extract(self, text: str) -> dict:
        result: dict[str, Any] = dict.fromkeys(self.template)
        result[NEEDS_REVIEW] = []
        if not text.strip():
            return result
        try:
            raw = json.loads(self.generate(text))
        except json.JSONDecodeError:
            result[NEEDS_REVIEW].append(UNPARSEABLE_OUTPUT)
            return result
        if not isinstance(raw, dict):
            return result

        for field, field_type in self.template.items():
            value, needs_review = _coerce(raw.get(field), field_type, text)
            result[field] = value
            if needs_review:
                result[NEEDS_REVIEW].append(field)
        return result


FLAN_MODEL = "google/flan-t5-base"
FLAN_MAX_NEW_TOKENS = 16  # each answer is one option, a number or a short place name
FLAN_NONE = "none"        # the answer Flan is told to give when the message doesn't say
FLAN_NULLABLE_CHOICES = ("activity_type",)  # list fields that also offer "none" (status has "neither")

# One question per template field. Flan-T5 answers questions rather than filling a JSON
# template, so the answers are assembled into the same JSON NuExtract would return.
FLAN_QUESTIONS = {
    "status": "Has the activity in this message already happened (completed), is it planned for "
              "the future (planned), or is there no activity (neither)?",
    "activity_type": "Does this message report a community activity? If so, which type? "
                     "Small talk, logistics and follow-up details are \"none\".",
    "location": "Where did the activity take place? Copy the place name from the message.",
    "participant_count": "How many people attended or were reached? Answer with a number.",
    "referrals": "How many people were referred to a clinic or need follow-up? This is not the number "
                 "who attended. Answer with a number.",
}


class FlanT5Extractor(Extractor):
    """Extractor backed by google/flan-t5-base: one question per field, asked in one batch.

    One-of-a-list fields are answered by scoring each option's likelihood under Flan's
    "OPTIONS:" prompt format (free generation nearly always picked the same option);
    activity_type also offers "none" so a message can have no activity, as NuExtract's null
    allows. Free-text answers are kept only if they appear in the message, matching
    NuExtract's verbatim-string type.
    Results go through the same validation as NuExtract's, so needs_review works the same.
    """

    model_name = FLAN_MODEL

    @cached_property
    def generate(self):
        if self._generate is not None:
            return self._generate
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = torch_device()
        tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL).to(device).eval()
        fields = [field for field in self.template if field in FLAN_QUESTIONS]

        def score(prompt: str, options: list[str]) -> str:
            """The option Flan finds most likely as the answer (mean log-likelihood per token)."""
            inputs = tokenizer([prompt] * len(options), return_tensors="pt", padding=True,
                               truncation=True).to(device)
            labels = tokenizer(options, return_tensors="pt", padding=True).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100  # padding isn't part of the answer
            with torch.no_grad():
                logits = model(**inputs, labels=labels).logits
            token_losses = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
            mean_losses = token_losses.sum(dim=1) / (labels != -100).sum(dim=1)
            return options[int(mean_losses.argmin())]

        def generate(text: str) -> str:
            answers = {}
            open_fields = [field for field in fields if not isinstance(self.template[field], list)]
            for field in fields:
                if isinstance(self.template[field], list):
                    options = _flan_options(field, self.template[field])
                    answers[field] = score(_flan_prompt(text, field, options), options)
            prompts = [_flan_prompt(text, field) for field in open_fields]
            if prompts:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                output = model.generate(**inputs, max_new_tokens=FLAN_MAX_NEW_TOKENS, do_sample=False)
                answers.update(zip(open_fields, tokenizer.batch_decode(output, skip_special_tokens=True)))
            return json.dumps({field: _flan_value(answers[field], self.template[field], text)
                               for field in fields})

        return generate


def _flan_options(field: str, field_type: list[str]) -> list[str]:
    """The answer choices for a one-of-a-list field, plus "none" where the label may be empty."""
    return ([FLAN_NONE] if field in FLAN_NULLABLE_CHOICES else []) + list(field_type)


def _flan_prompt(text: str, field: str, options: list[str] | None = None) -> str:
    if options:
        prompt = "Read the WhatsApp message and answer the question."
    else:
        prompt = f'Read the WhatsApp message and answer the question. If the message does not say, answer "{FLAN_NONE}".'
    prompt += f"\n\nMessage: {text}\n\nQuestion: {FLAN_QUESTIONS[field]}"
    if options:
        prompt += "\n\nOPTIONS:\n" + "\n".join(f"- {option}" for option in options)
    return prompt


def _flan_value(answer: str, field_type: Any, text: str) -> Any:
    """Turn one Flan answer into the value NuExtract would have put in its JSON (None if unsaid)."""
    answer = answer.strip().strip(".").strip()
    if not answer or answer.casefold() in (FLAN_NONE, "unknown", "no", "n/a", "not mentioned"):
        return None
    if isinstance(field_type, list):
        options = {option.casefold(): option for option in field_type}
        if answer.casefold() in options:
            return options[answer.casefold()]
        close = difflib.get_close_matches(answer.casefold(), options, n=1)
        return options[close[0]] if close else None
    if field_type == INTEGER:
        match = re.search(r"\d+", answer)
        if match:
            return int(match.group())
        try:
            return w2n.word_to_num(answer)
        except ValueError:
            return None
    return answer if answer.casefold() in text.casefold() else None


def _coerce(value: Any, field_type: Any, text: str) -> tuple[Any, bool]:
    """Validate one raw model value against its template type.

    Returns (value or None, whether a human should check it because it isn't in `text`).
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
    """An extractor by name ("nuextract" or "flan-t5"); neither downloads its model until first use."""
    return EXTRACTORS[backend]()


def context_for(messages: list[Message], index: int, size: int = CONTEXT_MESSAGES) -> str:
    """The `size` messages before messages[index] in the same conversation, anonymized ("" if size is 0)."""
    current = messages[index]
    earlier = [message for message in messages[max(0, index - size):index]
               if message.conversation == current.conversation]
    return "\n".join(f"{message.anonymized_sender or 'Unknown'}: {message.anonymized_text}" for message in earlier)


def extract(messages: list[Message], extractor: Extractor | None = None,
            context_size: int = CONTEXT_MESSAGES) -> None:
    """Fill in `extraction` for messages classified relevant; others are left as None.

    context_size: earlier messages shown when reading status; 0 halves the model calls.
    """
    require_anonymized(messages)
    extractor = extractor or make_extractor()
    relevant_indices = [index for index, message in enumerate(messages) if is_relevant(message)]
    for progress, index in enumerate(relevant_indices, start=1):
        print(f"\r  extracting {progress}/{len(relevant_indices)}", end="", flush=True)  # a few seconds per message
        context = context_for(messages, index, context_size)
        messages[index].extraction = extractor.extract(messages[index].anonymized_text, context)
    if relevant_indices:
        print()
