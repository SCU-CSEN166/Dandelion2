"""Step 5: extract structured activity fields with NuExtract (template-based JSON extraction).

Runs on anonymized text, so private locations come out as "[LOCATION]". Chat messages
are often fragments ("I think 9am but I'll double check"), so status is read with the
previous few messages as context. Everything else comes from the message alone: on
data/whatsapp_labels.json, context made activity_type worse (0.74 -> 0.35 accuracy
with 5 messages) while improving status (0.45 -> 0.58), and it would let numbers from
earlier messages leak into counts. Constructing an Extractor never downloads the
model; it loads on first use.
"""

import json
import re
from functools import cached_property
from typing import Any

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
    extractor = extractor or Extractor()
    relevant_indices = [index for index, message in enumerate(messages) if is_relevant(message)]
    for progress, index in enumerate(relevant_indices, start=1):
        print(f"\r  extracting {progress}/{len(relevant_indices)}", end="", flush=True)  # a few seconds per message
        context = context_for(messages, index, context_size)
        messages[index].extraction = extractor.extract(messages[index].anonymized_text, context)
    if relevant_indices:
        print()
