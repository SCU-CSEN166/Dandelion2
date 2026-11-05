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

from .ingest import Message

EXTRACTION_MODEL = "numind/NuExtract-2.0-2B"

# NuExtract template: a list means "one of these", other values are types.
TEMPLATE = {
    "status": ["completed", "planned", "neither"],
    "activity_type": ["family planning outreach", "youth health session", "savings group meeting",
                      "business training", "community health visit", "other"],
    "location": "verbatim-string",
    "participant_count": "integer",
    "referrals": "integer",
}
CONTEXT_FIELDS = ("status",)  # read with earlier messages as context
CONTEXT_MESSAGES = 5


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

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        processor = AutoProcessor.from_pretrained(EXTRACTION_MODEL, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(EXTRACTION_MODEL, dtype=dtype).to(device)
        template = json.dumps(self.template, indent=4)

        def generate(text: str) -> str:
            prompt = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], template=template, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

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
            if "unparseable_output" in with_context["needs_review"]:
                result["needs_review"].append("unparseable_output")
        return result

    def _extract(self, text: str) -> dict:
        result = dict.fromkeys(self.template)
        result["needs_review"] = []
        if not text.strip():
            return result
        try:
            raw = json.loads(self.generate(text))
        except json.JSONDecodeError:
            result["needs_review"].append("unparseable_output")
            return result

        for field, kind in self.template.items():
            value = raw.get(field) if isinstance(raw, dict) else None
            if value is None:
                continue
            if isinstance(kind, list):
                result[field] = value if value in kind else None
            elif kind == "integer":
                try:
                    result[field] = int(value)
                except (TypeError, ValueError):
                    continue
                if not re.search(rf"(?<!\d){result[field]}(?!\d)", text):
                    result["needs_review"].append(field)
            else:
                result[field] = str(value)
                if kind == "verbatim-string" and result[field].casefold() not in text.casefold():
                    result["needs_review"].append(field)
        return result


def context_for(messages: list[Message], i: int, size: int = CONTEXT_MESSAGES) -> str:
    """The `size` messages before messages[i] in the same conversation, anonymized."""
    earlier = [p for p in messages[max(0, i - size):i] if p.conversation == messages[i].conversation]
    return "\n".join(f"{p.anonymized_sender or 'Unknown'}: {p.anonymized_text}" for p in earlier)


def extract(messages: list[Message], extractor: Extractor | None = None,
            context_size: int = CONTEXT_MESSAGES) -> None:
    """Fill in `extraction` for messages classified relevant; others are left as None.

    context_size: earlier messages shown when reading status; 0 halves the model calls.
    """
    extractor = extractor or Extractor()
    for m in messages:
        if m.anonymized_text is None:
            raise ValueError(f"{m.id} has not been anonymized; run anonymize() first")
    targets = [i for i, m in enumerate(messages) if m.relevance and m.relevance["label"] == "relevant"]
    for n, i in enumerate(targets, start=1):
        print(f"\r  extracting {n}/{len(targets)}", end="", flush=True)  # a few seconds per message
        context = context_for(messages, i, context_size) if context_size else ""
        messages[i].extraction = extractor.extract(messages[i].anonymized_text, context)
    if targets:
        print()
