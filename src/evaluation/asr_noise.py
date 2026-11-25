"""Stress-test anonymization on text garbled the way speech-to-text garbles it.

    uv run -m src.evaluation.asr_noise data/synthetic_unanonymized.json

The synthetic records are clean typed text, but most real Dandelion reports are voice notes. On the
hidden test, Whisper lowercased whole messages, misspelled local names ("Faith Achieng" -> "faith
aching") and wrote phone numbers with commas. This rewrites every record in each of those styles,
moves the annotated entities with the text, and scores anonymization on each version, so we can say
how much privacy protection survives a noisy transcript, and which kind of noise hurts most.

The noise is deterministic (fixed seed), so reruns give the same numbers.
"""

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.anonymize import Anonymizer, CombinedNER, GlinerNER, bert_ner
from src.config import PUBLIC_ENTITIES
from src.evaluation.anonymization import _outcome, load_records
from src.evaluation.metrics import scores
from src.utils import write_json

DEFAULT_OUTPUT = Path("output/evaluations/anonymization_asr_noise.json")
SEED = 166
DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
NUMBER_LABELS = {"PHONE", "NATIONAL_ID"}
NAME_LABELS = {"PERSON"}
_PUNCTUATION = re.compile(r"[.,;:!?()\"]")


def _lowercase_no_punctuation(text: str, label: str | None, rng: random.Random) -> str:
    """Whisper on fast, informal speech: no capitals, no punctuation (hidden-test msg03)."""
    return _PUNCTUATION.sub("", text.lower()) if label is None else text.lower()


def _spoken_digits(text: str, label: str | None, rng: random.Random) -> str:
    """Numbers written out as words: "0721 448 903" -> "zero seven two one four four ..."."""
    if label not in NUMBER_LABELS:
        return text
    return " ".join(DIGIT_WORDS[int(digit)] for digit in re.findall(r"\d", text))


def _comma_digits(text: str, label: str | None, rng: random.Random) -> str:
    """Digits read slowly: "0721 448 903" -> "0721, 448, 903" (hidden-test msg02 before our fix)."""
    if label not in NUMBER_LABELS:
        return text
    return re.sub(r"[\s.-]+", ", ", text.strip())


def _misspelled_names(text: str, label: str | None, rng: random.Random) -> str:
    """Local names misheard: lowercase, plus one letter swapped, dropped or split per word."""
    if label not in NAME_LABELS:
        return text
    return " ".join(_mishear(word, rng) for word in text.split())


def _mishear(word: str, rng: random.Random) -> str:
    word = word.lower()
    if len(word) < 4:
        return word
    position = rng.randrange(1, len(word) - 1)
    edit = rng.choice(("swap_vowel", "drop", "split"))
    if edit == "swap_vowel" and word[position] in "aeiou":
        return word[:position] + rng.choice([v for v in "aeiou" if v != word[position]]) + word[position + 1:]
    if edit == "split":
        return word[:position] + " " + word[position:]
    return word[:position] + word[position + 1:]


NOISES = {
    "lowercase_no_punctuation": [_lowercase_no_punctuation],
    "spoken_digits": [_spoken_digits],
    "comma_digits": [_comma_digits],
    "misspelled_names": [_misspelled_names],
    "all_combined": [_misspelled_names, _spoken_digits, _lowercase_no_punctuation],
}


def add_noise(record: dict, steps, rng: random.Random) -> dict:
    """`record` with every segment rewritten by `steps`; entity offsets follow the new text."""
    text, entities = record["text"], sorted(record.get("entities", []), key=lambda entity: entity["start"])
    pieces, moved, cursor = [], [], 0
    for entity in entities:
        if entity["start"] < cursor:
            continue  # overlapping annotations: keep the first
        pieces.append(_apply(text[cursor:entity["start"]], None, steps, rng))
        start = sum(map(len, pieces))
        new_text = _apply(entity["text"], entity["label"], steps, rng)
        pieces.append(new_text)
        moved.append({**entity, "text": new_text, "start": start, "end": start + len(new_text)})
        cursor = entity["end"]
    pieces.append(_apply(text[cursor:], None, steps, rng))
    return {**record, "text": "".join(pieces), "entities": moved}


def _apply(text: str, label: str | None, steps, rng: random.Random) -> str:
    for step in steps:
        text = step(text, label, rng)
    return text


def score(records: list[dict], anonymizer: Anonymizer) -> dict:
    outcomes, by_label = Counter(), defaultdict(Counter)
    examples = []
    for record in records:
        masks = anonymizer.anonymize(record["text"])["redactions"]
        for entity in record["entities"]:
            outcome = _outcome(entity, masks)
            outcomes[outcome] += 1
            if entity["should_redact"]:
                by_label[entity["label"]][outcome] += 1
                if outcome == "FN" and len(examples) < 10:
                    examples.append({"id": record["id"], "label": entity["label"], "leaked": entity["text"]})
    recall_by_label = {label: round(counts["TP"] / (counts["TP"] + counts["FN"]), 3)
                       for label, counts in sorted(by_label.items())}
    metrics = scores(outcomes["TP"], outcomes["FP"], outcomes["FN"], outcomes["TN"])
    return {"metrics": {key: round(value, 3) if isinstance(value, float) else value for key, value in metrics.items()},
            "pii_recall_by_label": recall_by_label, "leaked_examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/synthetic_unanonymized.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_records(args.input)
    anonymizer = Anonymizer(ner=CombinedNER(bert_ner(), GlinerNER()), public_entities=PUBLIC_ENTITIES)
    results = {"clean": score(records, anonymizer)}
    for name, steps in NOISES.items():
        rng = random.Random(SEED)
        results[name] = score([add_noise(record, steps, rng) for record in records], anonymizer)
        results[name]["example_text"] = add_noise(records[0], steps, random.Random(SEED))["text"]

    write_json(args.output, {"dataset": str(args.input), "seed": SEED, "results": results})
    print(f"{'version':26} {'recall':>7} {'precision':>9} {'f1':>6}")
    for name, result in results.items():
        metrics = result["metrics"]
        print(f"{name:26} {metrics['recall']:7.3f} {metrics['precision']:9.3f} {metrics['f1']:6.3f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
