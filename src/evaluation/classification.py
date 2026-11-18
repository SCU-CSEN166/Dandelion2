"""Evaluate relevance classification against labeled messages.

    uv run -m src.evaluation.classification data/whatsapp_labels.json
    uv run -m src.evaluation.classification data/whatsapp_labels.json --count-plans --min-words 4 --skip-questions

Each message is anonymized first, as in the pipeline, and the masked text is classified.
"Relevant" is the positive class.
"""

import argparse
import json
from pathlib import Path

from src.anonymize import NER_MODEL, Anonymizer
from src.evaluation.anonymization import load_records
from src.evaluation.metrics import binary_counts, scores
from src.relevance import CLASSIFIER_MODEL, IRRELEVANT, RELEVANT, RelevanceClassifier
from src.utils import write_json

DEFAULT_OUTPUT = Path("output/evaluations/classification.json")


def evaluate(records: list[dict], anonymizer: Anonymizer, classifier: RelevanceClassifier) -> dict:
    outputs = []
    decisions = []
    for index, record in enumerate(records):
        anonymized_text = anonymizer.anonymize(record["text"])["text"]
        classification = classifier.classify(anonymized_text)
        outputs.append({"id": record.get("id", index), "text": anonymized_text, "classification": classification})
        label = record.get("relevance")
        if label is not None:
            if label not in (RELEVANT, IRRELEVANT):
                raise ValueError(f"relevance must be {RELEVANT} or {IRRELEVANT}")
            decisions.append((classification["label"] == RELEVANT, label == RELEVANT))
    return {"records": outputs, "metrics": {
        "classification": scores(*binary_counts(decisions)) if decisions else None,
        "classification_labeled_records": len(decisions)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public-entities", type=Path, help="reviewed JSON list of public places to leave visible")
    parser.add_argument("--count-plans", action="store_true", help="treat planned activities as relevant")
    parser.add_argument("--min-words", type=int, default=0, help="messages with fewer words are irrelevant (0 = off)")
    parser.add_argument("--skip-questions", action="store_true", help='messages ending in "?" are irrelevant')
    args = parser.parse_args()

    public = json.loads(args.public_entities.read_text()) if args.public_entities else []
    classifier = RelevanceClassifier(count_plans=args.count_plans, min_words=args.min_words,
                                     skip_questions=args.skip_questions)
    result = evaluate(load_records(args.input), Anonymizer(public_entities=public), classifier)
    report = {"models": {"ner": NER_MODEL, "classification": CLASSIFIER_MODEL}, "dataset": str(args.input),
              "metrics": result["metrics"], "records": result["records"]}
    write_json(args.output, report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
