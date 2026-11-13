"""Evaluate relevance classification (facebook/bart-large-mnli) against labeled records.

    uv run -m src.evaluation.classification data/whatsapp_labels.json

Original: Prableen's evaluate.py (commit 6c10379), the classification half; the
anonymization half is anonymization.py. Test data: Risha's data/whatsapp_labels.json,
the only dataset with relevance labels. As in the original, each message is
anonymized first and the masked text is classified. Every change from the original
is marked ADAPTED.
"""
import argparse
import json
from pathlib import Path

# ADAPTED: was `from Anonymization import MessageProcessor` (file since deleted).
from src.anonymize import NER_MODEL, Anonymizer
from src.evaluation.anonymization import load_records, metrics
from src.relevance import CLASSIFIER_MODEL, RelevanceClassifier


def evaluate(records, anonymizer, classifier):
    outputs = []
    relevance = [0, 0, 0, 0]
    labeled = 0
    for i, record in enumerate(records):
        # ADAPTED: was processor.process(text), which anonymized and then classified.
        anonymized = anonymizer.anonymize(record['text'])
        output = {'text': anonymized['text'], 'classification': classifier.classify(anonymized['text'])}
        outputs.append({'id': record.get('id', i), **output})
        gold = record.get('relevance')
        if gold is not None:
            if gold not in ('relevant', 'irrelevant'):
                raise ValueError('relevance must be relevant or irrelevant')
            labeled += 1
            predicted = output['classification']['label'] == 'relevant'
            actual = gold == 'relevant'
            relevance[(0 if predicted else 2) if actual else (1 if predicted else 3)] += 1
    return {'records': outputs, 'metrics': {
        'classification': metrics(*relevance) if labeled else None,
        'classification_labeled_records': labeled}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    # ADAPTED: default was output/baseline.json, shared with anonymization.
    parser.add_argument('--output', type=Path, default=Path('output/evaluations/classification.json'))
    parser.add_argument('--public-entities', type=Path, help='Reviewed JSON list, independent of test annotations')
    # Week 7 experiment: count planned activities as relevant.
    parser.add_argument('--count-plans', action='store_true', help='treat planned activities as relevant')
    # Week 8 experiment: rule out short chat and questions before asking the model.
    parser.add_argument('--min-words', type=int, default=0, help='messages with fewer words are irrelevant (0 = off)')
    parser.add_argument('--skip-questions', action='store_true', help='messages ending in "?" are irrelevant')
    args = parser.parse_args()
    public = json.loads(args.public_entities.read_text()) if args.public_entities else []
    result = evaluate(load_records(args.input), Anonymizer(public_entities=public),
                      RelevanceClassifier(count_plans=args.count_plans, min_words=args.min_words,
                                          skip_questions=args.skip_questions))
    # ADAPTED: metrics written before the per-record outputs.
    report = {'models': {'ner': NER_MODEL, 'classification': CLASSIFIER_MODEL}, 'dataset': str(args.input),
              'metrics': result['metrics'], 'records': result['records']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ADAPTED: encoding='utf-8' added here and in load_records; Windows' default can't handle the labels' emoji.
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
