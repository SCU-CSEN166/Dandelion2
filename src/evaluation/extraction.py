"""Evaluate activity-field extraction (numind/NuExtract-2.0-2B) against labeled messages.

    uv run -m src.evaluation.extraction data/whatsapp_labels.json

Not from an earlier commit: written in the same style as anonymization.py and
classification.py. Test data: Risha's data/whatsapp_labels.json, which labels status,
activity_type, location, participant_count and referrals for each message.

Each labeled message's text is extracted on its own: no earlier messages as context
and no anonymization, so this measures the model alone. A field counts as correct
only when the extracted value equals the label exactly (null must match null).
"""
import argparse
import json
from pathlib import Path

from src.extract import EXTRACTION_MODEL, Extractor


def evaluate(records, extractor):
    outputs = []
    fields = list(extractor.template)
    correct = dict.fromkeys(fields, 0)
    for i, record in enumerate(records):
        predicted = extractor.extract(record['text'])
        expected = {field: record[field] for field in fields}
        outputs.append({'id': record.get('message_id', i), 'predicted': predicted, 'expected': expected})
        for field in fields:
            correct[field] += predicted[field] == expected[field]
    total = len(records)
    return {'records': outputs, 'metrics': {
        'records': total,
        'accuracy': {field: correct[field] / total if total else None for field in fields}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/evaluations/extraction.json'))
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding='utf-8'))
    result = evaluate(records, Extractor())
    report = {'models': {'extraction': EXTRACTION_MODEL}, 'dataset': str(args.input),
              'metrics': result['metrics'], 'records': result['records']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
