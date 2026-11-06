"""Evaluate activity-field extraction (numind/NuExtract-2.0-2B) against labeled messages.

    uv run -m src.evaluation.extraction data/whatsapp_labels.json

Not from an earlier commit: written in the same style as anonymization.py and
classification.py. Test data: Risha's data/whatsapp_labels.json, which labels status,
activity_type, location, participant_count and referrals for each message.

Each labeled message's text is extracted on its own: no earlier messages as context
and no anonymization, so this measures the model alone. A value counts as correct
only when it equals the label exactly.

Most labels are null (e.g. 24 of 31 messages have no participant count), so accuracy
alone rewards extracting nothing. Each field is therefore scored like slot filling:
TP = value extracted and correct, FP = value extracted but wrong or not labeled,
FN = labeled value missed or wrong, TN = both null. A wrong value is both an FP and
an FN. Precision says how far extracted values can be trusted; recall says how many
labeled values were found. Accuracy (share of records matching exactly, nulls
included) is kept for reference. One-of-a-list fields (status, activity_type) also
get a confusion matrix: {labeled value: {extracted value: count}}.
"""
import argparse
import json
from pathlib import Path

from src.evaluation.anonymization import metrics
from src.extract import EXTRACTION_MODEL, Extractor

NULL = 'null'  # stands in for None as a confusion-matrix key (JSON keys must be strings)


def slot_counts(pairs):
    """(TP, FP, FN, TN) over (predicted, expected) pairs; see the module docstring."""
    tp = fp = fn = tn = 0
    for predicted, expected in pairs:
        if predicted is not None and predicted == expected:
            tp += 1
        elif predicted is None and expected is None:
            tn += 1
        else:
            fp += predicted is not None
            fn += expected is not None
    return tp, fp, fn, tn


def confusion_matrix(pairs):
    matrix = {}
    for predicted, expected in pairs:
        row = matrix.setdefault(NULL if expected is None else str(expected), {})
        column = NULL if predicted is None else str(predicted)
        row[column] = row.get(column, 0) + 1
    return matrix


def evaluate(records, extractor):
    outputs = []
    fields = list(extractor.template)
    pairs = {field: [] for field in fields}
    for i, record in enumerate(records):
        predicted = extractor.extract(record['text'])
        expected = {field: record[field] for field in fields}
        outputs.append({'id': record.get('message_id', i), 'predicted': predicted, 'expected': expected})
        for field in fields:
            pairs[field].append((predicted[field], expected[field]))
    total = len(records)
    scores = {}
    for field in fields:
        exact = sum(predicted == expected for predicted, expected in pairs[field])
        # metrics()' own accuracy would count a wrong value twice (as FP and FN), so use exact matches.
        scores[field] = {**metrics(*slot_counts(pairs[field])), 'accuracy': exact / total if total else None}
    return {'records': outputs, 'metrics': {
        'records': total,
        'fields': scores,
        'confusion_matrices': {field: confusion_matrix(pairs[field]) for field in fields
                               if isinstance(extractor.template[field], list)}}}


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
