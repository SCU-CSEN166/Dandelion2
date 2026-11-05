"""Evaluate original-text annotations and export sanitized model outputs."""
import argparse
import json
from pathlib import Path
from Anonymization import MessageProcessor


def metrics(tp, fp, fn, tn):
    def div(a, b):
        return a / b if b else None
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'accuracy': div(tp + tn, tp + fp + fn + tn),
            'precision': div(tp, tp + fp), 'recall': div(tp, tp + fn),
            'f1': div(2 * tp, 2 * tp + fp + fn)}


def evaluate(records, processor):
    outputs = []
    pii = [0, 0, 0, 0]
    relevance = [0, 0, 0, 0]
    full = total = preserved = safe = extra = 0
    labeled = 0
    for i, record in enumerate(records):
        text = record['text']  # Gold offsets refer to text, never translation_en.
        output = processor.process(text)
        outputs.append({'id': record.get('id', i), **output})
        spans = output['redactions']
        for entity in record.get('entities', []):
            a, b = entity['start'], entity['end']
            if not 0 <= a < b <= len(text) or text[a:b] != entity['text']:
                raise ValueError('Invalid gold offset in record %s' % record.get('id', i))
            overlap = any(s['start'] < b and a < s['end'] for s in spans)
            covered = all(any(s['start'] <= pos < s['end'] for s in spans) for pos in range(a, b))
            redact = entity['should_redact']
            # Positive requires complete coverage; partial PII masking is a miss.
            index = (0 if covered else 2) if redact else (1 if overlap else 3)
            pii[index] += 1
            if redact:
                total += 1
                full += covered
            else:
                safe += 1
                preserved += not overlap
        extra += sum(not any(s['start'] < e['end'] and e['start'] < s['end']
                             for e in record.get('entities', [])) for s in spans)
        gold = record.get('relevance')
        if gold is not None:
            if gold not in ('relevant', 'irrelevant'):
                raise ValueError('relevance must be relevant or irrelevant')
            labeled += 1
            predicted = output['classification']['label'] == 'relevant'
            actual = gold == 'relevant'
            relevance[(0 if predicted else 2) if actual else (1 if predicted else 3)] += 1
    return {'records': outputs, 'metrics': {
        'anonymization_annotated_entities': metrics(*pii),
        'fully_redacted_pii': full, 'total_pii': total,
        'preserved_safe_entities': preserved, 'total_safe_entities': safe,
        'redactions_outside_annotations': extra,
        'classification': metrics(*relevance) if labeled else None,
        'classification_labeled_records': labeled}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/baseline.json'))
    parser.add_argument('--public-entities', type=Path, help='Reviewed JSON list, independent of test annotations')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    public = json.loads(args.public_entities.read_text()) if args.public_entities else []
    report = evaluate(data['records'] if isinstance(data, dict) else data,
                      MessageProcessor(public_entities=public))
    report['models'] = {'ner': 'dslim/bert-base-NER', 'classification': 'facebook/bart-large-mnli'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
