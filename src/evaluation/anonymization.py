"""Evaluate anonymization (dslim/bert-base-NER + regex) against annotated PII.

    uv run -m src.evaluation.anonymization data/synthetic_unanonymized.json data/whatsapp_labels.json

Original: Prableen's evaluate.py (commit 6c10379), the anonymization half; the
classification half is classification.py. Test data: the synthetic records, and
Risha's data/whatsapp_labels.json. Every change from the original is marked ADAPTED.

Decision units are annotated entities:
TP = private entity fully covered, FN = private entity not fully covered,
FP = safe entity touched by a redaction, TN = safe entity untouched.
"""
import argparse
import json
from pathlib import Path

# ADAPTED: was `from Anonymization import MessageProcessor` (file since deleted).
from src.anonymize import GLINER_MODEL, NER_MODEL, Anonymizer, CombinedNER, GlinerNER


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
    full = total = preserved = safe = extra = 0
    for i, record in enumerate(records):
        text = record['text']  # Gold offsets refer to text, never translation_en.
        output = processor.anonymize(text)  # ADAPTED: was processor.process(text), which also classified
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
    return {'records': outputs, 'metrics': {
        'anonymization_annotated_entities': metrics(*pii),
        'fully_redacted_pii': full, 'total_pii': total,
        'preserved_safe_entities': preserved, 'total_safe_entities': safe,
        'redactions_outside_annotations': extra}}


# ADAPTED: Risha's data/whatsapp_labels.json lists PII by value rather than by offset,
# so it is converted to the records format above (conversion from commit 0b20a53).
def load_records(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    records = data['records'] if isinstance(data, dict) else data
    return [from_whatsapp_label(r) if 'message_id' in r else r for r in records]


def from_whatsapp_label(label):
    """Locate each labeled PII value in the text. A labeled location that isn't PII
    (e.g. "Nakuru") is a should-not-redact entity.
    """
    text = label['text']
    pii_values = {p['value'] for p in label['pii']}
    entities = [e for value in pii_values for e in _occurrences(text, value, should_redact=True)]
    location = label.get('location')
    if location and location not in pii_values:
        entities += _occurrences(text, location, should_redact=False, required=False)
    return {'id': label['message_id'], 'text': text, 'entities': entities, 'relevance': label['relevance']}


def _occurrences(text, value, should_redact, required=True):
    found, start = [], text.find(value)
    while start != -1:
        found.append({'text': value, 'start': start, 'end': start + len(value), 'should_redact': should_redact})
        start = text.find(value, start + 1)
    if required and not found:
        raise ValueError(f'labeled value {value!r} not found in {text!r}')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')  # ADAPTED: was one `input`; now every dataset in one report
    # ADAPTED: default was output/baseline.json, shared with classification.
    parser.add_argument('--output', type=Path, default=Path('output/evaluations/anonymization.json'))
    parser.add_argument('--public-entities', type=Path, help='Reviewed JSON list, independent of test annotations')
    # Week 7 experiment: swap BERT for GLiNER, or use both together.
    parser.add_argument('--ner', choices=['bert', 'gliner', 'both'], default='bert', help='which NER model(s) to use')
    args = parser.parse_args()
    public = json.loads(args.public_entities.read_text()) if args.public_entities else []
    if args.ner == 'gliner':
        ner = GlinerNER()
    elif args.ner == 'both':
        ner = CombinedNER(Anonymizer().ner, GlinerNER())  # Anonymizer().ner is the default BERT pipeline
    else:
        ner = None  # Anonymizer loads BERT itself
    anonymizer = Anonymizer(ner=ner, public_entities=public)
    # ADAPTED: was json.loads(args.input.read_text()) on a records file only; see load_records.
    reports = {str(path): evaluate(load_records(path), anonymizer) for path in args.inputs}
    # ADAPTED: one file for all datasets, metrics first; classification model moved to classification.py.
    ner_names = {'bert': NER_MODEL, 'gliner': GLINER_MODEL, 'both': f'{NER_MODEL} + {GLINER_MODEL}'}
    report = {'models': {'ner': ner_names[args.ner]},
              'metrics': {dataset: result['metrics'] for dataset, result in reports.items()},
              'records': {dataset: result['records'] for dataset, result in reports.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ADAPTED: encoding='utf-8' added here and in load_records; Windows' default can't handle the labels' emoji.
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
