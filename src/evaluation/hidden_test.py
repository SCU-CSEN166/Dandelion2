"""Evaluate against the hidden test solution (data/dandelion_audio/hidden_test_outputs.csv).

    uv run -m src.evaluation.hidden_test                                # score the pipeline's CSV
    uv run -m src.evaluation.hidden_test --models nuextract flan-t5    # also compare extractors

The solution labels the 10 voice notes in data/dandelion_audio with relevance ("Event" or
"Chit-chat"), activity_type, participants, duplicate_of, the PII items spoken, and the
expected anonymized text and transcript.

1) Pipeline: scores a CSV written by main.py (default output/HIDDEN_TEST_OUTPUT.csv, from
   `uv run main.py --folder data/dandelion_audio`), so every step's errors count:
   - relevance: Event = positive, as in classification.py.
   - activity_type: the solution's free-text labels are mapped by hand to our template's
     categories (ACTIVITY_LABELS), then scored as slot filling like extraction.py.
   - participants and duplicate_of: exact match, slot filling.
   - PII: an item leaks if it (or a near spelling, since speech-to-text changes names, e.g.
     "Faith Achieng" -> "faith aching") is still in anonymized_text. Each item is a TP if
     hidden and an FN if leaked, so recall is the share of PII hidden.
2) Extractors (--models): runs each extraction model on two texts, scoring activity_type
   and participant_count like extraction.py (every message is extracted, nulls expected
   for chit-chat):
   - "solution": the expected anonymized text, i.e. the model alone, with no speech-to-text
     or anonymization errors.
   - "pipeline": the anonymized text in the pipeline's CSV, i.e. what the model sees in
     the real pipeline.

Caveat: 10 messages, and the last five activity_type categories in src/extract.py were
added after looking at these voice notes, so this set is not blind for activity_type.
"""
import argparse
import csv
import difflib
import json
import re
from pathlib import Path

from src.evaluation.anonymization import metrics
from src.evaluation.extraction import confusion_matrix, slot_counts
from src.extract import EXTRACTORS, TEMPLATE, make_extractor

SOLUTION = Path('data/dandelion_audio/hidden_test_outputs.csv')
PIPELINE_OUTPUT = Path('output/HIDDEN_TEST_OUTPUT.csv')
OUTPUT = Path('output/evaluations/hidden_test.json')

EVENT = 'Event'  # the solution's relevant label (the other is "Chit-chat"); main.py writes the same
CSV_NONE = 'none'
# Solution label -> our template category (src/extract.py TEMPLATE), reviewed by hand.
ACTIVITY_LABELS = {
    'Family planning outreach': 'family planning outreach',
    "Women's group table banking (savings) meeting": 'savings group meeting',
    'Menstrual hygiene session and sanitary pad distribution': 'menstrual hygiene session',
    'Gender-based violence awareness dialogue': 'gender-based violence dialogue',
    'Cervical cancer screening camp': 'cancer screening',
    'Kitchen garden training': 'agriculture training',
}
LEAK_SIMILARITY = 0.8  # how close a span of the output must be to a PII item to count as leaked
EXTRACTION_FIELDS = {'activity_type': 'activity_type', 'participant_count': 'participants'}


def load_csv(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def value(cell):
    """A CSV cell as a value: '' and 'none' are None, whole numbers are ints."""
    cell = (cell or '').strip()
    if not cell or cell.casefold() == CSV_NONE:
        return None
    return int(cell) if cell.isdigit() else cell


def activity(cell):
    """A solution or pipeline activity_type as a template category (None if empty)."""
    label = value(cell)
    if label is None:
        return None
    category = ACTIVITY_LABELS.get(label, label.casefold())
    if category not in TEMPLATE['activity_type']:
        raise ValueError(f'no template category for activity_type {label!r}; add it to ACTIVITY_LABELS')
    return category


def pii_items(cell):
    """'Lucy Kiprop (also 'Lucy'); Tumaini (women's group name)' -> ['Lucy Kiprop', 'Lucy', 'Tumaini']."""
    items = []
    for item in cell.split(';'):
        items += re.findall(r"also '([^']+)'", item)
        item = re.sub(r'\(.*?\)', '', item).strip()
        if item:
            items.append(item)
    return items


def _words(text):
    return re.findall(r'[a-z0-9]+', text.casefold())


def leaked(item, text):
    """Whether `item` (or a near spelling of it) is still in `text`. Spaces are ignored, so
    "0711000482" matches "0711 000 482"; phone numbers must match exactly."""
    target = ''.join(_words(item))
    words = _words(text)
    size = len(_words(item))
    for length in range(max(1, size - 1), size + 2):
        for start in range(len(words) - length + 1):
            window = ''.join(words[start:start + length])
            if window == target:
                return True
            if not target.isdigit() and difflib.SequenceMatcher(None, window, target).ratio() >= LEAK_SIMILARITY:
                return True
    return False


def score_slots(pairs):
    exact = sum(predicted == expected for predicted, expected in pairs)
    return {**metrics(*slot_counts(pairs)), 'accuracy': exact / len(pairs) if pairs else None}


def evaluate_pipeline(solution, rows):
    by_id = {str(row['Message_id']): row for row in rows}
    missing = [record['message_id'] for record in solution if record['message_id'] not in by_id]
    if missing:
        raise ValueError(f'pipeline output has no row for message(s) {missing}')
    relevance = [0, 0, 0, 0]
    pairs = {'activity_type': [], 'participants': [], 'duplicate_of': []}
    pii = [0, 0, 0, 0]
    records = []
    for record in solution:
        row = by_id[record['message_id']]
        actual, predicted = record['relevance'] == EVENT, row['relevance'] == EVENT
        relevance[(0 if predicted else 2) if actual else (1 if predicted else 3)] += 1
        pairs['activity_type'].append((activity(row['activity_type']), activity(record['activity_type'])))
        for field in ('participants', 'duplicate_of'):
            pairs[field].append((value(row[field]), value(record[field])))
        leaks = [item for item in pii_items(record['pii_items']) if leaked(item, row['anonymized_text'])]
        items = pii_items(record['pii_items'])
        pii[0] += len(items) - len(leaks)
        pii[2] += len(leaks)
        records.append({'id': record['message_id'], 'relevance': [row['relevance'], record['relevance']],
                        'activity_type': [row['activity_type'], record['activity_type']],
                        'participants': [row['participants'], record['participants']],
                        'duplicate_of': [row['duplicate_of'], record['duplicate_of']],
                        'leaked_pii': leaks})
    return {'records': records, 'metrics': {
        'messages': len(solution),
        'relevance': metrics(*relevance),
        'fields': {field: score_slots(field_pairs) for field, field_pairs in pairs.items()},
        'pii': {**metrics(*pii), 'items': pii[0] + pii[2], 'leaked': pii[2]},
        'confusion_matrices': {'activity_type': confusion_matrix(pairs['activity_type'])}}}


def evaluate_extractor(solution, texts, extractor):
    pairs = {field: [] for field in EXTRACTION_FIELDS}
    records = []
    for record, text in zip(solution, texts):
        predicted = extractor.extract(text)
        expected = {'activity_type': activity(record['activity_type']), 'participant_count': value(record['participants'])}
        records.append({'id': record['message_id'], 'predicted': predicted, 'expected': expected})
        for field in EXTRACTION_FIELDS:
            pairs[field].append((predicted[field], expected[field]))
    return {'records': records, 'metrics': {
        'fields': {field: score_slots(field_pairs) for field, field_pairs in pairs.items()},
        'confusion_matrices': {'activity_type': confusion_matrix(pairs['activity_type'])}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--solution', type=Path, default=SOLUTION)
    parser.add_argument('--pipeline-output', type=Path, default=PIPELINE_OUTPUT, help="main.py's CSV for the voice notes")
    parser.add_argument('--models', nargs='*', choices=list(EXTRACTORS), default=[],
                        help='also run these extraction models on the solution and pipeline texts')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()

    solution = load_csv(args.solution)
    rows = load_csv(args.pipeline_output)
    report = {'solution': str(args.solution), 'pipeline_output': str(args.pipeline_output),
              'pipeline': evaluate_pipeline(solution, rows), 'extractors': {}}
    by_id = {str(row['Message_id']): row for row in rows}
    texts = {'solution': [record['expected_anonymized_text'] for record in solution],
             'pipeline': [by_id[record['message_id']]['anonymized_text'] for record in solution]}
    for name in args.models:
        extractor = make_extractor(name)
        report['extractors'][name] = {'model': extractor.model_name, **{
            source: evaluate_extractor(solution, source_texts, extractor) for source, source_texts in texts.items()}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = {'pipeline': report['pipeline']['metrics'],
               'extractors': {name: {source: result[source]['metrics']['fields'] for source in texts}
                              for name, result in report['extractors'].items()}}
    print(json.dumps(summary, indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
