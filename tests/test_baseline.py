import json
import unittest
from pathlib import Path
from Anonymization import MessageProcessor, RELEVANT, IRRELEVANT
from evaluate import evaluate, metrics


class BaselineTests(unittest.TestCase):
    def processor(self, entities=(), relevant=0.9, public=()):
        return MessageProcessor(ner=lambda text: entities,
            classifier=lambda text, **kw: {'labels': [IRRELEVANT, RELEVANT],
                                          'scores': [1-relevant, relevant]}, public_entities=public)

    def test_contacts_preserve_counts_and_dates(self):
        text = '28 women, 3 referrals on 2026-11-04. Call +254 721 448 903 or 0721-448-903; mail a@example.org.'
        result = self.processor().anonymize(text)['text']
        self.assertEqual(result, '28 women, 3 referrals on 2026-11-04. Call [PHONE] or [PHONE]; mail [EMAIL].')

    def test_public_policy_and_person(self):
        text = 'Grace in Nakuru'
        ner = [{'entity_group':'PER','start':0,'end':5}, {'entity_group':'LOC','start':9,'end':15}]
        self.assertEqual(self.processor(ner, public=['Nakuru']).anonymize(text)['text'], '[PERSON] in Nakuru')

    def test_overlapping_spans(self):
        ner = [{'entity_group':'PER','start':0,'end':5}]
        self.assertEqual(self.processor(ner).anonymize('a@example.org')['text'], '[EMAIL]')

    def test_classification_uses_label_not_order(self):
        self.assertEqual(self.processor().classify('28 served')['label'], 'relevant')
        self.assertEqual(self.processor(relevant=0.1).classify('hello')['label'], 'irrelevant')

    def test_classification_receives_anonymized_text(self):
        seen = []
        def classify(text, **kwargs):
            seen.append(text)
            return {'labels':[RELEVANT, IRRELEVANT], 'scores':[.9,.1]}
        p = MessageProcessor(ner=lambda t: [], classifier=classify)
        p.process('Call 0721448903')
        self.assertEqual(seen, ['Call [PHONE]'])

    def test_long_message_tail(self):
        calls = []
        def classify(text, **kwargs):
            calls.append(text)
            return {'labels':[RELEVANT, IRRELEVANT], 'scores':[.9 if 'served' in text else .1,.1]}
        p = MessageProcessor(ner=lambda t: [], classifier=classify)
        self.assertEqual(p.classify('hello '*400+'served')['label'], 'relevant')
        self.assertEqual(len(calls), 3)

    def test_partial_mask_is_miss(self):
        p = self.processor([{'entity_group':'PER','start':0,'end':3}])
        records = [{'text':'Grace Wanjiku', 'entities':[{'text':'Grace Wanjiku','start':0,'end':13,'should_redact':True}]}]
        result = evaluate(records,p)['metrics']
        self.assertEqual(result['anonymization_annotated_entities']['FN'], 1)
        self.assertIsNone(result['classification'])

    def test_subword_suffix_is_masked(self):
        p = self.processor([{'entity_group':'PER','start':0,'end':3}])
        self.assertEqual(p.anonymize('Wanjiku served 28')['text'], '[PERSON] served 28')

    def test_empty(self):
        self.assertEqual(self.processor().process('')['classification']['label'], 'irrelevant')
        self.assertIsNone(metrics(0,0,0,0)['recall'])

    def test_dataset_offsets(self):
        data=json.loads(Path('synthetic_unanonymized.json').read_text())
        for record in data['records']:
            for entity in record['entities']:
                self.assertEqual(record['text'][entity['start']:entity['end']], entity['text'])


if __name__ == '__main__':
    unittest.main()
