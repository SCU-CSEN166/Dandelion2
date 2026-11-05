import unittest

from src.anonymize import Anonymizer, anonymize
from src.evaluation.baseline import DATASETS, ConfusionMatrix, evaluate, from_whatsapp_label, load_records
from src.ingest import Message
from src.relevance import IRRELEVANT_HYPOTHESIS, RELEVANT_HYPOTHESIS, RelevanceClassifier, classify


def fake_anonymizer(entities=(), public=()):
    return Anonymizer(ner=lambda text: entities, public_entities=public)


def fake_classifier(relevant=0.9):
    return RelevanceClassifier(classifier=lambda text, **kwargs: {'labels': [IRRELEVANT_HYPOTHESIS, RELEVANT_HYPOTHESIS],
                                                              'scores': [1 - relevant, relevant]})


class BaselineTests(unittest.TestCase):
    def test_contacts_preserve_counts_and_dates(self):
        text = '28 women, 3 referrals on 2026-11-04. Call +254 721 448 903 or 0721-448-903; mail a@example.org.'
        result = fake_anonymizer().anonymize(text)['text']
        self.assertEqual(result, '28 women, 3 referrals on 2026-11-04. Call [PHONE] or [PHONE]; mail [EMAIL].')

    def test_public_policy_and_person(self):
        text = 'Grace in Nakuru'
        ner = [{'entity_group':'PER','start':0,'end':5}, {'entity_group':'LOC','start':9,'end':15}]
        self.assertEqual(fake_anonymizer(ner, public=['Nakuru']).anonymize(text)['text'], '[PERSON] in Nakuru')

    def test_overlapping_spans(self):
        ner = [{'entity_group':'PER','start':0,'end':5}]
        self.assertEqual(fake_anonymizer(ner).anonymize('a@example.org')['text'], '[EMAIL]')

    def test_classification_uses_label_not_order(self):
        self.assertEqual(fake_classifier().classify('28 served')['label'], 'relevant')
        self.assertEqual(fake_classifier(relevant=0.1).classify('hello')['label'], 'irrelevant')

    def test_classification_receives_anonymized_text(self):
        seen = []
        def classify_fn(text, **kwargs):
            seen.append(text)
            return {'labels': [RELEVANT_HYPOTHESIS, IRRELEVANT_HYPOTHESIS], 'scores': [.9, .1]}
        messages = [Message(id='m', source='test', text='Call 0721448903')]
        anonymize(messages, fake_anonymizer())
        classify(messages, RelevanceClassifier(classifier=classify_fn))
        self.assertEqual(seen, ['Call [PHONE]'])

    def test_classify_requires_anonymize(self):
        with self.assertRaises(ValueError):
            classify([Message(id='m', source='test', text='hi')], fake_classifier())

    def test_long_message_tail(self):
        calls = []
        def classify_fn(text, **kwargs):
            calls.append(text)
            return {'labels': [RELEVANT_HYPOTHESIS, IRRELEVANT_HYPOTHESIS], 'scores': [.9 if 'served' in text else .1, .1]}
        classifier = RelevanceClassifier(classifier=classify_fn)
        self.assertEqual(classifier.classify('hello '*400+'served')['label'], 'relevant')
        self.assertEqual(len(calls), 3)

    def test_partial_mask_is_miss(self):
        records = [{'text':'Grace Wanjiku', 'entities':[{'text':'Grace Wanjiku','start':0,'end':13,'should_redact':True}]}]
        result = evaluate(records, fake_anonymizer([{'entity_group':'PER','start':0,'end':3}]), fake_classifier())['metrics']
        self.assertEqual(result['anonymization_annotated_entities']['FN'], 1)
        self.assertIsNone(result['classification'])

    def test_subword_suffix_is_masked(self):
        anonymizer = fake_anonymizer([{'entity_group':'PER','start':0,'end':3}])
        self.assertEqual(anonymizer.anonymize('Wanjiku served 28')['text'], '[PERSON] served 28')

    def test_empty(self):
        self.assertEqual(fake_classifier().classify('')['label'], 'irrelevant')
        self.assertIsNone(ConfusionMatrix().metrics()['recall'])

    def test_senders_pseudonymized_and_masked_in_text(self):
        messages = [Message(id='1', source='test', sender='Grace Wanjiru', text='hello'),
                    Message(id='2', source='test', sender='Mary', text='thanks grace, see you'),
                    Message(id='3', source='test', sender='Grace Wanjiru', text='Wanjiru here')]
        anonymize(messages, fake_anonymizer())
        self.assertEqual([message.anonymized_sender for message in messages], ['Sender 1', 'Sender 2', 'Sender 1'])
        self.assertEqual(messages[1].anonymized_text, 'thanks [PERSON], see you')
        self.assertEqual(messages[2].anonymized_text, '[PERSON] here')

    def test_whatsapp_label_conversion(self):
        label = {'message_id': 'msg_1', 'text': 'Esther in Nakuru, Esther again', 'relevance': 'relevant',
                 'location': 'Nakuru', 'pii': [{'type': 'PERSON', 'value': 'Esther'}]}
        record = from_whatsapp_label(label)
        self.assertEqual([(entity['start'], entity['should_redact']) for entity in record['entities']],
                         [(0, True), (18, True), (10, False)])

    def test_dataset_offsets(self):
        for path in DATASETS.values():
            for record in load_records(path):
                for entity in record['entities']:
                    self.assertEqual(record['text'][entity['start']:entity['end']], entity['text'])


if __name__ == '__main__':
    unittest.main()
