import json
import unittest
from src.extract import Extractor, extract
from src.ingest import Message


def fake_extractor(output):
    return Extractor(generate=lambda text: output if isinstance(output, str) else json.dumps(output))


class ExtractTests(unittest.TestCase):
    def test_valid_output(self):
        extractor = fake_extractor({'status': 'completed', 'activity_type': 'youth health session',
                                    'location': 'Nakuru', 'participant_count': 42, 'referrals': '2'})
        result = extractor.extract('youth session in Nakuru, 42 students, 2 referrals')
        self.assertEqual(result, {'status': 'completed', 'activity_type': 'youth health session',
                                  'location': 'Nakuru', 'participant_count': 42, 'referrals': 2,
                                  'needs_review': []})

    def test_invalid_enum_dropped(self):
        result = fake_extractor({'status': 'done', 'activity_type': 'party'}).extract('done')
        self.assertIsNone(result['status'])
        self.assertIsNone(result['activity_type'])

    def test_values_missing_from_text_need_review(self):
        result = fake_extractor({'location': 'Baringo', 'participant_count': 30}).extract('we met 31 women')
        self.assertEqual(result['needs_review'], ['location', 'participant_count'])

    def test_unparseable_output(self):
        result = fake_extractor('not json').extract('31 women')
        self.assertEqual(result['needs_review'], ['unparseable_output'])
        self.assertIsNone(result['participant_count'])

    def test_only_relevant_messages_extracted(self):
        relevant = Message(id='1', source='t', anonymized_text='31 women', relevance={'label': 'relevant', 'score': .9})
        other = Message(id='2', source='t', anonymized_text='hi', relevance={'label': 'irrelevant', 'score': .1})
        extract([relevant, other], fake_extractor({'participant_count': 31}))
        self.assertEqual(relevant.extraction['participant_count'], 31)
        self.assertIsNone(other.extraction)

    def test_extract_requires_anonymize(self):
        with self.assertRaises(ValueError):
            extract([Message(id='1', source='t', text='31 women')], fake_extractor({}))


if __name__ == '__main__':
    unittest.main()
