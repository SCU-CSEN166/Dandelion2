"""Reusable Week 6 baselines. Importing this module never downloads models."""
import re
from functools import cached_property

RELEVANT = 'a report of completed community health, outreach, training or livelihood activities and outcomes'
IRRELEVANT = 'casual conversation, greetings, unrelated information or plans for future activities'


class MessageProcessor:
    def __init__(self, ner=None, classifier=None, public_entities=(), threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        self._ner = ner
        self._classifier = classifier
        self.public_entities = {s.casefold() for s in public_entities}
        self.threshold = threshold

    @cached_property
    def ner(self):
        if self._ner is not None:
            return self._ner
        from transformers import pipeline
        return pipeline('token-classification', model='dslim/bert-base-NER',
                        aggregation_strategy='simple', stride=64, device=-1)

    @cached_property
    def classifier(self):
        if self._classifier is not None:
            return self._classifier
        from transformers import pipeline
        return pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)

    def detect(self, text):
        """Return original character spans, without copying private surface text.

        Unknown locations/organizations are conservatively masked. Explicit,
        reviewed public entities can be preserved; never derive this list from gold labels.
        """
        if not text.strip():
            return []
        spans = []
        for entity in self.ner(text):
            kind = entity['entity_group']
            start, end = int(entity['start']), int(entity['end'])
            # BERT can return only a subword; mask the whole boundary word.
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            if kind not in {'PER', 'LOC', 'ORG'}:
                continue
            if kind != 'PER' and text[start:end].casefold() in self.public_entities:
                continue
            spans.append({'start': start, 'end': end,
                          'label': {'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORG'}[kind]})
        patterns = {
            'EMAIL': r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w-])',
            'PHONE': r'(?<!\w)(?:\+?254[\s.-]?[17]|0[17])(?:[\s.-]?\d){8}(?!\w)',
            'NATIONAL_ID': r'\b(?:ID\s*(?:No\.?|number)?\s*[:#.-]?\s*)(\d{7,8})\b',
        }
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.I):
                start, end = match.span(1 if label == 'NATIONAL_ID' else 0)
                if text[start:end].casefold() not in self.public_entities:
                    spans.append({'start': start, 'end': end, 'label': label})
        # Union overlapping spans so partial model predictions cannot expose suffixes.
        merged = []
        for span in sorted(spans, key=lambda s: (s['start'], -s['end'])):
            if merged and span['start'] < merged[-1]['end']:
                merged[-1]['end'] = max(merged[-1]['end'], span['end'])
            else:
                merged.append(dict(span))
        return merged

    def anonymize(self, text):
        spans = self.detect(text)
        result = text
        for span in reversed(spans):
            result = result[:span['start']] + '[' + span['label'] + ']' + result[span['end']:]
        return {'text': result, 'redactions': spans}

    def classify(self, text):
        """Classify anonymized text. Scores are model scores, not calibrated confidence."""
        if not text.strip():
            return {'label': 'irrelevant', 'score': 0.0, 'reason': 'empty_message'}
        # Chunk rather than silently dropping the tail of long messages.
        tokens = text.split()
        scores = []
        for offset in range(0, len(tokens), 180):
            result = self.classifier(' '.join(tokens[offset:offset + 180]),
                                     candidate_labels=[RELEVANT, IRRELEVANT], multi_label=False)
            scores.append(float(dict(zip(result['labels'], result['scores']))[RELEVANT]))
        score = max(scores)
        return {'label': 'relevant' if score >= self.threshold else 'irrelevant', 'score': score}

    def process(self, text):
        result = self.anonymize(text)
        return {**result, 'classification': self.classify(result['text'])}


"""
TODO: Presenting Week 6
1) Parse WhatsApp data
2) Transcribe when necessary
3) Anonymize Data
4) Determine Relevance
5) Extract to JSON


# AND SETUP GUIDE




# Test data (synthetic, human, WhatsApp, AfriSpeech data from Hugginface)
# Input that (capable of accepting WhatsApp primarily)
# GO THROUGH PIPELINE:
    1) Parse WhatsApp data
    2) Transcribe when necessary
    3) Anonymize Data
    4) Determine Relevance
    5) Extract to JSON
# Show processed and filtered output (in like a .json)
# Show results from confusion matrix where applicable
# Install guide
# Models used, models tried
# Attempts
# Contributions




Running Baseline &
Results: metrics: accuracy, recall, 

"""
