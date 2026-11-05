"""Step 4: zero-shot classification of whether a message reports field activity.

From Prableen's Week 6 baseline. Runs on anonymized text only.
"""

from functools import cached_property

from .ingest import Message

CLASSIFIER_MODEL = "facebook/bart-large-mnli"

RELEVANT = "a report of completed community health, outreach, training or livelihood activities and outcomes"
IRRELEVANT = "casual conversation, greetings, unrelated information or plans for future activities"


class RelevanceClassifier:
    def __init__(self, classifier=None, threshold: float = 0.5):
        """
        classifier: a zero-shot-classification pipeline (tests inject a fake one).
        threshold: minimum RELEVANT score; uncalibrated.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self._classifier = classifier
        self.threshold = threshold

    @cached_property
    def classifier(self):
        if self._classifier is not None:
            return self._classifier
        from transformers import pipeline
        return pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=-1)

    def classify(self, text: str) -> dict:
        """Scores are model scores, not calibrated confidence."""
        if not text.strip():
            return {"label": "irrelevant", "score": 0.0, "reason": "empty_message"}
        # Chunk rather than silently dropping the tail of long messages.
        tokens = text.split()
        scores = []
        for offset in range(0, len(tokens), 180):
            result = self.classifier(" ".join(tokens[offset:offset + 180]),
                                     candidate_labels=[RELEVANT, IRRELEVANT], multi_label=False)
            scores.append(float(dict(zip(result["labels"], result["scores"]))[RELEVANT]))
        score = max(scores)
        return {"label": "relevant" if score >= self.threshold else "irrelevant", "score": score}

def classify(messages: list[Message], classifier: RelevanceClassifier | None = None) -> None:
    """Fill in `relevance` for each message. Requires anonymize() to have run."""
    classifier = classifier or RelevanceClassifier()
    for m in messages:
        if m.anonymized_text is None:
            raise ValueError(f"{m.id} has not been anonymized; run anonymize() first")
        m.relevance = classifier.classify(m.anonymized_text)
