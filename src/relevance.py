"""Step 4: zero-shot classification of whether a message reports field activity.

From Prableen's Week 6 baseline. Runs on anonymized text only.
"""

from functools import cached_property

from .ingest import Message
from .utils import require_anonymized, torch_device

CLASSIFIER_MODEL = "facebook/bart-large-mnli"

# Hypotheses the zero-shot model scores each message against.
RELEVANT_HYPOTHESIS = "a report of completed community health, outreach, training or livelihood activities and outcomes"
IRRELEVANT_HYPOTHESIS = "casual conversation, greetings, unrelated information or plans for future activities"

# Week 7 experiment: also count planned activities as relevant. The `status` field
# extracted later still separates completed from planned events.
RELEVANT_WITH_PLANS_HYPOTHESIS = ("a message about a community health, outreach, training or livelihood "
                                  "activity that happened or is planned")
IRRELEVANT_WITHOUT_PLANS_HYPOTHESIS = "casual conversation, greetings or unrelated information"

# Labels stored in Message.relevance["label"].
RELEVANT = "relevant"
IRRELEVANT = "irrelevant"

DEFAULT_THRESHOLD = 0.5  # minimum relevant-hypothesis score; uncalibrated
# BART accepts 1024 tokens including the hypothesis; 180 words stays well under that.
MAX_WORDS_PER_CHUNK = 180


def is_relevant(message: Message) -> bool:
    return message.relevance is not None and message.relevance["label"] == RELEVANT


class RelevanceClassifier:
    def __init__(self, classifier=None, threshold: float = DEFAULT_THRESHOLD, count_plans: bool = False,
                 min_words: int = 0, skip_questions: bool = False):
        """
        classifier: a zero-shot-classification pipeline (tests inject a fake one).
        threshold: minimum relevant-hypothesis score; uncalibrated.
        count_plans: if True, planned activities also count as relevant (Week 7 experiment).
        min_words: messages shorter than this ("nice", "got it") are irrelevant without
            asking the model (Week 8 experiment). 0 turns the filter off.
        skip_questions: if True, messages ending in "?" are irrelevant without asking the
            model (Week 8 experiment). A question asks for field information; the answer
            is a separate message that gets classified on its own.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self._classifier = classifier
        self.threshold = threshold
        self.min_words = min_words
        self.skip_questions = skip_questions
        if count_plans:
            self.relevant_hypothesis = RELEVANT_WITH_PLANS_HYPOTHESIS
            self.irrelevant_hypothesis = IRRELEVANT_WITHOUT_PLANS_HYPOTHESIS
        else:
            self.relevant_hypothesis = RELEVANT_HYPOTHESIS
            self.irrelevant_hypothesis = IRRELEVANT_HYPOTHESIS

    @cached_property
    def classifier(self):
        if self._classifier is not None:
            return self._classifier
        from transformers import pipeline
        return pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=torch_device(allow_mps=False))

    def _relevant_score(self, text: str) -> float:
        result = self.classifier(text, candidate_labels=[self.relevant_hypothesis, self.irrelevant_hypothesis],
                                 multi_label=False)
        score_by_hypothesis = dict(zip(result["labels"], result["scores"]))
        return float(score_by_hypothesis[self.relevant_hypothesis])

    def classify(self, text: str) -> dict:
        """Scores are model scores, not calibrated confidence."""
        if not text.strip():
            return {"label": IRRELEVANT, "score": 0.0, "reason": "empty_message"}
        words = text.split()
        # Week 8 filters: rule out chat and questions before asking the model.
        if len(words) < self.min_words:
            return {"label": IRRELEVANT, "score": 0.0, "reason": "short_message"}
        if self.skip_questions and text.rstrip().endswith("?"):
            return {"label": IRRELEVANT, "score": 0.0, "reason": "question"}
        # Chunk rather than silently dropping the tail of long messages.
        chunks = [" ".join(words[offset:offset + MAX_WORDS_PER_CHUNK])
                  for offset in range(0, len(words), MAX_WORDS_PER_CHUNK)]
        score = max(self._relevant_score(chunk) for chunk in chunks)
        return {"label": RELEVANT if score >= self.threshold else IRRELEVANT, "score": score}


def classify(messages: list[Message], classifier: RelevanceClassifier | None = None) -> None:
    """Fill in `relevance` for each message. Requires anonymize() to have run."""
    require_anonymized(messages)
    classifier = classifier or RelevanceClassifier()
    for message in messages:
        message.relevance = classifier.classify(message.anonymized_text)
