"""Step 4: decide whether each message reports field activity (zero-shot classification).

    uv run main.py --message "Today we held a youth health session with 20 girls"
    uv run -m src.evaluation.classification data/whatsapp_labels.json

Runs on anonymized text only. Scores are raw model scores, not calibrated confidence.
"""

from functools import cached_property

from .ingest import Message
from .utils import require_anonymized, torch_device

CLASSIFIER_MODEL = "facebook/bart-large-mnli"

# (relevant descriptions, irrelevant descriptions) the model scores each message against.
# A message's relevant score is the summed probability of its relevant descriptions.
_ACTIVITY = "community health, outreach, training or livelihood activity"
COMPLETED_ONLY_HYPOTHESES = (
    ("a report of completed community health, outreach, training or livelihood activities and outcomes",),
    ("casual conversation, greetings, unrelated information or plans for future activities",),
)
WITH_PLANS_HYPOTHESES = (
    # Final: past and upcoming field activities are separate descriptions, and social plans are
    # spelled out, so "let's visit her this weekend" is not mistaken for a planned activity.
    # On data/relevance_plans_dev.json this went from 24/30 to 27/30 correct; the 74 labeled
    # WhatsApp messages and the 10 hidden-test messages got the same predictions as before.
    (f"a message about a {_ACTIVITY} that happened",
     f"a plan or schedule for an upcoming {_ACTIVITY}"),
    ("casual conversation, greetings, personal or social plans like visits, celebrations, gifts, food, "
     "travel or errands, or unrelated information",),
)

RELEVANT = "relevant"
IRRELEVANT = "irrelevant"

DEFAULT_THRESHOLD = 0.5
# Scores this close to the threshold are marked "uncertain" for a person to check. On our labeled
# data (74 WhatsApp + 30 plans messages) a 0.25 margin flags about a quarter of the messages and
# catches 9 of the 13 mistakes; most mistakes scored 0.5-0.75, never above 0.9.
REVIEW_MARGIN = 0.25
MAX_WORDS_PER_CHUNK = 180  # BART reads at most 1024 tokens, hypothesis included


def is_relevant(message: Message) -> bool:
    return message.relevance is not None and message.relevance["label"] == RELEVANT


class RelevanceClassifier:
    def __init__(self, classifier=None, threshold: float = DEFAULT_THRESHOLD, count_plans: bool = False,
                 min_words: int = 0, skip_questions: bool = False):
        """
        classifier: a zero-shot-classification pipeline; tests can pass a fake.
        threshold: lowest relevant score that counts as relevant.
        count_plans: planned activities count as relevant too (extraction's `status` tells them apart).
        min_words: shorter messages are irrelevant without asking the model; 0 turns this off.
        skip_questions: messages ending in "?" are irrelevant without asking the model.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self._classifier = classifier
        self.threshold = threshold
        self.min_words = min_words
        self.skip_questions = skip_questions
        hypotheses = WITH_PLANS_HYPOTHESES if count_plans else COMPLETED_ONLY_HYPOTHESES
        self.relevant_hypotheses, self.irrelevant_hypotheses = hypotheses

    @cached_property
    def classifier(self):
        if self._classifier is not None:
            return self._classifier
        from transformers import pipeline
        return pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=torch_device(allow_mps=False))

    def classify(self, text: str) -> dict:
        reason = self._rule_out(text)
        if reason:
            return {"label": IRRELEVANT, "score": 0.0, "reason": reason}
        # Long messages are split, not truncated, so a report at the end still counts.
        score = max(self._relevant_score(chunk) for chunk in _chunks(text.split()))
        return {"label": RELEVANT if score >= self.threshold else IRRELEVANT, "score": score,
                "uncertain": abs(score - self.threshold) < REVIEW_MARGIN}

    def _rule_out(self, text: str) -> str | None:
        """Why `text` is irrelevant without asking the model, or None."""
        if not text.strip():
            return "empty_message"
        if len(text.split()) < self.min_words:
            return "short_message"
        if self.skip_questions and text.rstrip().endswith("?"):
            return "question"
        return None

    def _relevant_score(self, text: str) -> float:
        labels = [*self.relevant_hypotheses, *self.irrelevant_hypotheses]
        result = self.classifier(text, candidate_labels=labels, multi_label=False)
        score_by_hypothesis = dict(zip(result["labels"], result["scores"]))
        return float(sum(score_by_hypothesis[hypothesis] for hypothesis in self.relevant_hypotheses))


def _chunks(words: list[str]) -> list[str]:
    return [" ".join(words[start:start + MAX_WORDS_PER_CHUNK])
            for start in range(0, len(words), MAX_WORDS_PER_CHUNK)]


def classify(messages: list[Message], classifier: RelevanceClassifier | None = None) -> None:
    """Fill in `relevance` for each message."""
    require_anonymized(messages)
    classifier = classifier or RelevanceClassifier()
    for message in messages:
        message.relevance = classifier.classify(message.anonymized_text)
