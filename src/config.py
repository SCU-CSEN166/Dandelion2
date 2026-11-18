"""Settings shared by the pipeline (main.py) and the evaluation scripts."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path("output")

# Where main.py --transcript-cache keeps transcripts between runs. They are raw, un-anonymized
# speech: fine to commit for our test data, never for real Dandelion messages.
TRANSCRIPT_CACHE = OUTPUT_DIR / "transcript_cache.json"

# Public places the anonymizer leaves visible, e.g. "Nakuru": Kenya's counties plus a few
# public organizations. Never build this list from evaluation labels; it would inflate the scores.
PUBLIC_ENTITIES: list[str] = json.loads((DATA_DIR / "public_entities.json").read_text(encoding="utf-8"))

# BERT and GLiNER each catch names the other misses, so run both.
USE_GLINER = True

RELEVANCE_COUNT_PLANS = True     # planned activities count as relevant, not only completed ones
RELEVANCE_MIN_WORDS = 4          # shorter messages ("nice", "got it") are irrelevant
RELEVANCE_SKIP_QUESTIONS = True  # questions are irrelevant; the answer is its own message

EXTRACTION_BACKEND = "nuextract"  # or "flan-t5"
