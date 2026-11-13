"""Settings shared by the pipeline (main.py) and the evaluation scripts."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Reviewed public places/organizations the anonymizer may leave unmasked, e.g. "Nakuru".
# Kenya's 47 counties plus a few public organizations. Never derive this list from evaluation labels.
PUBLIC_ENTITIES: list[str] = json.loads((DATA_DIR / "public_entities.json").read_text(encoding="utf-8"))

# Anonymization: run GLiNER alongside BERT (each catches names the other misses).
USE_GLINER = True

# Relevance: planned activities count as relevant, and short messages ("nice", "got it")
# and questions are ruled irrelevant before the model is asked.
RELEVANCE_COUNT_PLANS = True
RELEVANCE_MIN_WORDS = 4
RELEVANCE_SKIP_QUESTIONS = True

OUTPUT_DIR = Path("output")
