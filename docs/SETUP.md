# Week 6 classification and anonymization baseline

This covers Prableen's component, now pipeline steps 3 and 4: `src/anonymize.py`
masks a parsed message or transcript and `src/relevance.py` classifies the masked text.
`main.py` runs them after WhatsApp parsing (`src/ingest`) and ASR (`src/transcribe.py`),
then extracts activity fields from relevant messages (`src/extract.py`). It writes
`output/messages.json` (every message, anonymized) and `output/reports.json` (relevant
messages only, with status, activity type, location, participant count and referrals).

## Install and run

Use Python 3.12 (the repository's `.python-version`) and [uv](https://docs.astral.sh/uv/).
ffmpeg is needed to decode audio (`brew install ffmpeg`).

```sh
uv sync
uv run python -m unittest discover -s tests -v
uv run -m src.evaluation.baseline   # anonymization + relevance -> output/baseline_<dataset>.json
uv run -m src.evaluation.extraction # extraction fields -> output/extraction_whatsapp_labels.json
uv run -m src.evaluation.asr        # ASR word error rate on accented English
uv run main.py                      # full pipeline on WhatsAppExports/export2
```

First inference downloads both models from Hugging Face and requires network access
and several GB of disk/RAM. CPU is explicitly selected for reproducibility. Imports and unit tests do not download
models. `output/test_example.json` is a deterministic integration example. Unit tests inject deterministic model responses; they verify software behavior,
not pretrained model quality.

```python
from src.anonymize import Anonymizer
from src.relevance import RelevanceClassifier
masked = Anonymizer(public_entities=['Nakuru', 'Dandelion Africa']).anonymize('Grace served 28 women in Nakuru. Call 0721 448 903.')
label = RelevanceClassifier().classify(masked['text'])
```

`PUBLIC_ENTITIES` (in `main.py` and `src/evaluation/baseline.py`) is a reviewed list of
public place, organization, or contact strings. Matching is exact and case insensitive. Construct this
policy independently of evaluation labels. Unknown LOC and ORG spans are masked by
default, preserving safety at the cost of reporting detail. PER spans are always masked.
Counts, activity descriptions and general dates are not deliberately removed.

## Models and policy

- [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en):
  transcribes voice notes (needs ffmpeg).
- [numind/NuExtract-2.0-2B](https://huggingface.co/numind/NuExtract-2.0-2B): fills a JSON
  template (see `TEMPLATE` in `src/extract.py`) from anonymized text. Loaded without
  `trust_remote_code`; uses the GPU (CUDA or Apple MPS) when available. Numbers and
  locations not found in the message are listed in `needs_review`.
- [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli): zero-shot
  classification with descriptive relevant/irrelevant labels. Completed activity reports
  count as relevant; plans and casual chat count as irrelevant. Default relevant-score
  threshold is 0.5, uncalibrated. Long messages use 180-word chunks and maximum score.
- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER): aggregated person,
  location and organization spans, with overlapping token windows for long inputs.
- Regex adds Kenyan phone formats, email addresses and context-qualified national IDs.
  It does not recognize every international phone format, birth date, transaction code,
  clinic identifier or case number. NER can miss names or misidentify ordinary words.
  Output requires human review and is not guaranteed fully de-identified.

Only the above two models are used by this component. Anonymization.py previously
contained exploratory snippets for Whisper, Flan-T5, MiniLM, GLiNER and NuExtract;
those snippets are not evidence that alternatives were successfully evaluated.

## Evaluation and outputs

`output/baseline_<dataset>.json` contains message IDs, masked text, redaction offsets into the
original text (without entity surface text), classification decisions and aggregate
metrics. It omits input text and sender metadata. Missed identifiers may still appear
in masked output; do not publish results from private inputs without review.

Two labeled datasets are evaluated (see `DATASETS` in `src/evaluation/baseline.py`):

- `data/whatsapp_labels.json`: the 31 messages of the `WhatsAppExports/export2` group
  chat, with PII values and relevance labels. PII values are located in the text to get
  offsets; a labeled location that is not PII counts as a should-not-redact entity.
- `Archive/synthetic_unanonymized.json`: 20 synthetic records with denser PII and
  public-entity distractors, but no relevance labels.

The synthetic set has 20 synthetic records and entity annotations. Its authorship is not recorded here; confirm with Risha that this is her
intended test set. Gold offsets refer to `text`, not `translation_en`. Evaluation checks
offsets and never uses gold spans to generate predictions or public-entity policies.

The anonymization confusion matrix uses annotated entities as decision units:
TP = fully covered private entity, FN = incompletely covered private entity,
FP = safe entity touched by redaction, TN = safe entity untouched. Accuracy, precision,
recall and F1 apply to these units, not arbitrary text spans. Redactions outside all
annotations are reported separately. Undefined ratios are JSON null.

The synthetic set has no relevance labels, so its classification metrics are null; they
come from `data/whatsapp_labels.json`, whose records have `relevance` equal to `relevant` or `irrelevant`. Do not infer relevance
ground truth from entity labels. Use independently reviewed relevant and irrelevant
examples to measure classification quality.

For the presentation, report measured output from the evaluation command, demonstrate
one success and one failure, and describe the privacy-versus-utility tradeoff. Record
actual team contributions and experiment outcomes yourself in the course README.

## Measured local run

The available Python 3.9.6 interpreter ran Transformers 4.57.6 and Torch 2.8.0 on CPU;
the project targets Python 3.12, which was not available for this verification.
Ten unit tests passed. On the 20 supplied synthetic records, the final conservative
policy with an empty public allowlist produced TP=51, FP=26, FN=15, TN=10:
59.8% annotated-entity accuracy, 66.2% precision, 77.3% recall, and 71.3% F1.
It preserved 10 of 36 annotated safe entities and made 16 additional redactions outside
annotations. Classification metrics are unavailable without relevance labels.

An initial run using raw NER subword spans fully masked 43 of 66 private entities.
Inspection revealed exposed name suffixes; expanding predictions to whole boundary
words increased full coverage to 51 of 66. This is development-set iteration, not
held-out evidence. Results and attempt counts are saved in `output/metrics.json`;
full processed/filtered decisions are in the local `output/baseline.json`.
