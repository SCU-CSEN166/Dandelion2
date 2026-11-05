"""Word error rate of the ASR model on accented English.

    uv run -m src.evaluation.asr

Streams only the first NUM_CLIPS clips of the dataset, so the full dataset is
never downloaded.
"""

import re

from datasets import Audio, load_dataset
from jiwer import wer

from src.transcribe import Transcriber

DATASET = "benjaminogbonna/nigerian_accented_english_dataset"
SPLIT = "test"
NUM_CLIPS = 10


def normalize(text: str) -> str:
    """Lowercase and drop punctuation, so "Oil Rivers." and "oil rivers" count as a match."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def evaluate(transcriber: Transcriber, num_clips: int = NUM_CLIPS) -> dict:
    # decode=False yields the raw mp3 bytes, which the ASR pipeline decodes with
    # ffmpeg (decoding inside `datasets` would need torchcodec).
    dataset = load_dataset(DATASET, split=SPLIT, streaming=True).cast_column("audio", Audio(decode=False))

    rows = []
    for i, example in enumerate(dataset.take(num_clips), start=1):
        print(f"Transcribing clip {i}/{num_clips}...", flush=True)
        rows.append({
            "clip": example["path"],
            "reference": example["sentence"],
            "prediction": transcriber(example["audio"]["bytes"]),
        })

    references = [r["reference"] for r in rows]
    predictions = [r["prediction"] for r in rows]
    return {
        "model": transcriber.model,
        "dataset": f"{DATASET} ({SPLIT})",
        "clips": len(rows),
        "wer": wer(references, predictions),
        "wer_normalized": wer([normalize(r) for r in references], [normalize(p) for p in predictions]),
        "rows": rows,
    }


if __name__ == "__main__":
    results = evaluate(Transcriber())
    for row in results["rows"]:
        print("REFERENCE:", row["reference"])
        print("MODEL:    ", row["prediction"])
        print()
    print(f"{results['model']} on {results['clips']} clips of {results['dataset']}")
    print(f"WER:              {results['wer']:.3f}")
    print(f"WER (normalized): {results['wer_normalized']:.3f}")
