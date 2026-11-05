"""Compare ASR models on accented English and on our own WhatsApp voice notes.

    uv run -m src.evaluation.asr

Streams only the first NUM_CLIPS clips of the accented-English dataset, so the full
dataset is never downloaded. Every model in MODELS transcribes the same clips; WER and
CER are reported per model and per clip set, and all transcripts are saved to OUTPUT.
Model comparison and CER from Priyanka's audio evaluation.
"""

import csv
import gc
import re
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer

from src.transcribe import Transcriber

DATASET = "benjaminogbonna/nigerian_accented_english_dataset"
SPLIT = "test"
NUM_CLIPS = 10

MODELS = [
    "distil-whisper/distil-small.en",
    "openai/whisper-small.en",
]

# Our own WhatsApp voice notes: path -> exact words spoken.
# Leave a reference as "" to transcribe the clip without scoring it.
LOCAL_CLIPS = {
    "WhatsAppExports/export/00000010-AUDIO-2026-11-03-00-44-54.opus": "",
    "WhatsAppExports/export2/00003084-AUDIO-2026-11-04-16-05-11.opus": "",
    "WhatsAppExports/export2/00003085-AUDIO-2026-11-04-16-05-26.opus": "",
}

OUTPUT = Path("output/asr_comparison.csv")


def normalize(text: str) -> str:
    """Lowercase and drop punctuation, so "Oil Rivers." and "oil rivers" count as a match."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def load_clips(num_clips: int = NUM_CLIPS) -> list[dict]:
    """Every clip as {"set", "clip", "audio", "reference"}, loaded once and reused for each model."""
    # decode=False yields the raw mp3 bytes, which the ASR pipeline decodes with
    # ffmpeg (decoding inside `datasets` would need torchcodec).
    dataset = load_dataset(DATASET, split=SPLIT, streaming=True).cast_column("audio", Audio(decode=False))
    clips = [{"set": "accented", "clip": example["path"], "audio": example["audio"]["bytes"],
              "reference": example["sentence"]} for example in dataset.take(num_clips)]
    for path, reference in LOCAL_CLIPS.items():
        if Path(path).is_file():
            clips.append({"set": "whatsapp", "clip": path, "audio": path, "reference": reference})
        else:
            print(f"Skipping missing clip {path}")
    return clips


def score(rows: list[dict]) -> dict:
    """WER/CER over the rows that have a reference transcript."""
    scored = [r for r in rows if r["reference"]]
    if not scored:
        return {"scored": 0, "wer": None, "wer_normalized": None, "cer_normalized": None}
    refs = [r["reference"] for r in scored]
    preds = [r["prediction"] for r in scored]
    return {
        "scored": len(scored),
        "wer": wer(refs, preds),
        "wer_normalized": wer([normalize(t) for t in refs], [normalize(t) for t in preds]),
        "cer_normalized": cer([normalize(t) for t in refs], [normalize(t) for t in preds]),
    }


def evaluate(transcriber: Transcriber, clips: list[dict]) -> dict:
    rows = []
    for i, clip in enumerate(clips, start=1):
        print(f"  {transcriber.model}: clip {i}/{len(clips)}", flush=True)
        rows.append({**clip, "prediction": transcriber(clip["audio"])})
    sets = sorted({r["set"] for r in rows})
    return {"model": transcriber.model, "rows": rows,
            "scores": {s: score([r for r in rows if r["set"] == s]) for s in sets}}


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    clips = load_clips()
    results = []
    for model in MODELS:
        transcriber = Transcriber(model)
        results.append(evaluate(transcriber, clips))
        del transcriber  # one model in memory at a time
        free_memory()

    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    print(f"\n{'model':<34} {'set':<9} {'scored':>6} {'WER':>6} {'WER*':>6} {'CER*':>6}   (* = ignoring case/punctuation)")
    for result in results:
        for clip_set, s in result["scores"].items():
            print(f"{result['model']:<34} {clip_set:<9} {s['scored']:>6} {fmt(s['wer']):>6} "
                  f"{fmt(s['wer_normalized']):>6} {fmt(s['cer_normalized']):>6}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "clip", "reference"] + [f"{r['model']} transcript" for r in results])
        for i, clip in enumerate(clips):
            writer.writerow([clip["set"], clip["clip"], clip["reference"]] + [r["rows"][i]["prediction"] for r in results])
    print(f"\nSaved transcripts: {OUTPUT}")
