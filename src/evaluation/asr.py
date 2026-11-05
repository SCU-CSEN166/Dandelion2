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

from src.config import OUTPUT_DIR
from src.transcribe import Transcriber

DATASET = "benjaminogbonna/nigerian_accented_english_dataset"
SPLIT = "test"
NUM_CLIPS = 10

MODELS = [
    "distil-whisper/distil-small.en",
    "openai/whisper-small.en",
]

# Our own WhatsApp voice notes: path -> exact words spoken (from data/test_results.md).
# Leave a reference as "" to transcribe the clip without scoring it.
LOCAL_CLIPS = {
    "WhatsAppExports/export/00000010-AUDIO-2026-11-03-00-44-54.opus": "",
    "WhatsAppExports/export2/00003084-AUDIO-2026-11-04-16-05-11.opus":
        "lowkey me too im really stressed about what my grade is gonna be and i mean she said "
        "she's not gonna curve so im really nervous",
    "WhatsAppExports/export2/00003085-AUDIO-2026-11-04-16-05-26.opus":
        "wait yeah i was like losing so much sleep like oh my god guys before tuesdays class "
        "and then we had the quiz i was tweaking out",
}

OUTPUT = OUTPUT_DIR / "asr_comparison.csv"

# Score table layout
MODEL_COLUMN_WIDTH = 34
SET_COLUMN_WIDTH = 9
NUMBER_COLUMN_WIDTH = 6
SCORE_DECIMALS = 3


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
    scored = [row for row in rows if row["reference"]]
    if not scored:
        return {"scored": 0, "wer": None, "wer_normalized": None, "cer_normalized": None}
    references = [row["reference"] for row in scored]
    predictions = [row["prediction"] for row in scored]
    normalized_references = [normalize(text) for text in references]
    normalized_predictions = [normalize(text) for text in predictions]
    return {
        "scored": len(scored),
        "wer": wer(references, predictions),
        "wer_normalized": wer(normalized_references, normalized_predictions),
        "cer_normalized": cer(normalized_references, normalized_predictions),
    }


def evaluate(transcriber: Transcriber, clips: list[dict]) -> dict:
    rows = []
    for clip_number, clip in enumerate(clips, start=1):
        print(f"  {transcriber.model}: clip {clip_number}/{len(clips)}", flush=True)
        rows.append({**clip, "prediction": transcriber(clip["audio"])})
    clip_sets = sorted({row["set"] for row in rows})
    return {"model": transcriber.model, "rows": rows,
            "scores": {clip_set: score([row for row in rows if row["set"] == clip_set]) for clip_set in clip_sets}}


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.{SCORE_DECIMALS}f}"


def _table_row(model: str, clip_set: str, *numbers: str) -> str:
    cells = [f"{model:<{MODEL_COLUMN_WIDTH}}", f"{clip_set:<{SET_COLUMN_WIDTH}}"]
    cells += [f"{number:>{NUMBER_COLUMN_WIDTH}}" for number in numbers]
    return " ".join(cells)


def print_scores(results: list[dict]) -> None:
    print("\n" + _table_row("model", "set", "scored", "WER", "WER*", "CER*") + "   (* = ignoring case/punctuation)")
    for result in results:
        for clip_set, scores in result["scores"].items():
            print(_table_row(result["model"], clip_set, str(scores["scored"]), _format_score(scores["wer"]),
                             _format_score(scores["wer_normalized"]), _format_score(scores["cer_normalized"])))


def write_transcripts(clips: list[dict], results: list[dict]) -> None:
    """One CSV row per clip, with every model's transcript side by side."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["set", "clip", "reference"] + [f"{result['model']} transcript" for result in results])
        for clip_index, clip in enumerate(clips):
            transcripts = [result["rows"][clip_index]["prediction"] for result in results]
            writer.writerow([clip["set"], clip["clip"], clip["reference"]] + transcripts)


def main() -> None:
    clips = load_clips()
    results = []
    for model in MODELS:
        transcriber = Transcriber(model)
        results.append(evaluate(transcriber, clips))
        del transcriber  # one model in memory at a time
        free_memory()
    print_scores(results)
    write_transcripts(clips, results)
    print(f"\nSaved transcripts: {OUTPUT}")


if __name__ == "__main__":
    main()
