"""Save Nigerian-accented English clips into this folder as a fixed speech-to-text test set.

    uv run data/nigerian_accented/download.py [--count 10]

Source: benjaminogbonna/nigerian_accented_english_dataset (Apache-2.0), test split, in
dataset order, the same clips Priyanka streamed in her original `audiotranscription`
script (commit ffe3ee7). Clips are saved as the original audio bytes (no re-encoding),
with reference transcripts in ground_truth.json (same format as data/audio_ground_truth.json),
so src.evaluation.transcription can score them. The dataset's speaker IDs (client_id)
are deliberately not saved: its card asks users not to try to identify speakers.
"""
import argparse
import json
from pathlib import Path

DATASET = "benjaminogbonna/nigerian_accented_english_dataset"
SPLIT = "test"
DEFAULT_COUNT = 10
OUTPUT_DIR = Path(__file__).resolve().parent  # always this folder, wherever the script is run from
GROUND_TRUTH = OUTPUT_DIR / "ground_truth.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="number of clips to save")
    args = parser.parse_args()

    from datasets import Audio, load_dataset

    # decode=False keeps each clip's original file bytes, so no audio decoder is needed here.
    dataset = load_dataset(DATASET, split=SPLIT, streaming=True).cast_column("audio", Audio(decode=False))

    ground_truth = []
    for index, example in enumerate(dataset.take(args.count)):
        audio = example["audio"]
        suffix = Path(audio.get("path") or "").suffix or ".mp3"
        filename = f"clip_{index:02d}{suffix}"
        (OUTPUT_DIR / filename).write_bytes(audio["bytes"])
        ground_truth.append({"file": filename, "speaker": None, "transcript": example["sentence"]})
        print(f"Saved {filename}: {example['sentence']}")

    GROUND_TRUTH.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(ground_truth)} clips and {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
