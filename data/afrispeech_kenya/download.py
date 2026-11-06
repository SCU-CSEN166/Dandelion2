"""Save Kenyan-accented English conversations into this folder as a fixed speech-to-text test set.

    uv run data/afrispeech_kenya/download.py [--count 3]

Source: intronhealth/afrispeech-dialog (CC BY-NC-SA 4.0: non-commercial use, with
attribution), the conversations whose country is "KE" (all Swahili-accented, general
domain), in metadata order. Each is a 5-14 minute two-speaker dialogue, saved as the
original .wav (no re-encoding). Only the chosen .wav files are downloaded, not the
whole 2.2 GB dataset.

The dataset's transcripts interleave timestamps and speaker labels
("00:02:04\\n[Speaker 1]: On today's discussion..."); those are stripped so
ground_truth.json holds only the spoken words, in the same format as
data/audio_ground_truth.json, and src.evaluation.transcription can score them.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

DATASET = "intronhealth/afrispeech-dialog"
COUNTRY = "KE"
DEFAULT_COUNT = 3  # each clip is a 25-75 MB .wav (44.1 kHz, about 5 MB per minute)
OUTPUT_DIR = Path(__file__).resolve().parent  # always this folder, wherever the script is run from
GROUND_TRUTH = OUTPUT_DIR / "ground_truth.json"

# A leading "00:02:04" timestamp and/or "[Speaker 1]:" / "Speaker C:" label on a transcript line.
LINE_PREFIX = re.compile(r"^\s*(?:\d+:\d+:\d+\s*)?(?:\[?Speaker\s+\w+\]?:\s*)?")


def plain_transcript(transcript: str) -> str:
    """Only the spoken words: timestamps and speaker labels removed, whitespace collapsed."""
    return " ".join(" ".join(LINE_PREFIX.sub("", line) for line in transcript.splitlines()).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="number of conversations to save")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    metadata = hf_hub_download(DATASET, "metadata.csv", repo_type="dataset")
    with open(metadata, encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["country"] == COUNTRY]
    if args.count > len(rows):
        print(f"Only {len(rows)} conversations have country {COUNTRY}; saving all of them")

    ground_truth = []
    for index, row in enumerate(rows[:args.count]):
        filename = f"clip_{index:02d}{Path(row['file_name']).suffix}"
        shutil.copyfile(hf_hub_download(DATASET, row["file_name"], repo_type="dataset"), OUTPUT_DIR / filename)
        ground_truth.append({"file": filename, "speaker": None, "transcript": plain_transcript(row["transcript"])})
        print(f"Saved {filename}: {float(row['duration']) / 60:.1f} min, {row['accent']} accent")

    GROUND_TRUTH.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(ground_truth)} clips and {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
