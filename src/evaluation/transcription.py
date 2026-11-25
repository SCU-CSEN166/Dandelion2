"""Compare Whisper models on the same audio set (accuracy and elapsed time).

    uv run -m src.evaluation.transcription
    uv run -m src.evaluation.transcription --audio-dir data/nigerian_accented \
        --ground-truth data/nigerian_accented/ground_truth.json
    uv run -m src.evaluation.transcription --audio-dir data/dandelion_audio \
        --ground-truth data/dandelion_audio/asr_ground_truth.json \
        --entity-targets data/dandelion_audio/asr_entity_targets.json

Model selection should use a development set. If a separate holdout set is scored, report it
after choosing the model and do not use those results to tune or select the model.
Needs ffmpeg and internet access to download the selected Hugging Face models.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from src.transcribe import DEFAULT_CHUNK_SECONDS
from src.utils import torch_device, write_json

MODELS = {
    "distil-small": "distil-whisper/distil-small.en",
    "whisper-small": "openai/whisper-small.en",
    "whisper-large": "openai/whisper-large-v3",
}
DEFAULT_MODELS = ["distil-small", "whisper-small"]
SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono


def load_ground_truth(path: Path, audio_dir: Path) -> list[dict]:
    """The ground-truth entries whose recording is in `audio_dir`."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    missing = [entry["file"] for entry in entries if not (audio_dir / entry["file"]).is_file()]
    for name in missing:
        print(f"Skipping {name}: not in {audio_dir}")
    return [entry for entry in entries if (audio_dir / entry["file"]).is_file()]


def convert_to_wav(recordings: list[Path], wav_dir: Path) -> list[Path]:
    wav_paths = []
    for index, recording in enumerate(recordings):
        wav_path = wav_dir / f"clip_{index}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(recording), "-ac", "1", "-ar", str(SAMPLE_RATE),
                        str(wav_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wav_paths.append(wav_path)
    return wav_paths


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def transcribe_all(model_id: str, wav_paths: list[Path], names: list[str], device: str) -> dict:
    import torch
    from transformers import pipeline

    device_id = 0 if device == "cuda" else -1
    print(f"\nLoading {model_id} on {device}...")
    load_start = time.perf_counter()
    asr = pipeline("automatic-speech-recognition", model=model_id,
                   chunk_length_s=DEFAULT_CHUNK_SECONDS, device=device_id)
    load_seconds = time.perf_counter() - load_start

    transcripts, clip_seconds = [], []
    for wav_path, name in zip(wav_paths, names):
        clip_start = time.perf_counter()
        transcripts.append(asr(str(wav_path))["text"].strip())
        clip_seconds.append(time.perf_counter() - clip_start)
        print(f"\nFile: {name}\nTranscript: {transcripts[-1]}")

    del asr  # free the model before loading the next one
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"transcripts": transcripts, "load_seconds": load_seconds,
            "clip_seconds": clip_seconds, "inference_seconds": sum(clip_seconds)}


def error_rates(references: list[str], transcripts: list[str]) -> dict:
    from jiwer import cer, wer

    return {"wer": wer(references, transcripts), "cer": cer(references, transcripts)}


def _normalize_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


_DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
                "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _digit_runs(text: str) -> list[str]:
    """Normalize digits written numerically or as individual spoken digit words."""
    runs, current = [], ""
    for token in re.findall(r"\d+|[a-z]+", text.casefold()):
        if token.isdigit():
            current += token
        elif token in _DIGIT_WORDS:
            current += _DIGIT_WORDS[token]
        else:
            if current:
                runs.append(current)
                current = ""
    if current:
        runs.append(current)
    return runs


def entity_metrics(entries: list[dict], transcripts: list[str], targets_path: Path | None) -> dict | None:
    """Exact mention recall for manually labeled names, places, and phone numbers."""
    if targets_path is None:
        return None
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    by_file = {item["file"]: item for item in targets}
    results = {}
    categories = ("names", "locations", "phone_numbers")
    for category in categories:
        expected = found = 0
        for entry, transcript in zip(entries, transcripts):
            expected_items = by_file.get(entry["file"], {}).get(category, [])
            for item in expected_items:
                expected += 1
                if category == "phone_numbers":
                    digits = re.sub(r"\D", "", item)
                    found += any(digits in run for run in _digit_runs(transcript))
                else:
                    target = _normalize_phrase(item)
                    words = _normalize_phrase(transcript)
                    found += bool(target and re.search(rf"(?<!\w){re.escape(target)}(?!\w)", words))
        results[category] = {"exact_hits": found, "targets": expected,
                             "exact_recall": found / expected if expected else None}
    return results


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, default=Path("data/nigerian_accented"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/nigerian_accented/ground_truth.json"))
    parser.add_argument("--entity-targets", type=Path,
                        help="optional JSON with per-file names, locations, and phone_numbers")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=DEFAULT_MODELS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="score only the first N files; 0 means all files")
    parser.add_argument("--output", type=Path, default=Path("output/evaluations/transcription.json"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is not available")
    device = torch_device(allow_mps=False) if args.device == "auto" else args.device
    entries = load_ground_truth(args.ground_truth, args.audio_dir)
    if args.limit:
        entries = entries[:max(0, args.limit)]
    if not entries:
        parser.error("No ground-truth audio files were found; check --audio-dir and --ground-truth")

    wav_paths = convert_to_wav([args.audio_dir / entry["file"] for entry in entries], Path(tempfile.mkdtemp()))
    names = [entry["file"] for entry in entries]
    references = [entry.get("transcript", "") for entry in entries]
    scored = [index for index, reference in enumerate(references) if reference.strip()]
    durations = [audio_duration(path) for path in wav_paths]
    audio_seconds = sum(durations)
    results = {}
    per_model_transcripts = {}

    for key in args.models:
        model_id = MODELS[key]
        run = transcribe_all(model_id, wav_paths, names, device)
        transcripts = run.pop("transcripts")
        per_model_transcripts[key] = transcripts
        run["metrics"] = error_rates([references[i] for i in scored], [transcripts[i] for i in scored]) if scored else None
        run["audio_seconds"] = audio_seconds
        run["seconds_per_audio_second"] = run["inference_seconds"] / audio_seconds if audio_seconds else None
        run["seconds_per_clip_mean"] = run["inference_seconds"] / len(transcripts)
        run["entity_metrics"] = entity_metrics(entries, transcripts, args.entity_targets)
        results[key] = run

    report = {
        "models": {key: MODELS[key] for key in args.models},
        "dataset": str(args.ground_truth),
        "device": device,
        "recordings": len(entries),
        "recordings_scored": len(scored),
        "runtime_note": "inference time excludes ffmpeg conversion; model load time is reported separately",
        "results": results,
        "records": [{"filename": name, "reference_transcript": entry.get("transcript", ""),
                     **{key: per_model_transcripts[key][index] for key in args.models}}
                    for index, (name, entry) in enumerate(zip(names, entries))],
    }
    write_json(args.output, report)
    print(json.dumps({key: {"metrics": value["metrics"], "load_seconds": round(value["load_seconds"], 2),
                            "inference_seconds": round(value["inference_seconds"], 2),
                            "seconds_per_audio_second": round(value["seconds_per_audio_second"], 2),
                            "entity_metrics": value["entity_metrics"]}
                      for key, value in results.items()}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
