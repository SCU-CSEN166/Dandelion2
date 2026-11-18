"""Compare Distil-Whisper and Whisper Small on recorded speech (WER and CER).

    uv run -m src.evaluation.transcription
    uv run -m src.evaluation.transcription --audio-dir data/nigerian_accented \\
        --ground-truth data/nigerian_accented/ground_truth.json --output output/evaluations/transcription_accented.json

Scores every recording listed in the ground-truth file that exists in the audio folder.
Needs ffmpeg.
"""

import argparse
import gc
import json
import subprocess
import tempfile
from pathlib import Path

from src.transcribe import DEFAULT_CHUNK_SECONDS
from src.utils import write_json

MODELS = ["distil-whisper/distil-small.en", "openai/whisper-small.en"]
SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono
FIRST_GPU, CPU = 0, -1  # Hugging Face pipeline device numbers


def load_ground_truth(path: Path, audio_dir: Path) -> list[dict]:
    """The ground-truth entries whose recording is in `audio_dir`."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    for entry in entries:
        if not (audio_dir / entry["file"]).is_file():
            print(f"Skipping {entry['file']}: not in {audio_dir}")
    return [entry for entry in entries if (audio_dir / entry["file"]).is_file()]


def convert_to_wav(recordings: list[Path], wav_dir: Path) -> list[Path]:
    wav_paths = []
    for index, recording in enumerate(recordings):
        wav_path = wav_dir / f"clip_{index}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(recording), "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wav_paths.append(wav_path)
    return wav_paths


def transcribe_all(model_id: str, wav_paths: list[Path], names: list[str], device: int) -> list[str]:
    import torch
    from transformers import pipeline

    print(f"\nLoading {model_id}...")
    asr = pipeline("automatic-speech-recognition", model=model_id, chunk_length_s=DEFAULT_CHUNK_SECONDS,
                   device=device)
    transcripts = []
    for wav_path, name in zip(wav_paths, names):
        transcripts.append(asr(str(wav_path))["text"])
        print(f"\nFile: {name}\nTranscript: {transcripts[-1]}")
    del asr  # free the model before loading the next one
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transcripts


def error_rates(references: list[str], transcripts: list[str]) -> dict:
    from jiwer import cer, wer

    return {"wer": wer(references, transcripts), "cer": cer(references, transcripts)}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, default=Path("data/WhatsAppExports/export2"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/audio_ground_truth.json"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluations/transcription.json"))
    args = parser.parse_args()

    entries = load_ground_truth(args.ground_truth, args.audio_dir)
    wav_paths = convert_to_wav([args.audio_dir / entry["file"] for entry in entries], Path(tempfile.mkdtemp()))
    device = FIRST_GPU if torch.cuda.is_available() else CPU
    print("Using:", "GPU" if device == FIRST_GPU else "CPU")
    names = [entry["file"] for entry in entries]
    transcripts = {model_id: transcribe_all(model_id, wav_paths, names, device) for model_id in MODELS}

    scored = [index for index, entry in enumerate(entries) if entry["transcript"]]
    references = [entries[index]["transcript"] for index in scored]
    rates = {model_id: error_rates(references, [transcripts[model_id][index] for index in scored])
             for model_id in MODELS} if scored else {}
    for model_id, model_rates in rates.items():
        print(f"\n{model_id}\nWER: {model_rates['wer']:.3f}\nCER: {model_rates['cer']:.3f}")
    if not scored:
        print("\nNo reference transcripts, so WER and CER were not calculated.")

    report = {
        "models": MODELS,
        "dataset": args.ground_truth.as_posix(),
        "metrics": {"recordings_scored": len(scored), **rates},
        "records": [{"filename": entry["file"], "reference_transcript": entry["transcript"],
                     "transcripts": {model_id: transcripts[model_id][index] for model_id in MODELS}}
                    for index, entry in enumerate(entries)],
    }
    write_json(args.output, report)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
