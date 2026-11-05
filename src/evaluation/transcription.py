"""Compare Distil-Whisper and Whisper Small on recorded speech (WER and CER).

    uv run -m src.evaluation.transcription
    uv run -m src.evaluation.transcription --audio-dir data/nigerian_accented --ground-truth data/nigerian_accented/ground_truth.json --output output/evaluations/transcription_accented.json

Original: Priyanka's `audiotranscription` Colab notebook (commit 280f0a0). Default test
data: the voice notes in data/WhatsAppExports/export2, with Risha's reference transcripts
from data/audio_ground_truth.json. Second test set: 10 Nigerian-accented English clips in
data/nigerian_accented (saved by data/nigerian_accented/download.py).
Every change from the original is marked ADAPTED.
"""
# ADAPTED: removed Colab-only setup (`!apt-get -qq install -y ffmpeg`, `!pip install -q jiwer`,
# `from google.colab import files`); ffmpeg must be installed, and jiwer is a project dependency.
import argparse
import json
import tempfile
from pathlib import Path

from transformers import pipeline
from jiwer import wer, cer
import subprocess
import torch
import gc

# ADAPTED: was `uploaded = files.upload()`; the recordings are now the ones listed in a
# ground-truth file that exist in an audio folder (by default, our WhatsApp voice notes).
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--audio-dir", type=Path, default=Path("data/WhatsAppExports/export2"))
parser.add_argument("--ground-truth", type=Path, default=Path("data/audio_ground_truth.json"))
parser.add_argument("--output", type=Path, default=Path("output/evaluations/transcription.json"))
args = parser.parse_args()

audio_dir = args.audio_dir
ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
for entry in ground_truth:
    if not (audio_dir / entry["file"]).is_file():
        print(f"Skipping {entry['file']}: not in {audio_dir}")
ground_truth = [entry for entry in ground_truth if (audio_dir / entry["file"]).is_file()]
filenames = [entry["file"] for entry in ground_truth]

# Convert recordings to WAV
wav_paths = []
wav_dir = tempfile.mkdtemp()  # ADAPTED: was /content

for i, filename in enumerate(filenames):
    wav_path = f"{wav_dir}/clip_{i}.wav"

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_dir / filename), "-ac", "1", "-ar", "16000", wav_path],  # ADAPTED: path
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    wav_paths.append(wav_path)

device = 0 if torch.cuda.is_available() else -1
print("Using:", "GPU" if device == 0 else "CPU")

# Compare both models on the same recordings
model_ids = [
    "distil-whisper/distil-small.en",
    "openai/whisper-small.en",
]

predictions = {}

for model_id in model_ids:
    print(f"\nLoading {model_id}...")
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=15,  # ADAPTED: added, as in src/transcribe.py; clips over 30 s fail without it
        device=device,
    )

    model_predictions = []

    for filename, wav_path in zip(filenames, wav_paths):
        result = asr(wav_path)
        model_predictions.append(result["text"])
        print(f"\nFile: {filename}")
        print("Transcript:", result["text"])

    predictions[model_id] = model_predictions

    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Enter the correct transcript for each recording, if you know it
# ADAPTED: was typed in at an input() prompt per file; now Risha's reference transcripts.
references = [entry["transcript"] for entry in ground_truth]

# Calculate scores for recordings with a reference transcript
scored = [
    i for i, reference in enumerate(references)
    if reference
]

scores = {}  # ADAPTED: scores kept for the JSON report below, as well as printed

if scored:
    for model_id in model_ids:
        refs = [references[i] for i in scored]
        preds = [predictions[model_id][i] for i in scored]
        scores[model_id] = {"wer": wer(refs, preds), "cer": cer(refs, preds)}

        print(f"\n{model_id}")
        print(f"WER: {wer(refs, preds):.3f}")
        print(f"CER: {cer(refs, preds):.3f}")
else:
    print("\nNo reference transcripts entered, so WER and CER were not calculated.")

# Save transcripts and any references to a CSV
# ADAPTED: was a pandas DataFrame saved as audio_model_comparison.csv (and display(report), which
# only exists in notebooks); now JSON like the other evaluations, with the scores first.
report = {
    "models": model_ids,
    "dataset": args.ground_truth.as_posix(),
    "metrics": {"recordings_scored": len(scored), **scores},
    "records": [
        {"filename": filename, "reference_transcript": reference,
         "transcripts": {model_id: predictions[model_id][i] for model_id in model_ids}}
        for i, (filename, reference) in enumerate(zip(filenames, references))
    ],
}

output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nSaved: {output}")
