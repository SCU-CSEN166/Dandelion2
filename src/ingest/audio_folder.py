"""A folder of standalone voice notes / recordings: one message per file.

Video files (e.g. .mp4 recordings) are reduced to audio-only .mp3 with ffmpeg
before transcription, so the pipeline only ever sees audio. The .mp3 copies go
in `mp3_dir` (default output/audio/<folder name>/, which is not committed) and
are reused on later runs unless the original is newer.

    uv run main.py --folder data/dandelion_audio
"""

import shutil
import subprocess
from pathlib import Path

from ..config import OUTPUT_DIR
from .base import Source, register
from .schema import Attachment, Message, kind_from_filename

MEDIA_KINDS = ("audio", "video")
MP3_QUALITY = "4"  # libmp3lame VBR quality: 0 best .. 9 smallest; 4 is plenty for speech


def _media_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and not path.name.startswith(".") and kind_from_filename(path.name) in MEDIA_KINDS)


def to_mp3(source: Path, target: Path) -> Path:
    """Write the audio track of `source` to `target` as mp3 (video stream dropped)."""
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert video files to mp3 (brew install ffmpeg)")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-vn", "-codec:a", "libmp3lame", "-q:a", MP3_QUALITY, str(target)],
        check=True,
    )
    return target


@register
class AudioFolderSource(Source):
    name = "audio"

    def __init__(self, mp3_dir: str | Path | None = None):
        """mp3_dir: where converted .mp3 files go (default: output/audio/<folder name>)."""
        self.mp3_dir = Path(mp3_dir) if mp3_dir else None

    @classmethod
    def can_load(cls, path: Path) -> bool:
        return path.is_dir() and bool(_media_files(path))

    def load(self, path: Path) -> list[Message]:
        folder = Path(path)
        mp3_dir = self.mp3_dir or OUTPUT_DIR / "audio" / folder.name
        messages = []
        for media in _media_files(folder):
            audio = media if kind_from_filename(media.name) == "audio" else to_mp3(media, mp3_dir / f"{media.stem}.mp3")
            messages.append(Message(
                id=f"{self.name}:{folder.name}:{media.stem}",
                source=self.name,
                conversation=folder.name,
                attachments=[Attachment(kind="audio", filename=audio.name, path=str(audio))],
                metadata={"original_file": media.name},
            ))
        return messages
