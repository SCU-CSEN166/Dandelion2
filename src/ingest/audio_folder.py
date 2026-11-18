"""A folder of voice notes or recordings: one message per file.

    uv run main.py --input data/dandelion_audio

Videos (e.g. .mp4) are converted to audio-only .mp3 with ffmpeg, so later steps only
see audio. The copies go in output/audio/<folder name>/ and are reused until the
original changes.
"""

import shutil
import subprocess
from pathlib import Path

from ..config import OUTPUT_DIR
from .base import Source, register
from .schema import Attachment, Message, kind_from_filename

MEDIA_KINDS = ("audio", "video")
MP3_QUALITY = "4"  # ffmpeg VBR scale, 0 (best) to 9 (smallest); 4 is plenty for speech


def _media_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and not path.name.startswith(".") and kind_from_filename(path.name) in MEDIA_KINDS)


def to_mp3(source: Path, target: Path) -> Path:
    """Write the audio track of `source` to `target` as mp3, unless `target` is already up to date."""
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
