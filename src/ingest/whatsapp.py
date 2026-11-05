"""WhatsApp "Export chat" files: the .zip, an unzipped folder, or the bare .txt.

Both export layouts are supported:
    iOS:      [11/3/26, 12:43:47 AM] Name: text
    Android:  11/3/26, 12:43 AM - Name: text

Lines that don't start with a timestamp are continuations of the previous
message. Attached media is extracted next to the zip (or `media_dir`) so later
steps such as transcription can open it by path.
"""

import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

from .base import Source, register
from .schema import Attachment, AttachmentKind, Message, kind_from_filename

LRM = "‎"  # iOS prefixes system notices and attachment lines with this
_INVISIBLE = dict.fromkeys(map(ord, "‎‏﻿"))
_SPACES = {0x202F: " ", 0xA0: " "}  # newer exports put a narrow no-break space before AM/PM

_DATE = r"(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})"
_TIME = r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?: ?[AaPp]\.? ?[Mm]\.?)?)"
_HEADERS = {
    "ios": re.compile(rf"^\[{_DATE},? {_TIME}\] (?P<body>.*)$"),
    "android": re.compile(rf"^{_DATE},? {_TIME} - (?P<body>.*)$"),
}
_SENDER = re.compile(r"^(?P<sender>[^:\n]+?): (?P<text>.*)$", re.DOTALL)

_IOS_ATTACHED = re.compile(r"<attached: (?P<file>[^>]+)>")
_ANDROID_ATTACHED = re.compile(r"^(?P<file>\S[^\n]*?\.\w{2,5}) \(file attached\)$", re.MULTILINE)
_OMITTED = re.compile(r"<Media omitted>|\b(?P<kind>image|audio|video|GIF|sticker|document) omitted\b", re.IGNORECASE)

_OMITTED_KINDS: dict[str, AttachmentKind] = {
    "image": "image", "gif": "image", "sticker": "image",
    "audio": "audio", "video": "video", "document": "document",
}

Resolver = Callable[[str], str | None]


@register
class WhatsAppSource(Source):
    name = "whatsapp"

    def __init__(self, media_dir: str | Path | None = None, dayfirst: bool | None = None):
        """
        media_dir: where to extract attachments from a .zip (default: next to the
            zip, in a folder with the same name).
        dayfirst: whether dates are D/M/Y. Detected from the data when None.
        """
        self.media_dir = Path(media_dir) if media_dir else None
        self.dayfirst = dayfirst

    @classmethod
    def can_load(cls, path: Path) -> bool:
        try:
            text, _ = _open(path, None)
        except (OSError, ValueError, zipfile.BadZipFile, UnicodeDecodeError):
            return False
        first = next((line for line in text.splitlines() if line.strip()), "")
        return _match_header(first) is not None

    def load(self, path: Path) -> list[Message]:
        path = Path(path)
        text, resolve = _open(path, self.media_dir)
        conversation = path.stem

        # Group raw lines into (format, date, time, body, line number) entries.
        entries: list[list] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            header = _match_header(line)
            if header:
                fmt, m = header
                entries.append([fmt, m["date"], m["time"], m["body"], line_no])
            elif entries:
                entries[-1][3] += "\n" + line

        dayfirst = self.dayfirst if self.dayfirst is not None else _guess_dayfirst([e[1] for e in entries])

        messages = []
        for i, (fmt, date, time, body, line_no) in enumerate(entries):
            m = _SENDER.match(body)
            # iOS prefixes contacts not saved on the exporting phone with "~"
            sender, raw = (m["sender"].translate(_INVISIBLE).lstrip("~ "), m["text"]) if m else (None, body)
            text, attachments = _split_attachments(raw.translate(_INVISIBLE), resolve)
            messages.append(Message(
                id=f"{self.name}:{conversation}:{i}",
                source=self.name,
                text=text,
                sender=sender,
                timestamp=_parse_timestamp(date, time, dayfirst),
                conversation=conversation,
                attachments=attachments,
                # Android notices have no sender; iOS ones are marked with a leading LRM.
                is_system=sender is None or (raw.startswith(LRM) and not attachments),
                metadata={"export_format": fmt, "line": line_no},  # 1-based line in the chat .txt
            ))
        return messages


def _match_header(line: str) -> tuple[str, re.Match] | None:
    line = line.translate(_SPACES).lstrip(LRM)
    for fmt, pattern in _HEADERS.items():
        if m := pattern.match(line):
            return fmt, m
    return None


def _open(path: Path, media_dir: Path | None) -> tuple[str, Resolver]:
    """Return the chat text and a function mapping an attachment filename to a local path."""
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            text = zf.read(_pick_chat(names)).decode("utf-8-sig")
        members = {PurePosixPath(n).name: n for n in names}
        out_dir = media_dir or path.with_suffix("")

        def resolve(filename: str) -> str | None:
            # Keys are bare basenames, so a crafted "../x" filename never matches.
            if filename not in members:
                return None
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / filename
            with zipfile.ZipFile(path) as zf:
                out.write_bytes(zf.read(members[filename]))
            return str(out)

        return text, resolve

    chat = path if path.is_file() else path / _pick_chat([p.name for p in path.iterdir()])
    text = chat.read_text(encoding="utf-8-sig")

    def resolve(filename: str) -> str | None:
        f = chat.parent / filename
        return str(f) if Path(filename).name == filename and f.is_file() else None

    return text, resolve


def _pick_chat(names: list[str]) -> str:
    txts = [n for n in names if n.lower().endswith(".txt")]
    for preferred in ("_chat.txt", "WhatsApp Chat"):  # iOS, then Android naming
        for n in txts:
            if PurePosixPath(n).name.startswith(preferred):
                return n
    if txts:
        return txts[0]
    raise ValueError("no chat .txt file found")


def _split_attachments(text: str, resolve: Resolver) -> tuple[str, list[Attachment]]:
    attachments = []
    for pattern in (_IOS_ATTACHED, _ANDROID_ATTACHED):
        for m in pattern.finditer(text):
            filename = m["file"].strip()
            attachments.append(Attachment(kind_from_filename(filename), filename, resolve(filename)))
        text = pattern.sub("", text)
    for m in _OMITTED.finditer(text):
        kind = _OMITTED_KINDS.get((m["kind"] or "").lower(), "other")
        attachments.append(Attachment(kind))
    text = _OMITTED.sub("", text)
    return text.strip(), attachments


def _guess_dayfirst(dates: list[str]) -> bool:
    pairs = [[int(x) for x in re.split(r"[/.-]", d)[:2]] for d in dates]
    if any(a > 12 for a, _ in pairs):
        return True
    if any(b > 12 for _, b in pairs):
        return False
    # Still ambiguous: dotted dates are a day-first convention; slashes default to US order.
    return any("." in d for d in dates)


def _parse_timestamp(date: str, time: str, dayfirst: bool) -> datetime:
    a, b, year = (int(x) for x in re.split(r"[/.-]", date))
    day, month = (a, b) if dayfirst else (b, a)
    if year < 100:
        year += 2000

    time = time.upper().replace(".", "").replace(" ", "")
    meridiem = time[-2:] if time[-2:] in ("AM", "PM") else None
    parts = [int(x) for x in time.removesuffix(meridiem or "").split(":")]
    hour, minute, second = parts + [0] * (3 - len(parts))
    if meridiem:
        hour = hour % 12 + (12 if meridiem == "PM" else 0)
    return datetime(year, month, day, hour, minute, second)
