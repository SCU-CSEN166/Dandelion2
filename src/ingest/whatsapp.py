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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

from .base import Source, register
from .schema import Attachment, AttachmentKind, Message, kind_from_filename

LRM = "‎"  # left-to-right mark; iOS prefixes system notices and attachment lines with it
_INVISIBLE = dict.fromkeys(map(ord, "‎‏﻿"))  # LRM, RLM, BOM: deleted
_NARROW_NO_BREAK_SPACE = 0x202F  # newer exports put one before AM/PM
_NO_BREAK_SPACE = 0xA0
_SPACES = {_NARROW_NO_BREAK_SPACE: " ", _NO_BREAK_SPACE: " "}

_DATE_SEPARATOR = re.compile(r"[/.-]")
_DATE = r"(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})"
_TIME = r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?: ?[AaPp]\.? ?[Mm]\.?)?)"
_HEADERS = {
    "ios": re.compile(rf"^\[{_DATE},? {_TIME}\] (?P<body>.*)$"),
    "android": re.compile(rf"^{_DATE},? {_TIME} - (?P<body>.*)$"),
}
_SENDER = re.compile(r"^(?P<sender>[^:\n]+?): (?P<text>.*)$", re.DOTALL)
_UNSAVED_CONTACT_PREFIX = "~ "  # iOS marks contacts not saved on the exporting phone with "~"

_IOS_ATTACHED = re.compile(r"<attached: (?P<file>[^>]+)>")
_ANDROID_ATTACHED = re.compile(r"^(?P<file>\S[^\n]*?\.\w{2,5}) \(file attached\)$", re.MULTILINE)
_OMITTED = re.compile(r"<Media omitted>|\b(?P<kind>image|audio|video|GIF|sticker|document) omitted\b", re.IGNORECASE)

_OMITTED_KINDS: dict[str, AttachmentKind] = {
    "image": "image", "gif": "image", "sticker": "image",
    "audio": "audio", "video": "video", "document": "document",
}

# Chat file names, most specific first: iOS, then Android.
_PREFERRED_CHAT_NAMES = ("_chat.txt", "WhatsApp Chat")

MONTHS_PER_YEAR = 12
HOURS_PER_MERIDIEM = 12
TWO_DIGIT_YEAR_LIMIT = 100  # years below this are two-digit, e.g. "26"
TWO_DIGIT_YEAR_CENTURY = 2000

Resolver = Callable[[str], str | None]


@dataclass
class _Entry:
    """One message's raw header fields and body, before parsing."""
    export_format: str
    date: str
    time: str
    body: str
    line_number: int  # 1-based line in the chat .txt


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
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        return _match_header(first_line) is not None

    def load(self, path: Path) -> list[Message]:
        path = Path(path)
        text, resolve = _open(path, self.media_dir)
        conversation = path.stem
        entries = _group_entries(text)
        dayfirst = self.dayfirst if self.dayfirst is not None else _guess_dayfirst([entry.date for entry in entries])
        return [
            self._to_message(entry, f"{self.name}:{conversation}:{index}", conversation, resolve, dayfirst)
            for index, entry in enumerate(entries)
        ]

    def _to_message(self, entry: _Entry, message_id: str, conversation: str,
                    resolve: Resolver, dayfirst: bool) -> Message:
        sender_match = _SENDER.match(entry.body)
        if sender_match:
            sender = sender_match["sender"].translate(_INVISIBLE).lstrip(_UNSAVED_CONTACT_PREFIX)
            raw_text = sender_match["text"]
        else:
            sender, raw_text = None, entry.body
        text, attachments = _split_attachments(raw_text.translate(_INVISIBLE), resolve)
        return Message(
            id=message_id,
            source=self.name,
            text=text,
            sender=sender,
            timestamp=_parse_timestamp(entry.date, entry.time, dayfirst),
            conversation=conversation,
            attachments=attachments,
            # Android notices have no sender; iOS ones are marked with a leading LRM.
            is_system=sender is None or (raw_text.startswith(LRM) and not attachments),
            metadata={"export_format": entry.export_format, "line": entry.line_number},
        )


def _group_entries(text: str) -> list[_Entry]:
    """Split the chat into messages; lines without a header continue the previous one."""
    entries: list[_Entry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        header = _match_header(line)
        if header:
            export_format, match = header
            entries.append(_Entry(export_format, match["date"], match["time"], match["body"], line_number))
        elif entries:
            entries[-1].body += "\n" + line
    return entries


def _match_header(line: str) -> tuple[str, re.Match] | None:
    line = line.translate(_SPACES).lstrip(LRM)
    for export_format, pattern in _HEADERS.items():
        if match := pattern.match(line):
            return export_format, match
    return None


def _open(path: Path, media_dir: Path | None) -> tuple[str, Resolver]:
    """Return the chat text and a function mapping an attachment filename to a local path."""
    if path.is_file() and zipfile.is_zipfile(path):
        return _open_zip(path, media_dir or path.with_suffix(""))
    return _open_text(path)


def _open_zip(path: Path, media_dir: Path) -> tuple[str, Resolver]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        text = archive.read(_pick_chat(names)).decode("utf-8-sig")
    members = {PurePosixPath(name).name: name for name in names}

    def resolve(filename: str) -> str | None:
        # Keys are bare basenames, so a crafted "../x" filename never matches.
        if filename not in members:
            return None
        media_dir.mkdir(parents=True, exist_ok=True)
        extracted = media_dir / filename
        with zipfile.ZipFile(path) as archive:
            extracted.write_bytes(archive.read(members[filename]))
        return str(extracted)

    return text, resolve


def _open_text(path: Path) -> tuple[str, Resolver]:
    """`path` is the chat .txt itself, or an unzipped export folder containing it."""
    chat = path if path.is_file() else path / _pick_chat([child.name for child in path.iterdir()])
    text = chat.read_text(encoding="utf-8-sig")

    def resolve(filename: str) -> str | None:
        media = chat.parent / filename
        is_bare_name = Path(filename).name == filename  # reject "../x" and other paths
        return str(media) if is_bare_name and media.is_file() else None

    return text, resolve


def _pick_chat(names: list[str]) -> str:
    text_files = [name for name in names if name.lower().endswith(".txt")]
    for preferred in _PREFERRED_CHAT_NAMES:
        for name in text_files:
            if PurePosixPath(name).name.startswith(preferred):
                return name
    if text_files:
        return text_files[0]
    raise ValueError("no chat .txt file found")


def _split_attachments(text: str, resolve: Resolver) -> tuple[str, list[Attachment]]:
    attachments = []
    for pattern in (_IOS_ATTACHED, _ANDROID_ATTACHED):
        for match in pattern.finditer(text):
            filename = match["file"].strip()
            attachments.append(Attachment(kind_from_filename(filename), filename, resolve(filename)))
        text = pattern.sub("", text)
    for match in _OMITTED.finditer(text):
        kind = _OMITTED_KINDS.get((match["kind"] or "").lower(), "other")
        attachments.append(Attachment(kind))
    text = _OMITTED.sub("", text)
    return text.strip(), attachments


def _date_numbers(date: str) -> list[int]:
    """"11/3/26" -> [11, 3, 26]."""
    return [int(part) for part in _DATE_SEPARATOR.split(date)]


def _guess_dayfirst(dates: list[str]) -> bool:
    leading_pairs = [_date_numbers(date)[:2] for date in dates]
    if any(first > MONTHS_PER_YEAR for first, _ in leading_pairs):
        return True
    if any(second > MONTHS_PER_YEAR for _, second in leading_pairs):
        return False
    # Still ambiguous: dotted dates are a day-first convention; slashes default to US order.
    return any("." in date for date in dates)


def _parse_timestamp(date: str, time: str, dayfirst: bool) -> datetime:
    first, second, year = _date_numbers(date)
    day, month = (first, second) if dayfirst else (second, first)
    if year < TWO_DIGIT_YEAR_LIMIT:
        year += TWO_DIGIT_YEAR_CENTURY

    time = time.upper().replace(".", "").replace(" ", "")
    meridiem = time[-2:] if time[-2:] in ("AM", "PM") else None
    hour, minute, *optional_seconds = (int(part) for part in time.removesuffix(meridiem or "").split(":"))
    second_of_minute = optional_seconds[0] if optional_seconds else 0
    if meridiem:
        hour = hour % HOURS_PER_MERIDIEM + (HOURS_PER_MERIDIEM if meridiem == "PM" else 0)
    return datetime(year, month, day, hour, minute, second_of_minute)
