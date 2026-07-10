"""Local Windows calendar tools.

The tool writes an ``.ics`` file and can open it with the Windows default
calendar handler. Importing the event still requires user confirmation in the
calendar application.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
    description: str = ""
    location: str = ""


def write_ics(event: CalendarEvent, output_dir: str | Path = "runs/calendar") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_filename(event.title)}-{uuid.uuid4().hex[:8]}.ics"
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    content = "\r\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//long_task_1//Windows Calendar Agent//CN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "BEGIN:VEVENT",
            f"UID:{uuid.uuid4().hex}@long-task-1",
            f"DTSTAMP:{now}",
            f"DTSTART:{_format_dt(event.start)}",
            f"DTEND:{_format_dt(event.end)}",
            f"SUMMARY:{_escape_ics(event.title)}",
            f"DESCRIPTION:{_escape_ics(event.description)}",
            f"LOCATION:{_escape_ics(event.location)}",
            "END:VEVENT",
            "END:VCALENDAR",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8", newline="")
    return path


def open_calendar_file(path: str | Path) -> None:
    target = Path(path).resolve()
    if os.name != "nt":
        raise RuntimeError("open_calendar_file is intended for Windows.")
    # Use PowerShell Start-Process so Windows opens the default .ics handler.
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", "Start-Process", "-FilePath", str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Expected ISO datetime, got {value!r}") from exc


def _format_dt(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value.strftime("%Y%m%dT%H%M%S")


def _escape_ics(value: Optional[str]) -> str:
    text = value or ""
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return safe.strip("_") or "calendar-event"
