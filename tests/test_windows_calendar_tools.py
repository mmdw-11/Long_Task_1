from datetime import datetime

from examples.windows_calendar_agent.tools import CalendarEvent, write_ics


def test_write_ics_creates_calendar_file(tmp_path):
    event = CalendarEvent(
        title="Project meeting",
        start=datetime(2026, 7, 11, 15, 0, 0),
        end=datetime(2026, 7, 11, 16, 0, 0),
        description="Discuss roadmap",
        location="Office",
    )

    path = write_ics(event, tmp_path)

    text = path.read_text(encoding="utf-8")
    assert "BEGIN:VCALENDAR" in text
    assert "SUMMARY:Project meeting" in text
    assert "DTSTART:20260711T150000" in text
    assert "DTEND:20260711T160000" in text
