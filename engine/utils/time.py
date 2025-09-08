"""Time helpers for the engine."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def parse_event_time(date_str: str, time_str: str, tz: str = "Europe/Rome") -> datetime:
    """Parse date and time strings into timezone-aware datetime."""
    naive = datetime.fromisoformat(f"{date_str}T{time_str}")
    return naive.replace(tzinfo=ZoneInfo(tz))
