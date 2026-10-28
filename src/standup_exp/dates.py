"""Resolve spoken due-date phrases ("by Thursday", "tomorrow at half two") to dates.

One documented convention is used everywhere, both to compute gold dates in the
generator and to interpret extracted phrases in the normalisers:

* "today", "this morning", "this afternoon", "end of day", "after lunch", "tonight" -> meeting date
* "tomorrow" -> meeting date + 1
* bare weekday ("Thursday", "by Thurs", "on Friday") -> next occurrence strictly after today
* "next <weekday>" -> that weekday in the following calendar week (Mon-Sun weeks)
* "end of the week" / "by Friday this week" -> Friday of the current week (or today if later)
* "next week" (no weekday) -> Monday of next week
* "the 3rd" / "on the 21st" -> next occurrence of that day of month (today counts)
* anything else -> None

Clock times ("at 2pm", "by 10:30", "half two") are returned separately as "HH:MM".
Work hours are assumed, so an ambiguous "at 2" or "half two" means 14:00 / 14:30.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "weds": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_WD = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def resolve_date(phrase: str | None, meeting_date: date) -> date | None:
    if not phrase:
        return None
    p = phrase.lower()

    m = re.search(rf"\bnext\s+({_WD})\b", p)
    if m:
        target = WEEKDAYS[m.group(1)]
        next_monday = meeting_date + timedelta(days=7 - meeting_date.weekday())
        return next_monday + timedelta(days=target)

    if re.search(r"\b(end of (the )?week|this week|by the weekend)\b", p):
        friday = meeting_date + timedelta(days=4 - meeting_date.weekday())
        return max(friday, meeting_date)

    if re.search(r"\bnext week\b", p):
        return meeting_date + timedelta(days=7 - meeting_date.weekday())

    if re.search(r"\b(tomorrow|tmrw|tomoz)\b", p):
        return meeting_date + timedelta(days=1)

    m = re.search(rf"\b({_WD})\b", p)
    if m:
        delta = (WEEKDAYS[m.group(1)] - meeting_date.weekday()) % 7
        return meeting_date + timedelta(days=delta or 7)

    m = re.search(r"\b(?:the\s+)(\d{1,2})(?:st|nd|rd|th)\b", p)
    if m:
        day = int(m.group(1))
        y, mo = meeting_date.year, meeting_date.month
        for _ in range(3):
            try:
                candidate = date(y, mo, day)
            except ValueError:
                candidate = None
            if candidate and candidate >= meeting_date:
                return candidate
            mo, y = (1, y + 1) if mo == 12 else (mo + 1, y)
        return None

    if re.search(
        r"\b(today|this morning|this afternoon|end of (the )?day|eod|after lunch|"
        r"before lunch|tonight|first thing|straight away|right away|asap|now|"
        r"by close|end of shift|this shift)\b",
        p,
    ):
        return meeting_date

    # A bare clock time ("at 2pm") implies today.
    if resolve_time(p):
        return meeting_date
    return None


def resolve_time(phrase: str | None) -> str | None:
    if not phrase:
        return None
    p = phrase.lower()

    m = re.search(r"\bhalf\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\b", p)
    if m:
        h = _WORD_NUM.get(m.group(1)) or int(m.group(1))
        return f"{_work_hour(h):02d}:30"

    m = re.search(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", p)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{_apply_ampm(h, ap):02d}:{mi:02d}"

    m = re.search(r"\b(\d{1,2})\s*(am|pm|o'?clock)\b", p)
    if m:
        h, ap = int(m.group(1)), m.group(2)
        ap = ap if ap in ("am", "pm") else None
        return f"{_apply_ampm(h, ap):02d}:00"

    m = re.search(r"\b(?:at|by|for)\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", p)
    if m:
        return f"{_work_hour(_WORD_NUM[m.group(1)]):02d}:00"

    if re.search(r"\bnoon|midday\b", p):
        return "12:00"
    return None


def _apply_ampm(h: int, ap: str | None) -> int:
    if ap == "pm" and h < 12:
        return h + 12
    if ap == "am" and h == 12:
        return 0
    return _work_hour(h) if ap is None else h


def _work_hour(h: int) -> int:
    """Interpret 1-6 as afternoon (work-hours convention)."""
    return h + 12 if 1 <= h <= 6 else h
