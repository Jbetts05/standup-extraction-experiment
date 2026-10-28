"""Check that a rendered transcript really contains its answer key.

A generated transcript is only useful if every gold task is recoverable from
the text. For each task we look at its evidence turns (plus one turn either
side) and require:

* the component, site and owner reference words appear (case-insensitive),
  or, for volunteered tasks, the owner is one of the speakers;
* the due phrase resolves to the same date (and time) as the gold, when set.

Problems are returned as human-readable strings and fed back to the LLM renderer.
"""

from __future__ import annotations

import re

from ..dates import resolve_date, resolve_time
from ..ontology import Ontology
from ..schema import Meeting


def _contains(haystack: str, needle: str) -> bool:
    """Whole-token, case-insensitive phrase match."""
    tokens = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).split()  # noqa: E731
    h, n = tokens(haystack), tokens(needle)
    return any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))


def check_meeting(meeting: Meeting, onto: Ontology) -> list[str]:
    problems: list[str] = []
    utts = meeting.utterances
    if len(utts) < 15:
        problems.append(f"transcript too short ({len(utts)} turns)")
    for t in meeting.gold_tasks:
        if not t.evidence:
            problems.append(f"{t.id}: no evidence turns given")
            continue
        lo, hi = max(0, min(t.evidence) - 1), min(len(utts), max(t.evidence) + 2)
        window = utts[lo:hi]
        text = " ".join(u.text for u in window)
        speakers = {u.speaker_id for u in window}

        if t.component.text and not _contains(text, t.component.text):
            problems.append(f"{t.id}: component words '{t.component.text}' not found in its turns")
        if t.site.text and not _contains(text, t.site.text):
            problems.append(f"{t.id}: site words '{t.site.text}' not found in its turns")
        if t.owner.entity_id:
            volunteered = not t.assigned_by and not t.reassigned_from
            said = t.owner.text and _contains(text, t.owner.text)
            if not said and not (volunteered and t.owner.entity_id in speakers):
                problems.append(f"{t.id}: owner '{t.owner.text}' not named and not the speaker")
        if t.due_mention:
            got_date = resolve_date(text, meeting.date)
            # Resolve over each turn separately so other deadlines in the window don't interfere.
            per_turn = {resolve_date(u.text, meeting.date) for u in window}
            if t.due_date not in per_turn and got_date != t.due_date:
                problems.append(f"{t.id}: due phrase '{t.due_mention}' not clearly stated in its turns")
            if t.due_time and t.due_time not in {resolve_time(u.text) for u in window}:
                problems.append(f"{t.id}: time in '{t.due_mention}' not clearly stated")
    return problems
