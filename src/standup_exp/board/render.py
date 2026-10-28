"""Write each stand-up in a series: new jobs as before, plus updates on cards from earlier days.

* ``render_day_template`` - deterministic and offline (tests, demo). Stilted but aligned.
* ``render_day_llm``      - the Foundry writer model writes a natural meeting from the plan; the
  result is validated (``check_day``) and regenerated with feedback on failure.

Both fill ``day.utterances``, each new job's ``evidence`` and each update's ``evidence``.
"""

from __future__ import annotations

import json
import random

from pydantic import BaseModel

from ..dates import resolve_date
from ..generate.render import RENDER_INSTRUCTIONS, _plan, _timed
from ..generate.validate import _contains, check_meeting
from ..llm import Usage, default_model, structured
from ..ontology import Ontology
from ..paths import env
from .schema import GoldUpdate, StandupDay


# -------------------------------------------------------------------- template
def _update_lines(u: GoldUpdate, day: StandupDay, onto: Ontology, card_owner: str | None) -> list[tuple[str, str]]:
    ref, note = u.reference, f", {u.note}" if u.note else ""
    relay = card_owner and u.speaker_id != card_owner and u.kind != "reassigned"
    who = f"{u.owner_words} asked me to pass on that " if relay and u.owner_words else ""
    if u.kind == "done":
        return [(u.speaker_id, f"Update on an old one: {who}the job to {ref} is done{note}.")]
    if u.kind == "progress":
        return [(u.speaker_id, f"On the job to {ref}, {who}{u.note or 'still going'}.")]
    if u.kind == "blocked":
        return [(u.speaker_id, f"Blocker: {who}the job to {ref} is stuck{note}.")]
    if u.kind == "cancelled":
        return [(u.speaker_id, f"We can drop the job to {ref}{note}.")]
    if u.kind == "due_changed":
        return [(u.speaker_id, f"The job to {ref} is going to slip, {who}it'll be {u.due_mention} now.")]
    new = u.new_owner.text if u.new_owner else "someone"
    if "picked_up" in u.tags:
        return [(u.speaker_id, f"I'll pick up the job to {ref}, nobody had it.")]
    return [(u.speaker_id, f"{new}, can you take over the job to {ref}?"),
            (u.new_owner.entity_id if u.new_owner else day.chair_id, "Yeah, I'll take that over.")]


def render_day_template(day: StandupDay, onto: Ontology) -> StandupDay:
    from ..generate.render import render_template

    render_template(day, onto)  # chit-chat, distractors and new jobs
    owners = {c.id: c.owner_id for c in day.board_before}
    rows = [(u.speaker_id, u.text) for u in day.utterances]
    insert_at = 1 + sum(1 for d in day.distractors if d.kind == "chitchat") * 2
    extra: list[tuple[str, str]] = []
    spans: dict[str, list[int]] = {}
    for u in day.updates:
        lines = _update_lines(u, day, onto, owners.get(u.card_id))
        spans[u.id] = list(range(insert_at + len(extra), insert_at + len(extra) + len(lines)))
        extra += lines
    rows = rows[:insert_at] + extra + rows[insert_at:]
    for t in day.gold_tasks:
        t.evidence = [i + len(extra) if i >= insert_at else i for i in t.evidence]
    for u in day.updates:
        u.evidence = spans[u.id]
    day.utterances = _timed(rows, onto, random.Random(day.id))
    day.renderer = "template"
    return day


# ------------------------------------------------------------------------- LLM
class _LLMUtterance(BaseModel):
    speaker_id: str
    text: str


class _Evidence(BaseModel):
    id: str
    utterance_indices: list[int]


class _LLMDay(BaseModel):
    utterances: list[_LLMUtterance]
    task_evidence: list[_Evidence]
    update_evidence: list[_Evidence]


DAY_INSTRUCTIONS = RENDER_INSTRUCTIONS + """

This stand-up is one day in a run of daily stand-ups, so people also give updates on jobs
agreed at EARLIER stand-ups. The plan's "updates" list says what to report about each one.
- "describe_job_as" tells you WHICH job it is, not a script. Say it the way people really do:
  shortened, in their own words ("the fire alarm test", "those seals on the GT2 inlet valve",
  "the contractor for the GT2 valve"). Never say "the job to ...". When "equipment_words" is
  given, those exact words must appear in the update; otherwise do not name the equipment.
- done: it is clearly finished ("that's sorted", "boxed that off", "all done"). progress: work
  happened but it is still open (work the note in). blocked: it cannot go on (give the reason).
  cancelled: it is not happening any more (say why). reassigned: the job moves to new_owner, who
  accepts it. due_changed: the new deadline is stated with the given due phrase.
- Say each update naturally, once. Do NOT spell out the status for a minute-taker ("still open,
  not finished", "don't mark it as done", "that's separate from ..."). Real people don't.
- The speaker says the update. If the speaker is not the job's owner, make it clear they are
  passing it on for the owner (use owner_words), e.g. "Dave asked me to say ...".
- Do not mention any other earlier job, and do not turn an update into a new job.
- New jobs in "tasks" are brand new today, exactly as before.

Also return update_evidence: for every update id, the 0-based indices of the turns where it is said."""


def _day_plan(day: StandupDay, onto: Ontology) -> dict:
    name = lambda pid: onto.by_id[pid].name if pid else None  # noqa: E731
    cards = {c.id: c for c in day.board_before}
    plan = _plan(day, onto)
    plan["stand_up_number"] = f"day {day.day} of the run"
    plan["absent_today"] = [name(p) for p in day.team if p not in day.attendees]
    plan["updates"] = [
        {
            "update_id": u.id,
            "kind": u.kind,
            "job": cards[u.card_id].summary,
            "job_owner": name(cards[u.card_id].owner_id) or "nobody",
            "owner_words": u.owner_words,
            "speaker": {"id": u.speaker_id, "name": name(u.speaker_id)},
            "describe_job_as": u.reference,
            "equipment_words": u.component.text if u.component else None,
            "note": u.note,
            "new_owner": {"id": u.new_owner.entity_id, "refer_to_as": u.new_owner.text} if u.new_owner else None,
            "due_phrase": u.due_mention,
        }
        for u in day.updates
    ]
    return plan


def check_day(day: StandupDay, onto: Ontology) -> list[str]:
    problems = check_meeting(day, onto)
    utts = day.utterances
    for u in day.updates:
        if not u.evidence:
            problems.append(f"{u.id}: no evidence turns given")
            continue
        lo, hi = max(0, min(u.evidence) - 1), min(len(utts), max(u.evidence) + 2)
        window = utts[lo:hi]
        text = " ".join(x.text for x in window)
        speakers = {x.speaker_id for x in window}
        if u.component and u.component.text and not _contains(text, u.component.text):
            problems.append(f"{u.id}: equipment words '{u.component.text}' not found in its turns")
        if u.speaker_id not in speakers:
            problems.append(f"{u.id}: the update should be said by {onto.by_id[u.speaker_id].name}")
        if u.new_owner and u.new_owner.entity_id not in speakers and not _contains(text, u.new_owner.text or ""):
            problems.append(f"{u.id}: new owner '{u.new_owner.text}' not named and not a speaker")
        if u.kind == "due_changed" and u.due_date not in {resolve_date(x.text, day.date) for x in window}:
            problems.append(f"{u.id}: new deadline '{u.due_mention}' not clearly stated in its turns")
    return problems


def render_day_llm(day: StandupDay, onto: Ontology, max_attempts: int = 3) -> tuple[StandupDay, Usage]:
    rng = random.Random(day.id)
    usage = Usage()
    plan = json.dumps(_day_plan(day, onto), indent=1)
    writer = env("FOUNDRY_GEN_MODEL") or default_model()
    day.writer = writer
    feedback, problems = "", []
    for attempt in range(max_attempts):
        user_input = f"Plan:\n{plan}"
        if feedback:
            user_input += f"\n\nYour previous attempt had these problems, fix them:\n{feedback}"
        out, u = structured("standup_series_day", DAY_INSTRUCTIONS, user_input, _LLMDay, model=writer,
                            temperature=None)
        usage += u
        valid = set(day.attendees)
        rows = [(x.speaker_id if x.speaker_id in valid else day.chair_id, x.text) for x in out.utterances]
        day.utterances = _timed(rows, onto, rng)
        tev = {e.id: e.utterance_indices for e in out.task_evidence}
        uev = {e.id: e.utterance_indices for e in out.update_evidence}
        for t in day.gold_tasks:
            t.evidence = [i for i in tev.get(t.id, []) if 0 <= i < len(rows)]
        for up in day.updates:
            up.evidence = [i for i in uev.get(up.id, []) if 0 <= i < len(rows)]
        problems = check_day(day, onto)
        if not problems:
            day.renderer = "llm"
            day.generation_notes = [f"accepted on attempt {attempt + 1}"]
            return day, usage
        feedback = "\n".join(f"- {p}" for p in problems)
    day.renderer = "llm"
    day.generation_notes = ["FAILED validation", *problems]
    return day, usage
