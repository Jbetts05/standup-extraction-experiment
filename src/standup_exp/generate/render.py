"""Turn a sampled answer key into a Teams-style stand-up transcript.

Two renderers share one contract: given a ``Meeting`` with gold tasks and
distractors, fill ``meeting.utterances`` and each task's ``evidence`` indices.

* ``template`` - deterministic, offline, no LLM. Stilted but perfectly aligned.
  Use it for tests, the offline demo and a quick sanity check of every method.
* ``llm``      - asks the Foundry model to write a natural, messy conversation
  that must contain every planned task and distractor. The result is then
  checked by ``validate.check_meeting`` and regenerated with feedback on failure.
"""

from __future__ import annotations

import json
import random

from pydantic import BaseModel

from ..llm import Usage, default_model, structured
from ..ontology import Ontology
from ..paths import env
from ..schema import GoldTask, Meeting, Utterance
from .validate import check_meeting


# ------------------------------------------------------------------ timestamps
def _timed(rows: list[tuple[str, str]], onto: Ontology, rng: random.Random) -> list[Utterance]:
    """Assign plausible Teams timestamps: ~0.35s per word plus a short gap."""
    t = rng.uniform(2.0, 6.0)
    out = []
    for speaker_id, text in rows:
        duration = max(1.2, len(text.split()) * 0.35)
        out.append(
            Utterance(start=round(t, 2), end=round(t + duration, 2), speaker_id=speaker_id,
                      speaker=onto.by_id[speaker_id].name, text=text)
        )
        t += duration + rng.uniform(0.3, 2.5)
    return out


# -------------------------------------------------------------------- template
def render_template(meeting: Meeting, onto: Ontology) -> Meeting:
    rng = random.Random(meeting.id)
    rows: list[tuple[str, str]] = []
    evidence: dict[str, list[int]] = {t.id: [] for t in meeting.gold_tasks}
    chair = meeting.chair_id

    def say(speaker: str, text: str, task: GoldTask | None = None):
        if task is not None:
            evidence[task.id].append(len(rows))
        rows.append((speaker, text))

    def when(t: GoldTask) -> str:
        parts = [p for p in (t.due_mention, t.priority_cue) if p]
        return (" " + ", ".join(parts)) if parts else ""

    say(chair, rng.choice(["Morning all, let's get going.", "Right, morning everyone, quick round the room.",
                           "Okay, can everyone hear me? Let's crack on."]))
    for d in [d for d in meeting.distractors if d.kind == "chitchat"]:
        say(d.speaker_id, f"Before we start, did anyone see... well, {d.text_hint}, honestly.")
        say(rng.choice([p for p in meeting.attendees if p != d.speaker_id]), rng.choice(
            ["Ha, yeah.", "Don't get me started.", "Tell me about it.", "Classic."]))

    by_person: dict[str, list] = {p: [] for p in meeting.attendees}
    for d in meeting.distractors:
        if d.kind != "chitchat":
            by_person[d.speaker_id].append(d)

    ownerless = [t for t in meeting.gold_tasks if t.owner.entity_id is None]
    for person in meeting.attendees:
        for d in by_person[person]:
            lead = {"done": "Yesterday I", "observation": "Just so you know, I", "blocker": "Blocker from me,",
                    "cancelled": "Oh and"}[d.kind]
            say(person, f"{lead} {d.text_hint}.")
        for t in meeting.gold_tasks:
            if t.owner.entity_id != person:
                continue
            if t.reassigned_from:
                say(chair, f"{onto.by_id[t.reassigned_from].aliases[0]}, can you {t.spoken_action}{when(t)}?", t)
                say(t.reassigned_from, "I'm tied up at Northfield all day, sorry.", t)
                say(chair, f"No worries. {t.owner.text}, can you take that one instead?", t)
                say(person, "Yeah, I'll take it.", t)
            elif t.assigned_by:
                say(chair, f"{t.owner.text}, can you {t.spoken_action}{when(t)}?", t)
                say(person, rng.choice(["Yep, will do.", "No problem.", "Can do."]), t)
            else:
                say(person, f"{rng.choice(['Next up, I', 'From me, I', 'I'])}'ll {t.spoken_action}{when(t)}.", t)
    for t in ownerless:
        say(chair, f"Someone needs to {t.spoken_action}{when(t)}. We'll pick an owner later.", t)
    say(chair, "Great, thanks all. Same time tomorrow.")

    meeting.utterances = _timed(rows, onto, rng)
    for t in meeting.gold_tasks:
        t.evidence = evidence[t.id]
    meeting.renderer = "template"
    return meeting


# ------------------------------------------------------------------------- LLM
class _LLMUtterance(BaseModel):
    speaker_id: str
    text: str


class _TaskEvidence(BaseModel):
    task_id: str
    utterance_indices: list[int]


class _LLMTranscript(BaseModel):
    utterances: list[_LLMUtterance]
    task_evidence: list[_TaskEvidence]


RENDER_INSTRUCTIONS = """You write realistic Microsoft Teams transcripts of a daily morning stand-up at a UK factory.

The plan you receive lists attendees, the tasks that must be agreed in the meeting, and
distractors (chit-chat, finished work, observations, blockers, cancelled work).

Write the conversation so that:
- It sounds like real engineers and operators talking: informal British English, filler words
  ("um", "yeah so", "right"), interruptions, people talking over each other, a bit of banter,
  engineering jargon, half sentences. About 50-90 turns.
- Every planned task is clearly agreed. Use the exact given words for each component,
  site and person reference (these are the nicknames people really use). Put the due phrase
  and priority cue in naturally; you may reword slightly, but the meaning of the deadline and
  urgency must stay exactly the same. Tasks may be spread over several turns (asked, discussed,
  agreed).
- Tasks marked "volunteered" are stated by the owner in first person ("I'll ...").
  Tasks with "assigned_by" are asked by that person and accepted by the owner.
  Tasks with "reassigned_from" are first given to that person, who declines, then handed to the owner.
  Tasks with owner "nobody" are raised as needing doing but nobody takes them.
- Every distractor appears, and it is obvious it is NOT a new task for today
  (already done, just an observation, blocked, or cancelled).
- Do not invent extra tasks, deadlines or owners beyond the plan.
- speaker_id must be one of the attendee ids.

Also return task_evidence: for every task id, the 0-based indices of the turns where it is
raised, assigned or accepted."""


def _plan(meeting: Meeting, onto: Ontology) -> dict:
    name = lambda pid: onto.by_id[pid].name  # noqa: E731
    return {
        "meeting_date": meeting.date.strftime("%A %d %B %Y"),
        "chair": {"id": meeting.chair_id, "name": name(meeting.chair_id)},
        "attendees": [{"id": p, "name": name(p), "role": onto.by_id[p].extra.get("role")} for p in meeting.attendees],
        "tasks": [
            {
                "task_id": t.id,
                "action": t.spoken_action,
                "owner": ({"id": t.owner.entity_id, "refer_to_as": t.owner.text} if t.owner.entity_id else "nobody"),
                "mode": ("reassigned" if t.reassigned_from else "assigned" if t.assigned_by else "volunteered"),
                "assigned_by": t.assigned_by,
                "reassigned_from": t.reassigned_from,
                "component_words": t.component.text,
                "site_words": t.site.text,
                "due_phrase": t.due_mention,
                "priority_cue": t.priority_cue,
            }
            for t in meeting.gold_tasks
        ],
        "distractors": [{"kind": d.kind, "speaker_id": d.speaker_id, "content": d.text_hint} for d in meeting.distractors],
    }


def render_llm(meeting: Meeting, onto: Ontology, max_attempts: int = 3) -> tuple[Meeting, Usage]:
    rng = random.Random(meeting.id)
    usage = Usage()
    plan = json.dumps(_plan(meeting, onto), indent=1)
    writer = env("FOUNDRY_GEN_MODEL") or default_model()
    meeting.writer = writer
    feedback = ""
    for attempt in range(max_attempts):
        user_input = f"Plan:\n{plan}"
        if feedback:
            user_input += f"\n\nYour previous attempt had these problems, fix them:\n{feedback}"
        out, u = structured(
            "standup_transcript", RENDER_INSTRUCTIONS, user_input, _LLMTranscript,
            model=writer, temperature=None,
        )
        usage += u
        valid_ids = set(meeting.attendees)
        rows = [(u_.speaker_id if u_.speaker_id in valid_ids else meeting.chair_id, u_.text) for u_ in out.utterances]
        meeting.utterances = _timed(rows, onto, rng)
        ev = {e.task_id: e.utterance_indices for e in out.task_evidence}
        for t in meeting.gold_tasks:
            t.evidence = [i for i in ev.get(t.id, []) if 0 <= i < len(rows)]
        problems = check_meeting(meeting, onto)
        if not problems:
            meeting.renderer = "llm"
            meeting.generation_notes = [f"accepted on attempt {attempt + 1}"]
            return meeting, usage
        feedback = "\n".join(f"- {p}" for p in problems)
    meeting.renderer = "llm"
    meeting.generation_notes = ["FAILED validation"] + problems
    return meeting, usage
