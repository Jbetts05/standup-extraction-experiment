"""The board loop: each stand-up turns the board it inherited into the next day's board.

    transcript --[1. existing pipeline]--> new jobs (LLMExtractor -> LLMNormaliser, unchanged)
         |                                     |
         +--[2. update check]  open cards + transcript + new jobs
                  -> updates on open cards (done / progress / blocked / cancelled / reassigned / due_changed)
                  -> new jobs that are really an open card said again (merged, not created)
    board after = apply(board before, create events + update events)   --> input to tomorrow

Step 2 never sees card ids it could invent: ``card_id``, ``speaker_id`` and ``new_owner_id`` are
enums of the ids that exist, built per call into the strict JSON schema.

Modes
    chained   - tomorrow starts from *our* board, so mistakes carry forward (the real thing; default)
    isolated  - every day starts from the true board, so each day's errors are measured alone (diagnostic)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, create_model

from ..dates import resolve_date, resolve_time
from ..extractors import TranscriptView
from ..extractors.llm import LLMExtractor
from ..llm import Usage, cost_usd, structured
from ..normalisers.base import NormContext
from ..normalisers.llm import LLMNormaliser
from ..ontology import Ontology
from ..schema import NormalisedTask
from .schema import UPDATE_KINDS, BoardEvent, Card, StandupDay, apply

MODES = ("chained", "isolated")

UPDATE_INSTRUCTIONS = """You keep a team's task board up to date from its daily factory stand-up.

You get the board's OPEN cards (jobs agreed at earlier stand-ups), the team roster, today's
transcript, and the NEW jobs another step already extracted from this transcript.

1. For every open card that today's meeting talks about, report what happened to it:
   - done: it is finished.
   - progress: someone worked on it but it is still open. Put what they said in note.
   - blocked: it cannot go on right now. Put the reason in note.
   - cancelled: it is not going to happen any more. Put why in note.
   - reassigned: it moves to another person, or someone picks up an unowned job. Set new_owner_id.
   - due_changed: a new deadline was given. Copy the deadline words to due_mention and resolve due_date.
   A card may get more than one update (for example reassigned and due_changed). Cards that
   nobody mentions get nothing.
2. Only link words to a card when they clearly refer to that job. Several cards can share the same
   equipment, so decide by the job itself. People use nicknames for equipment and colleagues, and
   may report for someone else ("Dave asked me to say...").
3. Remarks about finished work, observations or chit-chat on equipment or jobs with no open card are
   NOT updates. "Nearly done" or "will finish today" is progress, not done. Only mark done or
   cancelled when the meeting clearly says so.
4. For each new job, set same_as_card to the open card it restates (the same job said again), or
   "none" if it really is new today.
5. note: one short plain sentence for the card's activity feed. evidence_quote: the transcript
   sentence the update comes from, copied verbatim. confidence: 0-1 that the update is right.

Date rules (relative to the meeting date): "today" = meeting date; "tomorrow" = +1 day; a bare
weekday = its next occurrence after today; "next <weekday>" = that weekday in the following
Monday-Sunday week; "end of the week" = this Friday; "next week" = next Monday; "the 3rd" = next
occurrence of that day of month. Times: "half two" = 14:30, bare 1-6 o'clock = afternoon."""


def _update_models(card_ids: list[str], team_ids: list[str], n_new: int) -> type[BaseModel]:
    CardId = Literal[tuple(card_ids)]  # type: ignore[valid-type]
    PersonId = Literal[tuple(team_ids)]  # type: ignore[valid-type]
    Kind = Literal[UPDATE_KINDS]  # type: ignore[valid-type]
    Update = create_model(
        "CardUpdate",
        card_id=(CardId, Field(description="The open card this update is about.")),
        kind=(Kind, ...),
        speaker_id=(PersonId | None, Field(description="Who said it.")),
        note=(str | None, Field(description="Short plain sentence for the card's activity feed.")),
        new_owner_id=(PersonId | None, Field(description="reassigned only: who takes the job.")),
        due_mention=(str | None, Field(description="due_changed only: the new deadline words as spoken.")),
        due_date=(str | None, Field(description="due_changed only: YYYY-MM-DD.")),
        due_time=(str | None, Field(description="due_changed only: HH:MM if a clock time was said.")),
        evidence_quote=(str, ...),
        confidence=(float, ...),
    )
    fields = {"updates": (list[Update], ...)}
    if n_new:
        Same = create_model("NewJobCheck", new_job_index=(int, ...),
                            same_as_card=(Literal[(*card_ids, "none")], ...))  # type: ignore[valid-type]
        fields["new_jobs"] = (list[Same], ...)
    return create_model("BoardUpdates", **fields)


def _iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


@dataclass
class StepCost:
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float | None = 0.0


@dataclass
class DayRun:
    day_id: str
    mode: str
    new_jobs: list[NormalisedTask]
    merged: dict[int, str]  # new job index -> open card it restates (not created)
    events: list[BoardEvent]
    board_before: list[Card]
    board_after: list[Card]
    steps: dict[str, StepCost] = field(default_factory=dict)
    error: str | None = None
    why: dict = field(default_factory=dict)  # context pipeline: mentions + decisions, for the UI and per-step scores


class BoardPipeline:
    """Step 1 = the experiment's existing extractor + matcher; step 2 = the update check."""

    def __init__(self, onto: Ontology, model: str):
        self.onto = onto
        self.model = model
        self.names = {p.id: p.name for p in onto.people}
        self.extractor = LLMExtractor(model)
        self.normaliser = LLMNormaliser(onto, None, model=model)
        self._new: dict[str, tuple[list[NormalisedTask], dict[str, StepCost]]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.extractor.name} -> {self.normaliser.name} + update@{self.model}"

    def prepare(self, day: StandupDay):
        return self.new_jobs(day)

    # ------------------------------------------------------------------ step 1
    def new_jobs(self, day: StandupDay) -> tuple[list[NormalisedTask], dict[str, StepCost]]:
        """New jobs from the existing pipeline. The board does not affect this, so it runs once per day."""
        with self._lock:
            if day.id in self._new:
                return self._new[day.id]
        ex = self.extractor.extract(TranscriptView.of(day, self.names))
        ctx = NormContext(day.id, day.date, tuple(day.attendees))
        t0 = time.perf_counter()
        tasks = self.normaliser.normalise_meeting(ex.tasks, ctx)
        tin, tout = self.normaliser.tokens_by_meeting.get(day.id, (0, 0))
        secs = self.normaliser.seconds_by_meeting.get(day.id, time.perf_counter() - t0)
        steps = {
            "extract": StepCost(ex.input_tokens, ex.output_tokens, ex.seconds,
                                cost_usd(ex.input_tokens, ex.output_tokens, self.model)),
            "match": StepCost(tin, tout, secs, cost_usd(tin, tout, self.model)) if ex.tasks else StepCost(),
        }
        with self._lock:
            self._new[day.id] = (tasks, steps)
        return tasks, steps

    # ------------------------------------------------------------------ step 2
    def check_updates(self, day: StandupDay, board: list[Card], new_jobs: list[NormalisedTask]
                      ) -> tuple[list[BoardEvent], dict[int, str], StepCost]:
        open_cards = sorted((c for c in board if c.is_open), key=lambda c: (c.created_on, c.id))
        if not open_cards:
            return [], {}, StepCost()
        name = lambda pid: self.names.get(pid) if pid else None  # noqa: E731
        by_id = self.onto.by_id
        team = day.team or day.attendees
        payload = {
            "meeting_date": f"{day.date.isoformat()} ({day.date:%A})",
            "team": [{"id": p, "name": name(p), "aliases": list(by_id[p].aliases), "at_meeting": p in day.attendees}
                     for p in team],
            "open_cards": [
                {
                    "card_id": c.id,
                    "job": c.summary,
                    "owner": name(c.owner_id) or "unassigned",
                    "equipment": by_id[c.component_id].name if c.component_id else None,
                    "part_of": by_id[p].name if c.component_id and (p := by_id[c.component_id].parent) else None,
                    "site": by_id[c.site_id].name if c.site_id else None,
                    "due": c.due_date.isoformat() if c.due_date else None,
                    "status": c.status,
                    "added": c.created_on.isoformat(),
                    "latest_note": c.history[-1].text if len(c.history) > 1 else None,
                }
                for c in open_cards
            ],
            "new_jobs_found_today": [
                {"new_job_index": i, "job": t.summary, "owner": name(t.owner_id),
                 "equipment": by_id[t.component_id].name if t.component_id else None, "evidence": t.evidence_quote}
                for i, t in enumerate(new_jobs)
            ],
        }
        view = TranscriptView.of(day, self.names)
        user_input = (f"{json.dumps(payload, indent=1)}\n\nTranscript (meeting date {day.date.isoformat()}, "
                      f"{day.date:%A}):\n{view.text}")
        model = _update_models([c.id for c in open_cards], list(team), len(new_jobs))
        out, usage = structured("board_updates", UPDATE_INSTRUCTIONS, user_input, model, model=self.model)

        merged: dict[int, str] = {}
        for s in getattr(out, "new_jobs", []) or []:
            if s.same_as_card != "none" and 0 <= s.new_job_index < len(new_jobs):
                merged[s.new_job_index] = s.same_as_card
        events: list[BoardEvent] = []
        seen: set[tuple[str, str]] = set()
        for u in out.updates:
            if (u.card_id, u.kind) in seen:
                continue
            owner = u.new_owner_id if u.kind == "reassigned" else None
            due = (_iso(u.due_date) or resolve_date(u.due_mention, day.date)) if u.kind == "due_changed" else None
            if (u.kind == "reassigned" and not owner) or (u.kind == "due_changed" and not due):
                continue
            seen.add((u.card_id, u.kind))
            events.append(BoardEvent(
                day_id=day.id, kind=u.kind, card_id=u.card_id, speaker_id=u.speaker_id, note=u.note,
                owner_id=owner, due_date=due,
                due_time=(u.due_time or resolve_time(u.due_mention)) if u.kind == "due_changed" else None,
                quote=u.evidence_quote, confidence=max(0.0, min(1.0, u.confidence)),
            ))
        cost = StepCost(usage.input_tokens, usage.output_tokens, usage.seconds,
                        cost_usd(usage.input_tokens, usage.output_tokens, self.model))
        return events, merged, cost

    # --------------------------------------------------------------- the loop
    def run_day(self, day: StandupDay, board_before: list[Card], mode: str) -> DayRun:
        new_jobs, steps = self.new_jobs(day)
        updates, merged, update_cost = self.check_updates(day, board_before, new_jobs)
        creates = [
            BoardEvent(
                day_id=day.id, kind="create", card_id=f"{day.id}-p{i + 1}", speaker_id=t.owner_id,
                summary=t.summary, owner_id=t.owner_id, component_id=t.component_id, site_id=t.site_id,
                due_date=t.due_date, due_time=t.due_time, priority=t.priority, category=t.category,
                quote=t.evidence_quote, confidence=t.raw.confidence, item_index=i,
            )
            for i, t in enumerate(new_jobs) if i not in merged
        ]
        updated = {e.card_id for e in updates}
        comments = [
            BoardEvent(day_id=day.id, kind="comment", card_id=card, note=f"Mentioned again: {new_jobs[i].summary}",
                       quote=new_jobs[i].evidence_quote, item_index=i)
            for i, card in merged.items() if card not in updated
        ]
        events = creates + updates + comments
        return DayRun(day.id, mode, new_jobs, merged, events, board_before, apply(board_before, events, day),
                      {**steps, "update": update_cost})

    def run_series(self, days: list[StandupDay], mode: str, store=None) -> list[DayRun]:
        runs: list[DayRun] = []
        board: list[Card] = []
        for day in days:
            before = [c.model_copy(deep=True) for c in (day.board_before if mode == "isolated" else board)]
            try:
                run = self.run_day(day, before, mode)
            except Exception as exc:  # noqa: BLE001 - a failed day keeps yesterday's board
                run = DayRun(day.id, mode, [], {}, [], before, before, error=str(exc)[:300])
            runs.append(run)
            board = run.board_after
        return runs


def total_cost(runs: list[DayRun]) -> dict:
    out: dict[str, StepCost] = {}
    for r in runs:
        for k, s in r.steps.items():
            t = out.setdefault(k, StepCost())
            t.input_tokens += s.input_tokens
            t.output_tokens += s.output_tokens
            t.seconds += s.seconds
            t.cost_usd = None if t.cost_usd is None or s.cost_usd is None else t.cost_usd + s.cost_usd
    return {k: v.__dict__ for k, v in out.items()}


__all__ = ["BoardPipeline", "DayRun", "MODES", "total_cost", "Usage"]
