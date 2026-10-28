"""The POC pipeline: extract what was said -> pull board context (code) -> pick the card -> fixed rules.

    Step 1  model  every job mention (new jobs AND updates), then the existing matcher links people/equipment
    Step 2  code   context_for(): the person's open cards + cards on the same equipment (board at start of stand-up)
    Step 3  model  for each mention with context, pick the card it is about from ITS list, or none
            rules  new + no match -> create      new + match -> comment (+ new due date / owner if given)
                   update + match -> update     update + no match -> ignore (never creates)

It produces the same ``DayRun`` as the whole-board pipeline (v0), so both are scored identically. With a
``BoardStore`` the board is read from and committed to the JSON database, one stand-up at a time.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, create_model

from ..extractors import TranscriptView
from ..llm import cost_usd, structured, trace_calls
from ..normalisers.base import NormContext
from ..normalisers.llm import LLMNormaliser
from ..ontology import Ontology
from ..schema import NormalisedTask, RawTask
from .context import context_for
from .pipeline import DayRun, StepCost
from .schema import BoardEvent, Card, StandupDay, apply

SAID = ("new", "progress", "done", "blocked", "cancelled", "reassigned", "due_changed")
SHORTLIST = 3  # matcher candidates per name / equipment used as Step 2 lookup keys
Category = Literal["operations", "safety", "organisation"]


# ------------------------------------------------------------------ step 1
class _Mention(BaseModel):
    said: Literal[SAID]  # type: ignore[valid-type]
    summary: str = Field(description="Short imperative description of the job being talked about.")
    speaker: str = Field(description="Display name of the person who said it, exactly as in the transcript.")
    owner_mention: str | None = Field(description="Whose job it is, as named ('I'll' = the speaker; 'Dave asked me to say' = Dave). Null if not said.")
    new_owner_mention: str | None = Field(description="reassigned only: who takes the job over or picks it up.")
    component_mention: str | None = Field(description="Equipment exactly as people referred to it; resolve 'it'. Null if none.")
    site_mention: str | None = Field(description="Off-site location, as said. Null if none.")
    due_mention: str | None = Field(description="Deadline words as spoken. Null if none.")
    due_date: str | None = Field(description="Deadline as YYYY-MM-DD using the meeting date and the date rules. Null if none.")
    due_time: str | None = Field(description="HH:MM (24h) only if a clock time was said, else null.")
    priority_mention: str | None = Field(description="Urgency words as spoken. Null if none.")
    priority: int | None = Field(description="new only: 1, 2 or 3 per the rubric; 2 when no urgency was expressed.")
    category: Category | None = Field(description="new only.")
    note: str | None = Field(description="Updates only: a few plain words for the card's history, e.g. 'waiting on the permit'.")
    evidence_quote: str = Field(description="The single transcript sentence it comes from, copied verbatim.")
    confidence: float = Field(description="0-1 confidence in this record.")


class _Mentions(BaseModel):
    mentions: list[_Mention]


MENTION_INSTRUCTIONS = """You extract every job mention from a factory morning stand-up transcript for a task board.

The board already holds jobs from earlier stand-ups, so people talk about NEW jobs and give UPDATES on
existing ones. Return one record per job mentioned, with `said`:
- new: a job agreed, asked for or needing doing (even with no owner yet)
- progress: someone worked on it but it is still open
- done: it is finished
- blocked: it cannot go on right now (put the reason in note)
- cancelled: it is not going to happen any more (put why in note)
- reassigned: it moves to someone else, or someone picks up an unowned job (set new_owner_mention)
- due_changed: a new deadline was given for an existing job (set due_mention)

Do NOT extract chit-chat or observations with no action ("it ran a bit warm but settled").
A remark that something is finished is `done` even if you do not know which job it was.
A job first raised in THIS meeting is always `new`, even if it is handed from one person to another or
someone declines it before another accepts ("Dave, can you...?" "Can't, I'm tied up." "Marc, take it?").
`reassigned`, `blocked` and the other update labels are only for jobs from earlier stand-ups.

Rules:
- One record per job per thing said. Merge the ask and the acceptance of the same new job into one record.
- If a job is handed from one person to another, the owner is whoever finally accepted it.
- speaker: the display name of whoever said it, exactly as in the transcript.
- Copy mentions verbatim; do not "correct" nicknames. Leave fields null when not said. Never guess.
- Date rules (relative to the meeting date): "today"/"this afternoon"/"end of day" = meeting date;
  "tomorrow" = +1 day; a bare weekday = its next occurrence after today; "next <weekday>" =
  that weekday in the following Monday-Sunday week; "end of the week" = this Friday;
  "next week" = next Monday; "the 3rd" = next occurrence of that day of month.
  Times: "half two" = 14:30, bare 1-6 o'clock = afternoon.

""" + Ontology.load().rubric()


# ------------------------------------------------------------------ step 3
PICK_INSTRUCTIONS = """You match things said at a factory stand-up to the cards on the team's task board.

For each mention you get only the cards it could be about: the person's open cards and cards on the
same equipment. Pick the card that is the SAME JOB, or "none".
- Same person or same equipment but a different job is not a match.
- A mention said as "new" matches only when it is clearly the same job said again.
- A remark about work that is not on any of its cards is "none".
- People use nicknames for equipment and colleagues.
- Give the reason first: one short sentence on why that card is the same job, or why none of them is."""


def _pick_model(ids: list[str]) -> type[BaseModel]:
    Pick = create_model("Pick", mention=(int, ...), reason=(str, ...),
                        card=(Literal[(*ids, "none")], ...))  # type: ignore[valid-type]
    return create_model("Picks", picks=(list[Pick], ...))


def _iso(v: str | None) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None


class ContextPipeline:
    """Same interface as ``BoardPipeline``: prepare(day) caches Step 1, run_series() chains the stand-ups."""

    def __init__(self, onto: Ontology, model: str):
        self.onto = onto
        self.model = model
        self.names = {p.id: p.name for p in onto.people}
        self.normaliser = LLMNormaliser(onto, None, model=model)
        self._step1: dict[str, tuple[list[dict], list[NormalisedTask], dict[str, StepCost]]] = {}
        self._traces: dict[str, list[dict]] = {}  # Step 1's model calls per stand-up (prompt, output, usage)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"context pipeline: mentions@{self.model} -> llm@{self.model} -> board context -> pick@{self.model}"

    # ---------------------------------------------------------------- step 1
    def _person(self, words: str | None, day: StandupDay) -> str | None:
        """Speaker display names map exactly; a first name or known alias of an attendee also counts."""
        if not words:
            return None
        w = words.strip().lower()
        for pid in (*day.attendees, *day.team, *self.names):
            p = self.onto.by_id[pid]
            if w == p.name.lower() or w in (a.lower() for a in p.aliases) or w == p.name.split()[0].lower():
                return pid
        return None

    def prepare(self, day: StandupDay):
        """Step 1 for one stand-up: mentions + the existing matcher. The board does not affect it."""
        with self._lock:
            if day.id in self._step1:
                return self._step1[day.id]
        view = TranscriptView.of(day, self.names)
        header = f"Meeting date: {day.date.isoformat()} ({day.date:%A})\nAttendees: {', '.join(view.attendees)}\n\nTranscript:\n{view.text}"
        with trace_calls() as calls:
            out, u1 = structured("standup_mentions", MENTION_INSTRUCTIONS, header, _Mentions, model=self.model)
            raws = [RawTask(summary=m.summary, owner_mention=m.owner_mention, component_mention=m.component_mention,
                            site_mention=m.site_mention, due_mention=m.due_mention, due_date=_iso(m.due_date),
                            due_time=m.due_time, priority_mention=m.priority_mention, priority=m.priority,
                            category=m.category, evidence_quote=m.evidence_quote, confidence=m.confidence)
                    for m in out.mentions]
            extra = {i: len(raws) + k for k, i in enumerate(i for i, m in enumerate(out.mentions) if m.new_owner_mention)}
            raws += [RawTask(summary=out.mentions[i].summary, owner_mention=out.mentions[i].new_owner_mention) for i in extra]
            ctx = NormContext(day.id, day.date, tuple(day.attendees))
            linked = self.normaliser.normalise_meeting(raws, ctx) if raws else []
        tin, tout = self.normaliser.tokens_by_meeting.get(day.id, (0, 0))
        top = lambda kind, words: self.normaliser.shortlist_ids(kind, words, ctx)[:SHORTLIST] if words else []  # noqa: E731
        rows = []
        for i, (m, t) in enumerate(zip(out.mentions, linked)):
            speaker = self._person(m.speaker, day)
            rows.append({
                "mention_id": f"{day.id}-m{i + 1}", "standup_id": day.id, "index": i, "said": m.said,
                "summary": m.summary, "speaker_id": speaker, "owner_id": t.owner_id or (speaker if m.said != "new" else None),
                "new_owner_id": linked[extra[i]].owner_id if i in extra else None,
                "component_id": t.component_id, "site_id": t.site_id, "due_date": t.due_date, "due_time": t.due_time,
                "priority": t.priority, "category": t.category, "note": m.note, "quote": m.evidence_quote,
                "confidence": m.confidence,
                "owner_words": m.owner_mention, "component_words": m.component_mention,
                "new_owner_words": m.new_owner_mention, "due_words": m.due_mention,
                "people_candidates": top("person", m.owner_mention) + top("person", m.new_owner_mention),
                "equipment_candidates": top("asset", m.component_mention),
            })
        steps = {
            "extract": StepCost(u1.input_tokens, u1.output_tokens, u1.seconds, cost_usd(u1.input_tokens, u1.output_tokens, self.model)),
            "match": StepCost(tin, tout, self.normaliser.seconds_by_meeting.get(day.id, 0.0), cost_usd(tin, tout, self.model)),
        }
        with self._lock:
            self._step1[day.id] = (rows, linked[:len(out.mentions)], steps)
            self._traces[day.id] = [{**c, "step": "extract" if k == 0 else "match"} for k, c in enumerate(calls)]
        return self._step1[day.id]

    # ---------------------------------------------------------------- step 3
    def _pick(self, day: StandupDay, mentions: list[dict], contexts: list[list[Card]]) -> tuple[dict[int, str], dict[int, str], StepCost]:
        asks = [i for i, c in enumerate(contexts) if c]
        if not asks:
            return {}, {}, StepCost()
        name = lambda pid: self.names.get(pid) if pid else None  # noqa: E731
        asset = lambda aid: self.onto.by_id[aid].name if aid else None  # noqa: E731
        payload = [{
            "mention": i, "said": mentions[i]["said"], "job": mentions[i]["summary"], "speaker": name(mentions[i]["speaker_id"]),
            "owner": name(mentions[i]["owner_id"]), "equipment": asset(mentions[i]["component_id"]),
            "quote": mentions[i]["quote"],
            "cards": [{"card": c.id, "job": c.summary, "owner": name(c.owner_id) or "unassigned",
                       "equipment": asset(c.component_id), "status": c.status,
                       "due": c.due_date.isoformat() if c.due_date else None} for c in contexts[i]],
        } for i in asks]
        ids = sorted({c.id for i in asks for c in contexts[i]})
        out, u = structured("board_pick", PICK_INSTRUCTIONS, json.dumps(payload, indent=1), _pick_model(ids), model=self.model)
        allowed = {i: {c.id for c in contexts[i]} for i in asks}
        picks = {p.mention: p.card for p in out.picks if p.card != "none" and p.card in allowed.get(p.mention, ())}
        reasons = {p.mention: p.reason for p in out.picks if p.mention in allowed}
        return picks, reasons, StepCost(u.input_tokens, u.output_tokens, u.seconds, cost_usd(u.input_tokens, u.output_tokens, self.model))

    # ---------------------------------------------------------------- one stand-up
    def run_day(self, day: StandupDay, board_before: list[Card], mode: str = "chained") -> DayRun:
        mentions, linked, steps = self.prepare(day)
        contexts = [context_for(board_before, self.onto,
                                {m["speaker_id"], m["owner_id"], m["new_owner_id"], *m.get("people_candidates", [])},
                                [m["component_id"], *m.get("equipment_candidates", [])],
                                include_unowned=m["said"] == "reassigned", words=m["summary"])
                    for m in mentions]
        with trace_calls() as pick_calls:
            picks, reasons, pick_cost = self._pick(day, mentions, contexts)
        traces = [*getattr(self, "_traces", {}).get(day.id, []), *({**c, "step": "pick"} for c in pick_calls)]
        cards = {c.id: c for c in board_before}
        new_jobs: list[NormalisedTask] = []
        merged: dict[int, str] = {}
        events: list[BoardEvent] = []
        decisions = []
        # each card's due date and owner as this stand-up changes them, and its last change (to skip exact repeats)
        now = {c.id: {"due": (c.due_date, c.due_time), "owner": c.owner_id} for c in board_before}
        last: dict[str, tuple] = {}

        def emit(**kw) -> int:
            events.append(BoardEvent(day_id=day.id, **kw))
            return len(events) - 1

        for i, (m, t) in enumerate(zip(mentions, linked)):
            card = picks.get(i)
            base = dict(speaker_id=m["speaker_id"], quote=m["quote"], confidence=m["confidence"], item_index=i)
            ev: list[int] = []
            if m["said"] == "new":
                k = len(new_jobs)
                new_jobs.append(t)
                if card is None:
                    card_id, action, rule = f"{day.id}-p{k + 1}", "create", "said new + no match → create"
                    ev.append(emit(kind="create", card_id=card_id, summary=t.summary, owner_id=t.owner_id,
                                   component_id=t.component_id, site_id=t.site_id, due_date=t.due_date,
                                   due_time=t.due_time, priority=t.priority, category=t.category, **base))
                else:
                    merged[k] = card
                    card_id, action, rule = card, "comment", "said new + match → comment, no duplicate"
                    ev.append(emit(kind="comment", card_id=card, note=f"Mentioned again: {t.summary}", **base))
                    if t.due_date and (t.due_date, t.due_time) != now[card]["due"]:
                        ev.append(emit(kind="due_changed", card_id=card, due_date=t.due_date, due_time=t.due_time,
                                       note="New due date when mentioned again", **base))
                        now[card]["due"] = (t.due_date, t.due_time)
                        rule += " + new due date"
                    if t.owner_id and t.owner_id != now[card]["owner"]:
                        ev.append(emit(kind="reassigned", card_id=card, owner_id=t.owner_id,
                                       note="New owner when mentioned again", **base))
                        now[card]["owner"] = t.owner_id
                        rule += " + new owner"
            elif card is None:
                card_id, action, rule = None, "ignore", "update + no match → ignore (never creates)"
            elif last.get(card) == (key := (m["said"], m["new_owner_id"], t.due_date, t.due_time, m["note"])):
                card_id, action, rule = card, "ignore", "same change as the one just made to this card"
            else:
                last[card] = key
                kind = m["said"]
                card_id, action, rule = card, "update", "update + match → update that card"
                if kind == "reassigned" and not m["new_owner_id"]:
                    kind, rule = "comment", "reassigned but new owner unclear → comment"
                if kind == "due_changed" and not t.due_date:
                    kind, rule = "comment", "new date unclear → comment"
                ev.append(emit(kind=kind, card_id=card, note=m["note"], owner_id=m["new_owner_id"] if kind == "reassigned" else None,
                               due_date=t.due_date if kind == "due_changed" else None,
                               due_time=t.due_time if kind == "due_changed" else None, **base))
                if kind == "due_changed":
                    now[card]["due"] = (t.due_date, t.due_time)
                elif kind == "reassigned":
                    now[card]["owner"] = m["new_owner_id"]
            decisions.append({"decision_id": f"{day.id}-d{i + 1}", "standup_id": day.id, "mention_id": m["mention_id"],
                              "mention": i, "context_card_ids": [c.id for c in contexts[i]], "matched_card_id": card,
                              "pick_reason": reasons.get(i), "action": action, "rule": rule, "card_id": card_id, "events": ev})
        return DayRun(day.id, mode, new_jobs, merged, events, board_before, apply(board_before, events, day),
                      {**steps, "pick": pick_cost}, why={"mentions": mentions, "decisions": decisions, "traces": traces})

    # ---------------------------------------------------------------- the loop
    def run_series(self, days: list[StandupDay], mode: str, store=None) -> list[DayRun]:
        runs: list[DayRun] = []
        board: list[Card] = []
        for day in days:
            if mode == "isolated":
                before = [c.model_copy(deep=True) for c in day.board_before]
            else:
                before = store.board(day.series_id) if store else [c.model_copy(deep=True) for c in board]
            t0 = time.perf_counter()
            try:
                run = self.run_day(day, before, mode)
            except Exception as exc:  # noqa: BLE001 - a failed day keeps yesterday's board
                run = DayRun(day.id, mode, [], {}, [], before, before, error=str(exc)[:300],
                             why={"mentions": [], "decisions": []})  # its true jobs still count as missed
            runs.append(run)
            board = run.board_after
            if store is not None and mode == "chained":
                cost = sum((s.cost_usd or 0) for s in run.steps.values())
                store.commit(day, run.board_after, run.why, {
                    "status": "failed" if run.error else "done", "error": run.error, "model": self.model,
                    "cost_usd": round(cost, 6), "seconds": round(time.perf_counter() - t0, 2)})
        return runs
