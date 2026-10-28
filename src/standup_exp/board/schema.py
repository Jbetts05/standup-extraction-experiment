"""Board data models: cards that persist across stand-ups, and the events that change them.

    StandupDay (a Meeting that knows the board before it)
       gold_tasks -> new cards agreed today (the existing answer-key format)
       updates    -> GoldUpdate on cards from earlier days
    board after the day = apply(board before, events)

Predictions use the same ``BoardEvent`` shape, so gold and predicted boards are built by the
same ``apply`` and can be compared card by card.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ..schema import Meeting, Mention

Status = Literal["todo", "in_progress", "blocked", "done", "cancelled"]
UpdateKind = Literal["progress", "blocked", "done", "cancelled", "reassigned", "due_changed"]
# "comment" adds to a card's history without changing it (e.g. a new job that restates an open card).
EventKind = Literal["create", "progress", "blocked", "done", "cancelled", "reassigned", "due_changed", "comment"]
UPDATE_KINDS: tuple[str, ...] = ("progress", "blocked", "done", "cancelled", "reassigned", "due_changed")
OPEN: frozenset[str] = frozenset({"todo", "in_progress", "blocked"})
CLOSING: frozenset[str] = frozenset({"done", "cancelled"})
STATUS_AFTER = {"progress": "in_progress", "blocked": "blocked", "done": "done", "cancelled": "cancelled"}
DEFAULT_NOTE = {
    "create": "Added from the stand-up",
    "progress": "Progress update",
    "blocked": "Blocked",
    "done": "Marked done",
    "cancelled": "Cancelled",
    "reassigned": "Reassigned",
    "due_changed": "Due date changed",
    "comment": "Mentioned again",
}


class Comment(BaseModel):
    """One entry in a card's activity feed, always traceable to a stand-up."""

    day_id: str
    date: date
    kind: EventKind
    text: str
    quote: str | None = None
    speaker_id: str | None = None
    confidence: float | None = None


class Card(BaseModel):
    id: str
    summary: str
    category: str | None = None
    priority: int | None = None
    owner_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None
    due_date: date | None = None
    due_time: str | None = None
    status: Status = "todo"
    created_in: str
    created_on: date
    updated_in: str | None = None
    history: list[Comment] = Field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status in OPEN


class GoldUpdate(BaseModel):
    """Something said today about a card from an earlier stand-up."""

    id: str
    kind: UpdateKind
    card_id: str
    speaker_id: str
    style: Literal["full", "activity"] = "full"  # full = names the equipment; activity = only what the job is
    reference: str = ""  # how the job is referred to aloud, e.g. "the vibration readings on the old turbine"
    component: Mention | None = None  # equipment words to use when style == "full"
    owner_words: str | None = None  # how the card's owner is referred to when someone else reports on it
    note: str | None = None  # progress detail, blocker reason or why it was cancelled
    new_owner: Mention | None = None  # reassigned
    due_mention: str | None = None  # due_changed
    due_date: date | None = None
    due_time: str | None = None
    tags: list[str] = Field(default_factory=list)  # hard-case labels for the breakdown
    evidence: list[int] = Field(default_factory=list)


class StandupDay(Meeting):
    series_id: str
    day: int  # 1-based position in the series
    team: list[str] = Field(default_factory=list)  # everyone on the team, present or not
    board_before: list[Card] = Field(default_factory=list)
    updates: list[GoldUpdate] = Field(default_factory=list)


class BoardEvent(BaseModel):
    """One change to the board. Gold and predicted events share this shape."""

    day_id: str
    kind: EventKind
    card_id: str
    speaker_id: str | None = None
    summary: str | None = None  # create only
    owner_id: str | None = None  # create, or the new owner for reassigned
    component_id: str | None = None
    site_id: str | None = None
    due_date: date | None = None  # create, or the new date for due_changed
    due_time: str | None = None
    priority: int | None = None
    category: str | None = None
    note: str | None = None
    quote: str | None = None
    confidence: float | None = None
    gold_id: str | None = None  # gold task / update id (gold events only)
    item_index: int | None = None  # extractor item it came from (predicted events only)


def apply(board: list[Card], events: list[BoardEvent], day: Meeting) -> list[Card]:
    """Return the board after ``events``; ``board`` is not modified."""
    cards = {c.id: c.model_copy(deep=True) for c in board}
    for e in events:
        comment = Comment(day_id=day.id, date=day.date, kind=e.kind, text=e.note or DEFAULT_NOTE[e.kind],
                          quote=e.quote, speaker_id=e.speaker_id, confidence=e.confidence)
        if e.kind == "create":
            cards[e.card_id] = Card(
                id=e.card_id, summary=e.summary or "", category=e.category, priority=e.priority,
                owner_id=e.owner_id, component_id=e.component_id, site_id=e.site_id,
                due_date=e.due_date, due_time=e.due_time, created_in=day.id, created_on=day.date,
                updated_in=day.id, history=[comment],
            )
            continue
        c = cards.get(e.card_id)
        if c is None:
            continue
        if e.kind in STATUS_AFTER:
            c.status = STATUS_AFTER[e.kind]
        elif e.kind == "reassigned":
            c.owner_id = e.owner_id
        elif e.kind == "due_changed":
            c.due_date, c.due_time = e.due_date, e.due_time
        c.updated_in = day.id
        c.history.append(comment)
    return list(cards.values())


def gold_events(day: StandupDay) -> list[BoardEvent]:
    """The answer key for one day as board events: creates first, then updates."""
    events = [
        BoardEvent(
            day_id=day.id, kind="create", card_id=t.id, speaker_id=t.owner.entity_id, summary=t.summary,
            owner_id=t.owner.entity_id, component_id=t.component.entity_id, site_id=t.site.entity_id,
            due_date=t.due_date, due_time=t.due_time, priority=t.priority, category=t.category, gold_id=t.id,
        )
        for t in day.gold_tasks
    ]
    for u in day.updates:
        events.append(BoardEvent(
            day_id=day.id, kind=u.kind, card_id=u.card_id, speaker_id=u.speaker_id, note=u.note,
            owner_id=u.new_owner.entity_id if u.new_owner else None,
            due_date=u.due_date, due_time=u.due_time, gold_id=u.id,
        ))
    return events


def gold_board_after(day: StandupDay) -> list[Card]:
    return apply(day.board_before, gold_events(day), day)
