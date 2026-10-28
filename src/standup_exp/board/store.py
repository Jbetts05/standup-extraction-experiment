"""The stand-in database: one JSON file with a list per table, shaped like the real project's database.

    standups      one row per stand-up that has run
    mentions      Step 1: every job mention (who, what, equipment, what was said, quote)
    decisions     Steps 2-3: the context cards pulled for a mention, the card it matched (with the model's reason), the action and rule
    cards         current state of every card (what the dashboard shows)
    card_history  every change to a card, with the quote it came from

Rows are flat and keyed by id, so they load straight into a real database or Power BI. The pipeline reads
the board from here (Step 2's context) and only Step 3's write commits to it. Saves are atomic (write a temp
file, then replace), so a page reading the file never sees half a write.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from pathlib import Path

from .schema import Card, Comment

TABLES = ("standups", "mentions", "decisions", "cards", "card_history")
INTERNAL = ("mention", "events")  # decision fields the UI uses but a database does not need
CARD_COLUMNS = ("card_id", "series_id", "number", "summary", "status", "owner_id", "component_id", "site_id",
                "due_date", "due_time", "priority", "category", "created_in", "created_on", "updated_in")


class BoardStore:
    def __init__(self, path: Path | None = None, fresh: bool = False):
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self.rows: dict[str, list[dict]] = {t: [] for t in TABLES if t not in ("cards", "card_history")}
        self.cards: dict[str, Card] = {}
        self.meta: dict[str, dict] = {}  # card_id -> {"series_id", "number"}
        if self.path and self.path.exists() and not fresh:
            self._load(json.loads(self.path.read_text(encoding="utf-8")))

    # ---------------------------------------------------------------- reads
    def board(self, series_id: str) -> list[Card]:
        """The series' cards as the pipeline sees them (copies, so callers cannot change the store)."""
        with self.lock:
            return [c.model_copy(deep=True) for cid, c in self.cards.items() if self.meta[cid]["series_id"] == series_id]

    def standups_run(self, series_id: str) -> list[str]:
        with self.lock:
            return [r["standup_id"] for r in self.rows["standups"] if r["series_id"] == series_id]

    # --------------------------------------------------------------- writes
    def commit(self, day, board_after: list[Card], why: dict, standup_row: dict) -> None:
        """Replace the series' board with ``board_after`` and record the stand-up, mentions and decisions."""
        with self.lock:
            sid = day.series_id
            self._drop(lambda r: r.get("standup_id") == day.id)
            for cid in [cid for cid, m in self.meta.items() if m["series_id"] == sid]:
                if cid not in {c.id for c in board_after}:
                    self.cards.pop(cid)
                    self.meta.pop(cid)
            number = max((m["number"] for m in self.meta.values() if m["series_id"] == sid), default=0)
            for c in sorted(board_after, key=lambda c: (c.created_in, c.id)):
                if c.id not in self.meta:
                    number += 1
                    self.meta[c.id] = {"series_id": sid, "number": number}
                self.cards[c.id] = c.model_copy(deep=True)
            self.rows["standups"].append({"standup_id": day.id, "series_id": sid, "day": day.day,
                                          "date": day.date.isoformat(), "chair_id": day.chair_id,
                                          "attendees": ", ".join(day.attendees), **standup_row})
            self.rows["mentions"] += [{**m, "series_id": sid} for m in why.get("mentions", [])]
            self.rows["decisions"] += [{k: v for k, v in d.items() if k not in INTERNAL} | {"series_id": sid}
                                       for d in why.get("decisions", [])]
            self.save()  # durable per stand-up

    def reset(self, series_id: str | None = None) -> None:
        with self.lock:
            keep = (lambda r: r.get("series_id") != series_id) if series_id else (lambda r: False)
            self._drop(lambda r: not keep(r))
            for cid in [cid for cid, m in self.meta.items() if not series_id or m["series_id"] == series_id]:
                self.cards.pop(cid)
                self.meta.pop(cid)
            self.save()

    def save(self) -> None:
        if not self.path:
            return
        with self.lock:
            text = json.dumps(self.to_dict(), indent=1, default=_json_default)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)

    # --------------------------------------------------------------- tables
    def to_dict(self) -> dict:
        with self.lock:
            cards, history = [], []
            for cid, c in sorted(self.cards.items(), key=lambda kv: (self.meta[kv[0]]["series_id"], self.meta[kv[0]]["number"])):
                m = self.meta[cid]
                row = c.model_dump(mode="json", exclude={"history"})
                cards.append({"card_id": cid, "series_id": m["series_id"], "number": m["number"],
                              **{k: row.get(k) for k in CARD_COLUMNS if k not in ("card_id", "series_id", "number")}})
                for k, h in enumerate(c.history):
                    history.append({"history_id": f"{cid}-h{k + 1}", "card_id": cid, "series_id": m["series_id"],
                                    "standup_id": h.day_id, **h.model_dump(mode="json", exclude={"day_id"})})
            return {"saved_at": datetime.now().isoformat(timespec="seconds"),
                    "standups": list(self.rows["standups"]), "mentions": list(self.rows["mentions"]),
                    "decisions": list(self.rows["decisions"]), "cards": cards, "card_history": history}

    def _load(self, data: dict) -> None:
        for t in self.rows:
            self.rows[t] = list(data.get(t, []))
        history: dict[str, list[Comment]] = {}
        for h in data.get("card_history", []):
            history.setdefault(h["card_id"], []).append(Comment(day_id=h["standup_id"], **{
                k: h.get(k) for k in ("date", "kind", "text", "quote", "speaker_id", "confidence")}))
        for r in data.get("cards", []):
            cid = r["card_id"]
            self.cards[cid] = Card(id=cid, history=history.get(cid, []),
                                   **{k: r.get(k) for k in CARD_COLUMNS if k not in ("card_id", "series_id", "number")})
            self.meta[cid] = {"series_id": r["series_id"], "number": r["number"]}

    def _drop(self, match) -> None:
        for t in self.rows:
            self.rows[t] = [r for r in self.rows[t] if not match(r)]


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(type(o))
