"""Build the mock task board: docs/04-board-pipeline/mock-board/index.html.

Reads the stand-up series (answer key + transcripts) and the chained run in
results/board/latest-<pipeline>-<model>.json, and embeds them (gzip + base64) into mock-board/template.html.
Nothing is re-run and no model is called.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

from ..ontology import Ontology
from ..paths import BOARD_DOCS as APPROACH_DOCS
from .runner import SERIES_DIR, load_series, results_path
from .schema import gold_board_after

BOARD_DOCS = APPROACH_DOCS / "mock-board"
CARD_KEYS = ("id", "summary", "category", "priority", "owner_id", "component_id", "site_id", "due_date", "due_time",
             "status", "created_in", "updated_in", "history")


def _card(c: dict) -> dict:
    return {k: c.get(k) for k in CARD_KEYS}


def export(data: Path = SERIES_DIR, model: str = "gpt-5.6-luna", pipeline: str = "context") -> dict:
    onto = Ontology.load()
    path = results_path(pipeline, model)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `standup-exp board run --pipeline {pipeline} --model {model}` first.")
    results = json.loads(path.read_text(encoding="utf-8"))
    chained = results["modes"].get("chained")
    if chained is None:
        raise SystemExit(f"{path} has no chained run. Re-run `standup-exp board run --model {model}`.")

    series = []
    for days in load_series(data):
        records = {r["day_id"]: r for r in chained["days"].get(days[0].series_id, [])}
        out_days = []
        for d in days:
            r = records.get(d.id)
            out_days.append({
                "id": d.id, "day": d.day, "date": d.date.isoformat(), "chair": d.chair_id,
                "attendees": d.attendees,
                "transcript": [{"t": round(u.start), "who": u.speaker_id, "text": u.text} for u in d.utterances],
                "gold": {
                    "board": [_card(c.model_dump(mode="json")) for c in gold_board_after(d)],
                    "creates": [{"id": t.id, "summary": t.summary, "owner": t.owner.entity_id, "evidence": t.evidence}
                                for t in d.gold_tasks],
                    "updates": [{"id": u.id, "kind": u.kind, "card": u.card_id, "speaker": u.speaker_id,
                                 "note": u.note, "evidence": u.evidence, "tags": u.tags} for u in d.updates],
                },
                "pred": None if r is None else {
                    "board": [_card(c) for c in r["board_after"]],
                    "events": r["events"],
                    "new_jobs": [{"summary": j["summary"], "quote": j["quote"]} for j in r["new_jobs"]],
                    "merged": r["merged"],
                    "score": r["score"],
                    "error": r["error"],
                    "why": r.get("why") or None,
                },
            })
        series.append({"id": days[0].series_id, "team": days[0].team, "days": out_days,
                       "metrics": chained["series"].get(days[0].series_id)})

    return {
        "meta": {**results["meta"], "metrics": chained["metrics"]},
        "people": {p.id: p.name for p in onto.people},
        "roles": {p.id: p.extra.get("role") for p in onto.people},
        "assets": {a.id: {"name": a.name, "parent": a.parent} for a in onto.assets},
        "sites": {s.id: s.name for s in onto.sites},
        "series": series,
    }


def build(data: Path = SERIES_DIR, model: str = "gpt-5.6-luna", pipeline: str = "context") -> Path:
    payload = json.dumps(export(data, model, pipeline), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(payload, 9)).decode("ascii")
    out = BOARD_DOCS / "index.html"
    template = (BOARD_DOCS / "template.html").read_text(encoding="utf-8")
    out.write_text(template.replace("__DATA_GZ_B64__", encoded), encoding="utf-8")
    return out
