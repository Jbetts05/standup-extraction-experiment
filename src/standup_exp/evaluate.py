"""Scoring.

Task matching
    Predicted tasks are paired with gold tasks one-to-one (Hungarian algorithm)
    per meeting. Pair similarity combines:
      0.5  the predicted evidence quote lands in (or next to) a gold evidence turn
      0.3  summary wording overlap (rapidfuzz token_set_ratio)
      0.1  same component (or its parent)
      0.1  same owner
    Pairs below ``MATCH_THRESHOLD`` are left unmatched (false positive + false negative).

Field scoring (on matched pairs, a correct null counts as correct)
    owner, component, component_hier (1 exact, 0.5 parent of the gold asset),
    site, due_date, due_time, priority, category, and all_fields (every field exact).

Headline metric
    dashboard_ready = matched tasks with all fields correct / gold tasks.
    This is the share of real jobs that would appear on the dashboard with no human fixes.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from .ontology import Ontology
from .schema import GoldTask, Meeting, NormalisedTask

MATCH_THRESHOLD = 0.45
FIELDS = ["owner", "component", "component_hier", "site", "due_date", "due_time", "priority", "category", "all_fields"]
LINKED = {"owner": "owner", "component": "component", "site": "site"}


def locate(quote: str | None, meeting: Meeting) -> int | None:
    if not quote:
        return None
    best, best_i = 0.0, None
    for i, u in enumerate(meeting.utterances):
        s = fuzz.partial_ratio(quote.lower(), u.text.lower())
        if s > best:
            best, best_i = s, i
    return best_i if best >= 80 else None


def _related(onto: Ontology, pred: str | None, gold: str | None) -> bool:
    return bool(pred and gold and (pred == gold or pred in onto.ancestors(gold)))


def similarity(onto: Ontology, g: GoldTask, p: NormalisedTask, p_turn: int | None) -> float:
    s = 0.0
    if p_turn is not None and g.evidence and min(g.evidence) - 1 <= p_turn <= max(g.evidence) + 1:
        s += 0.5
    s += 0.3 * max(fuzz.token_set_ratio(p.summary, g.summary), fuzz.token_set_ratio(p.summary, g.spoken_action)) / 100
    s += 0.1 * _related(onto, p.component_id, g.component.entity_id)
    s += 0.1 * (p.owner_id is not None and p.owner_id == g.owner.entity_id)
    return s


def match(onto: Ontology, meeting: Meeting, preds: list[NormalisedTask]) -> list[tuple[GoldTask, NormalisedTask]]:
    golds = meeting.gold_tasks
    if not golds or not preds:
        return []
    turns = [locate(p.evidence_quote, meeting) for p in preds]
    sim = np.array([[similarity(onto, g, p, t) for p, t in zip(preds, turns)] for g in golds])
    rows, cols = linear_sum_assignment(-sim)
    return [(golds[r], preds[c]) for r, c in zip(rows, cols) if sim[r, c] >= MATCH_THRESHOLD]


def field_scores(onto: Ontology, g: GoldTask, p: NormalisedTask) -> dict[str, float]:
    comp_hier = 1.0 if p.component_id == g.component.entity_id else (
        0.5 if p.component_id and g.component.entity_id and p.component_id in onto.ancestors(g.component.entity_id) else 0.0)
    s = {
        "owner": float(p.owner_id == g.owner.entity_id),
        "component": float(p.component_id == g.component.entity_id),
        "component_hier": comp_hier,
        "site": float(p.site_id == g.site.entity_id),
        "due_date": float(p.due_date == g.due_date),
        "due_time": float(p.due_time == g.due_time),
        "priority": float(p.priority == g.priority),
        "category": float(p.category == g.category),
    }
    s["all_fields"] = float(all(s[k] == 1.0 for k in ("owner", "component", "site", "due_date", "priority", "category")))
    return s


class Scoreboard:
    """Accumulates counts over meetings and produces a flat metrics dict."""

    def __init__(self, onto: Ontology):
        self.onto = onto
        self.tp = self.fp = self.fn = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.by_split: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # key -> [correct, n]
        self.seconds = 0.0
        self.input_tokens = self.output_tokens = 0
        self.meetings = 0

    def add(self, meeting: Meeting, preds: list[NormalisedTask]) -> list[dict]:
        pairs = match(self.onto, meeting, preds)
        self.meetings += 1
        self.tp += len(pairs)
        self.fp += len(preds) - len(pairs)
        self.fn += len(meeting.gold_tasks) - len(pairs)
        details = []
        for g, p in pairs:
            s = field_scores(self.onto, g, p)
            for k, v in s.items():
                self.sums[k] += v
            for fld in LINKED:
                mention = getattr(g, fld)
                cell = self.by_split[f"{fld}@{mention.split}"]
                cell[0] += s[fld]
                cell[1] += 1
            details.append({
                "gold_id": g.id, "pred_summary": p.summary, **s,
                # gold -> predicted values and the raw mention, for error analysis
                "values": {
                    "owner": [g.owner.entity_id, p.owner_id, p.raw.owner_mention],
                    "component": [g.component.entity_id, p.component_id, p.raw.component_mention, g.component.text],
                    "site": [g.site.entity_id, p.site_id, p.raw.site_mention],
                    "due_date": [g.due_date, p.due_date, p.raw.due_mention],
                    "priority": [g.priority, p.priority, p.raw.priority_mention],
                    "category": [g.category, p.category],
                },
            })
        return details

    def metrics(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out = {
            "meetings": self.meetings, "gold_tasks": self.tp + self.fn, "predicted_tasks": self.tp + self.fp,
            "precision": p, "recall": r, "f1": f1,
            "dashboard_ready": self.sums["all_fields"] / (self.tp + self.fn) if self.tp + self.fn else 0.0,
            "seconds_per_meeting": self.seconds / self.meetings if self.meetings else 0.0,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
        }
        for k in FIELDS:
            out[f"acc_{k}"] = self.sums[k] / self.tp if self.tp else 0.0
        for key, (c, n) in sorted(self.by_split.items()):
            out[f"acc_{key}"] = c / n if n else None
            out[f"n_{key}"] = int(n)
        return out


class LinkScoreboard:
    """Normalisation-only ("oracle") scores: gold mentions in, compare linked ids."""

    def __init__(self):
        self.cells: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.seconds = 0.0
        self.input_tokens = self.output_tokens = 0

    def add(self, g: GoldTask, p: NormalisedTask, onto: Ontology) -> None:
        s = field_scores(onto, g, p)
        for fld in ("owner", "component", "component_hier", "site"):
            mention = getattr(g, "component" if fld == "component_hier" else fld)
            if mention.split == "none":
                key = f"{fld}@none"
            else:
                key = f"{fld}@{'heldout' if mention.split == 'heldout' else 'known'}"
            self.cells[key][0] += s[fld]
            self.cells[key][1] += 1
        for fld in ("category", "priority", "due_date"):
            self.cells[fld][0] += s[fld]
            self.cells[fld][1] += 1

    def metrics(self) -> dict:
        out = {k: (c / n if n else None) for k, (c, n) in sorted(self.cells.items())}
        out.update({f"n_{k}": int(n) for k, (_, n) in sorted(self.cells.items())})
        out.update(seconds=self.seconds, input_tokens=self.input_tokens, output_tokens=self.output_tokens)
        return out
