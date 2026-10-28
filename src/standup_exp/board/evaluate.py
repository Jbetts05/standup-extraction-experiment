"""Score the board loop against the answer key, day by day.

Card identity
    Predicted cards are tied to true cards when they are created: the day's new-card predictions
    are matched to the day's true new jobs with the experiment's existing Hungarian matcher
    (``evaluate.match``). That mapping carries forward, so in chained mode an update on a
    predicted card can be checked against the true card it stands for.

New jobs (the existing pipeline's output, after merging)
    created / missed / extra, dashboard-ready (every field right), and:
    duplicate    - a new card whose quote is one of today's true *updates* (a card made twice)
    wrong merge  - a real new job that step 2 folded into an existing card

Updates on existing cards (step 2)
    exact        - right card, right change, right value (new owner / new date)
    right card   - linked to the right card, whatever the change
    missed       - a true update with no predicted change on that card
    extra        - a predicted change with no true update on that card
    false close  - done/cancelled on a card that was not finished or cancelled today (matched or not)
    ("comment" events only add history and are not scored)

Board
    card correct - status and every field (owner, equipment, site, due, priority, category) right
    board accuracy = correct cards / (true cards + extra cards), after each day, so invented cards count
    against it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from ..evaluate import MATCH_THRESHOLD, field_scores, similarity
from ..ontology import Ontology
from ..schema import Meeting, NormalisedTask
from .pipeline import DayRun
from .schema import CLOSING, Card, StandupDay, gold_board_after

CARD_FIELDS = ("owner_id", "component_id", "site_id", "due_date", "priority", "category")
SAME_JOB = 50  # wording similarity (0-100) for a mention to count as the same job in the per-step scores


def locate(quote: str | None, meeting: Meeting) -> int | None:
    """The turn a quote came from.

    Unlike ``standup_exp.evaluate.locate``, a short turn that merely appears inside the quote
    ("Understood.") cannot win: turns much shorter than the quote are compared whole.
    """
    if not quote:
        return None
    q = quote.lower()
    best, best_i = 0.0, None
    for i, u in enumerate(meeting.utterances):
        t = u.text.lower()
        s = fuzz.partial_ratio(q, t) if len(t) >= 0.4 * len(q) else fuzz.ratio(q, t)
        if s > best:
            best, best_i = s, i
    return best_i if best >= 80 else None


def match(onto: Ontology, meeting: Meeting, preds: list[NormalisedTask]):
    """``standup_exp.evaluate.match`` (same similarity and threshold) with the locator above."""
    golds = meeting.gold_tasks
    if not golds or not preds:
        return []
    turns = [locate(p.evidence_quote, meeting) for p in preds]
    sim = np.array([[similarity(onto, g, p, t) for p, t in zip(preds, turns)] for g in golds])
    rows, cols = linear_sum_assignment(-sim)
    return [(golds[r], preds[c]) for r, c in zip(rows, cols) if sim[r, c] >= MATCH_THRESHOLD]


def _near(turn: int | None, evidence: list[int]) -> bool:
    return turn is not None and bool(evidence) and min(evidence) - 1 <= turn <= max(evidence) + 1


def score_day(onto: Ontology, day: StandupDay, run: DayRun, mapping: dict[str, str]) -> tuple[Counter, dict]:
    """Score one day. ``mapping`` (predicted card id -> true card id) is extended with today's new cards."""
    c: Counter = Counter()
    detail: dict = {"creates": [], "updates": [], "extra_events": [], "cards": {}, "extra_cards": []}

    # --------------------------------------------------------------- new jobs
    created = [(i, t) for i, t in enumerate(run.new_jobs) if i not in run.merged]
    pairs = match(onto, day, [t for _, t in created])
    idx_of = {id(t): i for i, t in created}
    gold_of: dict[int, str] = {}
    for g, p in pairs:
        i = idx_of[id(p)]
        gold_of[i] = g.id
        mapping[f"{day.id}-p{i + 1}"] = g.id
        s = field_scores(onto, g, p)
        c["create_all_fields"] += s["all_fields"]
        detail["creates"].append({"pred": f"{day.id}-p{i + 1}", "gold": g.id, "outcome": "matched",
                                  "wrong": [k for k in ("owner", "component", "site", "due_date", "priority",
                                                        "category") if s[k] < 1]})
    c["gold_creates"] += len(day.gold_tasks)
    c["pred_creates"] += len(created)
    c["create_tp"] += len(pairs)
    update_windows = [u.evidence for u in day.updates]
    for i, t in created:
        if i in gold_of:
            continue
        dup = any(_near(locate(t.evidence_quote, day), w) for w in update_windows)
        c["duplicates" if dup else "extra_creates"] += 1
        detail["creates"].append({"pred": f"{day.id}-p{i + 1}", "gold": None,
                                  "outcome": "duplicate" if dup else "extra"})
    matched_gold = {g for g in gold_of.values()}
    for g in day.gold_tasks:
        if g.id not in matched_gold:
            detail["creates"].append({"pred": None, "gold": g.id, "outcome": "missed"})
    if run.merged:
        all_pairs = match(onto, day, run.new_jobs)
        paired = {id(p): g.id for g, p in all_pairs}
        for i, card in run.merged.items():
            real_new = paired.get(id(run.new_jobs[i]))
            wrong = real_new is not None and real_new not in matched_gold
            c["wrong_merges" if wrong else "merges"] += 1
            detail["creates"].append({"pred": None, "merged_into": card, "new_job": i,
                                      "gold": real_new if wrong else None,
                                      "outcome": "wrong_merge" if wrong else "merged"})

    # ---------------------------------------------------------------- updates
    preds = [(k, e) for k, e in enumerate(run.events) if e.kind not in ("create", "comment")]
    used: set[int] = set()
    closing_gold = {u.card_id for u in day.updates if u.kind in CLOSING}
    gold_cards_today = {u.card_id for u in day.updates}
    # A false close is any predicted done/cancelled on a real card that was not closed today,
    # including one matched to a true update of another kind (done said when it was progress).
    false_close = {k for k, e in preds
                   if e.kind in CLOSING and (t := mapping.get(e.card_id)) is not None and t not in closing_gold}
    c["false_closes"] += len(false_close)
    c["gold_updates"] += len(day.updates)
    c["pred_updates"] += len(preds)
    for u in day.updates:
        same_card = [(k, e) for k, e in preds if k not in used and mapping.get(e.card_id) == u.card_id]
        exact = [(k, e) for k, e in same_card if e.kind == u.kind]
        outcome, hit = "missed", None
        if exact:
            hit = exact[0]
            value_ok = ((u.kind != "reassigned" or hit[1].owner_id == (u.new_owner.entity_id if u.new_owner else None))
                        and (u.kind != "due_changed" or hit[1].due_date == u.due_date))
            outcome = "exact" if value_ok else "wrong_value"
        elif same_card:
            hit, outcome = same_card[0], "wrong_change"
        if hit:
            used.add(hit[0])
            c["right_card"] += 1
        c[f"update_{outcome}"] += 1
        c[f"kind_n:{u.kind}"] += 1
        c[f"kind_exact:{u.kind}"] += outcome == "exact"
        for tag in u.tags or ["plain"]:
            c[f"tag_n:{tag}"] += 1
            c[f"tag_exact:{tag}"] += outcome == "exact"
        detail["updates"].append({"gold": u.id, "card": u.card_id, "kind": u.kind, "outcome": outcome,
                                  "event": hit[0] if hit else None,
                                  "pred_kind": hit[1].kind if hit else None, "tags": u.tags,
                                  "false_close": bool(hit and hit[0] in false_close)})
    for k, e in preds:
        if k in used:
            continue
        target = mapping.get(e.card_id)
        c["extra_updates"] += 1
        detail["extra_events"].append({"event": k, "card": e.card_id, "gold_card": target, "kind": e.kind,
                                       "false_close": k in false_close,
                                       "on_updated_card": target in gold_cards_today})
    c["pred_closes"] += sum(e.kind in CLOSING for _, e in preds)

    # ------------------------------------------------------------------ board
    gold_after = {g.id: g for g in gold_board_after(day)}
    inverse = {mapping[p.id]: p for p in run.board_after if p.id in mapping}
    for gid, g in gold_after.items():
        p: Card | None = inverse.get(gid)
        wrong = [f for f in CARD_FIELDS if p is None or getattr(p, f) != getattr(g, f)]
        status_ok = p is not None and p.status == g.status
        ok = status_ok and not wrong
        c["cards"] += 1
        c["cards_found"] += p is not None
        c["cards_status_ok"] += status_ok
        c["cards_ok"] += ok
        if g.is_open:
            c["open_cards"] += 1
            c["open_cards_ok"] += ok
        detail["cards"][gid] = {"pred": p.id if p else None, "status_ok": status_ok, "wrong": wrong if p else [],
                                "pred_status": p.status if p else None, "ok": ok}
    for p in run.board_after:
        if p.id not in mapping:
            c["extra_cards"] += 1
            c["extra_open_cards"] += p.is_open
            detail["extra_cards"].append(p.id)
    detail["board_accuracy"] = _rate(c["cards_ok"], c["cards"] + c["extra_cards"]) or 0.0
    return c, detail


def align_mentions(day: StandupDay, why: dict, mapping: dict[str, str]) -> dict[str, int | None]:
    """Which Step 1 mention each true new job / update came from (true task or update id -> mention index).

    A mention counts only if its quote is at the item's turns AND it is about the same job: similar wording,
    or the pipeline tied it to our card for that true job. Used by the per-step scores and the explorer.
    """
    mentions, decisions = why.get("mentions", []), why.get("decisions", [])
    turns = [locate(m["quote"], day) for m in mentions]
    ours = {g: p for p, g in mapping.items()}
    cards = {x.id: x for x in day.board_before}
    gold = [(t.id, t.evidence, None, t.summary) for t in day.gold_tasks] + [
        (u.id, u.evidence, u.card_id, cards[u.card_id].summary if u.card_id in cards else "") for u in day.updates]
    used: set[int] = set()
    out: dict[str, int | None] = {}
    for gid, evidence, gold_card, job in gold:
        target = ours.get(gold_card) if gold_card else ours.get(gid)
        near = [((target is not None and decisions[i]["card_id"] == target),
                 fuzz.token_set_ratio(mentions[i]["summary"].lower(), job.lower()), i)
                for i, t in enumerate(turns) if i not in used and _near(t, evidence)]
        near = [x for x in near if x[0] or x[1] >= SAME_JOB]
        out[gid] = max(near)[2] if near else None
        if near:
            used.add(out[gid])
    return out


def score_steps(day: StandupDay, run: DayRun, mapping: dict[str, str]) -> Counter:
    """Context pipeline only: for each true new job or update, where did it go missing?

    Step 1: was it extracted (a mention whose quote is in its turns) and with the right ``said``?
    Step 2: for updates, was our card for the true card in the mention's context list?
    Step 3: when it was in the list, did the pick choose it?
    """
    c: Counter = Counter()
    if not run.why:
        return c
    mentions, decisions = run.why["mentions"], run.why["decisions"]
    ours = {g: p for p, g in mapping.items()}
    found = align_mentions(day, run.why, mapping)
    gold = [("new", t.id, None) for t in day.gold_tasks] + [(u.kind, u.id, u.card_id) for u in day.updates]
    for kind, gid, gold_card in gold:
        group = "new" if kind == "new" else "upd"
        c[f"gold_{group}"] += 1
        i = found[gid]
        if i is None:
            continue
        c[f"found_{group}"] += 1
        c["said_ok"] += mentions[i]["said"] == kind
        if gold_card is None:
            continue
        target = ours.get(gold_card)
        if target is None:
            c["card_not_on_board"] += 1  # an earlier stand-up never created this card
            continue
        ctx = decisions[i]["context_card_ids"]
        c["ctx_n"] += 1
        c["ctx_size"] += len(ctx)
        if target in ctx:
            c["ctx_hit"] += 1
            c["pick_ok"] += decisions[i]["matched_card_id"] == target
    used = {i for i in found.values() if i is not None}
    for i, m in enumerate(mentions):
        if i not in used:
            c["extra_mentions"] += 1
            if m["said"] != "new":
                c["extra_update_mentions"] += 1
                c["extra_update_ignored"] += decisions[i]["action"] == "ignore"
    return c


def step_metrics(c: Counter) -> dict:
    found = c["found_new"] + c["found_upd"]
    return {
        "step1_new_jobs_found": _rate(c["found_new"], c["gold_new"]),
        "step1_updates_found": _rate(c["found_upd"], c["gold_upd"]),
        "step1_said_correct": _rate(c["said_ok"], found),
        "step2_context_recall": _rate(c["ctx_hit"], c["ctx_n"]),
        "step2_context_size": _rate(c["ctx_size"], c["ctx_n"]),
        "step2_card_not_on_board": c["card_not_on_board"],
        "step3_pick_accuracy": _rate(c["pick_ok"], c["ctx_hit"]),
        "extra_mentions": c["extra_mentions"],
        "extra_update_mentions": c["extra_update_mentions"],
        "extra_update_mentions_ignored": c["extra_update_ignored"],
    }


def score_series(onto: Ontology, days: list[StandupDay], runs: list[DayRun]) -> tuple[Counter, list[dict]]:
    total: Counter = Counter()
    details = []
    mapping: dict[str, str] = {}
    for day, run in zip(days, runs):
        if run.mode == "isolated":
            mapping = {c.id: c.id for c in day.board_before}
        counts, detail = score_day(onto, day, run, mapping)
        detail["mapping"] = dict(mapping)
        total += counts
        total += Counter({f"step:{k}": v for k, v in score_steps(day, run, mapping).items()})
        details.append(detail)
    if details:
        total["final_cards"] += counts["cards"]
        total["final_cards_ok"] += counts["cards_ok"]
        total["final_extra_cards"] += counts["extra_cards"]
    return total, details


def _rate(a: float, b: float) -> float | None:
    return a / b if b else None


def metrics(c: Counter) -> dict:
    """Flat metrics from summed counts (sum ``score_series`` totals over series first)."""
    out = {
        "stand_ups": c["days"],
        "gold_creates": c["gold_creates"], "gold_updates": c["gold_updates"],
        "create_precision": _rate(c["create_tp"], c["pred_creates"]),
        "create_recall": _rate(c["create_tp"], c["gold_creates"]),
        "create_dashboard_ready": _rate(c["create_all_fields"], c["gold_creates"]),
        "duplicates": c["duplicates"], "extra_creates": c["extra_creates"],
        "merges": c["merges"], "wrong_merges": c["wrong_merges"],
        "update_precision": _rate(c["update_exact"], c["pred_updates"]),
        "update_recall": _rate(c["update_exact"], c["gold_updates"]),
        "update_right_card": _rate(c["right_card"], c["gold_updates"]),
        "updates_missed": c["update_missed"], "updates_wrong_change": c["update_wrong_change"],
        "updates_wrong_value": c["update_wrong_value"], "extra_updates": c["extra_updates"],
        "false_closes": c["false_closes"], "pred_closes": c["pred_closes"],
        "false_close_share": _rate(c["false_closes"], c["pred_closes"]),
        "board_accuracy_daily": _rate(c["cards_ok"], c["cards"] + c["extra_cards"]),
        "status_accuracy_daily": _rate(c["cards_status_ok"], c["cards"]),
        "board_accuracy_final": _rate(c["final_cards_ok"], c["final_cards"] + c["final_extra_cards"]),
        "final_cards": c["final_cards"], "final_extra_cards": c["final_extra_cards"],
    }
    by_kind = {k.split(":", 1)[1]: {"n": c[k], "exact": _rate(c[f"kind_exact:{k.split(':', 1)[1]}"], c[k])}
               for k in sorted(c) if k.startswith("kind_n:")}
    by_tag = {k.split(":", 1)[1]: {"n": c[k], "exact": _rate(c[f"tag_exact:{k.split(':', 1)[1]}"], c[k])}
              for k in sorted(c) if k.startswith("tag_n:")}
    out["by_kind"], out["by_tag"] = by_kind, by_tag
    steps = Counter({k[5:]: v for k, v in c.items() if k.startswith("step:")})
    if steps:
        out["steps"] = step_metrics(steps)
    return out


def pooled(per_series: list[Counter], days: int) -> dict:
    total: Counter = sum(per_series, Counter())
    total["days"] = days
    return metrics(total)
