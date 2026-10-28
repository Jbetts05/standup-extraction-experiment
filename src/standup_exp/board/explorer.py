"""Build the dataset & trace explorer: docs/04-board-pipeline/trace-explorer/index.html.

One row per answer-key item (every true new job and update) plus every extra mention the pipeline made.
Each row follows the data through the pipeline and is checked against the answer key at every stage:

    raw text -> Step 1 extracted -> Step 1 matched to master data -> Step 2 board context
             -> Step 3 pick + rule -> board result

and links to the trace of the model calls behind it (prompt, input, schema, output, tokens, latency, cost),
recorded when the pipeline ran. Reads results/board/latest-context-<model>.json; no model is called.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

from ..ontology import Ontology
from ..paths import BOARD_DOCS
from .context import related
from .evaluate import align_mentions, locate
from .runner import SERIES_DIR, load_series, results_path

EXPLORER_DOCS = BOARD_DOCS / "trace-explorer"
OPEN = {"todo", "in_progress", "blocked"}
KIND_LABEL = {"new": "new job", "progress": "progress", "done": "done", "blocked": "blocked", "cancelled": "cancelled",
              "reassigned": "reassigned", "due_changed": "new due date"}


def _iso(v) -> str | None:
    return str(v)[:10] if v else None


def _stage(key, label, status, text, **data) -> dict:
    return {"key": key, "label": label, "status": status, "text": text, **data}


def _json(value):
    if not isinstance(value, str):
        return value or {}
    try:
        return json.loads(value)
    except ValueError:
        return {}


def export_explorer(data: Path = SERIES_DIR, model: str = "gpt-5.6-luna") -> dict:
    onto = Ontology.load()
    path = results_path("context", model)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `standup-exp board run --pipeline context --model {model}` first.")
    results = json.loads(path.read_text(encoding="utf-8"))
    chained = results["modes"].get("chained")
    if chained is None:
        raise SystemExit(f"{path} has no chained run.")
    name = lambda pid: onto.by_id[pid].name if pid and pid in onto.by_id else None  # noqa: E731
    texts: dict[str, str] = {}

    def text_id(value) -> str:
        s = value if isinstance(value, str) else json.dumps(value, indent=1)
        key = hashlib.sha1(s.encode()).hexdigest()[:10]
        texts[key] = s
        return key

    standups, rows = [], []
    card_numbers: dict[str, int] = {}
    for days in load_series(data):
        records = {r["day_id"]: r for r in chained["days"].get(days[0].series_id, [])}
        last = records.get(days[-1].id)
        order = sorted(last["board_after"], key=lambda c: (c["created_in"], c["id"])) if last else []
        number = {c["id"]: k + 1 for k, c in enumerate(order)}
        card_numbers.update(number)
        prev_board: dict[str, dict] = {}
        for day in days:
            rec = records.get(day.id)
            if rec is None:
                continue
            why = rec.get("why") or {}
            mentions, decisions = why.get("mentions", []), why.get("decisions", [])
            score, events = rec["score"], rec["events"]
            mapping = score["mapping"]
            ours = {g: p for p, g in mapping.items()}
            after = {c["id"]: c for c in rec["board_after"]}
            calls = []
            for c in why.get("traces", []):
                calls.append({**{k: v for k, v in c.items() if k not in ("instructions", "schema")},
                              "instructions": text_id(c["instructions"]), "schema": text_id(c["schema"])})
            standups.append({
                "id": day.id, "series": day.series_id, "day": day.day, "date": day.date.isoformat(),
                "transcript": [{"t": round(u.start), "who": u.speaker_id, "text": u.text} for u in day.utterances],
                "open_before": sum(1 for c in prev_board.values() if c["status"] in OPEN),
                "calls": calls,
            })
            align = align_mentions(day, why, mapping)

            def card(cid):
                c = after.get(cid) or prev_board.get(cid)
                return None if c is None else {"id": cid, "number": number.get(cid), "summary": c["summary"],
                                               "owner_id": c.get("owner_id"), "component_id": c.get("component_id"),
                                               "status": c.get("status"), "due_date": c.get("due_date")}

            def ctx_stage(m, dec):
                people = {m["speaker_id"], m["owner_id"], m["new_owner_id"]} - {None}
                cands = set(m.get("people_candidates") or []) - people
                comps = [x for x in [m["component_id"], *(m.get("equipment_candidates") or [])] if x]
                pulled = []
                for cid in dec["context_card_ids"]:
                    c = prev_board.get(cid, {})
                    why_ = []
                    if c.get("owner_id") in people:
                        why_.append(f"owner is {name(c['owner_id'])}")
                    elif c.get("owner_id") in cands:
                        why_.append(f"owner {name(c['owner_id'])} is on the name shortlist")
                    hit = next((x for x in comps if related(onto, c.get("component_id"), x)), None)
                    if hit:
                        why_.append(f"equipment {onto.by_id[c['component_id']].name}" if c.get("component_id") else "equipment")
                    if not why_ and c.get("owner_id") is None:
                        why_.append("nobody owns it (reassigned mention)")
                    pulled.append({**(card(cid) or {"id": cid}), "why": "; ".join(why_)})
                keys = {"people": sorted(name(p) for p in people), "people_shortlist": sorted(name(p) for p in cands),
                        "equipment": [onto.by_id[x].name for x in comps if x in onto.by_id]}
                return pulled, keys

            used: set[int] = set()
            match_call = next((c for c in why.get("traces", []) if c["step"] == "match"), None)
            mtasks = {t["task_index"]: t for t in _json(match_call["input"]).get("tasks", [])} if match_call else {}
            extra_task = {i: len(mentions) + k for k, i in enumerate(i for i, x in enumerate(mentions) if x["new_owner_words"])}

            def in_shortlist(m_idx, field, expected):
                """Was the right id among the candidates the matcher was shown? None when the field has no shortlist."""
                task, key, want = {
                    "owner": (m_idx, "owner_candidates", expected.get("owner_id")),
                    "equipment": (m_idx, "component_candidates", expected.get("component_id")),
                    "site": (m_idx, "site_candidates", expected.get("site_id")),
                    "new owner": (extra_task.get(m_idx), "owner_candidates", expected.get("new_owner_id")),
                }.get(field, (None, None, None))
                if want is None or task not in mtasks or not mtasks[task].get(key.replace("candidates", "words")):
                    return None
                return want in {c["id"] for c in mtasks[task].get(key, [])}

            def make_row(kind, rid, m_idx, expected, gold_turns, result, extra=False):
                m = mentions[m_idx] if m_idx is not None else None
                dec = decisions[m_idx] if m_idx is not None else None
                if m_idx is not None:
                    used.add(m_idx)
                st = []
                turns = gold_turns or ([t] if m is not None and (t := locate(m["quote"], day)) is not None else [])
                st.append(_stage("raw", "Raw text", "na", " / ".join(day.utterances[i].text for i in turns[:3]) if turns
                                 else (m["quote"] if m else ""), turns=turns))
                # Step 1 · extracted
                if m is None:
                    st.append(_stage("extract", "Step 1 · Extracted", "bad", "Not extracted: no mention for this item"))
                elif extra:
                    st.append(_stage("extract", "Step 1 · Extracted", "na", f"said: {m['said']} · {m['summary']}"))
                else:
                    ok = m["said"] == kind
                    st.append(_stage("extract", "Step 1 · Extracted", "ok" if ok else "bad",
                                     f"said: {m['said']} · {m['summary']}" + ("" if ok else f" (expected: {kind})")))
                # Step 1 · matched to master data
                fields = []
                if m is not None and not extra:
                    if kind == "new":
                        fields = [("owner", m["owner_words"], name(m["owner_id"]), name(expected["owner_id"]), m["owner_id"] == expected["owner_id"]),
                                  ("equipment", m["component_words"], onto.by_id[m["component_id"]].name if m["component_id"] else None,
                                   onto.by_id[expected["component_id"]].name if expected["component_id"] else None,
                                   m["component_id"] == expected["component_id"]),
                                  ("site", None, name(m["site_id"]), name(expected["site_id"]), m["site_id"] == expected["site_id"]),
                                  ("due date", m["due_words"], _iso(m["due_date"]), expected["due_date"], _iso(m["due_date"]) == expected["due_date"]),
                                  ("priority", None, m["priority"], expected["priority"], m["priority"] == expected["priority"]),
                                  ("category", None, m["category"], expected["category"], m["category"] == expected["category"])]
                    elif kind == "reassigned":
                        fields = [("new owner", m["new_owner_words"], name(m["new_owner_id"]), name(expected["new_owner_id"]),
                                   m["new_owner_id"] == expected["new_owner_id"])]
                    elif kind == "due_changed":
                        fields = [("new due date", m["due_words"], _iso(m["due_date"]), expected["due_date"],
                                   _iso(m["due_date"]) == expected["due_date"])]
                fields = [{"field": f, "words": w, "got": g, "expected": e, "ok": ok,
                           "shortlist": None if ok else in_shortlist(m_idx, f, expected)} for f, w, g, e, ok in fields]
                wrong = [f["field"] + (" (right one not on the shortlist)" if f["shortlist"] is False else "")
                         for f in fields if not f["ok"]]
                if m is None:
                    st.append(_stage("normalise", "Step 1 · Matched to master data", "skip", "–"))
                elif not fields:
                    st.append(_stage("normalise", "Step 1 · Matched to master data", "na",
                                     " · ".join(x for x in [name(m["owner_id"]), onto.by_id[m["component_id"]].name if m["component_id"] else None,
                                                            _iso(m["due_date"])] if x) or "no ids to check", fields=fields))
                else:
                    st.append(_stage("normalise", "Step 1 · Matched to master data", "bad" if wrong else "ok",
                                     ("wrong: " + ", ".join(wrong)) if wrong else "all fields right", fields=fields))
                # Step 2 · board context
                target = ours.get(expected.get("gold_card")) if kind not in ("new", None) else None
                if m is None:
                    st.append(_stage("context", "Step 2 · Board context", "skip", "–"))
                else:
                    pulled, keys = ctx_stage(m, dec)
                    if extra or kind == "new":
                        status, text = "na", f"{len(pulled)} card{'s' if len(pulled) != 1 else ''} pulled"
                    elif target is None:
                        status, text = "bad", "the card was never created on our board"
                    elif target in dec["context_card_ids"]:
                        status, text = "ok", f"right card #{number.get(target)} in context ({len(pulled)} pulled)"
                    else:
                        status, text = "bad", f"card #{number.get(target)} not in context ({len(pulled)} pulled)"
                    st.append(_stage("context", "Step 2 · Board context", status, text, cards=pulled, keys=keys,
                                     open_before=sum(1 for c in prev_board.values() if c["status"] in OPEN)))
                # Step 3 · pick + rule
                if m is None:
                    st.append(_stage("decide", "Step 3 · Pick + rule", "skip", "–"))
                else:
                    act, matched, cid = dec["action"], dec["matched_card_id"], dec["card_id"]
                    num = lambda x: f"#{number.get(x, '?')}"  # noqa: E731
                    if extra:
                        status = "ok" if act in ("ignore", "comment") else "bad"
                        text = {"ignore": "ignored: correct, not in the answer key", "comment": "commented only",
                                "create": f"created {num(cid)}: not in the answer key",
                                "update": f"changed {num(cid)}: not in the answer key"}[act]
                    elif kind == "new":
                        status = "ok" if act == "create" else "bad"
                        text = {"create": f"no match → created {num(cid)}", "comment": f"merged into {num(cid)} instead of creating",
                                "ignore": "ignored instead of creating", "update": f"updated {num(cid)} instead of creating"}[act]
                    else:
                        if matched == target and act == "update":
                            status, text = "ok", f"picked {num(matched)} → update"
                        elif matched is None:
                            status, text = "bad", f"no match (expected {num(target)}) → ignored"
                        elif matched != target:
                            status, text = "bad", f"picked {num(matched)}, expected {num(target)}"
                        else:
                            status, text = "bad", f"picked {num(matched)} but action was {act}"
                    st.append(_stage("decide", "Step 3 · Pick + rule", status, text, rule=dec["rule"], action=act,
                                     matched=matched, card_id=cid, reason=dec.get("pick_reason")))
                st.append(_stage("result", "Board result", result[0], result[1]))
                my_events = [dict(e) for e in events if m_idx is not None and e.get("item_index") == m_idx]
                return {
                    "id": rid, "standup": day.id, "series": day.series_id, "day": day.day, "kind": "extra" if extra else kind,
                    "expected": expected, "mention": m_idx, "owner_task": extra_task.get(m_idx), "m": m, "stages": st,
                    "status": result[0], "failed_at": next((s["key"] for s in st if s["status"] == "bad"), None),
                    "events": my_events, "card_after": card(dec["card_id"]) if dec and dec["card_id"] else None,
                    "decision": {**{k: dec[k] for k in ("action", "rule", "matched_card_id", "card_id", "context_card_ids")},
                                 "pick_reason": dec.get("pick_reason")} if dec else None,
                }

            for t in day.gold_tasks:
                entries = [c for c in score["creates"] if c.get("gold") == t.id]
                ce = next((c for c in entries if c["outcome"] == "matched"), None) or \
                    next((c for c in entries if c["outcome"] == "wrong_merge"), None) or (entries[0] if entries else None)
                m_idx = None
                if ce and ce.get("pred"):
                    m_idx = next((k for k, d in enumerate(decisions) if d["card_id"] == ce["pred"] and d["action"] == "create"), None)
                if m_idx is None:
                    m_idx = align.get(t.id)
                if ce and ce["outcome"] == "matched":
                    names = {"owner": "owner", "component": "equipment", "site": "site", "due_date": "due date",
                             "priority": "priority", "category": "category"}
                    res = ("ok", "card created, every field right") if not ce["wrong"] else \
                        ("bad", "card created, but wrong: " + ", ".join(names.get(w, w) for w in ce["wrong"]))
                elif ce and ce["outcome"] == "wrong_merge":
                    res = ("bad", "no card: merged into an existing card")
                else:
                    res = ("bad", "no card created")
                expected = {"type": "new", "summary": t.summary, "owner_id": t.owner.entity_id, "component_id": t.component.entity_id,
                            "site_id": t.site.entity_id, "due_date": _iso(t.due_date), "due_time": t.due_time,
                            "priority": t.priority, "category": t.category,
                            "words": {"owner": t.owner.text, "component": t.component.text, "due": t.due_mention}}
                rows.append(make_row("new", t.id, m_idx, expected, t.evidence, res))
            cards_before = {c.id: c for c in day.board_before}
            for u in day.updates:
                ue = next(x for x in score["updates"] if x["gold"] == u.id)
                m_idx = events[ue["event"]].get("item_index") if ue["event"] is not None else None
                if m_idx is None:
                    m_idx = align.get(u.id)
                label = {"exact": ("ok", f"{KIND_LABEL[u.kind]} applied to the right card"),
                         "wrong_change": ("bad", f"right card, wrong change: {KIND_LABEL.get(ue['pred_kind'], ue['pred_kind'])} "
                                                  f"instead of {KIND_LABEL[u.kind]}"),
                         "wrong_value": ("bad", "right card and change, wrong new owner or date"),
                         "missed": ("bad", "no change made")}[ue["outcome"]]
                if ue.get("false_close"):
                    label = ("bad", label[1] + " (a false close)")
                gc = cards_before.get(u.card_id)
                expected = {"type": u.kind, "gold_card": u.card_id, "our_card": ours.get(u.card_id),
                            "our_number": number.get(ours.get(u.card_id)), "summary": gc.summary if gc else u.card_id,
                            "new_owner_id": u.new_owner.entity_id if u.new_owner else None, "due_date": _iso(u.due_date),
                            "note": u.note, "tags": u.tags, "speaker_id": u.speaker_id}
                rows.append(make_row(u.kind, u.id, m_idx, expected, u.evidence, label))
            extra_by_event = {x["event"]: x for x in score["extra_events"]}
            bad_creates = {c["pred"]: c["outcome"] for c in score["creates"] if c.get("outcome") in ("extra", "duplicate")}
            for k in range(len(mentions)):
                if k in used:
                    continue
                mine = [j for j, e in enumerate(events) if e.get("item_index") == k]
                res = ("ok", "no change to the board")
                for j in mine:
                    e = events[j]
                    if e["kind"] == "create" and e["card_id"] in bad_creates:
                        res = ("bad", f"{bad_creates[e['card_id']]} card created")
                    elif j in extra_by_event:
                        res = ("bad", "false close on a card that was not finished" if extra_by_event[j]["false_close"]
                               else f"{e['kind']} made that is not in the answer key")
                    elif e["kind"] == "comment":
                        res = ("ok", "comment added to the card's history")
                expected = {"type": None, "summary": "Not in the answer key"}
                rows.append(make_row(None, f"{day.id}-x{k + 1}", k, expected, [], res, extra=True))
            prev_board = after

    m = chained["metrics"]
    return {
        "meta": {**results["meta"], "metrics": {k: v for k, v in m.items() if k not in ("by_kind", "by_tag")}},
        "people": {p.id: p.name for p in onto.people},
        "assets": {a.id: {"name": a.name, "parent": a.parent} for a in onto.assets},
        "sites": {s.id: s.name for s in onto.sites},
        "texts": texts, "standups": standups, "rows": rows, "card_numbers": card_numbers,
    }


def build_explorer(data: Path = SERIES_DIR, model: str = "gpt-5.6-luna") -> Path:
    payload = json.dumps(export_explorer(data, model), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(payload, 9)).decode("ascii")
    out = EXPLORER_DOCS / "index.html"
    template = (EXPLORER_DOCS / "template.html").read_text(encoding="utf-8")
    out.write_text(template.replace("__DATA_GZ_B64__", encoded), encoding="utf-8")
    return out
