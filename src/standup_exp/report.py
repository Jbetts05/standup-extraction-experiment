"""Build docs/01-extraction-experiment/RESULTS.md from the latest run files.

The report answers one question: which pipeline (extractor + normaliser, and which
LLM) is best on quality, cost and latency.

* Quality  = dashboard-ready: share of gold tasks found with every field correct.
* Cost     = USD per meeting from token usage x models.yaml prices (local models: $0).
* Latency  = seconds per meeting (sum of API/compute time; parallelism not counted).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .dataset import load_meetings
from .llm import model_ids
from .paths import EXPERIMENT_DOCS, MEETINGS_DIR, RESULTS_DIR


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _usd(v) -> str:
    if v is None:
        return "?"
    return "$0" if v == 0 else (f"${v:.4f}" if v < 0.01 else f"${v:.3f}")


def _sec(v) -> str:
    return "–" if v is None else f"{v:.1f}s"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _pipeline(r: dict) -> str:
    return r["extractor"] if r["normaliser"] == "passthrough" else f"{r['extractor']} → {r['normaliser']}"


def _pareto(rows: list[dict]) -> set[int]:
    """Indices of rows not beaten on both quality (higher) and cost (lower); unknown cost excluded."""
    known = [(i, r) for i, r in enumerate(rows) if r.get("cost_per_meeting") is not None]
    front = set()
    for i, r in known:
        dominated = any(
            o["dashboard_ready"] >= r["dashboard_ready"] and o["cost_per_meeting"] <= r["cost_per_meeting"]
            and (o["dashboard_ready"] > r["dashboard_ready"] or o["cost_per_meeting"] < r["cost_per_meeting"])
            for _, o in known
        )
        if not dominated:
            front.add(i)
    return front


def _headline(rows: list[dict]) -> list[str]:
    best = max(rows, key=lambda r: (r["dashboard_ready"], -(r.get("cost_per_meeting") or 1e9)))
    priced = [r for r in rows if r.get("cost_per_meeting") is not None]
    good = [r for r in priced if r["dashboard_ready"] >= best["dashboard_ready"] - 0.05]
    value = min(good, key=lambda r: (r["cost_per_meeting"], r["seconds_per_meeting"])) if good else None
    fast = [r for r in rows if r["dashboard_ready"] >= best["dashboard_ready"] - 0.10]
    fastest = min(fast, key=lambda r: r["seconds_per_meeting"])
    local = [r for r in rows if not r.get("model")]
    best_local = max(local, key=lambda r: r["dashboard_ready"]) if local else None

    def line(label, r):
        return (f"- **{label}:** `{_pipeline(r)}` — {_pct(r['dashboard_ready'])} dashboard-ready, "
                f"task F1 {_pct(r['f1'])}, {_usd(r.get('cost_per_meeting'))}/meeting, "
                f"{_sec(r['seconds_per_meeting'])}/meeting")

    out = ["### Verdict", "", line("Best quality", best)]
    if value:
        out.append(line("Best value (within 5 pts of best, cheapest)", value))
    out.append(line("Fastest (within 10 pts of best)", fastest))
    if best_local:
        out.append(line("Best with no LLM (runs on a laptop, $0)", best_local))
    ft = [r for r in rows if r["extractor"] == "gliner_ft" and r["normaliser"] == "classifier"]
    zs = [r for r in rows if r["extractor"] == "gliner" and r["normaliser"] == "classifier"]
    if ft and zs:
        f, z = ft[0], zs[0]
        out.append(
            f"- **GLiNER 2.5 fine-tuned vs zero-shot** (both → classifier): task F1 {_pct(z['f1'])} → {_pct(f['f1'])} "
            f"(precision {_pct(z['precision'])} → {_pct(f['precision'])}, recall {_pct(z['recall'])} → "
            f"{_pct(f['recall'])}); dashboard-ready {_pct(z['dashboard_ready'])} → {_pct(f['dashboard_ready'])}; "
            f"{_sec(z['seconds_per_meeting'])} → {_sec(f['seconds_per_meeting'])}/meeting on CPU")
    return out + [""]


SET_LABEL = {"meetings": "GPT-5.4-written set", "meetings-astra": "GPT-6 Astra-written set"}


def _moe(p: float, n: int) -> float:
    """95% margin of error for a proportion p over n items (normal approximation)."""
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def _sets() -> list[str]:
    names = sorted(p.name[len("latest-e2e-"):-len(".json")] for p in RESULTS_DIR.glob("latest-e2e-*.json"))
    return [n for n in names if n != "meetings-demo"]


def _load(stage: str, set_: str) -> dict | None:
    p = RESULTS_DIR / f"latest-{stage}-{set_}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _set_section(set_: str, e2e: dict | None, norm: dict | None) -> list[str]:
    meta = (e2e or norm or {}).get("meta", {})
    n_gold = meta.get("gold_tasks") or 0
    parts = [f"## {SET_LABEL.get(set_, set_)} · {meta.get('meetings', '?')} test meetings, {n_gold} real jobs", "",
             f"Transcripts written by **{meta.get('writer', '?')}**; trained components used "
             f"{meta.get('train_meetings', '?')} training meetings.", ""]
    if e2e:
        rows = sorted(e2e["rows"], key=lambda r: r["dashboard_ready"], reverse=True)
        parts += _headline(rows)

        # --- leaderboard
        front = _pareto(rows)
        parts += [f"### Leaderboard ({e2e['meta']['split']} split, {e2e['meta']['meetings']} meetings)", "",
                  "★ = on the quality/cost frontier (nothing else is both better and cheaper).", ""]
        parts.append(_table(
            ["#", "Pipeline", "Model", "Dashboard-ready", "Task F1", "Owner", "Component", "Comp (held-out)",
             "Due", "Priority", "Category", "$/meeting", "s/meeting", "Failed calls", ""],
            [[str(i + 1), f"`{_pipeline(r)}`", r.get("model") or "local", f"**{_pct(r['dashboard_ready'])}**",
              _pct(r["f1"]), _pct(r["acc_owner"]), _pct(r["acc_component"]),
              _pct(r.get("acc_component@heldout")), _pct(r["acc_due_date"]), _pct(r["acc_priority"]),
              _pct(r["acc_category"]), _usd(r.get("cost_per_meeting")), _sec(r["seconds_per_meeting"]),
              str(r.get("errors", 0) or ""), "★" if i in front else ""] for i, r in enumerate(rows)],
        ))
        parts.append("")

        # --- per model
        by = {(r["extractor"], r["normaliser"]): r for r in rows}
        n_by = {r["normaliser"]: r for r in (norm or {}).get("rows", [])}

        def cells(r):
            """ready / $ / s cells, or a failure note when every call failed."""
            if not r:
                return ["–", "–", "–"]
            if r.get("errors", 0) >= r["meetings"]:
                return [f"0% (all {r['meetings']} calls failed)", "–", "–"]
            ready = _pct(r["dashboard_ready"])
            if r.get("errors"):
                ready += f" ({r['errors']}/{r['meetings']} failed)"
            return [ready, _usd(r.get("cost_per_meeting")), _sec(r["seconds_per_meeting"])]

        model_rows = []
        for m in model_ids():
            two = by.get((f"llm@{m}", f"llm@{m}"))
            direct = by.get((f"llm_direct@{m}", "passthrough"))
            oracle = n_by.get(f"llm@{m}")
            if not (two or direct or oracle):
                continue
            model_rows.append([m, _pct(two["f1"]) if two else "–", *cells(two), *cells(direct),
                               _pct(oracle.get("component@heldout")) if oracle else "–"])
        if model_rows:
            parts += ["### Foundry models", "",
                      "*Two-step* = `llm@model` extracts mentions, `llm@model` links them via a fuzzy+embedding "
                      "shortlist. *One-step* = `llm_direct@model` sees the vocabulary and returns IDs. "
                      "*Held-out linking* = normalisation-only accuracy on nicknames absent from the vocabulary.", ""]
            parts.append(_table(
                ["Model", "Task F1", "Two-step ready", "Two-step $/mtg", "Two-step s/mtg",
                 "One-step ready", "One-step $/mtg", "One-step s/mtg", "Held-out linking"],
                model_rows,
            ))
            parts.append("")

        # --- local models
        local = [r for r in rows if not r.get("model")]
        if local:
            parts += ["### Local models (no LLM, $0)", ""]
            parts.append(_table(
                ["Pipeline", "Dashboard-ready", "Task F1", "Precision", "Recall", "s/meeting"],
                [[f"`{_pipeline(r)}`", _pct(r["dashboard_ready"]), _pct(r["f1"]), _pct(r["precision"]),
                  _pct(r["recall"]), _sec(r["seconds_per_meeting"])] for r in local],
            ))
            parts.append("")

    if norm:
        parts += [f"### Normalisation only ({norm['meta']['split']} split, gold mentions in)", "",
                  "Isolates linking from extraction. *Known* = official name or listed alias; "
                  "*held-out* = a nickname absent from the vocabulary.", ""]
        nrows = sorted(norm["rows"], key=lambda r: (r.get("component@heldout") or 0), reverse=True)
        parts.append(_table(
            ["Normaliser", "Asset (known)", "Asset (held-out)", "Held-out w/ parent credit", "Person (held-out)",
             "Site (held-out)", "Category", "Priority", "$/meeting", "s/meeting"],
            [[r["normaliser"], _pct(r.get("component@known")), _pct(r.get("component@heldout")),
              _pct(r.get("component_hier@heldout")), _pct(r.get("owner@heldout")), _pct(r.get("site@heldout")),
              _pct(r.get("category")), _pct(r.get("priority")), _usd(r.get("cost_per_meeting")),
              _sec(r.get("seconds_per_meeting"))] for r in nrows],
        ))
        parts.append("")

    return parts


def _dataset_lines() -> list[str]:
    from .paths import dataset_dirs

    out = ["## Datasets", ""]
    for d in dataset_dirs():
        ms = load_meetings(d)
        if not ms:
            continue
        splits = Counter(m.split for m in ms)
        tasks = sum(len(m.gold_tasks) for m in ms)
        turns = sum(len(m.utterances) for m in ms) / len(ms)
        held = sum(1 for m in ms if m.split == "test" for g in m.gold_tasks if g.component.split == "heldout")
        out.append(f"- **{SET_LABEL.get(d.name, d.name)}** (`data/{d.name}`, written by {ms[0].writer}): "
                   f"{len(ms)} meetings ({splits['train']} train / {splits['test']} test), ~{turns:.0f} turns each, "
                   f"{tasks} jobs; {held} test jobs use a held-out equipment nickname")
    return out + ["- Trained components (GLiNER fine-tune, text classifier) use the train meetings of every writer; "
                  "test meetings are never used for training.", ""]


def build_report(sets: list[str] | None = None, write: bool = True) -> str:
    sets = sets or _sets()
    loaded = {s: (_load("e2e", s), _load("normalise", s)) for s in sets}
    parts = ["# Results", "",
             "Generated by `standup-exp report`. **Dashboard-ready** = share of real jobs found with *every* field "
             "correct (owner, equipment, site, due date, priority, category). Cost uses `data/models.yaml` prices; "
             "latency is summed API/compute time per meeting. ± = 95% margin of error for that test set's size.", ""]
    # Cross-writer comparison: same pipeline, transcripts written by different models.
    with_rows = [s for s in sets if loaded[s][0]]
    if len(with_rows) > 1:
        tables = {s: {(r["extractor"], r["normaliser"]): r for r in loaded[s][0]["rows"]} for s in with_rows}
        common = set.intersection(*(set(t) for t in tables.values()))
        primary = max(with_rows, key=lambda s: loaded[s][0]["meta"].get("gold_tasks") or 0)
        top = sorted(common, key=lambda k: tables[primary][k]["dashboard_ready"], reverse=True)[:20]
        parts += ["## Does the writer matter?", "",
                  "Same pipelines on test sets written by different models. A pipeline whose rank only holds on its own "
                  "writer's transcripts is benefiting from familiarity, not skill.", ""]
        parts.append(_table(
            ["Pipeline", *[f"{SET_LABEL.get(s, s)} ({loaded[s][0]['meta'].get('gold_tasks')} jobs)" for s in with_rows]],
            [[f"`{_pipeline(tables[primary][k])}`",
              *[f"{_pct(tables[s][k]['dashboard_ready'])} ±{round(100 * _moe(tables[s][k]['dashboard_ready'], loaded[s][0]['meta'].get('gold_tasks') or 1))}"
                for s in with_rows]] for k in top]))
        parts.append("")
    for s in sets:
        e2e, norm = loaded[s]
        if e2e or norm:
            parts += _set_section(s, e2e, norm)
    if write:
        parts += _dataset_lines()
        parts += ["## Caveats", "",
                  "- Transcripts are LLM-written from an answer key; a writer can add a deadline (\"today\") the key "
                  "does not have, which costs every method a little on *Due*.",
                  "- Latency is per-call time summed per meeting; local models ran on a 4-core CPU without a GPU.",
                  "- Prices: `data/models.yaml` list prices; models with unknown price show `?`.",
                  "- Failed calls (timeouts) score as \"found nothing\" for that meeting and are listed per row. "
                  "Grok 4.6 ran with a 600 s timeout (others 240 s).",
                  "- One-call (`llm_direct`) IDs are kept only when the matching words were also returned.",
                  ""]
    if not any(any(v) for v in loaded.values()):
        parts.append("No runs yet. Run `standup-exp run` first.")
    text = "\n".join(parts) + "\n"
    if write:
        (EXPERIMENT_DOCS / "RESULTS.md").write_text(text, encoding="utf-8")
    return text