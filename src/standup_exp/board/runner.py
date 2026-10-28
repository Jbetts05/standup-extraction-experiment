"""Generate, store and run stand-up series.

    data/series/<day id>.json + .vtt        one file per stand-up (answer key + transcript)
    results/board/latest-<pipeline>-<model>.json  predicted events, boards and scores for every day and mode
    results/board/runs/<timestamp>/           the same, kept per run
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import groupby
from pathlib import Path

from ..dataset import save_meeting
from ..llm import Usage
from ..ontology import Ontology
from ..paths import DATA_DIR, RESULTS_DIR, rel
from .evaluate import metrics, score_series
from .pipeline import MODES, BoardPipeline, DayRun, total_cost
from .render import render_day_llm, render_day_template
from .schema import StandupDay, gold_board_after
from .simulate import sample_all

SERIES_DIR = DATA_DIR / "series"
BOARD_RESULTS = RESULTS_DIR / "board"


def load_series(directory: Path = SERIES_DIR) -> list[list[StandupDay]]:
    days = [StandupDay.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.json"))]
    days.sort(key=lambda d: (d.series_id, d.day))
    return [list(g) for _, g in groupby(days, key=lambda d: d.series_id)]


def generate(n_series: int, days: int, seed: int, renderer: str, out: Path, workers: int, overwrite: bool,
             log=print) -> None:
    onto = Ontology.load()
    plans = [d for s in sample_all(onto, n_series, days, seed) for d in s]
    todo = [d for d in plans if overwrite or not (out / f"{d.id}.json").exists()]
    for d in plans:
        if d not in todo:
            log(f"{d.id} exists, skipping (use --overwrite to regenerate)")

    def one(day: StandupDay):
        if renderer == "template":
            render_day_template(day, onto)
            u = Usage()
        else:
            try:
                _, u = render_day_llm(day, onto)
            except Exception as exc:  # noqa: BLE001 - keep going; the day can be regenerated later
                log(f"{day.id} ERROR {str(exc)[:160]}")
                return None
        save_meeting(day, out)
        log(f"{day.id} {day.date} {len(day.gold_tasks)} new {len(day.updates)} updates {len(day.utterances):>3} turns"
            f"  {day.generation_notes[0] if day.generation_notes else 'ok'}")
        return u

    with ThreadPoolExecutor(max(1, workers if renderer == "llm" else 1)) as pool:
        usages = [u for u in pool.map(one, todo) if u]
    total = Usage()
    for u in usages:
        total += u
    log(f"\nWrote {len(usages)} stand-ups to {out}")
    if total.input_tokens:
        log(f"Generation tokens: {total.input_tokens} in / {total.output_tokens} out")


def _run_record(run: DayRun, detail: dict) -> dict:
    return {
        "day_id": run.day_id,
        "error": run.error,
        "new_jobs": [{"summary": t.summary, "owner_id": t.owner_id, "component_id": t.component_id,
                      "site_id": t.site_id, "due_date": t.due_date, "due_time": t.due_time, "priority": t.priority,
                      "category": t.category, "quote": t.evidence_quote, "confidence": t.raw.confidence,
                      "words": {"owner": t.raw.owner_mention, "component": t.raw.component_mention,
                                "site": t.raw.site_mention, "due": t.raw.due_mention}}
                     for t in run.new_jobs],
        "merged": {str(k): v for k, v in run.merged.items()},
        "events": [e.model_dump(mode="json") for e in run.events],
        "board_after": [c.model_dump(mode="json") for c in run.board_after],
        "steps": {k: v.__dict__ for k, v in run.steps.items()},
        "score": detail,
        "why": run.why,
    }


PIPELINES = ("context", "whole-board")


def results_path(pipeline: str, model: str) -> Path:
    return BOARD_RESULTS / f"latest-{pipeline}-{model}.json"


def db_path(pipeline: str, model: str) -> Path:
    return BOARD_RESULTS / f"db-{pipeline}-{model}.json"


def make_pipeline(onto: Ontology, model: str, pipeline: str):
    if pipeline == "context":
        from .context_pipeline import ContextPipeline

        return ContextPipeline(onto, model)
    if pipeline == "whole-board":
        return BoardPipeline(onto, model)
    raise SystemExit(f"Unknown pipeline '{pipeline}'; choose from {', '.join(PIPELINES)}")


def run(model: str, data: Path = SERIES_DIR, modes: tuple[str, ...] = ("chained",), workers: int = 6,
        log=print, pipeline: str = "context") -> Path:
    onto = Ontology.load()
    series = load_series(data)
    if not series:
        raise SystemExit(f"No stand-up series in {data}. Run `standup-exp board generate` first.")
    unknown = set(modes) - set(MODES)
    if unknown:
        raise SystemExit(f"Unknown mode(s) {', '.join(sorted(unknown))}; choose from {', '.join(MODES)}")
    failed = [d.id for s in series for d in s if any(n.startswith("FAILED") for n in d.generation_notes)]
    if failed:
        log(f"WARNING: {len(failed)} stand-ups failed transcript validation, so their answer key may not match "
            f"the transcript: {', '.join(failed)}. Regenerate them with `board generate --overwrite`.")
    pipe = make_pipeline(onto, model, pipeline)
    all_days = [d for s in series for d in s]
    log(f"{pipe.name}: step 1 on {len(all_days)} stand-ups ...")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(pipe.prepare, all_days))

    results: dict = {"meta": {"model": model, "pipeline": pipeline, "pipeline_name": pipe.name, "data": rel(data),
                              "modes": list(modes), "failed_validation": failed,
                              "created": datetime.now().isoformat(timespec="seconds")},
                     "modes": {}}
    for mode in modes:
        log(f"board loop, {mode} ...")
        store = None
        if pipeline == "context" and mode == "chained":
            from .store import BoardStore

            store = BoardStore(db_path(pipeline, model), fresh=True)
        with ThreadPoolExecutor(len(series)) as pool:
            runs_by_series = list(pool.map(lambda s: pipe.run_series(s, mode, store), series))
        if store is not None:
            store.save()
            results["meta"]["database"] = rel(store.path)
        per_series, records, total = {}, {}, Counter()
        for days, runs in zip(series, runs_by_series):
            counts, details = score_series(onto, days, runs)
            counts["days"] = len(days)
            total += counts
            per_series[days[0].series_id] = metrics(counts)
            records[days[0].series_id] = [_run_record(r, d) for r, d in zip(runs, details)]
        flat_runs = [r for rs in runs_by_series for r in rs]
        m = metrics(total)
        m["cost"] = total_cost(flat_runs)
        m["cost_per_stand_up"] = (sum(v["cost_usd"] or 0 for v in m["cost"].values()) / len(flat_runs)
                                  if flat_runs else 0.0)
        m["seconds_per_stand_up"] = (sum(v["seconds"] for v in m["cost"].values()) / len(flat_runs)
                                     if flat_runs else 0.0)
        m["failed_days"] = sum(1 for r in flat_runs if r.error)
        results["modes"][mode] = {"metrics": m, "series": per_series, "days": records}
        log(_summary(mode, m))

    BOARD_RESULTS.mkdir(parents=True, exist_ok=True)
    run_dir = BOARD_RESULTS / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{pipeline}-{model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=1, default=str)
    (run_dir / "board.json").write_text(text, encoding="utf-8")
    latest = results_path(pipeline, model)
    latest.write_text(text, encoding="utf-8")
    return latest


def _pct(x) -> str:
    return "  -  " if x is None else f"{100 * x:4.0f}%"


def _summary(mode: str, m: dict) -> str:
    s = m.get("steps")
    steps = (f"\n    steps: 1) new jobs found {_pct(s['step1_new_jobs_found'])}, updates found {_pct(s['step1_updates_found'])}, "
             f"said right {_pct(s['step1_said_correct'])} | 2) right card in context {_pct(s['step2_context_recall'])} "
             f"(avg {s['step2_context_size'] or 0:.1f} cards, {s['step2_card_not_on_board']} not on our board) | "
             f"3) right pick {_pct(s['step3_pick_accuracy'])} | extra update mentions ignored "
             f"{s['extra_update_mentions_ignored']}/{s['extra_update_mentions']}") if s else ""
    return (
        f"  [{mode}] new jobs: dashboard-ready {_pct(m['create_dashboard_ready'])}, duplicates {m['duplicates']}, "
        f"wrong merges {m['wrong_merges']} | updates: exact {_pct(m['update_recall'])} of true, precision "
        f"{_pct(m['update_precision'])}, right card {_pct(m['update_right_card'])}, false closes "
        f"{m['false_closes']}/{m['pred_closes']} | board accuracy final {_pct(m['board_accuracy_final'])}, "
        f"daily {_pct(m['board_accuracy_daily'])} | ${m['cost_per_stand_up']:.4f}/stand-up"
        + (f" | {m['failed_days']} failed days" if m["failed_days"] else "") + steps
    )


def show(day_id: str, data: Path = SERIES_DIR) -> str:
    onto = Ontology.load()
    day = next((d for s in load_series(data) for d in s if d.id == day_id), None)
    if day is None:
        raise SystemExit(f"No stand-up {day_id} in {data}")
    name = lambda pid: onto.by_id[pid].name if pid else "-"  # noqa: E731
    lines = [f"# {day.id} (series {day.series_id}, day {day.day}) {day.date:%A %d %B %Y}  renderer={day.renderer}",
             "", "## Open cards before the meeting"]
    lines += [f"- {c.id} [{c.status}] {c.summary} | owner={name(c.owner_id)} | due={c.due_date}"
              for c in day.board_before if c.is_open] or ["(none)"]
    lines += ["", day.transcript_text(), "", "## New jobs"]
    lines += [f"- {t.id} {t.summary} | owner={name(t.owner.entity_id)} | due={t.due_date} | turns={t.evidence}"
              for t in day.gold_tasks]
    lines += ["", "## Updates"]
    lines += [f"- {u.id} {u.kind} {u.card_id} by {name(u.speaker_id)} ('{u.reference}') {u.note or ''} "
              f"{u.tags} turns={u.evidence}" for u in day.updates] or ["(none)"]
    lines += ["", "## Board after", *[f"- {c.id} [{c.status}] {c.summary}" for c in gold_board_after(day)]]
    return "\n".join(lines)
