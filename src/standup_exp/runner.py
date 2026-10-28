"""Run experiments and write results.

Two stages:

* ``e2e``       - transcript -> extractor -> normaliser -> scored against gold.
                  Extraction output is cached per (extractor, meeting) so trying a
                  new normaliser never re-runs extraction.
* ``normalise`` - "oracle" stage: gold mentions are fed straight into each
                  normaliser, isolating linking/classification quality from
                  extraction errors.

Each run writes ``results/runs/<timestamp>-<stage>/`` with ``metrics.json``
(one row per method combination) and ``details.jsonl`` (per matched task), and
updates ``results/latest-<stage>-<set>.json`` (one per test set, e.g. ``latest-e2e-meetings-astra.json``),
which ``report`` and ``walkthrough`` read.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .dataset import load_all_train, load_meetings
from .extractors import TranscriptView, get_extractor
from .evaluate import LinkScoreboard, Scoreboard, field_scores, locate, match
from .llm import cost_usd, default_model
from .normalisers import NormContext, get_normaliser
from .ontology import Ontology
from .paths import CACHE_DIR, MEETINGS_DIR, RESULTS_DIR, dataset_dirs, rel, set_name
from .schema import ExtractionResult, Meeting, NormalisedTask, RawTask

# llm_direct links IDs itself; passthrough only makes sense with it.
DIRECT_EXTRACTORS = {"llm_direct"}
LLM_EXTRACTORS = {"llm", "llm_direct"}
CPU_HEAVY_NORMALISERS = {"gliner", "gliner_decide"}
WORKERS = 8  # threads for network-bound (LLM) methods


def model_of(method: str, kind: str) -> str | None:
    """The LLM a method uses (plain ``llm`` = catalogue default), or None if it uses no LLM."""
    base, _, model = method.partition("@")
    uses_llm = base in LLM_EXTRACTORS if kind == "extractor" else base == "llm"
    return (model or default_model()) if uses_llm else None


def _compatible(extractor: str, normaliser: str) -> bool:
    base = extractor.partition("@")[0]
    if (base in DIRECT_EXTRACTORS) != (normaliser == "passthrough"):
        return False
    # GLiNER matchers make one CPU call per mention: pair them only with the local extractors
    # (their matching quality on every mention is measured separately in the normalise stage).
    if normaliser in CPU_HEAVY_NORMALISERS and base in LLM_EXTRACTORS:
        return False
    # Never mix two different LLMs in one pipeline: keeps the matrix small and costs attributable.
    ex_model, no_model = model_of(extractor, "extractor"), model_of(normaliser, "normaliser")
    return not (ex_model and no_model and ex_model != no_model)


def _map(obj, fn, items):
    """Run ``fn`` over items, in threads if ``obj`` is network bound."""
    if getattr(obj, "parallel", False):
        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(fn, items))
    return [fn(i) for i in items]


def _train_for(data_dir: Path) -> list[Meeting]:
    """Standard datasets share every writer's train split; any other dir (e.g. the demo) uses its own."""
    if Path(data_dir).resolve() in {d.resolve() for d in dataset_dirs()}:
        return load_all_train()
    return load_meetings(data_dir, "train")


def _ctx(meeting: Meeting) -> NormContext:
    return NormContext(meeting.id, meeting.date, tuple(meeting.attendees))


def _extract_cached(extractor, meeting: Meeting, names: dict[str, str], refresh: bool) -> ExtractionResult:
    # Only slow/paid extractors are cached; cheap ones rerun so code edits show up immediately.
    if not getattr(extractor, "cacheable", True):
        return extractor.extract(TranscriptView.of(meeting, names))
    digest = hashlib.sha256(meeting.transcript_text().encode()).hexdigest()[:12]
    path = CACHE_DIR / "extract" / extractor.name / f"{meeting.id}-{digest}.json"
    if path.exists() and not refresh:
        return ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    try:
        result = extractor.extract(TranscriptView.of(meeting, names))
    except Exception as exc:  # noqa: BLE001 - a failed call scores as "found nothing"; not cached
        print(f"    ! {extractor.name} failed on {meeting.id}: {str(exc)[:160]}")
        return ExtractionResult(meeting_id=meeting.id, extractor=extractor.name, tasks=[],
                                seconds=time.perf_counter() - t0, error=str(exc)[:500])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=1), encoding="utf-8")
    return result


def _normalise_timed(normaliser, tasks: list[RawTask], meeting: Meeting, ex_name: str = "") -> tuple[list, float, bool, tuple]:
    """Normalise one meeting -> (predictions, seconds, failed, (input_tokens, output_tokens)).

    Latency uses the recorded API time when the LLM reply was replayed from cache.
    CPU-heavy local matchers (GLiNER) cache their outputs and measured time.
    A failed call scores as no predictions.
    """
    cache = None
    if getattr(normaliser, "cache_outputs", False):
        key = hashlib.sha256(json.dumps([normaliser.name, ex_name, meeting.id, meeting.attendees,
                                         [t.model_dump(mode="json") for t in tasks]], sort_keys=True).encode()).hexdigest()[:24]
        cache = CACHE_DIR / "norm" / normaliser.name / f"{key}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            return [NormalisedTask.model_validate(p) for p in data["preds"]], data["seconds"], False, (0, 0)
    t0 = time.perf_counter()
    try:
        preds = normaliser.normalise_meeting(tasks, _ctx(meeting))
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {normaliser.name} failed on {meeting.id}: {str(exc)[:160]}")
        return [], time.perf_counter() - t0, True, (0, 0)
    elapsed = time.perf_counter() - t0
    recorded = getattr(normaliser, "seconds_by_meeting", {}).get(meeting.id)
    tokens = getattr(normaliser, "tokens_by_meeting", {}).get(meeting.id, (0, 0))
    seconds = max(elapsed, recorded or 0.0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"preds": [p.model_dump(mode="json") for p in preds], "seconds": seconds}),
                         encoding="utf-8")
    return preds, seconds, False, tokens


def raw_record(t: RawTask, meeting: Meeting) -> dict:
    """What an extractor returned for one job, as stored for the walkthrough."""
    return {
        "summary": t.summary, "owner": t.owner_mention, "component": t.component_mention, "site": t.site_mention,
        "due": t.due_mention, "due_date": t.due_date.isoformat() if t.due_date else None, "due_time": t.due_time,
        "urgency": t.priority_mention, "priority": t.priority, "category": t.category, "quote": t.evidence_quote,
        "turn": locate(t.evidence_quote, meeting), "confidence": t.confidence,
        "ids": {k: v for k, v in (("owner", t.owner_id), ("component", t.component_id), ("site", t.site_id)) if v},
    }


def meeting_record(onto: Ontology, meeting: Meeting, preds: list[NormalisedTask]) -> dict:
    """Per-meeting outcome of one pipeline: predictions, pairing with gold, field scores."""
    pairs = match(onto, meeting, preds)
    index = {id(p): i for i, p in enumerate(preds)}
    matched_p, matched_g = {id(p) for _, p in pairs}, {g.id for g, _ in pairs}
    scored = []
    for g, p in pairs:
        s = field_scores(onto, g, p)
        scored.append({"gold": g.id, "pred": index[id(p)],
                       "scores": {k: s[k] for k in ("owner", "component", "site", "due_date", "priority", "category",
                                                    "component_hier", "all_fields")}})
    return {
        "preds": [{"owner": p.owner_id, "component": p.component_id, "site": p.site_id,
                   "due_date": p.due_date.isoformat() if p.due_date else None, "due_time": p.due_time,
                   "priority": p.priority, "category": p.category} for p in preds],
        "pairs": scored,
        "fp": [i for i, p in enumerate(preds) if id(p) not in matched_p],
        "fn": [g.id for g in meeting.gold_tasks if g.id not in matched_g],
        "ready": sum(1 for x in scored if x["scores"]["all_fields"] == 1.0),
    }


def _cost_fields(row: dict, model: str | None) -> dict:
    cost = cost_usd(row["input_tokens"], row["output_tokens"], model)
    n = row["meetings"] or 1
    return {"model": model, "cost_usd": cost, "cost_per_meeting": None if cost is None else cost / n}

def _run_dir(stage: str) -> Path:
    d = RESULTS_DIR / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{stage}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(stage: str, run_dir: Path, rows: list[dict], details: list[dict], meta: dict, merge: bool = False) -> None:
    meta = {**meta, "run_dirs": [rel(run_dir)]}
    payload = {"meta": meta, "rows": rows}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    with (run_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, default=str) + "\n")
    latest = RESULTS_DIR / f"latest-{stage}-{meta['set']}.json"
    if merge and latest.exists():
        # Replace matching rows in the previous results, keep the rest; remember every run dir merged in
        # so the per-meeting records (pipelines.jsonl etc.) of all rows can be found again.
        old = json.loads(latest.read_text(encoding="utf-8"))
        key = (lambda r: (r.get("extractor"), r["normaliser"]))
        new_keys = {key(r) for r in rows}
        prev = old["meta"].get("run_dirs") or [old["meta"].get("run_dir")]
        payload = {"meta": {**old["meta"], "run_dirs": [rel(d) for d in prev if d] + [rel(run_dir)]},
                   "rows": [r for r in old["rows"] if key(r) not in new_keys] + rows}
    latest.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")


def run_e2e(extractors: list[str], normalisers: list[str], split: str = "test", data_dir: Path = MEETINGS_DIR,
            refresh: bool = False, log=print, merge: bool = False) -> list[dict]:
    onto = Ontology.load()
    names = {p.id: p.name for p in onto.people}
    meetings = load_meetings(data_dir, split)
    train = _train_for(data_dir)
    if not meetings:
        raise SystemExit(f"No '{split}' meetings in {data_dir}. Run `standup-exp generate` first.")

    run_dir = _run_dir("e2e")
    rows, details, records, ex_records = [], [], [], []
    norm_cache: dict = {}
    for ex_name in extractors:
        pairs = [n for n in normalisers if _compatible(ex_name, n)]
        if not pairs:
            continue
        extractor = get_extractor(ex_name)
        log(f"extract  {ex_name:<24} on {len(meetings)} {split} meetings")
        results = _map(extractor, lambda m: _extract_cached(extractor, m, names, refresh), meetings)
        extracted = {m.id: r for m, r in zip(meetings, results)}
        ex_model = model_of(ex_name, "extractor")
        for m in meetings:
            ex = extracted[m.id]
            ex_records.append({"extractor": ex_name, "meeting": m.id, "seconds": round(ex.seconds, 2),
                               "tokens": [ex.input_tokens, ex.output_tokens], "error": ex.error,
                               "cost": cost_usd(ex.input_tokens, ex.output_tokens, ex_model),
                               "tasks": [raw_record(t, m) for t in ex.tasks]})
        for no_name in pairs:
            if no_name not in norm_cache:
                norm_cache[no_name] = get_normaliser(no_name, onto, train)
            normaliser = norm_cache[no_name]
            outputs = _map(normaliser, lambda m: _normalise_timed(normaliser, extracted[m.id].tasks, m, ex_name), meetings)
            board = Scoreboard(onto)
            errors = sum(1 for m in meetings if extracted[m.id].error)
            model = ex_model or model_of(no_name, "normaliser")
            for m, (preds, seconds, failed, ntok) in zip(meetings, outputs):
                errors += failed
                ex = extracted[m.id]
                board.seconds += ex.seconds + seconds
                tin, tout = ex.input_tokens + ntok[0], ex.output_tokens + ntok[1]
                board.input_tokens += tin
                board.output_tokens += tout
                for d in board.add(m, preds):
                    details.append({"extractor": ex_name, "normaliser": no_name, "meeting": m.id, **d})
                records.append({"extractor": ex_name, "normaliser": no_name, "meeting": m.id,
                                **meeting_record(onto, m, preds), "failed": bool(failed or ex.error),
                                "seconds": round(ex.seconds + seconds, 2), "match_seconds": round(seconds, 2),
                                "tokens": [tin, tout], "cost": cost_usd(tin, tout, model)})
            row = {"extractor": ex_name, "normaliser": no_name, **board.metrics(), "errors": errors}
            row.update(_cost_fields(row, model))
            rows.append(row)
            log(f"  + {no_name:<24} F1 {row['f1']:.2f}  dashboard-ready {row['dashboard_ready']:.2f}  "
                f"{row['seconds_per_meeting']:.1f}s/meeting")

    meta = {"stage": "e2e", "split": split, "data_dir": rel(data_dir), "set": set_name(data_dir),
            "writer": meetings[0].writer, "meetings": len(meetings), "gold_tasks": sum(len(m.gold_tasks) for m in meetings),
            "train_meetings": len(train), "run_dir": rel(run_dir)}
    for fname, items in (("pipelines.jsonl", records), ("extractions.jsonl", ex_records)):
        with (run_dir / fname).open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, default=str) + "\n")
    _write("e2e", run_dir, rows, details, meta, merge)
    return rows


def gold_as_raw(meeting: Meeting) -> list[tuple]:
    """Turn gold tasks into the RawTask a perfect extractor would have produced."""
    out = []
    for g in meeting.gold_tasks:
        out.append((g, RawTask(
            summary=g.spoken_action,
            owner_mention=g.owner.text,
            component_mention=g.component.text,
            site_mention=g.site.text,
            due_mention=g.due_mention,
            priority_mention=g.priority_cue,
            evidence_quote=g.spoken_action,
        )))
    return out


def run_normalise(normalisers: list[str], split: str = "test", data_dir: Path = MEETINGS_DIR, log=print,
                  merge: bool = False) -> list[dict]:
    onto = Ontology.load()
    meetings = load_meetings(data_dir, split)
    train = _train_for(data_dir)
    run_dir = _run_dir("normalise")
    rows, picks = [], []
    for no_name in normalisers:
        if no_name == "passthrough":
            continue
        normaliser = get_normaliser(no_name, onto, train)
        gold = {m.id: gold_as_raw(m) for m in meetings}
        outputs = _map(normaliser, lambda m: _normalise_timed(normaliser, [r for _, r in gold[m.id]], m, "gold"), meetings)
        board = LinkScoreboard()
        errors = 0
        for m, (preds, seconds, failed, _tok) in zip(meetings, outputs):
            errors += failed
            board.seconds += seconds
            for (g, _), p in zip(gold[m.id], preds):
                board.add(g, p, onto)
            picks.append({"normaliser": no_name, "meeting": m.id, "seconds": round(seconds, 2),
                          "picks": [{"gold": g.id, "owner": p.owner_id, "component": p.component_id, "site": p.site_id,
                                     "category": p.category, "priority": p.priority}
                                    for (g, _), p in zip(gold[m.id], preds)]})
        usage = getattr(normaliser, "usage", None)
        if usage is not None:
            board.input_tokens, board.output_tokens = usage.input_tokens, usage.output_tokens
        row = {"normaliser": no_name, "meetings": len(meetings), **board.metrics(), "errors": errors}
        row["seconds_per_meeting"] = row["seconds"] / max(len(meetings), 1)
        row.update(_cost_fields(row, model_of(no_name, "normaliser")))
        rows.append(row)
        log(f"normalise {no_name:<24} asset known {row.get('component@known') or 0:.2f}  "
            f"heldout {row.get('component@heldout') or 0:.2f}")
    meta = {"stage": "normalise", "split": split, "data_dir": rel(data_dir), "set": set_name(data_dir),
            "writer": meetings[0].writer, "meetings": len(meetings), "train_meetings": len(train), "run_dir": rel(run_dir)}
    with (run_dir / "picks.jsonl").open("w", encoding="utf-8") as f:
        for r in picks:
            f.write(json.dumps(r, default=str) + "\n")
    _write("normalise", run_dir, rows, [], meta, merge)
    return rows
