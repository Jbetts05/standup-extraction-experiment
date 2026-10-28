"""Export everything the walkthrough deck needs into one self-contained HTML file.

    standup-exp walkthrough            ->  docs/02-walkthrough/v1-deck/index.html
    standup-exp walkthrough --reuse    ->  re-render from the last export (template edits only)

Nothing is re-run and nothing is sent to Azure. The export reads the per-meeting records the
last ``standup-exp run`` wrote for each test set (``pipelines.jsonl``, ``extractions.jsonl``,
``picks.jsonl``), every meeting (train and test, all writers), the exact prompts and JSON
schemas the LLMs received, and the recorded costs. The payload is gzip-compressed and embedded
in ``docs/02-walkthrough/v1-deck/template.html`` so the deck opens from disk with no server.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from .dataset import load_all_train, load_meetings
from .llm import CACHE_DIR, MODELS_PATH, embedding_spec, strict_schema
from .ontology import Ontology
from .paths import RESULTS_DIR, WALKTHROUGH_DOCS, dataset_dirs, from_rel, set_name

TEMPLATE = WALKTHROUGH_DOCS / "v1-deck" / "template.html"
OUTPUT = WALKTHROUGH_DOCS / "v1-deck" / "index.html"
DATA_CACHE = RESULTS_DIR / "walkthrough-data.json"
SET_LABEL = {"meetings": "GPT-5.4-written", "meetings-astra": "GPT-6 Astra-written"}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mention(m) -> dict:
    return {"id": m.entity_id, "text": m.text, "split": m.split}


def _find_turn(meeting, window: list[int], phrase: str | None) -> int | None:
    if not phrase:
        return None
    for i in window:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", meeting.utterances[i].text, re.I):
            return i
    return None


def _checks(meeting, g) -> list[dict]:
    """Human-readable version of the validator's checks for one gold task."""
    if not g.evidence:
        return []
    lo, hi = max(0, min(g.evidence) - 1), min(len(meeting.utterances), max(g.evidence) + 2)
    window = list(range(lo, hi))
    out = []
    for label, phrase in (("equipment", g.component.text), ("site", g.site.text), ("owner", g.owner.text),
                          ("deadline", g.due_mention), ("urgency", g.priority_cue)):
        if phrase:
            out.append({"what": label, "phrase": phrase, "turn": _find_turn(meeting, window, phrase)})
    return out


def _meeting_json(meeting, set_: str) -> dict:
    return {
        "id": meeting.id, "set": set_, "date": meeting.date.isoformat(), "weekday": meeting.date.strftime("%A"),
        "split": meeting.split, "writer": meeting.writer, "chair": meeting.chair_id,
        "attendees": meeting.attendees, "notes": meeting.generation_notes,
        "turns": [{"t": round(u.start), "sid": u.speaker_id, "who": u.speaker, "text": u.text}
                  for u in meeting.utterances],
        "gold": [{
            "id": g.id, "summary": g.summary, "spoken": g.spoken_action, "category": g.category,
            "priority": g.priority, "owner": _mention(g.owner), "component": _mention(g.component),
            "site": _mention(g.site), "due_date": g.due_date.isoformat() if g.due_date else None,
            "due_time": g.due_time, "due_mention": g.due_mention, "priority_cue": g.priority_cue,
            "assigned_by": g.assigned_by, "reassigned_from": g.reassigned_from, "evidence": g.evidence,
            "checks": _checks(meeting, g),
        } for g in meeting.gold_tasks],
        "distractors": [{"kind": d.kind, "sid": d.speaker_id, "text": d.text_hint} for d in meeting.distractors],
    }


def _generation_costs() -> dict:
    """Tokens and cost of writing transcripts, per writer, from the LLM reply cache."""
    from .llm import cost_usd

    out: dict = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0})
    for f in (CACHE_DIR / "llm").glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "utterances" not in (data.get("output") or {}):
            continue
        w = data.get("model") or "gpt-5.4"  # early cache entries predate the model field (all GPT-5.4)
        u = data["usage"]
        o = out[w]
        o["calls"] += 1
        o["input_tokens"] += u["input_tokens"]
        o["output_tokens"] += u["output_tokens"]
        o["seconds"] += u["seconds"]
    for w, o in out.items():
        o["cost"] = cost_usd(o["input_tokens"], o["output_tokens"], w)
    return dict(out)


def _embedding_usage() -> dict:
    rows = _jsonl(CACHE_DIR / "embed" / "usage.jsonl")
    spec = embedding_spec()
    tokens = sum(r["tokens"] for r in rows)
    vectors = sum(1 for _ in (CACHE_DIR / "embed" / spec.get("deployment", "_")).glob("*.npy"))
    return {"model": spec.get("id"), "deployment": spec.get("deployment"), "tokens": tokens,
            "calls": len(rows), "vectors": vectors, "cost": tokens / 1e6 * float(spec.get("price") or 0),
            "note": None if rows else "ledger started after the first calls; token count is a lower bound"}


def _finetune_json(log: Path) -> dict:
    out: dict = {"eval_loss": [], "examples": None, "meetings": None, "seconds": None, "epochs": None}
    parts = [p for p in (log, log.with_suffix(".err")) if p.exists()]
    if not parts:
        return out
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in parts)
    out["eval_loss"] = [float(x) for x in re.findall(r"New best eval_loss: ([\d.]+)", text)]
    m = re.search(r"training examples: (\d+)", text)
    out["examples"] = int(m.group(1)) if m else None
    m = re.search(r"train meetings: (\d+)", text)
    out["meetings"] = int(m.group(1)) if m else None
    m = re.search(r"'total_time_seconds': ([\d.]+)", text)
    out["seconds"] = float(m.group(1)) if m else None
    m = re.search(r"'total_epochs': (\d+)", text)
    out["epochs"] = int(m.group(1)) if m else None
    return out


def _prompts(onto: Ontology) -> dict:
    """The exact instructions and JSON schemas the LLMs received."""
    from .extractors import llm as ex_llm
    from .generate import render
    from .normalisers import llm as no_llm

    direct = ex_llm.LLMDirectExtractor(ontology=onto).instructions
    cut = direct.find("\n{")
    return {
        "writer": {"instructions": render.RENDER_INSTRUCTIONS, "schema": strict_schema(render._LLMTranscript)},
        "extract": {"instructions": ex_llm.INSTRUCTIONS, "schema": strict_schema(ex_llm._Tasks),
                    "input": "Meeting date: <date> (<weekday>)\nAttendees: <names>\n\nTranscript:\n[HH:MM:SS] Name: text ..."},
        "direct": {"instructions": direct[:cut] if cut > 0 else direct, "vocab_chars": len(direct) - max(cut, 0),
                   "schema": strict_schema(ex_llm._DirectTasks)},
        "match": {"instructions": f"{no_llm.INSTRUCTIONS}\n\n{onto.rubric()}", "schema": strict_schema(no_llm._Choices)},
    }


def _shortlists(onto: Ontology, sets: dict) -> dict:
    """Candidates the LLM / GLiNER matchers chose from, per extracted mention (deterministic)."""
    from .normalisers import NormContext
    from .normalisers.embedding import HybridShortlist

    hs = HybridShortlist(onto, [])
    out: dict = {}
    for set_, s in sets.items():
        meetings = {m["id"]: m for m in s["meetings"]}
        for rec in s["extractions_raw"]:
            m = meetings.get(rec["meeting"])
            if not m:
                continue
            ctx = NormContext(m["id"], datetime.fromisoformat(m["date"]).date(), tuple(m["attendees"]))
            lists = []
            for t in rec["tasks"]:
                c = {}
                for field, kind in (("owner", "person"), ("component", "asset"), ("site", "site")):
                    if t.get(field):
                        ids = hs.shortlist_ids(kind, t[field], ctx)
                        if kind == "asset":
                            ids += [p for i in ids if (p := onto.by_id[i].parent) and p not in ids]
                        c[field] = ids
                lists.append(c)
            out.setdefault(set_, {}).setdefault(rec["meeting"], {})[rec["extractor"]] = lists
    return out


def export() -> dict:
    onto = Ontology.load()
    catalogue = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    sets: dict = {}
    all_meetings = []
    for d in dataset_dirs():
        ms = load_meetings(d)
        name = set_name(d)
        all_meetings += [_meeting_json(m, name) for m in ms]
        latest = RESULTS_DIR / f"latest-e2e-{name}.json"
        if not latest.exists():
            continue
        e2e = json.loads(latest.read_text(encoding="utf-8"))
        run_dirs = [from_rel(d) for d in (e2e["meta"].get("run_dirs") or [e2e["meta"]["run_dir"]])]
        norm_path = RESULTS_DIR / f"latest-normalise-{name}.json"
        norm = json.loads(norm_path.read_text(encoding="utf-8")) if norm_path.exists() else {"rows": [], "meta": {}}
        norm_dirs = [from_rel(d) for d in (norm["meta"].get("run_dirs") or [norm["meta"].get("run_dir")]) if d]
        test_ids = {m.id for m in ms if m.split == "test"}
        sets[name] = {
            "label": SET_LABEL.get(name, name), "writer": e2e["meta"].get("writer"), "meta": e2e["meta"],
            "leaderboard": e2e["rows"], "normalise": norm["rows"],
            "meetings": [x for x in all_meetings if x["set"] == name and x["id"] in test_ids],
            # records from every run dir merged into the latest file; later runs win
            "extractions_raw": [r for d in run_dirs for r in _jsonl(d / "extractions.jsonl")],
            "pipelines_raw": [r for d in run_dirs for r in _jsonl(d / "pipelines.jsonl")],
            "picks_raw": [r for d in norm_dirs for r in _jsonl(d / "picks.jsonl")],
        }
    shortlists = _shortlists(onto, sets)
    for name, s in sets.items():
        ex: dict = defaultdict(dict)
        for r in s.pop("extractions_raw"):
            ex[r["meeting"]][r["extractor"]] = {k: r[k] for k in ("seconds", "tokens", "error", "cost", "tasks")}
        pipes: dict = defaultdict(dict)
        for r in s.pop("pipelines_raw"):
            pipes[r["meeting"]][f"{r['extractor']}|{r['normaliser']}"] = {
                k: r[k] for k in ("preds", "pairs", "fp", "fn", "ready", "seconds", "match_seconds", "tokens", "cost", "failed")}
        picks: dict = defaultdict(dict)
        for r in s.pop("picks_raw"):
            picks[r["meeting"]][r["normaliser"]] = r["picks"]
        s.update(extractions=ex, pipelines=pipes, picks=picks, shortlists=shortlists.get(name, {}))

    train = load_all_train()
    return {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "plant": onto.plant,
        "ontology": {
            "categories": onto.categories, "priorities": onto.priorities,
            "people": [{"id": e.id, "name": e.name, "role": e.extra.get("role"), "aliases": list(e.aliases),
                        "heldout": list(e.heldout_aliases)} for e in onto.people],
            "sites": [{"id": e.id, "name": e.name, "aliases": list(e.aliases), "heldout": list(e.heldout_aliases)}
                      for e in onto.sites],
            "assets": [{"id": e.id, "name": e.name, "parent": e.parent, "kind": e.kind, "aliases": list(e.aliases),
                        "heldout": list(e.heldout_aliases)} for e in onto.assets],
        },
        "models": [{"id": m["id"], "deployment": m["deployment"], "api": m.get("api", "responses"),
                    "endpoint": m["endpoint"], "price": m.get("price"), "timeout": m.get("timeout", 240)}
                   for m in catalogue["models"]],
        "embeddings": catalogue.get("embeddings", []),
        "meetings": all_meetings,
        "train_count": len(train),
        "train_writers": dict(Counter(m.writer for m in train)),
        "sets": sets,
        "prompts": _prompts(onto),
        "generation": _generation_costs(),
        "embedding_usage": _embedding_usage(),
        "finetune": {"gliner_ft": _finetune_json(RESULTS_DIR / "finetune.log"),
                     "gliner_ft_all": _finetune_json(RESULTS_DIR / "v2-ft.log")},
    }


def render(data: dict) -> Path:
    raw = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA_GZ_B64__", payload)
    OUTPUT.write_text(html, encoding="utf-8")
    return OUTPUT


def _from_deck() -> dict:
    """The data already embedded in the built deck, so --reuse works on a fresh clone without the per-run records."""
    m = re.search(r'id="payload"[^>]*>([^<]+)<', OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else None
    if not m:
        raise SystemExit(f"--reuse needs {DATA_CACHE} or a built {OUTPUT}; run without --reuse to export from the results.")
    return json.loads(gzip.decompress(base64.b64decode(m.group(1).strip())))


def build(data_dir: Path | None = None, log=print, reuse: bool = False) -> Path:
    if reuse:  # never calls a model: the saved export, or the data inside the deck that's already built
        data = json.loads(DATA_CACHE.read_text(encoding="utf-8")) if DATA_CACHE.exists() else _from_deck()
    else:
        data = export()
        DATA_CACHE.write_text(json.dumps(data, default=str), encoding="utf-8")
    out = render(data)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB; {len(data['sets'])} test sets, "
        f"{len(data['meetings'])} meetings)")
    return out
