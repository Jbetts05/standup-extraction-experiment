"""Command line interface: ``standup-exp <command>``.

    generate   sample answer keys and render transcripts
    show       print one meeting's transcript and its gold tasks
    run        score extractor x normaliser combinations (and/or normalisation only)
    finetune   fine-tune GLiNER2 on the train split
    report     rebuild docs/01-extraction-experiment/RESULTS.md from the latest runs
    demo       offline end-to-end: template transcripts + non-LLM methods
    board      task board over a series of stand-ups: generate / show / run / ui
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .paths import DATA_DIR, MEETINGS_DIR

OFFLINE_EXTRACTORS = ["rules"]
OFFLINE_NORMALISERS = ["alias", "fuzzy", "classifier"]
ALL_EXTRACTORS = ["rules", "gliner", "gliner_decide", "gliner_ft", "gliner_ft_all", "llm", "llm_direct"]
ALL_NORMALISERS = ["alias", "fuzzy", "embedding", "embedding_oai", "classifier", "gliner", "gliner_decide", "llm",
                   "passthrough"]


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_generate(args) -> None:
    import os
    from concurrent.futures import ThreadPoolExecutor

    from .dataset import save_meeting
    from .generate.render import render_llm, render_template
    from .generate.scenario import sample_dataset, sample_extra_train
    from .llm import Usage
    from .ontology import Ontology

    if getattr(args, "writer", None):
        os.environ["FOUNDRY_GEN_MODEL"] = args.writer
    onto = Ontology.load()
    prefix = getattr(args, "prefix", "m")
    if getattr(args, "extra_train", 0):
        meetings = sample_extra_train(onto, args.n, args.extra_train, args.seed, prefix)
    else:
        meetings = sample_dataset(onto, args.n, args.seed, args.test_fraction, prefix)
    todo = []
    for m in meetings:
        if (Path(args.out) / f"{m.id}.json").exists() and not getattr(args, "overwrite", False):
            print(f"{m.id} exists, skipping (use --overwrite to regenerate)")
        else:
            todo.append(m)

    def one(m):
        if args.renderer == "template":
            render_template(m, onto)
            u = Usage()
        else:
            try:
                _, u = render_llm(m, onto)
            except Exception as exc:  # noqa: BLE001 - one failed meeting must not stop the batch
                print(f"{m.id} ERROR {str(exc)[:160]}")
                return None
        save_meeting(m, args.out)
        status = m.generation_notes[0] if m.generation_notes else "ok"
        print(f"{m.id} {m.split:<5} {m.date} {len(m.gold_tasks)} tasks {len(m.utterances):>3} turns  {status}", flush=True)
        return m, u

    workers = getattr(args, "workers", 1) if args.renderer == "llm" else 1
    with ThreadPoolExecutor(max(1, workers)) as pool:
        done = [r for r in pool.map(one, todo) if r]
    total = Usage()
    for _, u in done:
        total += u
    failed = sum(any(n.startswith("FAILED") for n in m.generation_notes) for m, _ in done)
    print(f"\nWrote {len(done)} meetings to {args.out}"
          + (f" ({failed} failed validation and will be skipped)" if failed else ""))
    if total.input_tokens:
        print(f"Generation tokens: {total.input_tokens} in / {total.output_tokens} out")


def cmd_show(args) -> None:
    from .dataset import load_meetings
    from .ontology import Ontology

    onto = Ontology.load()
    m = next((m for m in load_meetings(args.data) if m.id == args.meeting), None)
    if m is None:
        raise SystemExit(f"No meeting {args.meeting} in {args.data}")
    print(f"# {m.id} ({m.split}) {m.date:%A %d %B %Y}  renderer={m.renderer}\n")
    print(m.transcript_text())
    print("\n# Gold tasks")
    for g in m.gold_tasks:
        owner = onto.by_id[g.owner.entity_id].name if g.owner.entity_id else "-"
        print(f"- [{g.category}/P{g.priority}] {g.summary} | owner={owner} ({g.owner.text}) "
              f"| component={g.component.entity_id} ('{g.component.text}', {g.component.split}) "
              f"| site={g.site.entity_id} | due={g.due_date} {g.due_time or ''} | turns={g.evidence}")


def _expand(names: list[str], llm_bases: set[str], models: list[str]) -> list[str]:
    """``llm`` -> ``llm@m1, llm@m2, ...`` for every requested model."""
    if not models:
        return names
    out = []
    for n in names:
        out += [f"{n}@{m}" for m in models] if n in llm_bases else [n]
    return out


def cmd_run(args) -> None:
    import os

    from .llm import model_ids
    from .report import build_report
    from .runner import run_e2e, run_normalise

    if args.cache_only:
        os.environ["LLM_CACHE_ONLY"] = "1"

    models = model_ids() if args.models == "all" else _csv(args.models or "")
    extractors = _expand(_csv(args.extractors), {"llm", "llm_direct"}, models)
    normalisers = _expand(_csv(args.normalisers), {"llm"}, models)
    if args.stage in ("e2e", "both"):
        run_e2e(extractors, normalisers, args.split, args.data, args.refresh, merge=args.merge)
    if args.stage in ("normalise", "both"):
        run_normalise(normalisers, args.split, args.data, merge=args.merge)
    build_report()
    print("\nUpdated docs/01-extraction-experiment/RESULTS.md")


def cmd_finetune(args) -> None:
    from .finetune import finetune

    path = finetune(args.data, args.epochs, args.batch_size, out_name=args.out_name,
                    negatives_per_job=args.negatives, eval_steps=args.eval_steps)
    print(f"Fine-tuned checkpoint: {path}")


def cmd_report(args) -> None:
    from .report import build_report

    print(build_report())


def cmd_walkthrough(args) -> None:
    from .walkthrough import build

    path = build(args.data, reuse=args.reuse)
    print(f"Open {path.as_uri()}")


def cmd_demo(args) -> None:
    """Fully offline: no Azure, no model downloads. Results go to their own 'meetings-demo' set."""
    from .report import build_report
    from .runner import run_e2e, run_normalise

    out = DATA_DIR / "meetings-demo"
    cmd_generate(argparse.Namespace(n=args.n, seed=args.seed, test_fraction=0.4, renderer="template", out=out,
                                    overwrite=True))
    run_e2e(OFFLINE_EXTRACTORS, OFFLINE_NORMALISERS, "test", out)
    run_normalise(OFFLINE_NORMALISERS, "test", out)
    print(build_report(sets=["meetings-demo"], write=False))


def main(argv: list[str] | None = None) -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(prog="standup-exp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create synthetic meetings")
    g.add_argument("--n", type=int, default=20, help="number of meetings (default 20)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--test-fraction", type=float, default=0.4)
    g.add_argument("--renderer", choices=["llm", "template"], default="llm")
    g.add_argument("--out", type=Path, default=MEETINGS_DIR)
    g.add_argument("--overwrite", action="store_true", help="regenerate meetings that already exist")
    g.add_argument("--writer", help="models.yaml id of the LLM that writes transcripts (default: catalogue default)")
    g.add_argument("--prefix", default="m", help="meeting id prefix, e.g. 'a' for a second dataset")
    g.add_argument("--extra-train", type=int, default=0,
                   help="add this many TRAIN meetings with ids starting at --n (test split untouched)")
    g.add_argument("--workers", type=int, default=6, help="parallel LLM writers")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("show", help="print a meeting and its answer key")
    s.add_argument("meeting", help="meeting id, e.g. m012")
    s.add_argument("--data", type=Path, default=MEETINGS_DIR)
    s.set_defaults(func=cmd_show)

    r = sub.add_parser("run", help="run and score methods")
    r.add_argument("--extractors", default=",".join(ALL_EXTRACTORS),
                   help=f"comma list from: {', '.join(ALL_EXTRACTORS)}")
    r.add_argument("--normalisers", default=",".join(ALL_NORMALISERS),
                   help=f"comma list from: {', '.join(ALL_NORMALISERS)}")
    r.add_argument("--models", default=None,
                   help="LLM ids from data/models.yaml (comma list) or 'all'; expands llm/llm_direct per model")
    r.add_argument("--stage", choices=["e2e", "normalise", "both"], default="both")
    r.add_argument("--split", choices=["train", "test", "all"], default="test")
    r.add_argument("--data", type=Path, default=MEETINGS_DIR)
    r.add_argument("--refresh", action="store_true", help="ignore cached extraction output")
    r.add_argument("--cache-only", action="store_true",
                   help="replay recorded LLM replies only; calls that never completed count as failures")
    r.add_argument("--merge", action="store_true",
                   help="update matching rows in the latest results instead of replacing them")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("finetune", help="fine-tune GLiNER 2.5 base on train meetings")
    f.add_argument("--data", type=Path, default=None, help="one dataset dir (default: train meetings from every writer)")
    f.add_argument("--epochs", type=int, default=1)
    f.add_argument("--eval-steps", type=int, default=90, help="evaluate + keep best every N steps (0 = per epoch)")
    f.add_argument("--batch-size", type=int, default=8)
    f.add_argument("--out-name", default="gliner-standup-all", help="folder under models/")
    f.add_argument("--negatives", type=int, default=2, help="non-job turns sampled per job turn")
    f.set_defaults(func=cmd_finetune)

    sub.add_parser("report", help="rebuild docs/01-extraction-experiment/RESULTS.md").set_defaults(func=cmd_report)

    w = sub.add_parser("walkthrough", help="build the slide-deck demo (docs/02-walkthrough/v1-deck/index.html)")
    w.add_argument("--data", type=Path, default=MEETINGS_DIR)
    w.add_argument("--reuse", action="store_true", help="re-render from the last export (after template edits)")
    w.set_defaults(func=cmd_walkthrough)

    d = sub.add_parser("demo", help="offline demo with template transcripts")
    d.add_argument("--n", type=int, default=20)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)

    from .board.cli import register as register_board

    register_board(sub)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
