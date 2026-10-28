"""``standup-exp board ...``: the task-board experiment (create cards, then keep them up to date)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def cmd_generate(args) -> None:
    from .runner import generate

    if args.writer:
        os.environ["FOUNDRY_GEN_MODEL"] = args.writer
    generate(args.series, args.days, args.seed, args.renderer, args.out, args.workers, args.overwrite)


def cmd_show(args) -> None:
    from .runner import show

    print(show(args.day, args.data))


def cmd_run(args) -> None:
    from .runner import run

    if args.cache_only:
        os.environ["LLM_CACHE_ONLY"] = "1"
    path = run(args.model, args.data, tuple(m.strip() for m in args.modes.split(",") if m.strip()), args.workers,
               pipeline=args.pipeline)
    print(f"\nWrote {path}")


def cmd_ui(args) -> None:
    from .export import build

    path = build(args.data, args.model, args.pipeline)
    print(f"Open {path.as_uri()}")
    if args.pipeline == "context":  # the explorer needs the context pipeline's recorded steps and traces
        from .explorer import build_explorer

        print(f"Open {build_explorer(args.data, args.model).as_uri()}")


def cmd_demo(args) -> None:
    from .demo import serve

    serve(args.model, args.data, args.port, args.open)


def register(sub: argparse._SubParsersAction) -> None:
    from .runner import SERIES_DIR

    b = sub.add_parser("board", help="task board over a series of stand-ups (create + update cards)")
    bs = b.add_subparsers(dest="board_command", required=True)

    g = bs.add_parser("generate", help="simulate stand-up series and write their transcripts")
    g.add_argument("--series", type=int, default=3, help="number of series (teams), default 3")
    g.add_argument("--days", type=int, default=8, help="consecutive working days per series, default 8")
    g.add_argument("--seed", type=int, default=21)
    g.add_argument("--renderer", choices=["llm", "template"], default="llm")
    g.add_argument("--writer", help="models.yaml id of the transcript writer (default: catalogue default)")
    g.add_argument("--out", type=Path, default=SERIES_DIR)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(func=cmd_generate)

    s = bs.add_parser("show", help="print one stand-up with its board and answer key")
    s.add_argument("day", help="stand-up id, e.g. s01d03")
    s.add_argument("--data", type=Path, default=SERIES_DIR)
    s.set_defaults(func=cmd_show)

    r = bs.add_parser("run", help="run the board loop with one model and score it")
    r.add_argument("--pipeline", choices=["context", "whole-board"], default="context",
                   help="context = POC (extract -> board context -> pick); whole-board = v0")
    r.add_argument("--model", default="gpt-5.6-luna", help="models.yaml id for every step (default gpt-5.6-luna)")
    r.add_argument("--data", type=Path, default=SERIES_DIR)
    r.add_argument("--modes", default="chained",
                   help="chained (default: tomorrow starts from our board) and/or isolated (diagnostic)")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--cache-only", action="store_true", help="replay recorded LLM replies only")
    r.set_defaults(func=cmd_run)

    u = bs.add_parser("ui", help="build the static pages: mock board + dataset & trace explorer (docs/04-board-pipeline/)")
    u.add_argument("--pipeline", choices=["context", "whole-board"], default="context")
    u.add_argument("--data", type=Path, default=SERIES_DIR)
    u.add_argument("--model", default="gpt-5.6-luna", help="which results/board/latest-<pipeline>-<model>.json to show")
    u.set_defaults(func=cmd_ui)

    d = bs.add_parser("demo", help="live local demo: run stand-ups one at a time and watch the board update")
    d.add_argument("--model", default="gpt-5.6-luna")
    d.add_argument("--data", type=Path, default=SERIES_DIR)
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--open", action="store_true", help="open the browser")
    d.set_defaults(func=cmd_demo)
