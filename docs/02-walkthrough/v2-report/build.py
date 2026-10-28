"""Build the v2 experiment report (docs/02-walkthrough/v2-report/index.html).

Uses exactly the same data export as the original deck, so nothing is re-run:

    python docs/02-walkthrough/v2-report/build.py                 # results/walkthrough-data.json (written by `standup-exp walkthrough`)
    python docs/02-walkthrough/v2-report/build.py --from-index    # re-use the payload already embedded in docs/02-walkthrough/v1-deck/index.html
    .venv/Scripts/python docs/02-walkthrough/v2-report/build.py --fresh   # export the latest results/latest-*.json now (needs the project venv)

--fresh calls standup_exp.walkthrough.export(), which only reads files; it does not rewrite the
original deck or results/walkthrough-data.json.

Standard library only. The original deck (docs/02-walkthrough/v1-deck/) is not touched.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEMPLATE = HERE / "template.html"
OUTPUT = HERE / "index.html"
DATA_CACHE = ROOT / "results" / "walkthrough-data.json"
OLD_INDEX = HERE.parent / "v1-deck" / "index.html"


def _from_index(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    m = re.search(r'id="payload"[^>]*>([^<]+)<', html)
    if not m:
        raise SystemExit(f"No embedded payload found in {path}")
    return json.loads(gzip.decompress(base64.b64decode(m.group(1).strip())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-index", action="store_true", help="read the payload embedded in docs/02-walkthrough/v1-deck/index.html")
    ap.add_argument("--data", type=Path, help="explicit walkthrough-data.json path")
    ap.add_argument("--fresh", action="store_true", help="export straight from the latest results (project venv required)")
    args = ap.parse_args()

    if args.fresh:
        sys.path.insert(0, str(ROOT / "src"))
        from standup_exp.walkthrough import export
        data, src = json.loads(json.dumps(export(), default=str)), "a fresh export of results/latest-*.json"
    elif args.data:
        data = json.loads(args.data.read_text(encoding="utf-8"))
        src = args.data.resolve()
    elif args.from_index or not DATA_CACHE.exists():
        data, src = _from_index(OLD_INDEX), OLD_INDEX
    else:
        data, src = json.loads(DATA_CACHE.read_text(encoding="utf-8")), DATA_CACHE

    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    OUTPUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA_GZ_B64__", payload), encoding="utf-8")
    print(f"{OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB, data from {src})")


if __name__ == "__main__":
    main()
