"""Project paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ONTOLOGY_PATH = DATA_DIR / "ontology.yaml"
MEETINGS_DIR = DATA_DIR / "meetings"
# Every dataset directory: data/meetings (written by GPT-5.4) plus data/meetings-* (other writers).
# Train splits from all of them feed fine-tuning and the trained classifier.
def dataset_dirs() -> list[Path]:
    return [d for d in [MEETINGS_DIR, *sorted(DATA_DIR.glob("meetings-*"))]
            if d.is_dir() and d.name != "meetings-demo" and any(d.glob("*.json"))]


def set_name(data_dir: Path) -> str:
    """Stable id for a test set, used in result file names: 'meetings', 'meetings-astra', ..."""
    return Path(data_dir).name
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
MODELS_DIR = ROOT / "models"

# Write-ups, reports and UIs live under docs/, one folder per approach.
DOCS_DIR = ROOT / "docs"
EXPERIMENT_DOCS = DOCS_DIR / "01-extraction-experiment"   # RESULTS.md
WALKTHROUGH_DOCS = DOCS_DIR / "02-walkthrough"            # v1-deck/, v2-report/
AGENT_DOCS = DOCS_DIR / "03-agent"
BOARD_DOCS = DOCS_DIR / "04-board-pipeline"               # mock-board/

load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def flag(name: str) -> bool:
    """An on/off switch from the environment: 1, true, yes or on (any case) mean on; anything else, including 0, is off."""
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def rel(path) -> str:
    """A path as it's stored in shared results: relative to the repo with forward slashes, so no machine paths leak."""
    p = Path(path)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def from_rel(path: str) -> Path:
    """Undo ``rel``: repo-relative paths resolve against the repo; absolute paths (older results) are kept."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
