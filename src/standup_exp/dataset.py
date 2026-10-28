"""Read and write generated meetings.

Each meeting is stored as ``<id>.json`` (answer key + transcript) and
``<id>.vtt`` (the same transcript as a Teams export, for eyeballing).
"""

from __future__ import annotations

from pathlib import Path

from .paths import MEETINGS_DIR, dataset_dirs
from .schema import Meeting


def save_meeting(meeting: Meeting, directory: Path = MEETINGS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{meeting.id}.json").write_text(meeting.model_dump_json(indent=1), encoding="utf-8")
    (directory / f"{meeting.id}.vtt").write_text(meeting.to_vtt(), encoding="utf-8")


def load_meetings(directory: Path = MEETINGS_DIR, split: str | None = None) -> list[Meeting]:
    """Load meetings; ``split`` is 'train', 'test' or None/'all'."""
    meetings = [Meeting.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.json"))]
    meetings = [m for m in meetings if not any(n.startswith("FAILED") for n in m.generation_notes)]
    if split and split != "all":
        meetings = [m for m in meetings if m.split == split]
    return meetings


def load_all_train() -> list[Meeting]:
    """Train meetings from every dataset (all writers). Test meetings are never included."""
    return [m for d in dataset_dirs() for m in load_meetings(d, "train")]
