"""Data models shared by every stage.

Flow of data through the experiment:

    GoldTask  (answer key, created first by the scenario sampler)
       |  rendered into
    Meeting   (Teams-style transcript + the gold tasks it contains)
       |  extractor
    RawTask   (what people *said*: free-text mentions, no IDs)
       |  normaliser
    NormalisedTask (canonical IDs, category, priority, resolved dates)
       |  evaluate
    metrics
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AliasSplit = Literal["name", "seen", "heldout", "none"]


# --------------------------------------------------------------------------- gold
class Mention(BaseModel):
    """How a canonical entity was actually referred to in the transcript."""

    entity_id: str | None
    text: str | None
    split: AliasSplit = "none"


class GoldTask(BaseModel):
    """One task that genuinely should end up on the dashboard."""

    id: str
    summary: str  # canonical wording, uses official names
    spoken_action: str  # how it is said aloud, uses the chosen mentions
    category: str
    priority: int
    owner: Mention
    component: Mention
    site: Mention
    due_date: date | None = None
    due_time: str | None = None  # "HH:MM", only when a clock time was stated
    due_mention: str | None = None
    priority_cue: str | None = None
    assigned_by: str | None = None  # person id of whoever asked; None = volunteered
    reassigned_from: str | None = None  # person id of the original (wrong) owner
    evidence: list[int] = Field(default_factory=list)  # utterance indices


class Distractor(BaseModel):
    """Transcript content that must NOT become a task."""

    kind: Literal["chitchat", "done", "observation", "blocker", "cancelled"]
    speaker_id: str
    text_hint: str
    component: Mention | None = None


class Utterance(BaseModel):
    start: float  # seconds from meeting start
    end: float
    speaker_id: str
    speaker: str
    text: str


class Meeting(BaseModel):
    id: str
    date: date
    split: Literal["train", "test"]
    renderer: str
    writer: str | None = None  # models.yaml id of the LLM that wrote the transcript (None = template)
    attendees: list[str]  # person ids
    chair_id: str
    gold_tasks: list[GoldTask]
    distractors: list[Distractor]
    utterances: list[Utterance] = Field(default_factory=list)
    generation_notes: list[str] = Field(default_factory=list)

    # --------------------------------------------------------------- rendering
    def transcript_text(self) -> str:
        """The text an extractor sees: Teams display format, one line per turn."""
        lines = []
        for u in self.utterances:
            lines.append(f"[{_hms(u.start)}] {u.speaker}: {u.text}")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """The same transcript as a Teams .vtt export (for realism / other tools)."""
        out = ["WEBVTT", ""]
        for u in self.utterances:
            out.append(f"{_hms(u.start, ms=True)} --> {_hms(u.end, ms=True)}")
            out.append(f"<v {u.speaker}>{u.text}</v>")
            out.append("")
        return "\n".join(out)


def _hms(seconds: float, ms: bool = False) -> str:
    td = timedelta(seconds=seconds)
    h, rem = divmod(int(td.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    if ms:
        return f"{h:02d}:{m:02d}:{s:02d}.{int((seconds % 1) * 1000):03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------ extraction
class RawTask(BaseModel):
    """Extractor output. Mentions are verbatim (or pronoun-resolved) surface text.

    Extractors that link directly to the ontology (e.g. ``llm_direct``) may also fill
    the ``*_id`` fields; the ``passthrough`` normaliser then keeps them as-is.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str
    owner_mention: str | None = None
    component_mention: str | None = None
    site_mention: str | None = None
    due_mention: str | None = None
    due_date: date | None = None
    due_time: str | None = None
    priority_mention: str | None = None
    priority: int | None = None
    category: str | None = None
    evidence_quote: str | None = None
    confidence: float | None = None
    owner_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None


class ExtractionResult(BaseModel):
    meeting_id: str
    extractor: str
    tasks: list[RawTask]
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None  # set when the call failed; the meeting then scores as "found nothing"


# --------------------------------------------------------------- normalisation
class NormalisedTask(BaseModel):
    summary: str
    owner_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None
    due_date: date | None = None
    due_time: str | None = None
    priority: int | None = None
    category: str | None = None
    evidence_quote: str | None = None
    raw: RawTask
    link_scores: dict[str, float] = Field(default_factory=dict)
