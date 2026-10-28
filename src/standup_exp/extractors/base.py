"""Extractors turn a transcript into ``RawTask`` records (surface mentions, no IDs).

To keep the comparison honest, extractors never see a ``Meeting`` (which holds
the answer key). They receive a ``TranscriptView`` with only what a real system
would have: the meeting date, attendee display names and the transcript.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import date

from ..schema import ExtractionResult, Meeting


@dataclass(frozen=True)
class Turn:
    index: int
    time: str
    speaker: str
    text: str


@dataclass(frozen=True)
class TranscriptView:
    meeting_id: str
    date: date
    attendees: tuple[str, ...]  # display names
    turns: tuple[Turn, ...]
    text: str

    @classmethod
    def of(cls, meeting: Meeting, names: dict[str, str]) -> "TranscriptView":
        lines = meeting.transcript_text().splitlines()
        turns = tuple(
            Turn(i, line[1:9], u.speaker, u.text) for i, (u, line) in enumerate(zip(meeting.utterances, lines))
        )
        return cls(meeting.id, meeting.date, tuple(names[p] for p in meeting.attendees), turns, meeting.transcript_text())


class Extractor:
    """Base class. Subclasses implement ``extract``."""

    name = "base"

    def extract(self, view: TranscriptView) -> ExtractionResult:  # pragma: no cover - interface
        raise NotImplementedError


# name -> "module:Class". Imported lazily so heavy optional deps load only when used.
REGISTRY = {
    "rules": "standup_exp.extractors.rules:RulesExtractor",
    "llm": "standup_exp.extractors.llm:LLMExtractor",
    "llm_direct": "standup_exp.extractors.llm:LLMDirectExtractor",
    "gliner": "standup_exp.extractors.gliner:GlinerExtractor",
    "gliner_ft": "standup_exp.extractors.gliner:FineTunedGlinerExtractor",
    "gliner_ft_all": "standup_exp.extractors.gliner:FineTunedAllGlinerExtractor",
    "gliner_decide": "standup_exp.extractors.gliner:DecideGlinerExtractor",
}
LLM_EXTRACTORS = {"llm", "llm_direct"}


def get_extractor(name: str, **kwargs) -> Extractor:
    """``name`` may pin an LLM: ``llm@gpt-6-sol`` (ids from data/models.yaml)."""
    base, _, model = name.partition("@")
    if base not in REGISTRY:
        raise KeyError(f"Unknown extractor '{name}'. Choose from: {', '.join(REGISTRY)}")
    if model:
        if base not in LLM_EXTRACTORS:
            raise KeyError(f"'{base}' does not use an LLM, so '@{model}' makes no sense.")
        kwargs["model"] = model
    module, cls = REGISTRY[base].split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)
