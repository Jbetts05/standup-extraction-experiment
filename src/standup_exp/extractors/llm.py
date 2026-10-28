"""LLM extractors using a fixed JSON schema (Azure AI Foundry structured outputs).

* ``llm``        - extracts surface mentions only; a separate normaliser links them.
* ``llm_direct`` - single-step baseline: the prompt also contains the business
                   vocabulary and the model returns canonical IDs directly.
                   Use it with the ``passthrough`` normaliser.
"""

from __future__ import annotations

import json
import time
from typing import Literal

from pydantic import BaseModel, Field

from ..llm import structured
from ..ontology import Ontology
from ..schema import ExtractionResult, RawTask
from .base import Extractor, TranscriptView

Category = Literal["operations", "safety", "organisation"]


class _Task(BaseModel):
    summary: str = Field(description="Short imperative description of the job, e.g. 'Replace seals on GT2 water valve'.")
    owner_mention: str | None = Field(description="Who will do it, as named in the meeting (full name if known). Null if nobody took it.")
    component_mention: str | None = Field(description="Equipment/asset exactly as people referred to it. Resolve 'it'/'that one' to the thing meant. Null if none.")
    site_mention: str | None = Field(description="Off-site location to visit, as said. Null if none.")
    due_mention: str | None = Field(description="Deadline words as spoken, e.g. 'by Thursday', 'tomorrow at half two'. Null if none.")
    due_date: str | None = Field(description="Deadline as YYYY-MM-DD using the meeting date and the date rules. Null if none.")
    due_time: str | None = Field(description="Clock time as HH:MM (24h) only if a specific time was said, else null.")
    priority_mention: str | None = Field(description="Urgency words as spoken, e.g. 'no rush', 'top priority'. Null if none.")
    priority: int | None = Field(description="1, 2 or 3 per the priority rubric; 2 when no urgency was expressed.")
    category: Category | None
    evidence_quote: str = Field(description="The single transcript sentence where the job is stated or agreed, copied verbatim.")
    confidence: float = Field(description="0-1 confidence that this is a real new job for the dashboard.")


class _Tasks(BaseModel):
    tasks: list[_Task]


class _DirectTask(_Task):
    owner_id: str | None = Field(description="Person id from the people list, or null.")
    component_id: str | None = Field(description="Asset id from the asset list (use the parent id if the exact part is unclear), or null.")
    site_id: str | None = Field(description="Site id from the site list, or null.")


class _DirectTasks(BaseModel):
    tasks: list[_DirectTask]


INSTRUCTIONS = """You extract jobs from a factory morning stand-up transcript for a task dashboard.

A job is a piece of work someone agreed to do, was asked to do, or that the meeting said
needs doing (even with no owner yet).

Do NOT extract:
- work already finished ("tested it yesterday, all fine"),
- observations with no action ("it ran a bit warm but settled"),
- blockers that are just waiting on something,
- cancelled or postponed-indefinitely work, and chit-chat.

Rules:
- One job per action. Merge the ask and the acceptance of the same job into one record.
- If a job is handed from one person to another, the owner is whoever finally accepted it.
- If someone says "I'll ...", the owner is that speaker.
- Copy mentions verbatim from the transcript; do not "correct" nicknames.
- Leave any field null when it was not said. Never guess.
- Date rules (relative to the meeting date): "today"/"this afternoon"/"end of day" = meeting date;
  "tomorrow" = +1 day; a bare weekday = its next occurrence after today; "next <weekday>" =
  that weekday in the following Monday-Sunday week; "end of the week" = this Friday;
  "next week" = next Monday; "the 3rd" = next occurrence of that day of month.
  Times: "half two" = 14:30, bare 1-6 o'clock = afternoon.

""" + Ontology.load().rubric()


def _header(view: TranscriptView) -> str:
    return (
        f"Meeting date: {view.date.isoformat()} ({view.date.strftime('%A')})\n"
        f"Attendees: {', '.join(view.attendees)}\n\nTranscript:\n{view.text}"
    )


def _to_raw(t: _Task) -> RawTask:
    data = t.model_dump()
    if data.get("due_date"):
        try:
            RawTask.model_validate({"summary": "x", "due_date": data["due_date"]})
        except Exception:  # noqa: BLE001 - drop malformed dates rather than fail
            data["due_date"] = None
    return RawTask.model_validate(data)


def _model_name(base: str, model: str | None) -> str:
    return f"{base}@{model}" if model else base


class LLMExtractor(Extractor):
    name = "llm"
    parallel = True  # network bound: meetings run in threads

    def __init__(self, model: str | None = None):
        self.model = model
        self.name = _model_name("llm", model)

    def extract(self, view: TranscriptView) -> ExtractionResult:
        t0 = time.perf_counter()
        out, usage = structured("standup_tasks", INSTRUCTIONS, _header(view), _Tasks, model=self.model)
        return ExtractionResult(
            meeting_id=view.meeting_id, extractor=self.name, tasks=[_to_raw(t) for t in out.tasks],
            seconds=usage.seconds if usage.cached else time.perf_counter() - t0,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        )


class LLMDirectExtractor(Extractor):
    """Extract and link in one call. The vocabulary excludes held-out aliases."""

    name = "llm_direct"
    parallel = True

    def __init__(self, model: str | None = None, ontology: Ontology | None = None):
        self.model = model
        self.name = _model_name("llm_direct", model)
        onto = ontology or Ontology.load()
        vocab = {
            "people": [{"id": p.id, "name": p.name, "aliases": list(p.aliases)} for p in onto.people],
            "sites": [{"id": s.id, "name": s.name, "aliases": list(s.aliases)} for s in onto.sites],
            "assets": [{"id": a.id, "name": a.name, "parent": a.parent, "aliases": list(a.aliases)} for a in onto.assets],
        }
        self.instructions = (
            INSTRUCTIONS
            + "\n\nAlso link each mention to the business vocabulary below and fill owner_id, component_id and site_id."
            " People may use nicknames that are not listed; pick the best match or null.\n"
            + json.dumps(vocab)
        )

    def extract(self, view: TranscriptView) -> ExtractionResult:
        t0 = time.perf_counter()
        out, usage = structured("standup_tasks_direct", self.instructions, _header(view), _DirectTasks,
                                model=self.model)
        return ExtractionResult(
            meeting_id=view.meeting_id, extractor=self.name, tasks=[_to_raw(t) for t in out.tasks],
            seconds=usage.seconds if usage.cached else time.perf_counter() - t0,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        )
