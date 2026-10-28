"""LLM normaliser: hybrid shortlist, then the model picks the right entity.

Real asset registers have thousands of entries, so the model is never shown the
whole list. For each mention we retrieve fuzzy + embedding candidates (see
``HybridShortlist``) plus their parent assets; the model then picks one id, a
parent (when the exact part is unclear), or null. All tasks in a meeting are
resolved in one call. The model also assigns category and priority.
"""

from __future__ import annotations

import json
import threading
from typing import Literal

from pydantic import BaseModel, Field

from ..llm import Usage, structured
from ..schema import NormalisedTask, RawTask
from .base import NormContext
from .embedding import HybridShortlist


class _Choice(BaseModel):
    task_index: int
    owner_id: str | None
    component_id: str | None
    site_id: str | None
    category: Literal["operations", "safety", "organisation"]
    priority: int | None = Field(description="1, 2 or 3 per the priority rubric; 2 if no urgency was expressed")


class _Choices(BaseModel):
    choices: list[_Choice]


INSTRUCTIONS = """You link jobs extracted from a factory stand-up to the company's master data.

For each task you get the words people used and a shortlist of candidate ids.
- Pick the candidate that the words refer to, using the task wording as context.
- For equipment, if you cannot tell which specific part is meant, pick its parent asset id.
- Return null if none of the candidates fits. Never invent ids.
- Attendee ids are more likely owners than non-attendees.
- category and priority: follow the business rubric below exactly."""


class LLMNormaliser(HybridShortlist):
    name = "llm"
    parallel = True  # network bound: meetings run in threads

    def __init__(self, ontology, train=None, model: str | None = None):
        super().__init__(ontology, train)
        self.model = model
        self.name = f"llm@{model}" if model else "llm"
        self.usage = Usage()
        self.seconds_by_meeting: dict[str, float] = {}
        self.tokens_by_meeting: dict[str, tuple[int, int]] = {}
        self.instructions = f"{INSTRUCTIONS}\n\n{ontology.rubric()}"
        self._lock = threading.Lock()

    def _candidates(self, kind: str, mention: str | None, ctx: NormContext) -> list[dict]:
        if not mention:
            return []
        ids = self.shortlist_ids(kind, mention, ctx)
        if kind == "asset":
            ids += [p for eid in ids if (p := self.onto.by_id[eid].parent) and p not in ids]
        return [{"id": eid, "name": self.onto.by_id[eid].name, "aliases": self.onto.by_id[eid].aliases,
                 "parent": self.onto.by_id[eid].parent} for eid in ids]

    def normalise_meeting(self, tasks: list[RawTask], ctx: NormContext) -> list[NormalisedTask]:
        if not tasks:
            return []
        payload = {
            "attendee_ids": list(ctx.attendee_ids),
            "tasks": [
                {
                    "task_index": i,
                    "summary": t.summary,
                    "evidence": t.evidence_quote,
                    "owner_words": t.owner_mention,
                    "owner_candidates": self._candidates("person", t.owner_mention, ctx),
                    "component_words": t.component_mention,
                    "component_candidates": self._candidates("asset", t.component_mention, ctx),
                    "site_words": t.site_mention,
                    "site_candidates": self._candidates("site", t.site_mention, ctx),
                    "priority_words": t.priority_mention,
                }
                for i, t in enumerate(tasks)
            ],
        }
        out, usage = structured("link_tasks", self.instructions, json.dumps(payload, indent=1), _Choices,
                                model=self.model)
        with self._lock:
            self.usage += usage
            self.seconds_by_meeting[ctx.meeting_id] = usage.seconds  # true API latency, even when cached
            self.tokens_by_meeting[ctx.meeting_id] = (usage.input_tokens, usage.output_tokens)
        by_index = {c.task_index: c for c in out.choices}
        valid = self.onto.by_id
        results = []
        for i, t in enumerate(tasks):
            c = by_index.get(i)
            if c is None:
                results.append(self.normalise(t, ctx))  # model skipped it: fall back to fuzzy
                continue
            pick = lambda v: v if v in valid else None  # noqa: E731
            results.append(self._finish(
                t, ctx, pick(c.owner_id), pick(c.component_id), pick(c.site_id), {},
                category=c.category, priority=t.priority or c.priority,
            ))
        return results
