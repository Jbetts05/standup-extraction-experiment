"""Normalisers map a ``RawTask`` (surface mentions) onto the business vocabulary.

Every normaliser answers the same questions for each task:

* ``link(kind, mention)`` -> canonical id for a person, site or asset (or None to abstain)
* ``classify_category(task)`` -> operations / safety / organisation
* ``classify_priority(task)`` -> 1 / 2 / 3

Shared behaviour lives here: date resolution, "use the extractor's value if it
already gave one", and falling back to a parent asset when the top candidates
are near-tied siblings (a correct-but-general answer beats a wrong specific one).

Normalisers may only use ``Entity.known_surface_forms()`` - never held-out aliases.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from datetime import date

from ..dates import resolve_date, resolve_time
from ..ontology import Ontology
from ..schema import Meeting, NormalisedTask, RawTask

KINDS = ("person", "asset", "site")


@dataclass(frozen=True)
class NormContext:
    meeting_id: str
    date: date
    attendee_ids: tuple[str, ...]  # a real system gets these from the Teams roster


def clean(text: str | None) -> str:
    """Lowercase, drop punctuation and leading determiners."""
    if not text:
        return ""
    t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    t = re.sub(r"^(the|a|an|our|that|this)\s+", "", t.strip())
    return re.sub(r"\s+", " ", t).strip()


SAFETY_WORDS = re.compile(
    r"\b(risk assessment|permit|isolation|guard|guarding|interlock|near miss|lock ?out|tag ?out|loto|fire alarm|"
    r"pre ?use|ppe|hazard|incident|safety|coshh|harness)\b", re.I)
ORG_WORDS = re.compile(
    r"\b(order|book|schedule|chase|purchase|po\b|visit|go to|go over|pick up|collect|report|handover|toolbox|"
    r"training|contractor|meeting|rota|paperwork|quote|invoice|organise|organize|send)\b", re.I)
PRIORITY_RULES = [
    (1, re.compile(r"\b(urgent|top priority|straight away|asap|drop everything|safety critical|holding|first|now)\b", re.I)),
    (3, re.compile(r"\b(no rush|low priority|whenever|spare minute|no hurry|chance)\b", re.I)),
    (2, re.compile(r"\b(this week|normal|fairly important|slip|not an emergency)\b", re.I)),
]


def keyword_category(task: RawTask) -> str:
    text = f"{task.summary} {task.evidence_quote or ''}"
    if SAFETY_WORDS.search(text):
        return "safety"
    if ORG_WORDS.search(text):
        return "organisation"
    return "operations"


def keyword_priority(task: RawTask) -> int:
    """Map urgency words to 1/2/3; no (recognised) urgency words -> default 2."""
    text = task.priority_mention or ""
    for level, rx in PRIORITY_RULES:
        if rx.search(text):
            return level
    return 2


class Normaliser:
    name = "base"
    #: Relative score margin under which two sibling assets count as a tie.
    tie_margin = 0.03

    def __init__(self, ontology: Ontology, train: list[Meeting] | None = None):
        self.onto = ontology
        self.train = train or []
        # (clean surface form, entity id) for every kind; the only vocabulary normalisers may use.
        self.forms: dict[str, list[tuple[str, str]]] = {
            kind: [(clean(f), e.id) for e in ontology.entities_of(kind) for f in e.known_surface_forms()]
            for kind in KINDS
        }

    # ------------------------------------------------------------- interface
    def link(self, kind: str, mention: str | None, task: RawTask, ctx: NormContext) -> tuple[str | None, float]:
        raise NotImplementedError

    def classify_category(self, task: RawTask, ctx: NormContext) -> str:
        return keyword_category(task)

    def classify_priority(self, task: RawTask, ctx: NormContext) -> int | None:
        return keyword_priority(task)

    # ------------------------------------------------------------ pipeline
    def normalise_meeting(self, tasks: list[RawTask], ctx: NormContext) -> list[NormalisedTask]:
        return [self.normalise(t, ctx) for t in tasks]

    def normalise(self, task: RawTask, ctx: NormContext) -> NormalisedTask:
        scores: dict[str, float] = {}
        ids: dict[str, str | None] = {}
        for kind, mention in (("person", task.owner_mention), ("asset", task.component_mention), ("site", task.site_mention)):
            if not mention:
                ids[kind] = None
                continue
            ids[kind], scores[kind] = self.link(kind, mention, task, ctx)
        return self._finish(task, ctx, ids["person"], ids["asset"], ids["site"], scores)

    def _finish(self, task, ctx, owner_id, component_id, site_id, scores,
                category: str | None = None, priority: int | None = None) -> NormalisedTask:
        # An extractor-supplied category/priority wins, so normalisers are compared on what they add.
        category = (task.category if task.category in self.onto.category_ids else None) \
            or category or self.classify_category(task, ctx)
        priority = task.priority or priority or self.classify_priority(task, ctx)
        return NormalisedTask(
            summary=task.summary,
            owner_id=owner_id,
            component_id=component_id,
            site_id=site_id,
            due_date=task.due_date or resolve_date(task.due_mention, ctx.date),
            due_time=task.due_time or resolve_time(task.due_mention),
            priority=priority,
            category=category,
            evidence_quote=task.evidence_quote,
            raw=task,
            link_scores=scores,
        )

    # -------------------------------------------------------------- helpers
    def best_with_parent_fallback(self, kind: str, ranked: list[tuple[str, float]]) -> tuple[str | None, float]:
        """``ranked`` is [(entity_id, score in 0..1)] best first, one entry per entity."""
        if not ranked:
            return None, 0.0
        top_id, top = ranked[0]
        if kind == "asset" and len(ranked) > 1:
            second_id, second = ranked[1]
            p1, p2 = self.onto.by_id[top_id].parent, self.onto.by_id[second_id].parent
            if top - second < self.tie_margin and p1 and p1 == p2:
                return p1, top
        return top_id, top

    def person_boost(self, kind: str, entity_id: str, ctx: NormContext) -> float:
        """People at the meeting are much more likely owners than people who are not."""
        return 0.05 if kind == "person" and entity_id in ctx.attendee_ids else 0.0


class Passthrough(Normaliser):
    """Keep IDs chosen by a direct-linking extractor (``llm_direct``).

    Guard: an ID is only kept when the model also reported the words it came from
    (no site mentioned -> no site ID). Without this, Mistral Large 3 in JSON mode
    filled a site ID on almost every task.
    """

    name = "passthrough"

    def normalise(self, task: RawTask, ctx: NormContext) -> NormalisedTask:
        valid = self.onto.by_id

        def pick(eid, mention):
            return eid if eid in valid and clean(mention) else None

        return self._finish(task, ctx, pick(task.owner_id, task.owner_mention),
                            pick(task.component_id, task.component_mention), pick(task.site_id, task.site_mention), {})


REGISTRY = {
    "passthrough": "standup_exp.normalisers.base:Passthrough",
    "alias": "standup_exp.normalisers.lexical:AliasNormaliser",
    "fuzzy": "standup_exp.normalisers.lexical:FuzzyNormaliser",
    "embedding": "standup_exp.normalisers.embedding:EmbeddingNormaliser",
    "embedding_oai": "standup_exp.normalisers.embedding:OpenAIEmbeddingNormaliser",
    "classifier": "standup_exp.normalisers.classifier:ClassifierNormaliser",
    "gliner": "standup_exp.normalisers.gliner:GlinerNormaliser",
    "gliner_decide": "standup_exp.normalisers.gliner:DecideGlinerNormaliser",
    "llm": "standup_exp.normalisers.llm:LLMNormaliser",
}


def get_normaliser(name: str, ontology: Ontology, train: list[Meeting] | None = None) -> Normaliser:
    """``name`` may pin an LLM: ``llm@grok-4.6`` (ids from data/models.yaml)."""
    base, _, model = name.partition("@")
    if base not in REGISTRY:
        raise KeyError(f"Unknown normaliser '{name}'. Choose from: {', '.join(REGISTRY)}")
    module, cls = REGISTRY[base].split(":")
    klass = getattr(importlib.import_module(module), cls)
    if model:
        if base != "llm":
            raise KeyError(f"'{base}' does not use an LLM, so '@{model}' makes no sense.")
        return klass(ontology, train, model=model)
    return klass(ontology, train)
