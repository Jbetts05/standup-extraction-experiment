"""GLiNER2 normaliser: hybrid shortlist, then zero-shot classification picks one.

Linking: fuzzy + embedding retrieval gives candidate entities (see
``HybridShortlist``); GLiNER2 ``classify_text`` then chooses between their labels
("Official Name (alias, alias)"), using the mention and task wording as context.
Mentions that exactly match a known form skip the model.
Category and priority: ``classify_text`` with the business label descriptions.
"""

from __future__ import annotations

from ..extractors.gliner import _load
from ..paths import env
from ..schema import RawTask
from .base import NormContext, clean, keyword_priority
from .embedding import HybridShortlist


class GlinerNormaliser(HybridShortlist):
    name = "gliner"
    model_id = None  # None = GLINER_MODEL (default fastino/gliner2.5-base-v1)
    cache_outputs = True  # CPU-heavy: the runner caches predictions + measured time
    min_confidence = 0.35
    exact_score = 1.0  # WRatio 100 = the mention is a known form verbatim

    def __init__(self, ontology, train=None):
        super().__init__(ontology, train)
        self.model = _load(self.model_id or env("GLINER_MODEL", "fastino/gliner2.5-base-v1"))
        self.cat_labels = {c["id"]: c["description"] for c in ontology.categories}

    def _classify(self, text: str, labels: list[str]) -> tuple[str, float]:
        out = self.model.classify_text(text, {"choice": labels}, include_confidence=True)["choice"]
        return (out["label"], float(out["confidence"])) if isinstance(out, dict) else (out, 1.0)

    def link(self, kind: str, mention: str | None, task: RawTask, ctx: NormContext) -> tuple[str | None, float]:
        if not clean(mention):
            return None, 0.0
        top = self.ranked(kind, mention, ctx, limit=1)
        if top and top[0][1] >= self.exact_score:
            return top[0]  # a known alias: no need to ask the model
        # Label = official name plus known aliases, so the model sees the vocabulary it is choosing from.
        labels = {}
        for eid in self.shortlist_ids(kind, mention, ctx):
            e = self.onto.by_id[eid]
            labels[f"{e.name} ({', '.join(e.aliases[:3])})" if e.aliases else e.name] = eid
        # No "none of these" label: in probes the model chose it for almost every nickname,
        # so abstention is handled by the confidence threshold instead.
        label, conf = self._classify(f"{mention} - {task.summary}", list(labels))
        if conf < self.min_confidence:
            return None, conf
        return labels[label], conf

    def classify_category(self, task: RawTask, ctx: NormContext) -> str:
        out = self.model.classify_text(task.summary, {"category": self.cat_labels})
        label = out["category"]
        return label["label"] if isinstance(label, dict) else label

    def classify_priority(self, task: RawTask, ctx: NormContext) -> int | None:
        if not task.priority_mention:
            return keyword_priority(task)
        labels = {"1": "urgent, today, safety critical", "2": "normal, this week", "3": "low priority, no rush"}
        out = self.model.classify_text(task.priority_mention, {"priority": labels})["priority"]
        return int(out["label"] if isinstance(out, dict) else out)


class DecideGlinerNormaliser(GlinerNormaliser):
    """Same shortlist-then-choose matcher, with GLiNER2.5-Decide doing the choosing."""

    name = "gliner_decide"
    model_id = "fastino/GLiNER2.5-Decide"
