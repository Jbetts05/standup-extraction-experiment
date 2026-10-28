"""Lexical normalisers: exact alias lookup and fuzzy string matching (rapidfuzz)."""

from __future__ import annotations

from rapidfuzz import fuzz

from ..schema import RawTask
from .base import NormContext, Normaliser, clean


class AliasNormaliser(Normaliser):
    """Exact match on a known name/alias, else the longest known alias contained in the mention.

    Cheap, precise, and brittle: anything not already in the alias table is missed.
    """

    name = "alias"

    def link(self, kind: str, mention: str | None, task: RawTask, ctx: NormContext) -> tuple[str | None, float]:
        m = clean(mention)
        if not m:
            return None, 0.0
        exact = {eid for form, eid in self.forms[kind] if form == m}
        if len(exact) == 1:
            return exact.pop(), 1.0
        padded = f" {m} "
        contained = [(len(form), eid) for form, eid in self.forms[kind] if form and f" {form} " in padded]
        if contained:
            return max(contained)[1], 0.9
        return None, 0.0


class FuzzyNormaliser(Normaliser):
    """Best rapidfuzz WRatio against every known surface form, with an abstain threshold.

    Handles typos, word order and partial phrases ("turbine two water valve"),
    but not nicknames with no lexical overlap ("the Arburg").
    """

    name = "fuzzy"
    thresholds = {"person": 0.85, "asset": 0.72, "site": 0.75}

    def ranked(self, kind: str, mention: str, ctx: NormContext, limit: int | None = None) -> list[tuple[str, float]]:
        m = clean(mention)
        best: dict[str, float] = {}
        for form, eid in self.forms[kind]:
            if not form:
                continue
            score = fuzz.WRatio(m, form) / 100 + self.person_boost(kind, eid, ctx)
            if score > best.get(eid, 0.0):
                best[eid] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit] if limit else ranked

    def link(self, kind: str, mention: str | None, task: RawTask, ctx: NormContext) -> tuple[str | None, float]:
        if not clean(mention):
            return None, 0.0
        eid, score = self.best_with_parent_fallback(kind, self.ranked(kind, mention, ctx))
        return (eid, score) if score >= self.thresholds[kind] else (None, score)
