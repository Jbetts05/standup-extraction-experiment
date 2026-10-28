"""Step 2 · pull board context: which open cards could a mention be about? Plain code, no model.

    open cards where
          owner is the speaker, the named owner or the named new owner
       or the equipment is the same, a part of it, or what it is part of
       or (said = reassigned and nobody owns the card)       # "I'll pick up the one nobody had"
    ranked by wording similarity, at most ``cap`` cards

Names and equipment come from the matcher's pick plus its top shortlist candidates: Step 2 aims to catch
the right card (nicknames like "Haley" or "the boss" are often mis-linked); Step 3 then picks precisely.
The board is the one at the start of the stand-up, so the order of mentions does not matter.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from ..ontology import Ontology
from .schema import Card

CAP = 8


def related(onto: Ontology, a: str | None, b: str | None) -> bool:
    """Same equipment, or one is part of the other (GT1 and GT1 Compressor Section)."""
    return bool(a and b and (a == b or a in onto.ancestors(b) or b in onto.ancestors(a)))


def context_for(board: list[Card], onto: Ontology, people: set[str], component_ids,
                include_unowned: bool = False, words: str = "", cap: int = CAP) -> list[Card]:
    """``component_ids``: one id or several (the matcher's pick plus its shortlist), any may be None."""
    people = {p for p in people if p}
    comps = {c for c in ([component_ids] if isinstance(component_ids, str) or component_ids is None else component_ids) if c}
    hits = [c for c in board if c.is_open and (
        c.owner_id in people or any(related(onto, c.component_id, x) for x in comps)
        or (include_unowned and c.owner_id is None))]
    hits.sort(key=lambda c: (-fuzz.token_set_ratio(words.lower(), c.summary.lower()), c.id))
    return hits[:cap]
