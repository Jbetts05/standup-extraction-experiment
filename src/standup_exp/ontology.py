"""Load the plant ontology and provide lookups used by generators and normalisers."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

from .paths import ONTOLOGY_PATH


@dataclass(frozen=True)
class Entity:
    """A canonical thing a mention can be linked to: a person, site or asset."""

    id: str
    name: str
    aliases: tuple[str, ...] = ()
    heldout_aliases: tuple[str, ...] = ()
    parent: str | None = None
    kind: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def known_surface_forms(self) -> list[str]:
        """Names a normaliser is allowed to know about (never the held-out aliases)."""
        return [self.name, self.id, *self.aliases]


@dataclass
class Ontology:
    plant: dict
    categories: list[dict]
    priorities: list[dict]
    people: list[Entity]
    sites: list[Entity]
    assets: list[Entity]

    @classmethod
    def load(cls, path: Path = ONTOLOGY_PATH) -> "Ontology":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        def entities(items: list[dict]) -> list[Entity]:
            out = []
            for item in items:
                item = dict(item)
                out.append(
                    Entity(
                        id=str(item.pop("id")),
                        name=item.pop("name"),
                        aliases=tuple(item.pop("aliases", []) or []),
                        heldout_aliases=tuple(item.pop("heldout_aliases", []) or []),
                        parent=item.pop("parent", None),
                        kind=item.pop("kind", ""),
                        extra=item,
                    )
                )
            return out

        return cls(
            plant=raw["plant"],
            categories=raw["categories"],
            priorities=raw["priorities"],
            people=entities(raw["people"]),
            sites=entities(raw["sites"]),
            assets=entities(raw["assets"]),
        )

    # ------------------------------------------------------------------ lookups
    @cached_property
    def by_id(self) -> dict[str, Entity]:
        return {e.id: e for e in [*self.people, *self.sites, *self.assets]}

    @property
    def category_ids(self) -> list[str]:
        return [c["id"] for c in self.categories]

    @property
    def priority_ids(self) -> list[int]:
        return [int(p["id"]) for p in self.priorities]

    def rubric(self) -> str:
        """The business category/priority definitions, as given to every LLM prompt."""
        cats = "\n".join(f"  - {c['id']}: {c['description']}" for c in self.categories)
        pris = "\n".join(f"  - {p['id']}: {p['description']}" for p in self.priorities)
        return (
            "Business rubric.\n"
            f"category (by the purpose of the job):\n{cats}\n"
            "  Safety checks and tests (fire alarm tests, pre-use checks, guarding, lock-out) are safety;\n"
            "  ordering, booking, scheduling, chasing, visits and paperwork are organisation.\n"
            f"priority (only from what was said about urgency):\n{pris}\n"
            "  If nobody said anything about urgency, priority is 2. "
            "\"Not an emergency but don't let it slip\" is 2."
        )

    def entities_of(self, kind: str) -> list[Entity]:
        """kind is one of 'person', 'site', 'asset'."""
        return {"person": self.people, "site": self.sites, "asset": self.assets}[kind]

    def ancestors(self, asset_id: str) -> list[str]:
        """Parent chain of an asset, nearest first."""
        chain, current = [], self.by_id.get(asset_id)
        while current is not None and current.parent:
            chain.append(current.parent)
            current = self.by_id.get(current.parent)
        return chain

    def leaf_assets(self) -> list[Entity]:
        return [a for a in self.assets if a.kind in ("component", "system")]
