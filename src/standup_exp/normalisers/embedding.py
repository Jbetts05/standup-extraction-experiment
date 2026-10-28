"""Embedding normalisers: nearest known surface form by cosine similarity.

Two embedding models:

* ``embedding``      - ``sentence-transformers/all-MiniLM-L6-v2`` (22M params, local CPU)
* ``embedding_oai``  - OpenAI ``text-embedding-3-large`` on Azure AI Foundry (Foundry resource B),
                       every vector cached on disk

Each known surface form is indexed as "form (Official Name)" so a descriptive nickname
("the pump in the pit") can still land near its entity. Categories are chosen by
similarity to each category's description.

Acceptance thresholds are calibrated per model, never on test data: every known alias is
queried with itself left out of the index (it simulates an unseen nickname), and the
threshold is set so 95% of those genuine matches are accepted.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..llm import embed_openai
from ..paths import env
from ..schema import RawTask
from .base import NormContext, Normaliser, clean
from .lexical import FuzzyNormaliser


@lru_cache(maxsize=1)
def _minilm():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def embed_minilm(texts: list[str]) -> np.ndarray:
    return np.asarray(_minilm().encode(texts, normalize_embeddings=True, show_progress_bar=False))


BACKENDS = {"minilm": embed_minilm, "openai": embed_openai}


class EmbeddingNormaliser(Normaliser):
    name = "embedding"
    backend = "minilm"
    default_thresholds = {"person": 0.75, "asset": 0.55, "site": 0.55}

    def __init__(self, ontology, train=None, backend: str | None = None):
        super().__init__(ontology, train)
        self.backend = backend or self.backend
        self._embed = BACKENDS[self.backend]
        self.index: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}
        for kind, forms in self.forms.items():
            texts = [f"{form} ({ontology.by_id[eid].name})" for form, eid in forms]
            self.index[kind] = (self._embed(texts), [eid for _, eid in forms], [f for f, _ in forms])
        self.cat_ids = ontology.category_ids
        self.cat_vecs = self._embed([f"{c['id']}: {c['description']}" for c in ontology.categories])
        self.thresholds = self._calibrate()

    def _calibrate(self) -> dict[str, float]:
        """Leave-one-alias-out on the known vocabulary -> 5th percentile of genuine-match scores."""
        out = {}
        for kind, (vecs, ids, forms) in self.index.items():
            aliases = [(i, f) for i, f in enumerate(forms)
                       if f != self.onto.by_id[ids[i]].name and f != ids[i] and sum(x == ids[i] for x in ids) > 2]
            if len(aliases) < 5:
                out[kind] = self.default_thresholds[kind]
                continue
            q = self._embed([f for _, f in aliases])
            scores = []
            for (i, _), qv in zip(aliases, q):
                sims = vecs @ qv
                keep = [j for j in range(len(ids)) if j != i]
                j = max(keep, key=lambda j: sims[j])
                if ids[j] == ids[i]:
                    scores.append(float(sims[j]))
            out[kind] = float(np.percentile(scores, 5)) if len(scores) >= 5 else self.default_thresholds[kind]
        return out

    def ranked(self, kind: str, mention: str, ctx: NormContext, task: RawTask | None = None) -> list[tuple[str, float]]:
        vecs, ids, _ = self.index[kind]
        q = self._embed([clean(mention)])[0]
        sims = vecs @ q
        best: dict[str, float] = {}
        for eid, s in zip(ids, sims):
            s = float(s) + self.person_boost(kind, eid, ctx)
            best[eid] = max(best.get(eid, -1.0), s)
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def link(self, kind: str, mention: str | None, task: RawTask, ctx: NormContext) -> tuple[str | None, float]:
        if not clean(mention):
            return None, 0.0
        eid, score = self.best_with_parent_fallback(kind, self.ranked(kind, mention, ctx, task))
        return (eid, score) if score >= self.thresholds[kind] else (None, score)

    def classify_category(self, task: RawTask, ctx: NormContext) -> str:
        q = self._embed([task.summary])[0]
        return self.cat_ids[int(np.argmax(self.cat_vecs @ q))]


class OpenAIEmbeddingNormaliser(EmbeddingNormaliser):
    name = "embedding_oai"
    backend = "openai"


class HybridShortlist(FuzzyNormaliser):
    """Retrieve-then-rerank base: candidates = interleaved fuzzy and embedding top hits.

    Character similarity finds typos and abbreviations ("GT2 wtr valve"); embeddings find
    meaning ("the cold unit" ~ chiller). A reranker (GLiNER, LLM) can only be as good as
    the shortlist it is given, so both are used. Embeddings: OpenAI text-embedding-3-large
    (``SHORTLIST_EMBED=minilm`` switches back to the local model).
    """

    shortlist = 10

    def __init__(self, ontology, train=None):
        super().__init__(ontology, train)
        self._emb = EmbeddingNormaliser(ontology, train, backend=env("SHORTLIST_EMBED", "openai"))

    def shortlist_ids(self, kind: str, mention: str, ctx: NormContext) -> list[str]:
        fuzzy = [eid for eid, _ in self.ranked(kind, mention, ctx)]
        semantic = [eid for eid, _ in self._emb.ranked(kind, mention, ctx)]
        out: list[str] = []
        for a, b in zip(fuzzy, semantic):
            for eid in (a, b):
                if eid not in out:
                    out.append(eid)
            if len(out) >= self.shortlist:
                break
        return out[: self.shortlist]
