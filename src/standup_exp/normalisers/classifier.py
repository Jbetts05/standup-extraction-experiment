"""Trained classifier normaliser (scikit-learn, trains in seconds on CPU).

* Entity linking: character n-gram TF-IDF + logistic regression, one model per
  kind (person / asset / site). Training examples are every known surface form
  plus the mentions used in the *train* split (held-out aliases never appear
  there). Abstains when the top class probability is below a threshold.
* Category and priority: word + character TF-IDF + logistic regression trained
  on train-split task wording and priority cues.

This is the "can a small supervised model learn the plant's vocabulary?" baseline.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from ..schema import RawTask
from .base import NormContext, Normaliser, clean, keyword_priority


def _char_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0),
    )


def _text_model():
    return make_pipeline(
        FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]),
        LogisticRegression(max_iter=2000, C=5.0),
    )


class ClassifierNormaliser(Normaliser):
    name = "classifier"
    thresholds = {"person": 0.30, "asset": 0.12, "site": 0.30}

    def __init__(self, ontology, train=None):
        super().__init__(ontology, train)
        examples: dict[str, list[tuple[str, str]]] = {k: list(v) for k, v in self.forms.items()}
        cat_x, cat_y, pri_x, pri_y = [], [], [], []
        for meeting in self.train:
            for t in meeting.gold_tasks:
                for kind, m in (("person", t.owner), ("asset", t.component), ("site", t.site)):
                    if m.entity_id and m.text:
                        examples[kind].append((clean(m.text), m.entity_id))
                cat_x += [t.summary, t.spoken_action]
                cat_y += [t.category, t.category]
                if t.priority_cue:
                    pri_x.append(t.priority_cue)
                    pri_y.append(t.priority)

        self.linkers = {}
        for kind, pairs in examples.items():
            x, y = zip(*[(f, e) for f, e in pairs if f])
            self.linkers[kind] = _char_model().fit(list(x), list(y))
        self.category_model = _text_model().fit(cat_x, cat_y) if len(set(cat_y)) > 1 else None
        self.priority_model = _text_model().fit(pri_x, pri_y) if len(set(pri_y)) > 1 else None

    def link(self, kind: str, mention: str | None, task: RawTask, ctx: NormContext) -> tuple[str | None, float]:
        m = clean(mention)
        if not m:
            return None, 0.0
        model = self.linkers[kind]
        probs = model.predict_proba([m])[0]
        classes = model.classes_
        ranked = sorted(
            ((str(c), float(p) + self.person_boost(kind, str(c), ctx)) for c, p in zip(classes, probs)),
            key=lambda kv: kv[1], reverse=True,
        )
        eid, score = self.best_with_parent_fallback(kind, ranked)
        return (eid, score) if score >= self.thresholds[kind] else (None, score)

    def classify_category(self, task: RawTask, ctx: NormContext) -> str:
        if self.category_model is None:
            return super().classify_category(task, ctx)
        return str(self.category_model.predict([task.summary])[0])

    def classify_priority(self, task: RawTask, ctx: NormContext) -> int | None:
        if not task.priority_mention or self.priority_model is None:
            return keyword_priority(task)
        return int(self.priority_model.predict([task.priority_mention])[0])
