"""GLiNER2 extractors (local encoder model, CPU friendly).

For each transcript turn:
1. ``classify_text`` decides whether the turn raises a new job;
2. ``extract_entities`` pulls equipment, person, deadline, site and urgency spans.

Conversation structure (who owns a job) uses the same light heuristics as the
rules baseline: "I'll ..." means the speaker, "X, can you ..." means X, a bare
"yep" accepts the previous request, "take that instead" hands it over.

* ``gliner``    - zero-shot, pretrained checkpoint (``GLINER_MODEL``, default
                  fastino/gliner2.5-base-v1).
* ``gliner_ft`` - the same pipeline on a checkpoint fine-tuned on the train split
                  by ``standup-exp finetune`` (``models/gliner-standup/best``).
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from ..paths import MODELS_DIR, env
from ..schema import ExtractionResult, RawTask
from .base import Extractor, TranscriptView
from .rules import ACCEPT, COMMIT, HANDOVER, LEADING_NAME, _summary

ENTITY_LABELS = {
    "equipment": "A machine, asset, valve, pump, line, or piece of plant equipment, including nicknames",
    "person": "The first name, nickname or full name of a colleague",
    "deadline": "When the work is due, e.g. today, tomorrow morning, by Thursday, at half two",
    "site": "An off-site location, depot, supplier or substation to visit",
    "urgency": "Words that say how urgent the work is, e.g. top priority, no rush",
}

# Zero-shot model: picked by a small probe (14 hand-written turns; "action item / status
# update / small talk" scored 12/14 vs 10-11/14 for finer-grained label sets).
ZERO_SHOT_TURN_LABELS = {"action item": True, "status update": False, "small talk": False}
# Fine-tuned model: trained on this binary pair (see finetune.py).
FT_TURN_LABELS = {"new job": True, "not a job": False}

PRIORITY_WORDS = [
    (1, ("urgent", "top priority", "straight away", "drop everything", "safety critical", "holding", "asap")),
    (3, ("no rush", "low priority", "whenever", "spare minute", "no hurry", "when you get a chance")),
    (2, ("this week", "normal", "fairly important", "slip", "not an emergency")),
]


@lru_cache(maxsize=2)
def _load(checkpoint: str):
    from gliner2 import AutoExtractor

    return AutoExtractor.from_pretrained(checkpoint)


def _first(entities: dict, label: str, exclude: str | None = None) -> str | None:
    items = entities.get(label) or []
    best = None
    for item in items:
        text, conf = (item["text"], item.get("confidence", 1.0)) if isinstance(item, dict) else (item, 1.0)
        if exclude and text.lower() in exclude.lower():
            continue
        if best is None or conf > best[1]:
            best = (text, conf)
    return best[0] if best else None


def _priority_from(text: str | None) -> int | None:
    if not text:
        return None
    t = text.lower()
    for level, words in PRIORITY_WORDS:
        if any(w in t for w in words):
            return level
    return None


class GlinerExtractor(Extractor):
    name = "gliner"
    turn_labels = ZERO_SHOT_TURN_LABELS
    threshold = 0.4

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint or env("GLINER_MODEL", "fastino/gliner2.5-base-v1")

    def extract(self, view: TranscriptView) -> ExtractionResult:
        model = _load(str(self.checkpoint))
        t0 = time.perf_counter()
        turns = view.turns
        texts = [t.text for t in turns]

        # Two batched passes: classify every turn, then pull entities from the job turns.
        classified = model.batch_classify_text(texts, {"turn": list(self.turn_labels)}, include_confidence=True)
        is_job = [
            self.turn_labels.get(c["turn"]["label"], False) and c["turn"]["confidence"] >= self.threshold
            and not (ACCEPT.search(t) and len(t.split()) <= 6) and not HANDOVER.search(t)
            for c, t in zip(classified, texts)
        ]
        wanted = [i for i, t in enumerate(texts) if is_job[i] or HANDOVER.search(t)]
        extracted = model.batch_extract_entities(
            [texts[i] for i in wanted], ENTITY_LABELS, include_confidence=True, threshold=self.threshold
        ) if wanted else []
        entities = {i: e.get("entities", {}) for i, e in zip(wanted, extracted)}

        tasks: list[RawTask] = []
        for i, turn in enumerate(turns):
            text = turn.text
            if HANDOVER.search(text):
                name = _first(entities.get(i, {}), "person", exclude=turn.speaker)
                if name and tasks:
                    tasks[-1].owner_mention = name
                continue
            if not is_job[i]:
                continue
            ents = entities[i]
            if COMMIT.search(text):
                owner = turn.speaker
            else:
                m = LEADING_NAME.search(text)
                owner = m.group(1) if m else _first(ents, "person", exclude=turn.speaker)
                if owner is None and i + 1 < len(turns) and ACCEPT.search(turns[i + 1].text):
                    owner = turns[i + 1].speaker
            urgency = _first(ents, "urgency")
            tasks.append(
                RawTask(
                    summary=_summary(text),
                    owner_mention=owner,
                    component_mention=_first(ents, "equipment"),
                    site_mention=_first(ents, "site"),
                    due_mention=_first(ents, "deadline"),
                    priority_mention=urgency,
                    priority=_priority_from(urgency),
                    evidence_quote=text,
                    confidence=float(classified[i]["turn"]["confidence"]),
                )
            )
        return ExtractionResult(
            meeting_id=view.meeting_id, extractor=self.name, tasks=tasks, seconds=time.perf_counter() - t0
        )


class FineTunedGlinerExtractor(GlinerExtractor):
    """GLiNER 2.5 base fine-tuned on the 12 GPT-5.4 training meetings (models/gliner-standup)."""

    name = "gliner_ft"
    turn_labels = FT_TURN_LABELS
    folder = "gliner-standup"

    def __init__(self, checkpoint: str | None = None):
        path = Path(checkpoint) if checkpoint else MODELS_DIR / self.folder / "best"
        if not path.exists():
            raise FileNotFoundError(f"No fine-tuned model at {path}. Run `standup-exp finetune` first.")
        super().__init__(str(path))


class FineTunedAllGlinerExtractor(FineTunedGlinerExtractor):
    """GLiNER 2.5 base fine-tuned on every writer's training meetings (models/gliner-standup-all)."""

    name = "gliner_ft_all"
    folder = "gliner-standup-all"


DECIDE_MODEL = "fastino/GLiNER2.5-Decide"


class DecideGlinerExtractor(GlinerExtractor):
    """GLiNER2.5-Decide (340M, fastino): the classification-specialist GLiNER2.5 checkpoint.

    Same pipeline and the same turn labels as zero-shot ``gliner``, so the two differ only
    in the checkpoint. Decide also tags spans (it is a SpanExtractor under the hood).
    """

    name = "gliner_decide"

    def __init__(self, checkpoint: str | None = None):
        super().__init__(checkpoint or env("GLINER_DECIDE_MODEL", DECIDE_MODEL))
