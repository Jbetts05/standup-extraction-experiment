"""Rules / regex baseline extractor.

No model, no ontology: only generic engineering vocabulary and conversational
patterns. It shows how far plain pattern matching gets, and where it breaks
(paraphrase, pronouns, tasks spread over several turns).

Per turn:
1. skip turns that talk about the past, blockers, cancellations or banter;
2. detect a commitment ("I'll", "I'm going to"), a request ("X, can you ...")
   or an ownerless need ("someone needs to ...");
3. pull out the equipment noun phrase, site, deadline and urgency words;
4. handle hand-overs ("can you take that one instead?") and bare acceptances ("yep").
"""

from __future__ import annotations

import re
import time

from ..schema import ExtractionResult, RawTask
from .base import Extractor, TranscriptView

EQUIPMENT_NOUNS = (
    "valve|valves|pump|pumps|boiler|turbine|bearing|compressor|conveyor|belt|palletiser|palletizer|"
    "stacker|wrapper|spinner|robot|arm|guard|interlock|switch|chiller|unit|barrel|screw|dryer|board|"
    "switchboard|panel|forklifts?|trucks|counterbalances|tower|hydraulics|hpu|tcu|power pack|heater box|"
    "section|end|line [0-9]|line (?:one|two|three)|hall|plant|system|ring|pit|box|engel|arburg|atlas|hrsg|bfp|flts?"
)
_STOP = r"(?!(?:the|on|for|at|to|and|with|in)\b)"
# Lookahead so matches can overlap: every "the/on/for/at" is a possible start, and the
# phrase may not run across another preposition ("pressure drop on the ...").
EQUIPMENT = re.compile(
    rf"(?=\b(?:the|on|for|at)\s+((?:{_STOP}[\w'-]+\s+){{0,4}}(?:{EQUIPMENT_NOUNS})(?:\s+(?:{EQUIPMENT_NOUNS}))*)\b"
    rf"(\s+on\s+(?:one|two|three|[0-9]))?)",
    re.I,
)
ASSET_TAG = re.compile(r"\b(?:[A-Z]{2,4}-?\d{1,2}|line\s+(?:[0-9]|one|two|three))\b", re.I)
SITE = re.compile(
    r"\b(?:to|at|visit|over to|up to|down to)\s+((?:the\s+)?(?:[A-Z][\w]+(?:\s+(?:[A-Z][\w]+|in\s+[A-Z][\w]+)){0,3}|"
    r"substation|sub|warehouse|water works|treatment plant|overflow store|hydraulics lot))\b"
)
DUE = re.compile(
    r"\b(today|tonight|tomorrow(?:\s+(?:morning|afternoon))?|first thing(?:\s+tomorrow)?|this\s+(?:morning|afternoon|week)|"
    r"after lunch|before lunch|end of (?:the\s+)?(?:day|week|shift)|next\s+week|"
    r"(?:by|on|next)?\s*(?:mon|tues|wednes|thurs|fri)day|monday|"
    r"(?:at|by)\s+\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?|half\s+\w+|the\s+\d{1,2}(?:st|nd|rd|th))\b",
    re.I,
)
PRIORITY = [
    (1, re.compile(r"\b(urgent|top priority|straight away|asap|drop everything|safety critical|holding (?:production|us) up|first)\b", re.I)),
    (3, re.compile(r"\b(no rush|low priority|whenever|spare minute|when you get a chance|no hurry)\b", re.I)),
    (2, re.compile(r"\b(this week|normal priority|fairly important|don't let it slip|not an emergency)\b", re.I)),
]
COMMIT = re.compile(r"\b(i'll|i will|i'm going to|i am going to|i'm gonna|let me|i can do|i'll take)\b", re.I)
REQUEST = re.compile(r"\b(can|could|would) you\b|\bplease\b|\bcan i get you to\b", re.I)
OWNERLESS = re.compile(r"\b(someone|somebody|anyone) (needs|has|will need) to\b|\bneeds (doing|sorting)\b", re.I)
HANDOVER = re.compile(r"\b(take (?:that|it|this)(?: one)? (?:instead|on|over)|can you take (?:that|it|this))\b", re.I)
ACCEPT = re.compile(r"^\W*(yep|yeah|yes|sure|no problem|will do|can do|ok(?:ay)?|on it)\b", re.I)
NEGATIVE = re.compile(
    r"\b(yesterday|last night|already|finished|all fine|signed off|ignore it|cancelled|pushed to|"
    r"waiting on|still waiting|can't get to|hasn't|settled|came back on its own|nothing to action|tied up)\b",
    re.I,
)
ACTION = re.compile(
    r"\b(inspect|replace|swap|check|take|investigate|recalibrate|run|grease|clear|update|sort|write|do|order|"
    r"book|schedule|chase|go|pick|send|organise|organize|arrange|fix|look at|test|log)\b",
    re.I,
)
LEADING_NAME = re.compile(
    r"^\s*(?!(?:Next|Right|Okay|Ok|So|And|Also|First|Finally|Then|Actually|Great|Cheers|Nice|Sorted|Yeah|Yep|"
    r"Well|Now|Last|Lastly|Morning|Thanks|Sure|Fine|Good|Separate|Another)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,"
)
TRAILING_NAME = re.compile(r",\s*([A-Z][a-z]+)\s*[?.!]?\s*$")


def _equipment(text: str) -> str | None:
    best = None
    for m in EQUIPMENT.finditer(text):
        phrase = (m.group(1) + (m.group(2) or "")).strip()
        if best is None or len(phrase) > len(best):
            best = phrase
    if best:
        return best
    tag = ASSET_TAG.search(text)
    return tag.group(0) if tag else None


def _due(text: str) -> str | None:
    parts = [m.group(0).strip() for m in DUE.finditer(text)]
    return " ".join(parts) if parts else None


def _priority(text: str) -> tuple[int | None, str | None]:
    for level, pattern in PRIORITY:
        m = pattern.search(text)
        if m:
            return level, m.group(0)
    return None, None


def _summary(text: str) -> str:
    text = LEADING_NAME.sub("", text).strip()
    m = ACTION.search(text)
    return (text[m.start():] if m else text).rstrip(" ?.!").strip()[:160]


class RulesExtractor(Extractor):
    name = "rules"
    cacheable = False

    def extract(self, view: TranscriptView) -> ExtractionResult:
        t0 = time.perf_counter()
        tasks: list[RawTask] = []
        turns = view.turns
        for i, turn in enumerate(turns):
            text = turn.text
            if HANDOVER.search(text) and tasks:
                m = LEADING_NAME.search(text) or re.search(r"\b([A-Z][a-z]+),\s+can you take", text)
                if m:
                    tasks[-1].owner_mention = m.group(1)
                continue
            if NEGATIVE.search(text) or not ACTION.search(text):
                continue
            if ACCEPT.search(text) and len(text.split()) <= 6:
                continue  # "Yeah, I'll take it." - acceptance of an earlier ask

            owner = None
            if COMMIT.search(text):
                owner = turn.speaker
            elif REQUEST.search(text):
                m = LEADING_NAME.search(text) or TRAILING_NAME.search(text)
                if m:
                    owner = m.group(1)
                elif i + 1 < len(turns) and ACCEPT.search(turns[i + 1].text):
                    owner = turns[i + 1].speaker
            elif not OWNERLESS.search(text):
                continue

            priority, cue = _priority(text)
            site = SITE.search(text)
            tasks.append(
                RawTask(
                    summary=_summary(text),
                    owner_mention=owner,
                    component_mention=_equipment(text),
                    site_mention=site.group(1) if site else None,
                    due_mention=_due(text),
                    priority_mention=cue,
                    priority=priority,
                    evidence_quote=text,
                    confidence=0.5,
                )
            )
        return ExtractionResult(
            meeting_id=view.meeting_id, extractor=self.name, tasks=tasks, seconds=time.perf_counter() - t0
        )
