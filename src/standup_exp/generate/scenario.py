"""Sample the answer key for a meeting *before* any transcript text exists.

Each meeting gets:
* a date (a weekday in 2026), a chair and 7 attendees,
* 5-9 gold tasks built from templates, with a sampled owner, component, site,
  due phrase and priority cue, each referred to by a name/alias chosen per split,
* distractors: chit-chat, work finished yesterday, observations, blockers and
  cancelled work. None of these should be extracted.

Everything is driven by ``random.Random(seed)`` so a dataset is reproducible.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta

from ..dates import resolve_date, resolve_time
from ..ontology import Entity, Ontology
from ..schema import Distractor, GoldTask, Meeting, Mention


@dataclass(frozen=True)
class Template:
    category: str
    action: str  # "{asset}" and "{site}" placeholders
    assets: tuple[str, ...] = ()  # allowed asset ids; empty = any leaf asset
    needs_asset: bool = True
    needs_site: bool = False
    priorities: tuple[int, ...] = (1, 2, 3)


PUMPS_VALVES = ("GT1-FV-01", "GT2-WV-03", "WHB-01-FWP", "WHB-01-SV", "WTR-MV-01", "WTR-P-01", "IM1-HYD")
ROTATING = ("GT1", "GT1-CMP", "GT2", "GT2-BRG-02", "WHB-01-FWP", "WTR-P-01", "AIR-C-01", "PCK-CV-01", "IM3-SCR")
GUARDED = ("IM2-GRD", "IM2-ROB", "PCK-PAL-01", "PCK-WRP-01", "PCK-CV-01", "IM1", "IM2", "IM3")
ELECTRICAL = ("HV-SW-01", "FIRE-01", "IM2-GRD", "AIR-DRY-01", "IM1-MLD-T", "IM3-CHL")

TEMPLATES: list[Template] = [
    # ------------------------------------------------------------ operations
    Template("operations", "inspect the {asset} for leaks", PUMPS_VALVES),
    Template("operations", "replace the seals on the {asset}", PUMPS_VALVES, priorities=(1, 2)),
    Template("operations", "take vibration readings on the {asset}", ROTATING),
    Template("operations", "investigate the pressure drop on the {asset}", ("GT2-WV-03", "WTR-MV-01", "WTR-P-01", "WHB-01-FWP", "IM1-HYD", "AIR-C-01")),
    Template("operations", "recalibrate the sensors on the {asset}", ("IM1-MLD-T", "IM3-CHL", "GT2-WV-03", "WHB-01-SV", "AIR-DRY-01")),
    Template("operations", "swap the filter on the {asset}", ("AIR-C-01", "AIR-DRY-01", "IM1-HYD", "WTR-P-01", "WTR-CT-01")),
    Template("operations", "run a function test on the {asset}", ELECTRICAL + ("WHB-01-SV",)),
    Template("operations", "grease and check the bearings on the {asset}", ROTATING, priorities=(2, 3)),
    Template("operations", "clear the jam and check the sensors on the {asset}", ("PCK-CV-01", "PCK-PAL-01", "PCK-WRP-01", "IM2-ROB"), priorities=(1, 2)),
    # ---------------------------------------------------------------- safety
    Template("safety", "update the risk assessment for work on the {asset}", priorities=(1, 2)),
    Template("safety", "sort the permit to work and isolation for the {asset}", priorities=(1, 2)),
    Template("safety", "check the guarding on the {asset}", GUARDED, priorities=(1, 2)),
    Template("safety", "write up the near miss at the {asset}", GUARDED + ("PCK-FLT", "WHB-01-SV"), priorities=(1, 2)),
    Template("safety", "do the lock-out tag-out sign-off on the {asset}", ELECTRICAL + PUMPS_VALVES, priorities=(1,)),
    Template("safety", "run the monthly fire alarm test", needs_asset=False, priorities=(2,)),
    Template("safety", "do the pre-use checks on the {asset}", ("PCK-FLT",), priorities=(1, 2)),
    # ---------------------------------------------------------- organisation
    Template("organisation", "order spare parts for the {asset}", priorities=(2, 3)),
    Template("organisation", "book a contractor to look at the {asset}", priorities=(2, 3)),
    Template("organisation", "schedule downtime for the {asset} with production", priorities=(2, 3)),
    Template("organisation", "chase the purchase order for the {asset}", priorities=(2, 3)),
    Template("organisation", "go to {site} to pick up the replacement parts for the {asset}", needs_site=True, priorities=(1, 2)),
    Template("organisation", "do a site visit at {site} with the contractor", needs_asset=False, needs_site=True, priorities=(2, 3)),
    Template("organisation", "send the shift handover report round", needs_asset=False, priorities=(2, 3)),
    Template("organisation", "organise the toolbox talk for the team", needs_asset=False, priorities=(2, 3)),
    Template("organisation", "sort the training records for the new starters", needs_asset=False, priorities=(3,)),
]

DUE_PHRASES = [
    "today", "by end of day", "this morning", "after lunch", "this afternoon",
    "tomorrow", "tomorrow morning", "first thing tomorrow",
    "by Thursday", "on Friday", "by Wednesday", "Monday",
    "next Tuesday", "by the end of the week", "next week",
    "today at 2pm", "by 11am", "at half two", "tomorrow at 9:30",
    "on the 3rd", "by the 28th",
]

DEFAULT_PRIORITY = 2  # the business rule when nobody says how urgent a job is

PRIORITY_CUES = {
    1: ["it's urgent", "top priority please", "that's holding production up so it needs doing first",
        "that one's safety critical, straight away", "drop everything for that one"],
    2: ["fairly important", "it matters but it's not urgent", "normal priority", "not an emergency but don't let it slip"],
    3: ["no rush", "whenever you get a chance", "low priority", "if you've got a spare minute"],
}

CHITCHAT = [
    "weekend plans and the weather", "the football last night", "the broken coffee machine in the canteen",
    "someone's new car in the car park", "the charity bake sale on Friday", "traffic on the ring road",
    "a birthday cake in the office", "the heating in the portakabin",
]

DONE_TEMPLATES = [
    "tested the {asset} yesterday and it's all fine now",
    "finished the repair on the {asset} last night",
    "already swapped the filter on the {asset} yesterday",
    "signed off the inspection on the {asset} on Tuesday",
]
OBSERVATION_TEMPLATES = [
    "noticed the {asset} was running a bit warm on nights but it settled down",
    "saw the pressure on the {asset} dip for a few minutes, it came back on its own",
    "heard a slight rattle from the {asset}, nothing to action yet",
]
BLOCKER_TEMPLATES = [
    "still waiting on parts for the {asset}, nothing we can do until they arrive",
    "can't get to the {asset} until production gives us a window",
    "the supplier still hasn't confirmed the delivery for the {asset}",
]
CANCELLED_TEMPLATES = [
    "we were going to replace the belt on the {asset} but that's been pushed to next month so ignore it",
    "the shutdown on the {asset} is cancelled now, production need it running",
]


def fill(template: str, **values: str) -> str:
    """Format a template, collapsing "the the" when an alias already starts with "the"."""
    return re.sub(r"\b(the|The) the\b", r"\1", template.format(**values))


def _mention(rng: random.Random, entity: Entity | None, split: str) -> Mention:
    """Choose how an entity is referred to.

    Train meetings use the canonical name or a known alias. Test meetings use a
    held-out alias about half of the time.
    """
    if entity is None:
        return Mention(entity_id=None, text=None, split="none")
    if split == "test" and entity.heldout_aliases and rng.random() < 0.5:
        return Mention(entity_id=entity.id, text=rng.choice(entity.heldout_aliases), split="heldout")
    if entity.aliases and rng.random() < 0.75:
        return Mention(entity_id=entity.id, text=rng.choice(entity.aliases), split="seen")
    return Mention(entity_id=entity.id, text=entity.name, split="name")


def _meeting_date(rng: random.Random) -> date:
    d = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 330))
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def sample_meeting(onto: Ontology, index: int, seed: int, split: str, prefix: str = "m") -> Meeting:
    rng = random.Random(f"{seed}-{index}")
    meeting_date = _meeting_date(rng)

    chair = onto.people[0] if rng.random() < 0.7 else rng.choice(onto.people[:6])
    others = [p for p in onto.people if p.id != chair.id]
    attendees = [chair, *rng.sample(others, 6)]
    by_id = onto.by_id
    assets = {a.id: a for a in onto.leaf_assets()}

    tasks: list[GoldTask] = []
    used_templates: set[int] = set()
    for t_index in range(rng.randint(5, 9)):
        ti = rng.choice([i for i in range(len(TEMPLATES)) if i not in used_templates])
        used_templates.add(ti)
        tpl = TEMPLATES[ti]

        asset = None
        if tpl.needs_asset:
            asset = by_id[rng.choice(tpl.assets)] if tpl.assets else rng.choice(list(assets.values()))
        site = rng.choice(onto.sites[1:]) if tpl.needs_site else None

        # ~10% of tasks have nobody clearly owning them ("someone needs to...").
        owner = None if rng.random() < 0.1 else rng.choice(attendees)
        assigned_by = chair.id if owner and owner.id != chair.id and rng.random() < 0.55 else None
        reassigned_from = None
        if owner and rng.random() < 0.12:
            reassigned_from = rng.choice([p for p in attendees if p.id not in (owner.id, chair.id)]).id

        due_phrase = rng.choice(DUE_PHRASES) if rng.random() < 0.85 else None
        priority = rng.choice(tpl.priorities)
        cue = rng.choice(PRIORITY_CUES[priority]) if rng.random() < 0.8 else None
        if cue is None:
            priority = DEFAULT_PRIORITY  # nothing said about urgency -> business default

        component = _mention(rng, asset, split)
        site_m = _mention(rng, site, split)
        owner_m = _mention(rng, owner, split)

        summary = fill(tpl.action, asset=asset.name if asset else "", site=site.name if site else "")
        spoken = fill(tpl.action, asset=component.text or "", site=site_m.text or "")
        tasks.append(
            GoldTask(
                id=f"{prefix}{index:03d}-t{t_index + 1}",
                summary=summary[0].upper() + summary[1:],
                spoken_action=spoken,
                category=tpl.category,
                priority=priority,
                owner=owner_m,
                component=component,
                site=site_m,
                due_date=resolve_date(due_phrase, meeting_date),
                due_time=resolve_time(due_phrase),
                due_mention=due_phrase,
                priority_cue=cue,
                assigned_by=assigned_by,
                reassigned_from=reassigned_from,
            )
        )

    distractors: list[Distractor] = []
    for _ in range(rng.randint(2, 3)):
        distractors.append(Distractor(kind="chitchat", speaker_id=rng.choice(attendees).id, text_hint=rng.choice(CHITCHAT)))
    for kind, templates, count in [
        ("done", DONE_TEMPLATES, rng.randint(2, 3)),
        ("observation", OBSERVATION_TEMPLATES, rng.randint(1, 2)),
        ("blocker", BLOCKER_TEMPLATES, rng.randint(1, 2)),
        ("cancelled", CANCELLED_TEMPLATES, rng.randint(0, 1)),
    ]:
        for _ in range(count):
            asset = rng.choice(list(assets.values()))
            m = _mention(rng, asset, split)
            distractors.append(
                Distractor(
                    kind=kind,
                    speaker_id=rng.choice(attendees).id,
                    text_hint=fill(rng.choice(templates), asset=m.text),
                    component=m,
                )
            )

    return Meeting(
        id=f"{prefix}{index:03d}",
        date=meeting_date,
        split=split,
        renderer="",
        attendees=[p.id for p in attendees],
        chair_id=chair.id,
        gold_tasks=tasks,
        distractors=distractors,
    )


def sample_dataset(onto: Ontology, n: int, seed: int, test_fraction: float, prefix: str = "m") -> list[Meeting]:
    n_test = max(1, round(n * test_fraction))
    return [sample_meeting(onto, i, seed, "test" if i >= n - n_test else "train", prefix) for i in range(n)]


def sample_extra_train(onto: Ontology, start: int, n: int, seed: int, prefix: str = "m") -> list[Meeting]:
    """More training meetings with ids after the existing ones; the test split is untouched."""
    return [sample_meeting(onto, i, seed, "train", prefix) for i in range(start, start + n)]
