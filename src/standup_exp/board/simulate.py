"""Sample a run of consecutive stand-ups and what happens to every card, before any text exists.

A series is one team over N working days. Day 1 starts from an empty board. Each later day:

* every open card gets a fate: done, progress, blocked, cancelled, reassigned, due changed,
  or not mentioned at all (weighted by status and whether it is due);
* 2-4 new jobs are agreed (some on equipment that already has an open card: "twins");
* distractors: chit-chat, plus "done"/observation remarks about equipment that has no card,
  which must NOT be matched to anything.

Updates carry hard-case tags (third_party, owner_absent, twin, activity_only, heldout,
picked_up, slipped) so the scores can be broken down by what makes linking hard.
Everything is driven by ``random.Random`` and is reproducible from the seed.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta

from ..dates import resolve_date, resolve_time
from ..generate.scenario import (CHITCHAT, DEFAULT_PRIORITY, DONE_TEMPLATES, DUE_PHRASES, OBSERVATION_TEMPLATES,
                                 PRIORITY_CUES, TEMPLATES, Template, _mention, fill)
from ..ontology import Ontology
from ..schema import Distractor, GoldTask
from .schema import Card, GoldUpdate, StandupDay, gold_board_after

MAX_UPDATES_PER_DAY = 6

PROGRESS_NOTES = [
    "about halfway through", "made a start on it", "got the first half done", "still on it, going fine so far",
    "done the prep, just the write-up left", "had a first look, nothing major so far",
]
BLOCKED_NOTES = [
    "waiting on the permit to be signed", "the parts haven't turned up", "production won't give us a window",
    "the contractor can't get here until the job's rescheduled", "need an electrician to isolate it first",
    "can't get hold of the supplier",
]
CANCELLED_NOTES = [
    "production changed the plan so it's not needed", "the other shift already sorted it",
    "the supplier's doing it under warranty instead", "it's been rolled into next month's shutdown",
]
DONE_NOTES = [None, None, None, "all fine", "no issues found", "all signed off"]


@dataclass
class _CardInfo:
    """Generator-side facts about a card that the board itself does not store."""

    template: int
    asset_id: str | None
    site_words: str = ""


def reference(tpl_index: int, component_words: str | None, site_words: str, style: str) -> str:
    """How a job is referred to aloud: with the equipment words, or by the activity alone."""
    action = TEMPLATES[tpl_index].action
    if style == "full" or "{asset}" not in action:
        return fill(action, asset=component_words or "", site=site_words)
    action = re.sub(r"\s*\b(on|for|at|to look at) the \{asset\}", "", action)
    action = re.sub(r"\bthe \{asset\}\s*", "", action)
    return fill(action, site=site_words).strip()


def _working_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _related(onto: Ontology, a: str | None, b: str | None) -> bool:
    """Same equipment, or one is part of the other (GT1 vs GT1 Compressor Section)."""
    return bool(a and b and (a == b or a in onto.ancestors(b) or b in onto.ancestors(a)))


def _weights(card: Card, today: date) -> dict[str, float]:
    if card.owner_id is None:
        return {"reassigned": 4, "cancelled": 0.6, "silent": 5}
    if card.status == "blocked":
        return {"done": 2, "progress": 3, "cancelled": 0.8, "silent": 4}
    due = card.due_date is not None and card.due_date <= today
    return {"done": 3 + 2.5 * due, "progress": 2, "blocked": 0.8, "cancelled": 0.35, "reassigned": 0.5,
            "due_changed": 0.6 + 0.9 * due, "silent": 3.2}


def _pick(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, [weights[k] for k in keys])[0]


def _future_due(rng: random.Random, today: date, current: date | None) -> str | None:
    options = [p for p in DUE_PHRASES if (d := resolve_date(p, today)) and d > today and d != current]
    return rng.choice(options) if options else None


def _new_task(rng: random.Random, onto: Ontology, task_id: str, today: date, attendees: list, chair,
              tpl_index: int, asset_id: str | None) -> GoldTask:
    tpl: Template = TEMPLATES[tpl_index]
    asset = onto.by_id[asset_id] if asset_id else None
    site = rng.choice(onto.sites[1:]) if tpl.needs_site else None
    owner = None if rng.random() < 0.08 else rng.choice(attendees)
    assigned_by = chair.id if owner and owner.id != chair.id and rng.random() < 0.5 else None
    reassigned_from = None
    if owner and rng.random() < 0.08:
        pool = [p for p in attendees if p.id not in (owner.id, chair.id)]
        reassigned_from = rng.choice(pool).id if pool else None
    due_phrase = rng.choice(DUE_PHRASES) if rng.random() < 0.85 else None
    priority = rng.choice(tpl.priorities)
    cue = rng.choice(PRIORITY_CUES[priority]) if rng.random() < 0.8 else None
    if cue is None:
        priority = DEFAULT_PRIORITY
    component = _mention(rng, asset, "test")
    site_m = _mention(rng, site, "test")
    summary = fill(tpl.action, asset=asset.name if asset else "", site=site.name if site else "")
    spoken = fill(tpl.action, asset=component.text or "", site=site_m.text or "")
    return GoldTask(
        id=task_id, summary=summary[0].upper() + summary[1:], spoken_action=spoken, category=tpl.category,
        priority=priority, owner=_mention(rng, owner, "test"), component=component, site=site_m,
        due_date=resolve_date(due_phrase, today), due_time=resolve_time(due_phrase), due_mention=due_phrase,
        priority_cue=cue, assigned_by=assigned_by, reassigned_from=reassigned_from,
    )


def _choose_template(rng: random.Random, onto: Ontology, open_info: list[_CardInfo], twin: bool
                     ) -> tuple[int, str | None]:
    """A (template, asset) pair that is not already an open card; ``twin`` reuses open equipment."""
    taken = {(i.template, i.asset_id) for i in open_info}
    leaf = [a.id for a in onto.leaf_assets()]
    if twin:
        for info in rng.sample(open_info, len(open_info)):
            if not info.asset_id:
                continue
            fits = [i for i, t in enumerate(TEMPLATES)
                    if t.needs_asset and i != info.template and (not t.assets or info.asset_id in t.assets)
                    and (i, info.asset_id) not in taken]
            if fits:
                return rng.choice(fits), info.asset_id
    for _ in range(200):
        i = rng.randrange(len(TEMPLATES))
        t = TEMPLATES[i]
        asset = (rng.choice(t.assets) if t.assets else rng.choice(leaf)) if t.needs_asset else None
        if (i, asset) not in taken and not (not t.needs_asset and any(x.template == i for x in open_info)):
            return i, asset
    raise RuntimeError("could not find a free job template")


def sample_series(onto: Ontology, index: int, seed: int, days: int = 8, prefix: str = "s") -> list[StandupDay]:
    rng = random.Random(f"{seed}-series-{index}")
    start = date(2026, 2, 2) + timedelta(weeks=rng.randrange(0, 38))  # a Monday
    dates = _working_days(start, days)
    chair = onto.people[0] if rng.random() < 0.7 else rng.choice(onto.people[:6])
    team = [chair, *rng.sample([p for p in onto.people if p.id != chair.id], 7)]
    by_id = onto.by_id

    board: list[Card] = []
    info: dict[str, _CardInfo] = {}
    series: list[StandupDay] = []
    for d, today in enumerate(dates, start=1):
        day_id = f"{prefix}{index:02d}d{d:02d}"
        absent = set(rng.sample([p.id for p in team[1:]], rng.choice([0, 1, 1, 2]))) if d > 1 else set()
        attendees = [p for p in team if p.id not in absent]
        present = {p.id for p in attendees}
        open_cards = sorted((c for c in board if c.is_open), key=lambda c: c.id)

        # ------------------------------------------------------------ updates
        planned: list[tuple[Card, str]] = []
        for card in open_cards:
            fate = _pick(rng, _weights(card, today))
            if fate == "silent":
                continue
            if card.owner_id and card.owner_id not in present and rng.random() > 0.35:
                continue  # owner is off today and nobody mentions it
            planned.append((card, fate))
        if len(planned) > MAX_UPDATES_PER_DAY:
            planned = rng.sample(planned, MAX_UPDATES_PER_DAY)
            planned.sort(key=lambda cf: cf[0].id)

        updates: list[GoldUpdate] = []
        for card, fate in planned:
            ci = info[card.id]
            tags: list[str] = []
            others = [p for p in attendees if p.id != card.owner_id]
            new_owner = None
            if fate == "reassigned":
                target = rng.choice(others)
                new_owner = _mention(rng, target, "test")
                if card.owner_id is None:
                    speaker, tags = target.id, ["picked_up"]
                else:
                    speaker = rng.choice([chair.id, card.owner_id]) if card.owner_id in present else chair.id
            elif card.owner_id is None or card.owner_id not in present:
                speaker = chair.id
                tags.append("owner_absent" if card.owner_id else "unowned")
            elif rng.random() < 0.2:
                speaker = rng.choice(others).id
            else:
                speaker = card.owner_id
            if card.owner_id and speaker != card.owner_id and fate != "reassigned":
                tags.append("third_party")

            twins = [c for c in open_cards if c.id != card.id and _related(onto, c.component_id, card.component_id)]
            same_job_twin = any(info[c.id].template == ci.template for c in twins)
            if twins:
                tags.append("twin")
            style = "full"
            if not ci.asset_id:
                style = "activity"
            elif not same_job_twin and rng.random() < 0.35:
                style, tags = "activity", [*tags, "activity_only"]
            component = _mention(rng, by_id[ci.asset_id], "test") if style == "full" and ci.asset_id else None
            if component is not None and component.split == "heldout":
                tags.append("heldout")

            due_phrase = _future_due(rng, today, card.due_date) if fate == "due_changed" else None
            if fate == "due_changed" and due_phrase is None:
                fate = "progress"
            if fate == "due_changed":
                tags.append("slipped")
            note = {"progress": rng.choice(PROGRESS_NOTES), "blocked": rng.choice(BLOCKED_NOTES),
                    "cancelled": rng.choice(CANCELLED_NOTES), "done": rng.choice(DONE_NOTES)}.get(fate)
            owner_words = _mention(rng, by_id[card.owner_id], "test").text if card.owner_id else None
            updates.append(GoldUpdate(
                id=f"{day_id}-u{len(updates) + 1}", kind=fate, card_id=card.id, speaker_id=speaker, style=style,
                reference=reference(ci.template, component.text if component else None, ci.site_words, style),
                component=component, owner_words=owner_words, note=note, new_owner=new_owner,
                due_mention=due_phrase, due_date=resolve_date(due_phrase, today), due_time=resolve_time(due_phrase),
                tags=tags,
            ))

        # ------------------------------------------------------------ new jobs
        still_open = [info[c.id] for c in open_cards if c.id not in {u.card_id for u in updates if u.kind in
                                                                        ("done", "cancelled")}]
        creates: list[GoldTask] = []
        for t in range(rng.randint(5, 7) if d == 1 else rng.randint(2, 4)):
            twin = d > 1 and rng.random() < 0.2
            tpl_index, asset_id = _choose_template(rng, onto, still_open, twin)
            task = _new_task(rng, onto, f"{day_id}-t{t + 1}", today, attendees, chair, tpl_index, asset_id)
            creates.append(task)
            site = onto.by_id[task.site.entity_id] if task.site.entity_id else None
            info[task.id] = _CardInfo(tpl_index, asset_id, (site.aliases[0] if site and site.aliases else
                                                            site.name if site else ""))
            still_open.append(info[task.id])

        # --------------------------------------------------------- distractors
        ever = {i.asset_id for i in info.values() if i.asset_id}
        spare = [a for a in onto.leaf_assets() if not any(_related(onto, a.id, e) for e in ever)]
        distractors = [Distractor(kind="chitchat", speaker_id=rng.choice(attendees).id, text_hint=rng.choice(CHITCHAT))
                       for _ in range(rng.randint(1, 2))]
        for kind, templates, p in (("done", DONE_TEMPLATES, 0.7), ("observation", OBSERVATION_TEMPLATES, 0.5)):
            if spare and rng.random() < p:
                asset = spare.pop(rng.randrange(len(spare)))
                m = _mention(rng, asset, "test")
                distractors.append(Distractor(kind=kind, speaker_id=rng.choice(attendees).id,
                                              text_hint=fill(rng.choice(templates), asset=m.text), component=m))

        day = StandupDay(
            id=day_id, date=today, split="test", renderer="", attendees=[p.id for p in attendees],
            chair_id=chair.id, gold_tasks=creates, distractors=distractors, series_id=f"{prefix}{index:02d}",
            day=d, team=[p.id for p in team], board_before=[c.model_copy(deep=True) for c in board], updates=updates,
        )
        board = gold_board_after(day)
        series.append(day)
    return series


def sample_all(onto: Ontology, n_series: int, days: int, seed: int, prefix: str = "s") -> list[list[StandupDay]]:
    return [sample_series(onto, i + 1, seed, days, prefix) for i in range(n_series)]
