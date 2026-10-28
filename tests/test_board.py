"""Offline tests for the task-board loop: no Azure calls."""

from datetime import date

import pytest

from standup_exp.board.evaluate import metrics, score_series
from standup_exp.board.pipeline import DayRun, _update_models
from standup_exp.board.render import check_day, render_day_template
from standup_exp.board.schema import BoardEvent, Card, apply, gold_board_after
from standup_exp.board.simulate import reference, sample_all
from standup_exp.generate.scenario import TEMPLATES
from standup_exp.llm import strict_schema
from standup_exp.ontology import Ontology
from standup_exp.schema import NormalisedTask, RawTask


@pytest.fixture(scope="module")
def onto():
    return Ontology.load()


@pytest.fixture(scope="module")
def series(onto):
    out = sample_all(onto, 2, 6, seed=5)
    for days in out:
        for d in days:
            render_day_template(d, onto)
    return out


def test_simulation_is_reproducible(onto):
    a = sample_all(onto, 2, 5, seed=9)
    b = sample_all(onto, 2, 5, seed=9)
    assert [d.model_dump_json() for s in a for d in s] == [d.model_dump_json() for s in b for d in s]


def test_board_carries_over_and_updates_target_open_cards(series):
    for days in series:
        assert days[0].board_before == []
        for prev, day in zip(days, days[1:]):
            assert [c.id for c in day.board_before] == [c.id for c in gold_board_after(prev)]
            open_ids = {c.id for c in day.board_before if c.is_open}
            assert {u.card_id for u in day.updates} <= open_ids
            assert len({u.card_id for u in day.updates}) == len(day.updates)
            assert day.date > prev.date and day.date.weekday() < 5


def test_template_transcripts_pass_validation(series, onto):
    for days in series:
        for d in days:
            problems = [p for p in check_day(d, onto) if not p.startswith("transcript too short")]
            assert problems == [], d.id


def test_activity_reference_drops_equipment():
    i = next(i for i, t in enumerate(TEMPLATES) if t.action == "take vibration readings on the {asset}")
    assert reference(i, "the old turbine", "", "full") == "take vibration readings on the old turbine"
    assert reference(i, None, "", "activity") == "take vibration readings"


def test_apply_moves_status_and_keeps_history():
    day = type("D", (), {"id": "x01", "date": date(2026, 3, 2)})()
    card = Card(id="c1", summary="Swap the filter", created_in="x00", created_on=date(2026, 2, 27))
    after = apply([card], [BoardEvent(day_id="x01", kind="blocked", card_id="c1", note="no parts"),
                           BoardEvent(day_id="x01", kind="reassigned", card_id="c1", owner_id="p02")], day)
    assert after[0].status == "blocked" and after[0].owner_id == "p02"
    assert [h.text for h in after[0].history] == ["no parts", "Reassigned"]
    assert card.status == "todo"  # input untouched


def _perfect(days, mode):
    """A pipeline that returns the answer key, expressed in its own card ids."""
    runs, board, g2p = [], [], {}
    for day in days:
        before = day.board_before if mode == "isolated" else board
        if mode == "isolated":
            g2p = {c.id: c.id for c in day.board_before}
        new_jobs, events = [], []
        for i, t in enumerate(day.gold_tasks):
            new_jobs.append(NormalisedTask(
                summary=t.summary, owner_id=t.owner.entity_id, component_id=t.component.entity_id,
                site_id=t.site.entity_id, due_date=t.due_date, due_time=t.due_time, priority=t.priority,
                category=t.category, evidence_quote=day.utterances[t.evidence[0]].text,
                raw=RawTask(summary=t.summary, confidence=1.0)))
            events.append(BoardEvent(day_id=day.id, kind="create", card_id=f"{day.id}-p{i + 1}", summary=t.summary,
                                     owner_id=t.owner.entity_id, component_id=t.component.entity_id,
                                     site_id=t.site.entity_id, due_date=t.due_date, due_time=t.due_time,
                                     priority=t.priority, category=t.category))
            g2p[t.id] = f"{day.id}-p{i + 1}"
        for u in day.updates:
            events.append(BoardEvent(day_id=day.id, kind=u.kind, card_id=g2p[u.card_id], due_date=u.due_date,
                                     due_time=u.due_time, owner_id=u.new_owner.entity_id if u.new_owner else None))
        after = apply(before, events, day)
        runs.append(DayRun(day.id, mode, new_jobs, {}, events, before, after))
        board = after
    return runs


@pytest.mark.parametrize("mode", ["chained", "isolated"])
def test_perfect_pipeline_scores_full_marks(series, onto, mode):
    for days in series:
        counts, _ = score_series(onto, days, _perfect(days, mode))
        m = metrics(counts)
        assert m["create_dashboard_ready"] == 1.0
        assert m["update_recall"] == 1.0 and m["update_precision"] == 1.0
        assert m["false_closes"] == 0 and m["duplicates"] == 0
        assert m["board_accuracy_final"] == 1.0 and m["final_extra_cards"] == 0


def test_false_close_is_caught(series, onto):
    days = series[0]
    runs = _perfect(days, "isolated")
    k = next(i for i, d in enumerate(days)
             if any(c.is_open and c.id not in {u.card_id for u in d.updates} for c in d.board_before))
    victim = next(c for c in days[k].board_before if c.is_open and c.id not in {u.card_id for u in days[k].updates})
    runs[k].events.append(BoardEvent(day_id=days[k].id, kind="done", card_id=victim.id))
    runs[k].board_after = apply(runs[k].board_before, runs[k].events, days[k])
    m = metrics(score_series(onto, days, runs)[0])
    assert m["false_closes"] == 1 and m["update_precision"] < 1.0
    assert m["board_accuracy_daily"] < 1.0


def test_done_said_for_progress_counts_as_false_close(series, onto):
    days = next(s for s in series if any(u.kind == "progress" for d in s for u in d.updates))
    runs = _perfect(days, "isolated")
    k, u = next((i, u) for i, d in enumerate(days) for u in d.updates if u.kind == "progress")
    e = next(e for e in runs[k].events if e.kind == "progress" and e.card_id == u.card_id)
    e.kind = "done"
    runs[k].board_after = apply(runs[k].board_before, runs[k].events, days[k])
    m = metrics(score_series(onto, days, runs)[0])
    assert m["false_closes"] == 1 and m["updates_wrong_change"] == 1


def test_extra_cards_lower_board_accuracy(series, onto):
    days = series[0]
    runs = _perfect(days, "chained")
    last = runs[-1]
    last.events.append(BoardEvent(day_id=days[-1].id, kind="create", card_id="invented", summary="Made up job"))
    last.board_after = apply(last.board_before, last.events, days[-1])
    m = metrics(score_series(onto, days, runs)[0])
    assert m["final_extra_cards"] == 1 and m["board_accuracy_final"] < 1.0


def test_comment_keeps_status():
    day = type("D", (), {"id": "x01", "date": date(2026, 3, 2)})()
    card = Card(id="c1", summary="Swap the filter", created_in="x00", created_on=date(2026, 2, 27))
    after = apply([card], [BoardEvent(day_id="x01", kind="comment", card_id="c1", note="Mentioned again")], day)
    assert after[0].status == "todo" and after[0].history[-1].text == "Mentioned again"


def test_locate_ignores_short_turns_inside_the_quote(series):
    from standup_exp.board.evaluate import locate
    from standup_exp.schema import Utterance

    day = series[0][0].model_copy(deep=True)
    day.utterances = [Utterance(start=0, end=1, speaker_id="p01", speaker="Sarah Okafor", text="Understood."),
                      Utterance(start=1, end=2, speaker_id="p02", speaker="David Whitfield",
                                text="Yeah, I'll schedule downtime for the Atlas at half two. Low priority, understood.")]
    assert locate("Yeah, I'll schedule downtime for the Atlas at half two. Low priority, understood.", day) == 1


def test_update_schema_only_allows_real_ids():
    schema = strict_schema(_update_models(["s01d01-t1", "s01d01-t2"], ["p01", "p02"], 1))
    text = str(schema)
    assert "'s01d01-t1'" in text and "'none'" in text and "'p02'" in text



# ------------------------------------------------------------ context pipeline (POC)
from standup_exp.board.context import context_for, related  # noqa: E402
from standup_exp.board.context_pipeline import ContextPipeline  # noqa: E402
from standup_exp.board.pipeline import StepCost  # noqa: E402
from standup_exp.board.store import TABLES, BoardStore  # noqa: E402


def _update_people(u, card):
    """Who the words name: the speaker, the owner if someone reports for them, the new owner."""
    people = {u.speaker_id}
    if u.owner_words and card.owner_id:
        people.add(card.owner_id)
    if u.new_owner:
        people.add(u.new_owner.entity_id)
    return people


def test_context_holds_the_right_card(series, onto):
    hits = n = 0
    for days in series:
        for d in days:
            cards = {c.id: c for c in d.board_before}
            for u in d.updates:
                ctx = context_for(d.board_before, onto, _update_people(u, cards[u.card_id]),
                                  u.component.entity_id if u.component else None, u.kind == "reassigned")
                n += 1
                hits += u.card_id in {c.id for c in ctx}
    assert n and hits == n


def test_related_includes_parts():
    o = Ontology.load()
    assert related(o, "GT1", "GT1-CMP") and related(o, "GT1-CMP", "GT1") and not related(o, "GT1", "GT2")


def _oracle(onto, days):
    """Perfect Step 1 (mentions from the answer key) and a perfect Step 3 pick, in the pipeline's own card ids."""
    ours: dict[str, str] = {}  # true card id -> our card id
    step1 = {}
    for d in days:
        mentions, linked = [], []
        cards = {c.id: c for c in d.board_before}
        k = 0
        for t in d.gold_tasks:
            mentions.append({"mention_id": f"{d.id}-m{len(mentions) + 1}", "said": "new", "summary": t.summary,
                             "speaker_id": t.owner.entity_id, "owner_id": t.owner.entity_id, "new_owner_id": None,
                             "component_id": t.component.entity_id, "quote": d.utterances[t.evidence[0]].text,
                             "note": None, "confidence": 1.0})
            linked.append(NormalisedTask(summary=t.summary, owner_id=t.owner.entity_id, component_id=t.component.entity_id,
                                         site_id=t.site.entity_id, due_date=t.due_date, due_time=t.due_time,
                                         priority=t.priority, category=t.category,
                                         evidence_quote=d.utterances[t.evidence[0]].text, raw=RawTask(summary=t.summary)))
            k += 1
            ours[t.id] = f"{d.id}-p{k}"
        for u in d.updates:
            card = cards[u.card_id]
            owner = card.owner_id if (u.owner_words or u.speaker_id == card.owner_id) else None
            mentions.append({"mention_id": f"{d.id}-m{len(mentions) + 1}", "said": u.kind, "summary": card.summary,
                             "speaker_id": u.speaker_id, "owner_id": owner,
                             "new_owner_id": u.new_owner.entity_id if u.new_owner else None,
                             "component_id": u.component.entity_id if u.component else None,
                             "quote": d.utterances[u.evidence[0]].text, "note": u.note, "confidence": 1.0,
                             "_gold_card": u.card_id})
            linked.append(NormalisedTask(summary=card.summary, due_date=u.due_date, due_time=u.due_time,
                                         raw=RawTask(summary=card.summary)))
        mentions.append({"mention_id": f"{d.id}-m{len(mentions) + 1}", "said": "done", "summary": "phantom remark",
                         "speaker_id": d.chair_id, "owner_id": d.chair_id, "new_owner_id": None, "component_id": None,
                         "quote": "nothing to see here at all", "note": None, "confidence": 1.0})
        linked.append(NormalisedTask(summary="phantom remark", raw=RawTask(summary="phantom remark")))
        step1[d.id] = (mentions, linked)

    def pick(day, mentions, contexts):
        return {i: ours[m["_gold_card"]] for i, m in enumerate(mentions)
                if m.get("_gold_card") and ours[m["_gold_card"]] in {c.id for c in contexts[i]}}

    p = ContextPipeline.__new__(ContextPipeline)
    p.onto, p.model, p.names = onto, "oracle", {x.id: x.name for x in onto.people}
    p.prepare = lambda day: (*step1[day.id], {})
    p._pick = lambda day, mentions, contexts: (pick(day, mentions, contexts), {}, StepCost())
    return p


def test_perfect_steps_give_a_perfect_board_through_the_database(series, onto, tmp_path):
    days = series[0]
    store = BoardStore(tmp_path / "db.json", fresh=True)
    runs = _oracle(onto, days).run_series(days, "chained", store)
    m = metrics(score_series(onto, days, runs)[0])
    assert m["create_dashboard_ready"] == 1.0 and m["update_recall"] == 1.0 and m["update_precision"] == 1.0
    assert m["false_closes"] == 0 and m["duplicates"] == 0 and m["board_accuracy_final"] == 1.0
    s = m["steps"]
    assert s["step1_new_jobs_found"] == 1.0 and s["step1_updates_found"] == 1.0 and s["step1_said_correct"] == 1.0
    assert s["step2_context_recall"] == 1.0 and s["step3_pick_accuracy"] == 1.0
    assert s["extra_update_mentions"] == len(days) and s["extra_update_mentions_ignored"] == len(days)
    store.save()
    again = BoardStore(tmp_path / "db.json")
    assert set(again.to_dict()) >= set(TABLES)
    assert sorted(c.model_dump_json() for c in again.board(days[0].series_id)) == \
        sorted(c.model_dump_json() for c in runs[-1].board_after)
    assert len(again.rows["standups"]) == len(days) and len(again.rows["decisions"]) == len(again.rows["mentions"])


def test_new_mention_matching_a_card_comments_and_updates_due_date(onto):
    day = type("D", (), {"id": "x02", "date": date(2026, 3, 3), "series_id": "x", "attendees": ["p01"],
                         "team": ["p01"], "chair_id": "p01"})()
    card = Card(id="c1", summary="Swap the filter on the Cooling Tower", owner_id="p01", component_id="WTR-CT-01",
                due_date=date(2026, 3, 4), created_in="x01", created_on=date(2026, 3, 2))
    mention = {"mention_id": "x02-m1", "said": "new", "summary": "Swap the filter on the tower", "speaker_id": "p01",
               "owner_id": "p01", "new_owner_id": None, "component_id": "WTR-CT-01", "quote": "I'll swap it Friday",
               "note": None, "confidence": 0.9}
    ghost = {**mention, "mention_id": "x02-m2", "said": "done", "summary": "Test the long belt", "component_id": "PCK-CV-01",
             "owner_id": None, "speaker_id": "p09"}
    linked = [NormalisedTask(summary=mention["summary"], owner_id="p01", due_date=date(2026, 3, 6), raw=RawTask(summary="x")),
              NormalisedTask(summary="Test the long belt", raw=RawTask(summary="y"))]
    p = ContextPipeline.__new__(ContextPipeline)
    p.onto, p.model, p.names = onto, "stub", {x.id: x.name for x in onto.people}
    p.prepare = lambda d: ([mention, ghost], linked, {})
    p._pick = lambda d, ms, ctx: ({0: "c1"}, {0: "same filter job said again", 1: "no card for the belt test"}, StepCost())
    run = p.run_day(day, [card])
    assert [e.kind for e in run.events] == ["comment", "due_changed"]
    assert run.merged == {0: "c1"} and len(run.board_after) == 1 and run.board_after[0].due_date == date(2026, 3, 6)
    assert [d["action"] for d in run.why["decisions"]] == ["comment", "ignore"]
    assert [d["pick_reason"] for d in run.why["decisions"]] == ["same filter job said again", "no card for the belt test"]


def test_the_pick_schema_asks_for_the_reason_before_the_card():
    from standup_exp.board.context_pipeline import _pick_model

    item = strict_schema(_pick_model(["c1", "c2"]))["$defs"]["Pick"]
    assert list(item["properties"]) == ["mention", "reason", "card"] and item["required"] == ["mention", "reason", "card"]
    assert item["properties"]["card"]["enum"] == ["c1", "c2", "none"]



def test_step_scores_do_not_credit_an_unrelated_mention(series, onto):
    days = series[0]
    p = _oracle(onto, days)
    orig = p.prepare
    changed = {}

    def worse(day):
        ms, linked, s = orig(day)
        ms = [dict(m) for m in ms]
        for m in ms:
            if m.get("_gold_card") and day.id not in changed:  # unrelated wording, and the pick does not tie it
                m["summary"], m["_gold_card"], changed[day.id] = "chat about the football results", None, True
        return ms, linked, s

    p.prepare = worse
    s = metrics(score_series(onto, days, p.run_series(days, "chained"))[0])["steps"]
    assert changed and s["step1_updates_found"] < 1.0


def test_two_different_due_dates_for_one_card_both_apply(onto):
    day = type("D", (), {"id": "x02", "date": date(2026, 3, 3), "series_id": "x", "attendees": ["p01"],
                         "team": ["p01"], "chair_id": "p01"})()
    card = Card(id="c1", summary="Swap the filter", owner_id="p01", created_in="x01", created_on=date(2026, 3, 2))
    m1 = {"mention_id": "x02-m1", "said": "due_changed", "summary": "Swap the filter", "speaker_id": "p01",
          "owner_id": "p01", "new_owner_id": None, "component_id": None, "quote": "Thursday now", "note": None,
          "confidence": 1.0}
    m2 = {**m1, "mention_id": "x02-m2", "quote": "actually Friday"}
    linked = [NormalisedTask(summary="x", due_date=date(2026, 3, 5), raw=RawTask(summary="x")),
              NormalisedTask(summary="x", due_date=date(2026, 3, 6), raw=RawTask(summary="x"))]
    p = ContextPipeline.__new__(ContextPipeline)
    p.onto, p.model, p.names = onto, "stub", {x.id: x.name for x in onto.people}
    p.prepare = lambda d: ([m1, m2], linked, {})
    p._pick = lambda d, ms, ctx: ({0: "c1", 1: "c1"}, {}, StepCost())
    run = p.run_day(day, [card])
    assert [e.kind for e in run.events] == ["due_changed", "due_changed"]
    assert run.board_after[0].due_date == date(2026, 3, 6)



def _stub_day(onto, mentions, linked, picks, board):
    day = type("D", (), {"id": "x02", "date": date(2026, 3, 3), "series_id": "x", "attendees": ["p01"],
                         "team": ["p01"], "chair_id": "p01"})()
    p = ContextPipeline.__new__(ContextPipeline)
    p.onto, p.model, p.names = onto, "stub", {x.id: x.name for x in onto.people}
    p.prepare = lambda d: (mentions, linked, {})
    p._pick = lambda d, ms, ctx: (picks, {}, StepCost())
    return p.run_day(day, board)


def test_due_date_changes_apply_in_order_even_back_to_the_first(onto):
    card = Card(id="c1", summary="Swap the filter", owner_id="p01", created_in="x01", created_on=date(2026, 3, 2))
    base = {"said": "due_changed", "summary": "Swap the filter", "speaker_id": "p01", "owner_id": "p01",
            "new_owner_id": None, "component_id": None, "note": None, "confidence": 1.0}
    ms = [{**base, "mention_id": f"x02-m{i}", "quote": q} for i, q in enumerate(["Thursday", "no, Friday", "Thursday after all"], 1)]
    days = [date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 5)]
    linked = [NormalisedTask(summary="x", due_date=d, raw=RawTask(summary="x")) for d in days]
    run = _stub_day(onto, ms, linked, {0: "c1", 1: "c1", 2: "c1"}, [card])
    assert [e.kind for e in run.events] == ["due_changed"] * 3
    assert run.board_after[0].due_date == date(2026, 3, 5)


def test_restatement_compares_with_changes_made_earlier_in_the_meeting(onto):
    card = Card(id="c1", summary="Swap the filter", owner_id="p01", due_date=date(2026, 3, 5),
                created_in="x01", created_on=date(2026, 3, 2))
    moved = {"mention_id": "x02-m1", "said": "due_changed", "summary": "Swap the filter", "speaker_id": "p01",
             "owner_id": "p01", "new_owner_id": None, "component_id": None, "quote": "Friday now", "note": None,
             "confidence": 1.0}
    back = {**moved, "mention_id": "x02-m2", "said": "new", "quote": "I'll swap the filter Thursday"}
    linked = [NormalisedTask(summary="x", due_date=date(2026, 3, 6), raw=RawTask(summary="x")),
              NormalisedTask(summary="Swap the filter", owner_id="p01", due_date=date(2026, 3, 5), raw=RawTask(summary="y"))]
    run = _stub_day(onto, [moved, back], linked, {0: "c1", 1: "c1"}, [card])
    assert [e.kind for e in run.events] == ["due_changed", "comment", "due_changed"]
    assert run.board_after[0].due_date == date(2026, 3, 5)


def test_an_exact_repeat_of_the_last_change_is_skipped(onto):
    card = Card(id="c1", summary="Swap the filter", owner_id="p01", created_in="x01", created_on=date(2026, 3, 2))
    m = {"mention_id": "x02-m1", "said": "done", "summary": "Swap the filter", "speaker_id": "p01", "owner_id": "p01",
         "new_owner_id": None, "component_id": None, "quote": "done", "note": None, "confidence": 1.0}
    linked = [NormalisedTask(summary="x", raw=RawTask(summary="x"))] * 2
    run = _stub_day(onto, [m, {**m, "mention_id": "x02-m2"}], linked, {0: "c1", 1: "c1"}, [card])
    assert [e.kind for e in run.events] == ["done"] and run.why["decisions"][1]["action"] == "ignore"


# ---------------------------------------------------------------- traces + explorer
def test_trace_calls_records_only_inside_the_block(monkeypatch, tmp_path):
    from pydantic import BaseModel

    from standup_exp import llm

    class Out(BaseModel):
        answer: str

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("LLM_CACHE_ONLY", raising=False)
    monkeypatch.setattr(llm, "_call_responses", lambda spec, *a: (Out(answer="yes"), (10, 5)))
    model = next(m for m in llm.model_ids() if llm.get_model(m).api != "chat_json")
    with llm.trace_calls() as calls:
        llm.structured("probe", "Say yes.", "hello", Out, model=model)
        llm.structured("probe", "Say yes.", "hello", Out, model=model)  # second call comes from the cache
    llm.structured("probe", "Say yes.", "again", Out, model=model)
    assert [c["cached"] for c in calls] == [False, True]
    assert calls[0]["instructions"] == "Say yes." and calls[0]["input"] == "hello"
    assert calls[0]["output"] == {"answer": "yes"} and calls[0]["input_tokens"] == 10
    assert calls[0]["schema"]["properties"]["answer"]["type"] == "string"


def test_explorer_has_a_row_for_every_answer_key_item():
    from standup_exp.board.explorer import export_explorer
    from standup_exp.board.runner import results_path

    if not results_path("context", "gpt-5.6-luna").exists():
        pytest.skip("no recorded context-pipeline run")
    from standup_exp.board.runner import load_series

    d = export_explorer()
    gold = [r for r in d["rows"] if r["kind"] != "extra"]
    assert len({r["id"] for r in d["rows"]}) == len(d["rows"])
    assert len(gold) == sum(len(day.gold_tasks) + len(day.updates) for days in load_series() for day in days)
    for r in d["rows"]:
        assert r["status"] in ("ok", "bad") and r["stages"][0]["key"] == "raw"
        assert (r["failed_at"] is None) == (r["status"] == "ok")
