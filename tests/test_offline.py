"""Offline tests: no Azure calls, no model downloads."""

from datetime import date

import pytest

from standup_exp.dates import resolve_date, resolve_time
from standup_exp.evaluate import Scoreboard, match
from standup_exp.extractors import TranscriptView, get_extractor
from standup_exp.generate.render import render_template
from standup_exp.generate.scenario import fill, sample_dataset
from standup_exp.generate.validate import check_meeting
from standup_exp.llm import _parse, get_model, model_ids
from standup_exp.normalisers import NormContext, get_normaliser
from standup_exp.ontology import Ontology
from standup_exp.runner import _compatible, gold_as_raw
from standup_exp.schema import RawTask

THU = date(2026, 1, 8)  # a Thursday


@pytest.fixture(scope="module")
def onto():
    return Ontology.load()


@pytest.fixture(scope="module")
def meetings(onto):
    ms = sample_dataset(onto, 6, seed=3, test_fraction=0.5)
    for m in ms:
        render_template(m, onto)
    return ms


# ---------------------------------------------------------------- dates
@pytest.mark.parametrize("phrase,expected", [
    ("today", THU), ("tomorrow morning", date(2026, 1, 9)), ("by Thursday", date(2026, 1, 15)),
    ("end of the week", date(2026, 1, 9)), ("next week", date(2026, 1, 12)), (None, None),
])
def test_resolve_date(phrase, expected):
    assert resolve_date(phrase, THU) == expected


def test_resolve_time():
    assert resolve_time("at half two") == "14:30"
    assert resolve_time("by 11am") == "11:00"


# -------------------------------------------------------------- ontology
def test_ontology_hierarchy(onto):
    assert onto.by_id["GT2-WV-03"].parent is not None
    assert onto.ancestors("GT2-WV-03")[-1] == "CHP"


def test_heldout_aliases_are_hidden(onto):
    e = onto.by_id["IM3-CHL"]
    assert set(e.heldout_aliases).isdisjoint(e.known_surface_forms())


# ------------------------------------------------------------ generation
def test_fill_collapses_double_article():
    assert fill("check the {asset}", asset="the long belt") == "check the long belt"


def test_template_meetings_validate(onto, meetings):
    for m in meetings:
        assert check_meeting(m, onto) == [], m.id


def test_train_split_never_uses_heldout(meetings):
    for m in meetings:
        if m.split == "train":
            assert all(g.component.split != "heldout" for g in m.gold_tasks)


# ----------------------------------------------------------- extraction
def test_rules_extractor_finds_tasks(onto, meetings):
    names = {p.id: p.name for p in onto.people}
    ex = get_extractor("rules")
    m = meetings[0]
    result = ex.extract(TranscriptView.of(m, names))
    assert result.tasks, "rules extractor found nothing on a template meeting"


def test_view_hides_gold(onto, meetings):
    view = TranscriptView.of(meetings[0], {p.id: p.name for p in onto.people})
    assert not hasattr(view, "gold_tasks")


# -------------------------------------------------------- normalisation
@pytest.mark.parametrize("name", ["alias", "fuzzy", "classifier"])
def test_known_forms_link_perfectly(onto, meetings, name):
    train = [m for m in meetings if m.split == "train"]
    norm = get_normaliser(name, onto, train)
    for m in meetings:
        ctx = NormContext(m.id, m.date, tuple(m.attendees))
        for g, raw in gold_as_raw(m):
            if g.component.split in ("name", "seen"):
                assert norm.normalise(raw, ctx).component_id == g.component.entity_id


def test_fuzzy_survives_typo(onto):
    norm = get_normaliser("fuzzy", onto, [])
    ctx = NormContext("x", THU, ())
    out = norm.normalise(RawTask(summary="x", component_mention="boiler feedwater pmp"), ctx)
    assert out.component_id == "WHB-01-FWP"


# ------------------------------------------------------------ evaluation
def test_perfect_predictions_score_100(onto, meetings):
    norm = get_normaliser("alias", onto, [])
    board = Scoreboard(onto)
    m = next(m for m in meetings if m.split == "train")
    ctx = NormContext(m.id, m.date, tuple(m.attendees))
    preds = []
    for g, raw in gold_as_raw(m):
        p = norm.normalise(raw, ctx)
        preds.append(p.model_copy(update=dict(
            owner_id=g.owner.entity_id, component_id=g.component.entity_id, site_id=g.site.entity_id,
            due_date=g.due_date, due_time=g.due_time, priority=g.priority, category=g.category,
            evidence_quote=m.utterances[g.evidence[0]].text,
        )))
    assert len(match(onto, m, preds)) == len(m.gold_tasks)
    board.add(m, preds)
    assert board.metrics()["dashboard_ready"] == 1.0


# ------------------------------------------------------------ LLM wiring
def test_model_catalogue():
    ids = model_ids()
    assert "gpt-6-sol" in ids and "grok-4.6" in ids
    assert get_model("grok-4.6").api == "chat_json"
    assert get_model("unknown-deployment").deployment == "unknown-deployment"


def test_parse_unwraps_wrapper_key():
    from pydantic import BaseModel

    class R(BaseModel):
        msg: str

    assert _parse(R, '{"response": {"msg": "hi"}}').msg == "hi"


def test_pairing_rules():
    assert _compatible("llm@gpt-6-sol", "llm@gpt-6-sol")
    assert not _compatible("llm@gpt-6-sol", "llm@grok-4.6")
    assert _compatible("rules", "llm@grok-4.6")
    assert _compatible("llm_direct@gpt-5.2", "passthrough")
    assert not _compatible("llm_direct@gpt-5.2", "fuzzy")
    # CPU-heavy GLiNER matchers only pair with local extractors
    assert _compatible("gliner_decide", "gliner_decide")
    assert not _compatible("llm@gpt-5.4", "gliner")


def test_meeting_ids_with_prefix_load(tmp_path, onto):
    from standup_exp.dataset import load_meetings, save_meeting
    from standup_exp.generate.scenario import sample_dataset

    ms = sample_dataset(onto, 3, seed=5, test_fraction=0.34, prefix="a")
    for m in ms:
        render_template(m, onto)
        save_meeting(m, tmp_path)
    loaded = load_meetings(tmp_path)
    assert [m.id for m in loaded] == ["a000", "a001", "a002"]
    assert loaded[-1].split == "test"


def test_meeting_record_perfect(onto, meetings):
    from standup_exp.runner import meeting_record

    m = next(m for m in meetings if m.split == "train")
    norm = get_normaliser("alias", onto, [])
    ctx = NormContext(m.id, m.date, tuple(m.attendees))
    preds = [norm.normalise(raw, ctx).model_copy(update=dict(
        owner_id=g.owner.entity_id, component_id=g.component.entity_id, site_id=g.site.entity_id,
        due_date=g.due_date, due_time=g.due_time, priority=g.priority, category=g.category,
        evidence_quote=m.utterances[g.evidence[0]].text)) for g, raw in gold_as_raw(m)]
    rec = meeting_record(onto, m, preds)
    assert rec["ready"] == len(m.gold_tasks) and not rec["fp"] and not rec["fn"]


def test_on_off_switches_treat_0_as_off(monkeypatch):
    from standup_exp.paths import flag

    for value, on in [("0", False), ("", False), ("false", False), ("no", False), ("1", True), ("true", True), (" Yes ", True)]:
        monkeypatch.setenv("LLM_CACHE_ONLY", value)
        assert flag("LLM_CACHE_ONLY") is on, value
    monkeypatch.delenv("LLM_CACHE_ONLY")
    assert flag("LLM_CACHE_ONLY") is False
