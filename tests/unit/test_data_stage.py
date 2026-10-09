from pathlib import Path

import pytest

from research_ci.render.common import card_id
from research_ci.runner import plugins
from research_ci.stages import data as data_stage
from research_ci.stages.contract import Settings
from research_ci.stages.data import DataSettings
from research_ci.store import vault
from research_ci.workflow.base import Approval
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import REGISTRY, ToyHindsight, ToySwapped, label_contract

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1, consumer_cuts=1, pilot=((1, 3), (2, 2)))
RELAYED = Approval(by="human", channel="relayed", words="ok")

SWAP_TEST = '''
import polars as pl
from research_ci.api import DataTest


def drifting_legs(fields):
    """Venues of one symbol whose hourly returns do not move with the symbol's other venues."""
    close = fields.filter(pl.col("field") == "close").sort("entity", "event_time")
    r = close.with_columns(r=(pl.col("value") / pl.col("value").shift(1).over("entity")).log())
    r = r.with_columns(symbol=pl.col("entity").str.split(":").list.last())
    mean = r.group_by("symbol", "event_time").agg(pl.col("r").mean().alias("m"))
    r = r.join(mean, on=["symbol", "event_time"])
    corr = r.group_by("entity").agg(pl.corr("r", "m").alias("c"))
    bad = corr.filter(pl.col("c") < 0.8)["entity"]
    return close.filter(pl.col("entity").is_in(bad.implode())).select("entity", "event_time")


def nothing(fields):
    return fields.head(0).select("entity", "event_time")


TESTS = [
    DataTest("drifting legs", drifting_legs, covers=[("swap", "close")], on_fail="drop_entities"),
    {lazy}
]
'''
LAZY = 'DataTest("lazy", nothing, covers=[("swap", "close")], on_fail="mask"),'


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _idea(tmp_path: Path, settings: DataSettings | None = None) -> Idea:
    idea = Idea.init(tmp_path / "idea", "factor", settings=FAST, data_settings=settings)
    (idea.root / "contract.py").write_text(label_contract())
    idea.derive_contract()
    idea.approve("contract")
    return idea


def test_the_data_layer_runs_from_admission_to_a_passed_verdict(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    spec, card = idea.check_data()
    assert spec["source"]["status"] == "waiting" and spec["verdict"]["status"] == "blocked"
    assert "WAITING for a person" in card
    assert [(a.who, a.command) for a in idea.next_actions()] == [("human", "rci source approve")]
    assert idea.status().stage("data").state is State.CHALLENGED
    idea.approve_source(_relayed(idea, "source"))
    spec, card = idea.check_data()
    assert spec["verdict"]["status"] == "ready" and spec["decide"] == []
    assert {c["name"]: c["status"] for c in spec["checks"]}["hedged legs are one asset"] == "pass"
    assert "READY" in card
    assert idea.status().stage("data").state is State.CHECKED
    idea.approve("data", approval=_relayed(idea, "data"))
    assert idea.submit_data() is State.PASSED
    assert idea.status().stage("data").budget is None and idea.ledger.n_trials("data") == 1
    assert idea.next_actions()[0].stage == "feat"


def test_a_source_that_fails_admission_blocks_the_data_and_cannot_be_admitted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyHindsight})
    idea = _idea(tmp_path)
    spec, _ = idea.check_data()
    assert spec["source"]["status"] == "not admitted"
    assert set(spec["source"]["failed"]) == {"replay as of past times", "independent source agrees"}
    with pytest.raises(WorkflowError, match="cannot be admitted"):
        idea.approve_source(RELAYED)
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("data", approval=RELAYED)


def test_changing_a_plugin_after_the_check_blocks_approval(tmp_path: Path, monkeypatch) -> None:
    idea = _idea(tmp_path)
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    idea.check_data()
    with monkeypatch.context() as m:
        m.setattr(data_stage, "code_hash", lambda plugin: "changed")
        with pytest.raises(WorkflowError, match="changed since the data check"):
            idea.approve("data", approval=Approval())


def test_a_swapped_leg_is_filtered_and_its_influence_measured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path)
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    spec, card = idea.check_data()
    guard = next(t for t in spec["treatments"] if t["name"] == "hedge guard at each decision time")
    assert guard["rows"] > 0 and set(guard["influence"]) == {"carry 8h", "carry 24h"}
    assert all(v["mde"] >= 0 for v in guard["influence"].values())
    assert all(p["mde"] < p["mde_before"] for p in spec["power"])
    assert "hedge guard" in card


def test_a_treatment_that_matters_needs_a_person(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path, DataSettings(materiality=1e-6))
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    spec, _ = idea.check_data()
    assert spec["verdict"]["status"] == "decide" and spec["decide"]
    with pytest.raises(WorkflowError, match="asks a person"):
        idea.approve("data", approval=Approval(by="agent", channel="auto"))
    with pytest.raises(WorkflowError, match="must quote the card"):
        idea.approve("data", approval=RELAYED)
    with pytest.raises(WorkflowError, match="must quote the card"):
        idea.approve("data", approval=Approval(by="human", channel="relayed", words="ok", card="00000000"))
    idea.approve("data", approval=_relayed(idea, "data"))


def test_a_person_can_keep_a_treatment_from_being_applied(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path)
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    applied, _ = idea.check_data()
    kept, card = idea.check_data(keep=("hedge guard at each decision time",))
    assert kept["kept"] == ["hedge guard at each decision time"]
    assert next(t for t in kept["treatments"] if t["name"].startswith("hedge"))["kept"] is True
    assert kept["panel"]["exclusions"] != applied["panel"]["exclusions"]
    assert all(p["mde"] == p["mde_before"] for p in kept["power"])
    assert "kept, not applied" in card
    with pytest.raises(data_stage.DataError, match="no treatment named"):
        idea.check_data(keep=("nothing like this",))


def test_treated_data_of_a_superseded_check_leaves_the_vault_unless_frozen(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path)
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    first, _ = idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))
    second, _ = idea.check_data(keep=("hedge guard at each decision time",))
    third, _ = idea.check_data()
    assert vault.has("toy.data.exclusions", first["panel"]["exclusions"])
    assert not vault.has("toy.data.exclusions", second["panel"]["exclusions"])
    assert third["panel"]["exclusions"] == first["panel"]["exclusions"]


@pytest.mark.parametrize(("lazy", "status"), [(False, "holds"), (True, "returned")])
def test_an_idea_test_must_catch_the_pits_it_claims(tmp_path: Path, monkeypatch, lazy: bool, status: str) -> None:
    idea = _idea(tmp_path, DataSettings(materiality=0.05, seeds=1))
    (idea.root / "data.py").write_text(SWAP_TEST.replace("{lazy}", LAZY if lazy else ""))
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    spec, card = idea.check_data()
    tests = {t["name"]: t for t in spec["tests"]}
    assert tests["drifting legs"]["status"] == "holds"
    assert [r["status"] for r in tests["drifting legs"]["calibration"]] == ["caught"]
    if lazy:
        assert tests["lazy"]["status"] == status
        assert spec["verdict"]["status"] == "blocked"
        assert "tests returned: lazy" in card
    else:
        assert spec["verdict"]["status"] == "ready"
