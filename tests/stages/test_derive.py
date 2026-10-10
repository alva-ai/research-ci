from pathlib import Path

import pytest

from research_ci.runner import plugins
from research_ci.runner.plugins import PluginError
from research_ci.stages import contract as stage
from research_ci.stages.contract import Settings
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import WorkflowError
from toy import REGISTRY, FlippedFlow, consumer_contract, label_contract

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1, consumer_cuts=1, pilot=((1, 3), (2, 2)))


def _idea(tmp_path: Path, source: str) -> Idea:
    idea = Idea.init(tmp_path / "idea", "factor", settings=FAST)
    (idea.root / "contract.py").write_text(source)
    (idea.root / "strategy.toml").write_text("[s]\nk = 2\nwindow = 24\n")
    (idea.root / "live.toml").write_text("[s]\nk = 2\nwindow = 12\nretries = 3\n")
    (idea.root / "strategy.py").write_text("K = 2\n")
    return idea


def test_a_label_contract_reports_power_per_horizon_without_revealing_the_target(tmp_path: Path) -> None:
    idea = _idea(tmp_path, label_contract())
    idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    rows = spec["decisive"][0]["horizons"]
    assert [r["horizon"] for r in rows] == ["8h", "24h"]
    assert {"mde", "se", "forward_days", "blocks"} <= rows[0].keys()
    assert not {"mean", "value", "estimate", "level"} & rows[0].keys()
    checks = {c["name"]: c["status"] for c in spec["checks"]}
    for name in ("universe point-in-time", "label carry: worked examples", "label carry: examples bite"):
        assert checks[name] == "pass", name
    assert checks["label carry: no data after t + h"] == "pass"
    assert spec["decisive"][0]["target"] == "ref.mean"
    assert spec["history"] == "can confirm"


def test_a_consumer_contract_tests_the_consumer_and_measures_how_noise_grows(tmp_path: Path) -> None:
    idea = _idea(tmp_path, consumer_contract())
    _, card = idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    checks = {c["name"]: c["status"] for c in spec["checks"]}
    for name in (
        "consumer deterministic",
        "net adds up",
        "fees match the cost model",
        "consumer matches labels (carry)",
        "consumer point-in-time",
        "consumer causal",
        "scores pass through",
    ):
        assert checks[name] == "pass", name
    pilot = spec["decisive"][0]["pilot"]
    assert pilot["design"] == "branch" and pilot["branches"] >= 5
    assert 0 < pilot["overlap"] <= 1
    assert spec["decisive"][0]["power"]["history_mde"] > 0
    assert spec["history"] == "explore-only"
    assert spec["baseline"]["live_config"] == {"common": 2, "differ": ["window"], "only_live": 1}
    assert spec["forward"] == {"source": "toy.live", "installed": False}
    assert "NOT INSTALLED" in card and "design: branch" in card


def test_a_consumer_that_fails_a_protocol_test_cannot_be_approved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.carry": FlippedFlow})
    idea = _idea(tmp_path, consumer_contract())
    _, card = idea.derive_contract()
    assert "FAIL consumer matches labels (carry)" in card
    assert "BLOCKED" in card
    with pytest.raises(WorkflowError, match="protocol tests"):
        idea.approve("contract")


def test_changes_to_plugin_code_or_baseline_files_after_derive_block_approval(tmp_path: Path, monkeypatch) -> None:
    idea = _idea(tmp_path, label_contract())
    idea.derive_contract()
    with monkeypatch.context() as m:
        m.setattr(stage, "code_hash", lambda plugin: "changed")
        with pytest.raises(WorkflowError, match="plugin:toy"):
            idea.approve("contract")
    b = _idea(tmp_path / "b", consumer_contract())
    b.derive_contract()
    (b.root / "strategy.toml").write_text("[s]\nk = 3\n")
    with pytest.raises(WorkflowError, match="baseline:config"):
        b.approve("contract")


def test_an_unknown_plugin_is_refused(tmp_path: Path) -> None:
    idea = _idea(tmp_path, label_contract().replace("'toy.pairs'", "'toy.nowhere'"))
    with pytest.raises(PluginError, match="toy.nowhere"):
        idea.derive_contract()
