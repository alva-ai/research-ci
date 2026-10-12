import math
from pathlib import Path

import pytest

from research_ci.runner import plugins
from research_ci.runner.plugins import PluginError
from research_ci.stages.contract import run as stage
from research_ci.stages.contract.run import Settings
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import REGISTRY, NeverRun, ToyRelaunched, ToyZeros, carry_contract, strategy_contract

FAST = Settings(workers=2, universe_cuts=2, outcome_cuts=1)
TOP = ", 'top': 0.34"


def _idea(tmp_path: Path, source: str) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST)
    (idea.root / "contract.py").write_text(source)
    return idea


def test_a_carry_contract_reports_power_per_horizon_without_revealing_the_target(tmp_path: Path) -> None:
    idea = _idea(tmp_path, carry_contract())
    _, card = idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    rows = spec["decisive"][0]["horizons"]
    assert [r["horizon"] for r in rows] == ["8h", "24h"]
    assert set(rows[0]) == {"horizon", "decisions", "blocks", "se", "mde", "concentration"}
    assert spec["verdict"]["status"] == "decidable"
    assert "ref.mean(cluster=symbol, top=0.34): beats random significantly" in card and "DECIDABLE" in card
    assert "outcome    ref.carry(flow=funding_rate) · H 8h 24h" in card
    assert "data       toy.data " in card and " rows in candles, funding · 9 insts" in card
    assert "FAIL if" not in card and "new data" not in card
    checks = {c["name"]: c["status"] for c in spec["checks"]}
    for name in ("universe point-in-time", "outcome ref.carry: worked examples", "outcome ref.carry: examples bite"):
        assert checks[name] == "pass", name
    assert checks["outcome ref.carry: no data after t + h"] == "pass"
    assert spec["decisive"][0]["target"] == "ref.mean"
    assert spec["history"] == "can confirm"


def test_a_contract_naming_a_consumer_takes_its_decision_times_and_runs_no_backtest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.carry": NeverRun})
    idea = _idea(tmp_path, strategy_contract())
    _, card = idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    assert spec["decisions"]["from"] == "consumer toy.carry"
    assert [d["kind"] for d in spec["decisive"]] == ["outcome"]
    assert not [c["name"] for c in spec["checks"] if c["name"].startswith("consumer")]
    assert spec["alternatives"]["floor"] == ["chance", "timing"]
    assert spec["history"] == "explore-only"
    assert spec["forward"] == {"source": "toy.live", "installed": False}
    assert "NOT INSTALLED" in card and "backtest" not in card


def test_a_target_that_ignores_the_scores_blocks_the_contract(tmp_path: Path) -> None:
    ignores = carry_contract().replace(TOP, "")
    assert ignores != carry_contract()
    idea = _idea(tmp_path, ignores)
    _, card = idea.derive_contract()
    checks = {c["name"]: c for c in idea.workspace.read_json("contract.json")["spec"]["checks"]}
    assert checks["target reads the score"]["status"] == "fail"
    assert "top" in checks["target reads the score"]["detail"]
    assert "FAIL target reads the score" in card and "BLOCKED" in card
    contract = idea.status().stage("contract")
    assert contract.state is State.OPEN and contract.note.startswith("blocked: target reads the score failed: ref.mean")
    why = "target reads the score failed: ref.mean gives the same number for any scores"
    assert why in card and "fix the contract, or the plugin it names, and derive again" in card
    assert [(a.who, a.command) for a in idea.next_actions()] == [("agent", "rci contract derive")]
    with pytest.raises(WorkflowError, match="cannot approve a contract stage that is OPEN"):
        idea.approve("contract")
    reads = _idea(tmp_path / "reads", carry_contract())
    _, card = reads.derive_contract()
    assert "ok   target reads the score: ref.mean moves with the scores" in card, card


def test_changes_to_plugin_code_after_derive_block_approval(tmp_path: Path, monkeypatch) -> None:
    idea = _idea(tmp_path, carry_contract())
    idea.derive_contract()
    with monkeypatch.context() as m:
        m.setattr(stage, "code_hash", lambda plugin: "changed")
        with pytest.raises(WorkflowError, match="plugin:toy"):
            idea.approve("contract")


def test_a_contract_whose_noise_cannot_be_measured_is_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stage.pw, "block_se", lambda values, h_ns: math.nan)
    idea = _idea(tmp_path, carry_contract())
    _, card = idea.derive_contract()
    verdict = idea.workspace.read_json("contract.json")["spec"]["verdict"]
    assert verdict["status"] == "blocked" and "BLOCKED" in card
    why = "ref.mean: too few blocks on every horizon to measure the noise; give the contract a longer explore period"
    assert verdict["notes"][0].startswith(why) and why in card
    assert idea.status().stage("contract").note.startswith(f"blocked: {why}")
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("contract")


def test_an_unknown_plugin_is_refused(tmp_path: Path) -> None:
    idea = _idea(tmp_path, carry_contract().replace("'toy.pairs'", "'toy.nowhere'"))
    with pytest.raises(PluginError, match="toy.nowhere"):
        idea.derive_contract()


def test_the_card_shows_the_detectable_gain_in_the_target_s_units_and_rolled_over_a_year(tmp_path: Path) -> None:
    idea = _idea(tmp_path, carry_contract())
    _, card = idea.derive_contract()
    d = idea.workspace.read_json("contract.json")["spec"]["decisive"][0]
    assert (d["per"], d["rolls"]) == ("entry", True)
    row = next(line for line in card.splitlines() if line.strip().startswith("8h "))
    yearly = d["horizons"][0]["mde"] * 365 * 24 / 8
    assert "MDE bp/entry  if rolled" in card
    assert f"{d['horizons'][0]['mde'] * 1e4:.1f}" in row and f"≈ {yearly:.1%}/yr" in row, card


def test_the_card_warns_when_one_cluster_carries_the_noise(tmp_path: Path, monkeypatch) -> None:
    clean = _idea(tmp_path, carry_contract())
    clean.derive_contract()
    (_, share), *_ = clean.workspace.read_json("contract.json")["spec"]["decisive"][0]["horizons"][1]["concentration"]
    assert share < 0.9
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyRelaunched})
    idea = _idea(tmp_path / "relaunched", carry_contract())
    _, card = idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    (name, share), *_ = spec["decisive"][0]["horizons"][1]["concentration"]
    assert name == "Z" and share > 0.9
    warning = next(w for w in spec["warnings"] if "of the noise the MDE rests on" in w)
    assert warning.startswith("ref.mean: symbol Z carries") and "rci data check" in warning
    assert spec["verdict"]["status"] == "decidable"


def test_a_number_that_is_not_finite_somewhere_is_left_out_of_the_power_not_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyZeros})
    idea = _idea(tmp_path, carry_contract())
    _, card = idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    assert spec["verdict"]["status"] == "decidable"
    assert all(math.isfinite(r["mde"]) for r in spec["decisive"][0]["horizons"])
    assert any(w.startswith("ref.mean: the number is not finite at some decision times") for w in spec["warnings"])


def test_an_outcome_option_that_names_no_column_of_the_sources_lists_the_columns(tmp_path: Path) -> None:
    """A misspelled flow still gives rows, with no flow in them: a warning. A misspelled price gives no rows: the
    contract is blocked."""
    columns = "their columns are close, funding_interval_hours, funding_rate"
    idea = _idea(tmp_path, carry_contract().replace("{'flow': 'funding_rate'}", "{'flow': 'funding_rat'}"))
    idea.derive_contract()
    checks = {c["name"]: c for c in idea.workspace.read_json("contract.json")["spec"]["checks"]}
    assert checks["outcome ref.carry: columns"]["status"] == "warn"
    assert (
        checks["outcome ref.carry: columns"]["detail"]
        == f"ref.carry option flow='funding_rat' is not a column of the sources; {columns}"
    )
    other = Idea.init(tmp_path / "other", settings=FAST)
    (other.root / "contract.py").write_text(carry_contract().replace("{'flow'", "{'price': 'closee', 'flow'"))
    _, card = other.derive_contract()
    assert (
        f"FAIL outcome ref.carry: columns: ref.carry option price='closee' is not a column of the sources; {columns}"
        in card
    )
    assert "the outcome gives no rows at 8h" in card and other.status().stage("contract").state is State.OPEN


def test_without_horizons_the_contract_takes_the_decision_period_times_1_4_and_16(tmp_path: Path) -> None:
    idea = _idea(tmp_path, carry_contract().replace(", horizons=('8h', '24h')", ""))
    _, card = idea.derive_contract()
    spec = idea.workspace.read_json("contract.json")["spec"]
    assert spec["outcome"]["horizons"] == ["1h", "4h", "16h"] and spec["outcome"]["default"]
    assert [r["horizon"] for r in spec["decisive"][0]["horizons"]] == ["1h", "4h", "16h"]
    assert "H 1h 4h 16h (none named: the decision period times 1, 4 and 16)" in card
