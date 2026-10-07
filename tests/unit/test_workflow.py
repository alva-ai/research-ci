from pathlib import Path

import pytest

from research_ci.api.contract import ContractError
from research_ci.render.card import card_id
from research_ci.store.ledger import LedgerError
from research_ci.workflow.commands import Approval, Idea, find_ideas
from research_ci.workflow.machine import State, WorkflowError
from toy import label_contract

S = State


def _contract(template: str = "factor", budget: int = 3, skip: tuple[str, ...] = ()) -> str:
    return label_contract(template, budget, skip)


def _idea(tmp_path: Path, template: str = "factor", budget: int = 3, skip: tuple[str, ...] = ()) -> Idea:
    idea = Idea.init(tmp_path / "tox", template)
    (idea.root / "contract.py").write_text(_contract(template, budget, skip))
    idea.derive_contract()
    idea.approve("contract")
    return idea


def _freeze(idea: Idea, stage: str) -> None:
    idea.register(stage)
    idea.challenge(stage)
    idea.check(stage, passed=True)
    idea.approve(stage)


def _pass(idea: Idea, stage: str) -> None:
    _freeze(idea, stage)
    assert idea.submit(stage, S.PASSED) is S.PASSED


def _pass_data(idea: Idea) -> None:
    idea.check_data()
    idea.approve_source(Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card("source"))))
    idea.check_data()
    idea.approve("data")
    assert idea.submit_data() is S.PASSED


def _who(idea: Idea) -> list[tuple[str, str, str]]:
    return [(a.who, a.stage, a.kind) for a in idea.next_actions()]


def test_init_writes_a_contract_skeleton_and_asks_for_it(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox", "factor")
    assert 'template="factor"' in (idea.root / "contract.py").read_text()
    assert _who(idea) == [("agent", "contract", "register")]
    assert (idea.workspace.root / ".gitignore").read_text() == "lock\n"
    assert [s.stage for s in idea.status().stages] == [
        "contract", "data", "feat", "signal", "algo", "seal", "forward", "capital", "monitor",
    ]  # fmt: skip
    with pytest.raises(WorkflowError, match="already"):
        Idea.init(tmp_path / "tox", "factor")


def test_the_skeleton_contract_is_rejected_until_the_claim_is_written(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox", "factor")
    with pytest.raises(ContractError, match="claim"):
        idea.derive_contract()


def test_contract_must_match_the_template_and_skip_only_optional_stages(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox", "factor")
    (idea.root / "contract.py").write_text(_contract("rule"))
    with pytest.raises(WorkflowError, match="template"):
        idea.derive_contract()
    (idea.root / "contract.py").write_text(_contract("factor", skip=("feat",)))
    with pytest.raises(WorkflowError, match="cannot skip feat"):
        idea.derive_contract()


def test_out_of_order_calls_are_rejected(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox", "factor")
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("contract")
    with pytest.raises(WorkflowError, match="contract is not approved"):
        idea.check_data()
    with pytest.raises(WorkflowError, match="rci data check"):
        idea.register("data")
    idea = _idea(tmp_path / "b")
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("data")
    with pytest.raises(WorkflowError, match="waiting for data"):
        idea.register("feat")
    _pass_data(idea)
    idea.register("feat")
    with pytest.raises(WorkflowError, match="cannot check"):
        idea.check("feat", passed=True)
    with pytest.raises(WorkflowError, match="cannot submit"):
        idea.submit("feat", S.PASSED)
    with pytest.raises(WorkflowError, match="not a stage"):
        idea.register("model")


def test_next_actions_walk_the_inner_loop(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    assert _who(idea) == [("agent", "data", "check")]
    _pass_data(idea)
    assert _who(idea) == [("agent", "feat", "register")]
    idea.register("feat")
    assert _who(idea) == [("challenger", "feat", "challenge")]
    idea.challenge("feat")
    assert _who(idea) == [("agent", "feat", "check")]
    idea.check("feat", passed=False, note="the stand-in survived")
    assert _who(idea) == [("agent", "feat", "check")]
    assert "stand-in survived" in idea.next_actions()[0].reason
    idea.check("feat", passed=True)
    assert _who(idea) == [("human", "feat", "approve")]
    idea.approve("feat")
    assert _who(idea) == [("agent", "feat", "submit")]
    idea.submit("feat", S.PASSED)
    assert _who(idea) == [("agent", "signal", "register")]


def test_the_budget_pauses_a_layer_until_a_person_extends_it(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=2)
    _pass_data(idea)
    _freeze(idea, "feat")
    assert idea.submit("feat", S.FAILED) is S.FAILED
    assert idea.submit("feat", S.FAILED) is S.PAUSED
    assert _who(idea) == [("human", "feat", "extend")]
    with pytest.raises(WorkflowError, match="cannot submit"):
        idea.submit("feat", S.PASSED)
    with pytest.raises(WorkflowError, match="cannot register"):
        idea.register("feat")
    assert idea.extend("feat", 1) is S.FAILED
    assert idea.status().stage("feat").trials_left == 1
    assert idea.submit("feat", S.PASSED) is S.PASSED
    assert idea.ledger.n_trials("feat") == 3


def test_trials_count_across_versions(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=3)
    _pass_data(idea)
    _freeze(idea, "feat")
    idea.submit("feat", S.FAILED)
    _freeze(idea, "feat")
    status = idea.status().stage("feat")
    assert (status.version, status.trials, status.trials_left) == (2, 1, 2)


def test_reopening_a_passed_stage_makes_downstream_stale(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    _pass(idea, "feat")
    _freeze(idea, "signal")
    idea.register("feat")
    signal = idea.status().stage("signal")
    assert (signal.state, idea.status().stage("algo").state) == (S.STALE, S.OPEN)
    with pytest.raises(WorkflowError, match="waiting for feat"):
        idea.register("signal")
    idea.challenge("feat")
    idea.check("feat", passed=True)
    idea.approve("feat")
    idea.submit("feat", S.PASSED)
    assert _who(idea) == [("agent", "signal", "register")]
    assert "feat was reopened" in idea.next_actions()[0].reason
    idea.register("signal")
    assert idea.status().stage("signal").version == 2


def test_checking_the_data_again_reopens_it_and_makes_feat_stale(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    _freeze(idea, "feat")
    idea.check_data()
    assert idea.status().stage("feat").state is S.STALE
    assert idea.status().stage("data").version == 2


def test_changes_to_frozen_dependencies_taint_the_submission(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    _freeze(idea, "feat")
    contract = idea.root / "contract.py"
    contract.write_text(contract.read_text() + "# edited\n")
    assert idea.submit("feat", S.PASSED) is S.TAINTED
    assert "contract" in idea.status().stage("feat").note


def test_judging_dependencies_are_frozen_and_the_research_environment_is_only_recorded(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    _freeze(idea, "feat")
    deps = idea.deps("feat")
    assert {"contract", "engine", "python", "engine_packages", "upstream:data"} <= deps.keys()
    assert not {"packages", "uv.lock"} & deps.keys()
    assert idea.submit("feat", S.FAILED) is S.FAILED
    assert "packages" in list(idea.ledger.records())[-1]["payload"]["environment"]
    assert idea.submit("feat", S.PASSED, extra={"evaluators": "e2"}) is S.TAINTED
    assert "evaluators" in idea.status().stage("feat").note


def test_a_contract_edited_after_derive_cannot_be_approved(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox", "factor")
    contract = idea.root / "contract.py"
    contract.write_text(_contract())
    idea.derive_contract()
    contract.write_text(_contract(budget=5))
    with pytest.raises(WorkflowError, match="derive"):
        idea.approve("contract")


def test_approval_trusts_the_ledger_not_the_copies_under_rci(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox", "factor")
    (idea.root / "contract.py").write_text(_contract(budget=3))
    idea.derive_contract()
    card = idea.workspace.read_json("contract.json")
    idea.workspace.write_json("contract.json", {**card, "contract": {**card["contract"], "budget": 1000}})
    assert idea.preview_approval("contract")["contract"]["budget"] == 3
    idea.approve("contract")
    assert idea.status().stage("feat").budget == 3


def test_reopening_the_contract_makes_everything_after_it_stale(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    idea.register("feat")
    idea.derive_contract()
    states = idea.status().states()
    assert (states["contract"], states["data"], states["feat"], states["signal"]) == (
        S.REGISTERED,
        S.STALE,
        S.STALE,
        S.OPEN,
    )


def test_optional_stages_can_be_skipped_by_the_contract(tmp_path: Path) -> None:
    idea = _idea(tmp_path, template="taker_arb", skip=("signal",))
    assert idea.status().stage("signal").state is S.SKIPPED
    _pass_data(idea)
    assert _who(idea) == [("agent", "algo", "register")]
    assert "upstream:signal" not in idea.deps("algo")
    assert "upstream:data" in idea.deps("algo")


def test_checkpoints_run_once_and_then_ask_for_a_rework(tmp_path: Path) -> None:
    idea = _idea(tmp_path, template="rule")
    _pass_data(idea)
    for stage in ("signal", "algo"):
        _pass(idea, stage)
    assert _who(idea) == [("agent", "seal", "register")]
    idea.register("seal")
    idea.approve("seal")
    assert _who(idea) == [("human", "seal", "submit")]
    assert idea.submit("seal", S.FAILED) is S.FAILED
    assert _who(idea) == [("human", "seal", "reopen")]
    with pytest.raises(WorkflowError):
        idea.submit("seal", S.PASSED)
    idea.register("algo")
    assert idea.status().stage("seal").state is S.STALE


def test_state_is_replayed_from_the_ledger_and_tampering_is_refused(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    assert Idea(idea.root).status() == idea.status()
    text = idea.ledger.path.read_text()
    idea.ledger.path.write_text(text.replace('"verdict":"PASSED"', '"verdict":"FAILED"'))
    with pytest.raises(LedgerError):
        Idea(idea.root)


def test_ideas_are_found_one_or_two_levels_down(tmp_path: Path) -> None:
    Idea.init(tmp_path / "a", "factor")
    Idea.init(tmp_path / "ideas" / "b", "rule")
    assert [p.name for p in find_ideas(tmp_path)] == ["a", "b"]


def test_deleting_the_last_failed_submission_is_caught(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=3)
    _pass_data(idea)
    _freeze(idea, "feat")
    before = idea.ledger.path.read_text()
    idea.submit("feat", S.FAILED)
    idea.submit("feat", S.FAILED)
    lines = idea.ledger.path.read_text().splitlines(keepends=True)
    idea.ledger.path.write_text("".join(lines[:-1]))
    with pytest.raises(LedgerError, match="removed or rewritten"):
        Idea(idea.root)
    idea.ledger.path.write_text(before)
    with pytest.raises(LedgerError, match="removed or rewritten"):
        Idea(idea.root)


def test_an_idea_keeps_its_anchor_when_the_directory_moves(tmp_path: Path, isolated_rci_home: Path) -> None:
    idea = _idea(tmp_path)
    moved = idea.root.rename(tmp_path / "renamed")
    assert Idea(moved).status().stage("contract").state is S.PASSED
    assert len(list((isolated_rci_home / "heads").glob("*.json"))) == 1
