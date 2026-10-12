from pathlib import Path

import pytest

from research_ci.api.contract import ContractError
from research_ci.render.common import card_id
from research_ci.store.ledger import Ledger, LedgerError
from research_ci.store.sources import Sources
from research_ci.store.workspace import Workspace, rci_home
from research_ci.workflow.base import LEDGER_FILE, Approval, find_ideas
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, Verdict, WorkflowError
from toy import lab, label_contract

S = State
V = Verdict


def _idea(tmp_path: Path, budget: int = 3) -> Idea:
    idea = Idea.init(tmp_path / "tox")
    (idea.root / "contract.py").write_text(label_contract(budget))
    idea.derive_contract()
    idea.approve("contract")
    lab(idea.root)
    return idea


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _pass_data(idea: Idea) -> None:
    idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))


def _pass(idea: Idea, stage: str) -> None:
    assert idea.submit(stage, V.PASSED) is S.REVIEW
    idea.approve(stage)


def _who(idea: Idea) -> list[tuple[str, str, str]]:
    return [(a.who, a.stage, a.kind) for a in idea.next_actions()]


def _state(idea: Idea, stage: str) -> State:
    return idea.status().stage(stage).state


def test_init_writes_a_contract_skeleton_and_every_idea_runs_the_one_pipeline(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox")
    assert "template" not in (idea.root / "contract.py").read_text()
    assert _who(idea) == [("agent", "contract", "derive")]
    assert (idea.workspace.root / ".gitignore").read_text() == "lock\n"
    assert [s.stage for s in idea.status().stages] == [
        "contract", "data", "feat", "signal", "algo", "seal", "forward", "capital", "monitor",
    ]  # fmt: skip
    assert next(idea.ledger.records())["payload"] == {"format": 2}
    with pytest.raises(WorkflowError, match="already"):
        Idea.init(tmp_path / "tox")


def test_the_skeleton_contract_is_rejected_until_the_claim_is_written(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox")
    with pytest.raises(ContractError, match="claim"):
        idea.derive_contract()


@pytest.mark.parametrize("field", ["template='factor'", "skip=('signal',)", "deadline='180 days'"])
def test_a_contract_naming_a_template_a_skip_or_a_deadline_does_not_load(tmp_path: Path, field: str) -> None:
    idea = Idea.init(tmp_path / "tox")
    source = label_contract().replace("CONTRACT = Contract(", f"CONTRACT = Contract({field}, ")
    (idea.root / "contract.py").write_text(source)
    with pytest.raises(ContractError, match="unexpected keyword"):
        idea.derive_contract()


def test_a_ledger_of_the_old_lifecycle_is_refused_with_a_way_out(tmp_path: Path) -> None:
    root = tmp_path / "old"
    ledger = Ledger(Workspace(root).path(LEDGER_FILE), anchors=rci_home() / "heads")
    ledger.append("init", "idea", {})
    ledger.append("register", "contract", {"from": "OPEN", "to": "REGISTERED", "version": 1})
    with pytest.raises(LedgerError, match="older Research CI.*fresh directory with `rci init`"):
        Idea(root)


def test_the_contract_is_derived_by_the_agent_and_approved_by_a_person(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox")
    (idea.root / "contract.py").write_text(label_contract())
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("contract")
    idea.derive_contract()
    assert _state(idea, "contract") is S.REVIEW
    assert _who(idea) == [("human", "contract", "approve")]
    idea.derive_contract()
    assert idea.status().stage("contract").version == 1, "deriving again before approval replaces the result"
    idea.approve("contract")
    assert _state(idea, "contract") is S.PASSED
    assert _who(idea) == [("agent", "data", "check")]


def test_out_of_order_calls_are_rejected(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox")
    with pytest.raises(WorkflowError, match="contract is not approved"):
        idea.check_data()
    idea = _idea(tmp_path / "b")
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("data")
    with pytest.raises(WorkflowError, match="waiting for data"):
        idea.submit("feat", V.PASSED)
    _pass_data(idea)
    with pytest.raises(WorkflowError, match="cannot register"):
        idea.register("feat")
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("feat")
    with pytest.raises(WorkflowError, match="not a stage"):
        idea.submit("model", V.PASSED)


def test_the_data_layer_is_checked_then_approved_and_the_first_approval_admits_the_source(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    spec, card = idea.check_data()
    assert spec["source"]["status"] == "waiting" and "approving this card admits it for every idea" in card
    assert _state(idea, "data") is S.REVIEW and _who(idea) == [("human", "data", "approve")]
    key = spec["source"]["key"]
    before = Sources().latest(key)
    assert before is not None and before["approved"] is None
    idea.approve("data", approval=_relayed(idea, "data"))
    assert _state(idea, "data") is S.PASSED and _who(idea) == [("agent", "feat", "submit")]
    after = Sources().latest(key)
    assert after is not None
    approved = after["approved"]
    assert approved["payload"]["approval"]["words"] == "ok"
    assert list(idea.ledger.records())[-1]["payload"]["source_approval"] == approved["hash"]
    other = _idea(tmp_path / "other")
    spec, card = other.check_data()
    assert spec["source"]["status"] == "admitted" and "· admitted" in card
    other.approve("data", approval=Approval(by="agent", channel="auto"))
    assert [r["kind"] for r in Sources().ledger.records()] == ["admit", "approve"]


def test_a_failed_submission_keeps_the_layer_open_and_a_passed_one_goes_up_for_review(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=5)
    _pass_data(idea)
    assert idea.submit("feat", V.FAILED, note="carry: does not beat random") is S.OPEN
    action = idea.next_actions()[0]
    assert (action.who, action.command) == ("agent", "rci feat submit <file>")
    assert action.reason == (
        "submission 1 FAILED: carry: does not beat random; write a new features file in research/ and submit it "
        "(4 of 5 submissions left)"
    )
    assert idea.submit("feat", V.TAINTED) is S.OPEN
    assert idea.submit("feat", V.PASSED) is S.REVIEW
    assert _who(idea) == [("human", "feat", "approve")]
    assert idea.submit("feat", V.FAILED) is S.OPEN, "the agent may go on; a later result replaces the one in review"
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("feat")
    assert idea.submit("feat", V.PASSED) is S.REVIEW
    idea.approve("feat", approval=Approval(by="agent", channel="auto"))
    assert _state(idea, "feat") is S.PASSED and _who(idea) == [("agent", "signal", "wait")]


def test_a_person_accepts_only_a_passed_latest_submission(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    idea.submit("feat", V.PASSED)
    idea.ledger.append("submit", "feat", {"verdict": "FAILED", "from": "REVIEW", "to": "REVIEW", "version": 1})
    with pytest.raises(WorkflowError, match="latest submission of feat did not pass"):
        idea.approve("feat")


def test_the_budget_pauses_a_layer_until_a_person_extends_it(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=2)
    _pass_data(idea)
    assert idea.submit("feat", V.FAILED) is S.OPEN
    assert idea.submit("feat", V.FAILED) is S.PAUSED
    assert _who(idea) == [("human", "feat", "extend")]
    with pytest.raises(WorkflowError, match="cannot submit"):
        idea.submit("feat", V.PASSED)
    assert idea.extend("feat", 1) is S.OPEN
    assert idea.status().stage("feat").trials_left == 1
    assert idea.submit("feat", V.PASSED) is S.REVIEW, "a pass on the last trial goes up for review, not PAUSED"
    with pytest.raises(WorkflowError, match="all 3 submissions of feat are used"):
        idea.submit("feat", V.PASSED)
    assert idea.ledger.n_trials("feat") == 3


def test_trials_count_across_versions(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=3)
    _pass_data(idea)
    idea.submit("feat", V.FAILED)
    _pass_data(idea)
    assert _state(idea, "feat") is S.STALE
    assert idea.submit("feat", V.FAILED) is S.OPEN
    status = idea.status().stage("feat")
    assert (status.version, status.trials, status.trials_left) == (2, 2, 1)


def test_reopening_a_passed_stage_makes_started_stages_after_it_stale(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=5)
    _pass_data(idea)
    _pass(idea, "feat")
    idea.submit("signal", V.FAILED)
    assert _state(idea, "signal") is S.OPEN
    idea.submit("feat", V.FAILED)
    states = idea.status().states()
    assert (states["feat"], states["signal"], states["algo"]) == (S.OPEN, S.STALE, S.OPEN)
    assert idea.status().stage("feat").version == 2
    with pytest.raises(WorkflowError, match="waiting for feat"):
        idea.submit("signal", V.PASSED)
    _pass(idea, "feat")
    action = idea.next_actions()[0]
    assert (action.who, action.stage, action.kind) == ("agent", "signal", "submit")
    assert action.reason.startswith("feat was reopened; submit again to start v2")
    idea.submit("signal", V.PASSED)
    assert idea.status().stage("signal").version == 2


def test_deriving_a_passed_contract_again_makes_every_started_stage_stale(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    idea.submit("feat", V.FAILED)
    _, card = idea.derive_contract()
    states = idea.status().states()
    assert (states["contract"], states["data"], states["feat"], states["signal"]) == (
        S.REVIEW,
        S.STALE,
        S.STALE,
        S.OPEN,
    )
    assert card.startswith("ContractCard · tox · v2") and idea.status().stage("contract").version == 2
    assert _who(idea) == [("human", "contract", "approve")]
    idea.approve("contract")
    action = idea.next_actions()[0]
    assert (action.stage, action.command) == ("data", "rci data check")
    assert action.reason == "contract was reopened; check again to start v2"
    _pass_data(idea)
    assert idea.submit("feat", V.FAILED) is S.OPEN
    feat = idea.status().stage("feat")
    assert (feat.version, feat.trials) == (2, 2)


def test_judging_dependencies_are_frozen_and_the_research_environment_is_only_recorded(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    deps = idea.deps("feat")
    assert {"contract", "engine", "python", "engine_packages", "upstream:data"} <= deps.keys()
    assert not {"packages", "uv.lock"} & deps.keys()
    idea.submit("feat", V.FAILED)
    assert "packages" in list(idea.ledger.records())[-1]["payload"]["environment"]


def test_a_contract_edited_after_derive_cannot_be_approved(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox")
    contract = idea.root / "contract.py"
    contract.write_text(label_contract())
    idea.derive_contract()
    contract.write_text(label_contract(budget=5))
    with pytest.raises(WorkflowError, match="derive"):
        idea.approve("contract")


def test_approval_trusts_the_ledger_not_the_copies_under_rci(tmp_path: Path) -> None:
    idea = Idea.init(tmp_path / "tox")
    (idea.root / "contract.py").write_text(label_contract(budget=3))
    idea.derive_contract()
    card = idea.workspace.read_json("contract.json")
    idea.workspace.write_json("contract.json", {**card, "contract": {**card["contract"], "budget": 1000}})
    assert idea.preview_approval("contract")["contract"]["budget"] == 3
    idea.approve("contract")
    assert idea.status().stage("feat").budget == 3


def test_checkpoints_register_a_plan_a_person_approves_run_once_and_then_ask_for_a_rework(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    for stage in ("feat", "signal", "algo"):
        _pass(idea, stage)
    assert {"upstream:data", "upstream:feat", "upstream:signal"} <= idea.deps("algo").keys()
    assert _who(idea) == [("agent", "seal", "register")]
    assert idea.register("seal") is S.REVIEW
    assert _who(idea) == [("human", "seal", "approve")]
    assert idea.approve("seal") and _state(idea, "seal") is S.OPEN
    assert _who(idea) == [("human", "seal", "run")]
    with pytest.raises(WorkflowError, match="its plan is approved"):
        idea.register("seal")
    assert idea.run("seal", V.FAILED) is S.REVIEW
    assert _who(idea) == [("human", "seal", "reopen")]
    for again in (lambda: idea.run("seal", V.PASSED), lambda: idea.approve("seal"), lambda: idea.register("seal")):
        with pytest.raises(WorkflowError):
            again()
    idea.submit("algo", V.PASSED)
    assert _state(idea, "seal") is S.STALE
    assert _who(idea) == [("human", "algo", "approve")]


def test_a_checkpoint_run_is_tainted_when_its_frozen_plan_moved(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    for stage in ("feat", "signal", "algo"):
        _pass(idea, stage)
    idea.register("seal")
    idea.approve("seal", extra={"plan": "p1"})
    assert idea.run("seal", V.PASSED, extra={"plan": "p2"}) is S.REVIEW
    assert "TAINTED" in idea.status().stage("seal").note and "plan" in idea.status().stage("seal").note


def test_state_is_replayed_from_the_ledger_and_tampering_is_refused(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _pass_data(idea)
    assert Idea(idea.root).status() == idea.status()
    text = idea.ledger.path.read_text()
    idea.ledger.path.write_text(text.replace('"to":"PASSED"', '"to":"OPEN"'))
    with pytest.raises(LedgerError):
        Idea(idea.root)


def test_ideas_are_found_one_or_two_levels_down(tmp_path: Path) -> None:
    Idea.init(tmp_path / "a")
    Idea.init(tmp_path / "ideas" / "b")
    assert [p.name for p in find_ideas(tmp_path)] == ["a", "b"]


def test_deleting_the_last_failed_submission_is_caught(tmp_path: Path) -> None:
    idea = _idea(tmp_path, budget=3)
    _pass_data(idea)
    before = idea.ledger.path.read_text()
    idea.submit("feat", V.FAILED)
    idea.submit("feat", V.FAILED)
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
