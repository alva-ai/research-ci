"""The feat layer's commands: register a method, submit and explore features, and what approving a method shows
and pins."""

from pathlib import Path
from typing import Any

from research_ci.api.contract import Contract
from research_ci.render.feat import explore_card, method_card, result_card
from research_ci.stages import feat as fs
from research_ci.stages.contract import jsonable
from research_ci.store.freeze import drift
from research_ci.workflow.base import DATA, FEAT, IdeaBase, StageFreeze
from research_ci.workflow.machine import Command, State, WorkflowError, starts_new_version
from research_ci.workflow.next import StageStatus

FEAT_CARD = "feat.json"
FEAT_TEXT = "feat-card.txt"


class FeatCommands(IdeaBase):
    def register_feat(self) -> tuple[dict[str, Any], str]:
        """Fix how feat is judged: load the criteria and the null worlds, check every alternative is kept by a world,
        calibrate every criterion in every world, and register the result. The layer ends CHECKED when a person may
        approve the method, else stays CHALLENGED."""
        contract = self._approved_contract()
        self._allowed(FEAT, Command.REGISTER)
        frozen = self._frozen_feat(contract)
        spec = jsonable(fs.method(self.root, frozen, self.feat_settings))
        deps = fs.method_deps(self.root, frozen)
        current = self.status().stage(FEAT)
        version = current.version + starts_new_version(current.state, Command.REGISTER)
        text = method_card(spec, self.root.name, version, current.budget)

        def card() -> dict[str, Any]:
            self.workspace.write_json(FEAT_CARD, spec)
            self.workspace.path(FEAT_TEXT).write_text(text + "\n")
            return {"spec": spec, "deps": deps}

        self._apply(FEAT, Command.REGISTER, effect=card)
        self._apply(FEAT, Command.CHALLENGE, payload={"worlds": spec["worlds"], "nulls_file": deps["nulls_file"]})
        status = spec["verdict"]["status"]
        note = "" if status != "blocked" else spec["verdict"]["notes"][0]
        self.check(FEAT, status != "blocked", note, {"verdict": status})
        return spec, text

    def submit_feat(self, path: Path) -> tuple[dict[str, Any], str, State]:
        """Judge one file of features with the approved method and record the verdict. Refuses, without spending a
        submission, when anything the method was approved with has changed, or the scores repeat an earlier one."""
        current = self._allowed(FEAT, Command.SUBMIT, State.FAILED)
        record = self._feat_method()
        approval = self._last(Command.APPROVE, FEAT)
        if approval is None or approval["payload"]["deps"].get("feat_method") != record["hash"]:
            raise WorkflowError("the approved feat method is not the registered one; register and approve it again")
        extra = _feat_deps(record)
        moved = drift(approval["payload"]["deps"], self.deps(FEAT, extra))
        if moved:
            raise WorkflowError(f"changed since the method was approved: {', '.join(moved)}; register it again")
        contract = self._approved_contract()
        frozen = self._frozen_feat(contract)
        earlier = {
            r["payload"]["result"]["scores"]["hash"]: r["payload"]["result"]["n"]
            for r in self.ledger.records()
            if r["kind"] == Command.SUBMIT and r["stage"] == FEAT and "result" in r["payload"]
        }
        n = current.trials + 1
        spec = record["payload"]["spec"]
        result = jsonable(fs.submit(self.root, frozen, spec, Path(path), n, earlier, self.feat_settings))
        text = result_card(result, self.root.name, current.version, current.budget)
        self.workspace.write_json(f"feat-result-{n}.json", result)
        self.workspace.path(f"feat-result-{n}.txt").write_text(text + "\n")
        state = self.submit(FEAT, State(result["verdict"]["status"]), extra=extra, payload={"result": result})
        return result, text, state

    def explore_feat(self, path: Path) -> tuple[dict[str, Any], str]:
        """The checks a submission must pass, run on a file of features without judging it or recording anything."""
        status = self.status()
        if status.stage(DATA).state is not State.PASSED:
            raise WorkflowError("feat reads the frozen data; the data layer has not passed")
        spec = jsonable(fs.explore(self._frozen_feat(self._approved_contract()), Path(path), self.feat_settings))
        return spec, explore_card(spec, self.root.name)

    def results(self) -> list[str]:
        """The result card of every feat submission, oldest first."""
        out = []
        for r in self.ledger.records():
            if r["kind"] == Command.SUBMIT and r["stage"] == FEAT and "result" in r["payload"]:
                path = self.workspace.path(f"feat-result-{r['payload']['result']['n']}.txt")
                out.append(path.read_text().rstrip() if path.exists() else "")
        return out

    def _stage_freeze(self, stage: str, current: StageStatus) -> StageFreeze | None:
        if stage != FEAT or not self._feat_registered():
            return super()._stage_freeze(stage, current)
        record = self._feat_method()
        card = method_card(record["payload"]["spec"], self.root.name, current.version, current.budget)
        return StageFreeze(card=card, deps=_feat_deps(record))

    def _frozen_feat(self, contract: Contract) -> fs.Frozen:
        approval = self._last(Command.APPROVE, DATA)
        if approval is None:
            raise WorkflowError("the data is not frozen yet")
        deps = approval["payload"]["deps"]
        return fs.frozen(contract, deps["panel"], deps["exclusions"], self.registry, self.feat_settings)

    def _feat_registered(self) -> bool:
        record = self._last(Command.REGISTER, FEAT)
        return record is not None and "spec" in record["payload"]

    def _feat_method(self) -> dict[str, Any]:
        """The feat method as recorded in the ledger. Refuses if it was blocked, or if the criteria, the nulls or the
        code of a plugin it runs changed since."""
        record = self._last(Command.REGISTER, FEAT)
        if record is None or "spec" not in record["payload"]:
            raise WorkflowError("the feat method has not been registered; run `rci feat register` first")
        spec = record["payload"]["spec"]
        if spec["verdict"]["status"] == "blocked":
            raise WorkflowError(f"the feat method is blocked: {'; '.join(spec['verdict']['notes'])}")
        recorded = record["payload"]["deps"]
        now = fs.current_method_deps(self._approved_contract(), self.root, recorded, self.registry)
        moved = drift(recorded, now)
        if moved:
            raise WorkflowError(f"changed since `rci feat register`: {', '.join(moved)}; register it again")
        return record


def _feat_deps(record: dict[str, Any]) -> dict[str, str]:
    """What a feat freeze pins beyond the usual: the method's ledger record, the criteria and nulls files, and the
    code of every plugin feat runs."""
    return {"feat_method": record["hash"], **record["payload"]["deps"]}
