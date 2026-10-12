"""The feat layer's commands: submit and explore features, and what accepting a submission shows and pins."""

from pathlib import Path
from typing import Any

from research_ci.api.contract import Contract
from research_ci.render.common import adder
from research_ci.render.feat import calibration_lines, explore_card, result_card
from research_ci.stages.common import jsonable
from research_ci.stages.feat import run as fs
from research_ci.store.freeze import drift
from research_ci.store.workspace import hash_file
from research_ci.workflow.base import DATA, FEAT, IdeaBase, StageFreeze
from research_ci.workflow.machine import Command, State, Verdict, WorkflowError, starts_new_version
from research_ci.workflow.next import StageStatus

CALIBRATE = "calibrate"


class FeatCommands(IdeaBase):
    def submit_feat(self, path: Path) -> tuple[dict[str, Any], str, State]:
        """Judge one file of features against the floor world and record the verdict, one trial. The data check
        calibrated the decisive effect, and submissions reuse it; when anything it rests on changed since, the first
        submission calibrates it again, and a failed calibration refuses the submission without spending a trial. So do
        repeated scores, and, before anything heavy runs, a features file that is still the example, a ``feat.py``
        without a valid lab eval, and an alternative without a null world; row-by-row code is a warning on the card."""
        current = self._allowed(FEAT, Command.SUBMIT)
        fs.screen(self.root, Path(path), self.registry)
        contract = self._approved_contract()
        fs.ready(self.root, contract, self.registry)
        frozen = self._frozen_feat(contract)
        calibration = self._calibration(frozen)
        version = current.version + starts_new_version(current.state, Command.SUBMIT)
        submissions = self._submissions()
        earlier = [r["payload"]["result"] for r in submissions]
        n = current.trials + 1
        this = {
            e["scores"]["hash"]: e["n"]
            for r, e in zip(submissions, earlier, strict=True)
            if r["payload"]["version"] == version
        }
        before = {
            e["scores"]["hash"]: e["n"]
            for r, e in zip(submissions, earlier, strict=True)
            if r["payload"]["version"] != version
        }
        result = fs.submit(self.root, frozen, Path(path), n, this, self.feat_settings, again=before)
        n = result["n"]
        first = (
            all(e["calibration"]["record"] != calibration["hash"] for e in earlier) and calibration["kind"] == CALIBRATE
        )
        cal = calibration["payload"]
        result["calibration"] = {"record": calibration["hash"], "ran": first, "by": calibration["kind"]}
        result["calibration"] |= {k: cal[k] for k in ("rows", "checks")}
        result["budget"] = current.budget
        result = jsonable(result)
        deps = self.deps(FEAT, frozen.deps | fs.files(self.root) | fs.eval_files(self.root, result))
        deps |= {"features": result["features"]["hash"], "calibration": calibration["hash"]}
        text = result_card(result, self.root.name, version)
        name = f"feat-result-{n}-v{version}" if result["again"] else f"feat-result-{n}"
        self.workspace.write_json(f"{name}.json", result)
        self.workspace.path(f"{name}.txt").write_text(text + "\n")
        verdict = Verdict(result["verdict"]["status"])
        payload = {"result": result, "deps": deps, "again": result["again"]}
        state = self.submit(FEAT, verdict, payload, "; ".join(result["verdict"]["notes"]))
        return result, text, state

    def explore_feat(self, path: Path) -> tuple[dict[str, Any], str]:
        """The checks a submission must pass, run on a file of features without judging it or recording anything."""
        status = self.status()
        if status.stage(DATA).state is not State.PASSED:
            raise WorkflowError("feat reads the frozen data; the data layer has not passed")
        fs.screen(self.root, Path(path), self.registry, fs.EXPLORED)
        frozen = self._frozen_feat(self._approved_contract())
        spec = jsonable(fs.explore(self.root, frozen, Path(path), self.feat_settings))
        return spec, explore_card(spec, self.root.name)

    def results(self) -> list[str]:
        """The result card of every feat submission, oldest first, rendered from the ledger; first the latest
        calibration of the decisive effect when no result card shows it, such as one that failed."""
        submissions = self._submissions()
        out = [result_card(r["payload"]["result"], self.root.name, r["payload"]["version"]) for r in submissions]
        calibration = self._last(CALIBRATE, FEAT)
        if calibration is not None and all(
            r["payload"]["result"]["calibration"]["record"] != calibration["hash"] for r in submissions
        ):
            lines: list[str] = []
            adder(lines)("calibrate", *calibration_lines(calibration["payload"]))
            out.insert(0, "\n".join(lines))
        return out

    def _submissions(self) -> list[dict[str, Any]]:
        return [
            r
            for r in self.ledger.records()
            if r["kind"] == Command.SUBMIT and r["stage"] == FEAT and "result" in r["payload"]
        ]

    def _calibration(self, frozen: fs.Frozen) -> dict[str, Any]:
        """The calibration for what this submission rests on: the one the frozen data check ran, while the contract,
        the frozen data, the engine, the plugins feat runs, and the settings are unchanged; otherwise the first
        submission runs and records one, and later ones reuse it. A failed one refuses the submission until one of them
        changes."""
        key = self._calibration_key(frozen)
        checked = self._frozen_check()
        cal = checked["payload"]["spec"].get("calibration") if checked else None
        if checked is not None and cal is not None and cal["key"] == key and cal["ok"]:
            return {"kind": "check", "hash": checked["hash"], "payload": cal}
        found = None
        for r in self.ledger.records():
            if r["kind"] == CALIBRATE and r["stage"] == FEAT and r["payload"]["key"] == key:
                found = r
        if found is None:
            spec = jsonable(fs.calibration(frozen, self.feat_settings))
            note = {} if spec["ok"] else {"note": f"calibration failed: {'; '.join(spec['notes'])}"}
            found = self._record(FEAT, CALIBRATE, {**spec, "key": key, **note})
        if not found["payload"]["ok"]:
            raise WorkflowError(
                f"the decisive effect failed its calibration in the floor world: {'; '.join(found['payload']['notes'])}"
                "; fix the target or the contract; no submission was spent"
            )
        return found

    def _frozen_check(self) -> dict[str, Any] | None:
        """The record of the data check the data freeze pinned."""
        approval = self._last(Command.APPROVE, DATA)
        pinned = approval["payload"]["deps"].get("data_check") if approval else None
        return next((r for r in self.ledger.records() if r["hash"] == pinned), None) if pinned else None

    def _stage_freeze(self, stage: str, current: StageStatus) -> StageFreeze | None:
        if stage != FEAT:
            return super()._stage_freeze(stage, current)
        record = self._last(Command.SUBMIT, FEAT)
        if record is None or "result" not in record["payload"]:
            return super()._stage_freeze(stage, current)
        result, recorded = record["payload"]["result"], record["payload"]["deps"]
        moved = drift(recorded, self._feat_deps_now(result, recorded))
        if moved:
            raise WorkflowError(f"changed since submission {result['n']}: {', '.join(moved)}; submit again")
        card = result_card(result, self.root.name, record["payload"]["version"])
        return StageFreeze(card=card, deps={**recorded, "feat_submission": record["hash"]})

    def _feat_deps_now(self, result: dict[str, Any], recorded: dict[str, str]) -> dict[str, str]:
        path = Path(result["features"]["path"])
        now = fs.current_deps(self._approved_contract(), self.root, recorded, self.registry)
        now |= {"features": hash_file(path) if path.is_file() else "missing", "calibration": recorded["calibration"]}
        return self.deps(FEAT, now)

    def _frozen_feat(self, contract: Contract) -> fs.Frozen:
        approval = self._last(Command.APPROVE, DATA)
        if approval is None:
            raise WorkflowError("the data is not frozen yet")
        deps = approval["payload"]["deps"]
        return fs.frozen(contract, deps["panel"], deps["exclusions"], self.registry, self.feat_settings)
