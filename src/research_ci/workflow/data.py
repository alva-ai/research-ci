"""The data layer's commands: admit the source, check the data, and what approving the check shows and pins."""

from dataclasses import asdict
from typing import Any

from research_ci.api.contract import as_use
from research_ci.render.data import data_card, source_card
from research_ci.runner.plugins import code_hash, load
from research_ci.stages.contract import jsonable
from research_ci.stages.data import check_data, current_data_deps
from research_ci.store import vault
from research_ci.store.freeze import drift
from research_ci.store.sources import Sources, source_key
from research_ci.workflow.base import DATA, Approval, IdeaBase, StageFreeze
from research_ci.workflow.machine import Command, State, WorkflowError, starts_new_version
from research_ci.workflow.next import StageStatus

DATA_CARD = "data.json"
DATA_TEXT = "data-card.txt"


class DataCommands(IdeaBase):
    def check_data(self, keep: tuple[str, ...] = ()) -> tuple[dict[str, Any], str]:
        """Admit the contract's data source if it has no admission yet, check the data against the contract, and
        register the result: the data layer ends CHECKED when a person may approve it, else stays CHALLENGED.
        ``keep`` names treatments a person chose not to apply. Treated data of earlier checks that was never frozen
        is removed from the vault."""
        contract = self._approved_contract()
        self._allowed(DATA, Command.REGISTER)
        previous = self._last(Command.REGISTER, DATA)
        checked = check_data(contract, self.root, self.registry, self.data_settings, keep=keep)
        spec = jsonable(checked.spec)
        current = self.status().stage(DATA)
        version = current.version + starts_new_version(current.state, Command.REGISTER)
        text = data_card(spec, self.root.name, version)

        def card() -> dict[str, Any]:
            self.workspace.write_json(DATA_CARD, spec)
            self.workspace.path(DATA_TEXT).write_text(text + "\n")
            return {"spec": spec, "deps": checked.deps}

        self._apply(DATA, Command.REGISTER, effect=card)
        if previous is not None:
            self._drop_unfrozen(as_use(contract.data).name, previous["payload"]["spec"]["panel"], spec["panel"])
        canaries = [c for c in spec["checks"] if c["name"].startswith("canary")]
        self._apply(DATA, Command.CHALLENGE, payload={"tests": spec["tests"], "canaries": canaries})
        status = spec["verdict"]["status"]
        note = "" if status != "blocked" else spec["verdict"]["notes"][0]
        self.check(DATA, status != "blocked", note, {"verdict": status})
        return spec, text

    def source_card(self) -> str:
        record = self._source_record()
        return source_card(record["payload"]["spec"])

    def card(self, stage: str) -> str:
        return self.source_card() if stage == "source" else super().card(stage)

    def approve_source(self, approval: Approval | None = None) -> dict[str, Any]:
        """A person admits the contract's data source, once for this plugin, options and code."""
        who = approval or Approval()
        record = self._source_record()
        spec = record["payload"]["spec"]
        if spec["verdict"]["status"] != "admissible":
            raise WorkflowError(f"{spec['plugin']} cannot be admitted: {'; '.join(spec['verdict']['notes'])}")
        self._shown(who, source_card(spec))
        if record["approved"] is not None:
            return record["approved"]
        return Sources().approve(record["stage"], record["hash"], asdict(who) | {"note": who.describe()})

    def submit_data(self) -> State:
        """Record the frozen data check as passed: the treated data it froze is what feat reads."""
        record = self._checked_data()
        panel = record["payload"]["spec"]["panel"]
        return self.submit(DATA, State.PASSED, extra=_data_deps(record), payload={"panel": panel})

    def _stage_freeze(self, stage: str, current: StageStatus) -> StageFreeze | None:
        if stage != DATA:
            return super()._stage_freeze(stage, current)
        record = self._checked_data()
        spec = record["payload"]["spec"]
        decide = spec["verdict"]["status"] == "decide"
        return StageFreeze(
            card=data_card(spec, self.root.name, current.version),
            deps=_data_deps(record),
            person_only="the data card asks a person to decide; an agent cannot approve it with --auto"
            if decide
            else "",
        )

    def _drop_unfrozen(self, source: str, old: dict[str, Any], new: dict[str, Any]) -> None:
        deps = [r["payload"]["deps"] for r in self.ledger.records() if r["kind"] == "approve"]
        for key, kind in (("hash", "panel"), ("exclusions", "exclusions")):
            frozen = {d.get(kind) for d in deps}
            if old[key] != new[key] and old[key] not in frozen:
                vault.drop(f"{source}.{kind}", old[key])

    def _source_record(self) -> dict[str, Any]:
        use = as_use(self._approved_contract().data)
        key = source_key(use.name, dict(use.options), code_hash(load(use, self.registry)))
        record = Sources().latest(key)
        if record is None:
            raise WorkflowError(f"{use.name} has no admission for its current code; run `rci data check` first")
        return record

    def _checked_data(self) -> dict[str, Any]:
        """The data check as recorded in the ledger. Refuses if it was blocked, or if a plugin's code, the data,
        the source's admission or the treated data changed since."""
        record = self._last(Command.REGISTER, DATA)
        if record is None:
            raise WorkflowError("the data has not been checked; run `rci data check` first")
        spec = record["payload"]["spec"]
        if spec["verdict"]["status"] == "blocked":
            raise WorkflowError(f"the data check is blocked: {'; '.join(spec['verdict']['notes'])}")
        recorded = record["payload"]["deps"]
        moved = drift(recorded, current_data_deps(self._approved_contract(), recorded, spec, self.registry))
        if moved:
            raise WorkflowError(f"changed since the data check: {', '.join(moved)}; run `rci data check` again")
        return record


def _data_deps(record: dict[str, Any]) -> dict[str, str]:
    """What a data freeze pins beyond the usual: the check's ledger record and the treated data it produced."""
    panel = record["payload"]["spec"]["panel"]
    return {"data_check": record["hash"], "panel": panel["hash"], "exclusions": panel["exclusions"]}
