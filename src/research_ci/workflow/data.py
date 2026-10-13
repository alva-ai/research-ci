"""The data layer's commands: check the data, and what approving the check shows, pins, and records about the data
source."""

from dataclasses import asdict
from typing import Any

from research_ci.api.contract import as_use
from research_ci.render.data import data_card
from research_ci.stages.common import jsonable
from research_ci.stages.data.run import PRODUCTS, check_data, current_data_deps
from research_ci.store import vault
from research_ci.store.freeze import drift
from research_ci.store.sources import Sources
from research_ci.workflow.base import DATA, Approval, IdeaBase, StageFreeze
from research_ci.workflow.machine import Command, WorkflowError, starts_new_version
from research_ci.workflow.next import StageStatus

DATA_CARD = "data.json"
DATA_TEXT = "data-card.txt"


class DataCommands(IdeaBase):
    def check_data(self, keep: tuple[str, ...] = ()) -> tuple[dict[str, Any], str]:
        """Run the source's admission checks if it has none yet, check the data against the contract, calibrate the
        decisive test on the treated data, and record the result: the DataCard goes up for review, or the layer stays
        OPEN with the reason when the check is blocked. ``keep`` names treatments a person chose not to apply. Treated
        data of earlier checks that was never frozen is removed from the vault."""
        contract = self._approved_contract()
        current = self._allowed(DATA, Command.CHECK)
        previous = self._last(Command.CHECK, DATA)
        checked = check_data(
            contract,
            self.root,
            self.registry,
            self.data_settings,
            keep=keep,
            feat_settings=self.feat_settings,
            key=self._calibration_key,
        )
        spec = jsonable(checked.spec)
        version = current.version + starts_new_version(current.state, Command.CHECK)
        text = data_card(spec, self.root.name, version)
        verdict = spec["verdict"]
        ready = verdict["status"] != "blocked"

        def card() -> dict[str, Any]:
            self.workspace.write_json(DATA_CARD, spec)
            self.workspace.path(DATA_TEXT).write_text(text + "\n")
            return {"spec": spec, "deps": checked.deps}

        note = "" if ready else f"blocked: {verdict['notes'][0]}"
        self._apply(DATA, Command.CHECK, ready, {"note": note}, effect=card)
        if previous is not None:
            self._drop_unfrozen(as_use(contract.data).name, previous["payload"]["spec"]["products"], spec["products"])
        return spec, text

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
            also=lambda who: _admit(spec["source"], who),
        )

    def _drop_unfrozen(self, source: str, old: dict[str, Any], new: dict[str, Any]) -> None:
        deps = [r["payload"]["deps"] for r in self.ledger.records() if r["kind"] == "approve"]
        for kind in PRODUCTS:
            frozen = {d.get(kind) for d in deps}
            if old[kind]["hash"] != new[kind]["hash"] and old[kind]["hash"] not in frozen:
                vault.drop(f"{source}.{kind}", old[kind]["hash"])

    def _checked_data(self) -> dict[str, Any]:
        """The data check as recorded in the ledger. Refuses if a plugin's code, the data, the source's admission or
        the treated data changed since."""
        record = self._last(Command.CHECK, DATA)
        if record is None:
            raise WorkflowError("the data has not been checked; run `rci data check` first")
        spec = record["payload"]["spec"]
        recorded = record["payload"]["deps"]
        moved = drift(recorded, current_data_deps(self._approved_contract(), recorded, spec, self.registry))
        if moved:
            raise WorkflowError(f"changed since the data check: {', '.join(moved)}; run `rci data check` again")
        return record


def _admit(source: dict[str, Any], who: Approval) -> dict[str, Any]:
    """Approving the data of a source whose admission no one approved yet admits the source too, for every idea that
    uses it, in the shared sources ledger."""
    if source["status"] == "admitted":
        return {}
    sources = Sources()
    record = sources.latest(source["key"])
    if record is None or record["hash"] != source["admission"]:
        raise WorkflowError("the source's admission changed since the data check; run `rci data check` again")
    approval = asdict(who) | {"note": who.describe()}
    approved = record["approved"] or sources.approve(source["key"], record["hash"], approval)
    return {"source_approval": approved["hash"]}


def _data_deps(record: dict[str, Any]) -> dict[str, str]:
    """What a data freeze pins beyond the usual: the check's ledger record and the products it stored in the vault
    (``PRODUCTS``)."""
    products = record["payload"]["spec"]["products"]
    return {"data_check": record["hash"]} | {kind: products[kind]["hash"] for kind in PRODUCTS}
