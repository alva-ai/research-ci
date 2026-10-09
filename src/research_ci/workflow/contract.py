"""The contract's commands: derive it, and what approving it shows and pins."""

from dataclasses import asdict
from typing import Any

from research_ci.api.contract import Contract
from research_ci.render.contract import contract_card
from research_ci.stages.contract import current_deps, derive, jsonable
from research_ci.store.freeze import drift
from research_ci.store.workspace import hash_file
from research_ci.workflow.base import CONTRACT_FILE, IdeaBase, StageFreeze, load_contract
from research_ci.workflow.machine import Command, WorkflowError
from research_ci.workflow.next import StageStatus
from research_ci.workflow.templates import CONTRACT, get_template

CONTRACT_CARD = "contract.json"
CARD_TEXT = "contract-card.txt"


class ContractCommands(IdeaBase):
    def derive_contract(self) -> tuple[Contract, str]:
        path = self.root / CONTRACT_FILE
        contract = load_contract(path)
        template = get_template(self.status().template)
        if contract.template != template.name:
            raise WorkflowError(f"contract.py says template {contract.template!r}; this idea is {template.name!r}")
        extra = set(contract.skip) - template.optional
        if extra:
            optional = ", ".join(sorted(template.optional)) or "none"
            raise WorkflowError(
                f"cannot skip {', '.join(sorted(extra))}; optional stages of {template.name}: {optional}"
            )
        digest = hash_file(path)
        derived = derive(contract, self.root, self.registry, self.settings)
        spec, fields = jsonable(derived.spec), jsonable(asdict(contract))
        version = self.status().stage(CONTRACT).version

        def card() -> dict[str, Any]:
            self.workspace.write_json(CONTRACT_CARD, {"contract": fields, "hash": digest, "spec": spec})
            text = contract_card(spec, self.root.name, version)
            self.workspace.path(CARD_TEXT).write_text(text + "\n")
            return {"contract": digest, "fields": fields, "spec": spec, "deps": derived.deps}

        self._apply(CONTRACT, Command.REGISTER, effect=card)
        return contract, contract_card(spec, self.root.name, version)

    def _stage_freeze(self, stage: str, current: StageStatus) -> StageFreeze | None:
        if stage != CONTRACT:
            return super()._stage_freeze(stage, current)
        record = self._derived_record()
        fields = record["fields"]
        return StageFreeze(
            card=contract_card(record["spec"], self.root.name, current.version),
            payload={"budget": fields["budget"], "skip": fields["skip"], "derived": self._derived_hash()},
            preview={"contract": fields},
            skip=tuple(fields["skip"]),
        )

    def _derived_hash(self) -> str:
        record = self._last(Command.REGISTER, CONTRACT)
        return record["hash"] if record else "none"

    def _derived_contract(self) -> dict[str, Any]:
        return self._derived_record()["fields"]

    def _derived_record(self) -> dict[str, Any]:
        """The contract as recorded in the ledger at derive time; files under .rci are only copies. Refuses if the
        contract file, a plugin's code, a baseline file, or the data snapshot changed since."""
        record = self._last(Command.REGISTER, CONTRACT)
        if record is None:
            raise WorkflowError("the contract has not been derived; run `rci contract derive` first")
        path = self.root / CONTRACT_FILE
        if not path.exists() or record["payload"]["contract"] != hash_file(path):
            raise WorkflowError("contract.py changed since it was derived; run `rci contract derive` again")
        recorded = record["payload"]["deps"]
        moved = drift(recorded, current_deps(load_contract(path), self.root, recorded, self.registry))
        if moved:
            raise WorkflowError(f"changed since derive: {', '.join(moved)}; run `rci contract derive` again")
        failed = [c["name"] for c in record["payload"]["spec"]["checks"] if c["status"] == "fail"]
        if failed:
            raise WorkflowError(f"plugins failed protocol tests ({', '.join(failed)}); fix them and derive again")
        return record["payload"]
