"""The contract's commands: derive it, and what approving it shows and pins."""

from dataclasses import asdict
from typing import Any

from research_ci.api.contract import Contract
from research_ci.render.contract import contract_card
from research_ci.stages.common import jsonable
from research_ci.stages.contract import current_deps, derive
from research_ci.store.freeze import drift
from research_ci.store.workspace import hash_file
from research_ci.workflow.base import CONTRACT_FILE, IdeaBase, StageFreeze, load_contract
from research_ci.workflow.lineage import CONTRACT
from research_ci.workflow.machine import Command, WorkflowError, starts_new_version
from research_ci.workflow.next import StageStatus

CONTRACT_CARD = "contract.json"
CARD_TEXT = "contract-card.txt"


class ContractCommands(IdeaBase):
    def derive_contract(self) -> tuple[Contract, str]:
        """Load contract.py, derive what the data can decide, and record it: the contract goes up for review, or
        stays OPEN with the reason when derive found it BLOCKED. Deriving a PASSED contract starts a new version."""
        path = self.root / CONTRACT_FILE
        contract = load_contract(path)
        digest = hash_file(path)
        derived = derive(contract, self.registry, self.settings)
        spec, fields = jsonable(derived.spec), jsonable(asdict(contract))
        current = self._allowed(CONTRACT, Command.DERIVE)
        version = current.version + starts_new_version(current.state, Command.DERIVE)
        text = contract_card(spec, self.root.name, version)
        verdict = spec["verdict"]
        ready = verdict["status"] != "blocked"

        def card() -> dict[str, Any]:
            self.workspace.write_json(CONTRACT_CARD, {"contract": fields, "hash": digest, "spec": spec})
            self.workspace.path(CARD_TEXT).write_text(text + "\n")
            return {"contract": digest, "fields": fields, "spec": spec, "deps": derived.deps}

        note = "" if ready else f"blocked: {verdict['notes'][0]}"
        self._apply(CONTRACT, Command.DERIVE, ready, {"note": note}, effect=card)
        return contract, text

    def _stage_freeze(self, stage: str, current: StageStatus) -> StageFreeze | None:
        if stage != CONTRACT:
            return super()._stage_freeze(stage, current)
        record = self._derived_record()
        payload = record["payload"]
        return StageFreeze(
            card=contract_card(payload["spec"], self.root.name, current.version),
            payload={"budget": payload["fields"]["budget"], "derived": record["hash"]},
            preview={"contract": payload["fields"]},
        )

    def _derived_record(self) -> dict[str, Any]:
        """The contract as recorded in the ledger at derive time; files under .rci are only copies. Refuses if the
        contract file, a plugin's code, or the data snapshot changed since."""
        record = self._last(Command.DERIVE, CONTRACT)
        if record is None:
            raise WorkflowError("the contract has not been derived; run `rci contract derive` first")
        path = self.root / CONTRACT_FILE
        if not path.exists() or record["payload"]["contract"] != hash_file(path):
            raise WorkflowError("contract.py changed since it was derived; run `rci contract derive` again")
        recorded = record["payload"]["deps"]
        moved = drift(recorded, current_deps(load_contract(path), recorded, self.registry))
        if moved:
            raise WorkflowError(f"changed since derive: {', '.join(moved)}; run `rci contract derive` again")
        return record
