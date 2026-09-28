import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research_ci.api.contract import Contract, ContractError
from research_ci.store.environment import environment_record, judging_deps
from research_ci.store.freeze import Freeze, drift, make_freeze, write_freeze
from research_ci.store.ledger import Ledger, LedgerError
from research_ci.store.workspace import Workspace, hash_file, rci_home
from research_ci.workflow.lineage import blockers, invalidated, upstream
from research_ci.workflow.machine import Command, Kind, State, WorkflowError, starts_new_version, transition
from research_ci.workflow.next import Action, IdeaStatus, StageStatus, next_actions
from research_ci.workflow.templates import CONTRACT, get_template, kind_of

CONTRACT_FILE = "contract.py"
CONTRACT_CARD = "contract.json"
LEDGER_FILE = "ledger.jsonl"
SKELETON = """from research_ci.api.contract import Contract

CONTRACT = Contract(
    template="{template}",
    claim="TODO: state the claim in one sentence",
    explore_end="2025-06-30T00:00:00Z",
    seal_end="2025-12-31T00:00:00Z",
)
"""

Record = tuple[str, str, dict[str, Any]]


class Idea:
    """One idea's lifecycle. Every state change is a ledger record, and the state is replayed from the ledger."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.workspace = Workspace(self.root)
        self.ledger = _ledger(self.workspace)
        if not self.ledger.path.exists():
            raise WorkflowError(f"{self.root} is not a Research CI idea; run `rci init` first")
        self.ledger.verify()

    @classmethod
    def init(cls, root: Path, template: str) -> "Idea":
        get_template(template)
        workspace = Workspace(Path(root))
        with workspace.lock():
            ledger = _ledger(workspace)
            if ledger.path.exists():
                raise WorkflowError(f"{root} is already a Research CI idea")
            contract = Path(root) / CONTRACT_FILE
            if not contract.exists():
                contract.write_text(SKELETON.format(template=template))
            ledger.append("init", "idea", {"template": template})
            (workspace.root / ".gitignore").write_text("lock\n")
        return cls(root)

    def status(self) -> IdeaStatus:
        records = list(self.ledger.records())
        if not records or records[0]["kind"] != "init":
            raise LedgerError("line 1: the ledger must start with init")
        template = get_template(records[0]["payload"]["template"])
        replay = {name: _Replay() for name in template.stages}
        budget = None
        for record in records[1:]:
            kind, payload, r = record["kind"], record["payload"], replay[record["stage"]]
            r.state, r.version, r.note = State(payload["to"]), payload["version"], payload.get("note", "")
            r.trials += kind == Command.SUBMIT
            r.extra += payload["trials"] if kind == Command.EXTEND else 0
            if kind == Command.APPROVE and record["stage"] == CONTRACT:
                budget = payload["budget"]
        stages = []
        for name, r in replay.items():
            layer_budget = budget + r.extra if budget is not None and kind_of(name) is Kind.LAYER else None
            stages.append(StageStatus(name, kind_of(name), r.state, r.version, r.trials, layer_budget, r.note))
        return IdeaStatus(idea=self.root.name, template=template.name, stages=tuple(stages))

    def next_actions(self) -> list[Action]:
        return next_actions(self.status())

    def derive_contract(self) -> Contract:
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

        def card() -> dict[str, Any]:
            self.workspace.write_json(CONTRACT_CARD, {"contract": asdict(contract), "hash": digest})
            return {"contract": digest, "fields": asdict(contract)}

        self._apply(CONTRACT, Command.REGISTER, effect=card)
        return contract

    def register(self, stage: str, payload: Mapping[str, Any] | None = None) -> State:
        if stage == CONTRACT:
            raise WorkflowError("the contract is registered by `rci contract derive`")
        return self._apply(stage, Command.REGISTER, payload=payload)

    def challenge(self, stage: str, payload: Mapping[str, Any] | None = None) -> State:
        return self._apply(stage, Command.CHALLENGE, payload=payload)

    def check(self, stage: str, passed: bool, note: str = "", payload: Mapping[str, Any] | None = None) -> State:
        result = State.PASSED if passed else State.FAILED
        return self._apply(stage, Command.CHECK, result, {**(payload or {}), "passed": passed, "note": note})

    def preview_approval(self, stage: str, extra: Mapping[str, str] | None = None) -> dict[str, Any]:
        """What ``approve`` would freeze. Raises if approving is not allowed right now."""
        current = self._allowed(stage, Command.APPROVE)
        preview: dict[str, Any] = {"stage": stage, "version": current.version, "deps": self.deps(stage, extra)}
        if stage == CONTRACT:
            preview["contract"] = self._derived_contract()
        return preview

    def approve(self, stage: str, extra: Mapping[str, str] | None = None) -> Freeze:
        self._allowed(stage, Command.APPROVE)
        fields = self._derived_contract() if stage == CONTRACT else None
        freeze = make_freeze(stage, self.deps(stage, extra))

        def write() -> dict[str, Any]:
            write_freeze(self.workspace, freeze)
            out: dict[str, Any] = {"freeze": freeze.hash, "deps": freeze.deps}
            if fields is not None:
                out |= {"budget": fields["budget"], "skip": fields["skip"]}
            return out

        self._apply(stage, Command.APPROVE, effect=write, skip=fields["skip"] if fields else ())
        return freeze

    def submit(
        self,
        stage: str,
        verdict: State,
        extra: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> State:
        """Record a verdict. It becomes TAINTED if anything frozen at approval has changed since."""
        approval = self._last(Command.APPROVE, stage)
        note = ""
        if approval is not None:
            moved = drift(approval["payload"]["deps"], self.deps(stage, extra))
            if moved:
                verdict, note = State.TAINTED, f"frozen dependencies changed: {', '.join(moved)}"
        record = {**(payload or {}), "verdict": verdict, "note": note, "environment": environment_record(self.root)}
        return self._apply(stage, Command.SUBMIT, verdict, record)

    def extend(self, stage: str, trials: int) -> State:
        if trials < 1:
            raise WorkflowError("extend by at least one submission")
        last = self._last(Command.SUBMIT, stage)
        verdict = State(last["payload"]["verdict"]) if last else None
        return self._apply(stage, Command.EXTEND, verdict, {"trials": trials, "note": f"budget extended by {trials}"})

    def deps(self, stage: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        contract = self.root / CONTRACT_FILE
        deps = {"contract": hash_file(contract) if contract.exists() else "missing"}
        if stage != CONTRACT:
            deps |= judging_deps()
            status = self.status()
            for up in upstream(status.names, stage):
                if up != CONTRACT and status.stage(up).state is not State.SKIPPED:
                    approval = self._last(Command.APPROVE, up)
                    deps[f"upstream:{up}"] = approval["payload"]["freeze"] if approval else "none"
        return deps | dict(extra or {})

    def _allowed(self, stage: str, command: Command, result: State | None = None) -> StageStatus:
        status = self.status()
        waiting = blockers(status.names, status.states(), stage)
        if waiting:
            raise WorkflowError(f"{stage} is waiting for {', '.join(waiting)} to pass")
        current = status.stage(stage)
        transition(current.kind, current.state, command, result)
        return current

    def _apply(
        self,
        stage: str,
        command: Command,
        result: State | None = None,
        payload: Mapping[str, Any] | None = None,
        effect: Callable[[], dict[str, Any]] | None = None,
        skip: tuple[str, ...] | list[str] = (),
    ) -> State:
        with self.workspace.lock():
            current = self._allowed(stage, command, result)
            after = transition(current.kind, current.state, command, result)
            if command is Command.SUBMIT and current.kind is Kind.LAYER and current.budget is not None:
                if after is not State.PASSED and current.trials + 1 >= current.budget:
                    after = State.PAUSED
            version = current.version + starts_new_version(current.state, command)
            body = {**(payload or {}), **(effect() if effect else {})}
            self.ledger.append(command, stage, {**body, "from": current.state, "to": after, "version": version})
            for kind, name, extra in self._followups(stage, command, current.state, tuple(skip)):
                self.ledger.append(kind, name, extra)
            return after

    def _followups(self, stage: str, command: Command, before: State, skip: tuple[str, ...]) -> list[Record]:
        status = self.status()
        out: list[Record] = []
        if command is Command.REGISTER and before is State.PASSED:
            for name in invalidated(status.names, status.states(), stage):
                s = status.stage(name)
                out.append(("stale", name, _move(s, State.STALE, f"{stage} was reopened")))
        if command is Command.APPROVE and stage == CONTRACT:
            for name in sorted(get_template(status.template).optional):
                s = status.stage(name)
                if name in skip and s.state is not State.SKIPPED:
                    out.append(("skip", name, _move(s, State.SKIPPED, "skipped by the contract")))
                elif name not in skip and s.state is State.SKIPPED:
                    out.append(("skip", name, _move(s, State.OPEN, "no longer skipped by the contract")))
        return out

    def _last(self, kind: str, stage: str) -> dict[str, Any] | None:
        last = None
        for record in self.ledger.records():
            if record["kind"] == kind and record["stage"] == stage:
                last = record
        return last

    def _derived_contract(self) -> dict[str, Any]:
        """The contract as recorded in the ledger at derive time; files under .rci are only copies."""
        record = self._last(Command.REGISTER, CONTRACT)
        if record is None:
            raise WorkflowError("the contract has not been derived; run `rci contract derive` first")
        path = self.root / CONTRACT_FILE
        if not path.exists() or record["payload"]["contract"] != hash_file(path):
            raise WorkflowError("contract.py changed since it was derived; run `rci contract derive` again")
        return record["payload"]["fields"]


def load_contract(path: Path) -> Contract:
    if not path.is_file():
        raise ContractError(f"{path} does not exist")
    spec = importlib.util.spec_from_file_location("_rci_contract", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"{path.name} failed to load: {exc}") from exc
    contract = getattr(module, "CONTRACT", None)
    if not isinstance(contract, Contract):
        raise ContractError(f"{path.name} must define CONTRACT = Contract(...)")
    return contract


def find_ideas(root: Path) -> list[Path]:
    patterns = (".rci/" + LEDGER_FILE, "*/.rci/" + LEDGER_FILE, "*/*/.rci/" + LEDGER_FILE)
    return sorted({p.parent.parent for pattern in patterns for p in Path(root).glob(pattern)})


@dataclass
class _Replay:
    state: State = State.OPEN
    version: int = 1
    trials: int = 0
    extra: int = 0
    note: str = ""


def _ledger(workspace: Workspace) -> Ledger:
    return Ledger(workspace.path(LEDGER_FILE), anchors=rci_home() / "heads")


def _move(s: StageStatus, to: State, note: str) -> dict[str, Any]:
    return {"from": s.state, "to": to, "version": s.version, "note": note}
