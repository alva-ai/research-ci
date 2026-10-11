"""What every stage of an idea shares: the ledger, the state replayed from it, and the commands that apply
transitions. Each stage adds its own commands in a module of its own; ``Idea`` puts them together."""

import importlib.util
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

from research_ci.api.contract import Contract, ContractError
from research_ci.render.common import card_id
from research_ci.runner.plugins import Registry
from research_ci.stages.contract import Settings
from research_ci.stages.data import DataSettings
from research_ci.stages.feat import FeatSettings, Frozen, calibration_key
from research_ci.store.environment import environment_record, judging_deps
from research_ci.store.freeze import Freeze, drift, make_freeze, write_freeze
from research_ci.store.ledger import Ledger, LedgerError
from research_ci.store.workspace import Workspace, hash_file, rci_home
from research_ci.workflow.lineage import CONTRACT, STAGES, blockers, invalidated, kind_of, upstream
from research_ci.workflow.machine import Command, Kind, State, Verdict, WorkflowError, starts_new_version, transition
from research_ci.workflow.next import Action, IdeaStatus, StageStatus, next_actions
from research_ci.workflow.templates import CONTRACT_FILE, EXAMPLE, EXAMPLE_FILE, SKELETON, START, START_FILE

LEDGER_FILE = "ledger.jsonl"
FORMAT = 2
DATA = "data"
FEAT = "feat"
DEFAULT_BUDGET = 20


@dataclass(frozen=True)
class Approval:
    """Who approved: a person at a terminal; a person whose words an agent relayed (the words are kept, and so is the
    id of the card the person was shown); or, in fully automatic research, the agent itself."""

    by: str = "human"
    channel: str = "terminal"
    words: str = ""
    card: str = ""

    def __post_init__(self) -> None:
        if (self.by, self.channel) not in APPROVAL_KINDS:
            raise WorkflowError(f"unknown approval {self.by}/{self.channel}")
        if self.channel == "relayed" and not self.words.strip():
            raise WorkflowError("a relayed approval must quote what the person said")

    def describe(self) -> str:
        if self.channel == "relayed":
            return f"approved by a person, relayed by an agent: {self.words!r}"
        if self.by == "agent":
            return "approved by the agent (automatic research)"
        return "approved by a person at a terminal"


APPROVAL_KINDS = {("human", "terminal"), ("human", "relayed"), ("agent", "auto")}


@dataclass(frozen=True)
class StageFreeze:
    """What approving a stage shows and pins beyond the usual: the card a person approves, extra frozen
    dependencies, extra ledger payload and preview fields, when only a person may approve, why, and what else
    approving records (``also``, given who approved; its result joins the ledger payload)."""

    card: str
    deps: Mapping[str, str] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    preview: Mapping[str, Any] = field(default_factory=dict)
    person_only: str = ""
    also: Callable[["Approval"], Mapping[str, Any]] | None = None


class IdeaBase:
    """One idea's lifecycle. Every state change is a ledger record, and the state is replayed from the ledger."""

    def __init__(
        self,
        root: Path,
        registry: Registry | None = None,
        settings: Settings | None = None,
        data_settings: DataSettings | None = None,
        feat_settings: FeatSettings | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.registry, self.settings, self.data_settings = registry, settings, data_settings
        self.feat_settings = feat_settings
        self.workspace = Workspace(self.root)
        self.ledger = _ledger(self.workspace)
        if not self.ledger.path.exists():
            raise WorkflowError(_not_an_idea(self.root))
        self.ledger.verify()
        first = next(self.ledger.records(), None)
        if first is not None and first["kind"] == "init" and first["payload"].get("format") != FORMAT:
            raise LedgerError(
                f"line 1: {self.root.name} was started by an older Research CI, whose lifecycle this one does not "
                "replay; start the idea again in a fresh directory with `rci init`"
            )

    @classmethod
    def init(
        cls,
        root: Path,
        registry: Registry | None = None,
        settings: Settings | None = None,
        data_settings: DataSettings | None = None,
        feat_settings: FeatSettings | None = None,
    ) -> Self:
        workspace = Workspace(Path(root))
        with workspace.lock():
            ledger = _ledger(workspace)
            if ledger.path.exists():
                raise WorkflowError(f"{root} is already a Research CI idea")
            files = {
                CONTRACT_FILE: SKELETON,
                START_FILE: START.format(name=Path(root).resolve().name, idea=Path(root)),
                EXAMPLE_FILE: EXAMPLE,
            }
            for name, text in files.items():
                path = Path(root) / name
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
            ledger.append("init", "idea", {"format": FORMAT})
            (workspace.root / ".gitignore").write_text("lock\n")
        return cls(root, registry, settings, data_settings, feat_settings)

    def status(self) -> IdeaStatus:
        records = list(self.ledger.records())
        if not records or records[0]["kind"] != "init":
            raise LedgerError("line 1: the ledger must start with init")
        replay = {name: _Replay() for name in STAGES}
        budget = None
        for record in records[1:]:
            kind, payload, r = record["kind"], record["payload"], replay[record["stage"]]
            r.approved = r.approved and payload["version"] == r.version or kind == Command.APPROVE
            r.state, r.version, r.note = State(payload["to"]), payload["version"], payload.get("note", "")
            r.trials += kind == Command.SUBMIT
            r.extra += payload["trials"] if kind == Command.EXTEND else 0
            if kind == Command.APPROVE and record["stage"] == CONTRACT:
                budget = payload["budget"]
        stages = []
        for name, r in replay.items():
            layer_budget = None
            if budget is not None and kind_of(name) is Kind.LAYER and name != DATA:
                layer_budget = (budget.get(name, DEFAULT_BUDGET) if isinstance(budget, dict) else budget) + r.extra
            stages.append(
                StageStatus(name, kind_of(name), r.state, r.version, r.trials, layer_budget, r.note, r.approved)
            )
        return IdeaStatus(idea=self.root.name, stages=tuple(stages))

    def next_actions(self) -> list[Action]:
        return next_actions(self.status())

    def card(self, stage: str) -> str:
        """The card a person approves for ``stage``, as it is now."""
        status = self.status()
        found = self._stage_freeze(stage, status.stage(stage)) if stage in STAGES else None
        if found is None:
            raise WorkflowError(f"{stage} has no card")
        return found.card

    def register(self, stage: str, payload: Mapping[str, Any] | None = None) -> State:
        """Register a checkpoint's plan for a person to review."""
        return self._apply(stage, Command.REGISTER, True, payload)

    def preview_approval(self, stage: str, extra: Mapping[str, str] | None = None) -> dict[str, Any]:
        """What ``approve`` would freeze. Raises if approving is not allowed right now."""
        current = self._allowed(stage, Command.APPROVE)
        self._passed_last(stage)
        found = self._stage_freeze(stage, current)
        preview: dict[str, Any] = {"stage": stage, "version": current.version, "deps": self.deps(stage, extra)}
        if found is not None:
            preview["deps"] |= found.deps
            preview["card"] = found.card
            preview |= found.preview
        return preview

    def approve(self, stage: str, extra: Mapping[str, str] | None = None, approval: Approval | None = None) -> Freeze:
        """A person accepts the result under review, or a checkpoint's plan, and the stage is frozen. ``approval``
        says who approved and through which channel; the ledger keeps it."""
        current = self._allowed(stage, Command.APPROVE)
        self._passed_last(stage)
        who = approval or Approval()
        found = self._stage_freeze(stage, current)
        if found is not None:
            if found.person_only and who.by != "human":
                raise WorkflowError(found.person_only)
            self._shown(who, found.card)
            extra = {**(extra or {}), **found.deps}
        freeze = make_freeze(stage, self.deps(stage, extra))

        def write() -> dict[str, Any]:
            write_freeze(self.workspace, freeze)
            out: dict[str, Any] = {"freeze": freeze.hash, "deps": freeze.deps, "approval": asdict(who)}
            out["note"] = who.describe()
            if found is not None:
                out |= dict(found.payload) | dict(found.also(who) if found.also else {})
            return out

        self._apply(stage, Command.APPROVE, effect=write)
        return freeze

    def submit(self, stage: str, verdict: Verdict, payload: Mapping[str, Any] | None = None, note: str = "") -> State:
        """Record a layer submission and its verdict, one trial: PASSED puts the layer up for review; FAILED and
        TAINTED leave it OPEN, or PAUSED when that was the last trial."""
        n = self._allowed(stage, Command.SUBMIT, verdict is Verdict.PASSED).trials + 1
        head = f"submission {n} {verdict}"
        record = {
            **(payload or {}),
            "verdict": verdict,
            "note": f"{head}: {note}" if note else head,
            "environment": environment_record(self.root),
        }
        return self._apply(stage, Command.SUBMIT, verdict is Verdict.PASSED, record)

    def run(self, stage: str, verdict: Verdict, extra: Mapping[str, str] | None = None) -> State:
        """Record a checkpoint's one run. It is TAINTED if anything frozen when its plan was approved has changed."""
        approval = self._last(Command.APPROVE, stage)
        note = f"ran once: {verdict}"
        if approval is not None:
            moved = drift(approval["payload"]["deps"], self.deps(stage, extra))
            if moved:
                verdict, note = Verdict.TAINTED, f"ran once: TAINTED; frozen dependencies changed: {', '.join(moved)}"
        record = {"verdict": verdict, "note": note, "environment": environment_record(self.root)}
        return self._apply(stage, Command.RUN, verdict is Verdict.PASSED, record)

    def extend(self, stage: str, trials: int) -> State:
        if trials < 1:
            raise WorkflowError("extend by at least one submission")
        return self._apply(stage, Command.EXTEND, payload={"trials": trials, "note": f"budget extended by {trials}"})

    def deps(self, stage: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        contract = self.root / CONTRACT_FILE
        deps = {"contract": hash_file(contract) if contract.exists() else "missing"}
        if stage != CONTRACT:
            deps |= judging_deps()
            for up in upstream(stage):
                if up != CONTRACT:
                    approval = self._last(Command.APPROVE, up)
                    deps[f"upstream:{up}"] = approval["payload"]["freeze"] if approval else "none"
        return deps | dict(extra or {})

    def _stage_freeze(self, stage: str, current: StageStatus) -> StageFreeze | None:
        """A stage's own card and pins, from the module of the stage that has them."""
        return None

    def _gate_key(self, frozen: Frozen) -> dict[str, str]:
        """What the calibration of the gate, the contract's decisive effect in the floor world, rests on: the contract,
        the engine, the treated data, the code of the plugins feat runs, and the calibration settings."""
        return self.deps(CONTRACT) | judging_deps() | calibration_key(frozen, self.feat_settings)

    def _allowed(self, stage: str, command: Command, ready: bool = False) -> StageStatus:
        return self._check(self.status(), stage, command, ready)

    def _check(self, status: IdeaStatus, stage: str, command: Command, ready: bool = False) -> StageStatus:
        waiting = blockers(status.states(), stage)
        if waiting:
            raise WorkflowError(f"{stage} is waiting for {', '.join(waiting)} to pass")
        current = status.stage(stage)
        transition(current.kind, current.state, command, ready, current.approved)
        if command is Command.SUBMIT and current.budget is not None and current.trials >= current.budget:
            raise WorkflowError(f"all {current.budget} submissions of {stage} are used; a person reviews the last one")
        return current

    def _apply(
        self,
        stage: str,
        command: Command,
        ready: bool = False,
        payload: Mapping[str, Any] | None = None,
        effect: Callable[[], dict[str, Any]] | None = None,
    ) -> State:
        with self.workspace.lock():
            status = self.status()
            current = self._check(status, stage, command, ready)
            after = transition(current.kind, current.state, command, ready, current.approved)
            if command is Command.SUBMIT and not ready and current.budget is not None:
                after = State.PAUSED if current.trials + 1 >= current.budget else after
            reopens = starts_new_version(current.state, command)
            body = {**(payload or {}), **(effect() if effect else {})}
            version = current.version + reopens
            self.ledger.append(command, stage, {**body, "from": current.state, "to": after, "version": version})
            if reopens and current.state is State.PASSED:
                for name in invalidated(status.states(), self._started(status), stage):
                    self.ledger.append("stale", name, _move(status.stage(name), State.STALE, f"{stage} was reopened"))
            return after

    def _record(self, stage: str, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """A record that changes no state: it keeps the stage's note unless it brings one."""
        with self.workspace.lock():
            s = self.status().stage(stage)
            body = {"note": s.note, **payload, "from": s.state, "to": s.state, "version": s.version}
            return self.ledger.append(kind, stage, body)

    def _started(self, status: IdeaStatus) -> set[str]:
        """Stages with a record in their current version."""
        return {
            r["stage"]
            for r in self.ledger.records()
            if r["stage"] in STAGES and r["payload"].get("version") == status.stage(r["stage"]).version
        }

    def _passed_last(self, stage: str) -> None:
        """A person accepts a layer only when its latest submission PASSED; verdicts come from code."""
        if kind_of(stage) is not Kind.LAYER or stage == DATA:
            return
        last = self._last(Command.SUBMIT, stage)
        if last is None or last["payload"]["verdict"] != Verdict.PASSED:
            raise WorkflowError(f"the latest submission of {stage} did not pass; only a PASSED one can be accepted")

    def _last(self, kind: str, stage: str) -> dict[str, Any] | None:
        last = None
        for record in self.ledger.records():
            if record["kind"] == kind and record["stage"] == stage:
                last = record
        return last

    @staticmethod
    def _shown(who: "Approval", card: str) -> None:
        """A relayed approval must quote the id of the card it approves: the card the person was shown is this one."""
        if who.channel == "relayed" and who.card != card_id(card):
            got = f"got {who.card!r}" if who.card else "none was given"
            raise WorkflowError(f"a relayed approval must quote the card the person saw: card {card_id(card)}, {got}")

    def _approved_contract(self) -> Contract:
        approval = self._last(Command.APPROVE, CONTRACT)
        if approval is None:
            raise WorkflowError("the contract is not approved yet")
        path = self.root / CONTRACT_FILE
        if not path.exists() or hash_file(path) != approval["payload"]["deps"]["contract"]:
            raise WorkflowError("contract.py changed since it was approved; derive and approve it again")
        return load_contract(path)


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
    approved: bool = False


def _ledger(workspace: Workspace) -> Ledger:
    return Ledger(workspace.path(LEDGER_FILE), anchors=rci_home() / "heads")


def _not_an_idea(root: Path) -> str:
    """Why ``root`` is not an idea, and the likely fix when the current directory, or the directory above ``root``,
    is one."""
    head = f"{root} is not a Research CI idea"
    here = Path.cwd().resolve()
    if here != root and Workspace(here).path(LEDGER_FILE).exists():
        return f"{head}; you are already in the idea {here.name}; drop -i, or run from its parent"
    if Workspace(root.parent).path(LEDGER_FILE).exists():
        return f"{head}; {root.parent.name} is the idea; pass -i {os.path.relpath(root.parent, here)}"
    return f"{head}; run `rci init` first"


def _move(s: StageStatus, to: State, note: str) -> dict[str, Any]:
    return {"from": s.state, "to": to, "version": s.version, "note": note}
