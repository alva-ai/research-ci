import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research_ci.api.contract import Contract, ContractError, as_use
from research_ci.render.card import (
    card_id,
    contract_card,
    data_card,
    explore_card,
    method_card,
    result_card,
    source_card,
)
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import feat as fs
from research_ci.stages.contract import Settings, current_deps, derive, jsonable
from research_ci.stages.data import DataSettings, check_data, current_data_deps
from research_ci.store import vault
from research_ci.store.environment import environment_record, judging_deps
from research_ci.store.freeze import Freeze, drift, make_freeze, write_freeze
from research_ci.store.ledger import Ledger, LedgerError
from research_ci.store.sources import Sources, source_key
from research_ci.store.workspace import Workspace, hash_file, rci_home
from research_ci.workflow.lineage import blockers, invalidated, upstream
from research_ci.workflow.machine import Command, Kind, State, WorkflowError, starts_new_version, transition
from research_ci.workflow.next import Action, IdeaStatus, StageStatus, next_actions
from research_ci.workflow.templates import CONTRACT, get_template, kind_of

CONTRACT_FILE = "contract.py"
CONTRACT_CARD = "contract.json"
CARD_TEXT = "contract-card.txt"
DATA = "data"
DATA_CARD = "data.json"
DATA_TEXT = "data-card.txt"
FEAT = "feat"
FEAT_CARD = "feat.json"
FEAT_TEXT = "feat-card.txt"
LEDGER_FILE = "ledger.jsonl"
SKELETON = """from research_ci.api import Alternative, Contract, Effect, Execution, Forward, Label, Universe, Use

CONTRACT = Contract(
    template="{template}",
    claim="TODO: state the claim in one sentence",
    decisive=(Effect(on="return", at_least=0.001, unit="TODO: what the number measures"),),
    alternatives=(Alternative("TODO", "TODO: another reason the same result could appear"),),
    used=None,  # which data was already used to choose anything here; () if none
    data="TODO-data-plugin",
    costs=Use("ref.flat_costs", {{"commission_bps": 0.0, "slippage_bps": 0.0}}),
    execution=Execution(every="24h"),
    universe=Universe("TODO-universe-plugin", clusters="entity"),
    labels=(Label("return", Use("ref.return", {{"price": "close"}}), horizons=("24h",)),),
    forward=Forward("TODO-data-plugin"),
)
"""

Record = tuple[str, str, dict[str, Any]]
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


class Idea:
    """One idea's lifecycle. Every state change is a ledger record, and the state is replayed from the ledger."""

    def __init__(
        self,
        root: Path,
        registry: Registry | None = None,
        settings: Settings | None = None,
        data_settings: DataSettings | None = None,
        feat_settings: fs.FeatSettings | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.registry, self.settings, self.data_settings = registry, settings, data_settings
        self.feat_settings = feat_settings
        self.workspace = Workspace(self.root)
        self.ledger = _ledger(self.workspace)
        if not self.ledger.path.exists():
            raise WorkflowError(f"{self.root} is not a Research CI idea; run `rci init` first")
        self.ledger.verify()

    @classmethod
    def init(
        cls,
        root: Path,
        template: str,
        registry: Registry | None = None,
        settings: Settings | None = None,
        data_settings: DataSettings | None = None,
        feat_settings: fs.FeatSettings | None = None,
    ) -> "Idea":
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
        return cls(root, registry, settings, data_settings, feat_settings)

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
            r.trials += kind == Command.SUBMIT and record["stage"] != DATA
            r.extra += payload["trials"] if kind == Command.EXTEND else 0
            if kind == Command.APPROVE and record["stage"] == CONTRACT:
                budget = payload["budget"]
        stages = []
        for name, r in replay.items():
            layer_budget = None
            if budget is not None and kind_of(name) is Kind.LAYER and name != DATA:
                layer_budget = (budget.get(name, DEFAULT_BUDGET) if isinstance(budget, dict) else budget) + r.extra
            stages.append(StageStatus(name, kind_of(name), r.state, r.version, r.trials, layer_budget, r.note))
        return IdeaStatus(idea=self.root.name, template=template.name, stages=tuple(stages))

    def next_actions(self) -> list[Action]:
        return next_actions(self.status())

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

    def source_card(self) -> str:
        record = self._source_record()
        return source_card(record["payload"]["spec"])

    def card(self, stage: str) -> str:
        """The card a person approves for ``stage`` (``contract``, ``source`` or ``data``), as it is now."""
        if stage == "source":
            return self.source_card()
        version = self.status().stage(stage).version
        if stage == CONTRACT:
            return contract_card(self._derived_record()["spec"], self.root.name, version)
        if stage == DATA:
            return data_card(self._checked_data()["payload"]["spec"], self.root.name, version)
        if stage == FEAT:
            budget = self.status().stage(FEAT).budget
            return method_card(self._feat_method()["payload"]["spec"], self.root.name, version, budget)
        raise WorkflowError(f"{stage} has no card")

    def approve_source(self, approval: "Approval | None" = None) -> dict[str, Any]:
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

    def register(self, stage: str, payload: Mapping[str, Any] | None = None) -> State:
        if stage == CONTRACT:
            raise WorkflowError("the contract is registered by `rci contract derive`")
        if stage == DATA:
            raise WorkflowError("the data layer is registered by `rci data check`")
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
            record = self._derived_record()
            preview["contract"] = record["fields"]
            preview["card"] = contract_card(record["spec"], self.root.name, current.version)
        if stage == DATA:
            record = self._checked_data()
            preview["deps"] |= _data_deps(record)
            preview["card"] = data_card(record["payload"]["spec"], self.root.name, current.version)
        if stage == FEAT and self._feat_registered():
            record = self._feat_method()
            preview["deps"] |= _feat_deps(record)
            spec = record["payload"]["spec"]
            preview["card"] = method_card(spec, self.root.name, current.version, current.budget)
        return preview

    def approve(self, stage: str, extra: Mapping[str, str] | None = None, approval: Approval | None = None) -> Freeze:
        """Freeze a stage. ``approval`` says who approved and through which channel; the ledger keeps it, and the
        verdict machinery caps what agent-approved or relayed stages can reach."""
        current = self._allowed(stage, Command.APPROVE)
        fields = self._derived_contract() if stage == CONTRACT else None
        who = approval or Approval()
        if stage == CONTRACT:
            self._shown(who, contract_card(self._derived_record()["spec"], self.root.name, current.version))
        if stage == DATA:
            record = self._checked_data()
            spec = record["payload"]["spec"]
            if spec["verdict"]["status"] == "decide" and who.by != "human":
                raise WorkflowError("the data card asks a person to decide; an agent cannot approve it with --auto")
            self._shown(who, data_card(spec, self.root.name, current.version))
            extra = {**(extra or {}), **_data_deps(record)}
        if stage == FEAT and self._feat_registered():
            record = self._feat_method()
            self._shown(who, method_card(record["payload"]["spec"], self.root.name, current.version, current.budget))
            extra = {**(extra or {}), **_feat_deps(record)}
        freeze = make_freeze(stage, self.deps(stage, extra))

        def write() -> dict[str, Any]:
            write_freeze(self.workspace, freeze)
            out: dict[str, Any] = {"freeze": freeze.hash, "deps": freeze.deps, "approval": asdict(who)}
            out["note"] = who.describe()
            if fields is not None:
                out |= {"budget": fields["budget"], "skip": fields["skip"], "derived": self._derived_hash()}
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

    def _derived_hash(self) -> str:
        record = self._last(Command.REGISTER, CONTRACT)
        return record["hash"] if record else "none"

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

    def _drop_unfrozen(self, source: str, old: dict[str, Any], new: dict[str, Any]) -> None:
        deps = [r["payload"]["deps"] for r in self.ledger.records() if r["kind"] == "approve"]
        for key, kind in (("hash", "panel"), ("exclusions", "exclusions")):
            frozen = {d.get(kind) for d in deps}
            if old[key] != new[key] and old[key] not in frozen:
                vault.drop(f"{source}.{kind}", old[key])

    def _approved_contract(self) -> Contract:
        approval = self._last(Command.APPROVE, CONTRACT)
        if approval is None:
            raise WorkflowError("the contract is not approved yet")
        path = self.root / CONTRACT_FILE
        if not path.exists() or hash_file(path) != approval["payload"]["deps"]["contract"]:
            raise WorkflowError("contract.py changed since it was approved; derive and approve it again")
        return load_contract(path)

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


def _data_deps(record: dict[str, Any]) -> dict[str, str]:
    """What a data freeze pins beyond the usual: the check's ledger record and the treated data it produced."""
    panel = record["payload"]["spec"]["panel"]
    return {"data_check": record["hash"], "panel": panel["hash"], "exclusions": panel["exclusions"]}


def _feat_deps(record: dict[str, Any]) -> dict[str, str]:
    """What a feat freeze pins beyond the usual: the method's ledger record, the criteria and nulls files, and the
    code of every plugin feat runs."""
    return {"feat_method": record["hash"], **record["payload"]["deps"]}


def _ledger(workspace: Workspace) -> Ledger:
    return Ledger(workspace.path(LEDGER_FILE), anchors=rci_home() / "heads")


def _move(s: StageStatus, to: State, note: str) -> dict[str, Any]:
    return {"from": s.state, "to": to, "version": s.version, "note": note}
