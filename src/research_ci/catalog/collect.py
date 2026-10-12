"""What `rci methods --markdown` lists, read from the code: every stage's floor list with its default list where the
floor marks it, slot by slot, each step with its role, what it needs, and the eval functions it references; how lab
evals plug in (slot 3) and what holds them to their tests (slot 4), with the engine internals that carry worked
examples of their own; and every public function of the eval library with where it is used and what backs it."""

import dis
import inspect
import pkgutil
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from types import CodeType, FunctionType, ModuleType
from typing import Any

import research_ci.eval
from research_ci.catalog import listing
from research_ci.stages import lab, skeleton
from research_ci.stages.contract import defaults as contract_defaults
from research_ci.stages.contract import floor as contract_floor
from research_ci.stages.data import defaults as data_defaults
from research_ci.stages.data import floor as data_floor
from research_ci.stages.data import source as source_floor
from research_ci.stages.data import treatments
from research_ci.stages.feat import defaults as feat_defaults
from research_ci.stages.feat import floor as feat_floor
from research_ci.validation import calibration
from research_ci.validation import load as lab_evals
from research_ci.validation import tests as validation_tests

EVAL = "research_ci.eval"
STAGES = "research_ci.stages"
ORDER = ("protocol", "data", "statistics", "structure", "synthetic")
UNDOCUMENTED = {
    "protocol.deterministic": "Two identical consumer runs must give the same net.",
    "protocol.max_drawdown": "The consumer's deepest fall in equity from its running peak.",
    "protocol.total": "The consumer's net summed over the run.",
    "protocol.truncated": "The field table cut to the rows known by `t`.",
    "data.invariants.hedge_check": "The card's hedged-legs line, naming the pairs most often apart.",
}


def collect() -> dict[str, Any]:
    stages = [
        _stage("contract", "rci contract derive", contract_floor, contract_defaults),
        _stage("data", "rci data check", data_floor, data_defaults, admission=source_floor),
        _stage("feat", "rci feat submit", feat_floor, feat_defaults),
    ]
    evals = {
        "about": listing.LAB_EVAL,
        "example": listing.CASE_EXAMPLE,
        "layers": list(lab_evals.LAYERS),
        "declarations": listing.lab_evals(),
        "worlds": listing.nulls(),
    }
    tests = {"checks": tested(), "outcomes": dict(lab_evals.OUTCOMES), "internals": internals()}
    return {
        "stages": stages,
        "roles": dict(skeleton.ROLES),
        "slots": dict(skeleton.SLOTS),
        "lab": evals,
        "tests": tests,
        "library": library(stages),
    }


def tested() -> list[dict[str, str]]:
    """What holds a lab eval to its tests, slot 4, each with the function that does it."""
    found = [
        ("CASES, required", validation_tests.run_tests, "each worked example gives what it expects, within its tol"),
        (
            "PROPERTIES, optional",
            validation_tests.run_tests,
            "Size, Power and Recovers on synthetic markets, judged against the floor world as a submission is",
        ),
        (
            "kept by its files",
            lab_evals.validate,
            "the cases and properties that run the eval, found in its module and kept by the hash of its files",
        ),
        ("no tests, or the example", lab_evals.refuse_untested, "refused before anything runs, saying what to write"),
        (
            "PITS, data only",
            data_floor.pits_caught,
            "every pit of what it covers, built in or in its module's PITS, planted in clean data, is flagged",
        ),
        (
            "the decisive test's world",
            calibration.calibrate,
            "stand-ins that carry only what the world keeps die, and planted signals live",
        ),
    ]
    return [{"test": what, "file": _path_of(fn), "function": fn.__name__, "does": does} for what, fn, does in found]


def internals() -> list[dict[str, Any]]:
    """The engine's own machinery that carries worked examples like the library, but that no idea lists: the tests of
    evals and the data layer's treatments, influence and decisive numbers."""
    out = []
    for module in (validation_tests, calibration, treatments):
        backing = listing._backing(module)
        functions = [
            {"name": fn.__name__, "doc": listing.first_line(fn), "examples": backing.get(fn.__name__, (0, 0))[0]}
            for fn in public(module)
        ]
        out.append({"module": _path(module), "functions": functions})
    return out


def library(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every public function of every eval module, by package, with where it is used and what backs it."""
    by_stage: dict[str, list[str]] = {}
    for s in stages:
        for _, rows in s["sections"]:
            for row in rows:
                for m in row["methods"]:
                    by_stage.setdefault(m, []).append(s["stage"])
    research_code = {f"{_short(m['module'])}.{m['name']}" for m in listing.methods()}
    modules = {p: list(listing._modules(f"{EVAL}.{p}")) for p in packages()}
    inside, examples = _inside([m for ms in modules.values() for m in ms])
    secondary = (("inside eval", inside), ("worked examples", examples))
    out = []
    for p, ms in modules.items():
        entries = []
        for module in ms:
            backing = listing._backing(module)
            functions = []
            for fn in public(module):
                key = _key(fn)
                used = list(dict.fromkeys(by_stage.get(key, [])))
                used += ["research code"] if key in research_code else []
                used = used or [tag for tag, keys in secondary if key in keys]
                used = used or ["algo, planned" if p == "protocol" else "tests only"]
                cases, props = backing.get(fn.__name__, (0, 0))
                doc = listing.first_line(fn) or UNDOCUMENTED.get(key, "")
                functions.append(
                    {"key": key, "name": fn.__name__, "doc": doc, "used": used, "examples": cases, "properties": props}
                )
            entries.append({"module": module.__name__.rsplit(".", 1)[-1], "functions": functions})
        out.append({"package": p, "modules": entries})
    return out


def packages() -> list[str]:
    found = [m.name for m in pkgutil.iter_modules(research_ci.eval.__path__)]
    return sorted(found, key=lambda name: (ORDER.index(name) if name in ORDER else len(ORDER), name))


def public(module: ModuleType) -> list[FunctionType]:
    return [
        fn
        for name, fn in inspect.getmembers(module, inspect.isfunction)
        if fn.__module__ == module.__name__ and not name.startswith("_")
    ]


def reached(fn: Any) -> list[str]:
    """The eval functions a step's function reaches, as ``package.module.function`` under ``research_ci.eval``."""
    return [_key(f) for f in _reach(fn, STAGES) if _is_eval(f)]


def _stage(
    name: str, command: str, floor: ModuleType, defaults: ModuleType, admission: ModuleType | None = None
) -> dict[str, Any]:
    """A stage's lists as its table shows them, slot by slot: the floor list with the default list where the floor
    marks it, each step under its slot in the order it runs, and, for the data, the admission's list first."""
    sections = []
    if admission is not None:
        sections.append((f"1 {skeleton.SLOTS[1]}: admission, once per source", _rows(admission.STEPS, [], "floor")))
    rows = _rows(floor.STEPS, defaults.CHECKS, "floor")
    for slot, label in skeleton.SLOTS.items():
        listed = [r for r in rows if r["slot"] == slot]
        if listed:
            sections.append((f"{slot} {label}", listed))
    files = [_path(floor), _path(defaults)] + ([_path(admission)] if admission is not None else [])
    files += [_path(lab)] if any(s in floor.STEPS for s in lab.READY) else []
    return {"stage": name, "command": command, "files": files, "sections": sections}


def _rows(floor: Sequence[Any], defaults: Sequence[skeleton.Step[Any]], kind: str) -> list[dict[str, Any]]:
    out = []
    for item in floor:
        listed = [(item, kind)] if isinstance(item, skeleton.Step) else [(s, "default") for s in defaults]
        out += [
            {
                "item": s.name,
                "what": s.what,
                "role": s.role,
                "list": where,
                "slot": s.slot if s.slot is not None else 2 if where == "default" else 1,
                "needs": s.needs,
                "methods": reached(s.run),
            }
            for s, where in listed
        ]
    return out


def _path(module: ModuleType) -> str:
    return module.__name__.removeprefix("research_ci.").replace(".", "/") + ".py"


def _path_of(fn: Any) -> str:
    return fn.__module__.removeprefix("research_ci.").replace(".", "/") + ".py"


def _inside(modules: list[ModuleType]) -> tuple[set[str], set[str]]:
    """The eval functions other eval functions call, and those the worked examples and properties of other modules
    call."""
    inside: set[str] = set()
    examples: set[str] = set()
    for module in modules:
        for fn in public(module):
            inside |= {_key(f) for f in _reach(fn) if _is_eval(f) and f is not fn}
        for item in (*getattr(module, "CASES", []), *getattr(module, "PROPERTIES", [])):
            run = getattr(item, "run", None) or getattr(item, "of", None)
            examples |= {_key(f) for f in _reach(run) if _is_eval(f) and f.__module__ != module.__name__}
    return inside, examples


def _reach(fn: Any, through: str = "") -> list[FunctionType]:
    """The functions ``fn`` references, in the order they come: through the functions of its own module it calls, and
    of the modules under ``through``, and the methods of the classes its parameters are annotated with."""
    code = getattr(fn, "__code__", None)
    if not isinstance(code, CodeType):
        return []
    home = fn.__module__
    found: list[FunctionType] = []
    queue: list[tuple[CodeType, dict[str, Any], dict[str, Any]]] = [(code, fn.__globals__, _locals(fn))]
    seen: set[CodeType] = set()
    while queue:
        top, scope, local = queue.pop(0)
        for c in _codes(top):
            for f, owner in _named(c, scope, local if through else {}):
                if f not in found:
                    found.append(f)
                inner = f.__module__ == home or (through and (f.__module__ or "").startswith(f"{through}."))
                if inner and f.__code__ not in seen:
                    seen.add(f.__code__)
                    queue.append((f.__code__, f.__globals__, _locals(f, owner)))
    return found


class _Instance:
    """A value known only by its class, such as a parameter annotated with it."""

    def __init__(self, cls: type) -> None:
        self.cls = cls


LOADS_NAME = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})
LOADS_LOCAL = frozenset({"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_DEREF", "LOAD_CLOSURE"})
LOADS_ATTR = frozenset({"LOAD_ATTR", "LOAD_METHOD"})


def _named(code: CodeType, scope: dict[str, Any], local: dict[str, Any]) -> list[tuple[FunctionType, type | None]]:
    """The function objects one code object references, each with the class it was found on: a global it loads, the
    functions of a global table it loads, an attribute it loads of a module or class it loads (``module.function``,
    ``Class.method``), and a method or property of a local whose class is known from ``local`` (``seen.judge``,
    ``self.inputs``, ``seen.frozen.inputs`` through the field's annotation). A name alone is never enough: a function
    of another module, or of a class, with the same name is not one this code references."""
    out: list[tuple[FunctionType, type | None]] = []
    top: Any = None
    for ins in dis.get_instructions(code):
        name = str(ins.argval[-1] if isinstance(ins.argval, tuple) and ins.argval else ins.argval)
        if ins.opname in LOADS_NAME:
            top = scope.get(name)
            values = list(top.values()) if isinstance(top, dict) else top
            out += [(f, None) for f in (values if isinstance(values, list | tuple) else [top]) if inspect.isfunction(f)]
        elif ins.opname in LOADS_LOCAL:
            top = local.get(name)
        elif ins.opname in LOADS_ATTR:
            top, found = _attribute(top, name)
            out += [found] if found is not None else []
        else:
            top = None
    return list(dict.fromkeys(out))


def _attribute(top: Any, name: str) -> tuple[Any, tuple[FunctionType, type | None] | None]:
    """What loading ``name`` from ``top`` gives, and the function it references, if any."""
    if isinstance(top, ModuleType) and top.__name__.startswith("research_ci."):
        value = getattr(top, name, None)
        if inspect.isfunction(value) and value.__module__ == top.__name__:
            return value, (value, None)
        return value, None
    cls = top.cls if isinstance(top, _Instance) else top if inspect.isclass(top) else None
    if cls is None or not (cls.__module__ or "").startswith("research_ci."):
        return None, None
    owner = next((k for k in cls.__mro__ if name in vars(k)), None)
    if owner is not None and inspect.isfunction(f := _method(vars(owner)[name])):
        return None, (f, cls)
    hints = [vars(k).get("__annotations__", {}) for k in cls.__mro__]
    hint = next((h[name] for h in hints if name in h), None)
    return (_Instance(hint) if inspect.isclass(hint) else None), None


def _locals(fn: FunctionType, owner: type | None = None) -> dict[str, Any]:
    """What is known of a function's locals: the classes its parameters are annotated with, and ``self``."""
    hints = {k: _Instance(v) for k, v in getattr(fn, "__annotations__", {}).items() if inspect.isclass(v)}
    return hints | ({"self": _Instance(owner)} if owner is not None else {})


def _method(attr: Any) -> Any:
    """A class attribute as the function that runs: a method, or what a property computes."""
    if isinstance(attr, property):
        return attr.fget
    if isinstance(attr, cached_property):
        return attr.func
    return attr


def _codes(code: CodeType) -> Iterator[CodeType]:
    yield code
    for c in code.co_consts:
        if isinstance(c, CodeType):
            yield from _codes(c)


def _is_eval(fn: Any) -> bool:
    return inspect.isfunction(fn) and (fn.__module__ or "").startswith(f"{EVAL}.") and not fn.__name__.startswith("_")


def _key(fn: Callable[..., Any]) -> str:
    return f"{_short(fn.__module__)}.{fn.__name__}"


def _short(module: str) -> str:
    return module.removeprefix(f"{EVAL}.")
