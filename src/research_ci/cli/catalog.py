"""What `rci methods --markdown` lists, read from the code: every stage's floor list with its default list where the
floor marks it, each step with its role, what it needs, and the eval functions it reaches; and every public function
of the eval library with where it is used and what backs it."""

import inspect
import pkgutil
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from types import CodeType, FunctionType, ModuleType
from typing import Any

import research_ci.eval
from research_ci.runner import catalog
from research_ci.runner import evals as case_evals
from research_ci.stages import steps
from research_ci.stages.defaults import contract as contract_defaults
from research_ci.stages.defaults import data as data_defaults
from research_ci.stages.defaults import feat as feat_defaults
from research_ci.stages.floor import contract as contract_floor
from research_ci.stages.floor import data as data_floor
from research_ci.stages.floor import feat as feat_floor
from research_ci.stages.floor import source as source_floor

EVAL = "research_ci.eval"
STAGES = "research_ci.stages"
ORDER = ("protocol", "data", "calibration", "statistics", "structure", "synthetic", "obligations")
UNDOCUMENTED = {
    "protocol.deterministic": "Two identical consumer runs must give the same net.",
    "protocol.max_drawdown": "The consumer's deepest fall in equity from its running peak.",
    "protocol.total": "The consumer's net summed over the run.",
    "protocol.truncated": "The field table cut to the rows known by `t`.",
    "data.invariants.hedge_check": "The card's hedged-legs line, naming the pairs most often apart.",
    "data.treatments.apply_candidates": "The candidate table after every candidate treatment, in order.",
    "data.treatments.drop_entities": "Drop every row of the named entities.",
}


def collect() -> dict[str, Any]:
    stages = [
        _stage("contract", "rci contract derive", contract_floor, contract_defaults),
        _stage("data", "rci data check", data_floor, data_defaults, admission=source_floor),
        _stage("feat", "rci feat submit", feat_floor, feat_defaults),
    ]
    cases = {
        "about": catalog.CASE_EVAL,
        "example": catalog.CASE_EXAMPLE,
        "layers": list(case_evals.LAYERS),
        "declarations": catalog.case_evals(),
        "outcomes": dict(case_evals.OUTCOMES),
    }
    return {"stages": stages, "roles": dict(steps.ROLES), "cases": cases, "library": library(stages)}


def library(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every public function of every eval module, by package, with where it is used and what backs it."""
    by_stage: dict[str, list[str]] = {}
    for s in stages:
        for _, rows in s["sections"]:
            for row in rows:
                for m in row["methods"]:
                    by_stage.setdefault(m, []).append(s["stage"])
    research_code = {f"{_short(m['module'])}.{m['name']}" for m in catalog.methods()}
    modules = {p: list(catalog._modules(f"{EVAL}.{p}")) for p in packages()}
    inside, examples = _inside([m for ms in modules.values() for m in ms])
    secondary = (("inside eval", inside), ("worked examples", examples))
    out = []
    for p, ms in modules.items():
        entries = []
        for module in ms:
            backing = catalog._backing(module)
            functions = []
            for fn in public(module):
                key = _key(fn)
                used = list(dict.fromkeys(by_stage.get(key, [])))
                used += ["research code"] if key in research_code else []
                used = used or [tag for tag, keys in secondary if key in keys]
                used = used or ["algo, planned" if p == "protocol" else "tests only"]
                cases, props = backing.get(fn.__name__, (0, 0))
                doc = catalog.first_line(fn) or UNDOCUMENTED.get(key, "")
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
    """A stage's lists as its table shows them: the floor list with the default list where the floor marks it, and,
    for the data, the admission's list before them."""
    sections = []
    if admission is not None:
        sections.append(("admission, once per source", _rows(admission.STEPS, [], "floor")))
    sections.append(("", _rows(floor.STEPS, defaults.CHECKS, "floor")))
    files = [_path(floor), _path(defaults)] + ([_path(admission)] if admission is not None else [])
    return {"stage": name, "command": command, "files": files, "sections": sections}


def _rows(floor: Sequence[Any], defaults: Sequence[steps.Step[Any]], kind: str) -> list[dict[str, Any]]:
    out = []
    for item in floor:
        listed = [(item, kind)] if isinstance(item, steps.Step) else [(s, "default") for s in defaults]
        out += [
            {
                "item": s.name,
                "what": s.what,
                "role": s.role,
                "list": where,
                "needs": s.needs,
                "methods": reached(s.run),
            }
            for s, where in listed
        ]
    return out


def _path(module: ModuleType) -> str:
    return module.__name__.removeprefix("research_ci.").replace(".", "/") + ".py"


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
    """The functions ``fn`` names, in the order they come: through the functions of its own module it calls, and of
    the modules under ``through``, and the methods of their classes it names."""
    code = getattr(fn, "__code__", None)
    if not isinstance(code, CodeType):
        return []
    home = fn.__module__
    found: list[FunctionType] = []
    queue: list[tuple[CodeType, dict[str, Any]]] = [(code, fn.__globals__)]
    seen: set[CodeType] = set()
    while queue:
        top, scope = queue.pop(0)
        for c in _codes(top):
            for f in _named(c, scope, methods=bool(through)):
                if f not in found:
                    found.append(f)
                inner = f.__module__ == home or (through and (f.__module__ or "").startswith(f"{through}."))
                if inner and f.__code__ not in seen:
                    seen.add(f.__code__)
                    queue.append((f.__code__, f.__globals__))
    return found


def _named(code: CodeType, scope: dict[str, Any], methods: bool = False) -> list[FunctionType]:
    """The functions one code object names: globals, the values of global tables of functions, ``module.function``,
    and with ``methods``, the methods and properties of the classes its module defines."""
    out: list[FunctionType] = []
    modules = [
        m for n in code.co_names if isinstance(m := scope.get(n), ModuleType) and m.__name__.startswith("research_ci.")
    ]
    home = scope.get("__name__", "")
    classes = [c for c in scope.values() if inspect.isclass(c) and c.__module__ == home] if methods else []
    for n in code.co_names:
        value = scope.get(n)
        values = list(value.values()) if isinstance(value, dict) else value
        out += [f for f in (values if isinstance(values, list | tuple) else [value]) if inspect.isfunction(f)]
        out += [f for m in modules if inspect.isfunction(f := getattr(m, n, None)) and f.__module__ == m.__name__]
        out += [f for c in classes if inspect.isfunction(f := _method(vars(c).get(n)))]
    return list(dict.fromkeys(out))


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
