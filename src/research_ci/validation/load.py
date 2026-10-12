"""An idea's evals, loaded and validated before anything runs them.

An eval in ``EVALS`` is a general eval of the engine's library, trusted because the engine's CI holds it to its tests,
or a case eval, any other function: one of the idea's own, kept with its tests in its module. The tests of a case eval
are the cases of its module's ``CASES`` and the properties of its ``PROPERTIES`` that run it: a case whose ``run`` is
the eval, or reaches it through the functions of the module it calls or holds; a property whose ``of`` does. The engine
runs only those, never the idea's other code and never pytest. A case eval with no test is refused.

A validation is kept in the idea's ``.rci/evals/`` under the hash of what it rests on: the eval's module file and the
files of the idea and of installed labs it imports, the engine, and the interpreter. An eval whose files did not change
is not validated again.

The idea's files are loaded with the idea's directory, and the file's own, first on the import path, so ``data.py``
can import ``evals/<name>.py`` as ``from evals.<name> import <eval>``; the modules loaded from the idea are forgotten
afterwards, so two ideas never share one.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any, Literal

from research_ci.api.contract import ContractError
from research_ci.api.evals import Case, Eval, Power, Property, Recovers, Size, unwrapped
from research_ci.kernel.hashing import hash_obj
from research_ci.kernel.verdict.check import Check
from research_ci.runner import research
from research_ci.runner.plugins import Registry
from research_ci.store.environment import judging_deps
from research_ci.store.workspace import Workspace, hash_file
from research_ci.validation.tests import run_tests

LIBRARY = "research_ci.eval"
CACHE = "evals"
PIT_TESTS = ("built-in ", "pit ", "pits: ")
Source = Literal["library", "case"]
Status = Literal["valid", "invalid", "trusted"]
CASE_FORMAT = (
    "a case eval's module holds its tests next to it: CASES = [Case(name, eval, expect, given=frame), ...] (required) "
    "and PROPERTIES = [Size(eval), Power(eval), Recovers(eval, market, expect, tol)] (optional), from research_ci.api"
)


OUTCOMES: dict[str, str] = {
    "trusted": "an eval of the engine's library: its tests run in the engine's CI, never at research time; it runs",
    "valid n/n": "a case eval that passed all n of its tests: it runs, and its line on the card says so",
    "invalid": (
        "a case eval that failed a test, or a data eval that missed a pit of what it covers: it does not run and "
        "counts as no evidence; its line names the test, what it expected and what it got, and how to fix it. A data "
        "check with one is blocked; a feat submission shows it with no numbers"
    ),
    "no tests": (
        "a case eval no case or property of its module runs: refused before anything runs, saying where CASES go"
    ),
}
LAYERS: tuple[dict[str, str], ...] = (
    {
        "layer": "data",
        "file": "data.py",
        "gets": "the explore data, a FieldTable",
        "returns": "the rows it flags: entity, event_time, optionally field",
        "wraps": "on_fail (mask, drop_entities, ask), covers",
        "validated": "its CASES and PROPERTIES; every pit of what it covers that moves the decisive number",
    },
    {
        "layer": "feat",
        "file": "feat.py",
        "gets": "the evaluation table of one horizon",
        "returns": "t, value per decision time",
        "wraps": "label, horizons, combine (max, each)",
        "validated": "its CASES and PROPERTIES",
    },
)


class EvalsError(ContractError):
    pass


@dataclass(frozen=True)
class Validation:
    """How an eval stands before it runs: ``trusted`` (the engine's library), ``valid`` (a case eval that passed
    every test), or ``invalid`` (one failed: it does not run, and counts as no evidence). ``tests`` are its tests,
    each with what it expected and got when it failed; ``file`` is the case eval's module, as the cards name it."""

    status: Status
    file: str = ""
    tests: tuple[Check, ...] = field(default_factory=tuple)
    cached: bool = False

    @property
    def passed(self) -> int:
        return sum(t.status != "fail" for t in self.tests)

    @property
    def failed(self) -> list[Check]:
        return [t for t in self.tests if t.status == "fail"]

    def summary(self) -> str:
        """One line for a card: ``engine library``, ``valid 3/3``, or the first failed test and what it got."""
        count = f"{self.passed}/{len(self.tests)}"
        if self.status == "trusted":
            return f"engine library, {count} pits" if self.tests else "engine library"
        if self.status == "valid":
            return f"valid {count}"
        first = self.failed[0]
        more = f" (and {len(self.failed) - 1} more)" if len(self.failed) > 1 else ""
        return f"invalid {count}: {first.name}: {first.detail}{more}"

    def fix(self) -> str:
        """What an invalid eval's author does next: fix the eval or its cases, or what it claims to cover."""
        failed = [t.name for t in self.failed]
        if failed and all(n.startswith(PIT_TESTS) for n in failed):
            return "make it catch the pits it claims in covers, or claim less"
        where = f" in {self.file}" if self.file else ""
        check = f"; `rci eval check {self.file}` runs them for free" if self.file else ""
        return f"fix the eval or its CASES{where}{check}"

    def with_tests(self, tests: Sequence[Check]) -> "Validation":
        """The same with more tests, such as the pits a data eval covers; one that fails makes it invalid."""
        status: Status = "invalid" if any(t.status == "fail" for t in tests) else self.status
        return Validation(status if self.status != "invalid" else "invalid", self.file, (*self.tests, *tests))

    def record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "file": self.file,
            "tests": [asdict(t) for t in self.tests],
            "summary": self.summary(),
            "fix": self.fix() if self.status == "invalid" else "",
        }


def source_of(fn: Callable[..., Any]) -> Source:
    """An eval of the engine's library, or a case eval."""
    module = str(getattr(unwrapped(fn), "__module__", "") or "")
    return "library" if module == LIBRARY or module.startswith(f"{LIBRARY}.") else "case"


def file_of(fn: Callable[..., Any]) -> Path | None:
    """The file of the module a function is defined in."""
    code = getattr(unwrapped(fn), "__code__", None)
    return Path(code.co_filename).resolve() if isinstance(code, CodeType) else None


def tests_of(fn: Callable[..., Any]) -> tuple[list[Case], list[Property]]:
    """The cases and properties of the function's module that run it."""
    base = unwrapped(fn)
    scope = getattr(base, "__globals__", None)
    if scope is None:
        return [], []
    cases, properties = scope.get("CASES", []), scope.get("PROPERTIES", [])
    where = _shown_file(base)
    if not isinstance(cases, list | tuple) or not all(isinstance(c, Case) for c in cases):
        raise EvalsError(f"CASES in {where} must be a list of Case; {CASE_FORMAT}")
    if not isinstance(properties, list | tuple) or not all(isinstance(p, Size | Power | Recovers) for p in properties):
        raise EvalsError(f"PROPERTIES in {where} must be a list of Size, Power, or Recovers; {CASE_FORMAT}")
    return [c for c in cases if _runs(c.run, base)], [p for p in properties if _runs(p.of, base)]


def refuse_untested(evals: Sequence[Eval], root: Path, layer_file: str, after: str = "") -> None:
    """Refuses, before anything runs, a case eval with no test, saying where its tests go."""
    for e in evals:
        if source_of(e.of) != "case":
            continue
        cases, properties = tests_of(e.of)
        if cases or properties:
            continue
        where = _relative(file_of(e.of), root) or layer_file
        raise EvalsError(
            f"{layer_file}: the case eval {e.name!r} has no tests, so it cannot run: write CASES in {where} with at "
            f"least one Case that runs it, such as Case('<what it shows>', {_name(e.of)}, <expected>, given=<frame>); "
            f"{CASE_FORMAT}; `rci eval check {where}` runs them{after}"
        )


def validate(
    evals: Sequence[Eval], root: Path, registry: Registry | None = None, cache: bool = True
) -> dict[str, Validation]:
    """Every eval's validation by name: a library eval is trusted; a case eval is validated by its tests, or the
    validation kept for files that did not change is used."""
    out: dict[str, Validation] = {}
    engine: dict[str, str] = {}
    for e in evals:
        if source_of(e.of) == "library":
            out[e.name] = Validation("trusted")
            continue
        engine = engine or judging_deps()
        out[e.name] = _validated(e, Path(root), registry, engine, cache)
    return out


def checked_file(path: Path, root: Path, registry: Registry | None = None) -> tuple[list[tuple[str, Validation]], str]:
    """Every public function of a module, as an eval, with its validation, run now and kept nowhere; and what is wrong
    with the module as a whole, if anything."""
    path, root = Path(path).resolve(), Path(root).resolve()
    module = load_module(path, root)
    functions = [
        fn
        for name, fn in inspect.getmembers(module, inspect.isfunction)
        if fn.__module__ == module.__name__ and not name.startswith("_")
    ]
    problem = "" if hasattr(module, "CASES") else f"{_relative(path, root)} defines no CASES; {CASE_FORMAT}"
    out = []
    for fn in functions:
        cases, properties = tests_of(fn)
        if not cases and not properties:
            out.append((fn.__name__, Validation("invalid", _relative(path, root), ())))
            continue
        out.append((fn.__name__, _run(fn.__name__, cases, properties, _relative(path, root))))
    return out, problem


class _Fresh(importlib.machinery.SourceFileLoader):
    """Runs a file of the idea from its source as it is now, never from bytecode cached for an earlier version."""

    def get_code(self, fullname: str) -> CodeType:
        return compile(self.get_data(self.path), self.path, "exec", dont_inherit=True)


class _IdeaFinder(importlib.abc.MetaPathFinder):
    """Finds the idea's modules on the import path as usual, and runs them with ``_Fresh``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        if spec is None or origin is None or not origin.is_relative_to(self.root) or origin.suffix != ".py":
            return None
        spec.loader = _Fresh(fullname, str(origin))
        return spec


@contextmanager
def idea_imports(root: Path, *dirs: Path) -> Iterator[None]:
    """The idea's directory, and ``dirs``, first on the import path while the idea's files load, run from their
    source as it is now; the modules loaded from the idea are forgotten afterwards."""
    root = Path(root).resolve()
    paths = [str(p) for p in dict.fromkeys([*(Path(d).resolve() for d in dirs), root])]
    before = set(sys.modules)
    finder = _IdeaFinder(root)
    sys.path[:0] = paths
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for p in paths:
            sys.path.remove(p)
        for name in set(sys.modules) - before:
            found = getattr(sys.modules.get(name), "__file__", None)
            if found and Path(found).resolve().is_relative_to(root):
                del sys.modules[name]
        importlib.invalidate_caches()


def load_module(path: Path, root: Path) -> ModuleType:
    """A file of the idea, run as a module from its source with the idea's directory and its own first on the import
    path. Raises ``EvalsError`` naming the file when it does not load."""
    path = Path(path).resolve()
    module_name = f"_rci_{abs(hash(str(path)))}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path, loader=_Fresh(module_name, str(path)))
    if spec is None or spec.loader is None:
        raise EvalsError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with idea_imports(root, path.parent):
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise EvalsError(f"{_relative(path, root) or path} failed to load: {type(exc).__name__}: {exc}") from exc
    return module


def _validated(e: Eval, root: Path, registry: Registry | None, engine: dict[str, str], cache: bool) -> Validation:
    path = file_of(e.of)
    shown = _relative(path, root)
    files = research.reached([path], root, registry) if path is not None else {}
    key = hash_obj(
        {
            "eval": getattr(unwrapped(e.of), "__qualname__", e.name),
            "files": {name: hash_file(p) for name, p in sorted(files.items())},
            "engine": engine.get("engine", ""),
            "python": engine.get("python", ""),
        }
    )
    store = Workspace(root)
    rel = f"{CACHE}/{key[:24]}.json"
    kept = store.read_json(rel) if cache else None
    if kept is not None and kept.get("key") == key:
        tests = tuple(Check(**t) for t in kept["tests"])
        return Validation(kept["status"], shown, tests, cached=True)
    cases, properties = tests_of(e.of)
    found = _run(e.name, cases, properties, shown)
    if cache:
        store.write_json(rel, {"key": key, "eval": e.name, "status": found.status, "tests": found.record()["tests"]})
    return found


def _run(name: str, cases: Sequence[Case], properties: Sequence[Property], shown: str) -> Validation:
    checks = run_tests(name, cases, properties)
    named = [f"case {c.name!r}" for c in cases] + [c.name.removeprefix(f"{name}: ") for c in checks[len(cases) :]]
    tests = tuple(Check(n, c.status, c.detail) for n, c in zip(named, checks, strict=True))
    return Validation("invalid" if any(t.status == "fail" for t in tests) else "valid", shown, tests)


def _runs(run: Any, target: Any) -> bool:
    """Whether calling ``run`` can reach ``target``: it is ``target``, or names it or holds it, directly or through the
    functions of ``target``'s module it calls, and the partials and closures it holds."""
    scope = getattr(target, "__globals__", None)
    pending, seen = [run], set()
    while pending:
        obj = pending.pop()
        if isinstance(obj, partial):
            pending += [obj.func, *obj.args, *obj.keywords.values()]
            continue
        if obj is target:
            return True
        code = getattr(obj, "__code__", None)
        if not isinstance(code, CodeType) or code in seen:
            continue
        seen.add(code)
        own = getattr(obj, "__globals__", {})
        held = [c.cell_contents for c in getattr(obj, "__closure__", None) or () if _filled(c)]
        named = [own.get(n) for c in _codes(code) for n in c.co_names]
        for value in [*held, *named]:
            values = list(value.values()) if isinstance(value, dict) else value
            for v in values if isinstance(values, list | tuple) else [value]:
                if v is target or isinstance(v, partial) or getattr(v, "__globals__", None) is scope:
                    pending.append(v)
    return False


def _filled(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:
        return False
    return True


def _codes(code: CodeType) -> Iterator[CodeType]:
    yield code
    for c in code.co_consts:
        if isinstance(c, CodeType):
            yield from _codes(c)


def _relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def _shown_file(fn: Any) -> str:
    found = file_of(fn)
    return found.name if found is not None else "its module"


def _name(fn: Any) -> str:
    return str(getattr(unwrapped(fn), "__name__", "the eval"))
