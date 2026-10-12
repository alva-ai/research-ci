"""What an agent can use without reading the engine's source: every installed plugin with its options and
declarations (``rci plugins``), and the general evals of the engine's library an idea's ``EVALS`` can list or its case
evals can call, each with the worked examples and properties that back it, how a case eval and its tests are written,
and how a null world is written (``rci methods``)."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import MISSING, asdict, fields, is_dataclass
from importlib.metadata import entry_points
from types import CodeType, ModuleType
from typing import Any

from research_ci.api.evals import FORMAT, Case, Eval, Market, Power, Recovers, Size, Source
from research_ci.api.nulls import Null, PermuteWithin
from research_ci.api.plugins import ENTRY_POINTS
from research_ci.runner import plugins as loader
from research_ci.runner.plugins import Registry, annotation, signature

KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("data", ("fetch",)),
    ("universe", ("candidates",)),
    ("label", ("values", "cases")),
    ("target", ("reduce",)),
    ("cost", ("round_trip", "fees")),
    ("consumer", ("timing", "run")),
)
METHODS: tuple[tuple[str, str, str], ...] = (
    ("feat eval", "research_ci.eval.statistics", f"{FORMAT['feat']}"),
    ("feat eval", "research_ci.eval.structure", ""),
    ("data eval", "research_ci.eval.data.sanity", f"{FORMAT['data']}"),
    ("data eval", "research_ci.eval.data.coverage", ""),
    ("data eval", "research_ci.eval.data.identity", ""),
    (
        "pit",
        "research_ci.eval.data.pits",
        "nulls/data.py: PITS = [Pit(category, plant)]; plant(fields, field, rng) returns a copy with the pit planted",
    ),
)
CASE_EVAL = (
    "a function of the idea's own, in a module such as `evals/<name>.py`, with its tests next to it: CASES (required) "
    "and PROPERTIES (optional); data.py or feat.py imports it into EVALS (`from evals.<name> import <eval>`). Before "
    "anything runs, the engine runs its tests, the cases and properties that run it, and only a case eval that passes "
    "them all runs; one without tests is refused, an invalid one is shown with the test it failed and no result. "
    "`rci eval check <file>` runs them any time, for free; a validation is kept until the module's files change"
)
CASE_EXAMPLE = (
    'CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)], where SIX is a polars '
    "DataFrame of the evaluation table (t, entity, score, y) and thirds(SIX) returns t, value"
)
CASE_FIELDS: dict[type, str] = {
    Eval: "an entry of EVALS with what its layer reads: on_fail, covers (data.py); label, horizons, combine (feat.py)",
    Case: "",
    Size: "",
    Power: "",
    Recovers: "",
    Market: "",
    Source: "",
}

NULLS_FORMAT = (
    "nulls/feat.py: NULLS = [Null(explains, world)]; give each alternative of the contract one, or the ResultCard "
    "marks it uncovered"
)
NULLS_EXAMPLE = 'NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"))]'
NULL_FIELDS: dict[type, dict[str, str]] = {
    Null: {
        "explains": "the id of the contract's alternative this world answers",
        "world": "a PermuteWithin: how the scores are shuffled while the labels stay",
        "keeps": "the column the alternative lives in; None means the world's bucket_by",
        "evals": "the names of the evals this world judges; () means every eval of the layer, defaults included",
        "reps": "how many shuffles, at least 20; 50 by default",
        "note": "free text kept with the world in the submission's record",
    },
    PermuteWithin: {
        "bucket_by": 'shuffle the scores across entities inside each bucket of this column; "time" is one bucket per '
        "decision time",
        "clusters": "a column whose members move together and are shuffled as one",
        "block": 'repeat the same shuffle for every decision time inside a block this long; "horizon" uses each '
        "label horizon",
    },
}


def plugins(registry: Registry | None = None) -> list[dict[str, Any]]:
    """Every plugin ``load`` can find, by name: the registry's (or the test registry's), then the entry points'."""
    local = registry if registry is not None else loader.EXTRA
    found: dict[str, tuple[str, Callable[[], Any]]] = {
        ep.name: (f"entry point of {ep.dist.name if ep.dist else 'an unknown distribution'}", ep.load)
        for ep in entry_points(group=ENTRY_POINTS)
    }
    found |= {name: ("registry", lambda f=factory: f) for name, factory in local.items()}
    return [_plugin(name, origin, get) for name, (origin, get) in sorted(found.items())]


def methods() -> list[dict[str, Any]]:
    """The public functions of the modules research code may call, with what backs each."""
    out = []
    for use, name, _ in METHODS:
        for module in _modules(name):
            backing = _backing(module)
            for fname, fn in inspect.getmembers(module, inspect.isfunction):
                if fn.__module__ != module.__name__ or fname.startswith("_"):
                    continue
                cases, props = backing.get(fname, (0, 0))
                out.append(
                    {
                        "use": use,
                        "module": module.__name__,
                        "name": fname,
                        "import": f"from {module.__name__} import {fname}",
                        "signature": _signature(fn),
                        "doc": first_line(fn),
                        "examples": cases,
                        "properties": props,
                    }
                )
    return out


def nulls() -> list[dict[str, Any]]:
    """How a null world is written: ``Null`` and ``PermuteWithin``, with what each field means."""
    return [
        {
            "use": "null world",
            "module": "research_ci.api",
            "name": cls.__name__,
            "import": f"from research_ci.api import {cls.__name__}",
            "signature": _signature(cls).removesuffix(" -> None").replace(f"{Null.__module__}.", ""),
            "doc": first_line(cls),
            "fields": [{"name": name, "means": means} for name, means in fields.items()],
        }
        for cls, fields in NULL_FIELDS.items()
    ]


def formats() -> dict[str, str]:
    """How each kind of research code calls the methods listed for it."""
    return {use: text for use, _, text in METHODS if text}


def case_evals() -> list[dict[str, Any]]:
    """How a case eval is written: ``Eval``, and the tests a case eval's module declares, from ``api/evals.py``."""
    return [
        {
            "use": "case eval",
            "module": "research_ci.api",
            "name": cls.__name__,
            "import": f"from research_ci.api import {cls.__name__}",
            "signature": _fields(cls),
            "doc": doc or first_line(cls),
        }
        for cls, doc in CASE_FIELDS.items()
    ]


def _fields(cls: type) -> str:
    """A dataclass's fields as it is called: required ones by name, the others with their defaults."""
    shown = []
    for f in fields(cls):
        if f.default is not MISSING:
            shown.append(f"{f.name}={f.default!r}")
        elif f.default_factory is not MISSING:
            shown.append(f"{f.name}={type(f.default_factory()).__name__}()")
        else:
            shown.append(f.name)
    return f"({', '.join(shown)})"


def first_line(obj: Any) -> str:
    """The first sentence of an object's docstring, on one line."""
    doc = (inspect.getdoc(obj) or "").replace("``", "`")
    para = " ".join(doc.split("\n\n")[0].split())
    end = para.find(". ")
    return para if end < 0 else para[: end + 1]


def _plugin(name: str, origin: str, get: Callable[[], Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "origin": origin}
    try:
        factory = get()
    except Exception as exc:  # noqa: BLE001
        return out | {"kind": "error", "error": f"{type(exc).__name__}: {exc}"}
    cls = factory if inspect.isclass(factory) else None
    kinds = [k for k, attrs in KINDS if cls is not None and all(callable(getattr(cls, a, None)) for a in attrs)]
    out["kind"] = "/".join(kinds) or "unknown"
    out["doc"] = first_line(factory)
    out["signature"] = signature(name, factory)
    out["options"] = _options(factory)
    if cls is None:
        return out
    if "data" in kinds:
        out["fields"] = list(getattr(cls, "fields", None) or [])
        levels = getattr(cls, "levels", None)
        out["levels"] = asdict(levels) if is_dataclass(levels) and not isinstance(levels, type) else None
        out["reference"] = callable(getattr(cls, "reference", None))
    if "target" in kinds:
        out["per"] = getattr(cls, "per", None)
        out["rolls"] = bool(getattr(cls, "rolls", False))
    return out


def _options(factory: Callable[..., Any]) -> list[dict[str, Any]]:
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return []
    kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return [
        {
            "name": p.name,
            "type": None if p.annotation is inspect.Parameter.empty else annotation(p.annotation),
            "required": p.default is inspect.Parameter.empty,
            "default": None if p.default is inspect.Parameter.empty else repr(p.default),
        }
        for p in params
        if p.kind not in kinds
    ]


def _modules(name: str) -> Iterator[ModuleType]:
    module = importlib.import_module(name)
    if not hasattr(module, "__path__"):
        yield module
        return
    for info in sorted(pkgutil.iter_modules(module.__path__), key=lambda m: m.name):
        if not info.ispkg:
            yield importlib.import_module(f"{name}.{info.name}")


def _signature(fn: Callable[..., Any]) -> str:
    sig = inspect.signature(fn)
    params = []
    for p in sig.parameters.values():
        text = p.name
        if p.annotation is not inspect.Parameter.empty:
            text += f": {annotation(p.annotation)}"
        if p.default is not inspect.Parameter.empty:
            text += f" = {p.default!r}"
        params.append(text)
    back = "" if sig.return_annotation is inspect.Signature.empty else f" -> {annotation(sig.return_annotation)}"
    return f"({', '.join(params)}){back}"


def _backing(module: ModuleType) -> dict[str, tuple[int, int]]:
    """For each function of ``module``, how many of its ``CASES`` and ``PROPERTIES`` run it, directly or through the
    module's own helpers and tables of functions."""
    counts: dict[str, list[int]] = {}
    for i, items in enumerate((getattr(module, "CASES", []), getattr(module, "PROPERTIES", []))):
        for item in items:
            run = getattr(item, "run", None) or getattr(item, "of", None)
            for name in _reached(run, module):
                counts.setdefault(name, [0, 0])[i] += 1
    return {name: (c[0], c[1]) for name, c in counts.items()}


def _reached(fn: Any, module: ModuleType) -> set[str]:
    """The functions of ``module`` that calling ``fn`` can reach by name."""
    own = {
        name: obj for name, obj in vars(module).items() if inspect.isfunction(obj) and obj.__module__ == module.__name__
    }
    reached: set[str] = set()
    start = getattr(fn, "__name__", "")
    if start in own:
        reached.add(start)
    stack: list[CodeType] = [fn.__code__] if hasattr(fn, "__code__") else []
    while stack:
        code = stack.pop()
        stack += [c for c in code.co_consts if isinstance(c, CodeType)]
        for name in code.co_names:
            value = vars(module).get(name)
            values = list(value.values()) if isinstance(value, dict) else value
            found = values if isinstance(values, list | tuple) else [value]
            for obj in found:
                key = next((k for k, v in own.items() if v is obj), None)
                if key is not None and key not in reached:
                    reached.add(key)
                    stack.append(own[key].__code__)
    return reached
