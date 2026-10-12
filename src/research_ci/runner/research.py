"""Research code is free: pandas, loops, and any library run. Before the engine runs a features file, an idea's
criteria and null worlds, or its data tests and pits, it reads them, and the modules of the idea and of the installed
labs they import, without running anything, and finds code that goes row by row: polars ``iter_rows``, ``rows``,
``row``, ``rows_by_key``, ``to_dicts``, ``map_elements``, ``map_rows``; pandas ``iterrows``, ``itertuples``,
``apply``, ``applymap``, and ``map`` with a function; ``while`` loops; and ``for`` loops and comprehensions over
anything but a small list. What it finds is one warning on the card, with each file and line; it never refuses and
costs nothing. Only a file that does not parse is refused, since it could not run either.

A small list is a display such as ``HORIZONS = ("8h", "24h")`` or a list of expressions, a comprehension over one, a
frame's column names or schema, ``range`` of a literal size, sequences of these (``zip``, ``enumerate``, ``sorted``,
``itertools.product``, ``+``), and a name that every binding in its scope binds to one of these.

A lab is a distribution that registers plugins under ``research_ci.plugins``. A plugin's own module is not research
code: protocol tests hold it, and the contract pins its code; the reading stops there. Third-party libraries and the
engine are never read.
"""

import ast
import importlib.util
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from research_ci.api.contract import ContractError
from research_ci.api.plugins import ENTRY_POINTS
from research_ci.runner import plugins as loader
from research_ci.runner.plugins import Registry

EXAMPLE_MARKER = "rci:example"
ENGINE = "research_ci"
ROW_APIS = frozenset(
    {
        "iter_rows",
        "rows",
        "row",
        "rows_by_key",
        "to_dicts",
        "map_elements",
        "map_rows",
        "iterrows",
        "itertuples",
        "apply",
        "applymap",
    }
)
BUILTIN_FUNCTIONS = frozenset(
    {"abs", "bool", "float", "format", "hash", "int", "len", "max", "min", "repr", "round", "str", "sum", "type"}
)
SEQUENCES = frozenset({"zip", "enumerate", "sorted", "reversed", "list", "tuple", "set", "frozenset", "dict"})
COMBINATIONS = frozenset({"product", "combinations", "permutations"})
METADATA = frozenset({"columns", "schema", "dtypes"})
SPLITS = frozenset({"split", "rsplit"})
SHOWN = 8
FASTER = "polars expressions are usually much faster"


class ResearchError(ContractError):
    pass


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    what: str
    file: str = ""

    def __str__(self) -> str:
        return f"{self.file or self.path.name}:{self.line} {self.what}"


def warning(paths: Iterable[Path], idea: Path, registry: Registry | None = None) -> str:
    """The card's warning: every place the files, or the research modules they import, go row by row, with its file
    and line; empty when none does."""
    found = findings(paths, idea, registry)
    if not found:
        return ""
    parts, last = [], ""
    for f in found[:SHOWN]:
        parts.append(f"{'' if f.file == last else f.file}:{f.line} {f.what}")
        last = f.file
    more = f" · {len(found) - SHOWN} more" if len(found) > SHOWN else ""
    return f"row-wise code: {' · '.join(parts)}{more} — {FASTER}"


def still_example(path: Path) -> bool:
    return Path(path).is_file() and EXAMPLE_MARKER in Path(path).read_text()


def findings(paths: Iterable[Path], idea: Path, registry: Registry | None = None) -> list[Finding]:
    """What the files, and the modules of the idea and of installed labs they import, do row by row. A file that does
    not parse is refused."""
    lab = _Lab.installed(registry)
    idea = Path(idea).resolve()
    out: list[Finding] = []
    for path, tree in _walked(paths, idea, lab):
        out += _Scan(tree).run(path, _shown(path, idea, lab))
    return out


def reached(paths: Iterable[Path], idea: Path, registry: Registry | None = None) -> dict[str, Path]:
    """The files themselves and the modules of the idea and of installed labs they import, as the cards name them,
    found without running anything. A file that does not parse is refused."""
    lab = _Lab.installed(registry)
    idea = Path(idea).resolve()
    return {_shown(path, idea, lab): path for path, _ in _walked(paths, idea, lab)}


def shown(path: Path, idea: Path, registry: Registry | None = None) -> str:
    """A file as the cards name it: from the idea's directory, or from an installed lab's top package."""
    return _shown(Path(path).resolve(), Path(idea).resolve(), _Lab.installed(registry))


def _walked(paths: Iterable[Path], idea: Path, lab: "_Lab") -> Iterable[tuple[Path, ast.Module]]:
    queue = [Path(p).resolve() for p in paths if Path(p).is_file()]
    seen: set[Path] = set()
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            where = _shown(path, idea, lab)
            raise ResearchError(f"{where}:{exc.lineno}: {exc.msg}; fix the file and run the command again") from exc
        yield path, tree
        queue += [p for p in _imported(tree, path, idea, lab) if p not in seen]


@dataclass(frozen=True)
class _Lab:
    """The top-level packages of the installed labs, and the files of their plugin modules, where reading stops."""

    tops: frozenset[str]
    plugins: frozenset[Path]

    @classmethod
    def installed(cls, registry: Registry | None) -> "_Lab":
        modules = {ep.value.split(":")[0].strip() for ep in entry_points(group=ENTRY_POINTS)}
        files: set[Path] = set()
        for factory in (registry if registry is not None else loader.EXTRA).values():
            modules.add(getattr(factory, "__module__", "") or "")
            try:
                files.add(Path(inspect.getfile(factory)).resolve())
            except TypeError:
                pass
        tops = frozenset(m.split(".")[0] for m in modules if m and m.split(".")[0] != ENGINE)
        files |= {found[-1] for m in modules if m.split(".")[0] in tops and (found := _installed(m))}
        return cls(tops, frozenset(files))


def _imported(tree: ast.Module, path: Path, idea: Path, lab: _Lab) -> list[Path]:
    out: list[Path] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out += _resolve(alias.name.split("."), path, idea, lab)
        elif isinstance(node, ast.ImportFrom) and node.level:
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            parts = node.module.split(".") if node.module else []
            out += _walk(base, parts, lab) + [f for a in node.names for f in _walk(base, [*parts, a.name], lab)]
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            out += _resolve(parts, path, idea, lab)
            out += [f for a in node.names for f in _resolve([*parts, a.name], path, idea, lab, submodule=True)]
    return out


def _resolve(parts: list[str], path: Path, idea: Path, lab: _Lab, submodule: bool = False) -> list[Path]:
    """The files importing ``parts`` runs, from the idea's directories first, then from an installed lab; with
    ``submodule``, only when the last part is a module and not a name inside one."""
    for root in dict.fromkeys((path.parent, idea)):
        if (root / parts[0]).is_dir() or (root / f"{parts[0]}.py").is_file():
            found = _files(root, parts)
            return [] if submodule and not _complete(root, parts) else _until_plugin(found, lab)
    if parts[0] not in lab.tops:
        return []
    found = _installed(".".join(parts))
    return [] if submodule and len(found) < len(parts) else _until_plugin(found, lab)


def _walk(root: Path, parts: Sequence[str], lab: _Lab) -> list[Path]:
    return _until_plugin(_files(root, parts), lab)


def _files(root: Path, parts: Sequence[str]) -> list[Path]:
    """The files importing ``parts`` from ``root`` runs: each package's ``__init__.py`` on the way, then the module."""
    out = []
    for part in parts:
        if (root / part).is_dir():
            root = root / part
            if (root / "__init__.py").is_file():
                out.append((root / "__init__.py").resolve())
        elif (root / f"{part}.py").is_file():
            out.append((root / f"{part}.py").resolve())
            break
        else:
            break
    return out


def _until_plugin(files: list[Path], lab: _Lab) -> list[Path]:
    """Reading stops at a plugin's own module."""
    kept = []
    for file in files:
        if file in lab.plugins:
            break
        kept.append(file)
    return kept


def _complete(root: Path, parts: Sequence[str]) -> bool:
    *head, last = parts
    base = root.joinpath(*head)
    return (base / last).is_dir() or (base / f"{last}.py").is_file()


def _installed(dotted: str) -> list[Path]:
    """The files of each package on the way to the installed module ``dotted``, found without importing anything."""
    parts = dotted.split(".")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    top = [Path(spec.origin).resolve()] if spec.origin and spec.origin.endswith(".py") else []
    for location in spec.submodule_search_locations or []:
        found = _files(Path(location), parts[1:])
        if found:
            return top + found
    return top


Key = tuple[int, str]


class _Scan:
    """One file's row-by-row code. A loop may go over a small list; a name is what every binding in the scope it
    resolves to, by Python's rules, binds it to."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
        self.scope_of: dict[int, ast.AST] = {}
        self.parent: dict[int, ast.AST | None] = {id(tree): None}
        self.declared: dict[int, dict[str, str]] = {}
        self.bindings: dict[Key, list[Bound]] = {}
        self.literal_keys: frozenset[Key] = frozenset()
        self.small_keys: frozenset[Key] = frozenset()
        self._visit(tree, tree)
        self.literal_keys = self._greatest(self._literal, "literal_keys")
        self.small_keys = self._greatest(self._iterable, "small_keys")

    def run(self, path: Path, shown: str) -> list[Finding]:
        out = []
        for node in ast.walk(self.tree):
            what = self._what(node)
            if what:
                out.append(Finding(path, getattr(node, "lineno", 0), what, shown))
        return sorted(out, key=lambda f: f.line)

    def _what(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in ROW_APIS:
                return attr
            if attr == "map" and self._function(next(iter(node.args), None)):
                return "map with a function"
        if isinstance(node, ast.For | ast.AsyncFor) and not self._iterable(node.iter) and not _row_api(node.iter):
            return f"for over {_text(node.iter)}"
        if isinstance(node, ast.While):
            return "while loop"
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            for gen in node.generators:
                if not self._iterable(gen.iter) and not _row_api(gen.iter):
                    return f"comprehension over {_text(gen.iter)}"
        return ""

    def _function(self, node: ast.expr | None) -> bool:
        if isinstance(node, ast.Lambda | ast.Attribute):
            return True
        return isinstance(node, ast.Name) and (node.id in self.functions or node.id in BUILTIN_FUNCTIONS)

    def _iterable(self, node: "Bound") -> bool:
        """What a loop may go over, a small list: a literal, a display of anything, a comprehension over small lists,
        column names, names bound only to these, sequences and sums of these, and an element of a list of them."""
        if isinstance(node, _Element) or node is None:
            if self._literal(node):
                return True
            members = self._members(node.of) if isinstance(node, _Element) else None
            return members is not None and all(self._iterable(m) for m in members)
        if self._literal(node) or isinstance(node, ast.List | ast.Tuple | ast.Set | ast.Dict):
            return True
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            return all(self._iterable(g.iter) for g in node.generators)
        if isinstance(node, ast.Name):
            return self._key(node) in self.small_keys
        if _metadata(node):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._iterable(node.left) and self._iterable(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left, right = node.left, node.right
            return (self._iterable(left) and self._literal(right)) or (self._literal(left) and self._iterable(right))
        if isinstance(node, ast.IfExp):
            return self._iterable(node.body) and self._iterable(node.orelse)
        if isinstance(node, ast.Call):
            name = _called(node)
            if name in SEQUENCES or name in COMBINATIONS:
                return bool(node.args) and all(self._iterable(a) for a in node.args)
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("items", "keys", "values"):
                return self._iterable(node.func.value)
        return False

    def _literal(self, node: "Bound") -> bool:
        """Built only from constants and names bound only to such values."""
        if isinstance(node, _Element):
            return node is not _UNKNOWN and node.of is not None and self._literal(node.of)
        if node is None or isinstance(node, ast.Constant | ast.JoinedStr):
            return True
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            return all(self._literal(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return all(self._literal(k) for k in node.keys) and all(self._literal(v) for v in node.values)
        if isinstance(node, ast.Name):
            return self._key(node) in self.literal_keys
        if isinstance(node, ast.Starred):
            return self._literal(node.value)
        if isinstance(node, ast.BinOp):
            return self._literal(node.left) and self._literal(node.right)
        if isinstance(node, ast.UnaryOp):
            return self._literal(node.operand)
        if isinstance(node, ast.IfExp):
            return self._literal(node.body) and self._literal(node.orelse)
        if isinstance(node, ast.Subscript):
            return self._literal(node.value) and self._literal(node.slice)
        if isinstance(node, ast.Slice):
            return all(self._literal(x) for x in (node.lower, node.upper, node.step))
        if isinstance(node, ast.Call):
            name = _called(node)
            args = [*node.args, *(k.value for k in node.keywords)]
            if name == "range" or name == "len" or name in SEQUENCES or name in COMBINATIONS:
                return all(self._literal(a) for a in args)
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("items", "keys", "values", *SPLITS):
                return self._literal(node.func.value) and all(self._literal(a) for a in args)
        return False

    def _members(self, node: "Bound") -> list[ast.expr] | None:
        """What going over ``node`` gives, when the file shows it: the elements of a display, the keys or the values
        of a dict display, through names bound only to displays and their sums."""
        values = isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "values"
        if values and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.args:
            node = node.func.value
        displays = self._displays(node, frozenset())
        if displays is None:
            return None
        out: list[ast.expr] = []
        for d in displays:
            if isinstance(d, ast.Dict):
                out += d.values if values else [k for k in d.keys if k is not None]
            elif values or not isinstance(d, ast.List | ast.Tuple | ast.Set):
                return None
            else:
                out += d.elts
        return out

    def _displays(self, node: "Bound", seen: frozenset[Key]) -> list[ast.expr] | None:
        if isinstance(node, ast.List | ast.Tuple | ast.Set | ast.Dict):
            return [node]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._displays(node.left, seen), self._displays(node.right, seen)
            return None if left is None or right is None else left + right
        if not isinstance(node, ast.Name):
            return None
        key = self._key(node)
        if key is None:
            return None
        if key in seen:
            return []
        out: list[ast.expr] = []
        for value in self.bindings[key]:
            found = self._displays(value, seen | {key})
            if found is None:
                return None
            out += found
        return out

    def _greatest(self, holds: Callable[["Bound"], bool], attr: str) -> frozenset[Key]:
        """The largest set of names whose every binding ``holds`` given the set, so that a name grown from itself,
        such as ``cols = cols + ["extra"]``, keeps what it was."""
        keys = frozenset(self.bindings)
        while True:
            setattr(self, attr, keys)
            kept = frozenset(k for k in keys if all(holds(v) for v in self.bindings[k]))
            if kept == keys:
                return keys
            keys = kept

    def _key(self, node: ast.Name) -> Key | None:
        """The binding a name resolves to: the innermost scope that binds it, skipping class bodies from inside them,
        as Python does; ``None`` for a name the file never binds."""
        scope = self.scope_of.get(id(node))
        first = True
        while scope is not None:
            kind = self.declared.get(id(scope), {}).get(node.id)
            if kind == "global":
                key = (id(self.tree), node.id)
                return key if key in self.bindings else None
            if (
                kind is None
                and (first or not isinstance(scope, ast.ClassDef))
                and (id(scope), node.id) in self.bindings
            ):
                return (id(scope), node.id)
            first = False
            scope = self.parent.get(id(scope))
        return None

    def _visit(self, node: ast.AST, scope: ast.AST) -> None:
        self.scope_of[id(node)] = scope
        for child, inner in self._children(node, scope):
            self._visit(child, inner)

    def _children(self, node: ast.AST, scope: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
        """``node``'s children, each with the scope it runs in, binding the names ``node`` binds on the way."""
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            self.parent[id(node)] = scope
            a = node.args
            params = [p for p in (*a.posonlyargs, *a.args, a.vararg, *a.kwonlyargs, a.kwarg) if p is not None]
            for p in params:
                self._bind(node, p.arg, _UNKNOWN)
            outer: list[ast.AST] = [*a.defaults, *(d for d in a.kw_defaults if d is not None)]
            inner: list[ast.AST] = [*params]
            if isinstance(node, ast.Lambda):
                inner.append(node.body)
            else:
                self._bind(scope, node.name, _UNKNOWN)
                outer += [*node.decorator_list, *([node.returns] if node.returns else [])]
                inner += node.body
            return [(c, scope) for c in outer] + [(c, node) for c in inner]
        if isinstance(node, ast.ClassDef):
            self.parent[id(node)] = scope
            self._bind(scope, node.name, _UNKNOWN)
            outer = [*node.decorator_list, *node.bases, *node.keywords]
            return [(c, scope) for c in outer] + [(c, node) for c in node.body]
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            self.parent[id(node)] = scope
            out: list[tuple[ast.AST, ast.AST]] = []
            for i, gen in enumerate(node.generators):
                self.scope_of[id(gen)] = node
                self._bind_target(node, gen.target, _Element(gen.iter))
                out += [(gen.iter, scope if i == 0 else node), (gen.target, node), *((c, node) for c in gen.ifs)]
            parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            return out + [(c, node) for c in parts]
        self._binds(node, scope)
        return [(c, scope) for c in ast.iter_child_nodes(node)]

    def _binds(self, node: ast.AST, scope: ast.AST) -> None:
        """The names a statement or an expression binds in ``scope``; a value the file does not show, such as an
        import, is unknown."""
        if isinstance(node, ast.Assign):
            for t in node.targets:
                self._bind_target(scope, t, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self._bind_target(scope, node.target, node.value)
        elif isinstance(node, ast.AugAssign):
            grown = ast.BinOp(node.target, node.op, node.value)
            self.scope_of[id(grown)] = scope
            self._bind_target(scope, node.target, grown)
        elif isinstance(node, ast.NamedExpr):
            while isinstance(scope, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
                scope = self.parent[id(scope)] or self.tree
            self._bind_target(scope, node.target, node.value)
        elif isinstance(node, ast.For | ast.AsyncFor):
            self._bind_target(scope, node.target, _Element(node.iter))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                self._bind(scope, (alias.asname or alias.name).split(".")[0], _UNKNOWN)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            kind = "global" if isinstance(node, ast.Global) else "nonlocal"
            self.declared.setdefault(id(scope), {}).update(dict.fromkeys(node.names, kind))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            self._bind(scope, node.name, _UNKNOWN)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            self._bind_target(scope, node.optional_vars, _UNKNOWN)
        elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
            self._bind(scope, node.name, _UNKNOWN)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            self._bind(scope, node.rest, _UNKNOWN)

    def _bind_target(self, scope: ast.AST, target: ast.expr, value: "Bound") -> None:
        if isinstance(target, ast.Name):
            self._bind(scope, target.id, value)
        elif isinstance(target, ast.Tuple | ast.List):
            same = isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(target.elts)
            for i, t in enumerate(target.elts):
                part = value.elts[i] if same and isinstance(value, ast.Tuple | ast.List) else _Element(value)
                self._bind_target(scope, t, part)
        elif isinstance(target, ast.Starred):
            self._bind_target(scope, target.value, _Element(value))

    def _bind(self, scope: ast.AST, name: str, value: "Bound") -> None:
        kind = self.declared.get(id(scope), {}).get(name)
        if kind == "global":
            scope = self.tree
        elif kind == "nonlocal":
            scope = self._enclosing(scope, name)
        self.bindings.setdefault((id(scope), name), []).append(value)

    def _enclosing(self, scope: ast.AST, name: str) -> ast.AST:
        """The function a ``nonlocal`` name belongs to: the nearest enclosing one that binds it, or the nearest."""
        found: list[ast.AST] = []
        outer = self.parent.get(id(scope))
        while outer is not None and not isinstance(outer, ast.Module):
            if not isinstance(outer, ast.ClassDef):
                if (id(outer), name) in self.bindings:
                    return outer
                found.append(outer)
            outer = self.parent.get(id(outer))
        return found[0] if found else scope


@dataclass(frozen=True)
class _Element:
    """An element of what a loop or an unpacking goes over: literal when that is; ``None`` for a value the file does
    not show, such as a parameter."""

    of: "Bound"


Bound = ast.expr | _Element | None
_UNKNOWN = _Element(None)


def _called(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and node.func.attr in COMBINATIONS:
        return node.func.attr
    return ""


def _metadata(node: ast.expr) -> bool:
    """A frame's column names or schema, and their items."""
    if isinstance(node, ast.Attribute):
        return node.attr in METADATA
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("collect_schema", "names") and not node.args:
            return True
        if node.func.attr in ("items", "keys", "values"):
            return _metadata(node.func.value)
    return False


def _row_api(node: ast.AST) -> bool:
    """Whether a call of a row-by-row API, reported by itself, is inside ``node``."""
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ROW_APIS
        for n in ast.walk(node)
    )


def _text(node: ast.AST) -> str:
    text = ast.unparse(node)
    return text if len(text) <= 40 else text[:37] + "..."


def _shown(path: Path, idea: Path, lab: _Lab) -> str:
    """A file as the card names it: from the idea's directory, or from an installed lab's top package."""
    if path.is_relative_to(idea):
        return path.relative_to(idea).as_posix()
    parts = path.parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in lab.tops:
            return "/".join(parts[i:])
    return path.name if path.stem in lab.tops else str(path)
