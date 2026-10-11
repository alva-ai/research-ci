import difflib
import inspect
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from research_ci.api.contract import ContractError, Use
from research_ci.api.plugins import ENTRY_POINTS
from research_ci.store.workspace import hash_tree

Registry = Mapping[str, Callable[..., Any]]
EXTRA: dict[str, Callable[..., Any]] = {}
LIST_THEM = "`rci plugins` lists every installed plugin with its options"


class PluginError(ContractError):
    pass


def load(use: Use, registry: Registry | None = None) -> Any:
    """Build a plugin with the contract's options, from ``registry`` or ``EXTRA`` if they have it, else from its
    entry point."""
    factory = factory_of(use.name, registry)
    try:
        return factory(**use.options)
    except (TypeError, ValueError) as exc:
        raise PluginError(
            f"plugin {use.name!r} rejects its options {dict(use.options)!r}: {exc}; "
            f"it takes {signature(use.name, factory)}"
        ) from exc


def factory_of(name: str, registry: Registry | None = None) -> Callable[..., Any]:
    factory = (registry if registry is not None else EXTRA).get(name)
    if factory is not None:
        return factory
    found = entry_points(group=ENTRY_POINTS, name=name)
    if not found:
        close = difflib.get_close_matches(name, names(registry), n=3, cutoff=0.75)
        hint = f"did you mean {', '.join(map(repr, close))}? " if close else ""
        raise PluginError(
            f"no plugin named {name!r} is installed (entry-point group {ENTRY_POINTS}); {hint}{LIST_THEM}"
        )
    return next(iter(found)).load()


def names(registry: Registry | None = None) -> list[str]:
    """Every plugin name ``load`` can find: the registry's (or ``EXTRA``'s), then the entry points'."""
    local = registry if registry is not None else EXTRA
    return sorted(set(local) | {ep.name for ep in entry_points(group=ENTRY_POINTS)})


def signature(name: str, factory: Callable[..., Any]) -> str:
    """The options a plugin is built with, as ``name(option: type = default, ...)``."""
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return f"{name}(...)"
    return f"{name}({', '.join(_param(p) for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD)})"


def code_hash(plugin: Any) -> str:
    """Hash of the directory holding the plugin's module: a change to its code is a change to the contract."""
    try:
        path = Path(inspect.getfile(type(plugin)))
    except TypeError:
        return "builtin"
    return hash_tree(path.parent)


def _param(p: inspect.Parameter) -> str:
    text = p.name if p.kind is not inspect.Parameter.VAR_POSITIONAL else f"*{p.name}"
    if p.annotation is not inspect.Parameter.empty:
        text += f": {annotation(p.annotation)}"
    if p.default is not inspect.Parameter.empty:
        text += f" = {p.default!r}"
    return text


def annotation(a: Any) -> str:
    """A type as people write it: ``pl.DataFrame``, ``pd.Timedelta``, ``str | None``."""
    text = a if isinstance(a, str) else inspect.formatannotation(a)
    for long, short in (
        ("polars.dataframe.frame.", "pl."),
        ("polars.lazyframe.frame.", "pl."),
        ("polars.series.series.", "pl."),
        ("polars.expr.expr.", "pl."),
        ("pandas._libs.tslibs.timedeltas.", "pd."),
        ("pandas._libs.tslibs.timestamps.", "pd."),
        ("pandas.core.frame.", "pd."),
        ("numpy.random._generator.", "np.random."),
        ("collections.abc.", ""),
    ):
        text = text.replace(long, short)
    return text
