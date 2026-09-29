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


class PluginError(ContractError):
    pass


def load(use: Use, registry: Registry | None = None) -> Any:
    """Build a plugin with the contract's options, from ``registry`` or ``EXTRA`` if they have it, else from its
    entry point."""
    factory = (registry if registry is not None else EXTRA).get(use.name)
    if factory is None:
        found = entry_points(group=ENTRY_POINTS, name=use.name)
        if not found:
            raise PluginError(f"no plugin named {use.name!r} is installed (entry-point group {ENTRY_POINTS})")
        factory = next(iter(found)).load()
    try:
        return factory(**use.options)
    except TypeError as exc:
        raise PluginError(f"plugin {use.name!r} rejects its options: {exc}") from exc


def code_hash(plugin: Any) -> str:
    """Hash of the directory holding the plugin's module: a change to its code is a change to the contract."""
    try:
        path = Path(inspect.getfile(type(plugin)))
    except TypeError:
        return "builtin"
    return hash_tree(path.parent)
