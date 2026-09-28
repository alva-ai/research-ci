import platform
import re
from importlib.metadata import distributions, requires, version
from pathlib import Path

import alpha_ci
from alpha_ci.kernel.hashing import hash_obj
from alpha_ci.store.workspace import hash_file, hash_tree

ENGINE = "alpha-ci"


def judging_deps() -> dict[str, str]:
    """What the verdict machinery depends on: engine source, interpreter, and the engine's own dependencies."""
    return {
        "engine": hash_tree(Path(alpha_ci.__file__).parent),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "engine_packages": hash_obj(engine_packages()),
    }


def engine_packages() -> dict[str, str]:
    names = [m.group(0) for r in requires(ENGINE) or [] if "extra ==" not in r and (m := re.match(r"[\w.-]+", r))]
    return {ENGINE: version(ENGINE)} | {name: version(name) for name in sorted(names)}


def environment_record(start: Path) -> dict[str, str]:
    """Everything installed, recorded with each submission. Research libraries may change between submissions."""
    installed = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]})
    lock = next((p / "uv.lock" for p in [start, *start.parents] if (p / "uv.lock").is_file()), None)
    return {"packages": hash_obj(installed), "uv.lock": "none" if lock is None else hash_file(lock)}
