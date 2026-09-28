import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from research_ci.kernel.hashing import canonical_json

STATE_DIR = ".rci"
IGNORED = frozenset({STATE_DIR, "__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache"})


def rci_home() -> Path:
    """Per-user engine state outside any idea: ledger anchors, and later the data vault."""
    return Path(os.environ.get("RCI_HOME") or Path.home() / ".rci")


class Workspace:
    """Engine-owned state of one idea, kept under ``<idea>/.rci``."""

    def __init__(self, idea_dir: Path) -> None:
        self.idea_dir = Path(idea_dir)
        self.root = self.idea_dir / STATE_DIR

    def path(self, rel: str) -> Path:
        return self.root / rel

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize commands on one idea across processes."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path("lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def write_json(self, rel: str, obj: Any) -> Path:
        text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        return write_atomic(self.path(rel), text.encode())

    def read_json(self, rel: str) -> Any:
        path = self.path(rel)
        if not path.exists():
            return None
        return json.loads(path.read_text())


def write_atomic(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, ignored: frozenset[str] = IGNORED) -> str:
    """Hash of every file's relative path and content under ``root``, skipping caches and engine state."""
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and not ignored.intersection(rel.parts):
            entries.append([rel.as_posix(), hash_file(path)])
    return hashlib.sha256(canonical_json(entries)).hexdigest()
