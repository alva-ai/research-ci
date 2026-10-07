"""Admission records of data sources, shared by every idea on this machine: one hash-chained ledger under
``$RCI_HOME/sources``. A source is admitted per plugin, options and code; any change needs a new admission."""

from pathlib import Path
from typing import Any

from research_ci.kernel.hashing import hash_obj
from research_ci.store.ledger import Ledger
from research_ci.store.workspace import rci_home


def sources_root() -> Path:
    return rci_home() / "sources"


def source_key(name: str, options: dict[str, Any], code: str) -> str:
    """Plugin name, options and code: what an admission is about."""
    return f"{name}@{hash_obj({'options': options, 'code': code})[:16]}"


class Sources:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else sources_root()
        self.ledger = Ledger(self.root / "ledger.jsonl", anchors=rci_home() / "heads")
        if self.ledger.path.exists():
            self.ledger.verify()

    def record(self, key: str, spec: dict[str, Any]) -> dict[str, Any]:
        return self.ledger.append("admit", key, {"spec": spec, "admissible": spec["verdict"]["status"] == "admissible"})

    def approve(self, key: str, admission: str, approval: dict[str, Any]) -> dict[str, Any]:
        return self.ledger.append("approve", key, {"admission": admission, "approval": approval})

    def latest(self, key: str) -> dict[str, Any] | None:
        """The last admission run for ``key``, with ``approved`` set to its approval record, if any."""
        found: dict[str, Any] | None = None
        for r in self.ledger.records():
            if r["stage"] != key:
                continue
            if r["kind"] == "admit":
                found = {**r, "approved": None}
            elif r["kind"] == "approve" and found is not None and r["payload"]["admission"] == found["hash"]:
                found["approved"] = r
        return found
