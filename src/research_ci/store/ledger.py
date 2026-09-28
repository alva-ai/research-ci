import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from research_ci.kernel.hashing import canonical_json, sha256_hex
from research_ci.store.workspace import write_atomic

GENESIS = "0" * 64


class LedgerError(ValueError):
    pass


class Ledger:
    """Append-only JSON lines where each record carries the hash of the one before it.

    A hash chain alone cannot show that records were cut from the end. With ``anchors``, the latest head is also kept
    in a directory outside the idea, keyed by the first record, and a ledger that no longer contains it is refused.
    """

    def __init__(self, path: Path, anchors: Path | None = None) -> None:
        self.path = Path(path)
        self.anchors = None if anchors is None else Path(anchors)

    def append(self, kind: str, stage: str, payload: dict[str, Any], at_ns: int | None = None) -> dict[str, Any]:
        records = list(self.records())
        last = records[-1] if records else None
        record = {
            "seq": 0 if last is None else last["seq"] + 1,
            "at_ns": time.time_ns() if at_ns is None else at_ns,
            "kind": kind,
            "stage": stage,
            "payload": payload,
            "prev": GENESIS if last is None else last["hash"],
        }
        record["hash"] = sha256_hex(canonical_json(record))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.anchors is not None:
            self._pin(records[0]["hash"] if records else record["hash"], record["seq"], record["hash"])
        return record

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            for line in handle:
                yield json.loads(line)

    def verify(self) -> int:
        """Check the chain and, with anchors, that no record seen before is gone; then move the anchor forward."""
        heads: list[str] = []
        if self.path.exists():
            with self.path.open("rb") as handle:
                for n, line in enumerate(handle, start=1):
                    _check(line, n, heads[-1] if heads else GENESIS)
                    heads.append(json.loads(line)["hash"])
        if self.anchors is not None and heads:
            self._check_anchor(heads)
        return len(heads)

    def head(self) -> str:
        last = self._last()
        return GENESIS if last is None else last["hash"]

    def n_trials(self, stage: str) -> int:
        return sum(1 for r in self.records() if r["kind"] == "submit" and r["stage"] == stage)

    def _check_anchor(self, heads: list[str]) -> None:
        path = self._anchor(heads[0])
        known = json.loads(path.read_text()) if path.exists() else None
        if known is not None:
            seq, head = known["seq"], known["head"]
            if seq >= len(heads) or heads[seq] != head:
                raise LedgerError(
                    f"line {seq + 1}: records were removed or rewritten; this machine last saw record {seq} "
                    f"({head[:12]}) and the ledger now has {len(heads)} records. "
                    "A copied idea needs its own `rci init`."
                )
        if known is None or known["seq"] < len(heads) - 1:
            self._pin(heads[0], len(heads) - 1, heads[-1])

    def _pin(self, first: str, seq: int, head: str) -> None:
        body = {"seq": seq, "head": head, "ledger": str(self.path.resolve())}
        write_atomic(self._anchor(first), canonical_json(body) + b"\n")

    def _anchor(self, first: str) -> Path:
        assert self.anchors is not None
        return self.anchors / f"{first}.json"

    def _last(self) -> dict[str, Any] | None:
        last = None
        for record in self.records():
            last = record
        return last


def _check(line: bytes, n: int, prev: str) -> None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"line {n}: not valid JSON") from exc
    if not line.endswith(b"\n") or line[:-1] != canonical_json(record):
        raise LedgerError(f"line {n}: not in canonical form")
    body = {k: v for k, v in record.items() if k != "hash"}
    if record.get("seq") != n - 1:
        raise LedgerError(f"line {n}: expected seq {n - 1}, found {record.get('seq')}")
    if record.get("prev") != prev:
        raise LedgerError(f"line {n}: does not follow the previous record")
    if record.get("hash") != sha256_hex(canonical_json(body)):
        raise LedgerError(f"line {n}: content does not match its hash")
