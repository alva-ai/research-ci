"""A feat submission's verdict from its decisive effects: PASSED when every one beats random, FAILED otherwise, with
a line for each."""

import math
from typing import Any


def combine(decisive: list[dict[str, Any]], need: float) -> dict[str, Any]:
    lines = [
        f"{d['on']}: beats random, z {_num(d['z'])} ≥ {need:.2f}"
        if d["outcome"] == "passed"
        else f"{d['on']}: does not beat random, z {_num(d['z'])} < {need:.2f} (decisive test, floor): {_effect(d)}"
        for d in decisive
    ]
    return verdict("PASSED" if all(d["outcome"] == "passed" for d in decisive) else "FAILED", lines)


def _effect(d: dict[str, Any]) -> str:
    """What the decisive effect came to, for a FAILED verdict: the gain over random and its error at the best horizon,
    or why no z could be computed."""
    if d.get("best") is None:
        return "no horizon has enough blocks for a z; the evaluation tables are too short for this contract"
    per = f" per {d['per']}" if d.get("per") else ""
    scale, unit = (1e4, " bp") if d.get("per") else (1.0, "")
    gain, se = d["effect"] * scale, d["se"] * scale
    return f"gain over random {gain:.3g}{unit}{per} ± {se:.3g} at {d['best']}"


def verdict(status: str, notes: list[str]) -> dict[str, Any]:
    return {"status": status, "notes": notes}


def _num(x: Any) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.4g}"
