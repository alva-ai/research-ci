"""What every card shares: the fingerprint a relayed approval quotes, and the line helpers."""

import math
from typing import Any

from research_ci.api.contract import span
from research_ci.kernel.hashing import sha256_hex

MARK = {"pass": "ok", "fail": "FAIL", "warn": "[!]", "skip": "--"}


def card_id(text: str) -> str:
    """A short fingerprint of a card's text. An agent relaying a person's approval quotes it, so the approval is tied
    to the card the person was shown."""
    return sha256_hex(text.encode())[:8]


def adder(out: list[str]) -> Any:
    def add(key: str, *lines: str | None) -> None:
        body = [line for line in lines if line]
        for i, line in enumerate(body):
            out.append(f"{key if i == 0 else '':<11}{line}")

    return add


def check_line(c: dict[str, Any]) -> str:
    return f"{MARK[c['status']]:<4} {c['name']}: {c['detail']}"


def eval_line(c: dict[str, Any]) -> str:
    """A check with whose eval it is: the stage's ``floor`` or ``default`` list, or the idea's ``library`` or ``lab``
    eval."""
    return f"{c.get('by') or 'floor':<8} {check_line(c)}"


def failed_tests(validation: dict[str, Any] | None) -> list[str]:
    """The tests an invalid eval failed after the first, which its line already shows, one line each."""
    if not validation or validation.get("status") != "invalid":
        return []
    failed = [t for t in validation.get("tests", []) if t["status"] == "fail"][1:]
    return [f"{'':<14}also failed {t['name']}: {t['detail']}" for t in failed]


def bp(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 1e4:.1f}"


def num(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.4g}"


def figure(x: float | None, units: dict[str, Any]) -> str:
    """A target's number without its unit: basis points when the target says what one value is (``per``)."""
    return bp(x) if units.get("per") else num(x)


def amount(x: float | None, units: dict[str, Any]) -> str:
    """A target's number with its unit, such as ``5.9 bp/entry``; the raw number for a target without ``per``."""
    return f"{bp(x)} bp/{units['per']}" if units.get("per") else num(x)


def rolled(x: float | None, horizon: str, units: dict[str, Any]) -> str:
    """What a return over ``horizon`` comes to over a year, repeated back to back, for a target that ``rolls``."""
    if not units.get("rolls") or x is None or not math.isfinite(x):
        return ""
    return f"≈ {x * (span('365 days') / span(horizon)):.1%}/yr"
