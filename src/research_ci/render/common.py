"""What every card shares: the fingerprint a relayed approval quotes, and the line helpers."""

import math
from typing import Any

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


def bp(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 1e4:.1f}"


def days(days: float | None) -> str:
    if days is None or not math.isfinite(days):
        return "n/a"
    return f"{days:.0f} d" if days < 60 else f"{days / 30.4:.0f} mo"


def num(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.4g}"


def pct(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+.1%}"
