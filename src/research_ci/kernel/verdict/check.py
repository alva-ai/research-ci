from dataclasses import dataclass
from typing import Literal

Status = Literal["pass", "fail", "warn", "skip"]


@dataclass(frozen=True)
class Check:
    """One line of a card: what was checked, how it came out, and why. ``by`` says whose eval it is: the stage's
    ``floor`` or ``default`` list, or the idea's evals, a ``library`` one or a ``case`` one."""

    name: str
    status: Status
    detail: str
    by: str = ""
