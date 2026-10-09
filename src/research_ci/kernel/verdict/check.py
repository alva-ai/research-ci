from dataclasses import dataclass
from typing import Literal

Status = Literal["pass", "fail", "warn", "skip"]


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
