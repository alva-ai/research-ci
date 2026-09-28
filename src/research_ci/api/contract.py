from dataclasses import dataclass

import pandas as pd


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Contract:
    """The claim an idea makes and the rules its verdicts are held to. Written by the agent, frozen by a person.

    Explore data ends at ``explore_end``; the sealed segment runs to ``seal_end``. ``budget`` is the number of
    submissions each layer may use across all its versions. ``skip`` lists optional stages of the template.
    """

    template: str
    claim: str
    explore_end: str
    seal_end: str
    budget: int = 20
    skip: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim.strip() or self.claim.startswith("TODO"):
            raise ContractError("state the claim in one sentence")
        if _timestamp("seal_end", self.seal_end) <= _timestamp("explore_end", self.explore_end):
            raise ContractError("seal_end must be after explore_end")
        if self.budget < 1:
            raise ContractError("budget must allow at least one submission")
        object.__setattr__(self, "skip", tuple(self.skip))


def _timestamp(name: str, value: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except ValueError as exc:
        raise ContractError(f"{name} is not a timestamp: {value!r}") from exc
    if ts.tzinfo is None:
        raise ContractError(f"{name} needs a time zone, for example {value}Z")
    return ts
