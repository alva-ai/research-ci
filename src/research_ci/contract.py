"""The intentionally small contract boundary."""

from dataclasses import dataclass

ALLOWED_STAGES = ("data", "feat", "sig", "algo")


class ContractError(ValueError):
    """Raised when a contract cannot define a runnable stage sequence."""


@dataclass(frozen=True, slots=True)
class Contract:
    """Task boundary shared by all stages.

    The contract does not define labels, outcomes, horizons, methods, or
    economic results. Those belong to later stage-specific work.
    """

    mission: str
    stages: tuple[str, ...] = ("data", "feat", "sig", "algo")
    scope: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()


def validate_contract(contract: Contract) -> None:
    """Validate only structure and ordering, never research conclusions."""

    if not isinstance(contract, Contract):
        raise ContractError("CONTRACT must be an research_ci.Contract")
    if not contract.mission.strip():
        raise ContractError("mission must not be empty")
    if not contract.stages:
        raise ContractError("stages must contain at least one stage")
    unknown = sorted(set(contract.stages) - set(ALLOWED_STAGES))
    if unknown:
        raise ContractError(f"unknown stages: {', '.join(unknown)}")
    if len(set(contract.stages)) != len(contract.stages):
        raise ContractError("stages must not contain duplicates")
    positions = [ALLOWED_STAGES.index(stage) for stage in contract.stages]
    if positions != sorted(positions):
        raise ContractError("stages must follow data -> feat -> sig -> algo order")
    if any(not item.strip() for item in contract.invariants):
        raise ContractError("invariants must not contain empty values")
