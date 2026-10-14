"""Hard method policy for the Algo stage."""

from ..registry import StagePolicy

POLICY = StagePolicy(
    stage="algo",
    objective=(
        "convert approved signals into capital allocation, positions, orders, "
        "risk controls, costs, and execution evidence"
    ),
    input_boundary="approved signal tables and declared market/execution context",
    output_boundary="positions, orders, executions, and economic evidence",
    allowed_purposes=("quality", "economic"),
    forbidden_requirements=("target", "horizon"),
    next_stage=None,
)

__all__ = ["POLICY"]
