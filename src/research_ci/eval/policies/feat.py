"""Hard method policy for the Feat stage."""

from ..registry import StagePolicy

POLICY = StagePolicy(
    stage="feat",
    objective=(
        "build reusable point-in-time features and evaluate their quality, "
        "structure, redundancy, and target-free stability"
    ),
    input_boundary="the approved Data main table",
    output_boundary="keys and reusable feature columns without targets or signals",
    allowed_purposes=("quality", "structure", "stability"),
    forbidden_requirements=(
        "target",
        "horizon",
        "signals",
        "positions",
        "trades",
        "pnl",
    ),
    next_stage="sig",
)

__all__ = ["POLICY"]
