"""Hard method policy for the Sig stage."""

from ..registry import StagePolicy

POLICY = StagePolicy(
    stage="sig",
    objective=(
        "select features for an explicit recipe and produce a forecast, ranking, "
        "direction, or strength"
    ),
    input_boundary=(
        "the approved Feat table plus explicit label sidecars when the recipe "
        "is target-aware"
    ),
    output_boundary="keys and signal columns without positions, orders, or PnL",
    allowed_purposes=("quality", "predictive", "stability"),
    forbidden_requirements=("positions", "trades", "pnl"),
    next_stage="algo",
)

__all__ = ["POLICY"]
