"""Hard method policy for the Data stage."""

from ..registry import StagePolicy

POLICY = StagePolicy(
    stage="data",
    objective="validate source facts, alignment, and the research-ready data table",
    input_boundary="raw source tables and the assembled Data main table",
    output_boundary="an approved Data table without features, targets, or signals",
    allowed_purposes=("quality", "temporal"),
    forbidden_requirements=(
        "target",
        "horizon",
        "signals",
        "positions",
        "trades",
        "pnl",
    ),
    next_stage="feat",
)

__all__ = ["POLICY"]
