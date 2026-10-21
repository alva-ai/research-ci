"""General-method proposals for the Data stage."""

METHOD_NAMES = (
    "library.quality.required_columns",
    "library.quality.finite_columns",
    "library.quality.coverage",
    "library.temporal.key_grid",
    "library.temporal.cross_section_completeness",
)

__all__ = ["METHOD_NAMES"]
