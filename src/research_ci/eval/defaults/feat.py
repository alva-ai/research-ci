"""General-method proposals for the Feat stage."""

METHOD_NAMES = (
    "library.quality.finite_columns",
    "library.structure.pairwise_correlation",
    "library.association.pearson_ic",
    "library.association.spearman_ic",
)

__all__ = ["METHOD_NAMES"]
