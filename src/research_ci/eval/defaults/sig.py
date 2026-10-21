"""General-method proposals for the Sig stage."""

METHOD_NAMES = (
    "library.quality.required_columns",
    "library.quality.finite_columns",
    "library.quality.coverage",
    "library.quality.dispersion",
    "library.association.pearson_ic",
    "library.association.spearman_ic",
    "library.association.icir",
    "library.association.cumulative_ic",
    "library.association.bucket_spread",
    "library.information.feature_target_mutual_information",
    "library.stability.split_association",
    "library.stability.walk_forward_association",
    "library.stability.rank_autocorrelation",
    "library.stability.distribution_shift",
)

__all__ = ["METHOD_NAMES"]
