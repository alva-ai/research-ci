"""General-method proposals for the Feat stage."""

METHOD_NAMES = (
    "library.quality.required_columns",
    "library.quality.finite_columns",
    "library.quality.coverage",
    "library.quality.dispersion",
    "library.structure.pairwise_correlation",
    "library.structure.vif",
    "library.structure.pca_diagnostics",
    "library.structure.autocorrelation",
    "library.information.feature_pair_mutual_information",
    "library.stability.rank_autocorrelation",
    "library.stability.distribution_shift",
)

__all__ = ["METHOD_NAMES"]
