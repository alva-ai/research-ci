"""Reusable general eval methods, grouped by methodology."""

from .association import METHODS as ASSOCIATION_METHODS
from .association import (
    bucket_spread,
    context_pearson_ic,
    context_spearman_ic,
    cumulative_ic,
    icir,
    pearson_ic,
    spearman_ic,
)
from .information import METHODS as INFORMATION_METHODS
from .information import (
    context_feature_target_mutual_information,
    feature_pair_mutual_information,
    feature_target_mutual_information,
)
from .quality import METHODS as QUALITY_METHODS
from .quality import coverage, dispersion, finite_columns, required_columns
from .stability import METHODS as STABILITY_METHODS
from .stability import distribution_shift, rank_autocorrelation, split_association
from .structure import METHODS as STRUCTURE_METHODS
from .structure import autocorrelation, pairwise_correlation, pca_diagnostics, vif
from .temporal import METHODS as TEMPORAL_METHODS
from .temporal import key_grid

METHODS = (
    QUALITY_METHODS
    + STRUCTURE_METHODS
    + ASSOCIATION_METHODS
    + INFORMATION_METHODS
    + STABILITY_METHODS
    + TEMPORAL_METHODS
)

__all__ = [
    "ASSOCIATION_METHODS",
    "INFORMATION_METHODS",
    "METHODS",
    "QUALITY_METHODS",
    "STABILITY_METHODS",
    "STRUCTURE_METHODS",
    "TEMPORAL_METHODS",
    "autocorrelation",
    "bucket_spread",
    "context_pearson_ic",
    "context_spearman_ic",
    "context_feature_target_mutual_information",
    "coverage",
    "cumulative_ic",
    "dispersion",
    "distribution_shift",
    "feature_pair_mutual_information",
    "feature_target_mutual_information",
    "finite_columns",
    "icir",
    "key_grid",
    "pairwise_correlation",
    "pca_diagnostics",
    "pearson_ic",
    "rank_autocorrelation",
    "required_columns",
    "spearman_ic",
    "split_association",
    "vif",
]
