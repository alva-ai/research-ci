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
from .economic import METHODS as ECONOMIC_METHODS
from .economic import backtest_summary
from .information import METHODS as INFORMATION_METHODS
from .information import (
    context_feature_target_mutual_information,
    feature_pair_mutual_information,
    feature_target_mutual_information,
)
from .model import METHODS as MODEL_METHODS
from .model import classification_probability_diagnostics, regression_diagnostics
from .quality import METHODS as QUALITY_METHODS
from .quality import coverage, dispersion, finite_columns, required_columns
from .stability import METHODS as STABILITY_METHODS
from .stability import (
    distribution_shift,
    rank_autocorrelation,
    split_association,
    walk_forward_association,
)
from .structure import METHODS as STRUCTURE_METHODS
from .structure import autocorrelation, pairwise_correlation, pca_diagnostics, vif
from .temporal import METHODS as TEMPORAL_METHODS
from .temporal import cross_section_completeness, key_grid

METHODS = (
    QUALITY_METHODS
    + ECONOMIC_METHODS
    + STRUCTURE_METHODS
    + ASSOCIATION_METHODS
    + INFORMATION_METHODS
    + MODEL_METHODS
    + STABILITY_METHODS
    + TEMPORAL_METHODS
)

__all__ = [
    "ASSOCIATION_METHODS",
    "ECONOMIC_METHODS",
    "INFORMATION_METHODS",
    "METHODS",
    "MODEL_METHODS",
    "QUALITY_METHODS",
    "STABILITY_METHODS",
    "STRUCTURE_METHODS",
    "TEMPORAL_METHODS",
    "autocorrelation",
    "backtest_summary",
    "bucket_spread",
    "classification_probability_diagnostics",
    "context_pearson_ic",
    "context_spearman_ic",
    "context_feature_target_mutual_information",
    "coverage",
    "cross_section_completeness",
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
    "regression_diagnostics",
    "required_columns",
    "spearman_ic",
    "split_association",
    "walk_forward_association",
    "vif",
]
