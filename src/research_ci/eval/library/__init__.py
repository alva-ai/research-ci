"""Reusable general eval methods, grouped by methodology."""

from .association import METHODS as ASSOCIATION_METHODS
from .association import pearson_ic, spearman_ic
from .quality import METHODS as QUALITY_METHODS
from .quality import finite_columns, required_columns
from .structure import METHODS as STRUCTURE_METHODS
from .structure import pairwise_correlation

METHODS = QUALITY_METHODS + STRUCTURE_METHODS + ASSOCIATION_METHODS

__all__ = [
    "ASSOCIATION_METHODS",
    "METHODS",
    "QUALITY_METHODS",
    "STRUCTURE_METHODS",
    "finite_columns",
    "pairwise_correlation",
    "pearson_ic",
    "required_columns",
    "spearman_ic",
]
