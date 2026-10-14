import math

import polars as pl

from research_ci.eval.library.information import feature_pair_mutual_information
from research_ci.eval.library.structure import pca_diagnostics, vif


def _run(spec, table: pl.DataFrame):
    assert spec.runner is not None
    return spec.runner(table)


def test_mutual_information_uses_sklearn_after_polars_row_filtering() -> None:
    x = [float(value) for value in range(1, 61)]
    table = pl.DataFrame(
        {
            "x": [*x, None, float("inf")],
            "y": [*(value**2 for value in x), 1.0, 2.0],
        }
    )

    observation = _run(
        feature_pair_mutual_information(
            "x",
            "y",
            bins=12,
            normalized=False,
            min_rows=20,
            n_neighbors=4,
            random_state=17,
        ),
        table,
    )

    assert observation.status == "passed"
    assert (
        feature_pair_mutual_information("x", "y", normalized=False).execution_mode
        == "matrix_backend"
    )
    assert observation.value > 1.0
    assert observation.details["rows"] == 60
    assert observation.details["backend"] == "scikit-learn"
    assert (
        observation.details["estimator"]
        == "sklearn.feature_selection.mutual_info_regression"
    )
    assert observation.details["execution_mode"] == "matrix_backend"
    assert observation.details["random_state"] == 17
    assert observation.details["n_neighbors"] == 4
    assert observation.details["bins"] == 12
    assert observation.details["bins_used"] is False


def test_vif_and_pca_use_sklearn_matrix_backends() -> None:
    x = [float(value) for value in range(1, 61)]
    table = pl.DataFrame(
        {
            "x": [*x, None],
            "twice_x": [*(2.0 * value for value in x), 1.0],
            "wave": [*(math.sin(value) for value in x), 1.0],
        }
    )

    vif_observation = _run(vif(("x", "twice_x", "wave")), table)
    pca_observation = _run(pca_diagnostics(("x", "twice_x", "wave")), table)

    assert vif_observation.status == "passed"
    assert math.isinf(vif_observation.value["x"])
    assert vif_observation.details["rows"] == 60
    assert vif_observation.details["backend"] == "scikit-learn"
    assert vif_observation.details["execution_mode"] == "matrix_backend"
    assert (
        vif_observation.details["estimator"] == "sklearn.linear_model.LinearRegression"
    )

    assert pca_observation.status == "passed"
    assert pca_observation.details["rows"] == 60
    assert pca_observation.details["backend"] == "scikit-learn"
    assert pca_observation.details["execution_mode"] == "matrix_backend"
    assert pca_observation.details["estimator"] == "sklearn.decomposition.PCA"
    assert pca_observation.details["svd_solver"] == "full"
    assert pca_observation.value["effective_rank"] < 2.5
