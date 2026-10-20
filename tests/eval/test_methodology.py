import math

import polars as pl

from research_ci import CaseEvalTest, EvalContext, EvalPlan, EvalSpec, run_eval_plan
from research_ci.eval.assessment import (
    assess_features,
    assess_records,
    assess_signal_evidence,
)
from research_ci.eval.library.association import (
    context_pearson_ic,
    context_spearman_ic,
    cumulative_ic,
    icir,
)
from research_ci.eval.library.information import (
    context_feature_target_mutual_information,
    feature_pair_mutual_information,
)
from research_ci.eval.library.quality import coverage
from research_ci.eval.library.structure import pca_diagnostics, vif
from research_ci.eval.runner import EvalRecord


def _plan(*library: EvalSpec, stage: str = "feat") -> EvalPlan:
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=library or (EvalSpec.noop("library.placeholder"),),
        case=(EvalSpec("case.ok", expr=pl.col("x") > 0),),
        case_eval=(
            CaseEvalTest.from_rows(
                name="case.ok.test",
                case="case.ok",
                valid={"x": [1]},
                invalid={"x": [-1]},
            ),
        ),
    )


def test_mutual_information_pca_and_vif_produce_explicit_metrics() -> None:
    table = pl.DataFrame(
        {
            "x": [float(value) for value in range(1, 41)],
            "y": [float(value * 2) for value in range(1, 41)],
            "z": [math.sin(value) for value in range(1, 41)],
        }
    )

    result = run_eval_plan(
        table,
        _plan(
            feature_pair_mutual_information("x", "y", min_rows=10),
            vif(("x", "y", "z")),
            pca_diagnostics(("x", "y", "z")),
        ),
    )
    records = {record.name: record for record in result.records}

    method = records["library.information.feature_pair_mutual_information"]
    assert method.status == "passed"
    assert method.value > 0.9
    assert records["library.structure.vif"].details["max_vif"] > 10
    pca = records["library.structure.pca_diagnostics"]
    assert pca.status == "passed"
    assert pca.value["effective_rank"] < 2.5


def test_explicit_thresholds_create_review_reasons() -> None:
    table = pl.DataFrame({"x": [float(value) for value in range(1, 20)] + [None]})
    result = run_eval_plan(
        table,
        _plan(
            coverage(("x",), minimum=0.99),
            pca_diagnostics(("x",), minimum_effective_rank=2.0),
        ),
    )
    records = {record.name: record for record in result.records}

    assert records["library.quality.coverage"].status == "failed"
    assert records["library.structure.pca_diagnostics"].status == "failed"
    assessment = assess_records(result.records)
    assert "LOW_COVERAGE" in assessment["reason_codes"]
    assert "STRUCTURE_CONCENTRATED" in assessment["reason_codes"]


def test_grouped_ic_methods_require_explicit_groups() -> None:
    rows = []
    for period in range(6):
        for instrument in range(4):
            score = float(instrument + period * 0.01)
            target = score if period % 2 else float(3 - instrument)
            rows.append((period, instrument, score, target))
    table = pl.DataFrame(
        rows, schema=["period", "inst", "score", "target"], orient="row"
    )

    result = run_eval_plan(
        table,
        _plan(
            icir("score", "target", by="period", min_group_size=3),
            cumulative_ic("score", "target", by="period", min_group_size=3),
            stage="sig",
        ),
    )
    records = {record.name: record for record in result.records}

    assert records["library.association.icir"].status == "passed"
    assert records["library.association.cumulative_ic"].status == "passed"
    assert "cumulative_ic" in records["library.association.cumulative_ic"].details


def test_context_target_stays_out_of_main_table() -> None:
    features = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "score": [1.0, 2.0, 3.0, 4.0],
            "x": [1, 1, 1, 1],
        }
    )
    targets = pl.DataFrame(
        {"event_time": [1, 2, 3, 4], "forward_return": [0.1, 0.2, 0.3, 0.4]}
    )
    result = run_eval_plan(
        features,
        _plan(
            context_pearson_ic(
                "score",
                "forward_return",
                context_table="labels",
                join_on=("event_time",),
            ),
            context_spearman_ic(
                "score",
                "forward_return",
                context_table="labels",
                join_on=("event_time",),
            ),
            context_feature_target_mutual_information(
                "score",
                "forward_return",
                context_table="labels",
                min_rows=2,
            ),
            stage="sig",
        ),
        EvalContext(tables={"labels": targets}),
    )

    records = {record.name: record for record in result.records}
    assert records["library.association.context_pearson_ic"].status == "passed"
    assert records["library.association.context_pearson_ic"].value == 1.0
    assert records["library.association.context_spearman_ic"].status == "passed"
    assert (
        records["library.information.context_feature_target_mutual_information"].status
        == "passed"
    )
    assert features.columns == ["event_time", "score", "x"]


def test_assessment_separates_reviewable_missing_values_from_hard_failures() -> None:
    table = pl.DataFrame({"x": [1.0, None, 3.0]})
    result = run_eval_plan(table, _plan(coverage(("x",))))
    assessment = assess_records(result.records)

    assert assessment["execution_status"] == "review"
    assert assessment["research_disposition"] == "incomplete"
    assert "MISSING_VALUES_REVIEW" in assessment["reason_codes"]
    assert assessment["hard_failures"] == []


def test_feature_triage_explains_structural_candidates() -> None:
    table = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "complete": [1.0, 2.0, 3.0, 4.0],
            "sparse": [1.0, None, None, None],
            "constant": [2.0, 2.0, 2.0, 2.0],
        }
    )
    triage = assess_features(table, ("event_time",), ())

    assert triage["complete"]["status"] == "candidate_for_review"
    assert triage["sparse"]["status"] == "incomplete"
    assert "LOW_COVERAGE" in triage["sparse"]["reason_codes"]
    assert triage["constant"]["status"] == "weak_data"
    assert "NO_VARIATION" in triage["constant"]["reason_codes"]


def test_signal_evidence_keeps_metrics_and_split_instability_explicit() -> None:
    records = (
        EvalRecord(
            stage="sig",
            slot="library",
            name="signal.rank",
            status="passed",
            value=0.12,
            library_method="library.association.spearman_ic",
            details={"score": "signal", "target": "return_6h"},
        ),
        EvalRecord(
            stage="sig",
            slot="library",
            name="signal.split",
            status="passed",
            value=0.3,
            library_method="library.stability.split_association",
            details={
                "score": "signal",
                "target": "return_6h",
                "first_half": -0.1,
                "second_half": 0.2,
                "same_sign": False,
            },
        ),
        EvalRecord(
            stage="sig",
            slot="library",
            name="candidate.rank",
            status="passed",
            value=-0.2,
            library_method="library.association.spearman_ic",
            details={"score": "feature", "target": "return_6h"},
        ),
    )

    evidence = assess_signal_evidence(records, ("signal",), ("feature",))

    signal = evidence["signals"]["signal"]
    assert signal["targets"]["return_6h"]["rank_ic"] == 0.12
    assert signal["reason_codes"] == ["SPLIT_SIGN_CHANGE:return_6h"]
    assert evidence["candidate_rank_ic"]["return_6h"]["most_negative"] == [
        {"feature": "feature", "value": -0.2}
    ]


def test_known_no_variation_is_weak_evidence_not_missing_context() -> None:
    record = EvalRecord(
        stage="sig",
        slot="library",
        name="candidate.mi",
        status="not_applicable",
        library_method="library.information.feature_target_mutual_information",
        details={
            "reason": "mutual information input has no variation",
            "reason_code": "NO_VARIATION",
        },
    )

    assessment = assess_records((record,))

    assert assessment["research_disposition"] == "weak_data"
    assert assessment["reason_codes"] == ["NO_VARIATION"]


def test_specific_library_reason_does_not_duplicate_generic_failure() -> None:
    record = EvalRecord(
        stage="feat",
        slot="library",
        name="feature.correlation",
        status="failed",
        failure_policy="record",
        details={"reason_code": "HIGH_REDUNDANCY"},
    )

    assessment = assess_records((record,))

    assert assessment["reason_codes"] == ["HIGH_REDUNDANCY"]
