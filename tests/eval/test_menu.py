import json

import pytest

from research_ci import (
    CaseEvalTest,
    EvalPlan,
    EvalSpec,
    FeatureSet,
    SignalSet,
    compile_stage_menu,
)
from research_ci.cli import main
from research_ci.eval.defaults import DEFAULT_METHODS
from research_ci.eval.library.association import pearson_ic


def _entries(stage: str, declaration=None):
    menu = compile_stage_menu(stage, declaration)
    return {entry.method.name: entry for entry in menu.entries}


def test_feat_menu_exposes_structure_and_blocks_predictive_methods() -> None:
    entries = _entries(
        "feat",
        FeatureSet(keys=("event_time",), features=("x", "y")),
    )

    pair_mi = entries["library.information.feature_pair_mutual_information"]
    assert pair_mi.availability == "selectable"
    assert pair_mi.recommended

    rank_ic = entries["library.association.spearman_ic"]
    assert rank_ic.availability == "unavailable"
    assert any("stage(s): sig" in reason for reason in rank_ic.reasons)


def test_targetless_sig_menu_blocks_only_target_dependent_methods() -> None:
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        hypothesis="extreme values should mean revert",
    )
    entries = _entries("sig", declaration)

    assert entries["library.quality.coverage"].availability == "selectable"
    rank_ic = entries["library.association.spearman_ic"]
    assert rank_ic.availability == "unavailable"
    assert rank_ic.reasons == ("requires SignalSet.targets with an explicit horizon",)


def test_target_aware_sig_menu_enables_predictive_methods() -> None:
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        targets={"1h": "return_1h"},
    )
    entries = _entries("sig", declaration)

    assert entries["library.association.spearman_ic"].availability == "selectable"
    assert (
        entries["library.information.feature_target_mutual_information"].availability
        == "selectable"
    )
    assert entries["library.structure.vif"].availability == "unavailable"


def test_every_default_is_allowed_by_its_static_stage_menu() -> None:
    for stage, defaults in DEFAULT_METHODS.items():
        entries = _entries(stage)
        unavailable = [
            name for name in defaults if entries[name].availability != "selectable"
        ]
        assert unavailable == []


def test_targetless_signal_set_requires_an_explicit_hypothesis() -> None:
    with pytest.raises(ValueError, match="explicit hypothesis"):
        SignalSet(keys=("event_time",), features=("x",))


def test_cli_stage_menu_reports_disabled_methods(capsys) -> None:
    exit_code = main(["eval", "methods", "--stage", "feat", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "feat"
    methods = {method["name"]: method for method in payload["methods"]}
    assert methods["library.association.pearson_ic"]["availability"] == "unavailable"


def _plan_with_library(spec: EvalSpec, *, stage: str = "feat") -> EvalPlan:
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=(spec,),
        case=(EvalSpec.noop("case.rule"),),
        case_eval=(
            CaseEvalTest(
                name="case.rule_test",
                case="case.rule",
                valid=lambda table: table,
                invalid=lambda table: table,
            ),
        ),
    )


def test_library_slot_cannot_bypass_stage_policy() -> None:
    plan = _plan_with_library(pearson_ic("x", "return_1h"))

    with pytest.raises(ValueError, match="unavailable in stage 'feat'"):
        plan.validate()


def test_library_slot_rejects_an_unregistered_raw_spec() -> None:
    plan = _plan_with_library(EvalSpec.noop("library.raw_custom_check"))

    with pytest.raises(ValueError, match="registered Engine method"):
        plan.validate()
