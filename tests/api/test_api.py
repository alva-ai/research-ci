import json
import types
from dataclasses import asdict
from functools import partial

import pytest

from research_ci.api import (
    Alternative,
    Contract,
    ContractError,
    Effect,
    Eval,
    Execution,
    Forward,
    Null,
    PermuteWithin,
    Universe,
    Usage,
    Use,
    evals_of,
)
from research_ci.api.contract import default_horizons

CARRY = Use("ref.carry", {"flow": "funding_rate"})
COSTS = Use("ref.flat_costs", {"commission_bps": 5, "slippage_bps": 5, "legs": 2})
USED = Usage("2026-04-08T00:00:00Z", "2026-09-14T12:00:00Z", "fc parameters tuned", ("claim",))


def _contract(**kw: object) -> Contract:
    base: dict[str, object] = {
        "claim": "cross-venue funding spreads persist",
        "decisive": (Effect(target="perp.breakeven"),),
        "alternatives": (Alternative("tail", "a few symbols carry it"),),
        "used": (),
        "data": "fc.raw",
        "costs": COSTS,
        "execution": Execution(every="1h"),
        "universe": Universe(Use("perp.pairs", {"venues": ["a", "b"]}), clusters="symbol"),
        "outcome": CARRY,
        "horizons": ("8h", "24h", "72h"),
        "forward": Forward("fc.raw.live"),
    }
    return Contract(**{**base, **kw})  # type: ignore[arg-type]


def test_a_complete_contract_serializes_with_its_plugins_and_their_options() -> None:
    contract = _contract(budget={"feat": 12, "signal": 8})
    plain = json.loads(json.dumps(asdict(contract)))
    assert plain["universe"]["use"]["options"] == {"venues": ["a", "b"]}
    assert plain["outcome"] == {"name": "ref.carry", "options": {"flow": "funding_rate"}}
    assert plain["horizons"] == ["8h", "24h", "72h"]
    assert plain["costs"]["name"] == "ref.flat_costs"
    assert plain["decisive"][0]["target"] == {"name": "perp.breakeven", "options": {}}
    assert (contract.budget_for("feat"), contract.budget_for("algo")) == (12, 20)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"claim": "TODO: say it"}, "claim"),
        ({"decisive": ()}, "decisive"),
        ({"alternatives": ()}, "alternative"),
        ({"alternatives": (Alternative("chance", "luck"),)}, "engine adds chance"),
        ({"alternatives": (Alternative("timing", "luck"),)}, "engine adds timing"),
        ({"used": None}, "used=()"),
        ({"forward": None}, "forward"),
        ({"execution": None}, "execution"),
        ({"consumer": "fc.backtest"}, "not both"),
        ({"universe": None}, "per candidate"),
        ({"outcome": None}, "name the outcome plugin"),
        ({"horizons": ()}, "at least one horizon"),
        ({"horizons": ("24h", "8h")}, "ascending"),
        ({"decisive": (Effect(), Effect("ref.mean"))}, "ref.mean twice"),
        ({"budget": {"feat": 0}}, "budget"),
    ],
)
def test_contracts_are_refused_until_every_question_is_answered(change: dict[str, object], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        _contract(**change)


def test_a_sealed_segment_must_be_fresh_and_follow_a_fixed_explore_end() -> None:
    with pytest.raises(ContractError, match="fixed explore_end"):
        _contract(seal_end="2026-10-01T00:00:00Z")
    with pytest.raises(ContractError, match="overlaps data already used"):
        _contract(explore_end="2026-06-01T00:00:00Z", seal_end="2026-10-01T00:00:00Z", used=(USED,))
    assert _contract(explore_end="2026-09-15T00:00:00Z", seal_end="2026-10-01T00:00:00Z", used=(USED,), forward=None)


def test_a_contract_names_a_consumer_only_for_its_decision_times_and_reserves_only_chance_and_timing() -> None:
    b = _contract(execution=None, consumer="fc.backtest", used=(USED,))
    assert (b.consumer, b.decisive[0].name) == (Use("fc.backtest"), "perp.breakeven")
    assert _contract(alternatives=(Alternative("fewer", "fewer candidates alone carry it"),))
    with pytest.raises(ContractError, match="affects must be some of claim, costs, outcome"):
        Usage("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "tuned", ("baseline",))


def test_an_effect_names_only_its_target_and_goes_by_its_name() -> None:
    assert Effect().name == "ref.mean" and Effect(Use("perp.breakeven", {"k": 1})).name == "perp.breakeven"
    with pytest.raises(TypeError):
        Effect(on="carry")  # type: ignore[call-arg]


def test_without_horizons_a_contract_takes_the_decision_period_times_1_4_and_16() -> None:
    assert _contract(horizons=None).horizons is None
    assert default_horizons("1h") == ("1h", "4h", "16h")
    assert default_horizons("24h") == ("1d", "4d", "16d")
    assert default_horizons("15min") == ("15min", "1h", "4h")


def test_evals_and_nulls_are_validated() -> None:
    def ic(table):
        return table

    assert Eval(ic).name == "ic" and Eval(partial(ic), name="ic again").name == "ic again"
    with pytest.raises(ValueError, match="combine"):
        Eval(ic, combine="sum")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lambda"):
        Eval(lambda table: table)
    with pytest.raises(ValueError, match="on_fail"):
        Eval(ic, on_fail="drop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="explanation"):
        Null(explains=" ", world=PermuteWithin(bucket_by="time"))
    with pytest.raises(ValueError, match="20"):
        Null(explains="chance", world=PermuteWithin(bucket_by="time"), reps=5)
    with pytest.raises(ValueError, match="PermuteWithin"):
        Null(explains="chance", world="shift time")  # type: ignore[arg-type]
    assert Null(explains="tail", world=PermuteWithin(bucket_by="symbol"), evals=["a"]).evals == ("a",)  # type: ignore[arg-type]
    h0 = "the scores do not rank the label"
    listed = [Eval(ic, h0=h0), Eval(ic, name="b", h0=h0, horizons=["8h"])]
    feat = evals_of(types.SimpleNamespace(EVALS=listed), "feat")
    assert [(e.name, e.combine, e.horizons) for e in feat] == [("ic", "max", None), ("b", "max", ("8h",))]

    def rows(wide, sources):
        return wide

    data = evals_of(types.SimpleNamespace(EVALS=[Eval(rows, h0=h0, on_fail="block")]), "data")
    assert [(e.name, e.on_fail) for e in data] == [("rows", "block")]
    with pytest.raises(ValueError, match="eval 'ic' takes table: a data eval takes wide, sources, or both"):
        evals_of(types.SimpleNamespace(EVALS=[Eval(ic, h0=h0)]), "data")
    with pytest.raises(ValueError, match="unique; give these a name=: ic"):
        evals_of(types.SimpleNamespace(EVALS=[Eval(ic, h0=h0), Eval(ic, h0=h0)]), "feat")
    with pytest.raises(ValueError, match="sets on_fail, which only data.py reads"):
        evals_of(types.SimpleNamespace(EVALS=[Eval(ic, on_fail="ask")]), "feat")
    with pytest.raises(ValueError, match="sets label, which only feat.py reads"):
        evals_of(types.SimpleNamespace(EVALS=[Eval(ic, label="y_flow")]), "data")
    with pytest.raises(ValueError, match="label names the column read as y"):
        Eval(ic, label="carry")
    with pytest.raises(ValueError, match="label names the column read as y"):
        Eval(ic, label="y_end")
    with pytest.raises(ValueError, match="EVALS must be a list"):
        evals_of(types.SimpleNamespace(EVALS=ic), "data")
    with pytest.raises(ValueError, match="on_fail is mask, drop_insts, ask or block"):
        Eval(ic, on_fail="drop")  # type: ignore[arg-type]


def test_a_lab_eval_must_name_its_h0_and_a_library_eval_must_not() -> None:
    """A lab eval is a claim test and names the H0 it rejects; a library eval tests a generic H0, is shown, and never
    vetoes, so h0 and on_fail='block' are refused on one. Each refusal says what to write."""
    from research_ci.eval.data.sanity import nonpositive
    from research_ci.eval.statistics.ic import rank_ic

    def flags(sources):
        return sources

    for layer, example in (("feat", "the scores do not rank the flow part"), ("data", "every close is a price")):
        with pytest.raises(ValueError) as refused:
            evals_of(types.SimpleNamespace(EVALS=[flags]), layer)  # type: ignore[arg-type]
        said = str(refused.value)
        assert said.startswith("the lab eval 'flags' names no h0: a lab eval tests the claim and can veto it")
        assert "write Eval(flags, h0='<the H0>', ...) in EVALS, such as h0=" in said and example in said
    assert evals_of(types.SimpleNamespace(EVALS=[Eval(flags, h0="  the data \n is sound ")]), "data")[0].h0 == (
        "the data is sound"
    )
    assert evals_of(types.SimpleNamespace(EVALS=[rank_ic]), "feat")[0].h0 == ""
    with pytest.raises(ValueError, match="the library eval 'rank_ic' sets h0: an eval of the engine's library tests"):
        evals_of(types.SimpleNamespace(EVALS=[Eval(rank_ic, h0="x")]), "feat")
    with pytest.raises(ValueError, match="the library eval 'nonpositive' sets on_fail='block': only a lab eval"):
        evals_of(types.SimpleNamespace(EVALS=[Eval(partial(nonpositive, levels=("close",)), on_fail="block")]), "data")
    with pytest.raises(ValueError, match="h0 is one sentence"):
        Eval(flags, h0=1)  # type: ignore[arg-type]
