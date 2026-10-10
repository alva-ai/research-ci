import json
import types
from dataclasses import asdict

import pytest

from research_ci.api import (
    Alternative,
    Contract,
    ContractError,
    Criterion,
    Effect,
    Execution,
    Forward,
    Label,
    Null,
    PermuteWithin,
    Universe,
    Usage,
    Use,
    criteria_of,
)

CARRY = Label("carry", Use("ref.carry", {"flow": "funding_rate"}), horizons=("8h", "24h", "72h"))
COSTS = Use("ref.flat_costs", {"commission_bps": 5, "slippage_bps": 5, "legs": 2})
USED = Usage("2026-04-08T00:00:00Z", "2026-09-14T12:00:00Z", "fc parameters tuned", ("claim",))


def _contract(**kw: object) -> Contract:
    base: dict[str, object] = {
        "claim": "cross-venue funding spreads persist",
        "decisive": (Effect(on="carry", target="perp.breakeven"),),
        "alternatives": (Alternative("tail", "a few symbols carry it"),),
        "used": (),
        "data": "fc.raw",
        "costs": COSTS,
        "execution": Execution(every="1h"),
        "universe": Universe(Use("perp.pairs", {"venues": ["a", "b"]}), clusters="symbol"),
        "labels": (CARRY,),
        "forward": Forward("fc.raw.live"),
    }
    return Contract(**{**base, **kw})  # type: ignore[arg-type]


def test_a_complete_contract_serializes_with_its_plugins_and_their_options() -> None:
    contract = _contract(budget={"feat": 12, "signal": 8})
    plain = json.loads(json.dumps(asdict(contract)))
    assert plain["universe"]["use"]["options"] == {"venues": ["a", "b"]}
    assert plain["labels"][0]["use"] == {"name": "ref.carry", "options": {"flow": "funding_rate"}}
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
        ({"labels": (CARRY, CARRY)}, "unique"),
        ({"decisive": (Effect(on="return"),)}, "not a label"),
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
    assert (b.consumer, b.decisive[0].on) == (Use("fc.backtest"), "carry")
    assert _contract(alternatives=(Alternative("fewer", "fewer candidates alone carry it"),))
    with pytest.raises(ContractError, match="affects must be some of claim, costs, labels"):
        Usage("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "tuned", ("baseline",))


def test_a_backtest_is_refused_as_the_decisive_effect_because_the_algo_layer_judges_it() -> None:
    effect = Effect(on="consumer")
    with pytest.raises(ContractError, match="algo layer") as refused:
        _contract(decisive=(effect,), execution=None, consumer="fc.backtest")
    assert "label effect" in str(refused.value) and "top" in str(refused.value)


def test_criteria_and_nulls_are_validated() -> None:
    Criterion("predictive", evaluate=lambda table: table)
    with pytest.raises(ValueError, match="combine"):
        Criterion("predictive", evaluate=lambda table: table, combine="sum")
    with pytest.raises(ValueError, match="explanation"):
        Null(explains=" ", world=PermuteWithin(bucket_by="time"))
    with pytest.raises(ValueError, match="20"):
        Null(explains="chance", world=PermuteWithin(bucket_by="time"), reps=5)
    with pytest.raises(ValueError, match="PermuteWithin"):
        Null(explains="chance", world="shift time")  # type: ignore[arg-type]
    assert Null(explains="tail", world=PermuteWithin(bucket_by="symbol")).keeps == "symbol"
    ok = types.SimpleNamespace(CRITERIA=[Criterion("a", evaluate=lambda table: table)])
    assert [c.name for c in criteria_of(ok)] == ["a"]
    assert criteria_of(types.SimpleNamespace()) == [] == criteria_of(types.SimpleNamespace(CRITERIA=[]))
    with pytest.raises(ValueError, match="unique"):
        criteria_of(types.SimpleNamespace(CRITERIA=[Criterion("a", evaluate=lambda table: table)] * 2))
