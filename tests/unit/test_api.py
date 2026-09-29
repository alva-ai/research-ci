import json
import types
from dataclasses import asdict, replace

import pytest

from research_ci.api import (
    Alternative,
    Baseline,
    Branch,
    Contract,
    ContractError,
    Criterion,
    Effect,
    Execution,
    Forward,
    FullPath,
    Label,
    Null,
    PermuteWithin,
    Perturb,
    Universe,
    Usage,
    Use,
    criteria_of,
)

CARRY = Label("carry", Use("ref.carry", {"flow": "funding_rate"}), horizons=("8h", "24h", "72h"))
COSTS = Use("ref.flat_costs", {"commission_bps": 5, "slippage_bps": 5, "legs": 2})
USED = Usage("2026-04-08T00:00:00Z", "2026-09-14T12:00:00Z", "fc parameters tuned", ("baseline",))


def _contract(**kw: object) -> Contract:
    base: dict[str, object] = {
        "template": "factor",
        "claim": "cross-venue funding spreads persist",
        "decisive": (Effect(on="carry", at_least=0.001, unit="per entry", target="perp.breakeven"),),
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
        ({"used": None}, "used=()"),
        ({"forward": None}, "forward"),
        ({"execution": None}, "execution"),
        ({"consumer": "fc.backtest"}, "not both"),
        ({"universe": None}, "per candidate"),
        ({"labels": (CARRY, CARRY)}, "unique"),
        ({"decisive": (Effect(on="return", at_least=0.01, unit="x"),)}, "not a label"),
        ({"decisive": (Effect(on="carry", at_least=0.01, unit="x", pilot=Perturb("drop", 0.2)),)}, "pilot and design"),
        ({"budget": {"feat": 0}}, "budget"),
        ({"deadline": "-1 days"}, "deadline"),
        ({"skip": ("signal",)}, "existing strategy"),
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


def test_improving_an_existing_strategy_names_its_files_and_how_to_measure_chance() -> None:
    base = _contract()
    consumer = Effect(
        on="consumer",
        at_least=0.05,
        unit="per year over the baseline",
        pilot=Perturb("drop", 0.2),
        design=Branch(every="7d", horizon="7d", tail="7d"),
    )
    b = replace(
        base,
        decisive=(consumer,),
        execution=None,
        consumer="fc.backtest",
        baseline=Baseline("stable_score", config="fc.toml", code=("fc.py",)),
        skip=("signal", "algo"),
        used=(USED,),
    )
    assert b.skip == ("signal", "algo")
    with pytest.raises(ContractError, match="declare the pilot"):
        replace(b, decisive=(Effect(on="consumer", at_least=0.05, unit="x"),))
    with pytest.raises(ContractError, match="declare the design"):
        replace(b, decisive=(Effect(on="consumer", at_least=0.05, unit="x", pilot=Perturb("drop", 0.2)),))
    with pytest.raises(ContractError, match="tail"):
        Branch(every="7d", horizon="7d", tail="-1h")
    assert replace(b, decisive=(replace(consumer, design=FullPath()),)).decisive[0].design == FullPath()
    with pytest.raises(ContractError, match="target applies to label effects"):
        replace(b, decisive=(Effect(on="consumer", at_least=0.05, unit="x", target="t", pilot=Perturb("drop", 0.2)),))
    with pytest.raises(ContractError, match="positive"):
        Effect(on="consumer", at_least=0.0, unit="x")
    with pytest.raises(ContractError, match="share"):
        Perturb("shuffle", 1.5)


def test_criteria_and_nulls_are_validated() -> None:
    Criterion("predictive", evaluate=lambda outputs, ctx: 0.0)
    with pytest.raises(ValueError, match="limit"):
        Criterion("drawdown", evaluate=lambda outputs, ctx: 0.0, kind="limit")
    with pytest.raises(ValueError, match="explanation"):
        Null(explains=" ", world=PermuteWithin(bucket_by="time"))
    with pytest.raises(ValueError, match="20"):
        Null(explains="chance", world=PermuteWithin(bucket_by="time"), reps=5)
    ok = types.SimpleNamespace(CRITERIA=[Criterion("a", evaluate=lambda o, c: 0.0)])
    assert [c.name for c in criteria_of(ok)] == ["a"]
    with pytest.raises(ValueError, match="unique"):
        criteria_of(types.SimpleNamespace(CRITERIA=[Criterion("a", evaluate=lambda o, c: 0.0)] * 2))
