"""The steps derive takes on every contract, after its default list (``stages/contract/defaults.py``): the protocol
tests of the universe, the outcome and the targets, and the power of each decisive effect."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, Effect, Use, as_use, span
from research_ci.api.plugins import CostModel, OutcomePlugin, TargetPlugin
from research_ci.eval import protocol as pr
from research_ci.eval.data import concentration as cc
from research_ci.kernel.tables import values
from research_ci.kernel.time.clock import span_ns, time_series
from research_ci.kernel.verdict import power as pw
from research_ci.runner.plugins import Registry, factory_of
from research_ci.stages.common import UNSCORED, outcome_rows, per_t, quantile_times, units
from research_ci.stages.skeleton import DEFAULTS, Found, Step


@dataclass(frozen=True)
class Decisive:
    """A decisive effect as derive weighs it: the target it names, and the outcome's rows at each horizon joined with
    the candidates, in the contract's order."""

    effect: Effect
    use: Use
    target: TargetPlugin
    rows: tuple[tuple[str, pl.DataFrame], ...]


@dataclass(frozen=True)
class Seen:
    """What a contract step reads: the contract, the explore sources, when their newest row became known and now,
    their first time and the end of the explore data (all in ns); the decision period and times, the candidates and
    how the universe computes them on sources and decision times, building ``wide`` on them (``None`` without a
    universe), the times the universe checks cut the data at, how many times the outcome checks cut it at; the outcome
    plugin and the horizons, the cost model, the decisive effects, and the share of the noise above which one cluster
    is a warning."""

    contract: Contract
    registry: Registry | None
    sources: Mapping[str, pl.DataFrame]
    latest: int
    now: int
    start: int
    end: int
    period: int
    times: list[int]
    candidates: pl.DataFrame | None
    universe: Callable[[Mapping[str, pl.DataFrame], list[int]], pl.DataFrame] | None
    cuts: list[int]
    outcome_cuts: int
    outcome: OutcomePlugin
    horizons: tuple[str, ...]
    costs: CostModel
    decisive: tuple[Decisive, ...]
    heavy: float


def universe_pit(seen: Seen) -> Found:
    """With the sources cut at each cut point and ``wide`` built again on them, the candidates up to a period before
    the cut must equal those from all data."""
    if seen.universe is None or seen.candidates is None:
        return Found()
    check = pr.universe_pit(seen.universe, seen.sources, seen.candidates, seen.times, seen.cuts, seen.period)
    return Found([check])


def universe_causal(seen: Seen) -> Found:
    """Values that become known after the middle cut point must leave the candidates at it unchanged."""
    if seen.universe is None or seen.candidates is None or not seen.cuts:
        return Found()
    t = seen.cuts[len(seen.cuts) // 2]
    return Found([pr.universe_causal(seen.universe, seen.sources, seen.candidates, seen.times, t)])


def membership(seen: Seen) -> Found:
    """A candidate list fixed with hindsight (``membership="static"``) leaves history explore-grade for it."""
    u = seen.contract.universe
    if u is None or u.membership != "static":
        return Found()
    return Found(warnings=["the candidate list was fixed with hindsight; history is explore-grade for it"])


def outcome_cases(seen: Seen) -> Found:
    """The outcome's worked examples must hold, and must move when their inputs move."""
    return Found(pr.outcome_cases(_name(seen), _values(seen.outcome), seen.outcome.cases()))


def outcome_columns(seen: Seen) -> Found:
    """An outcome option, given or by default, that names text which is neither a value column of the sources nor a
    candidate column: a warning, or a failure when the outcome then gives no rows at all."""
    check = _named_columns(seen)
    return Found([check] if check is not None else [])


def outcome_truncation(seen: Seen) -> Found:
    """The outcome at its shortest and longest horizon: the outcomes whose window ends by a cut must be the same when
    the sources stop at the cut, and none may say it reads past ``t + h``."""
    if seen.candidates is None:
        return Found()
    cuts = quantile_times(time_series(seen.times), seen.outcome_cuts)
    horizons = sorted({seen.horizons[0], seen.horizons[-1]}, key=pd.Timedelta)
    values = _values(seen.outcome)
    return Found(
        [pr.outcome_truncation(_name(seen), values, seen.sources, seen.candidates, span(h), cuts) for h in horizons]
    )


def power(seen: Seen) -> Found:
    """Per decisive effect and horizon, the smallest gain over random the data can detect (MDE), from the dispersion
    of the target alone, and the cluster that carries the most of that noise. A number that is not finite at some
    decision times is left out of the power, and a cluster that carries more than ``heavy`` of the noise is a warning,
    unless it is the only one; the contract is blocked when no horizon of an effect can measure its noise."""
    found = Found()
    found.shown["decisive"] = [_power(seen, d, found.warnings) for d in seen.decisive]
    return found


def target_reads(seen: Seen) -> Found:
    """Each decisive target, on the outcome rows of its first horizon, must give other values for other scores; for
    events, scores that leave some rows without one."""
    found = Found()
    u = seen.contract.universe
    missing = UNSCORED if u is not None and u.shape == "events" else 0.0
    for d in seen.decisive:
        h, table = d.rows[0]
        reduce = partial(_reduce, d.target, span(h), seen.costs)
        found.checks.append(pr.target_reads_score(d.use.name, reduce, table, missing))
    return found


STEPS = (
    DEFAULTS,
    Step("universe point-in-time", "blocks", universe_pit, "Same candidates when the data is cut"),
    Step("universe causal", "blocks", universe_causal, "Later values leave earlier candidates unchanged"),
    Step("universe membership", "warns", membership, "A static candidate list makes history explore-grade"),
    Step("outcome: worked examples", "blocks", outcome_cases, "Worked examples hold, and move when inputs move"),
    Step("outcome: columns", "blocks", outcome_columns, "Outcome options name columns; fails when no rows result"),
    Step("outcome: no data after t + h", "blocks", outcome_truncation, "Outcomes unchanged when the data stops early"),
    Step("power", "blocks", power, "MDE per horizon; blocks when no noise is measurable"),
    Step("target reads the score", "blocks", target_reads, "The decisive target moves with the scores"),
)


def _name(seen: Seen) -> str:
    assert seen.contract.outcome is not None
    return as_use(seen.contract.outcome).name


def _values(plugin: OutcomePlugin) -> pr.Values:
    """The outcome's ``values``, its rows as the engine reads them."""
    return lambda sources, cands, h: outcome_rows(plugin.values(sources, cands, h))


def _reduce(target: TargetPlugin, horizon: pd.Timedelta, costs: CostModel, table: pl.DataFrame) -> Any:
    return target.reduce(table, horizon, costs)


def _power(seen: Seen, d: Decisive, warnings: list[str]) -> dict[str, Any]:
    u = seen.contract.universe
    assert u is not None
    width = seen.end - seen.start
    found, heavy, infinite = [], {}, []
    for h, joined in d.rows:
        horizon = span(h)
        raw = per_t(d.target.reduce(joined, horizon, seen.costs))
        numbers = raw.filter(pl.col("value").is_finite())
        if numbers.height < raw.height:
            infinite.append(f"{raw.height - numbers.height} at {h}")
        p = pw.power(pw.block_se(numbers, span_ns(horizon)), width, span_ns(horizon))
        shares = cc.shares(joined, u.clusters, span_ns(horizon))
        top = shares.head(1).rows()
        if top and top[0][1] > seen.heavy and shares.height > 1:
            heavy.setdefault(top[0][0], []).append(f"{top[0][1]:.0%} at {h}")
        found.append({"horizon": h, "decisions": numbers.height, **asdict(p), "concentration": top[:1]})
    if infinite:
        warnings.append(
            f"{d.effect.name}: the number is not finite at some decision times ({', '.join(infinite)}), left out of "
            "the power; rci data check looks at whether the data is fit for research"
        )
    for name, where in heavy.items():
        warnings.append(
            f"{d.effect.name}: {u.clusters} {name} carries {' and '.join(where)} of the noise the MDE rests on; rci "
            "data check looks at whether its data is fit for research"
        )
    out = {"kind": "outcome", **units(d.effect, d.target), "target": d.use.name, "options": dict(d.use.options)}
    return out | {"horizons": found}


def _named_columns(seen: Seen) -> pr.Check | None:
    use = as_use(seen.contract.outcome) if seen.contract.outcome is not None else None
    assert use is not None
    present = sorted(c for t in seen.sources.values() for c in values(t))
    columns = set(seen.candidates.columns) if seen.candidates is not None else set()
    options = _text_defaults(factory_of(use.name, seen.registry)) | {
        k: v for k, v in use.options.items() if isinstance(v, str)
    }
    absent = [f"{k}={v!r}" for k, v in sorted(options.items()) if v not in present and v not in columns]
    if not absent:
        return None
    h = seen.horizons[0]
    try:
        found = 0 if seen.candidates is None else seen.outcome.values(seen.sources, seen.candidates, span(h)).height
    except Exception:  # noqa: BLE001
        found = 0
    detail = (
        f"{use.name} option {', '.join(absent)} is not a column of the sources; their columns are {', '.join(present)}"
    )
    if found:
        return pr.Check(f"outcome {use.name}: columns", "warn", detail)
    return pr.Check(f"outcome {use.name}: columns", "fail", f"{detail}; the outcome gives no rows at {h}")


def _text_defaults(factory: Callable[..., Any]) -> dict[str, str]:
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return {}
    return {p.name: p.default for p in params if isinstance(p.default, str)}
