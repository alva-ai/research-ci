"""The steps derive takes on every contract, after its default list (``stages/defaults/contract.py``): the protocol
tests of the universe, the labels and the targets, and the power of each decisive effect."""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, Effect, Label, Use, as_use, span
from research_ci.api.plugins import CostModel, LabelPlugin, TargetPlugin
from research_ci.eval import protocol as pr
from research_ci.eval.data import concentration as cc
from research_ci.kernel.time.clock import span_ns, time_series
from research_ci.kernel.verdict import power as pw
from research_ci.runner.plugins import Registry, factory_of
from research_ci.stages.common import per_t, quantile_times, units
from research_ci.stages.steps import DEFAULTS, Found, Step


@dataclass(frozen=True)
class Decisive:
    """A decisive effect as derive weighs it: its label, the target it names, and the label's rows at each horizon
    joined with the candidates, in the label's order."""

    effect: Effect
    label: Label
    use: Use
    target: TargetPlugin
    rows: tuple[tuple[str, pl.DataFrame], ...]


@dataclass(frozen=True)
class Seen:
    """What a contract step reads: the contract, the explore data, when its newest row became known and now, its
    first time and the end of the explore data (all in ns); the decision period and times, the candidates and how the
    universe computes them (``None`` without a universe), the times the universe checks cut the data at, how many times
    the label checks cut it at; the label plugins by name, the cost model, the decisive effects, the share of the
    noise above which one cluster is a warning, and the label a step of ``PER_LABEL`` checks."""

    contract: Contract
    registry: Registry | None
    fields: pl.DataFrame
    latest: int
    now: int
    start: int
    end: int
    period: int
    times: list[int]
    candidates: pl.DataFrame | None
    universe: Callable[[pl.DataFrame, list[int]], pl.DataFrame] | None
    cuts: list[int]
    label_cuts: int
    labels: dict[str, LabelPlugin]
    costs: CostModel
    decisive: tuple[Decisive, ...]
    heavy: float
    label: Label | None = None

    @property
    def this_label(self) -> Label:
        assert self.label is not None
        return self.label


def universe_pit(seen: Seen) -> Found:
    """With the data cut at each cut point, the candidates up to a period before the cut must equal those from all
    data."""
    if seen.universe is None or seen.candidates is None:
        return Found()
    return Found([pr.universe_pit(seen.universe, seen.fields, seen.candidates, seen.cuts, seen.period)])


def universe_causal(seen: Seen) -> Found:
    """Values that become known after the middle cut point must leave the candidates at it unchanged."""
    if seen.universe is None or seen.candidates is None or not seen.cuts:
        return Found()
    return Found([pr.universe_causal(seen.universe, seen.fields, seen.candidates, seen.cuts[len(seen.cuts) // 2])])


def membership(seen: Seen) -> Found:
    """A candidate list fixed with hindsight (``membership="static"``) leaves history explore-grade for it."""
    u = seen.contract.universe
    if u is None or u.membership != "static":
        return Found()
    return Found(warnings=["the candidate list was fixed with hindsight; history is explore-grade for it"])


def label_cases(seen: Seen) -> Found:
    """The label's worked examples must hold, and must move when their inputs move."""
    plugin = seen.labels[seen.this_label.name]
    return Found(pr.label_cases(seen.this_label.name, plugin.values, plugin.cases()))


def label_fields(seen: Seen) -> Found:
    """A label option, given or by default, that names text which is neither a field of the data nor a candidate
    column: a warning, or a failure when the label then gives no rows at all."""
    check = _named_fields(seen, seen.this_label)
    return Found([check] if check is not None else [])


def label_truncation(seen: Seen) -> Found:
    """The label at its shortest and longest horizon: the labels whose window ends by a cut must be the same when the
    data stops at the cut."""
    if seen.candidates is None:
        return Found()
    label, cuts = seen.this_label, quantile_times(time_series(seen.times), seen.label_cuts)
    values = seen.labels[label.name].values
    horizons = sorted({label.horizons[0], label.horizons[-1]}, key=pd.Timedelta)
    return Found(
        [pr.label_truncation(label.name, values, seen.fields, seen.candidates, span(h), cuts) for h in horizons]
    )


def power(seen: Seen) -> Found:
    """Per decisive effect and horizon, the smallest gain over random the data can detect (MDE), from the dispersion
    of the target alone, and the cluster that carries the most of that noise. A number that is not finite at some
    decision times is left out of the power, and a cluster that carries more than ``heavy`` of the noise is a warning;
    the contract is blocked when no horizon of an effect can measure its noise."""
    found = Found()
    found.shown["decisive"] = [_power(seen, d, found.warnings) for d in seen.decisive]
    return found


def target_reads(seen: Seen) -> Found:
    """Each decisive target, on the label rows of its first horizon, must give other values for other scores."""
    found = Found()
    for d in seen.decisive:
        h, table = d.rows[0]
        reduce = partial(_reduce, d.target, span(h), seen.costs)
        found.checks.append(pr.target_reads_score(d.use.name, reduce, table))
    return found


PER_LABEL = (
    Step("label: worked examples", "blocks", label_cases, "Worked examples hold, and move when inputs move"),
    Step("label: fields", "blocks", label_fields, "Label options name fields; fails when no rows result"),
    Step("label: no data after t + h", "blocks", label_truncation, "Labels unchanged when the data stops early"),
)
STEPS = (
    DEFAULTS,
    Step("universe point-in-time", "blocks", universe_pit, "Same candidates when the data is cut"),
    Step("universe causal", "blocks", universe_causal, "Later values leave earlier candidates unchanged"),
    Step("universe membership", "warns", membership, "A static candidate list makes history explore-grade"),
    *PER_LABEL,
    Step("power", "blocks", power, "MDE per horizon; blocks when no noise is measurable"),
    Step("target reads the score", "blocks", target_reads, "The decisive target moves with the scores"),
)


def runs(listed: Sequence[Step[Seen]], seen: Seen) -> list[tuple[Step[Seen], Seen]]:
    """Each step with what it reads: the steps of ``PER_LABEL`` once per declared label, all of one label together, in
    the contract's order."""
    out: list[tuple[Step[Seen], Seen]] = []
    for step in listed:
        if step not in PER_LABEL:
            out.append((step, seen))
        elif step == PER_LABEL[0]:
            out += [(s, replace(seen, label=label)) for label in seen.contract.labels for s in PER_LABEL]
    return out


def _reduce(target: TargetPlugin, horizon: pd.Timedelta, costs: CostModel, table: pl.DataFrame) -> Any:
    return target.reduce(table, horizon, costs)


def _power(seen: Seen, d: Decisive, warnings: list[str]) -> dict[str, Any]:
    u = seen.contract.universe
    assert u is not None
    width = seen.end - seen.start
    rows, heavy, infinite = [], {}, []
    for h, joined in d.rows:
        horizon = span(h)
        raw = per_t(d.target.reduce(joined, horizon, seen.costs))
        values = raw.filter(pl.col("value").is_finite())
        if values.height < raw.height:
            infinite.append(f"{raw.height - values.height} at {h}")
        p = pw.power(pw.block_se(values, span_ns(horizon)), width, span_ns(horizon))
        top = cc.shares(joined, u.clusters, span_ns(horizon)).head(1).rows()
        if top and top[0][1] > seen.heavy:
            heavy.setdefault(top[0][0], []).append(f"{top[0][1]:.0%} at {h}")
        rows.append({"horizon": h, "decisions": values.height, **asdict(p), "concentration": top[:1]})
    if infinite:
        warnings.append(
            f"{d.effect.on}: the number is not finite at some decision times ({', '.join(infinite)}), left out of "
            "the power; rci data check looks at whether the data is fit for research"
        )
    for name, where in heavy.items():
        warnings.append(
            f"{d.effect.on}: {u.clusters} {name} carries {' and '.join(where)} of the noise the MDE rests on; rci "
            "data check looks at whether its data is fit for research"
        )
    out = {"kind": "label", **units(d.effect, d.target), "target": d.use.name, "options": dict(d.use.options)}
    out["horizons"] = rows
    return out | {"label": {"name": d.label.name, "plugin": as_use(d.label.use).name}}


def _named_fields(seen: Seen, label: Label) -> pr.Check | None:
    use = as_use(label.use)
    present = sorted(seen.fields["field"].unique().to_list())
    columns = set(seen.candidates.columns) if seen.candidates is not None else set()
    options = _text_defaults(factory_of(use.name, seen.registry)) | {
        k: v for k, v in use.options.items() if isinstance(v, str)
    }
    absent = [f"{k}={v!r}" for k, v in sorted(options.items()) if v not in present and v not in columns]
    if not absent:
        return None
    h = label.horizons[0]
    values = seen.labels[label.name].values
    try:
        rows = 0 if seen.candidates is None else values(seen.fields, seen.candidates, span(h)).height
    except Exception:  # noqa: BLE001
        rows = 0
    detail = f"{use.name} option {', '.join(absent)} is not a field of the data; its fields are {', '.join(present)}"
    if rows:
        return pr.Check(f"label {label.name}: fields", "warn", detail)
    return pr.Check(f"label {label.name}: fields", "fail", f"{detail}; the label gives no rows at {h}")


def _text_defaults(factory: Callable[..., Any]) -> dict[str, str]:
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return {}
    return {p.name: p.default for p in params if isinstance(p.default, str)}
