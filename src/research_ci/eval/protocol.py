"""Protocol tests the engine runs on plugins before using their output. Each returns a ``Check``. Plugins are only
called through the callables passed in, so the engine decides what data they see: the sources, cut or changed, from
which a universe's ``wide`` is built again; nothing here knows what a column means."""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import span
from research_ci.api.plugins import ConsumerResult, CostModel, OutcomeCase
from research_ci.kernel.tables import values
from research_ci.kernel.time.asof import known
from research_ci.kernel.time.clock import lit_time
from research_ci.kernel.verdict.check import Check, Status

Sources = Mapping[str, pl.DataFrame]
Candidates = Callable[[Sources, list[int]], pl.DataFrame]
Values = Callable[[Sources, pl.DataFrame, pd.Timedelta], pl.DataFrame]
Run = Callable[[Sources, pl.DataFrame | None], ConsumerResult]
COMPONENTS = ("flow", "price", "fees")


def changed(sources: Sources, where: pl.Expr, how: Callable[[pl.Expr], pl.Expr]) -> dict[str, pl.DataFrame]:
    """The sources with the values of some rows changed. Every value column of the rows ``where`` holds is changed by
    ``how``."""
    return {
        name: table.with_columns(
            pl.when(where).then(how(pl.col(c))).otherwise(pl.col(c)).alias(c) for c in values(table)
        )
        for name, table in sources.items()
    }


def universe_pit(
    candidates: Candidates,
    sources: Sources,
    full: pl.DataFrame,
    times: Sequence[int],
    cuts: Sequence[int],
    margin_ns: int,
) -> Check:
    """Candidates before the cut equal those from all data. With the sources cut at ``cut`` and ``wide`` built again on
    them up to ``cut - margin``, the candidates at ``cut - margin`` must equal those from all data. The margin lets a
    plugin that needs the next bar to finish one still pass; ``universe_causal`` covers shorter look-aheads."""
    numbers = [c for c, d in full.schema.items() if c not in ("t", "inst") and d.is_numeric()]
    bad = []
    for cut in cuts:
        t = cut - margin_ns
        part = candidates(known(sources, cut), [x for x in times if x <= t]).filter(pl.col("t") == lit_time(t))
        mismatch = _frame_mismatch(part, full.filter(pl.col("t") == lit_time(t)), ["t", "inst"], numbers)
        if mismatch:
            bad.append(f"{_iso(t)}: {mismatch}")
    if bad:
        return Check("universe point-in-time", "fail", "; ".join(bad))
    return Check("universe point-in-time", "pass", f"{len(cuts)} cut points give the same candidates")


def grid_pit(
    grid: Callable[[Sources], pl.DataFrame], sources: Sources, full: pl.DataFrame, cuts: Sequence[int]
) -> Check:
    """Decision times up to the cut equal those from all data. With the sources cut at ``cut``, the grid ``t | inst``
    the idea's sampler gives up to ``cut`` must equal the one it gives on all data: a decision time may depend only on
    rows known by then."""
    bad = []
    for cut in cuts:
        until = pl.col("t") <= lit_time(cut)
        mismatch = _frame_mismatch(grid(known(sources, cut)).filter(until), full.filter(until), ["t", "inst"], [])
        if mismatch:
            bad.append(f"up to {_iso(cut)}, {mismatch} (cut data vs all data)")
    if bad:
        why = "; ".join(bad)
        fix = "a decision time may depend only on rows known by then (available_at <= t)"
        return Check("grid point-in-time", "fail", f"the sources cut at a time change the decision times: {why}; {fix}")
    return Check("grid point-in-time", "pass", f"{len(cuts)} cut points give the same decision times")


def universe_causal(
    candidates: Candidates, sources: Sources, full: pl.DataFrame, times: Sequence[int], t: int
) -> Check:
    """Candidates at ``t`` ignore every value known after ``t``. Every value that becomes known after ``t`` is changed
    and ``wide`` built again: the candidates at ``t`` must not move."""
    moved = changed(sources, pl.col("available_at") > lit_time(t), lambda v: v * 1.01 + 1e-6)
    numbers = [c for c, d in full.schema.items() if c not in ("t", "inst") and d.is_numeric()]
    part = candidates(moved, list(times)).filter(pl.col("t") == lit_time(t))
    mismatch = _frame_mismatch(part, full.filter(pl.col("t") == lit_time(t)), ["t", "inst"], numbers)
    if mismatch:
        return Check("universe causal", "fail", f"values after {_iso(t)} changed the candidates at it: {mismatch}")
    return Check("universe causal", "pass", f"values after {_iso(t)} leave the candidates at it unchanged")


def outcome_cases(name: str, outcome: Values, cases: Sequence[OutcomeCase]) -> list[Check]:
    """Worked examples hold as written and move with their inputs. The author's worked examples must come out as
    written, and must move when their inputs move: shifted in time, perturbed inside the window, or missing a value
    column. Perturbing data after the window must change nothing."""
    if not cases:
        return [Check(f"outcome {name}: worked examples", "fail", "the outcome gives no worked examples")]
    wrong, blind = [], []
    for i, case in enumerate(cases):
        h = span(case.horizon)
        got = outcome(case.sources, case.candidates, h)
        columns = [c for c in case.expected.columns if c not in ("t", "inst")]
        mismatch = _frame_mismatch(got, case.expected, ["t", "inst"], columns)
        if mismatch:
            wrong.append(f"case {i}: {mismatch}")
            continue
        for mutation, tables, should_move in _mutations(case, h):
            moved = bool(_frame_mismatch(outcome(tables, case.candidates, h), case.expected, ["t", "inst"], columns))
            if moved != should_move:
                blind.append(f"case {i}: {mutation} {'did not change' if should_move else 'changed'} the outcome")
    checks = [
        Check(f"outcome {name}: worked examples", "fail" if wrong else "pass", "; ".join(wrong) or f"{len(cases)} ok")
    ]
    detail = "; ".join(blind) or "shift, in-window changes and missing columns move it; later data does not"
    checks.append(Check(f"outcome {name}: examples bite", "fail" if blind else "pass", detail))
    return checks


def outcome_truncation(
    name: str, outcome: Values, sources: Sources, cands: pl.DataFrame, horizon: pd.Timedelta, cuts: Sequence[int]
) -> Check:
    """Data stopping at ``cut`` leaves outcomes ending by it unchanged. Outcomes whose window ends by ``cut`` must be
    the same when the sources stop at ``cut``, and no row may say it reads past its ``t + h``: its ``y_end`` must be at
    most that."""
    h = int(horizon.value)
    bad = []
    for cut in cuts:
        rows = cands.filter(pl.col("t").dt.epoch("ns") + h <= cut)
        if rows.is_empty():
            continue
        a, b = outcome(known(sources, cut), rows, horizon), outcome(sources, rows, horizon)
        mismatch = _frame_mismatch(a, b, ["t", "inst"], ["y"])
        if mismatch:
            bad.append(f"cut {_iso(cut)}: {mismatch}")
        late = b.filter(pl.col("y_end") > pl.col("t") + pl.duration(nanoseconds=h, time_unit="ns")).height
        if late:
            bad.append(f"cut {_iso(cut)}: {late} rows have y_end after t + h")
    check = f"outcome {name}: no data after t + h"
    if bad:
        return Check(check, "fail", "; ".join(bad))
    return Check(check, "pass", f"{len(cuts)} cut points, horizon {_span_text(horizon)}")


def consumer_pit(run: Run, sources: Sources, full: ConsumerResult, cuts: Sequence[int], margin_ns: int) -> Check:
    """Cut data gives the full run's periods before the cut. A run on the sources cut at ``cut`` must match the full
    run for every period ending before ``cut - margin``."""
    bad = []
    for cut in cuts:
        part = run(known(sources, cut), None)
        until = lit_time(cut - margin_ns)
        mismatch = _frame_mismatch(
            part.net.filter(pl.col("t") <= until), full.net.filter(pl.col("t") <= until), ["t"], ["net"]
        )
        trades = _frame_mismatch(
            part.trades.filter(pl.col("t") <= until),
            full.trades.filter(pl.col("t") <= until),
            ["t", "inst"],
            ["to_weight"],
        )
        if mismatch or trades:
            bad.append(f"cut {_iso(cut)}: net {mismatch or 'same'}, trades {trades or 'same'}")
    if bad:
        return Check("consumer point-in-time", "fail", "; ".join(bad))
    return Check("consumer point-in-time", "pass", f"{len(cuts)} cut points match the full run before each cut")


def causal(run: Run, sources: Sources, base: ConsumerResult, at: int, period_ns: int) -> Check:
    """Changes known after ``at`` must move only what comes after. Every value that becomes known in
    ``(at, at + period]`` is changed: nothing up to ``at`` may move, and something after it must."""
    window = (pl.col("available_at") > lit_time(at)) & (pl.col("available_at") <= lit_time(at + period_ns))
    moved = changed(sources, window, lambda v: v * 1.01 + 1e-6)
    diff = _diff(run(moved, None).net, base.net).filter(pl.col("d").abs() > 1e-12)
    early = diff.filter(pl.col("t") <= lit_time(at))
    if not early.is_empty():
        return Check("consumer causal", "fail", f"data after {_iso(at)} changed the period ending {early['t'][0]}")
    if diff.is_empty():
        return Check("consumer causal", "fail", f"changing the data after {_iso(at)} changed nothing")
    return Check("consumer causal", "pass", f"data after {_iso(at)} first moves the period ending {diff['t'][0]}")


def busiest(sources: Sources, times: list[int], period: int) -> int:
    """A decision time whose next period brings the most sources. It is the one in the middle third whose next period
    brings rows of the most sources, so a causality test there touches every input, including sparse ones such as
    settlements."""
    middle = times[len(times) // 3 : 2 * len(times) // 3] or times
    t0 = times[0]
    at = pl.concat(
        [pl.DataFrame(schema={"a": pl.Int64, "source": pl.String})]
        + [t.select(pl.col("available_at").dt.epoch("ns").alias("a"), source=pl.lit(n)) for n, t in sources.items()]
    )
    slot = t0 + (pl.col("a") - 1 - t0) // period * period
    counts = at.with_columns(slot=slot).group_by("slot").agg(pl.col("source").n_unique().alias("kinds"))
    ranked = pl.DataFrame({"slot": middle}, schema={"slot": pl.Int64}).join(counts, on="slot", how="left")
    return int(ranked.fill_null(0).sort(["kinds", "slot"], descending=[True, False])["slot"][0])


def deterministic(a: ConsumerResult, b: ConsumerResult) -> Check:
    mismatch = _frame_mismatch(a.net, b.net, ["t"], ["net"])
    if mismatch:
        return Check("consumer deterministic", "fail", f"two identical runs differ: {mismatch}")
    return Check("consumer deterministic", "pass", "two identical runs give the same net")


def adds_up(result: ConsumerResult) -> list[Check]:
    """Components add up to net; the consumer's own checks pass."""
    present = [c for c in COMPONENTS if c in result.net.columns]
    if present:
        gap = result.net.select((pl.sum_horizontal(present) - pl.col("net")).abs().max()).item()
        status: Status = "pass" if gap is not None and gap < 1e-9 else "fail"
        out = [Check("net adds up", status, f"{' + '.join(present)} vs net: largest gap {gap:.2e}")]
    else:
        out = [Check("net adds up", "warn", "the consumer reports no components")]
    failed = [name for name, ok in result.checks.items() if not ok]
    out.append(
        Check("consumer self-checks", "fail" if failed else "pass", ", ".join(failed) or f"{len(result.checks)} ok")
    )
    return out


def fees_match(result: ConsumerResult, costs: CostModel) -> Check:
    """The consumer charges what the cost model says. What the consumer charged must be what the cost model says its
    trades cost."""
    if "fees" not in result.net.columns:
        return Check("fees match the cost model", "warn", "the consumer reports no fees")
    charged = -float(result.net["fees"].sum())
    model = costs.fees(result.trades)
    ratio = charged / model if model else (1.0 if charged == 0 else math.inf)
    status: Status = "pass" if abs(ratio - 1) < 0.01 else "fail"
    return Check("fees match the cost model", status, f"charged {charged:.4%}, cost model {model:.4%}")


def matches_outcome(result: ConsumerResult, outcome: pl.DataFrame, period_ns: int) -> Check:
    """The consumer books what the outcome says, period by period. The consumer and the outcome are written
    independently; on the positions held, they must agree on what each component booked every period earned (the
    outcome's part ``y_<component>``), period by period. A gap in the total is an error in size or sign; a gap that only
    shows period by period is money booked in another period than the outcome's, such as a flow at a decision time
    paid to the positions that decision opens. Either fails: verdicts are judged period by period."""
    shared = [c for c in result.marked if c in result.net.columns and f"y_{c}" in outcome.columns]
    if not shared:
        return Check("consumer matches the outcome", "skip", "no component booked every period by both")
    parts = outcome.select("t", "inst", *(pl.col(f"y_{c}").alias(f"{c}_outcome") for c in shared))
    held = result.positions.join(parts, on=["t", "inst"], how="left")
    missing = held.filter(pl.col(f"{shared[0]}_outcome").is_null()).height
    expected = held.group_by("t").agg(
        *(pl.col(f"{c}_outcome").fill_null(0.0).mul(pl.col("weight")).sum() for c in shared)
    )
    expected = expected.with_columns(pl.col("t") + pl.duration(nanoseconds=period_ns))
    joined = result.net.join(expected, on="t", how="left").fill_null(0.0)
    sums = joined.select(
        pl.concat_list(pl.lit(c) for c in shared).alias("c"),
        pl.concat_list((pl.col(c).sum() - pl.col(f"{c}_outcome").sum()) for c in shared).alias("more"),
        pl.concat_list(pl.max_horizontal(pl.col(c).abs().sum(), pl.lit(1e-12)) for c in shared).alias("scale"),
        pl.concat_list((pl.col(c) - pl.col(f"{c}_outcome")).abs().sum() for c in shared).alias("apart"),
    ).explode("c", "more", "scale", "apart", empty_as_null=False)
    sums = sums.with_columns(total=pl.col("more") / pl.col("scale"), per_period=pl.col("apart") / pl.col("scale"))
    bad = (pl.col("total").abs() > 0.02) | (pl.col("per_period") > 0.02)
    status = "fail" if sums.select(bad.any()).item() else "pass"
    notes = [
        f"{c}: consumer books {more:+.4%} of capital more than its outcome ({total:+.2%} of |{c}|); period by period "
        f"the gap is {per_period:.0%}"
        for c, more, total, per_period in sums.select("c", "more", "total", "per_period").iter_rows()
    ]
    if missing:
        notes.append(f"{missing} held rows without an outcome")
    return Check("consumer matches the outcome", status, "; ".join(notes))  # type: ignore[arg-type]


def passthrough(run: Run, sources: Sources, base: ConsumerResult, scores: pl.DataFrame) -> Check:
    """The consumer's own baseline scores must change nothing. Handing the consumer its own baseline scores must change
    nothing: the score path is wired correctly."""
    mismatch = _frame_mismatch(run(sources, scores).net, base.net, ["t"], ["net"])
    if mismatch:
        return Check("scores pass through", "fail", f"own scores change the result: {mismatch}")
    return Check("scores pass through", "pass", "the baseline's own scores reproduce the baseline")


def target_reads_score(
    name: str, reduce: Callable[[pl.DataFrame], pl.DataFrame], rows: pl.DataFrame, missing: float = 0.0
) -> Check:
    """A target must move with the scores. The same outcome rows with two draws of random scores give other values;
    with ``missing``, each draw leaves that share of the rows without a score (NaN), as events leave the rows between
    them."""
    rng = np.random.default_rng(0)

    def drawn() -> np.ndarray:
        scores = rng.standard_normal(rows.height)
        return np.where(rng.random(rows.height) < missing, np.nan, scores) if missing else scores

    a, b = (_by_t(reduce(rows.with_columns(score=drawn()))) for _ in range(2))
    if a.height == b.height and np.allclose(a["value"].to_numpy(), b["value"].to_numpy(), equal_nan=True):
        detail = (
            f"{name} gives the same number for any scores; name a target that reads them, "
            'such as Use("ref.mean", {"cluster": ..., "top": 0.2})'
        )
        return Check("target reads the score", "fail", detail)
    return Check("target reads the score", "pass", f"{name} moves with the scores")


def branch_diff(base: ConsumerResult, branch: ConsumerResult, start: int, end: int) -> tuple[float, float]:
    """The branch's extra net over ``(start, end]``, and its overlap. The overlap is how much of the baseline's
    positions it still holds there."""
    window = (pl.col("t") > lit_time(start)) & (pl.col("t") <= lit_time(end))
    diff = float(branch.net.filter(window)["net"].sum()) - float(base.net.filter(window)["net"].sum())
    a = base.positions.filter(window).select("t", "inst", pl.col("weight").abs().alias("a"))
    b = branch.positions.filter(window).select("t", "inst", pl.col("weight").abs().alias("b"))
    both = a.join(b, on=["t", "inst"], how="left").fill_null(0.0)
    total = float(a["a"].sum())
    overlap = float(both.select(pl.min_horizontal("a", "b")).to_series().sum()) / total if total else 1.0
    return diff, overlap


def perturb(scores: pl.DataFrame, kind: str, share: float, block_ns: int, seed: int) -> pl.DataFrame:
    """An uninformative change to scores: a share dropped or permuted. It drops ``share`` of the candidates, or permutes
    the scores of ``share`` of them among themselves at each time. The chosen candidates stay chosen for a whole block.
    """
    block = (pl.col("t").dt.epoch("ns") // block_ns).alias("block")
    u = pl.struct(pl.col("inst"), block).hash(seed) / float(2**64)
    if kind == "drop":
        return scores.filter(u >= share)
    chosen = scores.with_columns(u.alias("_u"), pl.struct(pl.col("inst"), block).hash(seed + 1).alias("_k"))
    picked = chosen.filter(pl.col("_u") < share)
    shuffled = picked.sort("t", "_k")["score"]
    moved = picked.sort("t", "inst").with_columns(score=shuffled)
    return pl.concat([chosen.filter(pl.col("_u") >= share), moved]).drop("_u", "_k").sort("t", "inst")


def total(result: ConsumerResult) -> float:
    return float(result.net["net"].sum())


def max_drawdown(result: ConsumerResult) -> float:
    equity = 1 + np.cumsum(result.net["net"].to_numpy())
    return float((equity / np.maximum.accumulate(equity) - 1).min()) if equity.size else 0.0


def _by_t(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    out = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    return out.select("t", pl.col("value").cast(pl.Float64)).sort("t")


def _mutations(case: OutcomeCase, h: pd.Timedelta) -> list[tuple[str, dict[str, pl.DataFrame], bool]]:
    tables = dict(case.sources)
    ts = case.candidates.select(pl.col("t").dt.epoch("ns"))["t"]
    lo, hi = int(ts.min()), int(ts.max()) + int(h.value)  # type: ignore[arg-type]
    at = pl.concat([t.select(pl.col("available_at").dt.epoch("ns")) for t in tables.values()])["available_at"]
    steps = np.diff(at.unique().sort().to_numpy())
    step = int(steps[steps > 0].min()) if (steps > 0).any() else int(h.value)
    inside = (pl.col("available_at") > lit_time(lo)) & (pl.col("available_at") <= lit_time(hi))
    after = pl.col("available_at") > lit_time(hi)
    later = pl.duration(nanoseconds=step)
    shifted = {
        n: t.with_columns(pl.col("event_time") + later, pl.col("available_at") + later) for n, t in tables.items()
    }
    removed = [
        (
            f"removing {name}.{c} inside the window",
            {**tables, name: t.with_columns(pl.when(inside).then(None).otherwise(pl.col(c)).alias(c))},
            True,
        )
        for name, t in tables.items()
        for c in values(t)
    ]
    return [
        ("shifting the data one step later", shifted, True),
        ("changing values inside the window", changed(tables, inside, lambda v: v * 1.01 + 1e-6), True),
        ("changing values after the window", changed(tables, after, lambda v: v * 1.5 + 1), False),
        *removed,
    ]


def _diff(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    joined = a.select("t", pl.col("net").alias("a")).join(b.select("t", pl.col("net").alias("b")), on="t", how="full")
    return joined.select(pl.coalesce("t", "t_right").alias("t"), (pl.col("a") - pl.col("b")).alias("d")).sort("t")


def _frame_mismatch(a: pl.DataFrame, b: pl.DataFrame, keys: list[str], values: list[str]) -> str:
    if a.height != b.height:
        return f"{a.height} rows vs {b.height}"
    if a.is_empty():
        return ""
    joined = a.select(*keys, *values).join(b.select(*keys, *values), on=keys, how="full", suffix="_b", coalesce=True)
    if joined.height != a.height:
        return "different keys"
    if not values:
        return ""
    differ = joined.select(_differ(v) for v in values).row(0)
    return next((f"{n} values of {v} differ" for v, n in zip(values, differ, strict=True) if n), "")


def _differ(v: str) -> pl.Expr:
    """How many values of ``v`` differ between the two sides: both NaN agree, otherwise within ``np.isclose``."""
    x, y = (pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in (v, f"{v}_b"))
    near = x.is_not_nan() & y.is_not_nan() & ((x - y).abs() <= 1e-12 + 1e-9 * y.abs())
    close = (x == y) | near | (x.is_nan() & y.is_nan())
    return (~close).sum().alias(v)


def _span_text(span: pd.Timedelta) -> str:
    hours = span / pd.Timedelta(hours=1)
    return f"{hours:g}h" if hours == int(hours) else str(span)


def _iso(t: int) -> str:
    return str(np.datetime64(t, "ns"))[:16] + "Z"
