"""Protocol tests the engine runs on plugins before using their output. Each returns a ``Check``. Plugins are only
called through the callables passed in, so the engine decides what data they see; nothing here knows what a field
means."""

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import span
from research_ci.api.plugins import ConsumerResult, CostModel, LabelCase
from research_ci.kernel.time.clock import lit_time
from research_ci.kernel.verdict.check import Check, Status

Candidates = Callable[[pl.DataFrame, list[int]], pl.DataFrame]
Values = Callable[[pl.DataFrame, pl.DataFrame, pd.Timedelta], pl.DataFrame]
Run = Callable[[pl.DataFrame, pl.DataFrame | None], ConsumerResult]
COMPONENTS = ("flow", "price", "fees")


def truncated(fields: pl.DataFrame, t: int) -> pl.DataFrame:
    return fields.filter(pl.col("available_at") <= lit_time(t))


def universe_pit(
    candidates: Candidates, fields: pl.DataFrame, full: pl.DataFrame, cuts: Sequence[int], margin_ns: int
) -> Check:
    """With the data cut at ``cut``, the candidates at ``cut - margin`` must equal those from all data. The margin
    lets a plugin that needs the next bar to finish one still pass; ``universe_causal`` covers shorter look-aheads."""
    values = [c for c, d in full.schema.items() if c not in ("t", "entity") and d.is_numeric()]
    bad = []
    for cut in cuts:
        t = cut - margin_ns
        part = candidates(truncated(fields, cut), [t]).filter(pl.col("t") == lit_time(t))
        mismatch = _frame_mismatch(part, full.filter(pl.col("t") == lit_time(t)), ["t", "entity"], values)
        if mismatch:
            bad.append(f"{_iso(t)}: {mismatch}")
    if bad:
        return Check("universe point-in-time", "fail", "; ".join(bad))
    return Check("universe point-in-time", "pass", f"{len(cuts)} cut points give the same candidates")


def universe_causal(candidates: Candidates, fields: pl.DataFrame, full: pl.DataFrame, t: int) -> Check:
    """Change every value that becomes known after ``t``: the candidates at ``t`` must not move."""
    later = pl.col("available_at") > lit_time(t)
    moved = fields.with_columns(pl.when(later).then(pl.col("value") * 1.01 + 1e-6).otherwise(pl.col("value")))
    values = [c for c, d in full.schema.items() if c not in ("t", "entity") and d.is_numeric()]
    part = candidates(moved, [t]).filter(pl.col("t") == lit_time(t))
    mismatch = _frame_mismatch(part, full.filter(pl.col("t") == lit_time(t)), ["t", "entity"], values)
    if mismatch:
        return Check("universe causal", "fail", f"values after {_iso(t)} changed the candidates at it: {mismatch}")
    return Check("universe causal", "pass", f"values after {_iso(t)} leave the candidates at it unchanged")


def label_cases(name: str, values: Values, cases: Sequence[LabelCase]) -> list[Check]:
    """The author's worked examples must come out as written, and must move when their inputs move: shifted in time,
    perturbed inside the window, or missing a field. Perturbing data after the window must change nothing."""
    if not cases:
        return [Check(f"label {name}: worked examples", "fail", "the label gives no worked examples")]
    wrong, blind = [], []
    for i, case in enumerate(cases):
        h = span(case.horizon)
        got = values(case.fields, case.candidates, h)
        columns = [c for c in case.expected.columns if c not in ("t", "entity")]
        mismatch = _frame_mismatch(got, case.expected, ["t", "entity"], columns)
        if mismatch:
            wrong.append(f"case {i}: {mismatch}")
            continue
        for mutation, fields, should_move in _mutations(case, h):
            moved = bool(_frame_mismatch(values(fields, case.candidates, h), case.expected, ["t", "entity"], columns))
            if moved != should_move:
                blind.append(f"case {i}: {mutation} {'did not change' if should_move else 'changed'} the label")
    checks = [
        Check(f"label {name}: worked examples", "fail" if wrong else "pass", "; ".join(wrong) or f"{len(cases)} ok")
    ]
    detail = "; ".join(blind) or "shift, in-window changes and missing fields move it; later data does not"
    checks.append(Check(f"label {name}: examples bite", "fail" if blind else "pass", detail))
    return checks


def label_truncation(
    name: str, values: Values, fields: pl.DataFrame, cands: pl.DataFrame, horizon: pd.Timedelta, cuts: Sequence[int]
) -> Check:
    """Labels whose window ends by ``cut`` must be the same when the data stops at ``cut``."""
    h = int(horizon.value)
    bad = []
    for cut in cuts:
        rows = cands.filter(pl.col("t").dt.epoch("ns") + h <= cut)
        if rows.is_empty():
            continue
        a, b = values(truncated(fields, cut), rows, horizon), values(fields, rows, horizon)
        mismatch = _frame_mismatch(a, b, ["t", "entity"], ["value"])
        if mismatch:
            bad.append(f"cut {_iso(cut)}: {mismatch}")
    check = f"label {name}: no data after t + h"
    if bad:
        return Check(check, "fail", "; ".join(bad))
    return Check(check, "pass", f"{len(cuts)} cut points, horizon {_span_text(horizon)}")


def consumer_pit(run: Run, fields: pl.DataFrame, full: ConsumerResult, cuts: Sequence[int], margin_ns: int) -> Check:
    """A run on data cut at ``cut`` must match the full run for every period ending before ``cut - margin``."""
    bad = []
    for cut in cuts:
        part = run(truncated(fields, cut), None)
        until = lit_time(cut - margin_ns)
        mismatch = _frame_mismatch(
            part.net.filter(pl.col("t") <= until), full.net.filter(pl.col("t") <= until), ["t"], ["net"]
        )
        trades = _frame_mismatch(
            part.trades.filter(pl.col("t") <= until),
            full.trades.filter(pl.col("t") <= until),
            ["t", "entity"],
            ["to_weight"],
        )
        if mismatch or trades:
            bad.append(f"cut {_iso(cut)}: net {mismatch or 'same'}, trades {trades or 'same'}")
    if bad:
        return Check("consumer point-in-time", "fail", "; ".join(bad))
    return Check("consumer point-in-time", "pass", f"{len(cuts)} cut points match the full run before each cut")


def causal(run: Run, fields: pl.DataFrame, base: ConsumerResult, at: int, period_ns: int) -> Check:
    """Change every value that becomes known in ``(at, at + period]``: nothing up to ``at`` may move, and something
    after it must."""
    window = (pl.col("available_at") > lit_time(at)) & (pl.col("available_at") <= lit_time(at + period_ns))
    moved = fields.with_columns(pl.when(window).then(pl.col("value") * 1.01 + 1e-6).otherwise(pl.col("value")))
    diff = _diff(run(moved, None).net, base.net).filter(pl.col("d").abs() > 1e-12)
    early = diff.filter(pl.col("t") <= lit_time(at))
    if not early.is_empty():
        return Check("consumer causal", "fail", f"data after {_iso(at)} changed the period ending {early['t'][0]}")
    if diff.is_empty():
        return Check("consumer causal", "fail", f"changing the data after {_iso(at)} changed nothing")
    return Check("consumer causal", "pass", f"data after {_iso(at)} first moves the period ending {diff['t'][0]}")


def deterministic(a: ConsumerResult, b: ConsumerResult) -> Check:
    mismatch = _frame_mismatch(a.net, b.net, ["t"], ["net"])
    if mismatch:
        return Check("consumer deterministic", "fail", f"two identical runs differ: {mismatch}")
    return Check("consumer deterministic", "pass", "two identical runs give the same net")


def adds_up(result: ConsumerResult) -> list[Check]:
    """Components add up to net, and the consumer's own checks pass."""
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
    """What the consumer charged must be what the cost model says its trades cost."""
    if "fees" not in result.net.columns:
        return Check("fees match the cost model", "warn", "the consumer reports no fees")
    charged = -float(result.net["fees"].sum())
    model = costs.fees(result.trades)
    ratio = charged / model if model else (1.0 if charged == 0 else math.inf)
    status: Status = "pass" if abs(ratio - 1) < 0.01 else "fail"
    return Check("fees match the cost model", status, f"charged {charged:.4%}, cost model {model:.4%}")


def matches_labels(result: ConsumerResult, labels: pl.DataFrame, period_ns: int) -> Check:
    """The consumer and the label are written independently; on the positions held, they must agree on what each
    component booked every period earned, period by period. A gap in the total is an error in size or sign; a gap
    that only shows period by period is money booked in another period than the label's, such as a flow at a decision
    time paid to the positions that decision opens. Either fails: verdicts are judged period by period."""
    shared = [c for c in result.marked if c in result.net.columns and c in labels.columns]
    if not shared:
        return Check("consumer matches labels", "skip", "no component booked every period by both")
    held = result.positions.join(labels, on=["t", "entity"], how="left")
    missing = held.filter(pl.col(shared[0]).is_null()).height
    expected = held.group_by("t").agg(*(pl.col(c).fill_null(0.0).mul(pl.col("weight")).sum() for c in shared))
    expected = expected.with_columns(pl.col("t") + pl.duration(nanoseconds=period_ns))
    joined = result.net.join(expected, on="t", how="left", suffix="_label").fill_null(0.0)
    notes, status = [], "pass"
    for c in shared:
        a, b = joined[c].to_numpy(), joined[f"{c}_label"].to_numpy()
        scale = max(float(np.abs(a).sum()), 1e-12)
        total = (float(a.sum()) - float(b.sum())) / scale
        per_period = float(np.abs(a - b).sum()) / scale
        if abs(total) > 0.02 or per_period > 0.02:
            status = "fail"
        notes.append(
            f"{c}: consumer books {float(a.sum()) - float(b.sum()):+.4%} of capital more than its labels "
            f"({total:+.2%} of |{c}|); period by period the gap is {per_period:.0%}"
        )
    if missing:
        notes.append(f"{missing} held rows without a label")
    return Check("consumer matches labels", status, "; ".join(notes))  # type: ignore[arg-type]


def passthrough(run: Run, fields: pl.DataFrame, base: ConsumerResult, scores: pl.DataFrame) -> Check:
    """Handing the consumer its own baseline scores must change nothing: the score path is wired correctly."""
    mismatch = _frame_mismatch(run(fields, scores).net, base.net, ["t"], ["net"])
    if mismatch:
        return Check("scores pass through", "fail", f"own scores change the result: {mismatch}")
    return Check("scores pass through", "pass", "the baseline's own scores reproduce the baseline")


def branch_diff(base: ConsumerResult, branch: ConsumerResult, start: int, end: int) -> tuple[float, float]:
    """The branch's extra net over ``(start, end]``, and how much of the baseline's positions it still holds there."""
    window = (pl.col("t") > lit_time(start)) & (pl.col("t") <= lit_time(end))
    diff = float(branch.net.filter(window)["net"].sum()) - float(base.net.filter(window)["net"].sum())
    a = base.positions.filter(window).select("t", "entity", pl.col("weight").abs().alias("a"))
    b = branch.positions.filter(window).select("t", "entity", pl.col("weight").abs().alias("b"))
    both = a.join(b, on=["t", "entity"], how="left").fill_null(0.0)
    total = float(a["a"].sum())
    overlap = float(both.select(pl.min_horizontal("a", "b")).to_series().sum()) / total if total else 1.0
    return diff, overlap


def perturb(scores: pl.DataFrame, kind: str, share: float, block_ns: int, seed: int) -> pl.DataFrame:
    """An uninformative change to scores: drop ``share`` of the candidates, or permute the scores of ``share`` of
    them among themselves at each time. The chosen candidates stay chosen for a whole block."""
    block = (pl.col("t").dt.epoch("ns") // block_ns).alias("block")
    u = pl.struct(pl.col("entity"), block).hash(seed) / float(2**64)
    if kind == "drop":
        return scores.filter(u >= share)
    chosen = scores.with_columns(u.alias("_u"), pl.struct(pl.col("entity"), block).hash(seed + 1).alias("_k"))
    picked = chosen.filter(pl.col("_u") < share)
    shuffled = picked.sort("t", "_k")["score"]
    moved = picked.sort("t", "entity").with_columns(score=shuffled)
    return pl.concat([chosen.filter(pl.col("_u") >= share), moved]).drop("_u", "_k").sort("t", "entity")


def total(result: ConsumerResult) -> float:
    return float(result.net["net"].sum())


def max_drawdown(result: ConsumerResult) -> float:
    equity = 1 + np.cumsum(result.net["net"].to_numpy())
    return float((equity / np.maximum.accumulate(equity) - 1).min()) if equity.size else 0.0


def _mutations(case: LabelCase, h: pd.Timedelta) -> list[tuple[str, pl.DataFrame, bool]]:
    f = case.fields
    ts = case.candidates.select(pl.col("t").dt.epoch("ns"))["t"]
    lo, hi = int(ts.min()), int(ts.max()) + int(h.value)  # type: ignore[arg-type]
    known = f.select(pl.col("available_at").dt.epoch("ns").unique().sort())["available_at"].to_numpy()
    steps = np.diff(known)
    step = int(steps[steps > 0].min()) if (steps > 0).any() else int(h.value)
    inside = (pl.col("available_at") > lit_time(lo)) & (pl.col("available_at") <= lit_time(hi))
    after = pl.col("available_at") > lit_time(hi)
    shift = pl.duration(nanoseconds=step)
    shifted = f.with_columns(pl.col("event_time") + shift, pl.col("available_at") + shift)
    nudged = f.with_columns(pl.when(inside).then(pl.col("value") * 1.01 + 1e-6).otherwise("value"))
    later = f.with_columns(pl.when(after).then(pl.col("value") * 1.5 + 1).otherwise("value"))
    out = [
        ("shifting the data one step later", shifted, True),
        ("changing values inside the window", nudged, True),
        ("changing values after the window", later, False),
    ]
    for name in f["field"].unique().sort().to_list():
        out.append((f"removing {name} inside the window", f.filter(~(inside & (pl.col("field") == name))), True))
    return out


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
    for v in values:
        x, y = joined[v].to_numpy().astype("float64"), joined[f"{v}_b"].to_numpy().astype("float64")
        same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12)
        if not same.all():
            return f"{int((~same).sum())} values of {v} differ"
    return ""


def _span_text(span: pd.Timedelta) -> str:
    hours = span / pd.Timedelta(hours=1)
    return f"{hours:g}h" if hours == int(hours) else str(span)


def _iso(t: int) -> str:
    return str(np.datetime64(t, "ns"))[:16] + "Z"
