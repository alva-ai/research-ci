"""Invariants a plugin declares and the engine checks. A universe whose candidates are one asset on two venues
declares the hedge (``research_ci.api.Hedge``): the field that prices the legs and how close they must stay. At every
decision time the legs must then trade at about the same level and, when both move enough to tell, move together over
the recent past, using only prices known by then."""

import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.check import Check


def hedged_legs(
    cands: pl.DataFrame,
    fields: pl.DataFrame,
    price: str,
    max_ratio: float,
    min_corr: float,
    window: int,
    min_obs: int,
    min_move: float,
    keep: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Per candidate row (``t``, ``entity``): the legs' price ratio and the correlation of their latest returns over
    the pair's last ``window`` decisions, and ``ok``. The correlation counts only when both legs' returns vary by at
    least ``min_move``; a pegged asset's returns are noise on both venues."""
    px = (
        fields.filter(pl.col("field") == price)
        .sort("entity", "available_at")
        .select(
            pl.col("entity").alias("leg"),
            pl.col("available_at").alias("t"),
            pl.col("value").alias("px"),
            (pl.col("value") / pl.col("value").shift(1).over("entity")).log().alias("r"),
        )
        .sort("t")
    )
    rows = cands.select("t", "entity", "long", "short", *keep).with_columns(pl.col("t").cast(TIME))
    q = rows.unpivot(index="t", on=["long", "short"], value_name="leg").select("t", "leg").unique().sort("t")
    known = q.join_asof(px, on="t", by="leg", strategy="backward", check_sortedness=False)
    rows = rows.join(known.rename({"leg": "long", "px": "px_long", "r": "r_long"}), on=["t", "long"], how="left")
    rows = rows.join(known.rename({"leg": "short", "px": "px_short", "r": "r_short"}), on=["t", "short"], how="left")
    a, b = pl.min_horizontal("long", "short"), pl.max_horizontal("long", "short")
    rows = rows.with_columns(pair=a + "|" + b).sort("t")
    corr = pl.rolling_corr(pl.col("r_long"), pl.col("r_short"), window_size=window, min_samples=min_obs)
    vol = [pl.col(f"r_{leg}").rolling_std(window, min_samples=min_obs).over("pair") for leg in ("long", "short")]
    rows = rows.with_columns(corr=corr.over("pair"), quiet=pl.min_horizontal(vol) < min_move)
    rows = rows.with_columns(ratio=pl.col("px_long") / pl.col("px_short"))
    level_ok = pl.col("ratio").is_between(1 / max_ratio, max_ratio)
    unknown = pl.col("corr").is_null() | pl.col("corr").is_nan() | pl.col("quiet").fill_null(True)
    move_ok = unknown | (pl.col("corr") >= min_corr)
    ok = (level_ok & move_ok).fill_null(False).alias("ok")
    return rows.select("t", "entity", "pair", *keep, "ratio", "corr", ok)


def hedge_check(legs: pl.DataFrame, price: str, max_ratio: float, min_corr: float, by: str = "pair") -> Check:
    bad = legs.filter(~pl.col("ok"))
    name = "hedged legs are one asset"
    rule = f"{price} within {max_ratio:g}x and returns correlated ≥ {min_corr:g}"
    if bad.is_empty():
        return Check(name, "pass", f"every candidate row holds ({rule})")
    worst = bad.group_by(by).len().sort("len", descending=True).head(6)[by].cast(pl.String).to_list()
    share = bad.height / max(legs.height, 1)
    return Check(name, "warn", f"{share:.2%} of candidate rows fail ({rule}); mostly {', '.join(worst)}")


def _bad_share() -> dict[str, float]:
    """e1 is e0 on another venue, a little apart; e2 is another asset at the same level."""
    f = source(Source(entities=3), 0).filter(pl.col("field") == "price")
    e0 = f.filter(pl.col("entity") == "e0")
    twin = e0.with_columns(entity=pl.lit("e1"), value=pl.col("value") * 1.002)
    f = pl.concat([e0, twin, f.filter(pl.col("entity") == "e2")])
    hours = pl.DataFrame({"t": pl.int_range(80, 300, eager=True) * HOUR_NS + START_NS})
    pairs = pl.DataFrame({"entity": ["e0>e1", "e0>e2"], "long": ["e0", "e0"], "short": ["e1", "e2"]})
    cands = hours.join(pairs, how="cross").with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"))
    legs = hedged_legs(cands, f, "price", 1.2, 0.3, 72, 48, 1e-3)
    bad = legs.group_by("entity").agg((~pl.col("ok")).mean().alias("bad"))
    return {str(e): float(v) for e, v in bad.iter_rows()}


CASES = [
    Case(
        "one asset on two venues holds; another asset at the same level is caught",
        lambda: {k: v > 0.5 for k, v in _bad_share().items()},
        {"e0>e1": False, "e0>e2": True},
    ),
]
