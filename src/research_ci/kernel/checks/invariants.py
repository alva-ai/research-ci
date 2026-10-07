"""Invariants a plugin declares and the engine checks. A universe whose candidates are one asset on two venues says
which field prices the legs; at every decision time the legs must then trade at about the same level and, when both
move enough to tell, move together over the recent past, using only prices known by then."""

import polars as pl

from research_ci.kernel.judge.protocol import Check
from research_ci.kernel.time.clock import TIME


def hedged_legs(
    cands: pl.DataFrame,
    fields: pl.DataFrame,
    price: str,
    max_ratio: float = 1.2,
    min_corr: float = 0.3,
    window: int = 72,
    min_obs: int = 48,
    keep: tuple[str, ...] = (),
    min_move: float = 1e-3,
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
    for leg in ("long", "short"):
        q = rows.select("t", pl.col(leg).alias("leg")).unique().sort("t")
        known = q.join_asof(px, on="t", by="leg", strategy="backward", check_sortedness=False)
        rows = rows.join(known.rename({"leg": leg, "px": f"px_{leg}", "r": f"r_{leg}"}), on=["t", leg], how="left")
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


def hedge_check(
    legs: pl.DataFrame, price: str, max_ratio: float = 1.2, min_corr: float = 0.3, by: str = "pair"
) -> Check:
    bad = legs.filter(~pl.col("ok"))
    name = "hedged legs are one asset"
    rule = f"{price} within {max_ratio:g}x and returns correlated ≥ {min_corr:g}"
    if bad.is_empty():
        return Check(name, "pass", f"every candidate row holds ({rule})")
    worst = bad.group_by(by).len().sort("len", descending=True).head(6)[by].cast(pl.String).to_list()
    share = bad.height / max(legs.height, 1)
    return Check(name, "warn", f"{share:.2%} of candidate rows fail ({rule}); mostly {', '.join(worst)}")
