"""Influence concentration: how much of the noise the decisive number's MDE rests on comes from one cluster, such as one
symbol. A cluster whose data is broken often carries most of it; so does a cluster where the effect really is. The
method only measures; what to do about it is for a person."""

import math
from collections.abc import Sequence

import polars as pl

from research_ci.api.methods import Case, Market
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.power import block_se

SHARE_SCHEMA = {"cluster": pl.String, "share": pl.Float64}


def shares(
    rows: pl.DataFrame, cluster: str, h_ns: int | str, label: str = "value", by: Sequence[str] = ()
) -> pl.DataFrame:
    """Each cluster's share of the noise, between 0 and 1. The share is the part of the variance of the block means (one
    horizon long, as the MDE takes them) of the label's mean per decision time ``t``, each cluster weighted once, that
    goes away when the cluster is left out. ``h_ns`` is the horizon, or the column that holds it; with ``by`` (such as
    the key of each horizon in stacked rows) each group is measured on its own. Largest share first.

    Every cluster is left out at once, by algebra on the means per decision time and cluster: the mean without a
    cluster is the sum of the cluster means less its own, over their count less one. A cluster that is alone at a
    decision time leaves nothing there, and one alone everywhere carries everything. The number measured is the
    default target's shape (``ref.mean``: each cluster weighted once); a target that picks some candidates sees part
    of these rows, and the shares are those of the label's noise."""
    keys = list(by) or ["_key"]
    schema = {**{k: rows.schema[k] for k in by}, **SHARE_SCHEMA}
    if cluster not in rows.columns:
        return pl.DataFrame(schema=schema)
    h = pl.lit(h_ns, dtype=pl.Int64) if isinstance(h_ns, int) else pl.col(h_ns).cast(pl.Int64)
    base = rows.lazy().with_columns(_key=pl.lit(0)) if not by else rows.lazy()
    base = base.filter(pl.col(label).is_not_null()).select(
        *keys,
        pl.col("t").dt.epoch("ns").alias("_t"),
        pl.col(cluster).alias("cluster"),
        label,
        h.alias("_h"),
    )
    per_c = base.group_by(*keys, "_t", "cluster").agg(pl.col(label).mean().alias("_m"), pl.col("_h").first())
    per_t = per_c.group_by(*keys, "_t").agg(pl.col("_m").sum().alias("_M"), pl.len().alias("_K"), pl.col("_h").first())
    block = (pl.col("_t") - pl.col("_t").min().over(keys)) // pl.col("_h")
    per_t = per_t.with_columns(_full=pl.col("_M") / pl.col("_K"), _b=block)
    blocks = per_t.group_by(*keys, "_b").agg(pl.col("_full").sum().alias("_S"), pl.len().alias("_n"))
    blocks = blocks.with_columns(_F=pl.col("_S") / pl.col("_n")).with_columns(
        _f=pl.col("_F") - pl.col("_F").mean().over(keys)
    )
    totals = blocks.group_by(*keys).agg(
        pl.len().alias("_B"),
        pl.col("_f").sum().alias("_T1"),
        (pl.col("_f") ** 2).sum().alias("_T2"),
        pl.col("_F").mean().alias("_c"),
        pl.col("_F").is_finite().all().alias("_finite"),
    )
    totals = totals.with_columns(_se2=_se2(pl.col("_T1"), pl.col("_T2"), pl.col("_B")))
    alone = pl.col("_K") == 1
    loo = (pl.col("_M") - pl.col("_m")) / (pl.col("_K") - 1) - pl.col("_full")
    own = per_c.join(per_t.select(*keys, "_t", "_M", "_K", "_full", "_b"), on=[*keys, "_t"])
    own = own.group_by(*keys, "cluster", "_b").agg(
        pl.col("_full").filter(alone).sum().alias("_A"),
        alone.sum().alias("_a"),
        loo.filter(~alone).sum().alias("_D"),
    )
    own = own.join(blocks.select(*keys, "_b", "_S", "_n", "_f"), on=[*keys, "_b"])
    own = own.join(totals.select(*keys, "_c"), on=keys)
    gone = pl.col("_n") == pl.col("_a")
    g = (pl.col("_S") - pl.col("_A") + pl.col("_D")) / (pl.col("_n") - pl.col("_a")) - pl.col("_c")
    per_cluster = (
        own.with_columns(_g=g)
        .group_by(*keys, "cluster")
        .agg(
            gone.sum().alias("_gone"),
            (pl.col("_g") - pl.col("_f")).filter(~gone).sum().alias("_d1"),
            pl.col("_f").filter(gone).sum().alias("_l1"),
            (pl.col("_g") ** 2 - pl.col("_f") ** 2).filter(~gone).sum().alias("_d2"),
            (pl.col("_f") ** 2).filter(gone).sum().alias("_l2"),
        )
    )
    joined = per_cluster.join(totals, on=keys)
    count = pl.col("_B") - pl.col("_gone")
    without = _se2(pl.col("_T1") + pl.col("_d1") - pl.col("_l1"), pl.col("_T2") + pl.col("_d2") - pl.col("_l2"), count)
    share = pl.when(without.is_nan()).then(1.0).otherwise((1 - without / pl.col("_se2")).clip(0.0, 1.0))
    measured = pl.col("_finite") & pl.col("_se2").is_finite() & (pl.col("_se2") > 0)
    out = joined.filter(measured).select(*by, pl.col("cluster").cast(pl.String), share.alias("share"))
    order = [*by, "share", "cluster"]
    return out.sort(order, descending=[False] * len(by) + [True, False]).collect().cast(schema)  # type: ignore[arg-type]


def _se2(sum1: pl.Expr, sum2: pl.Expr, n: pl.Expr) -> pl.Expr:
    """The squared standard error of a mean of ``n`` block means from their sum and sum of squares; NaN below two."""
    var = (sum2 - sum1**2 / n) / (n - 1)
    return pl.when(n >= 2).then(var.clip(lower_bound=0.0) / n).otherwise(math.nan)


def _mean(table: pl.DataFrame) -> pl.DataFrame:
    """The mean per decision time, each cluster weighted once."""
    per = table.group_by("t", "cluster").agg(pl.col("y").mean())
    return per.group_by("t").agg(pl.col("y").mean().alias("value"))


def _scaled(k: float, seed: int, m: Market | None = None) -> pl.DataFrame:
    """A synthetic market whose cluster c0 has labels ``k`` times as large."""
    table = market(m or Market(), seed)
    return table.with_columns(y=pl.when(pl.col("cluster") == "c0").then(pl.col("y") * k).otherwise("y"))


def _top(k: float, seed: int) -> tuple[str, float]:
    found = shares(_scaled(k, seed), "cluster", HOUR_NS, label="y")
    return str(found["cluster"][0]), float(found["share"][0])


def _blamed(seed: int) -> list[str]:
    """The clusters above half the noise on a market of three clusters of three, when c0 has labels thirty times as
    large."""
    table = _scaled(30.0, seed, Market(entities=9, clusters=3, groups=1))
    return shares(table, "cluster", HOUR_NS, label="y").filter(pl.col("share") > 0.5)["cluster"].to_list()


def _alone() -> list[tuple[str, float]]:
    table = market(Market(clusters=1, groups=1), 0)
    return shares(table, "cluster", HOUR_NS, label="y").rows()


CLUSTERS = ("c0", "c1", "c2", "c3")


def _direct(seed: int, h_ns: int) -> float:
    """The largest gap between the algebra and leaving each cluster out and computing again, on a market where some
    clusters are missing at some times."""
    table = market(Market(entities=12, clusters=4, groups=1), seed).filter(
        ~((pl.col("cluster") == "c1") & (pl.col("t").dt.hour() < 6))
    )
    se2 = block_se(_mean(table), h_ns) ** 2
    found = shares(table, "cluster", h_ns, label="y")
    direct = {c: max(0.0, 1 - block_se(_mean(table.filter(pl.col("cluster") != c)), h_ns) ** 2 / se2) for c in CLUSTERS}
    return max(abs(share - direct[c]) for c, share in found.rows())


def _stacked() -> float:
    """The largest gap between two horizons stacked under a key and each alone."""
    a, b = _scaled(5.0, 0), _scaled(1.0, 1)
    rows = pl.concat([a.with_columns(key=pl.lit("1h"), h=HOUR_NS), b.with_columns(key=pl.lit("4h"), h=4 * HOUR_NS)])
    both = shares(rows, "cluster", "h", label="y", by=["key"])
    alone = pl.concat(
        [
            shares(a, "cluster", HOUR_NS, label="y").with_columns(key=pl.lit("1h")),
            shares(b, "cluster", 4 * HOUR_NS, label="y").with_columns(key=pl.lit("4h")),
        ]
    )
    joined = both.join(alone, on=["key", "cluster"], how="full")
    return float((joined["share"] - joined["share_right"]).abs().max())  # type: ignore[arg-type]


CASES = [
    Case(
        "no cluster carries half the noise on markets without a dominant one",
        lambda: sum(_top(1.0, s)[1] >= 0.5 for s in range(10)),
        0,
    ),
    Case(
        "a cluster with labels thirty times as large carries nearly all of it, on every market",
        lambda: sum(c == "c0" and share > 0.9 for c, share in (_top(30.0, s) for s in range(10))),
        10,
    ),
    Case("every cluster is measured", lambda: shares(market(Market(), 0), "cluster", HOUR_NS, label="y").height, 20),
    Case(
        "on three clusters of three only the noisy one is blamed, on every market",
        lambda: sum(_blamed(s) == ["c0"] for s in range(10)),
        10,
    ),
    Case("a lone cluster carries everything", _alone, [("c0", 1.0)]),
    Case(
        "the algebra gives what leaving each cluster out and computing again gives, with clusters missing at times",
        lambda: max(_direct(s, h) for s in range(3) for h in (HOUR_NS, 6 * HOUR_NS)),
        0.0,
        1e-9,
    ),
    Case("horizons stacked under a key give what each gives alone", _stacked, 0.0, 1e-12),
]
