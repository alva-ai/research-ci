"""Coverage: whether the data reaches far enough for the contract. An entity whose whole history is shorter than the
longest horizon can never give a label there, and a decision time with only a few candidates gives a number that
rests on them."""

import polars as pl

from research_ci.api.evals import Case, Source
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME

SPAN_SCHEMA = {"entity": pl.String, "start": TIME, "end": TIME}


def short_lived(fields: pl.DataFrame, horizon_ns: int) -> pl.DataFrame:
    """Entities whose events span less than ``horizon_ns``. Events of every field count; the columns are ``entity``,
    ``start`` and ``end``, the first and last event."""
    spans = fields.group_by("entity").agg(
        pl.col("event_time").min().alias("start"), pl.col("event_time").max().alias("end")
    )
    short = (pl.col("end") - pl.col("start")).dt.total_nanoseconds() < horizon_ns
    return spans.filter(short).sort("entity").cast(SPAN_SCHEMA)  # type: ignore[arg-type]


def thin(candidates: pl.DataFrame, floor: int) -> pl.DataFrame:
    """Decision times with some candidates but fewer than ``floor``. Columns: ``t`` and ``n``."""
    per_t = candidates.group_by("t").agg(pl.len().cast(pl.Int64).alias("n"))
    return per_t.filter(pl.col("n") < floor).sort("t")


def _trimmed(hours: int) -> pl.DataFrame:
    """A synthetic source in which e1 has only its first ``hours`` hours."""
    f = source(Source(entities=3), 0)
    late = (pl.col("entity") == "e1") & (pl.col("event_time").dt.epoch("ns") > START_NS + hours * HOUR_NS)
    return f.filter(~late)


def _thin() -> list[int]:
    rows = [(h, e) for h in range(4) for e in range(5 - h)]
    frame = pl.DataFrame(rows, schema=["t", "entity"], orient="row")
    return thin(frame, 3)["t"].to_list()


CASES = [
    Case(
        "every entity of a full source covers a day", lambda: short_lived(source(Source(), 0), 24 * HOUR_NS).height, 0
    ),
    Case(
        "an entity with 20 hours of data is shorter than a day; one with 25 is not",
        lambda: [short_lived(_trimmed(h), 24 * HOUR_NS)["entity"].to_list() for h in (20, 25)],
        [["e1"], []],
    ),
    Case("times with fewer candidates than the floor are thin", _thin, [3]),
]
