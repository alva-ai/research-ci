import polars as pl


class FlatCosts:
    """Commission plus slippage per fill, in basis points of the traded notional, for candidates of ``legs`` legs
    traded together (weights are per leg)."""

    def __init__(self, commission_bps: float, slippage_bps: float, legs: int = 1) -> None:
        self.per_fill = (commission_bps + slippage_bps) / 1e4
        self.legs = legs

    def round_trip(self, candidates: pl.DataFrame) -> pl.Series:
        return pl.Series("round_trip", [2 * self.legs * self.per_fill] * candidates.height, dtype=pl.Float64)

    def fees(self, trades: pl.DataFrame) -> float:
        traded = trades.select((pl.col("to_weight") - pl.col("from_weight")).abs().sum()).item()
        return float(traded or 0.0) * self.legs * self.per_fill
