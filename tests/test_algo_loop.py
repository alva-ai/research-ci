from __future__ import annotations

import json

import polars as pl

from research_ci import AlgoSet
from research_ci.cli import main
from research_ci.eval.baseline import algo_main_output_shape
from research_ci.workflow import new_workflow, read_workflow, write_workflow


def test_algo_trade_contract_accepts_trade_keys_and_preserves_sig_values() -> None:
    signals = pl.DataFrame(
        {
            "event_time": [1, 2],
            "inst": ["left", "left"],
            "price": [10.0, 10.5],
            "signal": [0.25, -0.4],
        }
    )
    trades = pl.DataFrame(
        {
            "event_time": [1, 1, 2],
            "trade_id": ["one", "two", "three"],
            "inst": ["left", "left", "left"],
            "price": [10.0, 10.0, 10.5],
            "signal": [0.25, 0.25, -0.4],
            "signed_quantity": [1.0, -0.25, -0.5],
        }
    )
    declaration = AlgoSet(
        keys=("event_time", "trade_id"),
        context=("inst",),
        prices=("price",),
        signals=("signal",),
        positions=("signed_quantity",),
        strategies={"orders": ("signed_quantity",)},
        return_columns={"orders": "net_return"},
        equity_columns={"orders": "equity"},
        representation="trades",
        max_abs_position=2.0,
        max_gross=None,
    )

    assert declaration.backtest_keys == ("event_time",)
    assert algo_main_output_shape(signals, trades, declaration).status == "passed"

    changed_price = trades.with_columns(
        pl.when(pl.col("trade_id") == "two")
        .then(pl.lit(11.0))
        .otherwise(pl.col("price"))
        .alias("price")
    )
    result = algo_main_output_shape(signals, changed_price, declaration)
    assert result.status == "failed"
    assert not result.details["source_columns_preserved"]


def test_algo_cli_runs_external_backtester_and_writes_review_card(tmp_path) -> None:
    out = tmp_path / "run"
    (out / "sig").mkdir(parents=True)
    signal_table = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4, 5, 6],
            "left_inst": ["left"] * 6,
            "left_price": [10.0, 10.2, 10.1, 10.4, 10.5, 10.7],
            "right_inst": ["right"] * 6,
            "right_price": [5.0, 5.1, 5.2, 5.1, 5.0, 5.2],
            "signal": [0.0, 0.2, -0.4, 0.5, -0.3, 0.1],
        }
    )
    signal_table.write_parquet(out / "sig" / "main.parquet")
    state = new_workflow(("data", "feat", "sig", "algo"), "contract-card")
    state["stages"].update(
        {
            "contract": "passed",
            "data": "passed",
            "feat": "passed",
            "sig": "passed",
            "algo": "ready",
        }
    )
    write_workflow(out, state)

    builder = tmp_path / "algo_builder.py"
    builder.write_text(
        """
import polars as pl
from research_ci.kit import AlgoBundle

def build_algo(table):
    return AlgoBundle(
        main=table.with_columns(
            (pl.col("signal") / 2.0).alias("left_weight"),
            (-pl.col("signal") / 2.0).alias("right_weight"),
        ),
        metadata={"execution_lag_bars": 1},
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
    backtester = tmp_path / "backtest.py"
    backtester.write_text(
        """
import polars as pl
from research_ci.kit import BacktestResult

def run_backtest(table):
    path = table.select(
        "event_time",
        (pl.col("left_price").pct_change().fill_null(0.0)).alias("asset_return"),
        pl.col("left_weight").shift(1).fill_null(0.0).alias("held_weight"),
        (
            pl.col("left_weight").diff().abs().fill_null(pl.col("left_weight").abs())
            + pl.col("right_weight")
            .diff()
            .abs()
            .fill_null(pl.col("right_weight").abs())
        ).alias("pair_turnover"),
    ).with_columns(
        (pl.col("held_weight") * pl.col("asset_return")).alias("pair_net_return")
    ).with_columns(
        (pl.lit(1.0) + pl.col("pair_net_return")).cum_prod().alias("pair_equity")
    )
    summary = pl.DataFrame({
        "strategy": ["pair"],
        "total_return": [float(path.get_column("pair_equity")[-1] - 1.0)],
    })
    return BacktestResult(
        timeseries=path,
        summary=summary,
        metadata={"engine": "test_external"},
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "algo_plan.py"
    plan.write_text(
        """
import polars as pl
from research_ci import (
    AlgoSet,
    CaseEvalTest,
    EvalPlan,
    EvalSpec,
    LibrarySelection,
    SelectedMethod,
    expand_library,
)

ALGO_SET = AlgoSet(
    keys=("event_time",),
    context=("left_inst", "right_inst"),
    prices=("left_price", "right_price"),
    signals=("signal",),
    positions=("left_weight", "right_weight"),
    strategies={"pair": ("left_weight", "right_weight")},
    return_columns={"pair": "pair_net_return"},
    equity_columns={"pair": "pair_equity"},
    turnover_columns={"pair": "pair_turnover"},
)

LIBRARY_SELECTION = LibrarySelection((
    SelectedMethod(
        "library.quality.required_columns",
        reason="validate the declared Algo table",
        failure_policy="stop",
    ),
    SelectedMethod(
        "library.quality.finite_columns",
        reason="positions must be finite",
        on="positions",
        failure_policy="stop",
    ),
    SelectedMethod(
        "library.economic.backtest_summary",
        reason="summarize external net-return paths",
        parameters={"annualization_periods": 252},
    ),
))

def _break_neutrality(table):
    return table.with_columns((pl.col("left_weight") + 0.1).alias("left_weight"))

def make_plan(table, stage="algo", context=None):
    del table, context
    case = EvalSpec(
        "algo.case.dollar_neutral",
        expr=(pl.col("left_weight") + pl.col("right_weight")).abs() < 1e-12,
        failure_policy="stop",
    )
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=expand_library(ALGO_SET, LIBRARY_SELECTION),
        case=(case,),
        case_eval=(CaseEvalTest(
            name="algo.case_eval.dollar_neutral_has_counterexample",
            case=case.name,
            valid=lambda value: value,
            invalid=_break_neutrality,
        ),),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "algo",
            "run",
            "--builder",
            str(builder),
            "--plan",
            str(plan),
            "--backtester",
            str(backtester),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    main_table = pl.read_parquet(out / "algo" / "main.parquet")
    backtest = pl.read_parquet(out / "algo" / "backtest" / "timeseries.parquet")
    summary = pl.read_parquet(out / "algo" / "backtest" / "summary.parquet")
    assert main_table.height == backtest.height == signal_table.height
    assert summary.get_column("strategy").to_list() == ["pair"]
    assert (out / "algo" / "card.md").exists()
    manifest = json.loads((out / "algo" / "manifest.json").read_text())
    assert manifest["representation"] == "weights"
    assert manifest["strategies"] == ["pair"]
    assert manifest["backtester"] == str(backtester)
    assert read_workflow(out)["stages"]["algo"] == "review"
