"""
Regression tests for the capstone baseline strategies
(scripts/capstone/baseline_strategies.py -> research/capstone/baselines/).

Guards:
  1. all four output artifacts exist and cover both 4H module windows,
  2. every expected baseline strategy row is present per window,
  3. the T-bill row behaves like cash (small positive return, ~zero vol),
  4. each module_*_deployed row's total return matches the locked
     family_compare_clean deployed number (same trades, one metrics path),
  5. equity curves reconcile with the metrics table,
  6. the random baseline used the documented deterministic seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "research" / "capstone" / "baselines"

WINDOWS = ["momentum_expansion", "multi_ticker_swing_htf"]
CLEAN_SUMMARIES = {
    "momentum_expansion": REPO / "strategies/momentum_expansion/backtest/results/family_compare_clean/comparison_summary_clean.json",
    "multi_ticker_swing_htf": REPO / "strategies/multi_ticker_swing_htf/backtest/results/family_compare_clean/comparison_summary_clean.json",
}


@pytest.fixture(scope="module")
def metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "baseline_metrics.csv")


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((OUT / "baseline_summary.json").read_text())


def test_artifacts_exist():
    for name in ["baseline_metrics.csv", "baseline_equity_curves.csv",
                 "random_k_seeds.csv", "baseline_summary.json"]:
        assert (OUT / name).exists(), f"missing {name}"


def test_all_strategies_present_per_window(metrics):
    for w in WINDOWS:
        strats = set(metrics.loc[metrics.window == w, "strategy"])
        for expected in ["spy_buy_hold", "equal_weight_universe", "sector_neutral_etf", "tbill_3m"]:
            assert expected in strats, f"{w}: missing {expected}"
        assert any(s.startswith("largest_stock_") for s in strats), f"{w}: missing largest_stock_*"
        assert any(s.startswith("module_") for s in strats), f"{w}: missing module row"
        assert any(s.startswith("random_top") for s in strats), f"{w}: missing random_top_k row"
        assert any(s.startswith("best_hindsight_pool_stock_") for s in strats), \
            f"{w}: missing best_hindsight_pool_stock_* oracle row"


def test_hindsight_oracle_beats_every_other_row(metrics):
    """The oracle pick is chosen BY its own window return, so by construction it
    must be >= every other strategy's return in that window (a broken selection
    or windowing bug would show up as the oracle losing to something)."""
    for w in WINDOWS:
        sub = metrics[metrics.window == w]
        oracle = sub[sub.strategy.str.startswith("best_hindsight_pool_stock_")].iloc[0]
        others = sub[~sub.strategy.str.startswith("best_hindsight_pool_stock_")]
        assert oracle.total_return_pct >= others.total_return_pct.max(), \
            f"{w}: oracle {oracle.total_return_pct}% did not beat max other row {others.total_return_pct.max()}%"


def test_tbill_behaves_like_cash(metrics):
    for w in WINDOWS:
        row = metrics[(metrics.window == w) & (metrics.strategy == "tbill_3m")].iloc[0]
        assert 1.0 < row.cagr_pct < 6.5, f"{w}: tbill CAGR {row.cagr_pct}% implausible"
        assert row.ann_vol_pct < 0.1
        assert row.max_dd_pct >= -0.01


def test_module_rows_match_locked_totals(metrics):
    for w in WINDOWS:
        dep = json.loads(CLEAN_SUMMARIES[w].read_text())["deployed_winner_frozen_test"]
        row = metrics[(metrics.window == w) & metrics.strategy.str.startswith("module_")].iloc[0]
        assert row.total_return_pct == pytest.approx(dep["total_return_pct"], abs=0.01), \
            f"{w}: module row {row.total_return_pct} != locked {dep['total_return_pct']}"


def test_equity_curves_reconcile_with_metrics(metrics):
    curves = pd.read_csv(OUT / "baseline_equity_curves.csv")
    for (w, s), grp in curves.groupby(["window", "strategy"]):
        if "median_seed" in s:
            continue  # single representative seed; table row is the seed mean
        row = metrics[(metrics.window == w) & (metrics.strategy == s)]
        assert len(row) == 1, f"({w}, {s}) missing from metrics table"
        end_ret = (grp.sort_values("date")["equity"].iloc[-1] / 100_000.0 - 1.0) * 100
        assert end_ret == pytest.approx(row.iloc[0].total_return_pct, abs=0.05)


def test_random_seeds_deterministic(summary):
    assert summary["random_seeds"] == list(range(123, 133))
    seeds = pd.read_csv(OUT / "random_k_seeds.csv")
    assert set(seeds["seed"]) == set(range(123, 133))
    assert (seeds.groupby("module")["seed"].count() == 10).all()
