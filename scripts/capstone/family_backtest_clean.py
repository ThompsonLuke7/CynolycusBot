"""
Val-select / test-freeze patch for the momentum & HTF order-policy backtest.

strategies/momentum_expansion/backtest/run_family_compare.py sweeps the exit
policy grid (tp/sl/top_k/max_hold) and picks the winner BY ret_over_dd on the
SAME test window it then reports (leakage_audit.md §0.2 / §2 / §3 — the
existing comparison_summary.json numbers rail to the loosest grid edge,
e.g. momentum ret_over_dd=44.6x, because the test window is trend-favorable
and the policy was tuned to it). This script fixes that:

  1. sweep the full grid on the VALIDATION window (middle 20% by time) per
     family, using each family's already-selected best seed (model choice
     is a separate, already-documented selection — see leakage_audit.md §0.2),
  2. freeze the val-winning policy and simulate it, untouched, on the TEST
     window,
  3. report every family's frozen-test performance plus the deployed
     winner's frozen-test performance as the headline number.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/family_backtest_clean.py --strategy momentum
  PYTHONPATH=. .venv/bin/python scripts/capstone/family_backtest_clean.py --strategy htf
  PYTHONPATH=. .venv/bin/python scripts/capstone/family_backtest_clean.py --strategy all
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.backtest import family_backtest as fb
from strategies.momentum_expansion.backtest.run_family_compare import STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("family_compare_clean")

ROOT = Path(__file__).resolve().parents[2]


def compute_cutoffs(cfg: fb.StrategyConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    t = fb._read_reset(cfg.matrix_path, ["timestamp", cfg.target])
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.sort_values("timestamp").dropna(subset=[cfg.target]).reset_index(drop=True)
    n = len(t)
    train_end = t["timestamp"].iloc[int(n * cfg.train_frac)]
    test_start = t["timestamp"].iloc[int(n * (cfg.train_frac + cfg.val_frac))]
    return train_end, test_start


def load_window(cfg: fb.StrategyConfig, which: str, train_end: pd.Timestamp, test_start: pd.Timestamp) -> pd.DataFrame:
    cols = list(dict.fromkeys(["timestamp", "ticker", cfg.relevance_src] + cfg.feature_cols))
    if which == "val":
        filt = [("timestamp", ">=", train_end.to_pydatetime()), ("timestamp", "<", test_start.to_pydatetime())]
    elif which == "test":
        filt = [("timestamp", ">=", test_start.to_pydatetime())]
    else:
        raise ValueError(which)
    df = fb._read_reset(cfg.matrix_path, cols, filt)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    if cfg.exclude_low_price and "low_price_flag" in df.columns:
        keep = pd.to_numeric(df["low_price_flag"], errors="coerce").fillna(1.0) <= 0.0
        df = df[keep].reset_index(drop=True)
    df["relevance"] = (df[cfg.relevance_src] >= cfg.relevance_thr).astype(int)
    return df


def run(strategy: str) -> dict:
    spec = STRATEGIES[strategy]
    cfg = fb.StrategyConfig.from_manifest(
        spec["name"], spec["models_dir"], spec["matrix_path"],
        allow_short=spec["allow_short"], forward_window=spec["forward_window"],
    )
    out = ROOT / "strategies" / ("momentum_expansion" if strategy == "momentum" else "multi_ticker_swing_htf") \
        / "backtest" / "results" / "family_compare_clean"
    out.mkdir(parents=True, exist_ok=True)

    train_end, test_start = compute_cutoffs(cfg)
    logger.info("[%s] val window [%s, %s)  test window [%s, ->)", cfg.name, train_end, test_start, test_start)

    val_df = load_window(cfg, "val", train_end, test_start)
    test_df = load_window(cfg, "test", train_end, test_start)
    logger.info("[%s] val rows=%d  test rows=%d", cfg.name, len(val_df), len(test_df))

    seeds = fb.best_seed_per_family(cfg.models_dir)
    grid = fb.default_grid(cfg.forward_window)
    bars = fb.BarCache()

    per_family = []
    for family in fb.FAMILIES:
        seed = seeds[family]
        val_df["score"] = fb.score_family(val_df, cfg, family, seed)
        val_scored = val_df[["timestamp", "ticker", "score"]].copy()
        val_sweep = fb.sweep_family(val_scored, cfg, bars, grid, family)
        best_val = val_sweep.iloc[0]
        logger.info("[%s] %-16s s%d  VAL-picked policy: tp=%.1f sl=%.1f topk=%d hold=%d  "
                    "val ret/DD=%.2f val trades=%d",
                    cfg.name, family, seed, best_val.tp_atr_mult, best_val.sl_atr_mult,
                    int(best_val.top_k), int(best_val.max_hold), best_val.ret_over_dd, int(best_val.trades))

        test_df["score"] = fb.score_family(test_df, cfg, family, seed)
        test_scored = test_df[["timestamp", "ticker", "score"]].copy()
        sig = fb.select_signals(test_scored, int(best_val.top_k), cfg.allow_short)
        trades = fb.simulate(sig, bars, tp_mult=float(best_val.tp_atr_mult),
                             sl_mult=float(best_val.sl_atr_mult), max_hold=int(best_val.max_hold))
        m = fb.metrics(trades)
        m.update(
            family=family, seed=seed,
            selected_on="val", top_k=int(best_val.top_k),
            tp_atr_mult=float(best_val.tp_atr_mult), sl_atr_mult=float(best_val.sl_atr_mult),
            max_hold=int(best_val.max_hold),
            val_ret_over_dd=float(best_val.ret_over_dd), val_trades=int(best_val.trades),
        )
        per_family.append(m)
        trades.to_parquet(out / f"{family}_s{seed}_frozen_test_trades.parquet", index=False)

    per_family_df = pd.DataFrame(per_family)
    per_family_df.to_csv(out / "frozen_test_by_family.csv", index=False)

    with open(cfg.models_dir / "eval_metrics.json") as f:
        deployed_winner_family = json.load(f)["winner_family"]
    deployed_row = per_family_df[per_family_df["family"] == deployed_winner_family].iloc[0].to_dict()

    summary = {
        "strategy": cfg.name,
        "method": "policy selected on VAL window, frozen and reported on TEST window "
                  "(fixes leakage_audit.md selection-bias finding for the order-policy sweep)",
        "val_window": [str(train_end), str(test_start)],
        "test_window_start": str(test_start),
        "deployed_winner_family": deployed_winner_family,
        "deployed_winner_frozen_test": deployed_row,
        "all_families_frozen_test": per_family,
    }
    (out / "comparison_summary_clean.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("[%s] DONE -> %s", cfg.name, out)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], required=True)
    args = p.parse_args()
    targets = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    for s in targets:
        run(s)
