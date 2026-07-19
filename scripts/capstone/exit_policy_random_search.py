"""
Random hyperparameter search over the Meta Ranker 4H SHARE exit policy.

Why this exists
----------------
`exit_policy_grid.py` (committed, paper-cited fig11/fig13) is a small, fixed
31-row grid, and its `scaleout_grid` section deliberately omits stop-loss to
isolate the scale-out mechanic. The user asked for (a) a genuine random search
over the space, (b) stop-loss included as a tuned parameter alongside
target/scale_frac/trail/horizon, and (c) shares only (no option-premium path
exists in this harness anyway).

Leakage discipline: `exit_policy_grid.py` searches its entire holdout
(2025-07-01+) in one pass and is fine ONLY because it's a small, hand-picked
grid reported as-is. A random search over hundreds of configs on that same
single window and reporting the winner would be textbook test-set tuning.
This script instead splits the holdout in two on a strict time boundary:
  - VAL  2025-07-01 -> 2026-01-15  (search happens here, all N configs)
  - TEST 2026-01-15 -> end of data (frozen: only the val-selected winners are
    ever evaluated here, once each)
Both are still walk-forward relative to the underlying models' own training
cutoffs; this is an additional in-holdout split for the exit-policy search
specifically, not a substitute for that.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_random_search.py [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import backtest_exits as be  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-15", tz="UTC")
TEST_END = pd.Timestamp("2026-05-15", tz="UTC")  # exclusive upper bound, past end of scored data

OUT_VAL = REPO / "research/capstone/exit_policy_random_search_val.csv"
OUT_TEST = REPO / "research/capstone/exit_policy_random_search_test.csv"


def _install_bar_cache() -> None:
    """_ticker_path's ts_index arg is unused; cache on ticker only. ~5x speedup
    across repeated simulate() calls that otherwise re-read every parquet."""
    orig = be._ticker_path
    cache: dict[str, pd.DataFrame | None] = {}

    def cached(ticker: str, ts_index):
        if ticker not in cache:
            cache[ticker] = orig(ticker, ts_index)
        return cache[ticker]

    be._ticker_path = cached


def _load_member_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(be.SCORED).dropna(subset=["s_combo"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rk"] = df.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["in_top"] = df["rk"] <= be.TOPK
    return df[["timestamp", "ticker", "in_top"]]


def sample_configs(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        cfg: dict = {}
        cfg["stop"] = None if rng.random() < 0.15 else round(float(rng.uniform(0.10, 0.75)), 2)
        cfg["trail"] = None if rng.random() < 0.35 else round(float(rng.uniform(0.10, 0.50)), 2)
        cfg["target"] = round(float(rng.uniform(0.05, 0.50)), 2)
        cfg["scale_frac"] = round(float(rng.uniform(0.10, 1.00)), 2)
        cfg["horizon"] = int(rng.integers(10, 61))
        cfg["grace"] = None
        configs.append(cfg)
    return configs


def run_grid(member: pd.DataFrame, configs: list[dict], label: str) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        s = be.simulate(member, **cfg)
        if not s:
            continue
        rows.append({**cfg, **s})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i + 1}/{len(configs)}  ({elapsed:.0f}s elapsed)")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not be.SCORED.exists():
        raise SystemExit(
            f"{be.SCORED} missing — regenerate first:\n"
            "  PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py"
        )

    _install_bar_cache()

    val_member = _load_member_window(VAL_START, VAL_END)
    test_member = _load_member_window(VAL_END, TEST_END)
    print(f"val window   {VAL_START.date()} -> {VAL_END.date()}  rows={len(val_member):,}")
    print(f"test window  {VAL_END.date()} -> {TEST_END.date()}  rows={len(test_member):,}")

    # Fixed reference rows so the random search is judged against known baselines.
    reference = [
        dict(stop=None, trail=None, target=None, scale_frac=1.0, horizon=None, grace=0),  # current live rebalance
        dict(stop=0.50, trail=0.35, target=0.20, scale_frac=0.5, horizon=25, grace=None),  # deployed
    ]
    configs = reference + sample_configs(args.n, args.seed)
    print(f"\nrunning {len(configs)} configs on VAL ({args.n} random + {len(reference)} reference, seed={args.seed})...")
    val_df = run_grid(val_member, configs, "val")
    val_df.insert(0, "policy_id", range(len(val_df)))
    val_df.to_csv(OUT_VAL, index=False)
    print(f"saved {OUT_VAL} ({len(val_df)} rows)")

    # Val-select winners by 4 different objectives (min 20 trades to avoid noise picks).
    eligible = val_df[val_df["n"] >= 20]
    winners = {}
    for objective in ("mean", "median", "win", "ret_per_bar"):
        idx = eligible[objective].idxmax()
        winners[objective] = int(val_df.loc[idx, "policy_id"])
    # de-dupe while preserving which objective(s) picked each one
    winner_ids = sorted(set(winners.values()) | {0, 1})
    picked = val_df[val_df["policy_id"].isin(winner_ids)].to_dict("records")

    print(f"\nval-selected winners: { {k: v for k, v in winners.items()} }")
    print(f"re-running {len(picked)} configs on FROZEN TEST window (never touched during search)...")
    test_rows = []
    for rec in picked:
        cfg = {k: rec[k] for k in ("stop", "trail", "target", "scale_frac", "horizon", "grace")}
        cfg = {k: (None if pd.isna(v) else v) for k, v in cfg.items()}
        s = be.simulate(test_member, **cfg)
        if not s:
            continue
        won_by = [obj for obj, pid in winners.items() if pid == rec["policy_id"]]
        test_rows.append({"policy_id": rec["policy_id"], "won_by": ",".join(won_by) or "reference", **cfg, **s})
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUT_TEST, index=False)
    print(f"saved {OUT_TEST} ({len(test_df)} rows)\n")

    print(f"{'policy_id':>9} {'won_by':22} {'stop':>6} {'trail':>6} {'target':>7} {'scale':>6} {'hz':>4} "
          f"| {'val mean':>9} {'test mean':>9} | {'val med':>8} {'test med':>8} | {'val win':>8} {'test win':>8} "
          f"| {'val r/bar':>10} {'test r/bar':>10}")
    for rec in test_df.to_dict("records"):
        v = val_df[val_df["policy_id"] == rec["policy_id"]].iloc[0]
        print(f"{rec['policy_id']:>9} {rec['won_by']:22} {str(rec['stop']):>6} {str(rec['trail']):>6} "
              f"{rec['target']:>7.2f} {rec['scale_frac']:>6.2f} {str(rec['horizon']):>4} "
              f"| {v['mean']:>9.4f} {rec['mean']:>9.4f} | {v['median']:>8.4f} {rec['median']:>8.4f} "
              f"| {v['win']:>8.4f} {rec['win']:>8.4f} | {v['ret_per_bar']:>10.5f} {rec['ret_per_bar']:>10.5f}")


if __name__ == "__main__":
    main()
