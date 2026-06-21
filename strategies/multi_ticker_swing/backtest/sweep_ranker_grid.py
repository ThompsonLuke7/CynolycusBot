"""
Random grid search over the ranker's *order-policy* parameters (TP/SL/top_k/entry window).

These are execution knobs, not model parameters — nothing is trained here. Each sampled combo
is run through simulate.py on the competition test matrix with the TRADING_BLACKLIST applied,
and the resulting trade log is scored. Results are ranked by return/maxDD.

Usage:
  python -m strategies.multi_ticker_swing.backtest.sweep_ranker_grid --n 16
"""
from __future__ import annotations

import argparse
import itertools
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MOD = "strategies.multi_ticker_swing.backtest.simulate"
ROOT = Path(__file__).resolve().parents[3]
BT = ROOT / "strategies/multi_ticker_swing/backtest"
D = ROOT / "strategies/multi_ticker_swing/models/competition_20260615"

GRID = {
    "tp_atr_mult":    [1.5, 2.0, 2.5, 3.0, 3.5],
    "sl_atr_mult":    [2.0, 3.0, 4.0, 5.0],
    "top_k":          [5, 10, 15, 20],
    "entry_bars_max": [3, 6],
}


def score_run(results_dir: Path) -> dict:
    t = pd.read_parquet(results_dir / "trades.parquet")
    eq = pd.read_parquet(results_dir / "equity_curve.parquet")
    arr = eq["equity"].to_numpy(); peak = np.maximum.accumulate(arr); dd = ((arr - peak) / peak).min()
    pos = t.loc[t.pnl_dollar > 0, "pnl_dollar"].sum()
    neg = abs(t.loc[t.pnl_dollar <= 0, "pnl_dollar"].sum())
    pnl = t.pnl_dollar.sum()
    return {
        "trades": len(t),
        "win_rate": round((t.pnl_pct > 0).mean() * 100, 2),
        "profit_factor": round(pos / neg, 3) if neg else float("inf"),
        "exp_per_trade_pct": round(t.pnl_pct.mean() * 100, 4),
        "total_pnl": round(pnl, 0),
        "max_dd_pct": round(dd * 100, 2),
        "ret_over_dd": round((pnl / 100000) / abs(dd), 2) if dd else float("nan"),
    }


def run_combo(combo: dict, source: dict, out: Path) -> dict:
    """source = either {'model':..,'matrix':..,'features':..} or {'scores':..} + 'selection'."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", MOD, "--apply-blacklist", "--force",
        "--test-start", "2025-08-01", "--test-end", "2026-06-30",
        "--results-dir", str(out),
        "--selection", source["selection"],
        "--top-k", str(combo["top_k"]),
        "--tp-atr-mult", str(combo["tp_atr_mult"]),
        "--sl-atr-mult", str(combo["sl_atr_mult"]),
        "--entry-bars-max", str(combo["entry_bars_max"]),
    ]
    if source.get("scores"):
        cmd += ["--scores", str(source["scores"])]
    else:
        cmd += ["--matrix", str(source["matrix"]), "--features", str(source["features"]),
                "--model", str(source["model"]), "--model-kind", "ranker"]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return score_run(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16, help="number of random combos to sample")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--full", action="store_true", help="run every combo in the (possibly overridden) grid, no sampling")
    p.add_argument("--tp", default=None, help="comma list overriding tp_atr_mult grid")
    p.add_argument("--sl", default=None, help="comma list overriding sl_atr_mult grid")
    p.add_argument("--topk", default=None, help="comma list overriding top_k grid")
    p.add_argument("--entry", default=None, help="comma list overriding entry_bars_max grid")
    p.add_argument("--matrix", default=str(BT / "competition_20260615_test_matrix.parquet"))
    p.add_argument("--features", default=str(D / "selected_features.txt"))
    p.add_argument("--model", default=str(D / "model_xgb_ranker_seed44.joblib"))
    p.add_argument("--scores", default=None, help="precomputed OOF scores parquet (overrides model path)")
    p.add_argument("--selection", default="topk", choices=["topk", "topk_short", "bottomk", "both"])
    p.add_argument("--out", default=str(BT / "results_comp/grid"))
    args = p.parse_args()

    source = {"selection": args.selection}
    if args.scores:
        source["scores"] = args.scores
    else:
        source.update(matrix=args.matrix, features=args.features, model=args.model)

    grid = dict(GRID)
    if args.tp:    grid["tp_atr_mult"]    = [float(x) for x in args.tp.split(",")]
    if args.sl:    grid["sl_atr_mult"]    = [float(x) for x in args.sl.split(",")]
    if args.topk:  grid["top_k"]          = [int(x) for x in args.topk.split(",")]
    if args.entry: grid["entry_bars_max"] = [int(x) for x in args.entry.split(",")]

    baseline = {"tp_atr_mult": 2.5, "sl_atr_mult": 4.0, "top_k": 10, "entry_bars_max": 3}
    all_combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    if args.full:
        combos = all_combos
    else:
        random.Random(args.seed).shuffle(all_combos)
        # always evaluate the current default first, then random samples (deduped)
        combos = [baseline] + [c for c in all_combos if c != baseline][: max(args.n - 1, 0)]

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, combo in enumerate(combos, 1):
        tag = f"tp{combo['tp_atr_mult']}_sl{combo['sl_atr_mult']}_k{combo['top_k']}_e{combo['entry_bars_max']}"
        print(f"[{i}/{len(combos)}] {tag}", flush=True)
        try:
            m = run_combo(combo, source, out_root / tag)
            rows.append({**combo, **m})
            print(f"    -> pnl={m['total_pnl']:.0f} dd={m['max_dd_pct']} r/dd={m['ret_over_dd']} wr={m['win_rate']} n={m['trades']}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"    FAILED: {e}", flush=True)
        res = pd.DataFrame(rows).sort_values("ret_over_dd", ascending=False)
        res.to_csv(out_root / "grid_results.csv", index=False)  # checkpoint after each run

    print("\n=== GRID RESULTS (by return/maxDD) ===")
    print(pd.DataFrame(rows).sort_values("ret_over_dd", ascending=False).to_string(index=False))
    print(f"\nsaved -> {out_root / 'grid_results.csv'}")


if __name__ == "__main__":
    main()
