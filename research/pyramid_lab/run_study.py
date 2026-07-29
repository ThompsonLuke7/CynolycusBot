"""CLI: pyramiding ("add to winners") study — see HYPOTHESES.md.

Read-only against production data (each module's OOF preds, 4H bars). Writes
only under ``research/pyramid_lab/results/`` (a NEW directory; it does not
touch ``research/portfolio_lab/results/`` or ``research/capstone/``).

Run one module at a time to keep each command short:

  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.run_study --module momentum
  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.run_study --module htf
  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.run_study --module meta

then combine + BH-FDR across the whole family:

  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.aggregate
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation.folds import build_walk_forward_folds, fold_label

from research.pyramid_lab import arms as A
from research.pyramid_lab import metrics as M
from research.pyramid_lab import streams as S
from research.pyramid_lab.engine import BasePolicy, PyramidPolicy, simulate_ticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pyramid_lab.run_study")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COST_BPS = 10.0                 # per-fill, matches portfolio_backtest/run_comparison convention
BASE = BasePolicy()             # live ExecPolicy defaults (see HYPOTHESES.md)


def walk_forward_periods() -> list[dict]:
    """The repo's one fold spec, on the momentum training-matrix date range.

    Deliberately NOT built from an OOF stream's own (narrower) range: doing so
    leaves no room for the 2-year train lookback and silently collapses 7 folds
    into 2 — a trap already documented in ``regime_policy/run_study.py``.
    """
    from strategies.momentum_expansion.ablation.screen import load_training_matrix
    ts = load_training_matrix(columns=["ret_20"])["timestamp"]
    return build_walk_forward_folds(ts)


def run_arm(masks: dict, master: np.ndarray, idx_map: dict, pyr: PyramidPolicy,
            *, notional: float, cost_bps: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate one arm across every ticker; return (positions, pnl_by_bar,
    deployed_by_bar) on the master bar grid."""
    pnl = np.zeros(len(master))
    dep = np.zeros(len(master))
    rows = []
    for tkr, m in masks.items():
        n = len(m["close"])
        p_local = np.zeros(n)
        d_local = np.zeros(n)
        positions = simulate_ticker(
            m["close"], m["high"], m["low"], m["member"], ticker=tkr,
            base=BASE, pyr=pyr, notional=notional, cost_bps=cost_bps,
            pnl_by_bar=p_local, deployed_by_bar=d_local,
        )
        if not positions:
            continue
        ix = idx_map[tkr]
        pnl[ix] += p_local
        dep[ix] += d_local
        ts = m["ts"]
        for p in positions:
            rows.append({
                "ticker": p.ticker, "entry_ts": ts[p.entry_i], "exit_ts": ts[p.exit_i],
                "entry_price": p.entry_price, "exit_price": p.exit_price,
                "exit_reason": p.exit_reason, "bars_held": p.bars_held, "n_adds": p.n_adds,
                "initial_notional": p.initial_notional, "added_notional": p.added_notional,
                "peak_cost_basis": p.peak_cost_basis, "fill_notional": p.fill_notional,
                "pnl_gross": p.pnl_gross, "fees": p.fees, "pnl_net": p.pnl_net,
                "ret_on_initial": p.ret_on_initial,
            })
    return pd.DataFrame(rows), pnl, dep


def run_module(module: str, *, out_dir: Path, arms: dict[str, PyramidPolicy],
               notional: float = BASE.target_notional, cost_bps: float = COST_BPS,
               tag: str = "primary", save_positions: bool = True) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    stream = S.load_module_stream(module)
    bars = S.Bars()
    masks = S.build_member_masks(stream, bars)
    master = S.master_index(masks)
    logger.info("[%s] %d tickers with bars, master grid = %d bars (%s .. %s)",
                module, len(masks), len(master), master.min(), master.max())
    idx_map = {t: np.searchsorted(master, m["ts"]) for t, m in masks.items()}

    folds = walk_forward_periods()
    results: dict[str, tuple] = {}
    for arm_id, pyr in arms.items():
        t1 = time.time()
        pos, pnl, dep = run_arm(masks, master, idx_map, pyr, notional=notional, cost_bps=cost_bps)
        results[arm_id] = (pos, pnl, dep)
        logger.info("[%s/%s] %d positions, %d adds, net $%.0f, avg deployed $%.0f (%.1fs)",
                    module, arm_id, len(pos), int(pos["n_adds"].sum()) if len(pos) else 0,
                    pnl.sum(), dep.mean(), time.time() - t1)

    base_pos, base_pnl, base_dep = results[A.BASELINE_ARM]

    # "full" spans the module's OWN signal life, not the whole bar grid: the
    # union bar grid starts years before the first OOF score, and averaging
    # deployed capital over those empty bars would flatter every arm's
    # return-per-dollar-deployed identically but meaninglessly.
    full_lo = stream["timestamp"].min()
    full_hi = pd.Timestamp(base_pos["exit_ts"].max(), tz="UTC") if len(base_pos) else stream["timestamp"].max()

    period_rows, stat_rows = [], []
    windows = [("full", full_lo, full_hi)] + [
        (fold_label(f), f["test_start"], f["test_end"]) for f in folds
    ]
    for arm_id, (pos, pnl, dep) in results.items():
        for label, lo, hi in windows:
            met = M.arm_metrics(master, pnl, dep, pos, lo=lo, hi=hi)
            period_rows.append({"module": module, "tag": tag, "arm": arm_id, "period": label, **met})
            if arm_id == A.BASELINE_ARM:
                continue
            bw = M.weekly_pnl(master, base_pnl, lo=lo, hi=hi)
            aw = M.weekly_pnl(master, pnl, lo=lo, hi=hi)
            boot = M.weekly_diff_bootstrap(bw, aw)
            bm = M.arm_metrics(master, base_pnl, base_dep, base_pos, lo=lo, hi=hi)
            # CAPITAL-MATCHED statistic: the same week-block bootstrap, but on
            # the arm's weekly P&L rescaled so its average deployed capital
            # equals the baseline's. The raw statistic above rewards an arm
            # simply for putting more money to work in an up-market; this one
            # does not. Exact (not an approximation) because sizing is linear
            # under basis='entry' -- asserted in
            # tests/test_engine.py::test_capital_linearity_under_entry_basis.
            ratio = ((met.get("avg_deployed") or 0.0) / bm["avg_deployed"]
                     if bm.get("avg_deployed") else float("nan"))
            boot_cm = (M.weekly_diff_bootstrap(bw, aw / ratio)
                       if np.isfinite(ratio) and ratio > 0
                       else {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                             "n_weeks": 0, "excludes_zero": False, "p_value": np.nan})
            stat_rows.append({
                "module": module, "tag": tag, "arm": arm_id, "period": label,
                "baseline_trades": bm.get("trades", 0), "arm_trades": met.get("trades", 0),
                "baseline_pnl": bm.get("total_pnl_net"), "arm_pnl": met.get("total_pnl_net"),
                "baseline_avg_deployed": bm.get("avg_deployed"), "arm_avg_deployed": met.get("avg_deployed"),
                "baseline_rpd": bm.get("ret_per_dollar_deployed"), "arm_rpd": met.get("ret_per_dollar_deployed"),
                "baseline_sharpe": bm.get("sharpe_daily"), "arm_sharpe": met.get("sharpe_daily"),
                "baseline_maxdd": bm.get("max_dd_dollars"), "arm_maxdd": met.get("max_dd_dollars"),
                "baseline_peak_deployed": bm.get("peak_deployed"),
                "arm_peak_deployed": met.get("peak_deployed"),
                **boot,
                **{f"cm_{k}": v for k, v in boot_cm.items()},
            })

    period_df = pd.DataFrame(period_rows)
    stat_df = pd.DataFrame(stat_rows)
    period_df.to_csv(out_dir / f"{module}_{tag}_period_metrics.csv", index=False)
    stat_df.to_csv(out_dir / f"{module}_{tag}_weekly_diff.csv", index=False)
    if save_positions:
        for arm_id, (pos, _, _) in results.items():
            if arm_id in (A.BASELINE_ARM, "L20_a100_m1"):  # baseline + one illustrative arm
                pos.to_parquet(out_dir / f"{module}_{tag}_positions_{arm_id}.parquet", index=False)

    logger.info("[%s] done in %.1fs -> %s", module, time.time() - t0, out_dir)
    return {"period": period_df, "stats": stat_df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", required=True, choices=list(S.MODULES))
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    ap.add_argument("--notional", type=float, default=BASE.target_notional)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--arms", default="primary", choices=["primary", "blended", "smoke"])
    ap.add_argument("--blended-of", default="", help="comma-separated primary arm ids for --arms blended")
    args = ap.parse_args()

    if args.arms == "primary":
        arms = A.primary_arms()
    elif args.arms == "smoke":
        p = A.primary_arms()
        arms = {k: p[k] for k in (A.BASELINE_ARM, "L20_a100_m1", "RESEL_a50_m2")}
    else:
        ids = [s for s in args.blended_of.split(",") if s]
        arms = {A.BASELINE_ARM: A.primary_arms()[A.BASELINE_ARM], **A.blended_basis_arms(ids)}

    run_module(args.module, out_dir=args.out_dir, arms=arms, notional=args.notional,
               cost_bps=args.cost_bps, tag=args.tag)


if __name__ == "__main__":
    main()
