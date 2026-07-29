"""CLI: regime-conditional policy study (see HYPOTHESES.md).

Read-only against production data (oof_preds.parquet, daily_regime.parquet,
4H bars). Writes only under
``research/portfolio_lab/regime_policy/results/`` (a NEW results dir --
does not touch ``research/portfolio_lab/results/``).

Usage:
  PYTHONPATH=. .venv/bin/python -m research.portfolio_lab.regime_policy.run_study
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.backtest.family_backtest import BarCache, select_signals
from strategies.momentum_expansion.ablation.bootstrap import bh_fdr
from strategies.momentum_expansion.ablation.folds import build_walk_forward_folds, fold_label, train_test_masks

from research.portfolio_lab import portfolio_backtest as pb
from research.portfolio_lab.regime_policy import engine as E
from research.portfolio_lab.regime_policy import rules as R

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regime_policy.run_study")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _filter_by_ts(items, ts_attr, lo, hi):
    return [x for x in items if lo <= getattr(x, ts_attr) <= hi]


def _period_metrics(trades, snapshots, n_pool: int) -> dict:
    return pb.compute_metrics(trades, snapshots, capital=E.CAPITAL_REF, n_candidates_seen=n_pool)


def run(*, out_dir: Path = RESULTS_DIR, skip_h4: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("Loading signal stream (deployed model OOF, not trained here) ...")
    signals = E.load_signal_stream()
    logger.info("  %d rows, %d tickers, %s .. %s", len(signals), signals["ticker"].nunique(),
                signals["timestamp"].min(), signals["timestamp"].max())

    bar_cache = BarCache()

    # --- shared admission/sizing candidate stream (baseline + H1/H2/H3/H5) ---
    logger.info("Resolving fixed-stop candidate stream at top_k=%d (superset for H5-breadth) ...", E.WIDE_TOP_K)
    wide = E.resolve_fixed_stop_candidates(signals, bar_cache, top_k=E.WIDE_TOP_K)
    logger.info("  %d resolved candidates", len(wide))
    wide = E.attach_regime(wide, ts_col="signal_ts")
    n0 = len(wide)
    wide = wide.dropna(subset=E.REGIME_COLS).reset_index(drop=True)
    logger.info("  dropped %d/%d rows with NaN regime state (pre-warm-up); %d remain", n0 - len(wide), n0, len(wide))

    # Fold boundaries must be computed off the FULL training-matrix date range
    # (same as strategies.momentum_expansion.ablation.run_screen), not off the
    # OOF-derived candidate stream's own (narrower) date range -- the OOF span
    # IS the concatenation of every fold's test window, so using it as the
    # ts input to build_walk_forward_folds leaves no room for the 2-year
    # train lookback before the first fold and silently collapses 7 folds
    # into 2. Reuses the exact same fold spec/inputs as the prior regime
    # screen so results land on the SAME 7 periods already on record.
    from strategies.momentum_expansion.ablation.screen import load_training_matrix
    matrix_ts = load_training_matrix(columns=["ret_20"])["timestamp"]
    folds = build_walk_forward_folds(matrix_ts)
    logger.info("Built %d walk-forward folds (from training-matrix date range %s..%s)",
                len(folds), matrix_ts.min().date(), matrix_ts.max().date())

    baseline_trades, baseline_snaps = E.apply_rule(wide, E.baseline_rule())
    logger.info("Baseline: %d trades", len(baseline_trades))

    period_rows = []
    stat_rows = []

    def _score_rule(rule_id: str, group: str, rule_trades, rule_snaps, pool_len: int):
        for f in folds:
            label = fold_label(f)
            lo, hi = f["test_start"], f["test_end"]
            b_t = _filter_by_ts(baseline_trades, "signal_ts", lo, hi)
            b_s = [s for s in baseline_snaps if lo <= s.ts <= hi]
            r_t = _filter_by_ts(rule_trades, "signal_ts", lo, hi)
            r_s = [s for s in rule_snaps if lo <= s.ts <= hi]

            bm = _period_metrics(b_t, b_s, pool_len)
            rm = _period_metrics(r_t, r_s, pool_len)
            period_rows.append({"rule": rule_id, "group": group, "period": label, "arm": "baseline", **bm})
            period_rows.append({"rule": rule_id, "group": group, "period": label, "arm": "rule", **rm})

            boot = E.weekly_diff_bootstrap(b_t, r_t)
            stat_rows.append({"rule": rule_id, "group": group, "period": label,
                              "baseline_trades": len(b_t), "rule_trades": len(r_t), **boot})

    # --- H0: baseline vs itself is not a test; record baseline period metrics once ---
    for f in folds:
        label = fold_label(f)
        lo, hi = f["test_start"], f["test_end"]
        b_t = _filter_by_ts(baseline_trades, "signal_ts", lo, hi)
        b_s = [s for s in baseline_snaps if lo <= s.ts <= hi]
        period_rows.append({"rule": "baseline", "group": "baseline", "period": label, "arm": "baseline",
                            **_period_metrics(b_t, b_s, len(b_t))})

    # --- H1/H2/H3/H5 (admission + sizing rules over the shared wide stream) ---
    for rule in R.ADMISSION_SIZING_RULES:
        logger.info("Running rule %s (%s) ...", rule.rule_id, rule.group)
        rt, rs = E.apply_rule(wide, rule)
        logger.info("  %d trades (baseline had %d)", len(rt), len(baseline_trades))
        _score_rule(rule.rule_id, rule.group, rt, rs, len(wide))

    # --- H4 (variable stop distance -- separate resolution pass) ---
    if not skip_h4:
        logger.info("Resolving top_k=%d signal stream for H4 (variable stop) ...", E.BASE_TOP_K)
        sig10 = select_signals(signals[["timestamp", "ticker", "score"]], E.BASE_TOP_K, allow_short=False)
        sig10 = E.attach_regime(sig10, ts_col="timestamp")
        n0 = len(sig10)
        sig10 = sig10.dropna(subset=E.REGIME_COLS).reset_index(drop=True)
        logger.info("  %d signal rows (dropped %d NaN-regime)", len(sig10), n0 - len(sig10))

        for rule_id, sl_fn in R.H4_SL_MULT_FNS.items():
            logger.info("Running rule %s (H4) ...", rule_id)
            cands = E.resolve_variable_sl_candidates(sig10, bar_cache, sl_mult_fn=sl_fn)
            logger.info("  %d resolved candidates", len(cands))
            rt, rs = E.apply_rule(cands, E.Rule(rule_id, "H4", admit=lambda r: r.rank_in_bar <= E.BASE_TOP_K))
            logger.info("  %d trades (baseline had %d)", len(rt), len(baseline_trades))
            _score_rule(rule_id, "H4", rt, rs, len(sig10))
    else:
        logger.info("skip_h4=True -- H4 rules not run (smoke mode)")

    period_df = pd.DataFrame(period_rows)
    stat_df = pd.DataFrame(stat_rows)
    stat_df["q_fdr"] = bh_fdr(stat_df["p_value"])
    stat_df["survives_fdr"] = stat_df["q_fdr"] <= 0.10

    period_df.to_csv(out_dir / "period_metrics.csv", index=False)
    stat_df.to_csv(out_dir / "weekly_diff_bootstrap.csv", index=False)

    n_rules = stat_df["rule"].nunique()
    n_tests = len(stat_df)
    n_survive = int(stat_df["survives_fdr"].sum())
    logger.info("Wrote results to %s (%.1fs total)", out_dir, time.time() - t0)
    logger.info("%d rule variants x %d periods = %d tests in FDR family; %d survive q<=0.10",
                n_rules, len(folds), n_tests, n_survive)

    summary = {
        "n_folds": len(folds), "folds": [fold_label(f) for f in folds],
        "n_rule_variants": n_rules, "n_tests_in_fdr_family": n_tests, "n_survive_fdr_q10": n_survive,
        "baseline_trades_total": len(baseline_trades),
    }
    pd.Series(summary).to_json(out_dir / "summary.json", indent=2)
    return {"period_metrics": period_df, "stats": stat_df, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--skip-h4", action="store_true", help="smoke mode: skip the H4 variable-stop rules")
    args = ap.parse_args()
    run(out_dir=args.out_dir, skip_h4=args.skip_h4)


if __name__ == "__main__":
    main()
