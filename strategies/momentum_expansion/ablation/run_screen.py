"""CLI entry point for the WS-E training-free local screen (deliverable 2).

Reads production parquets READ-ONLY (never overwrites
``training_matrix_4h.parquet`` / ``features_4h.parquet``); writes only under
``strategies/momentum_expansion/ablation/results/``.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.run_screen
    .venv/bin/python -m strategies.momentum_expansion.ablation.run_screen --n-boot 200  # faster smoke run

Runtime: ~1-3 minutes on the real 2.3M-row training matrix + 1.6M-row OOF
table (both already cached on disk; nothing is rebuilt or fetched).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation import screen as S
from strategies.momentum_expansion.ablation.folds import build_walk_forward_folds, fold_label, train_test_masks

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(*, n_boot: int = 500, out_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("Loading training matrix (read-only) ...")
    matrix = S.load_training_matrix()
    logger.info("  %d rows, %s .. %s", len(matrix), matrix["timestamp"].min(), matrix["timestamp"].max())

    logger.info("Loading deployed model OOF predictions (the 'existing signal', not trained here) ...")
    oof = S.load_oof_scores()
    logger.info("  %d rows, %s .. %s, %d tickers", len(oof), oof["timestamp"].min(), oof["timestamp"].max(),
                oof["ticker"].nunique())

    logger.info("Joining market-regime / sector-context features onto both frames (read-only WS-A tables) ...")
    matrix = S.join_regime_and_sector_features(matrix)
    oof = S.join_market_wide_features(oof)

    logger.info("Empirical taxonomy check: sector_breadth_20/50 disagreement rate across sectors ...")
    disagreement = S.sector_breadth_disagreement_rate()
    logger.info("  %s", disagreement)

    folds = build_walk_forward_folds(matrix["timestamp"])
    logger.info("Built %d walk-forward folds from WALK_FORWARD_CONFIG (2.0y train / 21d embargo / 6mo test)", len(folds))
    for f in folds:
        logger.info("  fold %s (train %s..%s)", fold_label(f), f["train_start"].date(), f["train_end"].date())

    group1_bucket_rows: list[pd.DataFrame] = []
    group1_spread_rows: list[dict] = []
    group2_rows: list[dict] = []
    group3_rows: list[dict] = []

    for f in folds:
        label = fold_label(f)
        _, test_mask_oof = train_test_masks(oof["timestamp"], f)
        oof_period = oof[test_mask_oof]
        _, test_mask_matrix = train_test_masks(matrix["timestamp"], f)
        matrix_period = matrix[test_mask_matrix]

        # ---- Group 1: market-wide regime-conditional expectancy of the
        # existing deployed model's top-10 selection ----------------------
        if not oof_period.empty:
            for feat in S.MARKET_WIDE_FEATURES:
                buckets = S.market_wide_conditional_expectancy(
                    oof_period, feature_col=feat, score_col="oof_score",
                    label_col="fwd_max_alpha", win_col="fwd_close_return",
                    dd_col="fwd_max_drawdown", n_buckets=5, top_n=10,
                )
                if not buckets.empty:
                    buckets = buckets.assign(period=label, feature=feat)
                    group1_bucket_rows.append(buckets)
                spread = S.market_wide_spread_bootstrap(
                    oof_period, feature_col=feat, score_col="oof_score",
                    label_col="fwd_max_alpha", n_buckets=5, top_n=10, n_boot=n_boot,
                )
                spread.update({"period": label, "feature": feat, "group": "market_wide"})
                group1_spread_rows.append(spread)

        # ---- Group 2: sector-level rank IC (ticker-level AND sector-level) ---
        if not matrix_period.empty:
            for feat in S.SECTOR_LEVEL_FEATURES:
                ticker_level = S.rank_ic_with_bootstrap(
                    matrix_period, feature_col=feat, label_col=S.PRIMARY_LABEL, n_boot=n_boot,
                )
                ticker_level.update({"period": label, "feature": feat, "group": "sector_level", "granularity": "ticker"})
                group2_rows.append(ticker_level)

                sector_deduped = S.sector_level_dedup(matrix_period, feature_col=feat, label_col=S.PRIMARY_LABEL)
                sector_level = S.rank_ic_with_bootstrap(
                    sector_deduped, feature_col=feat, label_col=S.PRIMARY_LABEL, n_boot=n_boot,
                )
                sector_level.update({"period": label, "feature": feat, "group": "sector_level", "granularity": "sector"})
                group2_rows.append(sector_level)

            # ---- Group 3: interaction rank IC (ticker-level, primary metric) -
            for feat in S.INTERACTION_FEATURES:
                res = S.rank_ic_with_bootstrap(
                    matrix_period, feature_col=feat, label_col=S.PRIMARY_LABEL, n_boot=n_boot,
                )
                res.update({"period": label, "feature": feat, "group": "interaction", "granularity": "ticker"})
                group3_rows.append(res)

    group1_buckets_df = pd.concat(group1_bucket_rows, ignore_index=True) if group1_bucket_rows else pd.DataFrame()
    group1_spread_df = S.apply_bh_fdr(group1_spread_rows)
    group2_df = S.apply_bh_fdr(group2_rows)
    group3_df = S.apply_bh_fdr(group3_rows)

    group1_buckets_df.to_csv(out_dir / "group1_market_wide_buckets.csv", index=False)
    group1_spread_df.to_csv(out_dir / "group1_market_wide_spread.csv", index=False)
    group2_df.to_csv(out_dir / "group2_sector_level_rank_ic.csv", index=False)
    group3_df.to_csv(out_dir / "group3_interaction_rank_ic.csv", index=False)
    pd.DataFrame([disagreement]).to_csv(out_dir / "sector_breadth_disagreement_check.csv", index=False)

    logger.info("Wrote results to %s (%.1fs total)", out_dir, time.time() - t_start)
    return {
        "group1_buckets": group1_buckets_df,
        "group1_spread": group1_spread_df,
        "group2": group2_df,
        "group3": group3_df,
        "disagreement": disagreement,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    results = run(n_boot=args.n_boot, out_dir=args.out_dir)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Group 1 (market-wide) top-bottom spread, BH-FDR survivors ===")
    g1 = results["group1_spread"]
    if not g1.empty:
        print(g1[g1["survives_fdr"] == True][  # noqa: E712
            ["period", "feature", "point", "ci_lo", "ci_hi", "n_weeks_top", "n_weeks_bot", "p_value", "q_fdr"]
        ].to_string(index=False))
        print(f"({(g1['survives_fdr'] == True).sum()}/{len(g1)} tests survive BH-FDR q<=0.10)")  # noqa: E712

    print("\n=== Group 2 (sector-level) rank IC, BH-FDR survivors ===")
    g2 = results["group2"]
    if not g2.empty:
        print(g2[g2["survives_fdr"] == True][  # noqa: E712
            ["period", "feature", "granularity", "mean_ic", "n_weeks", "p_value", "q_fdr"]
        ].to_string(index=False))
        print(f"({(g2['survives_fdr'] == True).sum()}/{len(g2)} tests survive BH-FDR q<=0.10)")  # noqa: E712

    print("\n=== Group 3 (interactions) rank IC, BH-FDR survivors ===")
    g3 = results["group3"]
    if not g3.empty:
        print(g3[g3["survives_fdr"] == True][  # noqa: E712
            ["period", "feature", "mean_ic", "n_weeks", "p_value", "q_fdr"]
        ].to_string(index=False))
        print(f"({(g3['survives_fdr'] == True).sum()}/{len(g3)} tests survive BH-FDR q<=0.10)")  # noqa: E712


if __name__ == "__main__":
    main()
