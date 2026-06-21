"""
Rankers-vs-classifiers + best-order-policy experiment on the held-out test set.

Runs the shared engine (family_backtest.py) for one strategy:
  - load the test window (last 20% by timestamp),
  - score all 4 competition families (each at its best seed by test_ndcg@10),
  - sweep the order-policy grid per family,
  - pick each family's best policy and compare classifiers vs rankers,
  - persist scored frames + best-policy trades for the plotting step.

Usage:
  .venv/bin/python -m strategies.momentum_expansion.backtest.run_family_compare --strategy momentum
  .venv/bin/python -m strategies.momentum_expansion.backtest.run_family_compare --strategy htf
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.backtest import family_backtest as fb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("family_compare")

ROOT = Path(__file__).resolve().parents[3]

STRATEGIES = {
    "momentum": dict(
        name="momentum_expansion",
        models_dir=ROOT / "strategies/momentum_expansion/models/expansion_v1",
        matrix_path=ROOT / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet",
        allow_short=False, forward_window=25,
        results_dir=ROOT / "strategies/momentum_expansion/backtest/results/family_compare",
        plot_dir=ROOT / "strategies/momentum_expansion/plots/output/family_compare",
    ),
    "htf": dict(
        name="multi_ticker_swing_htf",
        models_dir=ROOT / "strategies/multi_ticker_swing_htf/models",
        matrix_path=ROOT / "strategies/multi_ticker_swing_htf/data/processed/training_matrix_4h.parquet",
        allow_short=True, forward_window=25,
        results_dir=ROOT / "strategies/multi_ticker_swing_htf/backtest/results/family_compare",
        plot_dir=ROOT / "strategies/multi_ticker_swing_htf/plots/family_compare",
    ),
}


def run(strategy: str) -> None:
    spec = STRATEGIES[strategy]
    cfg = fb.StrategyConfig.from_manifest(
        spec["name"], spec["models_dir"], spec["matrix_path"],
        allow_short=spec["allow_short"], forward_window=spec["forward_window"],
    )
    out = Path(spec["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)

    df = fb.load_test_window(cfg)
    seeds = fb.best_seed_per_family(cfg.models_dir)
    logger.info("[%s] best seeds: %s", cfg.name, seeds)
    grid = fb.default_grid(cfg.forward_window)
    bars = fb.BarCache()

    all_results, best_rows = [], []
    for family in fb.FAMILIES:
        seed = seeds[family]
        df["score"] = fb.score_family(df, cfg, family, seed)
        scored = df[["timestamp", "ticker", "score", "relevance", cfg.relevance_src]].copy()
        scored.to_parquet(out / "scores" / f"{family}_s{seed}.parquet", index=False)

        res = fb.sweep_family(scored[["timestamp", "ticker", "score"]], cfg, bars, grid, family)
        res["seed"] = seed
        all_results.append(res)
        best = res.iloc[0].to_dict()
        best_rows.append(best)
        logger.info("[%s] %-16s s%d  best: ret=%.1f%% maxDD=%.1f%% ret/DD=%.2f PF=%.2f trades=%d "
                    "(tp=%.1f sl=%.1f topk=%d hold=%d)",
                    cfg.name, family, seed, best["total_return_pct"], best["max_dd_pct"],
                    best["ret_over_dd"], best["profit_factor"], best["trades"],
                    best["tp_atr_mult"], best["sl_atr_mult"], best["top_k"], best["max_hold"])

    sweep_df = pd.concat(all_results, ignore_index=True)
    sweep_df.to_csv(out / "sweep_results.csv", index=False)
    best_df = pd.DataFrame(best_rows).sort_values("ret_over_dd", ascending=False).reset_index(drop=True)
    best_df.to_csv(out / "best_by_family.csv", index=False)

    # persist the overall-best family's best-policy trades for plotting
    overall = best_df.iloc[0]
    seed = int(overall["seed"])
    scored = pd.read_parquet(out / "scores" / f"{overall['family']}_s{seed}.parquet")
    sig = fb.select_signals(scored[["timestamp", "ticker", "score"]], int(overall["top_k"]), cfg.allow_short)
    trades = fb.simulate(sig, bars, tp_mult=float(overall["tp_atr_mult"]),
                         sl_mult=float(overall["sl_atr_mult"]), max_hold=int(overall["max_hold"]))
    trades.to_parquet(out / "best_trades.parquet", index=False)

    clf_best = best_df[best_df.family.str.contains("classifier")]["ret_over_dd"].max()
    rnk_best = best_df[best_df.family.str.contains("ranker")]["ret_over_dd"].max()

    # Robust head-to-head: fraction of *matched* policies where the classifier beats its
    # paired ranker (same algo, same tp/sl/top_k/hold) on ret_over_dd. This is far more
    # reliable than comparing single best-policy cells, where a noisy outlier can flip the
    # verdict (e.g. HTF: ranker edges on its single best cell but loses ~95% of matched policies).
    key = ["top_k", "tp_atr_mult", "sl_atr_mult", "max_hold"]
    piv = sweep_df.pivot_table(index=key, columns="family", values="ret_over_dd")
    matched = {}
    for algo in ("xgb", "lgbm"):
        c, r = f"{algo}_classifier", f"{algo}_ranker"
        if c in piv and r in piv:
            matched[f"{algo}_clf_beats_ranker_frac"] = round(float((piv[c] > piv[r]).mean()), 3)
    mean_frac = float(np.mean(list(matched.values()))) if matched else 0.5
    winner_type = "classifier" if mean_frac >= 0.5 else "ranker"

    # Also report the policy that maximizes per-trade profit factor (trade-count invariant),
    # which is less sensitive to the breadth/regime effect than ret_over_dd.
    pf_best = sweep_df.sort_values("profit_factor", ascending=False).iloc[0]

    summary = {
        "strategy": cfg.name,
        "allow_short": cfg.allow_short,
        "rank_metric": "ret_over_dd",
        "regime_note": ("Test window is the last 20% by time. tp/sl/hold tend to rail to looser "
                        "values because the window is trend-favorable; treat the absolute best "
                        "policy as regime-conditioned and prefer the matched-policy clf-vs-ranker "
                        "verdict and the profit-factor-best policy as the more robust read."),
        "overall_best_by_ret_over_dd": {k: overall[k] for k in
                         ["family", "seed", "top_k", "tp_atr_mult", "sl_atr_mult", "max_hold",
                          "total_return_pct", "max_dd_pct", "ret_over_dd", "profit_factor",
                          "win_rate", "trades", "avg_trade_pct"]},
        "best_by_profit_factor": {k: pf_best[k] for k in
                         ["family", "top_k", "tp_atr_mult", "sl_atr_mult", "max_hold",
                          "profit_factor", "win_rate", "avg_trade_pct", "ret_over_dd", "trades"]},
        "clf_vs_ranker": {"winner": winner_type, "best_classifier_ret_over_dd": float(clf_best),
                          "best_ranker_ret_over_dd": float(rnk_best),
                          "matched_policy_head_to_head": matched},
        "best_by_family": best_df.to_dict(orient="records"),
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("[%s] DONE. clf-vs-ranker winner=%s. Wrote %s", cfg.name, winner_type, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    args = p.parse_args()
    run(args.strategy)
