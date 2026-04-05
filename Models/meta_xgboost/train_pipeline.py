from __future__ import annotations

import argparse

from Models.meta_xgboost.train_entry import build_config as build_entry_config, run_entry_pipeline
from Models.meta_xgboost.train_exit import run_exit_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run meta XGBoost entry then exit pipelines.")
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--dataset-name", type=str, default="10min")
    p.add_argument("--processed-root", type=str, default=None)
    p.add_argument("--ga-model-root", type=str, default=None)
    p.add_argument("--model-root", type=str, default=None)
    p.add_argument("--pivot-label-dir", type=str, default="swing")
    p.add_argument("--tb-label-dir", type=str, default="tb")
    p.add_argument("--include-tb-probs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-vix-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--a-tp", type=float, default=1.6)
    p.add_argument("--b-sl", type=float, default=0.8)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--use-next-open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-cross-day", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--entry-mode", choices=["tp", "phase"], default="tp")
    p.add_argument("--hazard-k", type=int, default=1)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--initial-train-days", type=int, default=0)
    p.add_argument("--purge-days", type=int, default=0)
    p.add_argument("--threshold-min", type=float, default=0.50)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.05)
    p.add_argument("--threshold-objective", type=str, default="f1")
    p.add_argument("--min-oos-prob-coverage", type=float, default=0.85)
    p.add_argument("--fixed-long-threshold", type=float, default=None)
    p.add_argument("--fixed-short-threshold", type=float, default=None)
    p.add_argument("--trail-activate-atr", type=float, default=2.0)
    p.add_argument("--trail-atr", type=float, default=1.0)
    p.add_argument("--trail-atr-after-tp", type=float, default=0.8)
    p.add_argument("--use-tp-to-tighten-trail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--xgb-booster", choices=["gbtree", "dart"], default=None)
    p.add_argument("--xgb-rate-drop", type=float, default=None)
    p.add_argument("--xgb-skip-drop", type=float, default=None)
    p.add_argument("--xgb-one-drop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--xgb-sample-type", choices=["uniform", "weighted"], default=None)
    p.add_argument("--xgb-normalize-type", choices=["tree", "forest"], default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--early-stopping-rounds", type=int, default=200)
    p.add_argument("--early-stopping-val-fraction", type=float, default=0.20)
    p.add_argument("--early-stopping-min-val-rows", type=int, default=100)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_entry_config(args)
    entry_artifacts = run_entry_pipeline(cfg)
    exit_artifacts = run_exit_pipeline(
        cfg,
        entry_root=entry_artifacts["entry_root"],
        entry_long_threshold=None,
        entry_short_threshold=None,
        fixed_long_threshold=args.fixed_long_threshold,
        fixed_short_threshold=args.fixed_short_threshold,
        trail_activate_atr=float(args.trail_activate_atr),
        trail_atr=float(args.trail_atr),
        trail_atr_after_tp=float(args.trail_atr_after_tp),
        use_tp_to_tighten_trail=bool(args.use_tp_to_tighten_trail),
    )
    print(f"[META-PIPELINE] Entry probs: {entry_artifacts['entry_probs_path']}")
    print(f"[META-PIPELINE] Exit probs: {exit_artifacts['exit_probs_path']}")


if __name__ == "__main__":
    main()
