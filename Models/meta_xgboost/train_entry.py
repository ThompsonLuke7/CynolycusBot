from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from Policy.training_logging import log_training_run
from Models.meta_xgboost.common import (
    ENTRY_TARGETS,
    PipelineConfig,
    add_entry_targets,
    binary_metrics,
    build_base_feature_frame,
    choose_threshold,
    compute_entry_embargo_end_idx,
    resolve_meta_dataset_root,
    save_booster_artifacts,
    save_prob_frame,
    select_numeric_feature_columns,
    sweep_thresholds,
    train_walkforward_binary,
    xgb_params_from_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train meta ENTRY XGBoost models with walk-forward OOF.")
    p.add_argument("--ticker", type=str, default=PipelineConfig.ticker)
    p.add_argument("--dataset-name", type=str, default=PipelineConfig.dataset_name)
    p.add_argument("--processed-root", type=str, default=None)
    p.add_argument("--ga-model-root", type=str, default=None)
    p.add_argument("--model-root", type=str, default=None)
    p.add_argument("--pivot-label-dir", type=str, default=PipelineConfig.pivot_label_dir)
    p.add_argument("--tb-label-dir", type=str, default=PipelineConfig.tb_label_dir)
    p.add_argument("--include-vix-features", action=argparse.BooleanOptionalAction, default=PipelineConfig.include_vix_features)
    p.add_argument("--a-tp", type=float, default=PipelineConfig.a_tp)
    p.add_argument("--b-sl", type=float, default=PipelineConfig.b_sl)
    p.add_argument("--cost-bps", type=float, default=PipelineConfig.cost_bps)
    p.add_argument("--use-next-open", action=argparse.BooleanOptionalAction, default=PipelineConfig.use_next_open)
    p.add_argument("--allow-cross-day", action=argparse.BooleanOptionalAction, default=PipelineConfig.allow_cross_day)
    p.add_argument("--entry-mode", choices=["tp", "phase"], default=PipelineConfig.entry_mode)
    p.add_argument("--n-folds", type=int, default=PipelineConfig.n_folds)
    p.add_argument("--initial-train-days", type=int, default=PipelineConfig.initial_train_days)
    p.add_argument("--purge-days", type=int, default=PipelineConfig.purge_days)
    p.add_argument("--threshold-min", type=float, default=PipelineConfig.threshold_min)
    p.add_argument("--threshold-max", type=float, default=PipelineConfig.threshold_max)
    p.add_argument("--threshold-step", type=float, default=PipelineConfig.threshold_step)
    p.add_argument("--threshold-objective", type=str, default=PipelineConfig.threshold_objective)
    p.add_argument("--min-oos-prob-coverage", type=float, default=PipelineConfig.min_oos_prob_coverage)
    p.add_argument("--xgb-booster", choices=["gbtree", "dart"], default=None)
    p.add_argument("--xgb-rate-drop", type=float, default=None)
    p.add_argument("--xgb-skip-drop", type=float, default=None)
    p.add_argument("--xgb-one-drop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--xgb-sample-type", choices=["uniform", "weighted"], default=None)
    p.add_argument("--xgb-normalize-type", choices=["tree", "forest"], default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--random-state", type=int, default=PipelineConfig.random_state)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        ticker=args.ticker,
        dataset_name=args.dataset_name,
        processed_root=args.processed_root,
        ga_model_root=args.ga_model_root,
        model_root=args.model_root,
        pivot_label_dir=args.pivot_label_dir,
        tb_label_dir=args.tb_label_dir,
        include_vix_features=bool(args.include_vix_features),
        a_tp=float(args.a_tp),
        b_sl=float(args.b_sl),
        cost_bps=float(args.cost_bps),
        use_next_open=bool(args.use_next_open),
        allow_cross_day=bool(args.allow_cross_day),
        entry_mode=str(args.entry_mode),
        n_folds=int(args.n_folds),
        initial_train_days=int(args.initial_train_days),
        purge_days=int(args.purge_days),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        threshold_step=float(args.threshold_step),
        threshold_objective=str(args.threshold_objective),
        min_oos_prob_coverage=float(args.min_oos_prob_coverage),
        xgb_booster=args.xgb_booster,
        xgb_rate_drop=args.xgb_rate_drop,
        xgb_skip_drop=args.xgb_skip_drop,
        xgb_one_drop=args.xgb_one_drop,
        xgb_sample_type=args.xgb_sample_type,
        xgb_normalize_type=args.xgb_normalize_type,
        n_estimators=args.n_estimators,
        random_state=int(args.random_state),
    )


def run_entry_pipeline(cfg: PipelineConfig) -> dict[str, Path | dict[str, float] | pd.DataFrame]:
    frame = add_entry_targets(build_base_feature_frame(cfg), cfg)
    exclude = {
        "open", "high", "low", "close", "session_date", cfg.atr_col,
        *ENTRY_TARGETS,
        "y_exit_long", "y_exit_short", "y_exit_long_point", "y_exit_short_point",
        "exit_reason_long", "exit_reason_short", "tp_hit_before_exit_long", "tp_hit_before_exit_short",
    }
    feature_cols = select_numeric_feature_columns(frame, exclude=exclude)
    xgb_params = xgb_params_from_config(cfg)
    session_dates = frame["session_date"]

    entry_root = resolve_meta_dataset_root(cfg) / "entry"
    entry_root.mkdir(parents=True, exist_ok=True)

    target_to_prob = {
        "y_enter_long": "p_enter_long",
        "y_enter_short": "p_enter_short",
    }
    summary_key = {
        "y_enter_long": "enter_long",
        "y_enter_short": "enter_short",
    }
    combined_cols: dict[str, np.ndarray] = {}
    threshold_summary: dict[str, dict[str, float | None]] = {}
    metrics_summary: dict[str, dict[str, dict[str, float]]] = {}

    for target_col, prob_prefix in target_to_prob.items():
        embargo_end_idx = compute_entry_embargo_end_idx(frame, cfg, target_col=target_col)
        result = train_walkforward_binary(
            df=frame,
            feature_cols=feature_cols,
            target_col=target_col,
            session_dates=session_dates,
            cfg=cfg,
            xgb_params=xgb_params,
            embargo_end_idx=embargo_end_idx,
        )
        y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        sweep = sweep_thresholds(y_true=y[result.valid_mask], probs=result.oof_probs[result.valid_mask], cfg=cfg)
        threshold, best_row = choose_threshold(sweep, objective=cfg.threshold_objective)
        train_metrics = binary_metrics(y[result.valid_mask], result.full_probs[result.valid_mask], threshold=threshold)
        oof_metrics = binary_metrics(y[result.valid_mask], result.oof_probs[result.valid_mask], threshold=threshold)
        key = summary_key[target_col]
        metrics_summary[key] = {"full": train_metrics, "oof": oof_metrics}
        threshold_summary[key] = {"threshold": float(threshold), **best_row}

        side_dir = entry_root / ("long" if target_col.endswith("long") else "short")
        save_booster_artifacts(
            out_dir=side_dir,
            result=result,
            feature_cols=feature_cols,
            oof_name=f"{prob_prefix}_oof",
            full_name=f"{prob_prefix}_full",
        )
        sweep.to_csv(side_dir / "threshold_sweep.csv", index=False)
        combined_cols[f"{prob_prefix}_oof"] = result.oof_probs
        combined_cols[f"{prob_prefix}_full"] = result.full_probs

    prob_path = save_prob_frame(entry_root / "entry_probs.parquet", index=frame.index, columns=combined_cols)
    thresholds_path = entry_root / "entry_thresholds.json"
    thresholds_path.write_text(json.dumps(threshold_summary, indent=2), encoding="utf-8")
    labels_path = save_prob_frame(
        entry_root / "entry_labels.parquet",
        index=frame.index,
        columns={
            "y_enter_long": frame["y_enter_long"].to_numpy(dtype=np.int8),
            "y_enter_short": frame["y_enter_short"].to_numpy(dtype=np.int8),
        },
    )

    log_training_run(
        run_name="meta_xgboost_entry_train",
        output_dir=entry_root,
        hyperparameters={**asdict(cfg), "feature_count": len(feature_cols), "feature_columns": feature_cols},
        train_metrics={k: v["full"] for k, v in metrics_summary.items()},
        validation_metrics={k: v["oof"] for k, v in metrics_summary.items()},
        best_validation_metrics=threshold_summary,
        artifacts={
            "entry_root": str(entry_root),
            "entry_probs": str(prob_path),
            "entry_labels": str(labels_path),
            "thresholds": str(thresholds_path),
        },
        extra={
            "rows": int(len(frame)),
            "ticker": cfg.ticker,
            "dataset_name": cfg.dataset_name,
            "targets": list(summary_key.values()),
            "boundary_embargo": "exclude training rows whose label resolution crosses an eval fold start",
        },
    )
    return {
        "entry_root": entry_root,
        "entry_probs_path": prob_path,
        "entry_thresholds_path": thresholds_path,
        "entry_labels_path": labels_path,
        "threshold_summary": threshold_summary,
    }


def main() -> None:
    cfg = build_config(parse_args())
    artifacts = run_entry_pipeline(cfg)
    print(f"[META-ENTRY] Saved probabilities: {artifacts['entry_probs_path']}")
    print(f"[META-ENTRY] Saved thresholds: {artifacts['entry_thresholds_path']}")


if __name__ == "__main__":
    main()
