from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from Data.plots.plots import plot_model_inference
from Data.retrieve_data import normalize_ticker
from Policy.training_logging import log_training_run
from Models.meta_xgboost.common import (
    ENTRY_TARGETS,
    PipelineConfig,
    add_entry_targets,
    binary_metrics,
    build_base_feature_frame,
    choose_threshold,
    compute_entry_embargo_end_idx,
    load_prob_frame,
    resolve_meta_dataset_root,
    save_booster_artifacts,
    save_prob_frame,
    save_train_val_loss_plot,
    select_numeric_feature_columns,
    sweep_thresholds,
    train_full_fit_binary,
    train_walkforward_binary,
    xgb_params_from_config,
)

ENTRY_SIDES = ("long", "short")


def _build_entry_candidate_mask(
    frame: pd.DataFrame,
    *,
    side: str,
    cfg: PipelineConfig,
) -> np.ndarray | None:
    thr = cfg.entry_candidate_min_pivot_prob
    if thr is None:
        return None
    threshold = float(thr)
    if threshold <= 0.0:
        return None
    side_key = str(side).strip().lower()
    if side_key not in ENTRY_SIDES:
        raise ValueError(f"Unsupported side: {side}")
    prob_col = f"p_pivot_{side_key}"
    if prob_col not in frame.columns:
        raise KeyError(f"Missing candidate probability column: {prob_col}")
    lookback = max(0, int(cfg.entry_candidate_lookback_bars))
    prob_s = pd.to_numeric(frame[prob_col], errors="coerce")
    if lookback > 0:
        prob_s = prob_s.rolling(window=lookback + 1, min_periods=1).max()
    mask = prob_s.ge(threshold).fillna(False).to_numpy(dtype=bool)
    return mask


def _save_entry_eval_plot(
    *,
    frame: pd.DataFrame,
    entry_root: Path,
    combined_cols: dict[str, np.ndarray],
    threshold_summary: dict[str, dict[str, float | None]],
    cfg: PipelineConfig,
) -> Path | None:
    long_probs = None
    short_probs = None
    if "p_enter_long_oof" in combined_cols:
        long_probs = np.asarray(combined_cols["p_enter_long_oof"], dtype=np.float32).reshape(-1)
    if "p_enter_short_oof" in combined_cols:
        short_probs = np.asarray(combined_cols["p_enter_short_oof"], dtype=np.float32).reshape(-1)

    valid = np.zeros(len(frame), dtype=bool)
    if long_probs is not None and long_probs.size == len(frame):
        valid |= np.isfinite(long_probs)
    if short_probs is not None and short_probs.size == len(frame):
        valid |= np.isfinite(short_probs)
    if not np.any(valid):
        return None

    valid_idx = np.flatnonzero(valid)
    tail_idx = valid_idx[-300:] if valid_idx.size > 300 else valid_idx
    plot_df = frame.iloc[tail_idx]
    long_thr = threshold_summary.get("enter_long", {}).get("threshold")
    short_thr = threshold_summary.get("enter_short", {}).get("threshold")
    save_path = entry_root / "meta_entry_oof_eval.png"
    plot_model_inference(
        plot_df,
        long_probs[tail_idx] if long_probs is not None else None,
        short_probs[tail_idx] if short_probs is not None else None,
        long_actual=plot_df["y_enter_long"].to_numpy(dtype=np.int64) if long_probs is not None else None,
        short_actual=plot_df["y_enter_short"].to_numpy(dtype=np.int64) if short_probs is not None else None,
        long_label_name="ENTRY LONG",
        short_label_name="ENTRY SHORT",
        threshold=0.5,
        long_threshold=float(long_thr) if long_thr is not None else None,
        short_threshold=float(short_thr) if short_thr is not None else None,
        title=f"{normalize_ticker(cfg.ticker)} | Meta-XGB entry OOF eval ({cfg.dataset_name})",
        save_path=str(save_path),
    )
    if save_path.exists():
        return save_path
    print(f"[META-ENTRY] Warning: expected plot file was not found at {save_path}")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train meta ENTRY XGBoost models with walk-forward OOF.")
    p.add_argument("--ticker", type=str, default=PipelineConfig.ticker)
    p.add_argument("--dataset-name", type=str, default=PipelineConfig.dataset_name)
    p.add_argument("--processed-root", type=str, default=None)
    p.add_argument("--ga-model-root", type=str, default=None)
    p.add_argument("--model-root", type=str, default=None)
    p.add_argument("--pivot-label-dir", type=str, default=PipelineConfig.pivot_label_dir)
    p.add_argument("--tb-label-dir", type=str, default=PipelineConfig.tb_label_dir)
    p.add_argument("--include-tb-probs", action=argparse.BooleanOptionalAction, default=PipelineConfig.include_tb_probs)
    p.add_argument("--include-vix-features", action=argparse.BooleanOptionalAction, default=PipelineConfig.include_vix_features)
    p.add_argument("--a-tp", type=float, default=PipelineConfig.a_tp)
    p.add_argument("--b-sl", type=float, default=PipelineConfig.b_sl)
    p.add_argument("--entry-max-holding-bars", type=int, default=PipelineConfig.entry_max_holding_bars)
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
    p.add_argument("--drop-high-corr-features", action=argparse.BooleanOptionalAction, default=PipelineConfig.drop_high_corr_features)
    p.add_argument("--high-corr-threshold", type=float, default=PipelineConfig.high_corr_threshold)
    p.add_argument("--full-fit-only", action=argparse.BooleanOptionalAction, default=PipelineConfig.full_fit_only)
    p.add_argument("--entry-candidate-min-pivot-prob", type=float, default=PipelineConfig.entry_candidate_min_pivot_prob)
    p.add_argument("--entry-candidate-lookback-bars", type=int, default=PipelineConfig.entry_candidate_lookback_bars)
    p.add_argument("--sides", choices=["both", "long", "short"], default="both")
    p.add_argument("--plot-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--xgb-booster", choices=["gbtree", "dart"], default=None)
    p.add_argument("--xgb-rate-drop", type=float, default=None)
    p.add_argument("--xgb-skip-drop", type=float, default=None)
    p.add_argument("--xgb-one-drop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--xgb-sample-type", choices=["uniform", "weighted"], default=None)
    p.add_argument("--xgb-normalize-type", choices=["tree", "forest"], default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--early-stopping-rounds", type=int, default=PipelineConfig.early_stopping_rounds)
    p.add_argument("--early-stopping-val-fraction", type=float, default=PipelineConfig.early_stopping_val_fraction)
    p.add_argument("--early-stopping-min-val-rows", type=int, default=PipelineConfig.early_stopping_min_val_rows)
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
        include_tb_probs=bool(args.include_tb_probs),
        include_vix_features=bool(args.include_vix_features),
        a_tp=float(args.a_tp),
        b_sl=float(args.b_sl),
        entry_max_holding_bars=int(args.entry_max_holding_bars),
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
        drop_high_corr_features=bool(args.drop_high_corr_features),
        high_corr_threshold=float(args.high_corr_threshold),
        full_fit_only=bool(args.full_fit_only),
        entry_candidate_min_pivot_prob=args.entry_candidate_min_pivot_prob,
        entry_candidate_lookback_bars=int(args.entry_candidate_lookback_bars),
        xgb_booster=args.xgb_booster,
        xgb_rate_drop=args.xgb_rate_drop,
        xgb_skip_drop=args.xgb_skip_drop,
        xgb_one_drop=args.xgb_one_drop,
        xgb_sample_type=args.xgb_sample_type,
        xgb_normalize_type=args.xgb_normalize_type,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        early_stopping_val_fraction=args.early_stopping_val_fraction,
        early_stopping_min_val_rows=args.early_stopping_min_val_rows,
        random_state=int(args.random_state),
    )


def _resolve_sides(raw: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, tuple):
        sides = tuple(s.lower() for s in raw)
    else:
        key = str(raw).strip().lower()
        sides = ENTRY_SIDES if key == "both" else (key,)
    invalid = [s for s in sides if s not in ENTRY_SIDES]
    if invalid:
        raise ValueError(f"Unsupported side(s): {invalid}. Expected one of {ENTRY_SIDES}.")
    return tuple(dict.fromkeys(sides))


def replot_entry_eval_from_saved_artifacts(cfg: PipelineConfig) -> Path | None:
    entry_root = resolve_meta_dataset_root(cfg) / "entry"
    prob_path = entry_root / "entry_probs.parquet"
    thresholds_path = entry_root / "entry_thresholds.json"
    labels_path = entry_root / "entry_labels.parquet"

    if not prob_path.exists():
        raise FileNotFoundError(f"Missing saved probabilities for replot: {prob_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing saved thresholds for replot: {thresholds_path}")

    probs_df = load_prob_frame(prob_path)
    combined_cols = {
        col: pd.to_numeric(probs_df[col], errors="coerce").to_numpy(dtype=float)
        for col in probs_df.columns
    }

    try:
        frame = build_base_feature_frame(cfg)
    except ValueError as exc:
        if "OOS probability coverage below threshold" not in str(exc):
            raise
        relaxed_cfg = replace(cfg, min_oos_prob_coverage=0.0)
        print(
            "[META-ENTRY] Replot fallback: lowering min_oos_prob_coverage "
            f"from {cfg.min_oos_prob_coverage:.2f} to 0.00 for plotting."
        )
        frame = build_base_feature_frame(relaxed_cfg)
    if labels_path.exists():
        labels_df = load_prob_frame(labels_path).reindex(frame.index)
        for col in ("y_enter_long", "y_enter_short"):
            if col in labels_df.columns:
                frame[col] = (
                    pd.to_numeric(labels_df[col], errors="coerce")
                    .fillna(0)
                    .astype(np.int8)
                    .to_numpy()
                )
    if "y_enter_long" not in frame.columns or "y_enter_short" not in frame.columns:
        frame = add_entry_targets(frame, cfg)

    raw_thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    threshold_summary = raw_thresholds if isinstance(raw_thresholds, dict) else {}

    plot_path = _save_entry_eval_plot(
        frame=frame,
        entry_root=entry_root,
        combined_cols=combined_cols,
        threshold_summary=threshold_summary,
        cfg=cfg,
    )
    if plot_path is not None:
        print(f"[META-ENTRY] Rebuilt OOF eval plot: {plot_path}")
    return plot_path


def run_entry_pipeline(
    cfg: PipelineConfig,
    *,
    sides: tuple[str, ...] = ENTRY_SIDES,
) -> dict[str, Path | dict[str, float] | pd.DataFrame]:
    active_sides = _resolve_sides(sides)
    print(f"[META-ENTRY] Building feature frame for {cfg.ticker} {cfg.dataset_name} | sides={active_sides}")
    frame = add_entry_targets(build_base_feature_frame(cfg), cfg)
    exclude = {
        "open", "high", "low", "close", "session_date", cfg.atr_col,
        *ENTRY_TARGETS,
        "y_exit_long", "y_exit_short", "y_exit_long_point", "y_exit_short_point",
        "exit_reason_long", "exit_reason_short", "tp_hit_before_exit_long", "tp_hit_before_exit_short",
    }
    feature_cols = select_numeric_feature_columns(
        frame,
        exclude=exclude,
        corr_threshold=float(cfg.high_corr_threshold) if bool(cfg.drop_high_corr_features) else None,
        log_prefix="[META-ENTRY]",
    )
    xgb_params = xgb_params_from_config(cfg)
    # Deploy the saved entry model as a stable full-data gbtree fit with no
    # internal tail-split early stopping. The label and feature set stay
    # unchanged; only the final inference fit differs.
    deployed_fit_cfg = replace(cfg, xgb_booster="gbtree", early_stopping_rounds=None)
    deployed_fit_xgb_params = xgb_params_from_config(deployed_fit_cfg)
    session_dates = frame["session_date"]

    entry_root = resolve_meta_dataset_root(cfg) / "entry"
    entry_root.mkdir(parents=True, exist_ok=True)
    print(f"[META-ENTRY] Output dir: {entry_root}")

    target_specs: list[tuple[str, str, str]] = []
    if "long" in active_sides:
        target_specs.append(("y_enter_long", "p_enter_long", "enter_long"))
    if "short" in active_sides:
        target_specs.append(("y_enter_short", "p_enter_short", "enter_short"))
    if not target_specs:
        raise ValueError("No entry targets selected.")

    combined_cols: dict[str, np.ndarray] = {}
    threshold_summary: dict[str, dict[str, float | None]] = {}
    metrics_summary: dict[str, dict[str, dict[str, float]]] = {}
    loss_histories: dict[str, dict[str, list[float]] | None] = {}
    feature_columns_by_target: dict[str, list[str]] = {}
    trained_targets: list[str] = []

    for target_col, prob_prefix, key in target_specs:
        print(f"[META-ENTRY] Training {key}")
        side = "long" if target_col.endswith("long") else "short"
        condition_mask = _build_entry_candidate_mask(frame, side=side, cfg=cfg)
        if condition_mask is not None:
            print(
                f"[META-ENTRY] {key}: candidate training enabled "
                f"(thr={float(cfg.entry_candidate_min_pivot_prob):.3f}, "
                f"lookback_bars={int(cfg.entry_candidate_lookback_bars)}, "
                f"coverage={float(np.mean(condition_mask)):.2%})"
            )
        if bool(cfg.full_fit_only):
            result = train_full_fit_binary(
                df=frame,
                feature_cols=feature_cols,
                target_col=target_col,
                cfg=cfg,
                xgb_params=deployed_fit_xgb_params,
                fit_early_stopping_rounds=deployed_fit_cfg.early_stopping_rounds,
                condition_mask=condition_mask,
            )
        else:
            embargo_end_idx = compute_entry_embargo_end_idx(frame, cfg, target_col=target_col)
            result = train_walkforward_binary(
                df=frame,
                feature_cols=feature_cols,
                target_col=target_col,
                session_dates=session_dates,
                cfg=cfg,
                xgb_params=xgb_params,
                full_fit_xgb_params=deployed_fit_xgb_params,
                full_fit_early_stopping_rounds=deployed_fit_cfg.early_stopping_rounds,
                condition_mask=condition_mask,
                embargo_end_idx=embargo_end_idx,
            )
        y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        if bool(cfg.full_fit_only):
            existing_thresholds: dict[str, dict[str, float | None]] = {}
            thresholds_path = entry_root / "entry_thresholds.json"
            if thresholds_path.exists():
                loaded = json.loads(thresholds_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing_thresholds = loaded
            threshold_payload = existing_thresholds.get(key, {})
            threshold = float(threshold_payload.get("threshold", 0.5))
            best_row = dict(threshold_payload)
            sweep = pd.DataFrame()
        else:
            sweep = sweep_thresholds(y_true=y[result.valid_mask], probs=result.oof_probs[result.valid_mask], cfg=cfg)
            threshold, best_row = choose_threshold(sweep, objective=cfg.threshold_objective)
        train_metrics = binary_metrics(y[result.valid_mask], result.full_probs[result.valid_mask], threshold=threshold)
        oof_metrics = (
            binary_metrics(y[result.valid_mask], result.oof_probs[result.valid_mask], threshold=threshold)
            if np.isfinite(result.oof_probs[result.valid_mask]).any()
            else {}
        )
        metrics_summary[key] = {"full": train_metrics, "oof": oof_metrics}
        threshold_summary[key] = {"threshold": float(threshold), **best_row}
        loss_histories[key] = result.eval_history
        feature_columns_by_target[key] = list(feature_cols)
        trained_targets.append(key)
        print(
            f"[META-ENTRY] {key}: threshold={float(threshold):.4f} "
            f"train_logloss={train_metrics.get('logloss', float('nan')):.4f} "
            f"train_auc={train_metrics.get('auc', float('nan')):.4f} "
            f"train_f1={train_metrics.get('f1', float('nan')):.4f}"
        )
        print(
            f"[META-ENTRY] {key}: oof_logloss={oof_metrics.get('logloss', float('nan')):.4f} "
            f"oof_auc={oof_metrics.get('auc', float('nan')):.4f} "
            f"oof_f1={oof_metrics.get('f1', float('nan')):.4f} "
            f"oof_ap={oof_metrics.get('average_precision', float('nan')):.4f}"
        )

        side_dir = entry_root / ("long" if target_col.endswith("long") else "short")
        save_booster_artifacts(
            out_dir=side_dir,
            result=result,
            feature_cols=feature_cols,
            oof_name=f"{prob_prefix}_oof",
            full_name=f"{prob_prefix}_full",
            preserve_existing_oof=bool(cfg.full_fit_only),
        )
        if not sweep.empty:
            sweep.to_csv(side_dir / "threshold_sweep.csv", index=False)
        combined_cols[f"{prob_prefix}_oof"] = result.oof_probs
        combined_cols[f"{prob_prefix}_full"] = result.full_probs

    prob_path = entry_root / "entry_probs.parquet"
    merged_prob_cols = dict(combined_cols)
    if prob_path.exists():
        existing_probs = load_prob_frame(prob_path).reindex(frame.index)
        for col in existing_probs.columns:
            if col not in merged_prob_cols or (bool(cfg.full_fit_only) and col.endswith("_oof")):
                merged_prob_cols[col] = pd.to_numeric(existing_probs[col], errors="coerce").to_numpy(dtype=float)
    prob_path = save_prob_frame(prob_path, index=frame.index, columns=merged_prob_cols)

    thresholds_path = entry_root / "entry_thresholds.json"
    merged_thresholds: dict[str, dict[str, float | None]] = {}
    if thresholds_path.exists():
        loaded = json.loads(thresholds_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            merged_thresholds.update(loaded)
    merged_thresholds.update(threshold_summary)
    thresholds_path.write_text(json.dumps(merged_thresholds, indent=2), encoding="utf-8")

    labels_path = save_prob_frame(
        entry_root / "entry_labels.parquet",
        index=frame.index,
        columns={
            "y_enter_long": frame["y_enter_long"].to_numpy(dtype=np.int8),
            "y_enter_short": frame["y_enter_short"].to_numpy(dtype=np.int8),
        },
    )
    plot_path = _save_entry_eval_plot(
        frame=frame,
        entry_root=entry_root,
        combined_cols=merged_prob_cols,
        threshold_summary=merged_thresholds,
        cfg=cfg,
    )
    if plot_path is not None:
        print(f"[META-ENTRY] Saved OOF eval plot: {plot_path}")
    loss_plot_path = save_train_val_loss_plot(
        histories=loss_histories,
        save_path=entry_root / "meta_entry_train_val_loss.png",
        title=f"{normalize_ticker(cfg.ticker)} | Meta-XGB entry train vs validation logloss ({cfg.dataset_name})",
    )
    if loss_plot_path is not None:
        print(f"[META-ENTRY] Saved train/val loss plot: {loss_plot_path}")

    summary_paths = log_training_run(
        run_name="meta_xgboost_entry_train",
        output_dir=entry_root,
        hyperparameters={
            **asdict(cfg),
            "deployed_full_fit_xgb_booster": deployed_fit_cfg.xgb_booster,
            "deployed_full_fit_early_stopping_rounds": deployed_fit_cfg.early_stopping_rounds,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "feature_count_by_target": {k: len(v) for k, v in feature_columns_by_target.items()},
            "feature_columns_by_target": feature_columns_by_target,
        },
        train_metrics={k: v["full"] for k, v in metrics_summary.items()},
        validation_metrics={k: v["oof"] for k, v in metrics_summary.items()},
        best_validation_metrics=threshold_summary,
        artifacts={
            "entry_root": str(entry_root),
            "entry_probs": str(prob_path),
            "entry_labels": str(labels_path),
            "thresholds": str(thresholds_path),
            "oof_eval_plot": str(plot_path) if plot_path is not None else None,
            "train_val_loss_plot": str(loss_plot_path) if loss_plot_path is not None else None,
        },
        extra={
            "rows": int(len(frame)),
            "ticker": cfg.ticker,
            "dataset_name": cfg.dataset_name,
            "targets": trained_targets,
            "boundary_embargo": "exclude training rows whose label resolution crosses an eval fold start",
        },
    )
    print(f"[META-ENTRY] Saved summary: {summary_paths['versioned_path']}")
    return {
        "entry_root": entry_root,
        "entry_probs_path": prob_path,
        "entry_thresholds_path": thresholds_path,
        "entry_labels_path": labels_path,
        "training_summary_latest_path": summary_paths["latest_path"],
        "training_summary_versioned_path": summary_paths["versioned_path"],
        "threshold_summary": threshold_summary,
    }


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    if bool(args.plot_only):
        replot_entry_eval_from_saved_artifacts(cfg)
        return
    sides = ENTRY_SIDES if args.sides == "both" else (str(args.sides),)
    artifacts = run_entry_pipeline(cfg, sides=sides)
    print(f"[META-ENTRY] Saved probabilities: {artifacts['entry_probs_path']}")
    print(f"[META-ENTRY] Saved thresholds: {artifacts['entry_thresholds_path']}")
    print(f"[META-ENTRY] Saved versioned summary: {artifacts['training_summary_versioned_path']}")


if __name__ == "__main__":
    main()
