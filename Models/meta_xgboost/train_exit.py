from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from Data.plots.plots import plot_model_inference
from Data.retrieve_data import normalize_ticker
from Policy.training_logging import log_training_run
from Models.meta_xgboost.common import (
    EXIT_TARGETS,
    PipelineConfig,
    add_position_context_features,
    binary_metrics,
    build_base_feature_frame,
    choose_threshold,
    compute_exit_embargo_end_idx,
    load_prob_frame,
    resolve_meta_dataset_root,
    save_booster_artifacts,
    save_prob_frame,
    save_train_val_loss_plot,
    select_numeric_feature_columns,
    sweep_thresholds,
    train_walkforward_binary,
    xgb_params_from_config,
)
from Features.label_generations import build_meta_exit_labels


ENTRY_THRESHOLDS_FILENAME = "entry_thresholds.json"
ENTRY_PROBS_FILENAME = "entry_probs.parquet"
EXIT_SIDES = ("long", "short")


def _save_exit_eval_plot(
    *,
    frame: pd.DataFrame,
    exit_root: Path,
    combined_cols: dict[str, np.ndarray],
    threshold_summary: dict[str, dict[str, float | None]],
    cfg: PipelineConfig,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    long_probs = None
    short_probs = None
    if "p_exit_long_oof" in combined_cols:
        long_probs = np.asarray(combined_cols["p_exit_long_oof"], dtype=np.float32).reshape(-1)
    if "p_exit_short_oof" in combined_cols:
        short_probs = np.asarray(combined_cols["p_exit_short_oof"], dtype=np.float32).reshape(-1)

    if long_probs is not None and long_probs.size == len(frame) and "exit_long" in threshold_summary:
        long_valid = np.isfinite(long_probs)
        if np.any(long_valid):
            tail_idx = np.flatnonzero(long_valid)[-300:]
            plot_df = frame.iloc[tail_idx]
            long_save_path = exit_root / "meta_exit_oof_eval_long.png"
            plot_model_inference(
                plot_df,
                long_probs[tail_idx],
                None,
                long_entry_actual=plot_df["enter_long_trigger_oof"].to_numpy(dtype=np.int64),
                long_actual=plot_df["y_exit_long"].to_numpy(dtype=np.int64),
                long_entry_label_name="ENTRY LONG",
                long_label_name="EXIT LONG",
                threshold=float("nan"),
                long_threshold=float(threshold_summary["exit_long"]["threshold"]),
                title=f"{normalize_ticker(cfg.ticker)} | Meta-XGB long entry/exit OOF eval ({cfg.dataset_name})",
                save_path=str(long_save_path),
            )
            outputs["long"] = long_save_path

    if short_probs is not None and short_probs.size == len(frame) and "exit_short" in threshold_summary:
        short_valid = np.isfinite(short_probs)
        if np.any(short_valid):
            tail_idx = np.flatnonzero(short_valid)[-300:]
            plot_df = frame.iloc[tail_idx]
            short_save_path = exit_root / "meta_exit_oof_eval_short.png"
            plot_model_inference(
                plot_df,
                None,
                short_probs[tail_idx],
                short_entry_actual=plot_df["enter_short_trigger_oof"].to_numpy(dtype=np.int64),
                short_actual=plot_df["y_exit_short"].to_numpy(dtype=np.int64),
                short_entry_label_name="ENTRY SHORT",
                short_label_name="EXIT SHORT",
                threshold=float("nan"),
                short_threshold=float(threshold_summary["exit_short"]["threshold"]),
                title=f"{normalize_ticker(cfg.ticker)} | Meta-XGB short entry/exit OOF eval ({cfg.dataset_name})",
                save_path=str(short_save_path),
            )
            outputs["short"] = short_save_path
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train meta EXIT XGBoost models from OOF-derived entry triggers.")
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
    p.add_argument("--hazard-k", type=int, default=1)
    p.add_argument("--n-folds", type=int, default=PipelineConfig.n_folds)
    p.add_argument("--initial-train-days", type=int, default=PipelineConfig.initial_train_days)
    p.add_argument("--purge-days", type=int, default=PipelineConfig.purge_days)
    p.add_argument("--threshold-min", type=float, default=PipelineConfig.threshold_min)
    p.add_argument("--threshold-max", type=float, default=PipelineConfig.threshold_max)
    p.add_argument("--threshold-step", type=float, default=PipelineConfig.threshold_step)
    p.add_argument("--threshold-objective", type=str, default=PipelineConfig.threshold_objective)
    p.add_argument("--min-oos-prob-coverage", type=float, default=PipelineConfig.min_oos_prob_coverage)
    p.add_argument("--sides", choices=["both", "long", "short"], default="both")
    p.add_argument("--entry-root", type=str, default=None)
    p.add_argument("--entry-long-threshold", type=float, default=None)
    p.add_argument("--entry-short-threshold", type=float, default=None)
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
        include_vix_features=bool(args.include_vix_features),
        a_tp=float(args.a_tp),
        b_sl=float(args.b_sl),
        cost_bps=float(args.cost_bps),
        use_next_open=bool(args.use_next_open),
        allow_cross_day=bool(args.allow_cross_day),
        hazard_k=int(args.hazard_k),
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
        early_stopping_rounds=args.early_stopping_rounds,
        early_stopping_val_fraction=args.early_stopping_val_fraction,
        early_stopping_min_val_rows=args.early_stopping_min_val_rows,
        random_state=int(args.random_state),
    )


def _entry_root(cfg: PipelineConfig, cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    return resolve_meta_dataset_root(cfg) / "entry"


def _resolve_sides(raw: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, tuple):
        sides = tuple(s.lower() for s in raw)
    else:
        key = str(raw).strip().lower()
        sides = EXIT_SIDES if key == "both" else (key,)
    invalid = [s for s in sides if s not in EXIT_SIDES]
    if invalid:
        raise ValueError(f"Unsupported side(s): {invalid}. Expected one of {EXIT_SIDES}.")
    return tuple(dict.fromkeys(sides))


def _load_entry_thresholds(
    *,
    cfg: PipelineConfig,
    entry_root: Path,
    cli_long: float | None,
    cli_short: float | None,
) -> tuple[float, float, Path]:
    path = entry_root / ENTRY_THRESHOLDS_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    long_key = "enter_long" if "enter_long" in payload else "y_enter_long"
    short_key = "enter_short" if "enter_short" in payload else "y_enter_short"
    long_payload = payload.get(long_key, {}) if isinstance(payload, dict) else {}
    short_payload = payload.get(short_key, {}) if isinstance(payload, dict) else {}
    long_thr = float(cli_long) if cli_long is not None else float(long_payload.get("threshold", 1.0))
    short_thr = float(cli_short) if cli_short is not None else float(short_payload.get("threshold", 1.0))
    return long_thr, short_thr, path


def run_exit_pipeline(
    cfg: PipelineConfig,
    *,
    entry_root: Path,
    entry_long_threshold: float | None,
    entry_short_threshold: float | None,
    fixed_long_threshold: float | None,
    fixed_short_threshold: float | None,
    trail_activate_atr: float,
    trail_atr: float,
    trail_atr_after_tp: float,
    use_tp_to_tighten_trail: bool,
    sides: tuple[str, ...] = EXIT_SIDES,
) -> dict[str, Path | dict[str, float] | pd.DataFrame]:
    active_sides = _resolve_sides(sides)
    frame = build_base_feature_frame(cfg)
    entry_prob_df = load_prob_frame(entry_root / ENTRY_PROBS_FILENAME).reindex(frame.index)
    long_thr, short_thr, thresholds_path = _load_entry_thresholds(
        cfg=cfg,
        entry_root=entry_root,
        cli_long=entry_long_threshold,
        cli_short=entry_short_threshold,
    )
    frame["p_enter_long_oof"] = pd.to_numeric(
        entry_prob_df["p_enter_long_oof"] if "p_enter_long_oof" in entry_prob_df.columns else np.nan,
        errors="coerce",
    )
    frame["p_enter_short_oof"] = pd.to_numeric(
        entry_prob_df["p_enter_short_oof"] if "p_enter_short_oof" in entry_prob_df.columns else np.nan,
        errors="coerce",
    )
    frame["enter_long_trigger_oof"] = (frame["p_enter_long_oof"] >= float(long_thr)).fillna(False).astype(np.int8)
    frame["enter_short_trigger_oof"] = (frame["p_enter_short_oof"] >= float(short_thr)).fillna(False).astype(np.int8)

    label_cols = [c for c in {
        "open", "high", "low", "close", cfg.atr_col, "session_date",
        "enter_long_trigger_oof", "enter_short_trigger_oof",
    } if c in frame.columns]
    labels = build_meta_exit_labels(
        frame[label_cols].copy(),
        atr_col=cfg.atr_col,
        enter_long_col="enter_long_trigger_oof",
        enter_short_col="enter_short_trigger_oof",
        a_tp=cfg.a_tp,
        b_sl=cfg.b_sl,
        use_next_open=cfg.use_next_open,
        cost_bps=cfg.cost_bps,
        K=cfg.hazard_k,
        day_col="session_date",
        allow_cross_day=cfg.allow_cross_day,
        trail_activate_atr=float(trail_activate_atr),
        trail_atr=float(trail_atr),
        trail_atr_after_tp=float(trail_atr_after_tp),
        use_tp_to_tighten_trail=bool(use_tp_to_tighten_trail),
        overwrite_entry_with_used=True,
    )

    for col in (
        "y_exit_long", "y_exit_short", "y_exit_long_point", "y_exit_short_point",
        "exit_reason_long", "exit_reason_short",
        "tp_hit_before_exit_long", "tp_hit_before_exit_short",
        "enter_long_trigger_oof", "enter_short_trigger_oof",
    ):
        frame[col] = labels[col]

    frame = add_position_context_features(
        frame,
        side="long",
        enter_col="enter_long_trigger_oof",
        point_exit_col="y_exit_long_point",
        atr_col=cfg.atr_col,
        use_next_open=cfg.use_next_open,
        a_tp=cfg.a_tp,
        trail_activate_atr=float(trail_activate_atr),
        trail_atr=float(trail_atr),
        trail_atr_after_tp=float(trail_atr_after_tp),
        use_tp_to_tighten_trail=bool(use_tp_to_tighten_trail),
    )
    frame = add_position_context_features(
        frame,
        side="short",
        enter_col="enter_short_trigger_oof",
        point_exit_col="y_exit_short_point",
        atr_col=cfg.atr_col,
        use_next_open=cfg.use_next_open,
        a_tp=cfg.a_tp,
        trail_activate_atr=float(trail_activate_atr),
        trail_atr=float(trail_atr),
        trail_atr_after_tp=float(trail_atr_after_tp),
        use_tp_to_tighten_trail=bool(use_tp_to_tighten_trail),
    )

    xgb_params = xgb_params_from_config(cfg)
    session_dates = frame["session_date"]

    exit_root = resolve_meta_dataset_root(cfg) / "exit"
    exit_root.mkdir(parents=True, exist_ok=True)
    print(f"[META-EXIT] Output dir: {exit_root}")

    target_setup = {
        "y_exit_long": (
            "p_exit_long",
            fixed_long_threshold,
            frame["in_long_trade"].fillna(0).astype(int).to_numpy() == 1,
            compute_exit_embargo_end_idx(
                frame,
                side="long",
                enter_col="enter_long_trigger_oof",
                point_exit_col="y_exit_long_point",
                use_next_open=cfg.use_next_open,
            ),
            {
                "open", "high", "low", "close", "session_date", cfg.atr_col,
                *EXIT_TARGETS,
                "y_exit_long_point", "y_exit_short_point",
                "exit_reason_long", "exit_reason_short",
                "tp_hit_before_exit_long", "tp_hit_before_exit_short",
                "enter_long_trigger_oof", "enter_short_trigger_oof",
                "in_long_trade", "in_short_trade",
                *(c for c in frame.columns if c.startswith("short_")),
            },
        ),
        "y_exit_short": (
            "p_exit_short",
            fixed_short_threshold,
            frame["in_short_trade"].fillna(0).astype(int).to_numpy() == 1,
            compute_exit_embargo_end_idx(
                frame,
                side="short",
                enter_col="enter_short_trigger_oof",
                point_exit_col="y_exit_short_point",
                use_next_open=cfg.use_next_open,
            ),
            {
                "open", "high", "low", "close", "session_date", cfg.atr_col,
                *EXIT_TARGETS,
                "y_exit_long_point", "y_exit_short_point",
                "exit_reason_long", "exit_reason_short",
                "tp_hit_before_exit_long", "tp_hit_before_exit_short",
                "enter_long_trigger_oof", "enter_short_trigger_oof",
                "in_long_trade", "in_short_trade",
                *(c for c in frame.columns if c.startswith("long_")),
            },
        ),
    }
    selected_target_cols = []
    if "long" in active_sides:
        selected_target_cols.append("y_exit_long")
    if "short" in active_sides:
        selected_target_cols.append("y_exit_short")
    if not selected_target_cols:
        raise ValueError("No exit targets selected.")

    summary_key = {"y_exit_long": "exit_long", "y_exit_short": "exit_short"}
    combined_cols: dict[str, np.ndarray] = {}
    threshold_summary: dict[str, dict[str, float | None]] = {}
    metrics_summary: dict[str, dict[str, dict[str, float]]] = {}
    loss_histories: dict[str, dict[str, list[float]] | None] = {}
    feature_columns_by_target: dict[str, list[str]] = {}
    trained_targets: list[str] = []

    for target_col in selected_target_cols:
        prob_prefix, fixed_threshold, active_mask, embargo_end_idx, exclude_cols = target_setup[target_col]
        key = summary_key[target_col]
        print(f"[META-EXIT] Training {key}")
        feature_cols = select_numeric_feature_columns(frame, exclude=exclude_cols)
        feature_columns_by_target[key] = list(feature_cols)
        result = train_walkforward_binary(
            df=frame,
            feature_cols=feature_cols,
            target_col=target_col,
            session_dates=session_dates,
            cfg=cfg,
            xgb_params=xgb_params,
            condition_mask=active_mask,
            embargo_end_idx=embargo_end_idx,
        )
        y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        sweep = sweep_thresholds(y_true=y[result.valid_mask], probs=result.oof_probs[result.valid_mask], cfg=cfg)
        objective_key = str(cfg.threshold_objective).strip().lower().replace("-", "_").replace(".", "_")
        best_threshold, best_row = choose_threshold(sweep, objective=cfg.threshold_objective)
        if fixed_threshold is not None:
            threshold = float(fixed_threshold)
            threshold_row = sweep.iloc[(sweep["threshold"] - threshold).abs().argmin()].to_dict()
            threshold_source = "fixed_threshold"
            selection_objective = "fixed"
        else:
            threshold = float(best_threshold)
            threshold_row = dict(best_row)
            threshold_source = f"best_{objective_key}_sweep"
            selection_objective = objective_key
        train_metrics = binary_metrics(y[result.valid_mask], result.full_probs[result.valid_mask], threshold=threshold)
        oof_metrics = binary_metrics(y[result.valid_mask], result.oof_probs[result.valid_mask], threshold=threshold)
        metrics_summary[key] = {"full": train_metrics, "oof": oof_metrics}
        loss_histories[key] = result.eval_history
        trained_targets.append(key)
        threshold_summary[key] = {
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "selection_objective": selection_objective,
            "best_sweep_threshold": float(best_threshold),
            f"best_{objective_key}_sweep_threshold": float(best_threshold),
            **threshold_row,
        }
        print(
            f"[META-EXIT] {key}: threshold={float(threshold):.4f} "
            f"train_logloss={train_metrics.get('logloss', float('nan')):.4f} "
            f"train_auc={train_metrics.get('auc', float('nan')):.4f} "
            f"train_f1={train_metrics.get('f1', float('nan')):.4f}"
        )
        print(
            f"[META-EXIT] {key}: oof_logloss={oof_metrics.get('logloss', float('nan')):.4f} "
            f"oof_auc={oof_metrics.get('auc', float('nan')):.4f} "
            f"oof_f1={oof_metrics.get('f1', float('nan')):.4f} "
            f"oof_ap={oof_metrics.get('average_precision', float('nan')):.4f}"
        )

        side_dir = exit_root / ("long" if target_col.endswith("long") else "short")
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

    prob_path = exit_root / "exit_probs.parquet"
    merged_prob_cols = dict(combined_cols)
    if prob_path.exists():
        existing_probs = load_prob_frame(prob_path).reindex(frame.index)
        for col in existing_probs.columns:
            if col not in merged_prob_cols:
                merged_prob_cols[col] = pd.to_numeric(existing_probs[col], errors="coerce").to_numpy(dtype=float)
    prob_path = save_prob_frame(prob_path, index=frame.index, columns=merged_prob_cols)

    thresholds_out = exit_root / "exit_thresholds.json"
    merged_thresholds: dict[str, dict[str, float | None]] = {}
    if thresholds_out.exists():
        loaded = json.loads(thresholds_out.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            merged_thresholds.update(loaded)
    merged_thresholds.update(threshold_summary)
    thresholds_out.write_text(json.dumps(merged_thresholds, indent=2), encoding="utf-8")

    label_path = save_prob_frame(
        exit_root / "exit_labels.parquet",
        index=frame.index,
        columns={
            "enter_long_used_oof": frame["enter_long_trigger_oof"].to_numpy(dtype=np.int8),
            "enter_short_used_oof": frame["enter_short_trigger_oof"].to_numpy(dtype=np.int8),
            "in_long_trade": frame["in_long_trade"].to_numpy(dtype=np.int8),
            "in_short_trade": frame["in_short_trade"].to_numpy(dtype=np.int8),
            "y_exit_long": frame["y_exit_long"].to_numpy(dtype=np.int8),
            "y_exit_short": frame["y_exit_short"].to_numpy(dtype=np.int8),
            "y_exit_long_point": frame["y_exit_long_point"].to_numpy(dtype=np.int8),
            "y_exit_short_point": frame["y_exit_short_point"].to_numpy(dtype=np.int8),
            "tp_hit_before_exit_long": frame["tp_hit_before_exit_long"].fillna(0).astype(np.int8).to_numpy(),
            "tp_hit_before_exit_short": frame["tp_hit_before_exit_short"].fillna(0).astype(np.int8).to_numpy(),
            "exit_reason_long": frame["exit_reason_long"].astype("string").fillna("NONE").to_numpy(),
            "exit_reason_short": frame["exit_reason_short"].astype("string").fillna("NONE").to_numpy(),
        },
    )
    context_cols = {
        c: pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        for c in frame.columns
        if c.startswith("long_") or c.startswith("short_")
    }
    context_cols["p_enter_long_oof"] = pd.to_numeric(frame["p_enter_long_oof"], errors="coerce").to_numpy(dtype=float)
    context_cols["p_enter_short_oof"] = pd.to_numeric(frame["p_enter_short_oof"], errors="coerce").to_numpy(dtype=float)
    context_path = save_prob_frame(exit_root / "exit_context.parquet", index=frame.index, columns=context_cols)
    plot_paths = _save_exit_eval_plot(
        frame=frame,
        exit_root=exit_root,
        combined_cols=combined_cols,
        threshold_summary=threshold_summary,
        cfg=cfg,
    )
    if plot_paths:
        for side, path in plot_paths.items():
            print(f"[META-EXIT] Saved {side} OOF eval plot: {path}")
    loss_plot_path = save_train_val_loss_plot(
        histories=loss_histories,
        save_path=exit_root / "meta_exit_train_val_loss.png",
        title=f"{normalize_ticker(cfg.ticker)} | Meta-XGB exit train vs validation logloss ({cfg.dataset_name})",
    )
    if loss_plot_path is not None:
        print(f"[META-EXIT] Saved train/val loss plot: {loss_plot_path}")

    summary_paths = log_training_run(
        run_name="meta_xgboost_exit_train",
        output_dir=exit_root,
        hyperparameters={
            **asdict(cfg),
            "feature_count": max((len(v) for v in feature_columns_by_target.values()), default=0),
            "feature_columns": feature_columns_by_target.get("exit_short") or feature_columns_by_target.get("exit_long") or [],
            "feature_count_by_target": {k: len(v) for k, v in feature_columns_by_target.items()},
            "feature_columns_by_target": feature_columns_by_target,
            "entry_root": str(entry_root),
            "entry_long_threshold": float(long_thr),
            "entry_short_threshold": float(short_thr),
            "fixed_long_threshold": float(fixed_long_threshold) if fixed_long_threshold is not None else None,
            "fixed_short_threshold": float(fixed_short_threshold) if fixed_short_threshold is not None else None,
            "trail_activate_atr": float(trail_activate_atr),
            "trail_atr": float(trail_atr),
            "trail_atr_after_tp": float(trail_atr_after_tp),
            "use_tp_to_tighten_trail": bool(use_tp_to_tighten_trail),
        },
        train_metrics={k: v["full"] for k, v in metrics_summary.items()},
        validation_metrics={k: v["oof"] for k, v in metrics_summary.items()},
        best_validation_metrics=threshold_summary,
        artifacts={
            "entry_thresholds_used": str(thresholds_path),
            "exit_root": str(exit_root),
            "exit_probs": str(prob_path),
            "exit_labels": str(label_path),
            "exit_context": str(context_path),
            "thresholds": str(thresholds_out),
            "oof_eval_plots": {k: str(v) for k, v in plot_paths.items()},
            "train_val_loss_plot": str(loss_plot_path) if loss_plot_path is not None else None,
        },
        extra={
            "rows": int(len(frame)),
            "ticker": cfg.ticker,
            "dataset_name": cfg.dataset_name,
            "targets": trained_targets,
            "exit_reason_counts_long": frame["exit_reason_long"].value_counts(dropna=False).to_dict(),
            "exit_reason_counts_short": frame["exit_reason_short"].value_counts(dropna=False).to_dict(),
            "boundary_embargo": "exclude active training rows whose simulated exit lands in or after an eval fold start",
        },
    )
    print(f"[META-EXIT] Saved summary: {summary_paths['versioned_path']}")
    return {
        "exit_root": exit_root,
        "exit_probs_path": prob_path,
        "exit_thresholds_path": thresholds_out,
        "exit_labels_path": label_path,
        "exit_context_path": context_path,
        "training_summary_latest_path": summary_paths["latest_path"],
        "training_summary_versioned_path": summary_paths["versioned_path"],
        "threshold_summary": threshold_summary,
    }


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    sides = EXIT_SIDES if args.sides == "both" else (str(args.sides),)
    artifacts = run_exit_pipeline(
        cfg,
        entry_root=_entry_root(cfg, args.entry_root),
        entry_long_threshold=args.entry_long_threshold,
        entry_short_threshold=args.entry_short_threshold,
        fixed_long_threshold=args.fixed_long_threshold,
        fixed_short_threshold=args.fixed_short_threshold,
        trail_activate_atr=float(args.trail_activate_atr),
        trail_atr=float(args.trail_atr),
        trail_atr_after_tp=float(args.trail_atr_after_tp),
        use_tp_to_tighten_trail=bool(args.use_tp_to_tighten_trail),
        sides=sides,
    )
    print(f"[META-EXIT] Saved probabilities: {artifacts['exit_probs_path']}")
    print(f"[META-EXIT] Saved thresholds: {artifacts['exit_thresholds_path']}")
    print(f"[META-EXIT] Saved versioned summary: {artifacts['training_summary_versioned_path']}")


if __name__ == "__main__":
    main()
