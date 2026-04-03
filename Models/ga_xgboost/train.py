from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from Data.load_data import (
    get_ticker_processed_base_dir,
    get_ticker_processed_split_dir,
    get_ticker_processed_stats_dir,
)
from Data.retrieve_data import normalize_ticker
from Data.plots.plots import get_default_model_inference_plot_path, plot_model_inference
from Features.feature_scaling import apply_scaler_from_stats
from Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector
from Policy.training_logging import log_training_run


@dataclass(frozen=True)
class TrainConfig:
    ticker: str = "$SPY"
    dataset_name: str = "15min"
    x_filename: str = "X_15min_tree.parquet"
    label_mode: str = "swing"
    model_dirname: str = "ga_xgboost"
    n_folds: int = 5
    initial_train_size: int | None = None
    apply_scaler: bool = False
    update_scale_pos_weight: bool = True
    output_dirname: str = "probs"
    refresh_masks: bool = False
    no_ga_mask: bool = False
    super_pivot_weight: float = 2.0
    processed_root: str | None = None
    split_root: str | None = None
    stats_root: str | None = None
    model_root: str | None = None
    ga_population_size: int | None = None
    ga_generations: int | None = None
    ga_crossover_rate: float | None = None
    ga_mutation_rate: float | None = None
    ga_val_size: float | None = None
    ga_fitness_metric: str | None = None
    ga_feature_penalty: float | None = None
    ga_early_stopping_rounds: int | None = None
    ga_max_boost_round: int | None = None
    ga_max_features: int | None = None
    ga_selection: str | None = None
    ga_tournament_k: int | None = None
    ga_random_state: int | None = None
    ga_eval_workers: int | None = None
    ga_allow_parallel_gpu_eval: bool | None = None
    xgb_booster: str | None = None
    xgb_rate_drop: float | None = None
    xgb_skip_drop: float | None = None
    xgb_one_drop: bool | None = None
    xgb_sample_type: str | None = None
    xgb_normalize_type: str | None = None


_GA_PARAM_FIELDS: tuple[str, ...] = (
    "population_size",
    "generations",
    "crossover_rate",
    "mutation_rate",
    "val_size",
    "fitness_metric",
    "feature_penalty",
    "early_stopping_rounds",
    "max_boost_round",
    "max_features",
    "selection",
    "tournament_k",
    "random_state",
    "eval_workers",
    "allow_parallel_gpu_eval",
)

DART_DEFAULTS: dict[str, object] = {
    "rate_drop": 0.1,
    "skip_drop": 0.5,
    "one_drop": 1,
    "sample_type": "uniform",
    "normalize_type": "tree",
}


def _extract_ga_params(selector: GAXGBoostFeatureSelector) -> dict:
    return {name: getattr(selector, name, None) for name in _GA_PARAM_FIELDS}


def _ga_kwargs_from_config(cfg: TrainConfig) -> dict:
    kwargs: dict = {}
    if cfg.ga_population_size is not None:
        kwargs["population_size"] = int(cfg.ga_population_size)
    if cfg.ga_generations is not None:
        kwargs["generations"] = int(cfg.ga_generations)
    if cfg.ga_crossover_rate is not None:
        kwargs["crossover_rate"] = float(cfg.ga_crossover_rate)
    if cfg.ga_mutation_rate is not None:
        kwargs["mutation_rate"] = float(cfg.ga_mutation_rate)
    if cfg.ga_val_size is not None:
        kwargs["val_size"] = float(cfg.ga_val_size)
    if cfg.ga_fitness_metric is not None:
        kwargs["fitness_metric"] = str(cfg.ga_fitness_metric)
    if cfg.ga_feature_penalty is not None:
        kwargs["feature_penalty"] = float(cfg.ga_feature_penalty)
    if cfg.ga_early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(cfg.ga_early_stopping_rounds)
    if cfg.ga_max_boost_round is not None:
        kwargs["max_boost_round"] = int(cfg.ga_max_boost_round)
    if cfg.ga_max_features is not None:
        kwargs["max_features"] = (
            None if int(cfg.ga_max_features) <= 0 else int(cfg.ga_max_features)
        )
    if cfg.ga_selection is not None:
        kwargs["selection"] = str(cfg.ga_selection)
    if cfg.ga_tournament_k is not None:
        kwargs["tournament_k"] = int(cfg.ga_tournament_k)
    if cfg.ga_random_state is not None:
        kwargs["random_state"] = int(cfg.ga_random_state)
    if cfg.ga_eval_workers is not None:
        kwargs["eval_workers"] = int(cfg.ga_eval_workers)
    if cfg.ga_allow_parallel_gpu_eval is not None:
        kwargs["allow_parallel_gpu_eval"] = bool(cfg.ga_allow_parallel_gpu_eval)
    return kwargs


def _xgb_overrides_from_config(cfg: TrainConfig) -> dict:
    overrides: dict = {}
    has_dart_knob = any(
        v is not None
        for v in (
            cfg.xgb_rate_drop,
            cfg.xgb_skip_drop,
            cfg.xgb_one_drop,
            cfg.xgb_sample_type,
            cfg.xgb_normalize_type,
        )
    )
    booster = cfg.xgb_booster
    if booster is None and has_dart_knob:
        booster = "dart"
    if booster is not None:
        overrides["booster"] = str(booster)
    if str(booster).lower() == "dart":
        overrides.update(DART_DEFAULTS)
    if cfg.xgb_rate_drop is not None:
        overrides["rate_drop"] = float(cfg.xgb_rate_drop)
    if cfg.xgb_skip_drop is not None:
        overrides["skip_drop"] = float(cfg.xgb_skip_drop)
    if cfg.xgb_one_drop is not None:
        overrides["one_drop"] = int(bool(cfg.xgb_one_drop))
    if cfg.xgb_sample_type is not None:
        overrides["sample_type"] = str(cfg.xgb_sample_type)
    if cfg.xgb_normalize_type is not None:
        overrides["normalize_type"] = str(cfg.xgb_normalize_type)
    return overrides


def _sanitize_xgb_params(xgb_params: dict) -> dict:
    params = dict(xgb_params)
    # Force single-metric training behavior across fresh and reused artifacts.
    params["eval_metric"] = "logloss"
    booster = str(params.get("booster", "gbtree")).lower()
    if booster != "dart":
        params.pop("rate_drop", None)
        params.pop("skip_drop", None)
        params.pop("one_drop", None)
        params.pop("sample_type", None)
        params.pop("normalize_type", None)
    return params


def _normalize_ga_label_dir(label_mode_or_dir: str) -> str:
    token = str(label_mode_or_dir).strip().lower()
    if token in {"pivot", "pivots"}:
        return "pivots"
    if token == "tb":
        return "tb"
    if token == "swing":
        return "swing"
    return token


def _artifact_side_dir(model_root: Path, side: str, label_dir: str) -> Path:
    return model_root / side.lower() / label_dir


def _maybe_migrate_legacy_label_artifacts(
    model_root: Path, *, side: str, label_dir: str
) -> None:
    base_side = model_root / side.lower()
    target_dir = _artifact_side_dir(model_root, side, label_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    moved_any = False

    # Legacy layout v1: <side>/probs/<label>/...
    legacy_label_dir = base_side / "probs" / label_dir
    if legacy_label_dir.exists():
        for src in legacy_label_dir.iterdir():
            if not src.is_file():
                continue
            dst = target_dir / src.name
            if not dst.exists():
                shutil.move(str(src), str(dst))
                moved_any = True
    for name in ("best_mask.npy", "meta.json", "xgb_model.json", "selected_features.txt"):
        src = legacy_label_dir / name
        dst = target_dir / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            moved_any = True

    # Legacy layout v0: <side>/... (pre-label scoping, swing only).
    if label_dir == "swing":
        legacy_mask = base_side / "best_mask.npy"
        legacy_meta = base_side / "meta.json"
        if legacy_mask.exists() and legacy_meta.exists():
            for name in ("best_mask.npy", "meta.json", "xgb_model.json", "selected_features.txt"):
                src = base_side / name
                dst = target_dir / name
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
                    moved_any = True

    if moved_any:
        print(f"[GA-XGB] Migrated legacy {side.upper()} artifacts -> {target_dir}")


def _maybe_cleanup_legacy_probs_dir(model_root: Path, *, side: str, label_dir: str) -> None:
    legacy_label_dir = model_root / side.lower() / "probs" / label_dir
    if legacy_label_dir.exists():
        try:
            legacy_label_dir.rmdir()
        except OSError:
            pass
    legacy_probs_root = model_root / side.lower() / "probs"
    if legacy_probs_root.exists():
        try:
            legacy_probs_root.rmdir()
        except OSError:
            pass


def _maybe_migrate_legacy_tree(model_root: Path, *, label_dir: str) -> None:
    for side in ("long", "short"):
        _maybe_migrate_legacy_label_artifacts(
            model_root, side=side, label_dir=label_dir
        )
        _maybe_cleanup_legacy_probs_dir(model_root, side=side, label_dir=label_dir)


def load_feature_names(
    ticker: str,
    dataset_name: str,
    x_filename: str,
    *,
    processed_root: Path | None = None,
) -> list[str] | None:
    if processed_root is None:
        dataset_dir = get_ticker_processed_base_dir(ticker) / "datasets" / dataset_name
    else:
        dataset_dir = processed_root / "datasets" / dataset_name
    features_path = dataset_dir / f"features_{Path(x_filename).stem}.txt"
    if not features_path.exists():
        return None
    return [line.strip() for line in features_path.read_text().splitlines() if line.strip()]


def save_selector_artifacts(
    selector: GAXGBoostFeatureSelector,
    output_dir: Path,
    side_name: str,
    *,
    feature_names: list[str] | None = None,
    metadata: dict | None = None,
    label_dir: str | None = None,
) -> Path:
    if selector.xgb_model_ is None or selector.best_mask_ is None:
        raise RuntimeError("Model must be fit before saving artifacts.")

    side_dir = output_dir / side_name.lower()
    if label_dir:
        side_dir = _artifact_side_dir(output_dir, side_name, label_dir)
    side_dir.mkdir(parents=True, exist_ok=True)

    model = selector.xgb_model_
    if hasattr(model, "save_model"):
        model.save_model(str(side_dir / "xgb_model.json"))
    else:
        model.get_booster().save_model(str(side_dir / "xgb_model.json"))
    np.save(side_dir / "best_mask.npy", selector.best_mask_.astype(np.int8))

    meta = {
        "side": side_name,
        "best_score": selector.best_score_,
        "n_features": int(selector.best_mask_.size),
        "selected_features": int(selector.best_mask_.sum()),
        "ga_params": _extract_ga_params(selector),
        "xgb_params": selector.xgb_params,
    }
    if metadata:
        meta.update(metadata)
    (side_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if feature_names:
        selected = [
            name
            for name, keep in zip(feature_names, selector.best_mask_.tolist())
            if keep
        ]
        (side_dir / "selected_features.txt").write_text("\n".join(selected))

    return side_dir


def _load_norm_stats(
    stats_dir: Path, dataset_name: str, x_filename: str
) -> dict | None:
    x_stem = Path(x_filename).stem
    stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())


def load_dataset(
    *,
    ticker: str,
    dataset_name: str,
    x_filename: str,
    label_mode: str,
    apply_scaler: bool,
    super_pivot_weight: float = 1.0,
    processed_root: Path | None = None,
    stats_root: Path | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    clean = normalize_ticker(ticker)
    if processed_root is None:
        processed_dir = get_ticker_processed_base_dir(clean)
    else:
        processed_dir = processed_root
    dataset_dir = processed_dir / "datasets" / dataset_name
    x_path = dataset_dir / x_filename
    y_path = dataset_dir / "y.parquet"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_filename} or y.parquet in {dataset_dir}")

    X_df = pd.read_parquet(x_path)
    y_df = pd.read_parquet(y_path)
    plot_path = dataset_dir / "plot_frame.parquet"
    plot_df = None
    if plot_path.exists():
        plot_df = pd.read_parquet(plot_path)

    if not isinstance(X_df.index, pd.DatetimeIndex):
        if (
            plot_df is not None
            and isinstance(plot_df.index, pd.DatetimeIndex)
            and len(plot_df) == len(X_df)
        ):
            X_df = X_df.copy()
            X_df.index = plot_df.index
            if not isinstance(y_df.index, pd.DatetimeIndex) and len(y_df) == len(plot_df):
                y_df = y_df.copy()
                y_df.index = plot_df.index
            print(
                "[GA-XGB] X index was non-datetime; aligned to plot_frame index "
                "for time-consistent probability artifacts."
            )
        else:
            print(
                "[GA-XGB] Warning: X index is non-datetime and could not be aligned to "
                "plot_frame. Probability parquet indices may be non-temporal."
            )

    if apply_scaler:
        stats_dir = stats_root if stats_root is not None else get_ticker_processed_stats_dir(clean)
        stats = _load_norm_stats(stats_dir, dataset_name, x_filename)
        if stats:
            X_df = apply_scaler_from_stats(X_df, stats)
        else:
            x_stem = Path(x_filename).stem
            stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
            print(f"No scaler stats found at {stats_path}; using raw features.")

    X = X_df.to_numpy(dtype=np.float32)

    sample_weight_long = None
    sample_weight_short = None

    if label_mode == "swing":
        long_col, short_col = "long_swing_label", "short_swing_label"
        missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
        if missing_cols:
            raise KeyError(
                f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}"
            )
        y_long = y_df[long_col].to_numpy(dtype=np.int64)
        y_short = y_df[short_col].to_numpy(dtype=np.int64)
    elif label_mode == "pivot":
        long_col, short_col = "pivot_down", "pivot_up"
        missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
        if missing_cols:
            raise KeyError(
                f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}"
            )
        y_long = y_df[long_col].fillna(0).to_numpy(dtype=np.int64)
        y_short = y_df[short_col].fillna(0).to_numpy(dtype=np.int64)

        if super_pivot_weight != 1.0:
            super_long_col, super_short_col = "super_pivot_down", "super_pivot_up"
            missing_supers = [
                c for c in (super_long_col, super_short_col) if c not in y_df.columns
            ]
            if missing_supers:
                print(
                    "[GA-XGB] super_pivot_weight requested but missing columns: "
                    + ", ".join(missing_supers)
                )
            else:
                sample_weight_long = np.ones_like(y_long, dtype=np.float32)
                sample_weight_short = np.ones_like(y_short, dtype=np.float32)
                super_long = (
                    y_df[super_long_col].fillna(0).astype(int).to_numpy() == 1
                )
                super_short = (
                    y_df[super_short_col].fillna(0).astype(int).to_numpy() == 1
                )
                sample_weight_long[super_long] = float(super_pivot_weight)
                sample_weight_short[super_short] = float(super_pivot_weight)
    elif label_mode == "tb":
        long_col, short_col = "tb_long_label", "tb_short_label"
        missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
        if missing_cols:
            raise KeyError(
                f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}"
            )
        y_long = y_df[long_col].to_numpy(dtype=np.int64)
        y_short = y_df[short_col].to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    return X, y_long, y_short, plot_df, sample_weight_long, sample_weight_short


def _load_split_indices(
    ticker: str,
    dataset_name: str,
    x_filename: str,
    *,
    split_root: Path | None = None,
) -> dict[str, np.ndarray]:
    clean = normalize_ticker(ticker)
    split_root = split_root if split_root is not None else get_ticker_processed_split_dir(clean)
    x_stem = Path(x_filename).stem
    split_dirs = [
        split_root / dataset_name / x_stem,
        split_root / dataset_name,
    ]
    for split_dir in split_dirs:
        train_path = split_dir / "train_idx.npy"
        val_path = split_dir / "val_idx.npy"
        test_path = split_dir / "test_idx.npy"
        missing = [p.name for p in (train_path, val_path, test_path) if not p.exists()]
        if not missing:
            return {
                "train": np.load(train_path),
                "val": np.load(val_path),
                "test": np.load(test_path),
            }
    raise FileNotFoundError(
        f"Missing split files under {split_root / dataset_name} (x_stem={x_stem})."
    )


def _full_split_indices(n: int) -> dict[str, np.ndarray]:
    if n <= 0:
        raise ValueError("No samples available.")
    all_idx = np.arange(n)
    return {"train": all_idx, "val": np.array([], dtype=int), "test": np.array([], dtype=int)}


def load_model_artifacts(
    model_root: Path,
    side: str,
    *,
    label_dir: str,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, dict, dict]:
    side_dir = _artifact_side_dir(model_root, side, label_dir)
    mask_path = side_dir / "best_mask.npy"
    meta_path = side_dir / "meta.json"
    if not (mask_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing artifacts under {side_dir}")

    mask = np.load(mask_path).astype(bool)
    meta = json.loads(meta_path.read_text())
    xgb_params = dict(meta.get("xgb_params", {}))
    if feature_names is not None and mask.size != len(feature_names):
        selected_path = side_dir / "selected_features.txt"
        if not selected_path.exists():
            raise ValueError(
                f"{side.upper()} mask expects {mask.size} features but current dataset has "
                f"{len(feature_names)}, and {selected_path.name} is missing for remap."
            )
        selected_names = [
            line.strip()
            for line in selected_path.read_text().splitlines()
            if line.strip()
        ]
        feature_lookup = {name: idx for idx, name in enumerate(feature_names)}
        remapped = np.zeros(len(feature_names), dtype=bool)
        matched = 0
        missing: list[str] = []
        for name in selected_names:
            idx = feature_lookup.get(name)
            if idx is None:
                missing.append(name)
                continue
            remapped[idx] = True
            matched += 1
        if matched == 0:
            raise ValueError(
                f"{side.upper()} artifact remap failed: none of the saved selected features "
                f"were found in the current dataset."
            )
        msg = (
            f"[GA-XGB] Remapped {side.upper()} {label_dir} mask by feature name: "
            f"old_n_features={mask.size}, new_n_features={len(feature_names)}, "
            f"matched={matched}"
        )
        if missing:
            preview = ", ".join(missing[:5])
            extra = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
            msg += f", dropped={len(missing)} [{preview}{extra}]"
        print(msg)
        mask = remapped
    return mask, xgb_params, meta


def update_model_artifact_meta(
    model_root: Path,
    side: str,
    *,
    label_dir: str,
    xgb_params: dict,
    metadata_updates: dict | None = None,
) -> Path:
    side_dir = _artifact_side_dir(model_root, side, label_dir)
    side_dir.mkdir(parents=True, exist_ok=True)
    meta_path = side_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta["side"] = side.upper()
    meta["xgb_params"] = dict(xgb_params)
    if metadata_updates:
        meta.update(metadata_updates)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def _train_ga_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    xgb_params: dict,
    sample_weight: np.ndarray | None = None,
    ga_kwargs: dict | None = None,
) -> GAXGBoostFeatureSelector:
    selector = GAXGBoostFeatureSelector(
        xgb_params=xgb_params,
        **(ga_kwargs or {}),
    )
    selector.fit(X_train, y_train, sample_weight=sample_weight)
    return selector


def refresh_masks_and_params(
    *,
    X_train: np.ndarray,
    y_long_train: np.ndarray,
    y_short_train: np.ndarray,
    w_long_train: np.ndarray | None,
    w_short_train: np.ndarray | None,
    model_root: Path,
    feature_names: list[str] | None,
    metadata: dict,
    label_dir: str | None = None,
    full_fit: bool = False,
    scale_pos_weight: bool = True,
    ga_kwargs: dict | None = None,
    xgb_param_overrides: dict | None = None,
    train_long: bool = True,
    train_short: bool = True,
) -> tuple[np.ndarray | None, dict | None, np.ndarray | None, dict | None]:
    def _side_params(y_train: np.ndarray) -> dict:
        base = GAXGBoostFeatureSelector().xgb_params.copy()
        if xgb_param_overrides:
            base.update(xgb_param_overrides)
        base = _sanitize_xgb_params(base)
        if scale_pos_weight:
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            base["scale_pos_weight"] = neg / max(pos, 1)
        else:
            base["scale_pos_weight"] = 1.0
        return base

    def _save_checkpoint(side: str, selector: GAXGBoostFeatureSelector) -> None:
        # Persist each side immediately so long runs can resume after interruptions.
        checkpoint_dir = save_selector_artifacts(
            selector,
            model_root,
            side,
            feature_names=feature_names,
            metadata=metadata,
            label_dir=label_dir,
        )
        print(f"[GA-XGB] Checkpoint saved for {side.upper()} at {checkpoint_dir}")

    print("Refreshing GA-XGB masks/params on train split only...")
    long_selector: GAXGBoostFeatureSelector | None = None
    short_selector: GAXGBoostFeatureSelector | None = None
    if full_fit:
        if train_long:
            long_selector = GAXGBoostFeatureSelector(
                xgb_params=_side_params(y_long_train),
                **(ga_kwargs or {}),
            )
            long_selector.fit(X_train, y_long_train, sample_weight=w_long_train)
            _save_checkpoint("long", long_selector)

        if train_short:
            short_selector = GAXGBoostFeatureSelector(
                xgb_params=_side_params(y_short_train),
                **(ga_kwargs or {}),
            )
            short_selector.fit(X_train, y_short_train, sample_weight=w_short_train)
            _save_checkpoint("short", short_selector)
    else:
        if train_long:
            long_selector = _train_ga_selector(
                X_train,
                y_long_train,
                xgb_params=_side_params(y_long_train),
                sample_weight=w_long_train,
                ga_kwargs=ga_kwargs,
            )
            _save_checkpoint("long", long_selector)

        if train_short:
            short_selector = _train_ga_selector(
                X_train,
                y_short_train,
                xgb_params=_side_params(y_short_train),
                sample_weight=w_short_train,
                ga_kwargs=ga_kwargs,
            )
            _save_checkpoint("short", short_selector)

    long_mask = (
        long_selector.best_mask_.astype(bool) if long_selector is not None else None
    )
    short_mask = (
        short_selector.best_mask_.astype(bool) if short_selector is not None else None
    )
    long_params = (
        _sanitize_xgb_params(long_selector.xgb_params)
        if long_selector is not None
        else None
    )
    short_params = (
        _sanitize_xgb_params(short_selector.xgb_params)
        if short_selector is not None
        else None
    )
    return long_mask, long_params, short_mask, short_params


def _maybe_update_scale_pos_weight(
    xgb_params: dict, y_train: np.ndarray, *, enabled: bool
) -> dict:
    if not enabled:
        return xgb_params
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale = neg / max(pos, 1)
    updated = dict(xgb_params)
    updated["scale_pos_weight"] = scale
    return updated


def _fit_xgb_with_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_params: dict,
    sample_weight: np.ndarray | None = None,
    eval_set: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    ga_kwargs: dict | None = None,
) -> tuple[GAXGBoostFeatureSelector, object]:
    selector = GAXGBoostFeatureSelector(xgb_params=xgb_params, **(ga_kwargs or {}))
    model = selector._fit_xgb(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=eval_set,
    )
    return selector, model


def _print_label_stats(y: np.ndarray, name: str) -> None:
    total = int(y.size)
    pos = int((y == 1).sum())
    neg = total - pos
    pct = 100.0 * pos / max(total, 1)
    print(f"[GA-XGB] {name}: n={total}, pos={pos} ({pct:.2f}%), neg={neg}")


def _summarize_probs(probs: np.ndarray, name: str) -> None:
    if probs.size == 0:
        print(f"[GA-XGB] {name}: empty")
        return
    p = probs[np.isfinite(probs)]
    if p.size == 0:
        print(f"[GA-XGB] {name}: no finite values")
        return
    qs = np.quantile(p, [0.01, 0.1, 0.5, 0.9, 0.99])
    print(
        f"[GA-XGB] {name}: mean={float(np.mean(p)):.4f}, "
        f"p01={qs[0]:.4f}, p10={qs[1]:.4f}, p50={qs[2]:.4f}, "
        f"p90={qs[3]:.4f}, p99={qs[4]:.4f}"
    )


def _print_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    name: str,
    threshold: float = 0.5,
    threshold_metric_beta: float = 1.0,
) -> dict[str, float] | None:
    if probs.size == 0:
        print(f"[GA-XGB] {name}: empty")
        return None
    mask = np.isfinite(probs)
    y = y_true[mask]
    p = probs[mask]
    if y.size == 0:
        print(f"[GA-XGB] {name}: no finite values")
        return None

    pred = (p >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    fbeta = fbeta_score(y, pred, beta=threshold_metric_beta, zero_division=0)
    metric_name = "f1" if np.isclose(threshold_metric_beta, 1.0) else f"f{threshold_metric_beta:g}"
    try:
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        ap = float("nan")
    try:
        ll = log_loss(y, p, labels=[0, 1])
    except ValueError:
        ll = float("nan")
    if len(np.unique(y)) > 1:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    else:
        tn = int((y == 0).sum())
        tp = int((y == 1).sum())
        fp = 0
        fn = 0

    print(
        f"[GA-XGB] {name} (thr={threshold:.2f}): "
        f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, "
        f"{metric_name}={fbeta:.4f}, "
        f"auc={auc:.4f}, ap={ap:.4f}, logloss={ll:.4f}, "
        f"tp={tp}, fp={fp}, tn={tn}, fn={fn}"
    )
    return {
        "n": float(y.size),
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f_beta": float(fbeta),
        "f_beta_beta": float(threshold_metric_beta),
        "auc": float(auc),
        "average_precision": float(ap),
        "logloss": float(ll),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _find_best_fbeta_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    beta: float = 1.0,
    default_threshold: float = 0.5,
    grid_size: int = 199,
) -> tuple[float, float]:
    if beta <= 0:
        raise ValueError("beta must be > 0 for F-beta threshold search.")
    mask = np.isfinite(probs)
    y = y_true[mask]
    p = probs[mask]
    if y.size == 0:
        return float(default_threshold), float("nan")

    thresholds = np.linspace(0.01, 0.99, max(3, int(grid_size)))
    best_t = float(default_threshold)
    best_score = -1.0
    for t in thresholds:
        pred = (p >= t).astype(int)
        score = float(fbeta_score(y, pred, beta=beta, zero_division=0))
        if score > best_score:
            best_score = score
            best_t = float(t)
    if best_score < 0.0:
        return float(default_threshold), float("nan")
    return best_t, best_score


def walk_forward_oof_probs(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask: np.ndarray,
    xgb_params: dict,
    n_folds: int,
    initial_train_size: int | None,
    update_scale_pos_weight: bool,
    sample_weight: np.ndarray | None = None,
    ga_kwargs: dict | None = None,
) -> np.ndarray:
    train_end = X_train.shape[0]
    if train_end <= 1:
        raise ValueError("Not enough samples for OOF training.")
    if sample_weight is not None and sample_weight.shape[0] != train_end:
        raise ValueError("sample_weight must match X_train length.")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    if initial_train_size is None:
        initial_train_size = max(1, train_end // (n_folds + 1))
    if initial_train_size >= train_end:
        raise ValueError("initial_train_size must be < train_end")

    remaining = train_end - initial_train_size
    fold_size = remaining // n_folds
    if fold_size <= 0:
        raise ValueError("Fold size too small; reduce n_folds or initial_train_size.")

    oof = np.full(train_end, np.nan, dtype=np.float32)
    for fold in range(n_folds):
        fold_start = initial_train_size + fold * fold_size
        fold_end = initial_train_size + (fold + 1) * fold_size
        if fold == n_folds - 1:
            fold_end = train_end
        if fold_start >= fold_end:
            continue
        print(
            f"[GA-XGB] OOF fold {fold + 1}/{n_folds}: "
            f"fit_rows={fold_start}, pred_rows={fold_end - fold_start}"
        )

        X_fit = X_train[:fold_start][:, mask]
        y_fit = y_train[:fold_start]
        params = _maybe_update_scale_pos_weight(
            xgb_params, y_fit, enabled=update_scale_pos_weight
        )
        w_fit = None if sample_weight is None else sample_weight[:fold_start]
        selector, model = _fit_xgb_with_selector(
            X_fit, y_fit, params, sample_weight=w_fit, ga_kwargs=ga_kwargs
        )

        X_pred = X_train[fold_start:fold_end][:, mask]
        use_gpu = selector._use_gpu is True
        dmat = selector._make_dmatrix(X_pred, y=None, use_gpu=use_gpu)
        probs = selector._to_numpy(model.predict(dmat)).astype(np.float32)
        oof[fold_start:fold_end] = probs
        print(f"[GA-XGB] OOF fold {fold + 1}/{n_folds}: done")

    return oof


def train_final_and_predict_test(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mask: np.ndarray,
    xgb_params: dict,
    update_scale_pos_weight: bool,
    sample_weight: np.ndarray | None = None,
    eval_set: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    ga_kwargs: dict | None = None,
    save_model_path: Path | None = None,
) -> tuple[np.ndarray, dict | None]:
    if sample_weight is not None and sample_weight.shape[0] != X_train.shape[0]:
        raise ValueError("sample_weight must match X_train length.")

    X_fit = X_train[:, mask]
    y_fit = y_train
    params = _maybe_update_scale_pos_weight(
        xgb_params, y_fit, enabled=update_scale_pos_weight
    )
    eval_selected = None
    if eval_set is not None:
        X_val, y_val, w_val = eval_set
        if X_val.shape[0] > 0:
            eval_selected = (X_val[:, mask], y_val, w_val)

    selector, model = _fit_xgb_with_selector(
        X_fit,
        y_fit,
        params,
        sample_weight=sample_weight,
        eval_set=eval_selected,
        ga_kwargs=ga_kwargs,
    )
    eval_history = selector.last_evals_result_
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_model"):
            model.save_model(str(save_model_path))
        else:
            model.get_booster().save_model(str(save_model_path))
    if X_test.size == 0:
        return np.empty((0,), dtype=np.float32), eval_history

    X_test = X_test[:, mask]
    use_gpu = selector._use_gpu is True
    dmat = selector._make_dmatrix(X_test, y=None, use_gpu=use_gpu)
    probs = selector._to_numpy(model.predict(dmat)).astype(np.float32)
    return probs, eval_history


def _plot_train_val_logloss(
    *,
    long_history: dict | None,
    short_history: dict | None,
    save_path: Path,
) -> bool:
    def _extract(history: dict | None) -> tuple[list[float], list[float] | None]:
        if not history:
            return [], None
        train = history.get("train", {})
        val = history.get("val", {})
        train_ll = train.get("logloss", []) if isinstance(train, dict) else []
        val_ll = val.get("logloss", []) if isinstance(val, dict) else []
        if not isinstance(train_ll, list):
            train_ll = list(train_ll)
        if not isinstance(val_ll, list):
            val_ll = list(val_ll)
        return train_ll, (val_ll if len(val_ll) else None)

    long_train, long_val = _extract(long_history)
    short_train, short_val = _extract(short_history)
    if not long_train and not short_train:
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    sides = (
        ("LONG", long_train, long_val, axes[0]),
        ("SHORT", short_train, short_val, axes[1]),
    )
    for title, train_ll, val_ll, ax in sides:
        if train_ll:
            ax.plot(train_ll, label="train_logloss", color="#1f77b4")
        if val_ll:
            ax.plot(val_ll, label="val_logloss", color="#ff7f0e")
        ax.set_title(f"{title} XGBoost Logloss")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Logloss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("GA-XGB Final Fit: Train vs Validation Logloss")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return True


def _print_eval_history_summary(*, side: str, history: dict | None) -> None:
    if not history:
        print(f"[GA-XGB] {side} eval history: unavailable")
        return
    train = history.get("train", {})
    val = history.get("val", {})
    train_ll = train.get("logloss", []) if isinstance(train, dict) else []
    val_ll = val.get("logloss", []) if isinstance(val, dict) else []
    if not isinstance(train_ll, list):
        train_ll = list(train_ll)
    if not isinstance(val_ll, list):
        val_ll = list(val_ll)
    if not train_ll:
        print(f"[GA-XGB] {side} eval history: no train logloss series")
        return
    if val_ll:
        best_idx = int(np.argmin(val_ll))
        print(
            f"[GA-XGB] {side} final-fit logloss: rounds={len(train_ll)}, "
            f"best_val={val_ll[best_idx]:.5f} @ round {best_idx + 1}, "
            f"final_train={train_ll[-1]:.5f}, final_val={val_ll[-1]:.5f}"
        )
    else:
        print(
            f"[GA-XGB] {side} final-fit logloss: rounds={len(train_ll)}, "
            f"final_train={train_ll[-1]:.5f} (no val series)"
        )


def _save_series(
    *,
    output_dir: Path,
    prefix: str,
    train_oof: np.ndarray,
    test_probs: np.ndarray,
    full_probs: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    index: pd.Index | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_npy_path = output_dir / f"{prefix}_oof_train.npy"
    test_npy_path = output_dir / f"{prefix}_test.npy"
    full_npy_path = output_dir / f"{prefix}_full.npy"

    oof_to_save = train_oof
    if oof_npy_path.exists():
        current_has_signal = train_oof.size and np.isfinite(train_oof).any()
        if not current_has_signal:
            prev_oof = np.load(oof_npy_path)
            if prev_oof.size and np.isfinite(prev_oof).any():
                oof_to_save = prev_oof
                print(
                    f"[GA-XGB] Preserving existing OOF array in {oof_npy_path.name} "
                    "during full-fit overwrite."
                )

    test_to_save = test_probs
    if test_npy_path.exists():
        current_has_signal = test_probs.size and np.isfinite(test_probs).any()
        if not current_has_signal:
            prev_test = np.load(test_npy_path)
            if prev_test.size and np.isfinite(prev_test).any():
                test_to_save = prev_test
                print(
                    f"[GA-XGB] Preserving existing test array in {test_npy_path.name} "
                    "during full-fit overwrite."
                )

    np.save(oof_npy_path, oof_to_save)
    np.save(test_npy_path, test_to_save)
    np.save(full_npy_path, full_probs)

    if index is not None:
        oof_full = np.full_like(full_probs, np.nan, dtype=np.float32)
        test_full = np.full_like(full_probs, np.nan, dtype=np.float32)
        if train_idx.size and train_oof.size:
            oof_full[train_idx] = train_oof
        if test_idx.size and test_probs.size:
            test_full[test_idx] = test_probs
        parquet_path = output_dir / f"{prefix}_probs.parquet"
        if (
            parquet_path.exists()
            and not np.isfinite(oof_full).any()
            and not np.isfinite(test_full).any()
        ):
            existing = pd.read_parquet(parquet_path)
            existing = existing.reindex(index[: full_probs.shape[0]])
            prev_oof = pd.to_numeric(
                existing.get(f"{prefix}_oof_train", pd.Series(index=existing.index, dtype=float)),
                errors="coerce",
            ).to_numpy(dtype=np.float32)
            prev_test = pd.to_numeric(
                existing.get(f"{prefix}_test", pd.Series(index=existing.index, dtype=float)),
                errors="coerce",
            ).to_numpy(dtype=np.float32)
            if np.isfinite(prev_oof).any() or np.isfinite(prev_test).any():
                oof_full = prev_oof
                test_full = prev_test
                print(
                    f"[GA-XGB] Preserving existing OOF/test probability columns in "
                    f"{parquet_path.name} during full-fit overwrite."
                )
        df = pd.DataFrame(
            {
                f"{prefix}_oof_train": oof_full,
                f"{prefix}_test": test_full,
                f"{prefix}_full": full_probs,
            },
            index=index[: full_probs.shape[0]],
        )
        df.to_parquet(parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe OOF probs for GA-XGB.")
    parser.add_argument(
        "--ticker",
        type=str,
        default=TrainConfig.ticker,
        help="Ticker (default: SPY).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=TrainConfig.dataset_name,
        help="Dataset name under processed datasets/ (e.g., 10min).",
    )
    parser.add_argument(
        "--x-filename",
        type=str,
        default=TrainConfig.x_filename,
        help="Feature parquet filename under dataset dir (e.g., X_10min_tree.parquet).",
    )
    parser.add_argument(
        "--refresh-masks",
        action="store_true",
        help="Re-run GA feature selection on the train split to refresh masks/params.",
    )
    parser.add_argument(
        "--no-ga-mask",
        action="store_true",
        help="Bypass GA feature masking and train XGBoost on all features directly.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default=None,
        choices=["swing", "pivot", "tb"],
        help="Label mode to use (default: swing).",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["both", "long", "short"],
        help="Train/evaluate both sides (default), or only one side.",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=1.0,
        help=(
            "F-beta used to select OOF decision thresholds. "
            "beta<1 favors precision, beta>1 favors recall. Default: 1.0 (F1)."
        ),
    )
    parser.add_argument(
        "--super-pivot-weight",
        type=float,
        default=TrainConfig.super_pivot_weight,
        help="Sample-weight multiplier for super pivot events (pivot mode only).",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Override processed data root (contains datasets/).",
    )
    parser.add_argument(
        "--split-root",
        type=str,
        default=None,
        help="Override split root (contains <dataset>/<x_stem>/train_idx.npy).",
    )
    parser.add_argument(
        "--stats-root",
        type=str,
        default=None,
        help="Override stats root (contains norm_stats_*.json).",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=None,
        help="Override model output root (will create <model_dirname>/<dataset>).",
    )
    parser.add_argument(
        "--full-fit",
        action="store_true",
        help="Train GA-XGB on the full dataset (no holdout).",
    )
    parser.add_argument("--ga-population-size", type=int, default=None)
    parser.add_argument("--ga-generations", type=int, default=None)
    parser.add_argument("--ga-crossover-rate", type=float, default=None)
    parser.add_argument("--ga-mutation-rate", type=float, default=None)
    parser.add_argument("--ga-val-size", type=float, default=None)
    parser.add_argument(
        "--ga-fitness-metric",
        type=str,
        choices=["f1", "f1_penalized", "logloss", "logloss_penalized"],
        default=None,
    )
    parser.add_argument("--ga-feature-penalty", type=float, default=None)
    parser.add_argument("--ga-early-stopping-rounds", type=int, default=None)
    parser.add_argument("--ga-max-boost-round", type=int, default=None)
    parser.add_argument(
        "--ga-max-features",
        type=int,
        default=None,
        help="GA feature cap. Set <=0 to disable cap.",
    )
    parser.add_argument(
        "--ga-selection",
        type=str,
        choices=["tournament", "roulette"],
        default=None,
    )
    parser.add_argument("--ga-tournament-k", type=int, default=None)
    parser.add_argument("--ga-random-state", type=int, default=None)
    parser.add_argument(
        "--ga-eval-workers",
        type=int,
        default=None,
        help="Parallel workers for GA chromosome evaluation (best for CPU fallback mode).",
    )
    parser.add_argument(
        "--ga-allow-parallel-gpu-eval",
        action="store_true",
        help="Allow parallel chromosome evaluation when GPU is active (may increase memory use).",
    )
    parser.add_argument(
        "--xgb-booster",
        "--booster",
        type=str,
        choices=["gbtree", "dart"],
        default=None,
        help="XGBoost booster type. Use dart to enable tree-dropout regularization with default DART settings unless overridden.",
    )
    parser.add_argument(
        "--xgb-rate-drop",
        type=float,
        default=None,
        help="DART: fraction of previous trees to drop each boosting round.",
    )
    parser.add_argument(
        "--xgb-skip-drop",
        type=float,
        default=None,
        help="DART: probability of skipping dropout in a boosting round.",
    )
    parser.add_argument(
        "--xgb-one-drop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DART: enforce dropping at least one tree (true/false).",
    )
    parser.add_argument(
        "--xgb-sample-type",
        type=str,
        choices=["uniform", "weighted"],
        default=None,
        help="DART: tree sampling method for dropout.",
    )
    parser.add_argument(
        "--xgb-normalize-type",
        type=str,
        choices=["tree", "forest"],
        default=None,
        help="DART: normalization mode after dropout.",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        ticker=args.ticker,
        dataset_name=args.dataset_name,
        x_filename=args.x_filename,
        refresh_masks=bool(args.refresh_masks),
        no_ga_mask=bool(args.no_ga_mask),
        label_mode=args.label_mode or TrainConfig.label_mode,
        super_pivot_weight=float(args.super_pivot_weight),
        processed_root=args.processed_root,
        split_root=args.split_root,
        stats_root=args.stats_root,
        model_root=args.model_root,
        ga_population_size=args.ga_population_size,
        ga_generations=args.ga_generations,
        ga_crossover_rate=args.ga_crossover_rate,
        ga_mutation_rate=args.ga_mutation_rate,
        ga_val_size=args.ga_val_size,
        ga_fitness_metric=args.ga_fitness_metric,
        ga_feature_penalty=args.ga_feature_penalty,
        ga_early_stopping_rounds=args.ga_early_stopping_rounds,
        ga_max_boost_round=args.ga_max_boost_round,
        ga_max_features=args.ga_max_features,
        ga_selection=args.ga_selection,
        ga_tournament_k=args.ga_tournament_k,
        ga_random_state=args.ga_random_state,
        ga_eval_workers=args.ga_eval_workers,
        ga_allow_parallel_gpu_eval=args.ga_allow_parallel_gpu_eval,
        xgb_booster=args.xgb_booster,
        xgb_rate_drop=args.xgb_rate_drop,
        xgb_skip_drop=args.xgb_skip_drop,
        xgb_one_drop=args.xgb_one_drop,
        xgb_sample_type=args.xgb_sample_type,
        xgb_normalize_type=args.xgb_normalize_type,
    )
    ga_kwargs = _ga_kwargs_from_config(cfg)
    xgb_overrides = _xgb_overrides_from_config(cfg)
    side_mode = str(args.side).strip().lower()
    run_long = side_mode in {"both", "long"}
    run_short = side_mode in {"both", "short"}
    if not run_long and not run_short:
        raise ValueError("--side must be one of: both, long, short")
    threshold_beta = float(args.threshold_beta)
    if threshold_beta <= 0:
        raise ValueError("--threshold-beta must be > 0.")
    metric_label = "F1" if np.isclose(threshold_beta, 1.0) else f"F{threshold_beta:g}"
    label_dir_probs = _normalize_ga_label_dir(cfg.label_mode)
    artifact_label_dir = label_dir_probs
    processed_root = Path(cfg.processed_root) if cfg.processed_root else None
    split_root = Path(cfg.split_root) if cfg.split_root else None
    stats_root = Path(cfg.stats_root) if cfg.stats_root else None
    model_root_override = Path(cfg.model_root) if cfg.model_root else None

    X, y_long, y_short, plot_df, w_long, w_short = load_dataset(
        ticker=cfg.ticker,
        dataset_name=cfg.dataset_name,
        x_filename=cfg.x_filename,
        label_mode=cfg.label_mode,
        apply_scaler=cfg.apply_scaler,
        super_pivot_weight=cfg.super_pivot_weight,
        processed_root=processed_root,
        stats_root=stats_root,
    )
    plot_index = plot_df.index if plot_df is not None else None

    if args.full_fit:
        splits = _full_split_indices(X.shape[0])
    else:
        splits = _load_split_indices(
            cfg.ticker,
            cfg.dataset_name,
            cfg.x_filename,
            split_root=split_root,
        )
    train_idx = np.sort(splits["train"])
    val_idx = np.sort(splits["val"])
    test_idx = np.sort(splits["test"])
    if train_idx.size < 2:
        raise ValueError("Not enough training samples for OOF.")

    if model_root_override is None:
        model_root = REPO_ROOT / "Data" / "models"
    else:
        model_root = model_root_override
    model_root = model_root / cfg.model_dirname
    model_dataset_root = model_root / cfg.dataset_name
    _maybe_migrate_legacy_tree(model_dataset_root, label_dir=artifact_label_dir)
    feature_names = load_feature_names(
        cfg.ticker,
        cfg.dataset_name,
        cfg.x_filename,
        processed_root=processed_root,
    )
    common_meta = {
        "ticker": cfg.ticker,
        "dataset_name": cfg.dataset_name,
        "label_mode": cfg.label_mode,
        "super_pivot_weight": cfg.super_pivot_weight,
        "side_mode": side_mode,
    }

    train_val_idx = np.sort(np.concatenate([train_idx, val_idx]))
    X_train = X[train_val_idx]
    y_long_train = y_long[train_val_idx]
    y_short_train = y_short[train_val_idx]
    w_long_train = w_long[train_val_idx] if w_long is not None else None
    w_short_train = w_short[train_val_idx] if w_short is not None else None

    X_train_only = X[train_idx]
    y_long_train_only = y_long[train_idx]
    y_short_train_only = y_short[train_idx]
    w_long_train_only = w_long[train_idx] if w_long is not None else None
    w_short_train_only = w_short[train_idx] if w_short is not None else None

    if val_idx.size:
        X_val = X[val_idx]
        y_long_val = y_long[val_idx]
        y_short_val = y_short[val_idx]
        w_long_val = w_long[val_idx] if w_long is not None else None
        w_short_val = w_short[val_idx] if w_short is not None else None
    else:
        X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_long_val = np.empty((0,), dtype=y_long.dtype)
        y_short_val = np.empty((0,), dtype=y_short.dtype)
        w_long_val = None
        w_short_val = None
    if args.full_fit:
        X_test = np.empty((0, X.shape[1]), dtype=X.dtype)
    else:
        X_test = X[test_idx]

    scale_pos_weight = cfg.update_scale_pos_weight
    if cfg.label_mode == "tb":
        scale_pos_weight = False

    default_params = _sanitize_xgb_params(GAXGBoostFeatureSelector().xgb_params.copy())
    long_mask = np.ones(X.shape[1], dtype=bool)
    short_mask = np.ones(X.shape[1], dtype=bool)
    long_params = dict(default_params)
    short_params = dict(default_params)
    long_meta: dict = {"mode": "untrained", "ga_params": {}, "best_score": None}
    short_meta: dict = {"mode": "untrained", "ga_params": {}, "best_score": None}
    long_meta_path: Path | None = None
    short_meta_path: Path | None = None

    if cfg.no_ga_mask:
        if cfg.refresh_masks:
            print("[GA-XGB] --refresh-masks ignored because --no-ga-mask is enabled.")
        print(
            "[GA-XGB] --no-ga-mask enabled: using all features (GA bypassed)"
            f" for side={side_mode}."
        )
        if xgb_overrides:
            if run_long:
                long_params = _sanitize_xgb_params({**long_params, **xgb_overrides})
            if run_short:
                short_params = _sanitize_xgb_params({**short_params, **xgb_overrides})
        if run_long:
            long_meta = {
                "best_score": None,
                "ga_params": {},
                "xgb_params": long_params,
                "mode": "no_ga_mask",
            }
        if run_short:
            short_meta = {
                "best_score": None,
                "ga_params": {},
                "xgb_params": short_params,
                "mode": "no_ga_mask",
            }
    else:
        need_refresh = cfg.refresh_masks
        if not need_refresh:
            try:
                if run_long:
                    load_model_artifacts(
                        model_dataset_root,
                        "long",
                        label_dir=artifact_label_dir,
                        feature_names=feature_names,
                    )
                if run_short:
                    load_model_artifacts(
                        model_dataset_root,
                        "short",
                        label_dir=artifact_label_dir,
                        feature_names=feature_names,
                    )
            except FileNotFoundError:
                need_refresh = True

        if need_refresh:
            refresh_masks_and_params(
                X_train=X_train,
                y_long_train=y_long_train,
                y_short_train=y_short_train,
                w_long_train=w_long_train,
                w_short_train=w_short_train,
                model_root=model_dataset_root,
                feature_names=feature_names,
                metadata=common_meta,
                label_dir=artifact_label_dir,
                full_fit=args.full_fit,
                scale_pos_weight=scale_pos_weight,
                ga_kwargs=ga_kwargs,
                xgb_param_overrides=xgb_overrides,
                train_long=run_long,
                train_short=run_short,
            )
        if run_long:
            long_mask, long_params, long_meta = load_model_artifacts(
                model_dataset_root,
                "long",
                label_dir=artifact_label_dir,
                feature_names=feature_names,
            )
            long_params = _sanitize_xgb_params(long_params)
        if run_short:
            short_mask, short_params, short_meta = load_model_artifacts(
                model_dataset_root,
                "short",
                label_dir=artifact_label_dir,
                feature_names=feature_names,
            )
            short_params = _sanitize_xgb_params(short_params)
        if xgb_overrides:
            if run_long:
                long_params = _sanitize_xgb_params({**long_params, **xgb_overrides})
            if run_short:
                short_params = _sanitize_xgb_params({**short_params, **xgb_overrides})

    if run_long and long_mask.size != X.shape[1]:
        raise ValueError("LONG mask size does not match feature count.")
    if run_short and short_mask.size != X.shape[1]:
        raise ValueError("SHORT mask size does not match feature count.")

    if run_long:
        long_meta_path = update_model_artifact_meta(
            model_dataset_root,
            "long",
            label_dir=artifact_label_dir,
            xgb_params=long_params,
            metadata_updates={
                "mode": long_meta.get("mode", "ga_mask"),
                "best_score": long_meta.get("best_score"),
                "ga_params": dict(long_meta.get("ga_params", {})),
            },
        )
    if run_short:
        short_meta_path = update_model_artifact_meta(
            model_dataset_root,
            "short",
            label_dir=artifact_label_dir,
            xgb_params=short_params,
            metadata_updates={
                "mode": short_meta.get("mode", "ga_mask"),
                "best_score": short_meta.get("best_score"),
                "ga_params": dict(short_meta.get("ga_params", {})),
            },
        )

    print(
        "Split sizes: "
        f"train+val={train_val_idx.size}, test={test_idx.size}"
    )
    print(f"[GA-XGB] Side selection: {side_mode}")
    if run_long:
        _print_label_stats(y_long_train, "LONG labels (train+val)")
    if run_short:
        _print_label_stats(y_short_train, "SHORT labels (train+val)")
    if cfg.label_mode == "pivot" and cfg.super_pivot_weight != 1.0:
        if w_long_train is not None and w_short_train is not None:
            long_super = int((w_long_train > 1.0).sum())
            short_super = int((w_short_train > 1.0).sum())
            print(
                f"[GA-XGB] super_pivot_weight={cfg.super_pivot_weight:g} "
                f"(long super={long_super}, short super={short_super})"
            )
    print(
        f"[GA-XGB] XGBoost objective="
        f"{(long_params if run_long else short_params).get('objective', 'binary:logistic')} "
        f"eval_metric={(long_params if run_long else short_params).get('eval_metric', 'logloss')}"
    )
    if xgb_overrides:
        print(f"[GA-XGB] XGBoost overrides={xgb_overrides}")

    long_oof = np.full(train_val_idx.size, np.nan, dtype=np.float32)
    short_oof = np.full(train_val_idx.size, np.nan, dtype=np.float32)
    long_test = np.empty((0,), dtype=np.float32)
    short_test = np.empty((0,), dtype=np.float32)
    long_eval_history = None
    short_eval_history = None
    if not args.full_fit:
        if run_long:
            long_oof = walk_forward_oof_probs(
                X_train=X_train,
                y_train=y_long_train,
                mask=long_mask,
                xgb_params=long_params,
                n_folds=cfg.n_folds,
                initial_train_size=cfg.initial_train_size,
                update_scale_pos_weight=scale_pos_weight,
                sample_weight=w_long_train,
                ga_kwargs=ga_kwargs,
            )
        if run_short:
            short_oof = walk_forward_oof_probs(
                X_train=X_train,
                y_train=y_short_train,
                mask=short_mask,
                xgb_params=short_params,
                n_folds=cfg.n_folds,
                initial_train_size=cfg.initial_train_size,
                update_scale_pos_weight=scale_pos_weight,
                sample_weight=w_short_train,
                ga_kwargs=ga_kwargs,
            )

        if run_long:
            long_test, long_eval_history = train_final_and_predict_test(
                X_train=X_train_only,
                y_train=y_long_train_only,
                X_test=X_test,
                mask=long_mask,
                xgb_params=long_params,
                update_scale_pos_weight=scale_pos_weight,
                sample_weight=w_long_train_only,
                eval_set=(X_val, y_long_val, w_long_val) if val_idx.size else None,
                ga_kwargs=ga_kwargs,
                save_model_path=(
                    _artifact_side_dir(model_dataset_root, "long", artifact_label_dir)
                    / "xgb_model.json"
                ),
            )
        if run_short:
            short_test, short_eval_history = train_final_and_predict_test(
                X_train=X_train_only,
                y_train=y_short_train_only,
                X_test=X_test,
                mask=short_mask,
                xgb_params=short_params,
                update_scale_pos_weight=scale_pos_weight,
                sample_weight=w_short_train_only,
                eval_set=(X_val, y_short_val, w_short_val) if val_idx.size else None,
                ga_kwargs=ga_kwargs,
                save_model_path=(
                    _artifact_side_dir(model_dataset_root, "short", artifact_label_dir)
                    / "xgb_model.json"
                ),
            )
        if run_long:
            _print_eval_history_summary(side="LONG", history=long_eval_history)
        if run_short:
            _print_eval_history_summary(side="SHORT", history=short_eval_history)

    if run_long:
        _summarize_probs(long_oof, "LONG OOF probs")
    if run_short:
        _summarize_probs(short_oof, "SHORT OOF probs")
    if run_long:
        _summarize_probs(long_test, "LONG test probs")
    if run_short:
        _summarize_probs(short_test, "SHORT test probs")
    long_oof_metrics: dict[str, float] | None = None
    short_oof_metrics: dict[str, float] | None = None
    long_test_metrics: dict[str, float] | None = None
    short_test_metrics: dict[str, float] | None = None
    long_full_train_metrics: dict[str, float] | None = None
    short_full_train_metrics: dict[str, float] | None = None
    long_threshold = 0.5
    short_threshold = 0.5
    if not args.full_fit:
        if run_long:
            long_threshold, long_best_oof_score = _find_best_fbeta_threshold(
                y_long_train,
                long_oof,
                beta=threshold_beta,
            )
            print(
                f"[GA-XGB] LONG best OOF {metric_label} threshold={long_threshold:.4f} "
                f"({metric_label.lower()}={long_best_oof_score:.4f})"
            )
            long_oof_metrics = _print_binary_metrics(
                y_long_train,
                long_oof,
                name="LONG OOF metrics",
                threshold=long_threshold,
                threshold_metric_beta=threshold_beta,
            )
            long_test_metrics = _print_binary_metrics(
                y_long[test_idx],
                long_test,
                name="LONG test metrics",
                threshold=long_threshold,
                threshold_metric_beta=threshold_beta,
            )
        if run_short:
            short_threshold, short_best_oof_score = _find_best_fbeta_threshold(
                y_short_train,
                short_oof,
                beta=threshold_beta,
            )
            print(
                f"[GA-XGB] SHORT best OOF {metric_label} threshold={short_threshold:.4f} "
                f"({metric_label.lower()}={short_best_oof_score:.4f})"
            )
            short_oof_metrics = _print_binary_metrics(
                y_short_train,
                short_oof,
                name="SHORT OOF metrics",
                threshold=short_threshold,
                threshold_metric_beta=threshold_beta,
            )
            short_test_metrics = _print_binary_metrics(
                y_short[test_idx],
                short_test,
                name="SHORT test metrics",
                threshold=short_threshold,
                threshold_metric_beta=threshold_beta,
            )

    n_total = X.shape[0]
    if args.full_fit:
        long_full = np.full(n_total, np.nan, dtype=np.float32)
        short_full = np.full(n_total, np.nan, dtype=np.float32)
        if run_long:
            long_full, _ = train_final_and_predict_test(
                X_train=X_train,
                y_train=y_long_train,
                X_test=X_train,
                mask=long_mask,
                xgb_params=long_params,
                update_scale_pos_weight=scale_pos_weight,
                sample_weight=w_long_train,
                ga_kwargs=ga_kwargs,
                save_model_path=(
                    _artifact_side_dir(model_dataset_root, "long", artifact_label_dir)
                    / "xgb_model.json"
                ),
            )
            if long_full.size != n_total:
                raise ValueError("LONG full-fit predictions do not match dataset length.")
            long_full_train_metrics = _print_binary_metrics(
                y_long_train,
                long_full,
                name="LONG full-fit train metrics",
                threshold_metric_beta=threshold_beta,
            )
        if run_short:
            short_full, _ = train_final_and_predict_test(
                X_train=X_train,
                y_train=y_short_train,
                X_test=X_train,
                mask=short_mask,
                xgb_params=short_params,
                update_scale_pos_weight=scale_pos_weight,
                sample_weight=w_short_train,
                ga_kwargs=ga_kwargs,
                save_model_path=(
                    _artifact_side_dir(model_dataset_root, "short", artifact_label_dir)
                    / "xgb_model.json"
                ),
            )
            if short_full.size != n_total:
                raise ValueError("SHORT full-fit predictions do not match dataset length.")
            short_full_train_metrics = _print_binary_metrics(
                y_short_train,
                short_full,
                name="SHORT full-fit train metrics",
                threshold_metric_beta=threshold_beta,
            )
    else:
        long_full = np.full(n_total, np.nan, dtype=np.float32)
        short_full = np.full(n_total, np.nan, dtype=np.float32)
        if run_long:
            long_full[train_val_idx] = long_oof
            if long_test.size:
                long_full[test_idx] = long_test
        if run_short:
            short_full[train_val_idx] = short_oof
            if short_test.size:
                short_full[test_idx] = short_test

    probs_root = model_dataset_root
    label_dir = label_dir_probs
    if run_long:
        _save_series(
            output_dir=probs_root / "long" / label_dir,
            prefix="p_long",
            train_oof=long_oof,
            test_probs=long_test,
            full_probs=long_full,
            train_idx=train_val_idx,
            test_idx=test_idx,
            index=plot_index,
        )
    if run_short:
        _save_series(
            output_dir=probs_root / "short" / label_dir,
            prefix="p_short",
            train_oof=short_oof,
            test_probs=short_test,
            full_probs=short_full,
            train_idx=train_val_idx,
            test_idx=test_idx,
            index=plot_index,
        )

    missing_long = int(np.isnan(long_oof).sum()) if run_long else 0
    missing_short = int(np.isnan(short_oof).sum()) if run_short else 0
    if not args.full_fit and ((run_long and missing_long) or (run_short and missing_short)):
        print(
            "OOF gap detected (early bars without prior data). "
            f"long={missing_long}, short={missing_short}"
        )
    if plot_df is not None and not args.full_fit:
        test_df = plot_df.iloc[test_idx]
        y_long_test = y_long[test_idx] if run_long else np.empty((0,), dtype=y_long.dtype)
        y_short_test = y_short[test_idx] if run_short else np.empty((0,), dtype=y_short.dtype)
        tail = 200
        if len(test_df) > tail:
            test_df = test_df.tail(tail)
            long_test_tail = long_test[-tail:] if run_long else np.empty((0,), dtype=np.float32)
            short_test_tail = short_test[-tail:] if run_short else np.empty((0,), dtype=np.float32)
            y_long_test = y_long_test[-tail:]
            y_short_test = y_short_test[-tail:]
        else:
            long_test_tail = long_test if run_long else np.empty((0,), dtype=np.float32)
            short_test_tail = short_test if run_short else np.empty((0,), dtype=np.float32)

        save_path = get_default_model_inference_plot_path(
            cfg.ticker, f"ga_xgb_{cfg.label_mode}_test"
        )
        plot_model_inference(
            test_df,
            long_test_tail if run_long and long_test_tail.size else None,
            short_test_tail if run_short and short_test_tail.size else None,
            long_actual=y_long_test if run_long and y_long_test.size else None,
            short_actual=y_short_test if run_short and y_short_test.size else None,
            long_label_name="LONG",
            short_label_name="SHORT",
            threshold=0.5,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            title=f"{normalize_ticker(cfg.ticker)} | GA-XGB {cfg.label_mode} (test tail)",
            save_path=str(save_path),
        )

    loss_plot_path = model_dataset_root / f"ga_xgb_{cfg.label_mode}_train_vs_val_loss.png"
    saved_loss_plot = False
    if not args.full_fit:
        saved_loss_plot = _plot_train_val_logloss(
            long_history=long_eval_history,
            short_history=short_eval_history,
            save_path=loss_plot_path,
        )
        if saved_loss_plot:
            print(f"[GA-XGB] Saved loss curve plot: {loss_plot_path}")

    print(f"Saved OOF/test/full probability arrays under {probs_root}")
    trained_sides = [side for side, enabled in (("long", run_long), ("short", run_short)) if enabled]
    hyperparams = {
        **asdict(cfg),
        **vars(args),
        "scale_pos_weight_enabled": bool(scale_pos_weight),
        "model_dataset_root": str(model_dataset_root),
        "trained_sides": trained_sides,
        "long_ga_params": dict(long_meta.get("ga_params", {})) if run_long else None,
        "short_ga_params": dict(short_meta.get("ga_params", {})) if run_short else None,
    }
    train_metrics = (
        {
            "long_oof": long_oof_metrics if run_long else None,
            "short_oof": short_oof_metrics if run_short else None,
        }
        if not args.full_fit
        else {
            "long_full_train": long_full_train_metrics if run_long else None,
            "short_full_train": short_full_train_metrics if run_short else None,
        }
    )
    validation_metrics = (
        {
            "long_test": long_test_metrics if run_long else None,
            "short_test": short_test_metrics if run_short else None,
            "test_rows": float(test_idx.size),
            "val_rows": float(val_idx.size),
        }
        if not args.full_fit
        else {"skipped": "--full-fit"}
    )
    best_validation_metrics = {
        "long_selector_best_score": long_meta.get("best_score") if run_long else None,
        "short_selector_best_score": short_meta.get("best_score") if run_short else None,
    }
    log_paths = log_training_run(
        run_name="ga_xgboost_train",
        output_dir=probs_root,
        hyperparameters=hyperparams,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_validation_metrics=best_validation_metrics,
        artifacts={
            "probs_root": str(probs_root),
            "long_probs_dir": str(probs_root / "long" / label_dir) if run_long else None,
            "short_probs_dir": str(probs_root / "short" / label_dir) if run_short else None,
            "loss_curve_plot": str(loss_plot_path) if saved_loss_plot else None,
            "long_model_path": (
                str(probs_root / "long" / label_dir / "xgb_model.json") if run_long else None
            ),
            "short_model_path": (
                str(probs_root / "short" / label_dir / "xgb_model.json") if run_short else None
            ),
            "long_meta_path": (
                str(long_meta_path)
                if long_meta_path is not None and long_meta_path.exists()
                else None
            ),
            "short_meta_path": (
                str(short_meta_path)
                if short_meta_path is not None and short_meta_path.exists()
                else None
            ),
        },
        extra={
            "ticker": normalize_ticker(cfg.ticker),
            "dataset_name": cfg.dataset_name,
            "label_mode": cfg.label_mode,
            "feature_count": int(X.shape[1]),
            "train_rows": int(train_idx.size),
            "val_rows": int(val_idx.size),
            "test_rows": int(test_idx.size),
            "oof_missing_long": int(np.isnan(long_oof).sum()) if run_long else None,
            "oof_missing_short": int(np.isnan(short_oof).sum()) if run_short else None,
        },
    )
    print(f"[GA-XGB] Saved training run summary: {log_paths['latest_path']}")


if __name__ == "__main__":
    main()
