from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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

from Data.load_data import (
    get_ticker_processed_base_dir,
    get_ticker_processed_split_dir,
    get_ticker_processed_stats_dir,
)
from Data.retrieve_data import normalize_ticker
from Features.feature_scaling import apply_scaler_from_stats
from Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector


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
    update_scale_pos_weight: bool = False
    output_dirname: str = "probs"
    refresh_masks: bool = False


def load_feature_names(ticker: str, dataset_name: str, x_filename: str) -> list[str] | None:
    dataset_dir = get_ticker_processed_base_dir(ticker) / "datasets" / dataset_name
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
) -> Path:
    if selector.xgb_model_ is None or selector.best_mask_ is None:
        raise RuntimeError("Model must be fit before saving artifacts.")

    side_dir = output_dir / side_name.lower()
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index | None]:
    clean = normalize_ticker(ticker)
    processed_dir = get_ticker_processed_base_dir(clean)
    dataset_dir = processed_dir / "datasets" / dataset_name
    x_path = dataset_dir / x_filename
    y_path = dataset_dir / "y.parquet"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_filename} or y.parquet in {dataset_dir}")

    X_df = pd.read_parquet(x_path)
    if apply_scaler:
        stats_dir = get_ticker_processed_stats_dir(clean)
        stats = _load_norm_stats(stats_dir, dataset_name, x_filename)
        if stats:
            X_df = apply_scaler_from_stats(X_df, stats)
        else:
            x_stem = Path(x_filename).stem
            stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
            print(f"No scaler stats found at {stats_path}; using raw features.")

    X = X_df.to_numpy(dtype=np.float32)
    y_df = pd.read_parquet(y_path)

    if label_mode == "swing":
        long_col, short_col = "long_swing_label", "short_swing_label"
    elif label_mode == "leg":
        long_col, short_col = "leg_up_label", "leg_down_label"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}"
        )

    y_long = y_df[long_col].to_numpy(dtype=np.int64)
    y_short = y_df[short_col].to_numpy(dtype=np.int64)

    plot_path = dataset_dir / "plot_frame.parquet"
    plot_index = None
    if plot_path.exists():
        plot_index = pd.read_parquet(plot_path).index

    return X, y_long, y_short, plot_index


def _load_split_indices(
    ticker: str,
    dataset_name: str,
    x_filename: str,
) -> dict[str, np.ndarray]:
    clean = normalize_ticker(ticker)
    split_root = get_ticker_processed_split_dir(clean)
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


def load_model_artifacts(model_root: Path, side: str) -> tuple[np.ndarray, dict]:
    side_dir = model_root / side.lower()
    mask_path = side_dir / "best_mask.npy"
    meta_path = side_dir / "meta.json"
    if not mask_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing artifacts under {side_dir}")

    mask = np.load(mask_path).astype(bool)
    meta = json.loads(meta_path.read_text())
    xgb_params = dict(meta.get("xgb_params", {}))
    return mask, xgb_params


def _train_ga_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    xgb_params: dict,
) -> GAXGBoostFeatureSelector:
    selector = GAXGBoostFeatureSelector(
        population_size=24,
        generations=60,
        crossover_rate=0.6,
        mutation_rate=0.005,
        val_size=0.15,
        random_state=42,
        xgb_params=xgb_params,
        fitness_metric="f1_penalized",
        feature_penalty=0.0015,
        max_features=80,
        selection="tournament",
        tournament_k=3,
    )
    selector.fit(X_train, y_train)
    return selector


def refresh_masks_and_params(
    *,
    X_train: np.ndarray,
    y_long_train: np.ndarray,
    y_short_train: np.ndarray,
    model_root: Path,
    feature_names: list[str] | None,
    metadata: dict,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    def _side_params(y_train: np.ndarray) -> dict:
        base = GAXGBoostFeatureSelector().xgb_params.copy()
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        base["scale_pos_weight"] = neg / max(pos, 1)
        return base

    print("Refreshing GA-XGB masks/params on train split only...")
    long_selector = _train_ga_selector(
        X_train, y_long_train, xgb_params=_side_params(y_long_train)
    )
    short_selector = _train_ga_selector(
        X_train, y_short_train, xgb_params=_side_params(y_short_train)
    )

    save_selector_artifacts(
        long_selector,
        model_root,
        "long",
        feature_names=feature_names,
        metadata=metadata,
    )
    save_selector_artifacts(
        short_selector,
        model_root,
        "short",
        feature_names=feature_names,
        metadata=metadata,
    )

    long_mask = long_selector.best_mask_.astype(bool)
    short_mask = short_selector.best_mask_.astype(bool)
    long_params = long_selector.xgb_params
    short_params = short_selector.xgb_params
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
) -> tuple[GAXGBoostFeatureSelector, object]:
    selector = GAXGBoostFeatureSelector(xgb_params=xgb_params)
    model = selector._fit_xgb(X_train, y_train)
    return selector, model


def walk_forward_oof_probs(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask: np.ndarray,
    xgb_params: dict,
    n_folds: int,
    initial_train_size: int | None,
    update_scale_pos_weight: bool,
) -> np.ndarray:
    train_end = X_train.shape[0]
    if train_end <= 1:
        raise ValueError("Not enough samples for OOF training.")
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

        X_fit = X_train[:fold_start][:, mask]
        y_fit = y_train[:fold_start]
        params = _maybe_update_scale_pos_weight(
            xgb_params, y_fit, enabled=update_scale_pos_weight
        )
        selector, model = _fit_xgb_with_selector(X_fit, y_fit, params)

        X_pred = X_train[fold_start:fold_end][:, mask]
        use_gpu = selector._use_gpu is True
        dmat = selector._make_dmatrix(X_pred, y=None, use_gpu=use_gpu)
        probs = selector._to_numpy(model.predict(dmat)).astype(np.float32)
        oof[fold_start:fold_end] = probs

    return oof


def train_final_and_predict_test(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mask: np.ndarray,
    xgb_params: dict,
    update_scale_pos_weight: bool,
) -> np.ndarray:
    if X_test.size == 0:
        return np.empty((0,), dtype=np.float32)

    X_fit = X_train[:, mask]
    y_fit = y_train
    params = _maybe_update_scale_pos_weight(
        xgb_params, y_fit, enabled=update_scale_pos_weight
    )
    selector, model = _fit_xgb_with_selector(X_fit, y_fit, params)

    X_test = X_test[:, mask]
    use_gpu = selector._use_gpu is True
    dmat = selector._make_dmatrix(X_test, y=None, use_gpu=use_gpu)
    return selector._to_numpy(model.predict(dmat)).astype(np.float32)


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
    np.save(output_dir / f"{prefix}_oof_train.npy", train_oof)
    np.save(output_dir / f"{prefix}_test.npy", test_probs)
    np.save(output_dir / f"{prefix}_full.npy", full_probs)

    if index is not None:
        oof_full = np.full_like(full_probs, np.nan, dtype=np.float32)
        test_full = np.full_like(full_probs, np.nan, dtype=np.float32)
        oof_full[train_idx] = train_oof
        test_full[test_idx] = test_probs
        df = pd.DataFrame(
            {
                f"{prefix}_oof_train": oof_full,
                f"{prefix}_test": test_full,
                f"{prefix}_full": full_probs,
            },
            index=index[: full_probs.shape[0]],
        )
        df.to_parquet(output_dir / f"{prefix}_probs.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe OOF probs for GA-XGB.")
    parser.add_argument(
        "--refresh-masks",
        action="store_true",
        help="Re-run GA feature selection on the train split to refresh masks/params.",
    )
    args = parser.parse_args()

    cfg = TrainConfig(refresh_masks=bool(args.refresh_masks))
    X, y_long, y_short, plot_index = load_dataset(
        ticker=cfg.ticker,
        dataset_name=cfg.dataset_name,
        x_filename=cfg.x_filename,
        label_mode=cfg.label_mode,
        apply_scaler=cfg.apply_scaler,
    )

    splits = _load_split_indices(cfg.ticker, cfg.dataset_name, cfg.x_filename)
    train_idx = np.sort(splits["train"])
    val_idx = np.sort(splits["val"])
    test_idx = np.sort(splits["test"])
    if train_idx.size < 2:
        raise ValueError("Not enough training samples for OOF.")

    model_root = REPO_ROOT / "Data" / "models" / cfg.model_dirname
    model_dataset_root = model_root / cfg.dataset_name
    feature_names = load_feature_names(cfg.ticker, cfg.dataset_name, cfg.x_filename)
    common_meta = {
        "ticker": cfg.ticker,
        "dataset_name": cfg.dataset_name,
        "label_mode": cfg.label_mode,
    }

    X_train = X[train_idx]
    y_long_train = y_long[train_idx]
    y_short_train = y_short[train_idx]
    X_test = X[test_idx]

    need_refresh = cfg.refresh_masks
    if not need_refresh:
        try:
            long_mask, long_params = load_model_artifacts(model_dataset_root, "long")
            short_mask, short_params = load_model_artifacts(model_dataset_root, "short")
        except FileNotFoundError:
            need_refresh = True

    if need_refresh:
        long_mask, long_params, short_mask, short_params = refresh_masks_and_params(
            X_train=X_train,
            y_long_train=y_long_train,
            y_short_train=y_short_train,
            model_root=model_dataset_root,
            feature_names=feature_names,
            metadata=common_meta,
        )

    if long_mask.size != X.shape[1] or short_mask.size != X.shape[1]:
        raise ValueError("Mask size does not match feature count.")

    print(
        "Split sizes: "
        f"train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
    )

    long_oof = walk_forward_oof_probs(
        X_train=X_train,
        y_train=y_long_train,
        mask=long_mask,
        xgb_params=long_params,
        n_folds=cfg.n_folds,
        initial_train_size=cfg.initial_train_size,
        update_scale_pos_weight=cfg.update_scale_pos_weight,
    )
    short_oof = walk_forward_oof_probs(
        X_train=X_train,
        y_train=y_short_train,
        mask=short_mask,
        xgb_params=short_params,
        n_folds=cfg.n_folds,
        initial_train_size=cfg.initial_train_size,
        update_scale_pos_weight=cfg.update_scale_pos_weight,
    )

    long_test = train_final_and_predict_test(
        X_train=X_train,
        y_train=y_long_train,
        X_test=X_test,
        mask=long_mask,
        xgb_params=long_params,
        update_scale_pos_weight=cfg.update_scale_pos_weight,
    )
    short_test = train_final_and_predict_test(
        X_train=X_train,
        y_train=y_short_train,
        X_test=X_test,
        mask=short_mask,
        xgb_params=short_params,
        update_scale_pos_weight=cfg.update_scale_pos_weight,
    )

    n_total = X.shape[0]
    long_full = np.full(n_total, np.nan, dtype=np.float32)
    short_full = np.full(n_total, np.nan, dtype=np.float32)
    long_full[train_idx] = long_oof
    short_full[train_idx] = short_oof
    if long_test.size:
        long_full[test_idx] = long_test
    if short_test.size:
        short_full[test_idx] = short_test

    probs_root = model_dataset_root
    _save_series(
        output_dir=probs_root / "long" / cfg.output_dirname,
        prefix="p_long",
        train_oof=long_oof,
        test_probs=long_test,
        full_probs=long_full,
        train_idx=train_idx,
        test_idx=test_idx,
        index=plot_index,
    )
    _save_series(
        output_dir=probs_root / "short" / cfg.output_dirname,
        prefix="p_short",
        train_oof=short_oof,
        test_probs=short_test,
        full_probs=short_full,
        train_idx=train_idx,
        test_idx=test_idx,
        index=plot_index,
    )

    missing_long = int(np.isnan(long_oof).sum())
    missing_short = int(np.isnan(short_oof).sum())
    if missing_long or missing_short:
        print(
            "OOF gap detected (early bars without prior data). "
            f"long={missing_long}, short={missing_short}"
        )
    print(f"Saved OOF/test/full probability arrays under {probs_root}")


if __name__ == "__main__":
    main()
