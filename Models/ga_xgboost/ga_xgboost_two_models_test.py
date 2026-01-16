import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from Data.load_data import (
    get_ticker_processed_base_dir,
    get_ticker_processed_split_dir,
    get_ticker_processed_stats_dir,
)
from Data.retrieve_data import normalize_ticker
from Features.feature_scaling import apply_scaler_from_stats

from Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


def _load_norm_stats(
    stats_dir: Path, dataset_name: str, x_filename: str
) -> dict | None:
    x_stem = Path(x_filename).stem
    stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())


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
        missing = [
            p.name for p in (train_path, val_path, test_path) if not p.exists()
        ]
        if not missing:
            return {
                "train": np.load(train_path),
                "val": np.load(val_path),
                "test": np.load(test_path),
            }
    raise FileNotFoundError(
        f"Missing split files under {split_root / dataset_name} (x_stem={x_stem})."
    )


def _load_scaled_dataset_splits(
    *,
    ticker: str,
    dataset_name: str,
    label_mode: str,
    x_filename: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    clean = normalize_ticker(ticker)
    processed_dir = get_ticker_processed_base_dir(clean)
    dataset_dir = processed_dir / "datasets" / dataset_name
    x_path = dataset_dir / x_filename
    y_path = dataset_dir / "y.parquet"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_filename} or y.parquet in {dataset_dir}")

    X_df = pd.read_parquet(x_path)
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

    splits = _load_split_indices(clean, dataset_name, x_filename)
    return {name: (X[idx], y_long[idx], y_short[idx]) for name, idx in splits.items()}


def load_feature_names(ticker: str, dataset_name: str) -> list[str] | None:
    dataset_dir = get_ticker_processed_base_dir(ticker) / "datasets" / dataset_name
    features_path = dataset_dir / "features.txt"
    if not features_path.exists():
        return None
    return [
        line.strip() for line in features_path.read_text().splitlines() if line.strip()
    ]


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

    selector.xgb_model_.save_model(side_dir / "xgb_model.json")
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


# ------------------------------
# Load dataset + split indices
# ------------------------------
TICKER = "$SPY"
DATASET_NAME = "15min"
LABEL_MODE = "swing"
MODEL_NAME = "ga_xgboost_two_models"
X_FILENAME = "X_15min_tree.parquet"

REPO_ROOT = _resolve_repo_root()
MODEL_DIR = REPO_ROOT / "Data" / "models" / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

splits = _load_scaled_dataset_splits(
    ticker=TICKER,
    dataset_name=DATASET_NAME,
    label_mode=LABEL_MODE,
    x_filename=X_FILENAME,
)

X_train, y_long_train, y_short_train = splits["train"]
X_test, y_long_test, y_short_test = splits["test"]

print(f"Loaded train split: {X_train.shape}")
print(f"Loaded test split:  {X_test.shape}")


def train_and_eval_side(y_train, y_test, side_name):
    """
    Train one GA+XGB model for a single side (LONG or SHORT)
    and print basic metrics.
    """
    # Class imbalance: weight positives more inside XGBoost
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / max(pos, 1)
    print(f"\n=== {side_name} side ===")
    print(
        f"{side_name} train positives: {pos}, negatives: {neg}, scale_pos_weight: {scale:.2f}"
    )

    # Start from default XGB params and add scale_pos_weight
    base_selector = GAXGBoostFeatureSelector()
    xgb_params = base_selector.xgb_params.copy()
    xgb_params["scale_pos_weight"] = scale

    selector = GAXGBoostFeatureSelector(
        population_size=8,
        generations=30,
        crossover_rate=0.5,
        mutation_rate=0.375,
        val_size=0.08,  # last 8% of TRAIN used as GA validation
        random_state=42,
        xgb_params=xgb_params,
    )

    # GA + XGB training on this side
    selector.fit(X_train, y_train)

    print(f"{side_name} best GA val F1:", selector.best_score_)
    print(f"{side_name} selected features:", selector.best_mask_.sum())

    # Evaluate on the hold-out test window
    y_pred = selector.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"{side_name} TEST F1 (class 1): {f1:.4f}")
    print(f"{side_name} TEST confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return selector


# ------------------------------
# Train both models
# ------------------------------
long_selector = train_and_eval_side(y_long_train, y_long_test, "LONG")
short_selector = train_and_eval_side(y_short_train, y_short_test, "SHORT")

# ------------------------------
# Save artifacts
# ------------------------------
feature_names = load_feature_names(TICKER, DATASET_NAME)
common_meta = {
    "ticker": TICKER,
    "dataset_name": DATASET_NAME,
    "label_mode": LABEL_MODE,
}

long_dir = save_selector_artifacts(
    long_selector,
    MODEL_DIR,
    "long",
    feature_names=feature_names,
    metadata=common_meta,
)
short_dir = save_selector_artifacts(
    short_selector,
    MODEL_DIR,
    "short",
    feature_names=feature_names,
    metadata=common_meta,
)

print(f"\nSaved LONG artifacts to: {long_dir}")
print(f"Saved SHORT artifacts to: {short_dir}")

# ------------------------------
# Tiny inference example
# ------------------------------
# Use the last bar in the test split as an example input
x_latest = X_test[-1:].astype(np.float32)

long_pred = long_selector.predict(x_latest)[0]
short_pred = short_selector.predict(x_latest)[0]

print("\nLatest bar predictions:")
print("  LONG model:  1 = good long swing, 0 = not:", long_pred)
print("  SHORT model: 1 = good short swing, 0 = not:", short_pred)
