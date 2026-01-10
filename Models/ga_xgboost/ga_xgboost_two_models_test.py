import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from Data.load_data import get_ticker_processed_base_dir, load_dataset_splits

from Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


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

REPO_ROOT = _resolve_repo_root()
MODEL_DIR = REPO_ROOT / "Data" / "models" / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

splits = load_dataset_splits(
    ticker=TICKER,
    dataset_name=DATASET_NAME,
    label_mode=LABEL_MODE,
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
