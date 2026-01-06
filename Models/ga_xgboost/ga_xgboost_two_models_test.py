from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector

# ------------------------------
# Load pre-split features + labels
# ------------------------------
SPLIT_ROOT = Path("Data/processed/splits/spy_daily")


def load_split(name: str):
    """Load a specific split (train/val/test) if it exists."""
    split_dir = SPLIT_ROOT / name
    X = np.load(split_dir / f"X_spy_daily_{name}.npy")
    y_long = np.load(split_dir / f"y_spy_daily_long_{name}.npy")  # 1 = good long swing
    y_short = np.load(
        split_dir / f"y_spy_daily_short_{name}.npy"
    )  # 1 = good short swing
    return X, y_long, y_short


X_train, y_long_train, y_short_train = load_split("train")
X_test, y_long_test, y_short_test = load_split("test")

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
# Tiny inference example
# ------------------------------
# Use the last bar in the test split as an example input
x_latest = X_test[-1:].astype(np.float32)

long_pred = long_selector.predict(x_latest)[0]
short_pred = short_selector.predict(x_latest)[0]

print("\nLatest bar predictions:")
print("  LONG model:  1 = good long swing, 0 = not:", long_pred)
print("  SHORT model: 1 = good short swing, 0 = not:", short_pred)
