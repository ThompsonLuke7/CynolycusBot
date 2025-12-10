import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector

# ------------------------------
# Load shared features + 2 labels
# ------------------------------
X = np.load("data/processed/X_spy_daily.npy")
y_long = np.load("data/processed/y_spy_daily_long.npy")   # 1 = good long swing
y_short = np.load("data/processed/y_spy_daily_short.npy") # 1 = good short swing


# Chronological time-series split (no shuffling)
n = len(X)
split = int(n * 0.92)   # first 92% train, last 8% test

X_train, X_test = X[:split], X[split:]
y_long_train, y_long_test = y_long[:split], y_long[split:]
y_short_train, y_short_test = y_short[:split], y_short[split:]

print("Train window:", 0, "→", split)
print("Test window:", split, "→", n)


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
    print(f"{side_name} train positives: {pos}, negatives: {neg}, scale_pos_weight: {scale:.2f}")

    # Start from default XGB params and add scale_pos_weight
    base_selector = GAXGBoostFeatureSelector()
    xgb_params = base_selector.xgb_params.copy()
    xgb_params["scale_pos_weight"] = scale

    selector = GAXGBoostFeatureSelector(
        population_size=8,
        generations=30,
        crossover_rate=0.5,
        mutation_rate=0.375,
        val_size=0.08,           # last 8% of TRAIN used as GA validation
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
# Use the last bar in the full dataset as an example input
x_latest = X[-1:].astype(np.float32)

long_pred = long_selector.predict(x_latest)[0]
short_pred = short_selector.predict(x_latest)[0]

print("\nLatest bar predictions:")
print("  LONG model:  1 = good long swing, 0 = not:", long_pred)
print("  SHORT model: 1 = good short swing, 0 = not:", short_pred)
