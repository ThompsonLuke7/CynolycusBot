"""
ga_xgboost_compare.py

Compare:
  (1) Plain XGBoost on all features (no GA),
  (2) GA-based feature selection with LogisticRegression fitness
      but XGBoost as the final classifier,
  (3) GA + XGBoost wrapper feature selection (your current method).

Also:
  - Print confusion matrix for the GA+XGB model.
  - Plot actual SPY price vs a strategy equity curve that goes long
    when GA+XGB predicts "up".
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

X = np.load("data/processed/X_spy_daily.npy")
y = np.load("data/processed/y_spy_daily.npy")

# Close prices aligned with X/y (saved during feature engineering)
close = np.load("data/processed/close_spy_daily.npy")

# Use a fixed chronological split so the test set is a real "future" segment.
# (If your X/y are already shuffled, switch to shuffle=True in train_test_split instead.)
n = len(X)
split = int(n * 0.92)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
close_train, close_test = close[:split], close[split:]


# ---------------------------------------------------------------------
# 1) Plain XGBoost baseline (no GA)
# ---------------------------------------------------------------------

xgb_all = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
)

xgb_all.fit(X_train, y_train)
y_pred_all = xgb_all.predict(X_test)
test_f1 = f1_score(y_test, y_pred_all, pos_label=1)
print(f"[Baseline] XGB on all features - test F1 score: {test_f1:.4f}")


# ---------------------------------------------------------------------
# 2) GA with LogisticRegression fitness, XGB as final classifier
# ---------------------------------------------------------------------

def ga_lr_feature_selector(
    X_train,
    y_train,
    val_size=0.15,
    population_size=8,
    generations=30,
    crossover_rate=0.5,
    mutation_rate=0.375,
    random_state=42,
):
    """
    Simpler GA feature selector that uses LogisticRegression as the fitness model.
    Returns: best_mask, best_val_acc
    """
    rng = np.random.default_rng(random_state)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
        shuffle=True,
    )

    n_features = X_train.shape[1]

    # Initialize population
    pop = rng.integers(0, 2, size=(population_size, n_features), dtype=np.int8)

    def ensure_valid_population(pop):
        for i in range(pop.shape[0]):
            if not pop[i].any():
                idx = rng.integers(0, pop.shape[1])
                pop[i, idx] = 1
        return pop

    def eval_chromosome(mask):
        if not mask.any():
            return 0.0
        sel = mask.astype(bool)
        X_tr_sel = X_tr[:, sel]
        X_val_sel = X_val[:, sel]

        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(X_tr_sel, y_tr)
        y_hat = clf.predict(X_val_sel)
        return f1_score(y_val, y_hat, pos_label=1)

    pop = ensure_valid_population(pop)
    best_mask = None
    best_score = -np.inf

    for gen in range(generations):
        fitness = np.zeros(population_size, dtype=float)
        for i, mask in enumerate(pop):
            fitness[i] = eval_chromosome(mask)

        gen_best_idx = int(np.argmax(fitness))
        gen_best_score = fitness[gen_best_idx]
        gen_best_mask = pop[gen_best_idx].copy()

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_mask = gen_best_mask

        print(
            f"[GA-LR] Gen {gen+1}/{generations} "
            f"- best val acc: {gen_best_score:.4f}, global best: {best_score:.4f}"
        )

        # Selection (roulette wheel)
        probs = fitness / fitness.sum()
        parent_idx = rng.choice(
            population_size,
            size=population_size,
            replace=True,
            p=probs,
        )
        parents = pop[parent_idx]

        # Crossover (uniform)
        offspring = parents.copy()
        for i in range(0, population_size, 2):
            if i + 1 >= population_size:
                break
            if rng.random() < crossover_rate:
                mask = rng.integers(0, 2, size=n_features, dtype=bool)
                p1, p2 = offspring[i].copy(), offspring[i + 1].copy()
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
                offspring[i], offspring[i + 1] = c1, c2

        # Mutation
        mut_mask = rng.random(offspring.shape) < mutation_rate
        offspring[mut_mask] = 1 - offspring[mut_mask]

        offspring = ensure_valid_population(offspring)
        pop = offspring

    return best_mask.astype(bool), best_score


# Run GA-LR for feature selection
ga_lr_mask, ga_lr_best_val = ga_lr_feature_selector(X_train, y_train)
print(f"[GA-LR] Selected features: {ga_lr_mask.sum()}, best val acc: {ga_lr_best_val:.4f}")

# Train final XGB on GA-LR-selected features
xgb_ga_lr = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
)

X_train_ga_lr = X_train[:, ga_lr_mask]
X_test_ga_lr = X_test[:, ga_lr_mask]

xgb_ga_lr.fit(X_train_ga_lr, y_train)
y_pred_ga_lr = xgb_ga_lr.predict(X_test_ga_lr)
test_f1 = f1_score(y_test, y_pred_ga_lr, pos_label=1)
print(f"[GA-LR + XGB] test F1 score: {test_f1:.4f}")


# ---------------------------------------------------------------------
# 3) GA + XGBoost wrapper (your existing class)
# ---------------------------------------------------------------------

selector = GAXGBoostFeatureSelector(
    population_size=8,
    generations=30,
    crossover_rate=0.5,
    mutation_rate=0.375,
    random_state=42,
)

selector.fit(X_train, y_train)
print(f"[GA-XGB] best val F1 score: {selector.best_score_:.4f}")
print(f"[GA-XGB] selected features: {selector.best_mask_.sum()}")

y_pred_ga_xgb = selector.predict(X_test)
test_f1 = f1_score(y_test, y_pred_ga_xgb, pos_label=1)
print(f"[GA-XGB] test F1 score: {test_f1:.4f}")


# ---------------------------------------------------------------------
# Confusion matrix and classification report for GA+XGB
# ---------------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred_ga_xgb)
print("\n[GA-XGB] Confusion matrix:")
print(cm)

print("\n[GA-XGB] Classification report:")
print(classification_report(y_test, y_pred_ga_xgb, digits=4))


# ---------------------------------------------------------------------
# Plot: actual SPY vs strategy equity curve (GA+XGB signals)
# ---------------------------------------------------------------------

# Normalize prices for plotting
prices = close_test.astype(float)
prices = prices / prices[0]  # start at 1.0

# Build simple strategy: at day t, if model predicts "up",
# we are long SPY for the move from t -> t+1; else we are flat.
# That uses predictions for all but the last day.
returns = prices[1:] / prices[:-1] - 1.0
signals = y_pred_ga_xgb[:-1]  # prediction at t used for return t+1

strategy_rets = returns * (signals == 1)
bh_curve = (1.0 + returns).cumprod()
strategy_curve = (1.0 + strategy_rets).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(bh_curve, label="Buy & Hold SPY", linewidth=2)
plt.plot(strategy_curve, label="GA-XGB Strategy (long on 'up' signals')", linewidth=2)
plt.title("SPY vs GA-XGB Strategy Equity Curve (Test Set)")
plt.xlabel("Test Samples (chronological)")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
