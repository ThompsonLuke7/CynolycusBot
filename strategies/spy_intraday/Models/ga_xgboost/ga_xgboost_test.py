import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector

# X: shape (n_samples, n_features), y: shape (n_samples,)
# Assume you've already built your expanded TA feature matrix here.
X = np.load("data/processed/X_spy_daily.npy")
y = np.load("data/processed/y_spy_daily.npy")

# Chronological time-series split (no shuffling)
n = len(X)
split = int(n * 0.92)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train window:", 0, "→", split)
print("Test window:", split, "→", n)


selector = GAXGBoostFeatureSelector(
    population_size=8,
    generations=30,
    crossover_rate=0.5,
    mutation_rate=0.375,
    random_state=42,
)

selector.fit(X_train, y_train)

print("Best GA-XGB val acc:", selector.best_score_)
print("Number of selected features:", selector.best_mask_.sum())

# Evaluate final model on test set
y_pred = selector.predict(X_test)
test_f1 = f1_score(y_test, y_pred, pos_label=1)
print("Final test F1 score:", test_f1)
