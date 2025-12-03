import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector

# X: shape (n_samples, n_features), y: shape (n_samples,)
# Assume you've already built your expanded TA feature matrix here.
X = np.load("data/X_spy_daily.npy")
y = np.load("data/y_spy_daily.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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
test_acc = accuracy_score(y_test, y_pred)
print("Final test accuracy:", test_acc)
