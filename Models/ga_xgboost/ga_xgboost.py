# models/ga_xgboost/ga_xgboost.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from xgboost.core import XGBoostError


@dataclass
class GAXGBoostFeatureSelector:
    """
    GA + XGBoost wrapper-based feature selector (and classifier).

    GA details are based on Yun et al. 2021:
      - Chromosome = binary mask over features
      - Population size = 8
      - Generations = 30
      - Crossover rate = 0.5
      - Mutation rate = 0.375
      - Fitness = validation accuracy of XGBoost on selected features
    """
    population_size: int = 8
    generations: int = 30
    crossover_rate: float = 0.5
    mutation_rate: float = 0.375
    val_size: float = 0.08          # portion of given data used as validation
    random_state: Optional[int] = 42
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
    })

    # learned attributes
    best_mask_: Optional[np.ndarray] = field(init=False, default=None)
    best_score_: Optional[float] = field(init=False, default=None)
    xgb_model_: Optional[XGBClassifier] = field(init=False, default=None)
    _use_gpu: Optional[bool] = field(init=False, default=None)

    # ---------- Public API ----------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAXGBoostFeatureSelector":
        """
        Run GA to select an optimal feature subset and train final XGB model
        on all training data using that subset.
        """
        rng = np.random.default_rng(self.random_state)

        # Chronological split: first chunk = train, later chunk = validation
        n_samples = X.shape[0]
        split_idx = int(n_samples * (1.0 - self.val_size))

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]


        n_features = X.shape[1]

        # Initialize population: binary masks
        population = rng.integers(0, 2, size=(self.population_size, n_features), dtype=np.int8)
        population = self._ensure_valid_population(population, rng)

        best_mask = None
        best_score = -np.inf

        for gen in range(self.generations):
            fitness = np.zeros(self.population_size)

            # Evaluate fitness (validation accuracy) of each chromosome
            for i, mask in enumerate(population):
                fitness[i] = self._evaluate_chromosome(mask, X_train, y_train, X_val, y_val)

            # Track global best
            gen_best_idx = int(np.argmax(fitness))
            gen_best_score = fitness[gen_best_idx]
            gen_best_mask = population[gen_best_idx].copy()

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_mask = gen_best_mask

            print(f"[GA-XGB] Generation {gen+1}/{self.generations} "
                  f"- best val acc: {gen_best_score:.4f}, global best: {best_score:.4f}")

            # Selection (roulette-wheel over fitness)
            probs = fitness / fitness.sum()
            parent_indices = rng.choice(
                self.population_size,
                size=self.population_size,
                replace=True,
                p=probs,
            )
            parents = population[parent_indices]

            # Crossover
            offspring = self._crossover(parents, rng)

            # Mutation
            offspring = self._mutate(offspring, rng)

            # Ensure no all-zero children
            offspring = self._ensure_valid_population(offspring, rng)

            population = offspring

        # Save final results
        self.best_mask_ = best_mask
        self.best_score_ = best_score

        # Train final XGB on full data using selected features
        X_selected = X[:, self.best_mask_.astype(bool)]
        self.xgb_model_ = self._fit_xgb(X_selected, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X to the selected features.
        """
        if self.best_mask_ is None:
            raise RuntimeError("You must call fit() before transform().")
        return X[:, self.best_mask_.astype(bool)]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for fit + transform.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the final trained XGB model and selected features.
        """
        if self.xgb_model_ is None or self.best_mask_ is None:
            raise RuntimeError("You must call fit() before predict().")
        X_selected = self.transform(X)
        return self.xgb_model_.predict(X_selected)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the final trained XGB model.
        """
        if self.xgb_model_ is None or self.best_mask_ is None:
            raise RuntimeError("You must call fit() before predict_proba().")
        X_selected = self.transform(X)
        return self.xgb_model_.predict_proba(X_selected)

    # ---------- Internal helpers ----------

    def _ensure_valid_population(self, pop: np.ndarray, rng) -> np.ndarray:
        """
        Make sure no chromosome is all zeros; if so, randomly flip one gene to 1.
        """
        for i in range(pop.shape[0]):
            if not pop[i].any():
                idx = rng.integers(0, pop.shape[1])
                pop[i, idx] = 1
        return pop

    def _evaluate_chromosome(
        self,
        mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Train XGB on selected features and return validation accuracy.
        """
        if not mask.any():
            return 0.0

        selected = mask.astype(bool)
        X_tr = X_train[:, selected]
        X_v = X_val[:, selected]

        model = self._fit_xgb(X_tr, y_train)
        y_pred = model.predict(X_v)
        # Assuming label 1 = “good swing / up move”
        return f1_score(y_val, y_pred, pos_label=1)

    def _crossover(self, parents: np.ndarray, rng) -> np.ndarray:
        """
        Uniform crossover with probability self.crossover_rate.
        """
        pop_size, n_features = parents.shape
        offspring = parents.copy()

        for i in range(0, pop_size, 2):
            if i + 1 >= pop_size:
                break
            if rng.random() < self.crossover_rate:
                mask = rng.integers(0, 2, size=n_features, dtype=bool)
                parent1, parent2 = offspring[i].copy(), offspring[i + 1].copy()
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
                offspring[i], offspring[i + 1] = child1, child2

        return offspring

    def _mutate(self, pop: np.ndarray, rng) -> np.ndarray:
        """
        Bit-flip mutation for each gene independently.
        """
        mutation_mask = rng.random(pop.shape) < self.mutation_rate
        pop[mutation_mask] = 1 - pop[mutation_mask]
        return pop

    def _fit_xgb(self, X: np.ndarray, y: np.ndarray) -> XGBClassifier:
        """
        Train XGB with GPU when available; fall back to CPU once if GPU fails.
        """
        use_gpu = True if self._use_gpu is None else self._use_gpu
        if use_gpu:
            model = XGBClassifier(**self._xgb_params_for_mode(use_gpu=True))
            try:
                model.fit(X, y)
                self._use_gpu = True
                return model
            except XGBoostError as exc:
                if not self._is_gpu_error(exc):
                    raise
                self._use_gpu = False

        model = XGBClassifier(**self._xgb_params_for_mode(use_gpu=False))
        model.fit(X, y)
        return model

    def _xgb_params_for_mode(self, use_gpu: bool) -> Dict[str, Any]:
        params = dict(self.xgb_params)
        if use_gpu:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
            return params

        if params.get("tree_method") == "gpu_hist":
            params["tree_method"] = "hist"
        else:
            params.setdefault("tree_method", "hist")
        params.pop("predictor", None)
        params.pop("gpu_id", None)
        params.pop("device", None)
        return params

    @staticmethod
    def _is_gpu_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "gpu" in message or "cuda" in message
