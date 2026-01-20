# models/ga_xgboost/ga_xgboost.py

import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from sklearn.metrics import f1_score
import xgboost as xgb
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
      - Fitness = configurable (default: positive-class F1) with optional sparsity penalty
    """
    population_size: int = 8
    generations: int = 30
    crossover_rate: float = 0.5
    mutation_rate: float = 0.005  # per-bit mutation probability
    val_size: float = 0.15          # portion of given data used as validation
    fitness_metric: str = "f1"      # "f1" | "f1_penalized"
    feature_penalty: float = 0.001  # penalty per selected feature when using f1_penalized
    max_features: Optional[int] = 80  # hard cap; None disables
    selection: str = "tournament"   # "tournament" | "roulette"
    tournament_k: int = 3
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
    _last_printed_device: Optional[bool] = field(init=False, default=None)
    _gpu_error_printed: bool = field(init=False, default=False)

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

        # Initialize population with k-sparse masks near the cap (or a sane default).
        population = self._init_population(rng, n_features)
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

            metric_label = "val_fitness"
            if gen%10 ==0:
                print(f"[GA-XGB] Generation {gen+1}/{self.generations} "
                    f"- best {metric_label}: {gen_best_score:.4f}, global best: {best_score:.4f}")

            # Selection
            if self.selection == "roulette":
                total = float(fitness.sum())
                if total <= 0 or not np.isfinite(total):
                    # fallback to uniform if degenerate
                    probs = np.ones_like(fitness) / fitness.size
                else:
                    probs = fitness / total
                parent_indices = rng.choice(
                    self.population_size,
                    size=self.population_size,
                    replace=True,
                    p=probs,
                )
                parents = population[parent_indices]
            else:
                # tournament selection (more stable than roulette for sparse/flat fitness)
                parents = np.empty_like(population)
                for i in range(self.population_size):
                    cand = rng.choice(
                        self.population_size, size=self.tournament_k, replace=False
                    )
                    best = cand[int(np.argmax(fitness[cand]))]
                    parents[i] = population[best]

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
        Enforce max_features (if set) and avoid all-zero masks.
        """
        for i in range(pop.shape[0]):
            pop[i] = self._cap_mask(pop[i], rng)
        return pop

    def _init_population(self, rng, n_features: int) -> np.ndarray:
        """
        Initialize a k-sparse population centered near the max feature cap.
        """
        pop = np.zeros((self.population_size, n_features), dtype=np.int8)
        if self.max_features is None:
            k0 = min(60, n_features)
        else:
            k0 = min(self.max_features, 60, n_features)

        k_low = max(1, int(math.floor(0.5 * k0)))
        k_high = max(k_low, int(math.ceil(1.2 * k0)))
        k_high = min(k_high, n_features)

        for i in range(self.population_size):
            k = int(rng.integers(k_low, k_high + 1))
            idx = rng.choice(n_features, size=k, replace=False)
            pop[i, idx] = 1
        return pop

    def _cap_mask(self, mask: np.ndarray, rng) -> np.ndarray:
        """
        Enforce max_features and ensure at least one feature is selected.
        """
        if self.max_features is not None:
            ones = np.flatnonzero(mask)
            if ones.size > self.max_features:
                drop = rng.choice(ones, size=ones.size - self.max_features, replace=False)
                mask[drop] = 0

        if not mask.any():
            k = 1
            if self.max_features is not None:
                k = max(1, min(3, self.max_features, mask.size))
            idx = rng.choice(mask.size, size=k, replace=False)
            mask[idx] = 1
        return mask

    def _evaluate_chromosome(
        self,
        mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train XGB on selected features and return validation fitness."""
        if not mask.any():
            return 0.0

        selected = mask.astype(bool)
        n_selected = int(selected.sum())
        if self.max_features is not None and n_selected > self.max_features:
            return 0.0
        X_tr = X_train[:, selected]
        X_v = X_val[:, selected]

        model = self._fit_xgb(X_tr, y_train)
        y_pred = model.predict(X_v)
        base = f1_score(y_val, y_pred, pos_label=1)

        if self.fitness_metric == "f1_penalized":
            return float(base - self.feature_penalty * n_selected)
        return float(base)

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
        if self.mutation_rate <= 0:
            return pop
        mutation_mask = rng.random(pop.shape) < self.mutation_rate
        pop[mutation_mask] = 1 - pop[mutation_mask]
        return pop

    def _fit_xgb(self, X: np.ndarray, y: np.ndarray) -> XGBClassifier:
        """
        Train XGB with GPU when available; fall back to CPU once if GPU fails.
        """
        use_gpu = True if self._use_gpu is None else self._use_gpu
        if use_gpu:
            gpu_params = self._xgb_params_for_mode(use_gpu=True)
            model = XGBClassifier(**gpu_params)
            try:
                model.fit(X, y)
                self._use_gpu = True
                self._maybe_print_device(use_gpu=True, params=gpu_params)
                return model
            except XGBoostError as exc:
                if not self._is_gpu_error(exc):
                    raise
                self._maybe_print_gpu_error(exc)
                self._use_gpu = False

        cpu_params = self._xgb_params_for_mode(use_gpu=False)
        model = XGBClassifier(**cpu_params)
        model.fit(X, y)
        self._maybe_print_device(use_gpu=False, params=cpu_params)
        return model

    def _xgb_params_for_mode(self, use_gpu: bool) -> Dict[str, Any]:
        params = dict(self.xgb_params)
        if use_gpu:
            if self._xgb_supports_device_param():
                gpu_id = params.pop("gpu_id", None)
                params.pop("predictor", None)
                params["tree_method"] = "hist"
                if gpu_id is None:
                    params["device"] = "cuda"
                else:
                    params["device"] = f"cuda:{gpu_id}"
            else:
                params["tree_method"] = "gpu_hist"
                params["predictor"] = "gpu_predictor"
                params.pop("device", None)
            return params

        if params.get("tree_method") == "gpu_hist":
            params["tree_method"] = "hist"
        else:
            params.setdefault("tree_method", "hist")
        params.pop("predictor", None)
        params.pop("gpu_id", None)
        if self._xgb_supports_device_param():
            params["device"] = "cpu"
        else:
            params.pop("device", None)
        return params

    @staticmethod
    def _is_gpu_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "gpu" in message or "cuda" in message

    @staticmethod
    def _xgb_supports_device_param() -> bool:
        version = getattr(xgb, "__version__", "0")
        try:
            major = int(version.split(".", 1)[0])
        except ValueError:
            return False
        return major >= 2

    def _maybe_print_device(self, use_gpu: bool, params: Dict[str, Any]) -> None:
        if self._last_printed_device is not None and self._last_printed_device == use_gpu:
            return
        self._last_printed_device = use_gpu

        if use_gpu:
            device_param = params.get("device")
            if isinstance(device_param, str) and device_param.startswith("cuda"):
                device_label = device_param
            else:
                gpu_id = params.get("gpu_id", self.xgb_params.get("gpu_id", 0))
                device_label = f"cuda:{gpu_id}"
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                print(f"[GA-XGB] Using GPU {device_label} (CUDA_VISIBLE_DEVICES={cuda_visible})")
            else:
                print(f"[GA-XGB] Using GPU {device_label}")
        else:
            print("[GA-XGB] Using CPU")

    def _maybe_print_gpu_error(self, exc: Exception) -> None:
        if self._gpu_error_printed:
            return
        self._gpu_error_printed = True
        message = str(exc).splitlines()[0].strip()
        if message:
            print(f"[GA-XGB] GPU init failed, falling back to CPU: {message}")
        else:
            print("[GA-XGB] GPU init failed, falling back to CPU")
