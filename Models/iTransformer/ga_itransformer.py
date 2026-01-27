# ga_itransformer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from itransformer_dataset import SplitIndex, WindowedTimeSeries
from itransformer_model import iTransformerEncoder


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    task: str,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        if task == "binary":
            yb = y.view(-1, 1)
            loss = loss_fn(out, yb)
            probs = torch.sigmoid(out)
            preds = (probs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()
        else:
            yt = y.view_as(out)
            loss = loss_fn(out, yt)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        n += bs

    metrics = {"loss": total_loss / max(n, 1)}
    if task == "binary":
        metrics["acc"] = correct / max(total, 1)
    return metrics


@dataclass
class GAITransformerFeatureSelector:
    population_size: int = 8
    generations: int = 12
    crossover_rate: float = 0.5
    mutation_rate: float = 0.01
    max_features: Optional[int] = 80
    selection: str = "tournament"
    tournament_k: int = 3
    random_state: Optional[int] = 42
    fitness_metric: str = "neg_val_loss"  # "neg_val_loss" | "acc"
    feature_penalty: float = 0.0
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    use_var_embedding: bool = False
    batch_size: int = 256
    epochs: int = 6
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    clip: float = 1.0
    use_cuda: bool = True
    output_activation: Optional[str] = None
    _best_mask: Optional[np.ndarray] = field(init=False, default=None)
    _best_score: Optional[float] = field(init=False, default=None)

    @property
    def best_mask_(self) -> Optional[np.ndarray]:
        return self._best_mask

    @property
    def best_score_(self) -> Optional[float]:
        return self._best_score

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        task: str,
    ) -> "GAITransformerFeatureSelector":
        rng = np.random.default_rng(self.random_state)
        set_seed(self.random_state or 0)

        n_features = X_train.shape[1]
        population = self._init_population(rng, n_features)
        population = self._ensure_valid_population(population, rng)

        best_mask = None
        best_score = -np.inf

        for gen in range(self.generations):
            fitness = np.zeros(self.population_size, dtype=float)
            for i, mask in enumerate(population):
                fitness[i] = self._evaluate_chromosome(
                    mask, X_train, y_train, X_val, y_val, task=task
                )

            gen_best_idx = int(np.argmax(fitness))
            gen_best_score = float(fitness[gen_best_idx])
            gen_best_mask = population[gen_best_idx].copy()
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_mask = gen_best_mask

            if gen % 5 == 0:
                print(
                    f"[GA-iTransformer] Generation {gen+1}/{self.generations} "
                    f"- best fitness: {gen_best_score:.4f}, global best: {best_score:.4f}"
                )

            parents = self._select_parents(population, fitness, rng)
            offspring = self._crossover(parents, rng)
            offspring = self._mutate(offspring, rng)
            offspring = self._ensure_valid_population(offspring, rng)
            population = offspring

        self._best_mask = best_mask
        self._best_score = best_score
        return self

    # ---------- Internal helpers ----------

    def _device(self) -> torch.device:
        if self.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _init_population(self, rng, n_features: int) -> np.ndarray:
        pop = np.zeros((self.population_size, n_features), dtype=np.int8)
        k0 = min(60, n_features) if self.max_features is None else min(self.max_features, 60, n_features)
        k_low = max(1, int(np.floor(0.5 * k0)))
        k_high = max(k_low, int(np.ceil(1.2 * k0)))
        k_high = min(k_high, n_features)
        for i in range(self.population_size):
            k = int(rng.integers(k_low, k_high + 1))
            idx = rng.choice(n_features, size=k, replace=False)
            pop[i, idx] = 1
        return pop

    def _cap_mask(self, mask: np.ndarray, rng) -> np.ndarray:
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

    def _ensure_valid_population(self, pop: np.ndarray, rng) -> np.ndarray:
        for i in range(pop.shape[0]):
            pop[i] = self._cap_mask(pop[i], rng)
        return pop

    def _select_parents(self, population: np.ndarray, fitness: np.ndarray, rng) -> np.ndarray:
        if self.selection == "roulette":
            total = float(fitness.sum())
            if total <= 0 or not np.isfinite(total):
                probs = np.ones_like(fitness) / fitness.size
            else:
                probs = fitness / total
            parent_indices = rng.choice(
                self.population_size, size=self.population_size, replace=True, p=probs
            )
            return population[parent_indices]

        parents = np.empty_like(population)
        for i in range(self.population_size):
            cand = rng.choice(self.population_size, size=self.tournament_k, replace=False)
            best = cand[int(np.argmax(fitness[cand]))]
            parents[i] = population[best]
        return parents

    def _crossover(self, parents: np.ndarray, rng) -> np.ndarray:
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
        if self.mutation_rate <= 0:
            return pop
        mutation_mask = rng.random(pop.shape) < self.mutation_rate
        pop[mutation_mask] = 1 - pop[mutation_mask]
        return pop

    def _evaluate_chromosome(
        self,
        mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        task: str,
    ) -> float:
        if not mask.any():
            return -np.inf
        selected = mask.astype(bool)
        n_selected = int(selected.sum())
        if self.max_features is not None and n_selected > self.max_features:
            return -np.inf

        X_tr = X_train[:, selected]
        X_v = X_val[:, selected]
        if len(X_tr) < self.seq_len or len(X_v) < self.seq_len:
            return -np.inf

        # combine to allow val windows to use prior train history
        X_combined = np.concatenate([X_tr, X_v], axis=0)
        y_combined = np.concatenate([y_train, y_val], axis=0)
        split_idx = SplitIndex(train_end=len(X_tr), val_end=len(X_tr) + len(X_v))

        # normalize targets for regression using train stats only
        if task == "regression":
            y_mu = y_train.mean(axis=0, keepdims=True)
            y_std = y_train.std(axis=0, keepdims=True) + 1e-8
            y_combined = (y_combined - y_mu) / y_std

        ds_train = WindowedTimeSeries(X_combined, y_combined, self.seq_len, "train", split_idx)
        ds_val = WindowedTimeSeries(X_combined, y_combined, self.seq_len, "val", split_idx)
        if len(ds_train) == 0 or len(ds_val) == 0:
            return -np.inf

        device = self._device()
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

        out_dim = y_train.shape[1]
        model = iTransformerEncoder(
            seq_len=self.seq_len,
            num_variates=X_tr.shape[1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            use_var_embedding=self.use_var_embedding,
            out_dim=out_dim,
            output_activation=self.output_activation,
        ).to(device)

        if task == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.SmoothL1Loss()

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        model.train()
        for _ in range(self.epochs):
            for x, yb, _ in train_loader:
                x = x.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                if task == "binary":
                    target = yb.view(-1, 1)
                    loss = loss_fn(out, target)
                else:
                    target = yb.view_as(out)
                    loss = loss_fn(out, target)
                loss.backward()
                if self.clip is not None and self.clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                opt.step()

        metrics = _evaluate(model, val_loader, loss_fn, task, device)
        if task == "binary" and self.fitness_metric == "acc":
            fitness = float(metrics.get("acc", 0.0))
        else:
            fitness = -float(metrics["loss"])

        if self.feature_penalty:
            fitness -= float(self.feature_penalty) * n_selected

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return fitness
