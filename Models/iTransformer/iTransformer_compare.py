# iTransformer_compare.py
"""
Compare lucidrains iTransformer variants (original / 2D / FFT) on a held-out test split.

Targets:
- Forecasting-style by default: predict y[t + pred_horizon] from X[t - lookback + 1 : t + 1]
- Supports multivariate targets (y has shape (N, D)) or single target (N,) / (N,1)

Models (from lucidrains/iTransformer):
- iTransformer
- iTransformer2D
- iTransformerFFT

Notes
-----
- lucidrains models return a dict: {pred_len: Tensor[B, pred_len, num_variates]}
  This script uses pred_length=(pred_horizon,) and reads preds[pred_horizon] -> (B, pred_horizon, D)
- If your y is 1D, it is treated as D=1.
- For iTransformer2D you must set num_time_tokens that divides lookback_len (ideally).
- This is NOT the official THUML implementation; it's lucidrains' unofficial code. See repo README. 
  https://github.com/lucidrains/iTransformer

Usage example
-------------
python iTransformer_compare.py \
  --x_path /path/to/X.npy \
  --y_path /path/to/y.npy \
  --lookback_len 96 \
  --pred_horizon 1 \
  --train_frac 0.8 --val_frac 0.1 \
  --epochs 40 --batch_size 256 \
  --dim 256 --depth 6 --heads 8 --dim_head 64 \
  --use_rev_inorm 1 \
  --num_time_tokens 16

"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset (window by target-index, no leakage)
# -----------------------------
@dataclass(frozen=True)
class SplitIndex:
    train_end: int  # exclusive
    val_end: int    # exclusive


class ForecastWindowDataset(Dataset):
    """
    Windowed dataset built over full timeline, split by target index t.
    Input window ends at t and predicts y[t + pred_horizon].

    x_win: X[t - lookback + 1 : t + 1]          -> (lookback, C)
    y_fut: y[t + pred_horizon : t + pred_horizon + pred_horizon]
           -> (pred_horizon, D)

    So you can train multi-step horizons; this script uses pred_horizon=K and trains for K steps ahead.
    """

    def __init__(
        self,
        X: np.ndarray,              # (N, C)
        y: np.ndarray,              # (N, D) or (N,)
        lookback_len: int,
        pred_horizon: int,
        split: str,                 # "train" | "val" | "test"
        split_index: SplitIndex,
        device: Optional[str] = None,
    ):
        assert X.ndim == 2, "X must be (N, C)"
        if y.ndim == 1:
            y = y[:, None]
        assert y.ndim == 2, "y must be (N, D) or (N,)"
        assert X.shape[0] == y.shape[0], "X and y must align"
        assert split in {"train", "val", "test"}
        assert lookback_len >= 2
        assert pred_horizon >= 1

        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.lookback_len = lookback_len
        self.pred_horizon = pred_horizon
        self.device = device

        N = X.shape[0]

        # valid t must have enough left history AND enough future for y[t+pred_horizon ...]
        t_min = lookback_len - 1
        t_max = N - pred_horizon - 1  # so t + pred_horizon + (pred_horizon-1) <= N-1 -> t <= N - 2*pred_horizon
        # But since we define y_fut as pred_horizon steps starting at t+pred_horizon, we need:
        # t + pred_horizon + pred_horizon - 1 <= N - 1  =>  t <= N - 2*pred_horizon
        # For simplicity and typical use (pred_horizon=1), it's fine. Keep correct:
        t_max = N - 2 * pred_horizon
        if t_max < t_min:
            raise ValueError("Not enough data for the chosen lookback_len and pred_horizon.")

        if split == "train":
            t0, t1 = t_min, split_index.train_end - 1
        elif split == "val":
            t0, t1 = split_index.train_end, split_index.val_end - 1
        else:
            t0, t1 = split_index.val_end, t_max

        t0 = max(t0, t_min)
        t1 = min(t1, t_max)

        self.targets = np.arange(t0, t1 + 1, dtype=np.int64)
        if self.targets.size == 0:
            raise ValueError(f"No samples for split={split}. Check split boundaries and window sizes.")

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.targets[idx])

        x_win = self.X[t - self.lookback_len + 1 : t + 1]  # (lookback, C)
        y_start = t + self.pred_horizon
        y_fut = self.y[y_start : y_start + self.pred_horizon]  # (pred_horizon, D)

        x = torch.from_numpy(x_win)  # float32
        y = torch.from_numpy(y_fut)  # float32

        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def compute_metrics(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    """
    pred, tgt: (N, pred_horizon, D) or (N, pred_horizon, 1)
    """
    err = pred - tgt
    mse = torch.mean(err ** 2).item()
    mae = torch.mean(err.abs()).item()

    # R^2 (global)
    tgt_mean = torch.mean(tgt).item()
    ss_res = torch.sum((pred - tgt) ** 2).item()
    ss_tot = torch.sum((tgt - tgt_mean) ** 2).item()
    r2 = float("nan") if ss_tot <= 1e-12 else (1.0 - ss_res / ss_tot)

    return {"mse": mse, "mae": mae, "r2": r2}


# -----------------------------
# Training utils
# -----------------------------
class EarlyStopper:
    def __init__(self, patience: int = 6, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, val: float) -> bool:
        if self.best is None or val < self.best - self.min_delta:
            self.best = val
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def maybe_to_device(batch, device: str):
    if isinstance(batch, (tuple, list)):
        return tuple(maybe_to_device(x, device) for x in batch)
    return batch.to(device)


def extract_pred_dict_output(preds: Dict[int, torch.Tensor], pred_horizon: int) -> torch.Tensor:
    """
    lucidrains outputs: {pred_len: Tensor[B, pred_len, variate]}
    We requested pred_length=(pred_horizon,) so it should exist.
    """
    if pred_horizon not in preds:
        raise KeyError(
            f"Model did not return pred_len={pred_horizon}. Available keys: {list(preds.keys())}"
        )
    return preds[pred_horizon]


# -----------------------------
# Model factory (lucidrains)
# -----------------------------
@dataclass
class ModelConfig:
    dim: int = 256
    depth: int = 6
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.1
    num_tokens_per_variate: int = 1
    use_rev_inorm: bool = True
    # 2D only
    num_time_tokens: int = 16


def build_model(
    variant: str,
    num_variates: int,
    lookback_len: int,
    pred_horizon: int,
    cfg: ModelConfig,
):
    """
    variant: "original" | "2d" | "fft"
    """
    try:
        # lucidrains package name mirrors the repo
        from iTransformer import iTransformer, iTransformer2D, iTransformerFFT
    except Exception as e:
        raise RuntimeError(
            "Could not import lucidrains iTransformer package.\n"
            "Install with: pip install iTransformer\n"
            f"Original import error: {e}"
        )

    common = dict(
        num_variates=num_variates,
        lookback_len=lookback_len,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        pred_length=(pred_horizon,),
        dropout=cfg.dropout,
        use_reversible_instance_norm=cfg.use_rev_inorm,
    )

    if variant == "original":
        return iTransformer(
            **common,
            num_tokens_per_variate=cfg.num_tokens_per_variate,
        )
    elif variant == "2d":
        return iTransformer2D(
            **common,
            num_time_tokens=cfg.num_time_tokens,
        )
    elif variant == "fft":
        return iTransformerFFT(
            **common,
            num_tokens_per_variate=cfg.num_tokens_per_variate,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


# -----------------------------
# Train / eval loops
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    pred_horizon: int,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x, y = maybe_to_device((x, y), device)

        optimizer.zero_grad(set_to_none=True)
        out_dict = model(x)  # dict[int, tensor]
        pred = extract_pred_dict_output(out_dict, pred_horizon)  # (B, H, D)
        loss = loss_fn(pred, y)

        loss.backward()
        if clip_grad and clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    pred_horizon: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    preds_all = []
    tgts_all = []

    for x, y in loader:
        x, y = maybe_to_device((x, y), device)
        out_dict = model(x)
        pred = extract_pred_dict_output(out_dict, pred_horizon)

        loss = loss_fn(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds_all.append(pred.detach().cpu())
        tgts_all.append(y.detach().cpu())

    preds_cat = torch.cat(preds_all, dim=0) if preds_all else torch.empty(0)
    tgts_cat = torch.cat(tgts_all, dim=0) if tgts_all else torch.empty(0)

    metrics = compute_metrics(preds_cat, tgts_cat) if preds_all else {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    pred_horizon: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    clip_grad: float,
) -> Tuple[nn.Module, Dict[str, float]]:
    model = model.to(device)

    # Robust for markets; change to MSELoss if you prefer
    loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    stopper = EarlyStopper(patience=patience, min_delta=0.0)
    best_state = None
    best_val = float("inf")
    best_val_metrics = {}

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, pred_horizon, clip_grad=clip_grad
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, pred_horizon)
        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

        print(f"  epoch {ep:03d} | train_loss={tr_loss:.6f} | val={val_metrics}")

        if stopper.step(val_metrics["loss"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_metrics


# -----------------------------
# Main compare
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--x_path", type=str, required=True, help="npy file for X, shape (N, C)")
    ap.add_argument("--y_path", type=str, required=True, help="npy file for y, shape (N,) or (N, D)")

    ap.add_argument("--lookback_len", type=int, default=96)
    ap.add_argument("--pred_horizon", type=int, default=1)

    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    # training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # model params (shared across variants)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim_head", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_tokens_per_variate", type=int, default=1)
    ap.add_argument("--use_rev_inorm", type=int, default=1)

    # 2D only
    ap.add_argument("--num_time_tokens", type=int, default=16)

    # misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=0)

    # which variants
    ap.add_argument(
        "--variants",
        type=str,
        default="original,2d,fft",
        help="comma-separated: original,2d,fft",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(args.x_path)
    y = np.load(args.y_path)

    if X.ndim != 2:
        raise ValueError(f"X must be (N, C). Got shape {X.shape}")
    if y.ndim == 1:
        y = y[:, None]
    elif y.ndim != 2:
        raise ValueError(f"y must be (N,) or (N, D). Got shape {y.shape}")

    N, C = X.shape
    _, D = y.shape

    # lucidrains model predicts "num_variates" channels
    # In this compare script, we require D == num_variates.
    # If your y is a single target, set y_path to shape (N,1) and it will work.
    if D != C and D != 1:
        raise ValueError(
            f"y has D={D} targets but X has C={C} variates.\n"
            "lucidrains iTransformer predicts num_variates channels.\n"
            "Either make y have shape (N, C) for multivariate forecasting, or (N, 1) for single-target.\n"
            "If you want to predict a scalar label from features, you'd typically use a custom head (different task)."
        )

    num_variates = D  # forecast D channels

    train_end = int(N * args.train_frac)
    val_end = int(N * (args.train_frac + args.val_frac))
    split_idx = SplitIndex(train_end=train_end, val_end=val_end)

    ds_train = ForecastWindowDataset(X, y, args.lookback_len, args.pred_horizon, "train", split_idx)
    ds_val = ForecastWindowDataset(X, y, args.lookback_len, args.pred_horizon, "val", split_idx)
    ds_test = ForecastWindowDataset(X, y, args.lookback_len, args.pred_horizon, "test", split_idx)

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cfg = ModelConfig(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        dropout=args.dropout,
        num_tokens_per_variate=args.num_tokens_per_variate,
        use_rev_inorm=bool(args.use_rev_inorm),
        num_time_tokens=args.num_time_tokens,
    )

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    allowed = {"original", "2d", "fft"}
    for v in variants:
        if v not in allowed:
            raise ValueError(f"Unknown variant '{v}'. Allowed: {sorted(allowed)}")

    # sanity for 2D time tokens
    if "2d" in variants:
        if args.num_time_tokens <= 0:
            raise ValueError("--num_time_tokens must be > 0 for 2d variant")
        # lucidrains note: patch size ~ (lookback_len // num_time_tokens)
        # Not strictly required, but good to keep clean
        if args.lookback_len % args.num_time_tokens != 0:
            print(
                f"[warn] lookback_len={args.lookback_len} not divisible by num_time_tokens={args.num_time_tokens}.\n"
                "       2D variant may still run, but patching may be uneven."
            )

    print("\n=== Config ===")
    print(f"device={device}")
    print(f"N={N}, X.C={C}, y.D={D}, forecasting num_variates={num_variates}")
    print(f"lookback_len={args.lookback_len}, pred_horizon={args.pred_horizon}")
    print(f"splits: train_end={train_end}, val_end={val_end}, test_end={N}")
    print("model cfg:", asdict(cfg))
    print("variants:", variants)
    print()

    results: Dict[str, Dict[str, float]] = {}

    # train and evaluate each variant
    for variant in variants:
        print(f"\n==============================")
        print(f"Training variant: {variant}")
        print(f"==============================")

        model = build_model(
            variant=variant,
            num_variates=num_variates,
            lookback_len=args.lookback_len,
            pred_horizon=args.pred_horizon,
            cfg=cfg,
        )

        model, best_val_metrics = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            pred_horizon=args.pred_horizon,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            clip_grad=args.clip_grad,
        )

        # test
        loss_fn = nn.SmoothL1Loss()
        test_metrics = evaluate(model, test_loader, loss_fn, device, args.pred_horizon)

        print(f"\nBest VAL for {variant}: {best_val_metrics}")
        print(f"TEST for {variant}: {test_metrics}")

        results[variant] = {
            "val_loss": float(best_val_metrics.get("loss", float("nan"))),
            "val_mse": float(best_val_metrics.get("mse", float("nan"))),
            "val_mae": float(best_val_metrics.get("mae", float("nan"))),
            "val_r2": float(best_val_metrics.get("r2", float("nan"))),
            "test_loss": float(test_metrics["loss"]),
            "test_mse": float(test_metrics["mse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
        }

    # pretty print
    print("\n\n=== Summary (lower is better for loss/mse/mae; higher is better for r2) ===")
    header = ["variant", "val_loss", "test_loss", "test_mse", "test_mae", "test_r2"]
    print(" | ".join(f"{h:>10s}" for h in header))
    print("-" * (len(header) * 13))

    for v in variants:
        r = results[v]
        print(
            " | ".join(
                [
                    f"{v:>10s}",
                    f"{r['val_loss']:10.6f}",
                    f"{r['test_loss']:10.6f}",
                    f"{r['test_mse']:10.6f}",
                    f"{r['test_mae']:10.6f}",
                    f"{r['test_r2']:10.6f}" if not math.isnan(r["test_r2"]) else f"{'nan':>10s}",
                ]
            )
        )

    # pick best by test_loss
    best = min(results.items(), key=lambda kv: kv[1]["test_loss"])
    print(f"\nBest by test_loss: {best[0]}  (test_loss={best[1]['test_loss']:.6f})")


if __name__ == "__main__":
    main()
