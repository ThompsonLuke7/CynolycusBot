# iTransformer_compare.py
"""
Compare lucidrains iTransformer variants (original / 2D / FFT) on a held-out test split.

Targets:
- Forecasting-style by default: predict y[t + pred_horizon] from X[t - lookback + 1 : t + 1]
- For label-mode parquet inputs, target_offset defaults to 0 (predict current label).
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
- If X has C>1 and y has D=1, a small linear projection head is added so the model
  can accept C inputs and predict a single target (D=1).
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

Parquet + exhaustion labels (long/short)
----------------------------------------
python iTransformer_compare.py \
  --x_parquet /path/to/X_15min_tree.parquet \
  --y_parquet /path/to/y_labels.parquet \
  --label_mode exhaustion \
  --sides long,short \
  --lookback_len 96 \
  --pred_horizon 1 \
  --epochs 40 --batch_size 256 \
  --dim 256 --depth 6 --heads 8 --dim_head 64

"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
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


def configure_sdpa_backend(backend: str) -> None:
    """
    Configure PyTorch SDPA backend. Use 'math' for maximum compatibility.
    """
    if not torch.cuda.is_available():
        return
    mode = (backend or "auto").strip().lower()
    if mode == "auto":
        return
    if mode not in {"math", "flash", "mem_efficient"}:
        raise ValueError("sdpa_backend must be one of: auto, math, flash, mem_efficient")
    enable_flash = mode == "flash"
    enable_mem = mode == "mem_efficient"
    enable_math = mode == "math"
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(enable_flash)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(enable_math)


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()


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
    Input window ends at t and predicts y[t + target_offset].

    x_win: X[t - lookback + 1 : t + 1]          -> (lookback, C)
    y_fut: y[t + target_offset : t + target_offset + pred_horizon]
           -> (pred_horizon, D)

    So you can train multi-step horizons; this script uses pred_horizon=K and trains for
    K steps ahead, starting at t + target_offset.
    """

    def __init__(
        self,
        X: np.ndarray,              # (N, C)
        y: np.ndarray,              # (N, D) or (N,)
        lookback_len: int,
        pred_horizon: int,
        split: str,                 # "train" | "val" | "test"
        split_index: SplitIndex,
        target_offset: Optional[int] = None,
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
        self.target_offset = pred_horizon if target_offset is None else int(target_offset)
        if self.target_offset < 0:
            raise ValueError("target_offset must be >= 0")
        self.device = device

        N = X.shape[0]

        # valid t must have enough left history AND enough future for y[t+target_offset ...]
        t_min = lookback_len - 1
        # y_start = t + target_offset
        # y_start + pred_horizon - 1 <= N - 1  =>  t <= N - pred_horizon - target_offset
        t_max = N - pred_horizon - self.target_offset
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
        y_start = t + self.target_offset
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
def compute_metrics(
    pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    pred, tgt: (N, pred_horizon, D) or (N, pred_horizon, 1)
    mask: boolean tensor with same shape; metrics computed only where mask==True
    """
    if mask is not None:
        pred = pred[mask]
        tgt = tgt[mask]
    if pred.numel() == 0:
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}

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
# Data helpers
# -----------------------------
def parse_comma_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


LABEL_MODE_COLUMNS = {
    "swing": ("long_swing_label", "short_swing_label"),
    "leg": ("leg_up_label", "leg_down_label"),
    "continuation": ("cont_strength_long", "cont_strength_short"),
    "mfe": ("mfe_up_atr", "mfe_down_atr"),
    "mae": ("mae_down_atr", "mae_up_atr"),
    "exhaustion": ("exhaustion_progress_long", "exhaustion_progress_short"),
}


def resolve_label_columns(
    label_mode: str,
    long_label_col: Optional[str],
    short_label_col: Optional[str],
) -> Tuple[str, str]:
    if (long_label_col is None) ^ (short_label_col is None):
        raise ValueError("Provide both --long_label_col and --short_label_col, or neither.")
    if long_label_col and short_label_col:
        return long_label_col, short_label_col
    mode = (label_mode or "").strip().lower()
    if mode not in LABEL_MODE_COLUMNS:
        raise ValueError(
            f"Unknown label_mode '{label_mode}'. Supported: {sorted(LABEL_MODE_COLUMNS)}"
        )
    return LABEL_MODE_COLUMNS[mode]


def load_parquet_features(x_path: str, x_cols: Optional[str]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_parquet(x_path)
    cols = parse_comma_list(x_cols)
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing X columns in {x_path}: {', '.join(missing)}")
        df = df[cols]
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        dropped = [c for c in df.columns if c not in numeric_df.columns]
        if dropped:
            preview = ", ".join(dropped[:8])
            suffix = "..." if len(dropped) > 8 else ""
            print(f"[warn] Dropping non-numeric X columns: {preview}{suffix}")
        df = numeric_df
    if df.shape[1] == 0:
        raise ValueError("No numeric X columns found in parquet.")
    return df.to_numpy(dtype=np.float32), list(df.columns)


def load_parquet_labels(
    y_path: str,
    label_mode: str,
    long_label_col: Optional[str],
    short_label_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    df = pd.read_parquet(y_path)
    long_col, short_col = resolve_label_columns(label_mode, long_label_col, short_label_col)
    missing = [c for c in (long_col, short_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing label columns in {y_path}: {', '.join(missing)}")
    y_long = df[long_col].to_numpy(dtype=np.float32)
    y_short = df[short_col].to_numpy(dtype=np.float32)
    return y_long, y_short, long_col, short_col


def _summarize_array(arr: np.ndarray) -> Dict[str, float]:
    flat = arr.reshape(-1)
    total = flat.size
    nan_count = int(np.isnan(flat).sum())
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {
            "n": total,
            "nan_pct": 100.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "zero_pct": float("nan"),
        }
    zero_pct = float(np.mean(finite == 0.0) * 100.0)
    return {
        "n": total,
        "nan_pct": float(nan_count / max(total, 1) * 100.0),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "zero_pct": zero_pct,
    }


def print_label_stats(name: str, arr: np.ndarray, fill_value: float) -> None:
    raw_stats = _summarize_array(arr)
    filled = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    filled_zero_pct = float(np.mean(filled.reshape(-1) == 0.0) * 100.0)
    print(
        f"{name}: n={raw_stats['n']}, nan%={raw_stats['nan_pct']:.2f}, "
        f"mean={raw_stats['mean']:.6f}, std={raw_stats['std']:.6f}, "
        f"min={raw_stats['min']:.6f}, max={raw_stats['max']:.6f}, "
        f"zero%={raw_stats['zero_pct']:.2f}, zero_after_fill%={filled_zero_pct:.2f}"
    )


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


def maybe_apply_output_activation(
    pred: torch.Tensor, label_mode: Optional[str], use_sigmoid: bool
) -> torch.Tensor:
    if not use_sigmoid:
        return pred
    mode = (label_mode or "").strip().lower()
    if mode in {"exhaustion", "continuation"}:
        return torch.sigmoid(pred)
    return pred


def masked_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss_fn: nn.Module,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss_el = loss_fn(pred, tgt)
    if weight is not None:
        loss_el = loss_el * weight

    if mask is not None:
        if mask.sum().item() == 0:
            return (
                torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True),
                0.0,
            )
        loss_el = loss_el[mask]
        denom = weight[mask].sum() if weight is not None else mask.sum()
    else:
        denom = weight.sum() if weight is not None else torch.tensor(loss_el.numel(), device=pred.device)

    denom_val = float(denom.item()) if torch.is_tensor(denom) else float(denom)
    if denom_val <= 0:
        return (
            torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True),
            0.0,
        )
    return loss_el.sum() / denom, denom_val


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    loss_el = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if mask is None:
        return loss_el.mean()
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
    return loss_el[mask].mean()


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target - pred
        losses = []
        for q in self.quantiles:
            losses.append(torch.maximum((q - 1) * err, q * err))
        return torch.mean(torch.stack(losses, dim=0), dim=0)


def quantile_loss_components(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    quantiles: tuple[float, ...],
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    err = tgt - pred
    out: Dict[str, float] = {}
    for q in quantiles:
        loss_el = torch.maximum((q - 1) * err, q * err)
        if weight is not None:
            loss_el = loss_el * weight
        if mask is not None:
            if mask.sum().item() == 0:
                out[f"q{int(q*100):02d}"] = float("nan")
                continue
            loss_el = loss_el[mask]
            denom = weight[mask].sum() if weight is not None else mask.sum()
        else:
            denom = weight.sum() if weight is not None else torch.tensor(
                loss_el.numel(), device=loss_el.device
            )
        denom_val = float(denom.item()) if torch.is_tensor(denom) else float(denom)
        out[f"q{int(q*100):02d}"] = (
            float((loss_el.sum() / denom).item()) if denom_val > 0 else float("nan")
        )
    return out


def extract_pred_dict_output(
    preds: Dict[int, torch.Tensor] | torch.Tensor, pred_horizon: int
) -> torch.Tensor:
    """
    lucidrains outputs:
      - dict: {pred_len: Tensor[B, pred_len, variate]}
      - or tensor directly for some versions.
    """
    if torch.is_tensor(preds):
        return preds
    if pred_horizon not in preds:
        raise KeyError(
            f"Model did not return pred_len={pred_horizon}. Available keys: {list(preds.keys())}"
        )
    return preds[pred_horizon]


def extract_pred_and_valid(
    out: Dict[int, torch.Tensor] | torch.Tensor | Tuple[object, object],
    pred_horizon: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(out, tuple) and len(out) == 2:
        value_out, valid_out = out
        pred = extract_pred_dict_output(value_out, pred_horizon)
        valid = extract_pred_dict_output(valid_out, pred_horizon)
        return pred, valid
    pred = extract_pred_dict_output(out, pred_horizon)
    return pred, None


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


class OutputProjector(nn.Module):
    """Project model outputs from input variates -> target variates."""

    def __init__(self, base: nn.Module, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.base = base
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor] | torch.Tensor:
        out = self.base(x)
        if isinstance(out, dict):
            return {k: self.proj(v) for k, v in out.items()}
        if torch.is_tensor(out):
            return self.proj(out)
        raise TypeError(f"Unexpected model output type: {type(out)}")


class TwoHeadWrapper(nn.Module):
    """Two-head outputs: validity logits + value prediction."""

    def __init__(self, base: nn.Module, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.base = base
        self.value_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.valid_proj = nn.Linear(in_dim, out_dim)

    def _apply_proj(self, out, proj):
        if isinstance(out, dict):
            return {k: proj(v) for k, v in out.items()}
        if torch.is_tensor(out):
            return proj(out)
        raise TypeError(f"Unexpected model output type: {type(out)}")

    def forward(self, x: torch.Tensor):
        out = self.base(x)
        value_out = self._apply_proj(out, self.value_proj)
        valid_out = self._apply_proj(out, self.valid_proj)
        return value_out, valid_out


def build_model(
    variant: str,
    input_variates: int,
    lookback_len: int,
    pred_horizon: int,
    cfg: ModelConfig,
    output_variates: Optional[int] = None,
    two_head_validity: bool = False,
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

    if output_variates is None:
        output_variates = input_variates

    common = dict(
        num_variates=input_variates,
        lookback_len=lookback_len,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        pred_length=(pred_horizon,),
        use_reversible_instance_norm=cfg.use_rev_inorm,
    )

    if variant == "original":
        base = iTransformer(
            **common,
            num_tokens_per_variate=cfg.num_tokens_per_variate,
        )
    elif variant == "2d":
        base = iTransformer2D(
            **common,
            num_time_tokens=cfg.num_time_tokens,
        )
    elif variant == "fft":
        base = iTransformerFFT(
            **common,
            num_tokens_per_variate=cfg.num_tokens_per_variate,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if two_head_validity:
        return TwoHeadWrapper(base, input_variates, output_variates)
    if output_variates != input_variates:
        return OutputProjector(base, input_variates, output_variates)
    return base


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
    label_mode: Optional[str],
    use_sigmoid: bool,
    use_amp: bool,
    mask_nan_y: bool,
    valid_loss_fn: Optional[nn.Module],
    valid_loss_weight: float,
    spike_threshold: Optional[float],
    spike_weight_mult: float,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x, y = maybe_to_device((x, y), device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x)
            pred, valid_logits = extract_pred_and_valid(out, pred_horizon)  # (B, H, D)
            pred = maybe_apply_output_activation(pred, label_mode, use_sigmoid)
            mask = torch.isfinite(y) if mask_nan_y else None
            weight = None
            if spike_threshold is not None and spike_weight_mult > 1.0:
                thresh = torch.tensor(spike_threshold, device=y.device, dtype=y.dtype)
                weight = torch.where(y >= thresh, spike_weight_mult, 1.0).to(y.dtype)
            loss, denom_val = masked_loss(pred, y, mask, loss_fn, weight)
            if valid_logits is not None and mask is not None and valid_loss_fn is not None:
                valid_target = mask.to(dtype=valid_logits.dtype)
                loss = loss + valid_loss_weight * masked_bce_with_logits(
                    valid_logits, valid_target, mask
                )

        loss.backward()
        if clip_grad and clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        if denom_val <= 0:
            continue
        total += loss.item() * denom_val
        n += denom_val

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    pred_horizon: int,
    label_mode: Optional[str],
    use_sigmoid: bool,
    use_amp: bool,
    mask_nan_y: bool,
    valid_loss_fn: Optional[nn.Module],
    spike_threshold: Optional[float],
    spike_weight_mult: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    valid_bce_sum = 0.0
    valid_acc_sum = 0.0
    valid_batches = 0
    q_sums = {25: 0.0, 50: 0.0, 75: 0.0}
    q_counts = {25: 0, 50: 0, 75: 0}

    preds_all = []
    tgts_all = []
    masks_all = [] if mask_nan_y else None

    for x, y in loader:
        x, y = maybe_to_device((x, y), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x)
            pred, valid_logits = extract_pred_and_valid(out, pred_horizon)
            pred = maybe_apply_output_activation(pred, label_mode, use_sigmoid)

        mask = torch.isfinite(y) if mask_nan_y else None
        weight = None
        if spike_threshold is not None and spike_weight_mult > 1.0:
            thresh = torch.tensor(spike_threshold, device=y.device, dtype=y.dtype)
            weight = torch.where(y >= thresh, spike_weight_mult, 1.0).to(y.dtype)
        loss, denom_val = masked_loss(pred, y, mask, loss_fn, weight)
        if valid_logits is not None and mask is not None and valid_loss_fn is not None:
            valid_target = mask.to(dtype=valid_logits.dtype)
            valid_bce = masked_bce_with_logits(valid_logits, valid_target, mask).item()
            valid_probs = torch.sigmoid(valid_logits)
            valid_acc = (valid_probs >= 0.5).eq(valid_target >= 0.5).float().mean().item()
            valid_bce_sum += valid_bce
            valid_acc_sum += valid_acc
            valid_batches += 1

        if denom_val <= 0:
            continue
        total_loss += loss.item() * denom_val
        n += denom_val

        if isinstance(loss_fn, QuantileLoss):
            q_vals = quantile_loss_components(
                pred, y, loss_fn.quantiles, mask=mask, weight=weight
            )
            for key, val in q_vals.items():
                q_key = int(key[1:])
                if math.isnan(val):
                    continue
                q_sums[q_key] += val
                q_counts[q_key] += 1

        preds_all.append(pred.detach().cpu())
        tgts_all.append(y.detach().cpu())
        if mask_nan_y:
            masks_all.append(mask.detach().cpu())

    preds_cat = torch.cat(preds_all, dim=0) if preds_all else torch.empty(0)
    tgts_cat = torch.cat(tgts_all, dim=0) if tgts_all else torch.empty(0)
    mask_cat = torch.cat(masks_all, dim=0) if (mask_nan_y and masks_all) else None

    metrics = (
        compute_metrics(preds_cat, tgts_cat, mask_cat)
        if preds_all
        else {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    )
    if valid_loss_fn is not None and mask_nan_y and valid_batches > 0:
        metrics["valid_bce"] = valid_bce_sum / valid_batches
        metrics["valid_acc"] = valid_acc_sum / valid_batches
    if isinstance(loss_fn, QuantileLoss):
        for q_key in (25, 50, 75):
            metrics[f"q{q_key:02d}"] = (
                q_sums[q_key] / q_counts[q_key] if q_counts[q_key] > 0 else float("nan")
            )
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
    label_mode: Optional[str],
    use_sigmoid: bool,
    use_amp: bool,
    mask_nan_y: bool,
    valid_loss_weight: float,
    spike_threshold: Optional[float],
    spike_weight_mult: float,
) -> Tuple[nn.Module, Dict[str, float]]:
    model = model.to(device)

    # Quantile loss for regression
    loss_fn = QuantileLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss(reduction="none") if mask_nan_y else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    stopper = EarlyStopper(patience=patience, min_delta=0.0)
    best_state = None
    best_val = float("inf")
    best_val_metrics = {}

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            pred_horizon,
            label_mode,
            use_sigmoid,
            use_amp,
            mask_nan_y,
            valid_loss_fn,
            valid_loss_weight,
            spike_threshold,
            spike_weight_mult,
            clip_grad=clip_grad,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            pred_horizon,
            label_mode,
            use_sigmoid,
            use_amp,
            mask_nan_y,
            valid_loss_fn,
            spike_threshold,
            spike_weight_mult,
        )
        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

        loss_desc = "QuantileLoss(q=0.25,0.50,0.75)"
        if valid_loss_fn is not None:
            loss_desc += f" + ValidBCE(w={valid_loss_weight:.2f})"
        print(
            "  epoch {:03d} | train_loss={:.6f} | val={} | loss={}".format(
                ep, tr_loss, val_metrics, loss_desc
            )
        )

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

    ap.add_argument("--x_path", type=str, default=None, help="npy file for X, shape (N, C)")
    ap.add_argument("--y_path", type=str, default=None, help="npy file for y, shape (N,) or (N, D)")
    ap.add_argument("--x_parquet", type=str, default=None, help="parquet file for X (overrides x_path)")
    ap.add_argument("--y_parquet", type=str, default=None, help="parquet file for y labels (overrides y_path)")
    ap.add_argument("--x_cols", type=str, default=None, help="comma-separated X columns for parquet")
    ap.add_argument("--label_mode", type=str, default="exhaustion", help="label mode for y_parquet")
    ap.add_argument("--sides", type=str, default="", help="comma-separated: long,short")
    ap.add_argument("--long_label_col", type=str, default=None, help="override long label column")
    ap.add_argument("--short_label_col", type=str, default=None, help="override short label column")
    ap.add_argument("--fill_nan_y", type=float, default=0.0, help="fill value for NaNs in labels")
    ap.add_argument(
        "--mask_nan_y",
        type=int,
        default=1,
        help="mask NaN labels in loss/metrics instead of filling",
    )
    ap.add_argument(
        "--two_head_validity",
        type=int,
        default=1,
        help="use two-head model (validity + value) when mask_nan_y is enabled",
    )
    ap.add_argument(
        "--valid_loss_weight",
        type=float,
        default=.5,
        help="weight for validity head BCE loss",
    )
    ap.add_argument(
        "--spike_weight_pct",
        type=float,
        default=0.9,
        help="percentile threshold for spike up-weighting (e.g., 0.9 for top 10%)",
    )
    ap.add_argument(
        "--spike_weight_mult",
        type=float,
        default=2.0,
        help="multiplier for spike samples (1.0 disables weighting)",
    )

    ap.add_argument("--lookback_len", type=int, default=96)
    ap.add_argument("--pred_horizon", type=int, default=1)
    ap.add_argument(
        "--target_offset",
        type=int,
        default=None,
        help="offset of target relative to window end (default: pred_horizon for npy; 0 for parquet)",
    )

    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    # training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # model params (shared across variants)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
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
    ap.add_argument("--use_sigmoid", type=int, default=1, help="apply sigmoid for exhaustion/continuation")
    ap.add_argument("--amp_bf16", type=int, default=1, help="use bf16 autocast on cuda")
    ap.add_argument(
        "--sdpa_backend",
        type=str,
        default="math",
        help="SDPA backend: auto, math, flash, mem_efficient",
    )

    # which variants
    ap.add_argument(
        "--variants",
        type=str,
        default="original,2d,fft",
        help="comma-separated: original,2d,fft",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train_frac must be in (0, 1)")
    if not (0.0 <= args.val_frac < 1.0):
        raise ValueError("--val_frac must be in [0, 1)")
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("--train_frac + --val_frac must be < 1 (need a test split)")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_sdpa_backend(args.sdpa_backend)

    use_parquet = args.x_parquet is not None or args.y_parquet is not None
    if use_parquet:
        if not args.x_parquet or not args.y_parquet:
            raise ValueError("Provide both --x_parquet and --y_parquet.")
        X, x_cols = load_parquet_features(args.x_parquet, args.x_cols)
        y_long_raw, y_short_raw, long_col, short_col = load_parquet_labels(
            args.y_parquet, args.label_mode, args.long_label_col, args.short_label_col
        )
        if X.shape[0] != y_long_raw.shape[0]:
            raise ValueError(
                f"X rows ({X.shape[0]}) do not match y rows ({y_long_raw.shape[0]})."
            )
        print(f"[info] Loaded X parquet with {X.shape[1]} features.")
        print_label_stats(f"[labels] long ({long_col})", y_long_raw, args.fill_nan_y)
        print_label_stats(f"[labels] short ({short_col})", y_short_raw, args.fill_nan_y)
        sides = parse_comma_list(args.sides) or ["long", "short"]
        bad_sides = [s for s in sides if s not in {"long", "short"}]
        if bad_sides:
            raise ValueError(f"Unknown sides: {bad_sides}. Use long,short.")
        target_offset = args.target_offset if args.target_offset is not None else 0
    else:
        if not args.x_path or not args.y_path:
            raise ValueError("Provide --x_path and --y_path (npy) or parquet inputs.")
        X = np.load(args.x_path)
        y_raw = np.load(args.y_path)
        if X.ndim != 2:
            raise ValueError(f"X must be (N, C). Got shape {X.shape}")
        if y_raw.ndim == 1:
            y_raw = y_raw[:, None]
        elif y_raw.ndim != 2:
            raise ValueError(f"y must be (N,) or (N, D). Got shape {y_raw.shape}")
        y_long_raw = y_raw
        y_short_raw = y_raw
        sides = ["long"]
        if parse_comma_list(args.sides) not in ([], ["long"]):
            print("[warn] --sides ignored for npy inputs; using single target.")
        target_offset = args.target_offset if args.target_offset is not None else args.pred_horizon

    if X.ndim != 2:
        raise ValueError(f"X must be (N, C). Got shape {X.shape}")

    N, C = X.shape

    train_end = int(N * args.train_frac)
    val_end = int(N * (args.train_frac + args.val_frac))
    split_idx = SplitIndex(train_end=train_end, val_end=val_end)

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
        if args.lookback_len % args.num_time_tokens != 0:
            print(
                f"[warn] lookback_len={args.lookback_len} not divisible by num_time_tokens={args.num_time_tokens}.\n"
                "       2D variant may still run, but patching may be uneven."
            )

    use_sigmoid = bool(args.use_sigmoid)
    use_amp = (
        device.startswith("cuda") and torch.cuda.is_available() and bool(args.amp_bf16)
    )
    label_mode_for_activation = args.label_mode if use_parquet else None

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for side in sides:
        y_side_raw = y_long_raw if side == "long" else y_short_raw
        if bool(args.mask_nan_y):
            y_side = y_side_raw.astype(np.float32, copy=False)
        else:
            y_side = np.nan_to_num(
                y_side_raw,
                nan=args.fill_nan_y,
                posinf=args.fill_nan_y,
                neginf=args.fill_nan_y,
            )
        if y_side.ndim == 1:
            y_side = y_side[:, None]
        elif y_side.ndim != 2:
            raise ValueError(f"y must be (N,) or (N, D). Got shape {y_side.shape}")

        D = y_side.shape[1]

        # lucidrains model predicts "num_variates" channels, which must match input variates.
        # For D=1 with C>1, we add a small projection head to map C -> 1.
        if D != C and D != 1:
            raise ValueError(
                f"y has D={D} targets but X has C={C} variates.\n"
                "lucidrains iTransformer predicts num_variates channels.\n"
                "Either make y have shape (N, C) for multivariate forecasting, or (N, 1) for single-target.\n"
                "If you want to predict a scalar label from features, you'd typically use a custom head (different task)."
            )

        input_variates = C
        output_variates = D
        if D == 1 and C > 1:
            print(
                f"[info] X has C={C} variates and y has D=1. "
                "Adding a linear projection head (C -> 1) on model outputs."
            )

        if use_sigmoid and label_mode_for_activation in {"exhaustion", "continuation"}:
            finite = y_side[np.isfinite(y_side)]
            if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
                print("[warn] labels outside [0,1] with --use_sigmoid=1")

        spike_threshold = None
        if args.spike_weight_mult > 1.0:
            if not (0.0 < args.spike_weight_pct < 1.0):
                raise ValueError("--spike_weight_pct must be in (0, 1)")
            y_train = y_side[: split_idx.train_end]
            y_train = y_train[np.isfinite(y_train)]
            if y_train.size == 0:
                print("[warn] spike weighting enabled but no finite train targets.")
            else:
                spike_threshold = float(np.quantile(y_train, args.spike_weight_pct))
                print(
                    f"[info] spike weighting: pct={args.spike_weight_pct:.2f} "
                    f"threshold={spike_threshold:.6f} mult={args.spike_weight_mult:.2f}"
                )
        else:
            y_train = y_side[: split_idx.train_end]
            y_train = y_train[np.isfinite(y_train)]

        if y_train.size > 0:
            q50 = float(np.quantile(y_train, 0.50))
            q75 = float(np.quantile(y_train, 0.75))
            print(f"[info] train quantiles: q50={q50:.6f} q75={q75:.6f}")

        ds_train = ForecastWindowDataset(
            X,
            y_side,
            args.lookback_len,
            args.pred_horizon,
            "train",
            split_idx,
            target_offset=target_offset,
        )
        ds_val = ForecastWindowDataset(
            X,
            y_side,
            args.lookback_len,
            args.pred_horizon,
            "val",
            split_idx,
            target_offset=target_offset,
        )
        ds_test = ForecastWindowDataset(
            X,
            y_side,
            args.lookback_len,
            args.pred_horizon,
            "test",
            split_idx,
            target_offset=target_offset,
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        print(f"\n=== Config (side={side}) ===")
        print(f"device={device}")
        print(f"N={N}, X.C={C}, y.D={D}, model_in={input_variates}, model_out={output_variates}")
        print(f"lookback_len={args.lookback_len}, pred_horizon={args.pred_horizon}")
        print(f"target_offset={target_offset}")
        print(f"splits: train_end={train_end}, val_end={val_end}, test_end={N}")
        if use_parquet:
            print(f"label_mode={args.label_mode}")
        print(
            "loss: QuantileLoss(q=0.25,0.50,0.75) + valid_bce"
            if (bool(args.two_head_validity) and bool(args.mask_nan_y))
            else "loss: QuantileLoss(q=0.25,0.50,0.75)"
        )
        print("model cfg:", asdict(cfg))
        print("variants:", variants)
        print()

        results[side] = {}

        # train and evaluate each variant
        for variant in variants:
            print(f"\n==============================")
            print(f"Training variant: {variant} (side={side})")
            print(f"==============================")

            use_valid_head = bool(args.two_head_validity) and bool(args.mask_nan_y)
            model = build_model(
                variant=variant,
                input_variates=input_variates,
                lookback_len=args.lookback_len,
                pred_horizon=args.pred_horizon,
                cfg=cfg,
                output_variates=output_variates,
                two_head_validity=use_valid_head,
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
                label_mode=label_mode_for_activation,
                use_sigmoid=use_sigmoid,
                use_amp=use_amp,
                mask_nan_y=bool(args.mask_nan_y),
                valid_loss_weight=args.valid_loss_weight,
                spike_threshold=spike_threshold,
                spike_weight_mult=args.spike_weight_mult,
            )

            # test
            loss_fn = QuantileLoss()
            test_metrics = evaluate(
                model,
                test_loader,
                loss_fn,
                device,
                args.pred_horizon,
                label_mode_for_activation,
                use_sigmoid,
                use_amp,
                bool(args.mask_nan_y),
                nn.BCEWithLogitsLoss(reduction="none") if use_valid_head else None,
                spike_threshold,
                args.spike_weight_mult,
            )

            print(f"\nBest VAL for {variant}: {best_val_metrics}")
            print(f"TEST for {variant}: {test_metrics}")

        results[side][variant] = {
            "val_loss": float(best_val_metrics.get("loss", float("nan"))),
            "val_mse": float(best_val_metrics.get("mse", float("nan"))),
            "val_mae": float(best_val_metrics.get("mae", float("nan"))),
            "val_r2": float(best_val_metrics.get("r2", float("nan"))),
            "val_q25": float(best_val_metrics.get("q25", float("nan"))),
            "val_q50": float(best_val_metrics.get("q50", float("nan"))),
            "val_q75": float(best_val_metrics.get("q75", float("nan"))),
            "test_loss": float(test_metrics["loss"]),
            "test_mse": float(test_metrics["mse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
            "test_q25": float(test_metrics.get("q25", float("nan"))),
            "test_q50": float(test_metrics.get("q50", float("nan"))),
            "test_q75": float(test_metrics.get("q75", float("nan"))),
        }

        # free GPU memory before next variant
        del model, loss_fn
        cleanup_cuda()

        # pretty print per side
        print(
            "\n\n=== Summary (lower is better for loss/mse/mae; higher is better for r2) "
            f"| side={side} ==="
        )
        header = [
            "variant",
            "val_loss",
            "test_loss",
            "test_mse",
            "test_mae",
            "test_r2",
            "test_q25",
            "test_q50",
            "test_q75",
        ]
        print(" | ".join(f"{h:>10s}" for h in header))
        print("-" * (len(header) * 13))

        for v in variants:
            r = results[side][v]
            print(
                " | ".join(
                    [
                        f"{v:>10s}",
                        f"{r['val_loss']:10.6f}",
                        f"{r['test_loss']:10.6f}",
                    f"{r['test_mse']:10.6f}",
                    f"{r['test_mae']:10.6f}",
                    f"{r['test_r2']:10.6f}" if not math.isnan(r["test_r2"]) else f"{'nan':>10s}",
                    f"{r['test_q25']:10.6f}" if not math.isnan(r["test_q25"]) else f"{'nan':>10s}",
                    f"{r['test_q50']:10.6f}" if not math.isnan(r["test_q50"]) else f"{'nan':>10s}",
                    f"{r['test_q75']:10.6f}" if not math.isnan(r["test_q75"]) else f"{'nan':>10s}",
                ]
            )
        )

        best = min(results[side].items(), key=lambda kv: kv[1]["test_loss"])
        print(
            f"\nBest by test_loss (side={side}): {best[0]}  "
            f"(test_loss={best[1]['test_loss']:.6f})"
        )


if __name__ == "__main__":
    main()
