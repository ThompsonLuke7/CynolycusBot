from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass(frozen=True)
class OracleConfig:
    max_wait_min: int = 12
    horizon_min: int = 20
    mae_weight: float = 1.5
    cost_per_trade_ret: float = 0.0002


def _flip_indices(htf_dir: pd.Series) -> np.ndarray:
    d = pd.to_numeric(htf_dir, errors="coerce").fillna(0.0)
    return np.where((d != d.shift(1)) & (d != 0.0))[0]


def _trade_score(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    direction: int,
    mae_weight: float,
    cost_ret: float,
) -> float:
    if exit_idx <= entry_idx:
        return -np.inf
    entry_px = float(close[entry_idx])
    if not np.isfinite(entry_px) or entry_px <= 0.0:
        return -np.inf

    sl = slice(entry_idx + 1, exit_idx + 1)
    rel = direction * (close[sl] / entry_px - 1.0)
    mfe = float(np.nanmax(rel)) if rel.size else 0.0

    if direction > 0:
        adverse = np.maximum(0.0, (entry_px - low[sl]) / entry_px)
    else:
        adverse = np.maximum(0.0, (high[sl] - entry_px) / entry_px)
    mae = float(np.nanmax(adverse)) if adverse.size else 0.0
    return mfe - mae_weight * mae - cost_ret


def build_oracle_entry_labels(
    df: pd.DataFrame,
    *,
    cfg: OracleConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OracleConfig()
    out = df.copy()
    n = len(out)
    out["oracle_enter"] = 0
    out["oracle_score"] = np.nan
    if n < 3:
        return out

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    htf_dir = pd.to_numeric(out["htf_dir"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    flips = _flip_indices(pd.Series(htf_dir))
    for i in flips:
        direction = int(np.sign(htf_dir[i]))
        if direction == 0:
            continue
        wait_end = min(i + int(cfg.max_wait_min), n - 2)
        horizon_end = min(i + int(cfg.horizon_min), n - 2)
        if wait_end <= i:
            continue

        next_flip_rel = np.where(np.sign(htf_dir[i + 1 : horizon_end + 1]) != direction)[0]
        if next_flip_rel.size > 0:
            exit_idx = i + 1 + int(next_flip_rel[0])
        else:
            exit_idx = horizon_end
        if exit_idx <= i:
            continue

        best_j = None
        best_s = -np.inf
        for j in range(i, wait_end + 1):
            s = _trade_score(
                close=close,
                high=high,
                low=low,
                entry_idx=j,
                exit_idx=exit_idx,
                direction=direction,
                mae_weight=float(cfg.mae_weight),
                cost_ret=float(cfg.cost_per_trade_ret),
            )
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is None:
            continue
        out.at[best_j, "oracle_enter"] = 1
        out.at[best_j, "oracle_score"] = best_s
    return out


def train_oracle_sniper(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str = "oracle_enter",
    val_frac: float = 0.15,
    random_seed: int = 7,
    save_model_path: str | Path | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    x = df[feature_cols].astype(float)
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    n = len(df)
    if n < 50 or y.sum() == 0:
        return pd.Series(np.zeros(n, dtype=float), index=df.index, name="sniper_enter_prob"), {
            "oracle_rows": float(n),
            "oracle_pos_rate": float(y.mean()) if n else 0.0,
            "oracle_auc_val": float("nan"),
        }

    split = max(1, min(n - 1, int((1.0 - float(val_frac)) * n)))
    x_tr, x_va = x.iloc[:split], x.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    dtr = xgb.DMatrix(x_tr, label=y_tr)
    dva = xgb.DMatrix(x_va, label=y_va)
    dall = xgb.DMatrix(x)
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 10,
        "lambda": 2.0,
        "alpha": 0.0,
        "seed": int(random_seed),
        "tree_method": "hist",
        "scale_pos_weight": pos_weight,
    }
    model = xgb.train(
        params,
        dtr,
        num_boost_round=400,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    p_all = model.predict(dall)
    metrics = {
        "oracle_rows": float(n),
        "oracle_pos_rate": float(y.mean()),
        "oracle_best_iteration": float(model.best_iteration if model.best_iteration is not None else -1),
        "oracle_best_score": float(model.best_score) if model.best_score is not None else float("nan"),
    }
    if save_model_path is not None:
        out_path = Path(save_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(out_path)
    return pd.Series(p_all, index=df.index, name="sniper_enter_prob"), metrics

