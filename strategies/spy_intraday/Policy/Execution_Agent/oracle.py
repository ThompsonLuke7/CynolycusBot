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


def _segment_return(
    *,
    close: np.ndarray,
    start_idx: int,
    end_idx: int,
    direction: int,
) -> float:
    if direction == 0 or end_idx <= start_idx:
        return 0.0
    p0 = float(close[start_idx])
    p1 = float(close[end_idx])
    if (not np.isfinite(p0)) or (not np.isfinite(p1)) or p0 <= 0.0:
        return 0.0
    return float(direction) * (p1 / p0 - 1.0)


def _segment_mae(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_idx: int,
    end_idx: int,
    direction: int,
) -> float:
    if direction == 0 or end_idx <= start_idx:
        return 0.0
    p0 = float(close[start_idx])
    if (not np.isfinite(p0)) or p0 <= 0.0:
        return 0.0
    sl = slice(start_idx + 1, end_idx + 1)
    if direction > 0:
        adverse = np.maximum(0.0, (p0 - low[sl]) / p0)
    else:
        adverse = np.maximum(0.0, (high[sl] - p0) / p0)
    mae = float(np.nanmax(adverse)) if adverse.size else 0.0
    return mae


def _event_timing_score(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    event_idx: int,
    exec_idx: int,
    horizon_idx: int,
    prev_dir: int,
    cur_dir: int,
    mae_weight: float,
    cost_ret: float,
) -> float:
    # ENTER: 0 -> +/-1 (delay entry)
    if prev_dir == 0 and cur_dir != 0:
        agent_gross = _segment_return(
            close=close, start_idx=exec_idx, end_idx=horizon_idx, direction=cur_dir
        )
        base_gross = _segment_return(
            close=close, start_idx=event_idx, end_idx=horizon_idx, direction=cur_dir
        )
        agent_cost = float(cost_ret)
        base_cost = float(cost_ret)
        mae = _segment_mae(
            high=high,
            low=low,
            close=close,
            start_idx=exec_idx,
            end_idx=horizon_idx,
            direction=cur_dir,
        )
        return (agent_gross - agent_cost) - (base_gross - base_cost) - float(mae_weight) * mae

    # EXIT: +/-1 -> 0 (delay flatten)
    if prev_dir != 0 and cur_dir == 0:
        agent_gross = _segment_return(
            close=close, start_idx=event_idx, end_idx=exec_idx, direction=prev_dir
        )
        base_gross = 0.0
        agent_cost = float(cost_ret)
        base_cost = float(cost_ret)
        mae = _segment_mae(
            high=high,
            low=low,
            close=close,
            start_idx=event_idx,
            end_idx=exec_idx,
            direction=prev_dir,
        )
        return (agent_gross - agent_cost) - (base_gross - base_cost) - float(mae_weight) * mae

    # SWITCH: +/-1 -> -/+1 (delay switch timing)
    if prev_dir != 0 and cur_dir != 0 and prev_dir != cur_dir:
        agent_gross = _segment_return(
            close=close, start_idx=event_idx, end_idx=exec_idx, direction=prev_dir
        )
        agent_gross += _segment_return(
            close=close, start_idx=exec_idx, end_idx=horizon_idx, direction=cur_dir
        )
        base_gross = _segment_return(
            close=close, start_idx=event_idx, end_idx=horizon_idx, direction=cur_dir
        )
        agent_cost = 2.0 * float(cost_ret)
        base_cost = 2.0 * float(cost_ret)
        mae = _segment_mae(
            high=high,
            low=low,
            close=close,
            start_idx=event_idx,
            end_idx=exec_idx,
            direction=prev_dir,
        )
        mae += _segment_mae(
            high=high,
            low=low,
            close=close,
            start_idx=exec_idx,
            end_idx=horizon_idx,
            direction=cur_dir,
        )
        return (agent_gross - agent_cost) - (base_gross - base_cost) - float(mae_weight) * mae

    return -np.inf


def build_oracle_event_labels(
    df: pd.DataFrame,
    *,
    cfg: OracleConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OracleConfig()
    out = df.copy()
    n = len(out)
    out["oracle_enter"] = 0
    out["oracle_exit"] = 0
    out["oracle_score"] = np.nan
    out["oracle_exit_score"] = np.nan
    if n < 3:
        return out

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    htf_dir = pd.to_numeric(out["htf_dir"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sign = np.sign(htf_dir).astype(int)

    for i in range(1, n - 1):
        prev_dir = int(sign[i - 1])
        cur_dir = int(sign[i])
        if cur_dir == prev_dir:
            continue

        wait_end = min(i + int(cfg.max_wait_min), n - 2)
        horizon_end = min(i + int(cfg.horizon_min), n - 2)
        if wait_end <= i:
            continue

        next_flip_rel = np.where(sign[i + 1 : horizon_end + 1] != cur_dir)[0]
        if next_flip_rel.size > 0:
            event_horizon = i + 1 + int(next_flip_rel[0])
        else:
            event_horizon = horizon_end
        if event_horizon <= i:
            continue

        best_j = None
        best_s = -np.inf
        for j in range(i, wait_end + 1):
            s = _event_timing_score(
                close=close,
                high=high,
                low=low,
                event_idx=i,
                exec_idx=j,
                horizon_idx=event_horizon,
                prev_dir=prev_dir,
                cur_dir=cur_dir,
                mae_weight=float(cfg.mae_weight),
                cost_ret=float(cfg.cost_per_trade_ret),
            )
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is None:
            continue
        if prev_dir == 0 and cur_dir != 0:
            out.at[best_j, "oracle_enter"] = 1
            out.at[best_j, "oracle_score"] = best_s
        elif prev_dir != 0 and cur_dir != prev_dir:
            out.at[best_j, "oracle_exit"] = 1
            out.at[best_j, "oracle_exit_score"] = best_s
    return out


def build_oracle_entry_labels(
    df: pd.DataFrame,
    *,
    cfg: OracleConfig | None = None,
) -> pd.DataFrame:
    # Backward-compatible wrapper.
    return build_oracle_event_labels(df, cfg=cfg)


def _build_head_window_mask(
    df: pd.DataFrame,
    *,
    label_col: str,
    event_window_max_wait: int,
) -> pd.Series:
    n = len(df)
    if n <= 0:
        return pd.Series(dtype=bool)
    if "htf_dir" not in df.columns:
        return pd.Series(np.ones(n, dtype=bool), index=df.index)
    d = pd.to_numeric(df["htf_dir"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sign = np.sign(d).astype(int)
    mask = np.zeros(n, dtype=bool)
    w = max(0, int(event_window_max_wait))
    for i in range(1, n):
        prev_dir = int(sign[i - 1])
        cur_dir = int(sign[i])
        if cur_dir == prev_dir:
            continue
        is_enter_evt = prev_dir == 0 and cur_dir != 0
        is_exit_evt = prev_dir != 0 and cur_dir != prev_dir
        include = False
        if str(label_col) == "oracle_enter":
            include = is_enter_evt
        elif str(label_col) == "oracle_exit":
            include = is_exit_evt
        else:
            include = True
        if not include:
            continue
        j0 = int(i)
        j1 = min(int(i + w), n - 1)
        if j1 >= j0:
            mask[j0 : j1 + 1] = True
    return pd.Series(mask, index=df.index)


def train_oracle_sniper(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str = "oracle_enter",
    val_frac: float = 0.15,
    random_seed: int = 7,
    save_model_path: str | Path | None = None,
    event_window_max_wait: int | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    x = df[feature_cols].astype(float)
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if event_window_max_wait is not None:
        mask = _build_head_window_mask(
            df,
            label_col=label_col,
            event_window_max_wait=int(event_window_max_wait),
        )
    else:
        mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    x_fit_all = x.loc[mask].copy()
    y_fit_all = y.loc[mask].copy()
    n = len(df)
    if n < 50 or len(x_fit_all) < 30 or y_fit_all.sum() == 0:
        return pd.Series(np.zeros(n, dtype=float), index=df.index, name="sniper_enter_prob"), {
            "oracle_rows": float(n),
            "oracle_rows_masked": float(len(x_fit_all)),
            "oracle_pos_rate": float(y_fit_all.mean()) if len(y_fit_all) else 0.0,
            "oracle_auc_val": float("nan"),
        }

    n_fit = len(x_fit_all)
    split = max(1, min(n_fit - 1, int((1.0 - float(val_frac)) * n_fit)))
    x_tr, x_va = x_fit_all.iloc[:split], x_fit_all.iloc[split:]
    y_tr, y_va = y_fit_all.iloc[:split], y_fit_all.iloc[split:]

    dtr = xgb.DMatrix(x_tr, label=y_tr)
    dva = xgb.DMatrix(x_va, label=y_va)
    dall = xgb.DMatrix(x_fit_all)
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
    p_mask = model.predict(dall)
    p_all = np.zeros(n, dtype=float)
    p_all[mask.to_numpy()] = p_mask
    metrics = {
        "oracle_rows": float(n),
        "oracle_rows_masked": float(len(x_fit_all)),
        "oracle_pos_rate": float(y_fit_all.mean()),
        "oracle_best_iteration": float(model.best_iteration if model.best_iteration is not None else -1),
        "oracle_best_score": float(model.best_score) if model.best_score is not None else float("nan"),
    }
    if save_model_path is not None:
        out_path = Path(save_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(out_path)
    return pd.Series(p_all, index=df.index, name="sniper_enter_prob"), metrics


def train_oracle_sniper_walk_forward(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str = "oracle_enter",
    n_folds: int = 5,
    initial_train_size: int | None = None,
    random_seed: int = 7,
    save_full_model_path: str | Path | None = None,
    event_window_max_wait: int | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """
    Train oracle sniper with expanding-window walk-forward OOF and a separate full-fit model.

    Returns:
        oof_probs: walk-forward predictions aligned to df index (NaN for warmup segment)
        full_probs: predictions from model fit on all rows
        metrics: summary dictionary
    """
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if int(n_folds) < 1:
        raise ValueError("n_folds must be >= 1")

    x = df[feature_cols].astype(float)
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if event_window_max_wait is not None:
        mask = _build_head_window_mask(
            df,
            label_col=label_col,
            event_window_max_wait=int(event_window_max_wait),
        )
    else:
        mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    mask_np = mask.to_numpy(dtype=bool, copy=False)
    masked_n = int(mask_np.sum())
    n = len(df)
    y_mask = y.loc[mask]
    if n < 50 or masked_n < 30 or y_mask.sum() == 0:
        zeros = pd.Series(np.zeros(n, dtype=float), index=df.index, name="sniper_enter_prob")
        return zeros.copy(), zeros.copy(), {
            "oracle_rows": float(n),
            "oracle_rows_masked": float(masked_n),
            "oracle_pos_rate": float(y_mask.mean()) if len(y_mask) else 0.0,
            "oracle_oof_missing": float(0),
            "oracle_oof_coverage": float(1.0),
            "oracle_best_iteration_full": float("nan"),
            "oracle_best_score_full": float("nan"),
        }

    if initial_train_size is None or int(initial_train_size) <= 0:
        initial_train_size = max(1, n // (int(n_folds) + 1))
    initial_train_size = int(initial_train_size)
    if initial_train_size >= n:
        raise ValueError("initial_train_size must be < number of rows.")
    remaining = n - initial_train_size
    fold_size = remaining // int(n_folds)
    if fold_size <= 0:
        raise ValueError("Fold size too small; reduce n_folds or initial_train_size.")

    def _params(y_fit: pd.Series, seed_offset: int) -> dict[str, float | int | str]:
        pos = int(y_fit.sum())
        neg = int(len(y_fit) - pos)
        if pos <= 0:
            pos_weight = 1.0
        else:
            pos_weight = float(neg / max(1, pos))
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 10,
            "lambda": 2.0,
            "alpha": 0.0,
            "seed": int(random_seed) + int(seed_offset),
            "tree_method": "hist",
            "scale_pos_weight": pos_weight,
        }

    oof = np.full(n, np.nan, dtype=np.float32)
    idx_arr = np.arange(n, dtype=np.int64)
    for fold in range(int(n_folds)):
        fold_start = initial_train_size + fold * fold_size
        fold_end = initial_train_size + (fold + 1) * fold_size
        if fold == int(n_folds) - 1:
            fold_end = n
        if fold_start >= fold_end:
            continue
        fit_idx = np.where(mask_np & (idx_arr < fold_start))[0]
        pred_idx = np.where(mask_np & (idx_arr >= fold_start) & (idx_arr < fold_end))[0]
        if pred_idx.size <= 0:
            continue
        x_fit = x.iloc[fit_idx]
        y_fit = y.iloc[fit_idx]
        x_pred = x.iloc[pred_idx]

        if len(y_fit) <= 0:
            oof[pred_idx] = 0.0
            continue
        if int(y_fit.sum()) == 0 or int(y_fit.sum()) == len(y_fit):
            const_prob = float(y_fit.mean()) if len(y_fit) else 0.0
            oof[pred_idx] = const_prob
            continue

        dtr = xgb.DMatrix(x_fit, label=y_fit)
        dte = xgb.DMatrix(x_pred)
        model = xgb.train(
            _params(y_fit, seed_offset=fold),
            dtr,
            num_boost_round=300,
            verbose_eval=False,
        )
        oof[pred_idx] = model.predict(dte).astype(np.float32)

    if int(y_mask.sum()) == 0 or int(y_mask.sum()) == len(y_mask):
        full_preds = np.zeros(n, dtype=np.float32)
        full_preds[mask_np] = float(y_mask.mean()) if len(y_mask) else 0.0
        best_iter_full = float("nan")
        best_score_full = float("nan")
    else:
        x_mask = x.loc[mask]
        y_fit_all = y.loc[mask]
        dall = xgb.DMatrix(x_mask, label=y_fit_all)
        full_model = xgb.train(
            _params(y_fit_all, seed_offset=10_000),
            dall,
            num_boost_round=300,
            verbose_eval=False,
        )
        full_preds = np.zeros(n, dtype=np.float32)
        full_preds[mask_np] = full_model.predict(xgb.DMatrix(x_mask)).astype(np.float32)
        best_iter_full = float(
            full_model.best_iteration if full_model.best_iteration is not None else -1
        )
        best_score_full = float(
            full_model.best_score if full_model.best_score is not None else float("nan")
        )
        if save_full_model_path is not None:
            out_path = Path(save_full_model_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            full_model.save_model(out_path)

    oof[~mask_np] = 0.0
    oof_s = pd.Series(oof, index=df.index, name="sniper_enter_prob_oof")
    full_s = pd.Series(full_preds, index=df.index, name="sniper_enter_prob_full")
    missing = int(oof_s.isna().sum())
    coverage = 1.0 - (float(missing) / float(max(1, n)))
    metrics = {
        "oracle_rows": float(n),
        "oracle_rows_masked": float(masked_n),
        "oracle_pos_rate": float(y_mask.mean()) if len(y_mask) else 0.0,
        "oracle_oof_missing": float(missing),
        "oracle_oof_coverage": float(coverage),
        "oracle_best_iteration_full": best_iter_full,
        "oracle_best_score_full": best_score_full,
    }
    return oof_s, full_s, metrics
