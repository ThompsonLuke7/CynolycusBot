from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.retrieve_data import normalize_ticker
from Features.feature_matrix_regime import (
    AgentFeatureConfig,
    _add_pivot_features,
    _add_probability_confidence_features,
    build_agent_feature_matrix,
)
from Features.label_generations import (
    _load_long_short_thresholds_from_summary,
    add_trend_phase_labels,
    build_meta_entry_labels,
    build_meta_exit_labels,
)
from Models.ga_xgboost.ga_xgboost import GAXGBoostFeatureSelector


DART_DEFAULTS: dict[str, object] = {
    "rate_drop": 0.1,
    "skip_drop": 0.5,
    "one_drop": 1,
    "sample_type": "uniform",
    "normalize_type": "tree",
}
XGB_VERBOSE_EVAL_EVERY = 100

ENTRY_TARGETS = ("y_enter_long", "y_enter_short")
EXIT_TARGETS = ("y_exit_long", "y_exit_short")


@dataclass(frozen=True)
class PipelineConfig:
    ticker: str = "SPY"
    dataset_name: str = "10min"
    processed_root: str | None = None
    ga_model_root: str | None = None
    model_root: str | None = None
    pivot_label_dir: str = "swing"
    tb_label_dir: str = "tb"
    include_tb_probs: bool = True
    include_vix_features: bool = True
    session_tz: str = "America/New_York"
    atr_col: str = "atr"
    a_tp: float = 1.6
    b_sl: float = 0.8
    cost_bps: float = 2.0
    use_next_open: bool = True
    allow_cross_day: bool = True
    entry_mode: str = "tp"
    hazard_k: int = 1
    n_folds: int = 5
    initial_train_days: int = 0
    purge_days: int = 0
    threshold_min: float = 0.50
    threshold_max: float = 0.95
    threshold_step: float = 0.05
    threshold_objective: str = "f0_5"
    min_oos_prob_coverage: float = 0.95
    drop_high_corr_features: bool = True
    high_corr_threshold: float = 0.95
    xgb_booster: str | None = None
    xgb_rate_drop: float | None = None
    xgb_skip_drop: float | None = None
    xgb_one_drop: bool | None = None
    xgb_sample_type: str | None = None
    xgb_normalize_type: str | None = None
    n_estimators: int | None = None
    early_stopping_rounds: int | None = 200
    early_stopping_val_fraction: float = 0.20
    early_stopping_min_val_rows: int = 100
    random_state: int = 42


@dataclass(frozen=True)
class TrainResult:
    target_col: str
    model: xgb.Booster | None
    constant_prob: float | None
    params: dict[str, Any]
    num_boost_round: int
    valid_mask: np.ndarray
    oof_probs: np.ndarray
    full_probs: np.ndarray
    coverage: float
    fold_rows: list[dict[str, int]]
    eval_history: dict[str, list[float]] | None = None


def derive_session_date(index: pd.Index, *, tz: str) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Feature matrix must use DatetimeIndex for session derivation.")
    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert(tz)
    return idx.normalize().date


def resolve_ga_model_root(cfg: PipelineConfig) -> Path:
    if cfg.ga_model_root:
        return Path(cfg.ga_model_root)
    return REPO_ROOT / "Data" / "models" / "ga_xgboost" / cfg.dataset_name


def resolve_meta_dataset_root(cfg: PipelineConfig) -> Path:
    if cfg.model_root:
        return Path(cfg.model_root) / cfg.dataset_name
    return REPO_ROOT / "Data" / "models" / "meta_xgboost" / cfg.dataset_name


def threshold_grid(cfg: PipelineConfig) -> np.ndarray:
    lo = float(cfg.threshold_min)
    hi = float(cfg.threshold_max)
    step = float(cfg.threshold_step)
    if step <= 0.0:
        raise ValueError("threshold_step must be > 0")
    if hi < lo:
        raise ValueError("threshold_max must be >= threshold_min")
    steps = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return np.round(lo + np.arange(steps, dtype=float) * step, 6)


def ga_summary_path(cfg: PipelineConfig) -> Path:
    return resolve_ga_model_root(cfg) / "training_run_summary.json"


def _ga_prob_parquet_path(
    *,
    model_root: Path,
    side: str,
    label_dir: str,
    prefix: str,
) -> Path:
    side_root = model_root / side.lower()
    probe_dirs = [
        side_root / label_dir,
        side_root / "probs" / label_dir,
        side_root,
        side_root / "probs",
    ]
    for base in probe_dirs:
        path = base / f"{prefix}_probs.parquet"
        if path.exists():
            return path
    searched = ", ".join(str(base / f"{prefix}_probs.parquet") for base in probe_dirs)
    raise FileNotFoundError(f"Missing GA probability parquet for {side}/{label_dir}: {searched}")


def _load_ga_oos_prob_series(
    *,
    model_root: Path,
    side: str,
    label_dir: str,
    prefix: str,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    path = _ga_prob_parquet_path(
        model_root=model_root,
        side=side,
        label_dir=label_dir,
        prefix=prefix,
    )
    df = pd.read_parquet(path)
    oof_col = f"{prefix}_oof_train"
    test_col = f"{prefix}_test"
    if oof_col not in df.columns or test_col not in df.columns:
        raise KeyError(f"{path} must contain {oof_col} and {test_col}")
    oos = pd.to_numeric(df[oof_col], errors="coerce").combine_first(
        pd.to_numeric(df[test_col], errors="coerce")
    )
    aligned = oos.reindex(target_index)
    if aligned.notna().any():
        return aligned
    # Backward compatibility: older GA probability artifacts may be saved with
    # RangeIndex instead of DatetimeIndex; align by position when lengths match.
    if len(oos) == len(target_index):
        return pd.Series(oos.to_numpy(dtype=float), index=target_index, name=oos.name)
    return aligned


def _replace_meta_prob_features_with_oos(
    feat_df: pd.DataFrame,
    *,
    model_root: Path,
    pivot_label_dir: str,
    tb_label_dir: str,
    include_tb_probs: bool,
    min_coverage: float,
) -> pd.DataFrame:
    out = feat_df.copy()
    target_index = out.index
    replacements = {
        "p_pivot_long": _load_ga_oos_prob_series(
            model_root=model_root,
            side="long",
            label_dir=pivot_label_dir,
            prefix="p_long",
            target_index=target_index,
        ),
        "p_pivot_short": _load_ga_oos_prob_series(
            model_root=model_root,
            side="short",
            label_dir=pivot_label_dir,
            prefix="p_short",
            target_index=target_index,
        ),
    }
    if include_tb_probs:
        replacements.update(
            {
                "p_tb_long": _load_ga_oos_prob_series(
                    model_root=model_root,
                    side="long",
                    label_dir=tb_label_dir,
                    prefix="p_long",
                    target_index=target_index,
                ),
                "p_tb_short": _load_ga_oos_prob_series(
                    model_root=model_root,
                    side="short",
                    label_dir=tb_label_dir,
                    prefix="p_short",
                    target_index=target_index,
                ),
            }
        )
    def _fmt_index_value(idx: pd.Index, pos: int | None) -> str:
        if pos is None or pos < 0 or pos >= len(idx):
            return "None"
        return f"{pos}:{idx[pos]}"

    def _coverage_diag(series: pd.Series) -> dict[str, str | float | int]:
        present = series.notna().to_numpy(dtype=bool)
        coverage = float(present.mean()) if present.size else 1.0
        valid_idx = np.flatnonzero(present)
        missing_idx = np.flatnonzero(~present)
        first_valid = int(valid_idx[0]) if valid_idx.size else None
        first_missing = int(missing_idx[0]) if missing_idx.size else None
        last_missing = int(missing_idx[-1]) if missing_idx.size else None
        missing_segments = 0
        if missing_idx.size:
            missing_segments = int(np.sum(np.diff(missing_idx) > 1) + 1)
        front_warmup_only = bool(missing_idx.size and np.all(~present[: int(valid_idx[0])] if valid_idx.size else ~present))
        if missing_idx.size == 0:
            gap_type = "none"
        elif valid_idx.size == 0:
            gap_type = "all_missing"
        elif front_warmup_only and np.all(present[int(valid_idx[0]) :]):
            gap_type = "front_warmup_only"
        else:
            gap_type = "scattered"
        return {
            "coverage": coverage,
            "first_valid": _fmt_index_value(target_index, first_valid),
            "first_missing": _fmt_index_value(target_index, first_missing),
            "last_missing": _fmt_index_value(target_index, last_missing),
            "missing_segments": missing_segments,
            "gap_type": gap_type,
        }

    coverage_by_col: dict[str, float] = {
        col: (float(series.notna().mean()) if len(series) else 1.0)
        for col, series in replacements.items()
    }
    diag_by_col = {col: _coverage_diag(series) for col, series in replacements.items()}
    low_coverage = {
        col: cov for col, cov in coverage_by_col.items() if cov < float(min_coverage)
    }
    if low_coverage:
        details = ", ".join(
            (
                f"{col}={diag_by_col[col]['coverage']:.2%} "
                f"(first_valid={diag_by_col[col]['first_valid']}, "
                f"first_missing={diag_by_col[col]['first_missing']}, "
                f"last_missing={diag_by_col[col]['last_missing']}, "
                f"segments={diag_by_col[col]['missing_segments']}, "
                f"gap_type={diag_by_col[col]['gap_type']})"
            )
            for col in sorted(diag_by_col)
        )
        failing = ", ".join(
            (
                f"{col}={diag_by_col[col]['coverage']:.2%} "
                f"(gap_type={diag_by_col[col]['gap_type']}, "
                f"segments={diag_by_col[col]['missing_segments']})"
            )
            for col in sorted(low_coverage)
        )
        raise ValueError(
            "OOS probability coverage below threshold after reindex. "
            f"threshold={float(min_coverage):.2%}; failing: {failing}; all: {details}"
        )
    for col, series in replacements.items():
        out[col] = series
        out = _add_pivot_features(out, col)
    out = _add_probability_confidence_features(out)
    return out


def build_base_feature_frame(cfg: PipelineConfig) -> pd.DataFrame:
    clean = normalize_ticker(cfg.ticker)
    processed_root = Path(cfg.processed_root) if cfg.processed_root else None
    ga_model_root = resolve_ga_model_root(cfg)
    agent_cfg = AgentFeatureConfig(
        ticker=clean,
        dataset_name=cfg.dataset_name,
        model_name="ga_xgboost",
        processed_root=processed_root,
        model_root=ga_model_root,
        pivot_label_dir=cfg.pivot_label_dir,
        tb_label_dir=cfg.tb_label_dir,
        include_pivot_probs=True,
        include_tb_probs=bool(cfg.include_tb_probs),
        include_state_placeholders=False,
        include_vix_features=bool(cfg.include_vix_features),
        drop_na=False,
        tz=cfg.session_tz,
    )
    feat_df = build_agent_feature_matrix(config=agent_cfg)
    feat_df = _replace_meta_prob_features_with_oos(
        feat_df,
        model_root=ga_model_root,
        pivot_label_dir=cfg.pivot_label_dir,
        tb_label_dir=cfg.tb_label_dir,
        include_tb_probs=bool(cfg.include_tb_probs),
        min_coverage=float(cfg.min_oos_prob_coverage),
    )
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(feat_df.columns):
        raise KeyError(f"Agent feature matrix missing required OHLC columns: {sorted(needed - set(feat_df.columns))}")
    feat_df = feat_df.copy()
    if cfg.atr_col not in feat_df.columns:
        feat_df[cfg.atr_col] = ta.atr(feat_df["high"], feat_df["low"], feat_df["close"], length=14)
    feat_df = add_trend_phase_labels(
        feat_df,
        close_col="close",
        momentum_col="trend_phase_m",
        accel_col="trend_phase_a",
        phase_col="trend_phase_label",
        ignition_col="trend_phase_ignition",
        expansion_col="trend_phase_expansion",
        saturation_col="trend_phase_saturation",
        decay_col="trend_phase_decay",
        exit_long_col="trend_phase_exit_long",
        exit_short_col="trend_phase_exit_short",
        write_phase_columns=True,
        use_hazard_exit_labels=False,
    )
    # trend_phase_ret duplicates ret_1 for the current meta frame, so keep the simpler base return feature.
    if "trend_phase_ret" in feat_df.columns:
        feat_df = feat_df.drop(columns=["trend_phase_ret"])
    feat_df["session_date"] = derive_session_date(feat_df.index, tz=cfg.session_tz)
    return feat_df


def add_entry_targets(frame: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    label_cols = [c for c in {
        "open", "high", "low", "close", cfg.atr_col, "session_date",
        "p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short",
        "tb_long_label", "tb_short_label",
        "trend_phase_label", "trend_phase_ignition", "trend_phase_expansion",
        "trend_phase_m", "trend_phase_a",
    } if c in frame.columns]
    labels = build_meta_entry_labels(
        frame[label_cols].copy(),
        atr_col=cfg.atr_col,
        a_tp=cfg.a_tp,
        b_sl=cfg.b_sl,
        use_next_open=cfg.use_next_open,
        cost_bps=cfg.cost_bps,
        day_col="session_date",
        allow_cross_day=cfg.allow_cross_day,
        entry_mode=cfg.entry_mode,
        thresholds_summary_path=ga_summary_path(cfg),
        use_summary_thresholds=True,
    )
    out = frame.copy()
    for col in ENTRY_TARGETS:
        out[col] = labels[col].to_numpy(dtype=np.int8)
    return out


def _session_spans(frame: pd.DataFrame, *, allow_cross_day: bool) -> list[tuple[int, int]]:
    n = len(frame)
    if n == 0:
        return []
    if bool(allow_cross_day):
        return [(0, n)]
    if "session_date" not in frame.columns:
        raise KeyError("session_date is required when allow_cross_day=False")
    day_vals = pd.Series(frame["session_date"]).astype("string").to_numpy()
    boundaries = np.flatnonzero(day_vals[1:] != day_vals[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def compute_entry_embargo_end_idx(
    frame: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    target_col: str,
) -> np.ndarray:
    side = "long" if str(target_col).endswith("long") else "short"
    mode = str(cfg.entry_mode).strip().lower()
    n = len(frame)
    out = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return out

    if mode == "tp":
        open_px = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
        high_px = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
        low_px = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
        atr = pd.to_numeric(frame[cfg.atr_col], errors="coerce").to_numpy(dtype=float)
        entry_offset = 1 if bool(cfg.use_next_open) else 0
        tp_mult = float(cfg.a_tp)
        sl_mult = float(cfg.b_sl)

        for s, e in _session_spans(frame, allow_cross_day=cfg.allow_cross_day):
            if (e - s) <= entry_offset:
                continue
            for i in range(s, e):
                entry_idx = i + entry_offset
                if entry_idx >= e:
                    break
                entry = open_px[entry_idx]
                atr_i = atr[i]
                if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0.0:
                    continue
                tp_dist = tp_mult * atr_i
                sl_dist = sl_mult * atr_i
                if tp_dist <= 0.0 or sl_dist <= 0.0:
                    continue
                if side == "long":
                    tp = entry + tp_dist
                    sl = entry - sl_dist
                    for j in range(entry_idx + 1, e):
                        if high_px[j] >= tp or low_px[j] <= sl:
                            out[i] = int(j)
                            break
                    else:
                        out[i] = int(e - 1)
                else:
                    tp = entry - tp_dist
                    sl = entry + sl_dist
                    for j in range(entry_idx + 1, e):
                        if low_px[j] <= tp or high_px[j] >= sl:
                            out[i] = int(j)
                            break
                    else:
                        out[i] = int(e - 1)
        return out

    if mode != "phase":
        raise ValueError(f"Unsupported entry_mode for embargo computation: {cfg.entry_mode}")

    summary_long_thr, summary_short_thr, summary_mode = _load_long_short_thresholds_from_summary(
        ga_summary_path(cfg)
    )
    pivot_long_thr = 0.55
    pivot_short_thr = 0.55
    tb_long_thr = 0.50
    tb_short_thr = 0.50
    if summary_long_thr is not None and summary_short_thr is not None:
        if summary_mode in {"pivot", "pivots"}:
            pivot_long_thr = float(summary_long_thr)
            pivot_short_thr = float(summary_short_thr)
        elif summary_mode in {"tb", "triple", "triple_barrier"}:
            tb_long_thr = float(summary_long_thr)
            tb_short_thr = float(summary_short_thr)

    m_np = pd.to_numeric(frame.get("trend_phase_m"), errors="coerce").to_numpy(dtype=float)
    a_np = pd.to_numeric(frame.get("trend_phase_a"), errors="coerce").to_numpy(dtype=float)
    ignition = pd.to_numeric(frame.get("trend_phase_ignition", 0), errors="coerce").fillna(0).astype(int).to_numpy() == 1
    expansion = pd.to_numeric(frame.get("trend_phase_expansion", 0), errors="coerce").fillna(0).astype(int).to_numpy() == 1
    finite_a = np.isfinite(a_np)
    short_zero_band = float(np.nanquantile(np.abs(a_np[finite_a]), 0.35)) if np.any(finite_a) else 0.0

    def _prob_gate(prob_col: str, thr: float) -> np.ndarray:
        if prob_col not in frame.columns:
            return np.ones(n, dtype=bool)
        p = pd.to_numeric(frame[prob_col], errors="coerce").to_numpy(dtype=float)
        return np.isfinite(p) & (p >= float(thr))

    phase_long = ignition | expansion
    phase_short = np.isfinite(m_np) & np.isfinite(a_np) & (m_np < 0.0) & (
        (a_np < 0.0) | (np.abs(a_np) <= short_zero_band)
    )
    candidate_long = phase_long & _prob_gate("p_pivot_long", pivot_long_thr) & _prob_gate("p_tb_long", tb_long_thr)
    candidate_short = phase_short & _prob_gate("p_pivot_short", pivot_short_thr) & _prob_gate("p_tb_short", tb_short_thr)

    high_px = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low_px = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close_px = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(frame[cfg.atr_col], errors="coerce").to_numpy(dtype=float)
    max_holding = 10

    candidates = candidate_long if side == "long" else candidate_short
    for i in np.flatnonzero(candidates):
        ep = close_px[i]
        atr_i = atr[i]
        if not np.isfinite(ep) or not np.isfinite(atr_i) or atr_i <= 0.0:
            continue
        upper = ep + float(cfg.a_tp) * atr_i
        lower = ep - float(cfg.b_sl) * atr_i
        last_idx = min(i + max_holding, n - 1)
        out[i] = int(last_idx)
        for j in range(i + 1, last_idx + 1):
            if high_px[j] >= upper or low_px[j] <= lower:
                out[i] = int(j)
                break
    return out


def compute_exit_embargo_end_idx(
    frame: pd.DataFrame,
    *,
    side: str,
    enter_col: str,
    point_exit_col: str,
    use_next_open: bool,
) -> np.ndarray:
    side_key = str(side).strip().lower()
    if side_key not in {"long", "short"}:
        raise ValueError("side must be long or short")
    n = len(frame)
    out = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return out
    enter_sig = pd.to_numeric(frame[enter_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    point_exit = pd.to_numeric(frame[point_exit_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    entry_offset = 1 if bool(use_next_open) else 0

    i = 0
    while i < n:
        if not enter_sig[i]:
            i += 1
            continue
        entry_idx = i + entry_offset
        if entry_idx >= n:
            break
        exit_idx = None
        for j in range(entry_idx, n):
            if point_exit[j]:
                exit_idx = int(j)
                break
        if exit_idx is None:
            break
        if exit_idx > entry_idx:
            out[entry_idx:exit_idx] = int(exit_idx)
        i = max(i + 1, exit_idx)
    return out


def select_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: set[str] | None = None,
    corr_threshold: float | None = None,
    log_prefix: str = "[META-XGB]",
) -> list[str]:
    blocked = set(exclude or set())
    cols: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns available.")
    if corr_threshold is None:
        return cols

    threshold = float(corr_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("corr_threshold must be between 0 and 1.")

    corr = df.loc[:, cols].corr().abs()
    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for col in cols:
        drop_with: tuple[str, float] | None = None
        for kept_col in kept:
            corr_val = corr.at[col, kept_col]
            if pd.notna(corr_val) and float(corr_val) > threshold:
                drop_with = (kept_col, float(corr_val))
                break
        if drop_with is None:
            kept.append(col)
            continue
        dropped.append((col, drop_with[0], drop_with[1]))

    if dropped:
        print(
            f"{log_prefix} Correlation prune: dropped {len(dropped)} of {len(cols)} "
            f"numeric features with abs(corr) > {threshold:.2f}."
        )
        for dropped_col, kept_col, corr_val in dropped:
            print(
                f"{log_prefix} Correlation prune drop: {dropped_col} "
                f"(corr={corr_val:.4f} with kept {kept_col})"
            )
    else:
        print(
            f"{log_prefix} Correlation prune: dropped 0 of {len(cols)} numeric features "
            f"with abs(corr) > {threshold:.2f}."
        )
    return kept


def build_day_folds(
    session_dates: pd.Series | np.ndarray,
    *,
    valid_mask: np.ndarray,
    n_folds: int,
    initial_train_days: int,
    purge_days: int,
) -> list[dict[str, Any]]:
    day_index = pd.Index(pd.Series(session_dates).astype("string")).dropna().unique()
    n_days = int(len(day_index))
    if n_days < 2:
        raise ValueError("Need at least 2 distinct sessions for walk-forward training.")
    if int(n_folds) < 1:
        raise ValueError("n_folds must be >= 1")
    init_days = int(initial_train_days)
    if init_days <= 0:
        init_days = max(1, n_days // (int(n_folds) + 1))
    if init_days >= n_days:
        raise ValueError("initial_train_days must be < number of sessions")
    remaining = n_days - init_days
    fold_size = remaining // int(n_folds)
    if fold_size <= 0:
        raise ValueError("Fold size too small; reduce n_folds or initial_train_days.")

    session_arr = pd.Series(session_dates).astype("string").to_numpy()
    valid = np.asarray(valid_mask, dtype=bool)
    folds: list[dict[str, Any]] = []
    for fold_id in range(int(n_folds)):
        eval_start = init_days + fold_id * fold_size
        eval_end = init_days + (fold_id + 1) * fold_size
        if fold_id == int(n_folds) - 1:
            eval_end = n_days
        if eval_start >= eval_end:
            continue
        train_end = max(0, eval_start - max(0, int(purge_days)))
        train_days = day_index[:train_end]
        eval_days = day_index[eval_start:eval_end]
        eval_rows_all = np.flatnonzero(np.isin(session_arr, eval_days))
        if eval_rows_all.size == 0:
            continue
        fit_idx = np.flatnonzero(valid & np.isin(session_arr, train_days))
        pred_idx = np.flatnonzero(valid & np.isin(session_arr, eval_days))
        folds.append(
            {
                "fold_id": fold_id,
                "fit_idx": fit_idx,
                "pred_idx": pred_idx,
                "eval_start_row": int(eval_rows_all[0]),
                "eval_end_row": int(eval_rows_all[-1]),
                "train_day_count": int(len(train_days)),
                "eval_day_count": int(len(eval_days)),
            }
        )
    if not folds:
        raise ValueError("No walk-forward folds were created.")
    return folds


def _sanitize_xgb_params(xgb_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(xgb_params)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    booster = str(params.get("booster", "gbtree")).lower()
    if booster != "dart":
        params.pop("rate_drop", None)
        params.pop("skip_drop", None)
        params.pop("one_drop", None)
        params.pop("sample_type", None)
        params.pop("normalize_type", None)
    return params


def xgb_params_from_config(cfg: PipelineConfig) -> dict[str, Any]:
    params = GAXGBoostFeatureSelector().xgb_params.copy()
    params["seed"] = int(cfg.random_state)
    if cfg.n_estimators is not None and int(cfg.n_estimators) > 0:
        params["n_estimators"] = int(cfg.n_estimators)
    if cfg.xgb_booster is not None:
        params["booster"] = str(cfg.xgb_booster)
    if str(params.get("booster", "gbtree")).lower() == "dart":
        params.update(DART_DEFAULTS)
    if cfg.xgb_rate_drop is not None:
        params["rate_drop"] = float(cfg.xgb_rate_drop)
    if cfg.xgb_skip_drop is not None:
        params["skip_drop"] = float(cfg.xgb_skip_drop)
    if cfg.xgb_one_drop is not None:
        params["one_drop"] = int(bool(cfg.xgb_one_drop))
    if cfg.xgb_sample_type is not None:
        params["sample_type"] = str(cfg.xgb_sample_type)
    if cfg.xgb_normalize_type is not None:
        params["normalize_type"] = str(cfg.xgb_normalize_type)
    return _sanitize_xgb_params(params)


def _fit_booster(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: dict[str, Any],
    seed: int,
    early_stopping_rounds: int | None = None,
    early_stopping_val_fraction: float = 0.20,
    early_stopping_min_val_rows: int = 100,
) -> tuple[xgb.Booster | None, float | None, dict[str, Any], int]:
    if X.shape[0] == 0:
        return None, 0.0, _sanitize_xgb_params(params), int(params.get("n_estimators", 100))
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    params_local = dict(params)
    params_local["seed"] = int(seed)
    params_local["scale_pos_weight"] = (neg / max(pos, 1)) if pos > 0 else 1.0
    params_local = _sanitize_xgb_params(params_local)
    num_boost_round = int(params_local.pop("n_estimators", 100))
    n_jobs = params_local.pop("n_jobs", None)
    if n_jobs is not None:
        params_local["nthread"] = int(n_jobs)
    if pos == 0 or neg == 0:
        const_prob = float(np.mean(y)) if y.size else 0.0
        return None, const_prob, params_local, num_boost_round

    dtrain = xgb.DMatrix(X, label=y)
    evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
    early_stopping = None
    es_rounds = int(early_stopping_rounds) if early_stopping_rounds is not None else 0
    if es_rounds > 0:
        n_rows = int(X.shape[0])
        val_rows = max(
            int(early_stopping_min_val_rows),
            int(np.ceil(n_rows * float(early_stopping_val_fraction))),
        )
        if 0 < val_rows < n_rows:
            split_idx = n_rows - val_rows
            if split_idx > 0:
                y_fit = y[:split_idx]
                y_val = y[split_idx:]
                if len(np.unique(y_fit)) > 1 and len(np.unique(y_val)) > 1:
                    dtrain = xgb.DMatrix(X[:split_idx], label=y_fit)
                    dval = xgb.DMatrix(X[split_idx:], label=y_val)
                    evals = [(dtrain, "train"), (dval, "validation")]
                    early_stopping = es_rounds

    train_kwargs: dict[str, Any] = {
        "params": params_local,
        "dtrain": dtrain,
        "num_boost_round": num_boost_round,
        "evals": evals,
        "verbose_eval": max(1, min(int(XGB_VERBOSE_EVAL_EVERY), int(num_boost_round))),
    }
    if early_stopping is not None:
        train_kwargs["early_stopping_rounds"] = int(early_stopping)
    model = xgb.train(**train_kwargs)
    return model, None, params_local, num_boost_round


def build_final_eval_history(
    *,
    X: np.ndarray,
    y: np.ndarray,
    session_dates: pd.Series | np.ndarray,
    valid_mask: np.ndarray,
    params: dict[str, Any],
    seed: int,
    validation_fraction: float = 0.20,
    min_validation_days: int = 1,
    early_stopping_rounds: int | None = None,
) -> dict[str, list[float]] | None:
    valid_idx = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if valid_idx.size < 10:
        return None
    session_arr = pd.Series(session_dates).astype("string").to_numpy()
    valid_days = pd.Index(session_arr[valid_idx]).dropna().unique()
    if len(valid_days) < 2:
        return None
    val_days = max(int(np.ceil(len(valid_days) * float(validation_fraction))), int(min_validation_days))
    val_days = min(max(1, val_days), len(valid_days) - 1)
    train_days = valid_days[:-val_days]
    eval_days = valid_days[-val_days:]
    fit_idx = valid_idx[np.isin(session_arr[valid_idx], train_days)]
    eval_idx = valid_idx[np.isin(session_arr[valid_idx], eval_days)]
    if fit_idx.size == 0 or eval_idx.size == 0:
        return None

    y_fit = y[fit_idx]
    y_eval = y[eval_idx]
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
        return None

    params_local = dict(params)
    params_local["seed"] = int(seed)
    params_local["scale_pos_weight"] = float(np.sum(y_fit == 0) / max(int(np.sum(y_fit == 1)), 1))
    params_local = _sanitize_xgb_params(params_local)
    num_boost_round = int(params_local.pop("n_estimators", 100))
    n_jobs = params_local.pop("n_jobs", None)
    if n_jobs is not None:
        params_local["nthread"] = int(n_jobs)

    dtrain = xgb.DMatrix(X[fit_idx], label=y_fit)
    dval = xgb.DMatrix(X[eval_idx], label=y_eval)
    evals_result: dict[str, dict[str, list[float]]] = {}
    train_kwargs: dict[str, Any] = {
        "params": params_local,
        "dtrain": dtrain,
        "num_boost_round": num_boost_round,
        "evals": [(dtrain, "train"), (dval, "validation")],
        "evals_result": evals_result,
        "verbose_eval": max(1, min(int(XGB_VERBOSE_EVAL_EVERY), int(num_boost_round))),
    }
    es_rounds = int(early_stopping_rounds) if early_stopping_rounds is not None else 0
    if es_rounds > 0:
        train_kwargs["early_stopping_rounds"] = es_rounds
    xgb.train(**train_kwargs)
    train_logloss = [float(v) for v in evals_result.get("train", {}).get("logloss", [])]
    val_logloss = [float(v) for v in evals_result.get("validation", {}).get("logloss", [])]
    if not train_logloss or not val_logloss:
        return None
    return {
        "train_logloss": train_logloss,
        "val_logloss": val_logloss,
    }


def predict_probs(
    model: xgb.Booster | None,
    X: np.ndarray,
    *,
    constant_prob: float | None,
) -> np.ndarray:
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if model is None:
        p = 0.0 if constant_prob is None else float(constant_prob)
        return np.full(X.shape[0], p, dtype=np.float32)
    dmat = xgb.DMatrix(X)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and int(best_iteration) >= 0:
        try:
            return model.predict(dmat, iteration_range=(0, int(best_iteration) + 1)).astype(np.float32)
        except TypeError:
            pass
    best_ntree_limit = getattr(model, "best_ntree_limit", None)
    if best_ntree_limit is not None and int(best_ntree_limit) > 0:
        try:
            return model.predict(dmat, ntree_limit=int(best_ntree_limit)).astype(np.float32)
        except TypeError:
            pass
    return model.predict(dmat).astype(np.float32)


def train_walkforward_binary(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    session_dates: pd.Series | np.ndarray,
    cfg: PipelineConfig,
    xgb_params: dict[str, Any],
    condition_mask: np.ndarray | None = None,
    embargo_end_idx: np.ndarray | None = None,
) -> TrainResult:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    valid_mask = np.isfinite(X).all(axis=1) & ((y == 0) | (y == 1))
    if condition_mask is not None:
        valid_mask &= np.asarray(condition_mask, dtype=bool)
    print(
        f"[META-XGB] {target_col}: rows={len(df)} valid_rows={int(np.sum(valid_mask))} "
        f"pos_rate={float(np.mean(y[valid_mask])):.4f}" if np.any(valid_mask) else
        f"[META-XGB] {target_col}: rows={len(df)} valid_rows=0"
    )
    folds = build_day_folds(
        session_dates,
        valid_mask=valid_mask,
        n_folds=int(cfg.n_folds),
        initial_train_days=int(cfg.initial_train_days),
        purge_days=int(cfg.purge_days),
    )
    embargo_idx = None
    if embargo_end_idx is not None:
        embargo_idx = np.asarray(embargo_end_idx, dtype=np.int64)
        if embargo_idx.shape[0] != len(df):
            raise ValueError("embargo_end_idx must match dataframe length")
    oof = np.full(len(df), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, int]] = []
    for fold in folds:
        fit_idx = np.asarray(fold["fit_idx"], dtype=int)
        pred_idx = np.asarray(fold["pred_idx"], dtype=int)
        embargoed_rows = 0
        if embargo_idx is not None and fit_idx.size > 0:
            eval_start_row = int(fold["eval_start_row"])
            keep_mask = embargo_idx[fit_idx] < eval_start_row
            embargoed_rows = int(np.sum(~keep_mask))
            fit_idx = fit_idx[keep_mask]
        print(
            f"[META-XGB] {target_col}: fold={int(fold['fold_id']) + 1}/{len(folds)} "
            f"fit_rows={int(fit_idx.size)} pred_rows={int(pred_idx.size)} "
            f"embargoed={int(embargoed_rows)} eval_start_row={int(fold['eval_start_row'])}"
        )
        if fit_idx.size == 0 or pred_idx.size == 0:
            fold_rows.append(
                {
                    "fold_id": int(fold["fold_id"]),
                    "fit_rows": int(fit_idx.size),
                    "pred_rows": int(pred_idx.size),
                    "embargoed_rows": int(embargoed_rows),
                    "eval_start_row": int(fold["eval_start_row"]),
                }
            )
            continue
        model, constant_prob, _, _ = _fit_booster(
            X[fit_idx],
            y[fit_idx],
            params=xgb_params,
            seed=int(cfg.random_state) + int(fold["fold_id"]),
            early_stopping_rounds=cfg.early_stopping_rounds,
            early_stopping_val_fraction=cfg.early_stopping_val_fraction,
            early_stopping_min_val_rows=cfg.early_stopping_min_val_rows,
        )
        oof[pred_idx] = predict_probs(model, X[pred_idx], constant_prob=constant_prob)
        fold_rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "fit_rows": int(fit_idx.size),
                "pred_rows": int(pred_idx.size),
                "embargoed_rows": int(embargoed_rows),
                "eval_start_row": int(fold["eval_start_row"]),
            }
        )

    model_full, constant_prob_full, params_full, num_boost_round = _fit_booster(
        X[valid_mask],
        y[valid_mask],
        params=xgb_params,
        seed=int(cfg.random_state) + 10_000,
        early_stopping_rounds=cfg.early_stopping_rounds,
        early_stopping_val_fraction=cfg.early_stopping_val_fraction,
        early_stopping_min_val_rows=cfg.early_stopping_min_val_rows,
    )
    full_probs = np.full(len(df), np.nan, dtype=np.float32)
    full_probs[valid_mask] = predict_probs(
        model_full,
        X[valid_mask],
        constant_prob=constant_prob_full,
    )
    coverage = float(np.isfinite(oof[valid_mask]).mean()) if np.any(valid_mask) else 0.0
    print(
        f"[META-XGB] {target_col}: OOF coverage={coverage:.2%} "
        f"full_fit_rows={int(np.sum(valid_mask))}"
    )
    eval_history = build_final_eval_history(
        X=X,
        y=y,
        session_dates=session_dates,
        valid_mask=valid_mask,
        params=xgb_params,
        seed=int(cfg.random_state) + 20_000,
        validation_fraction=float(cfg.early_stopping_val_fraction),
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    return TrainResult(
        target_col=target_col,
        model=model_full,
        constant_prob=constant_prob_full,
        params=params_full,
        num_boost_round=num_boost_round,
        valid_mask=valid_mask,
        oof_probs=oof,
        full_probs=full_probs,
        coverage=coverage,
        fold_rows=fold_rows,
        eval_history=eval_history,
    )


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, *, threshold: float) -> dict[str, float]:
    mask = np.isfinite(probs)
    y = y_true[mask]
    p = probs[mask]
    out: dict[str, float] = {"n": float(y.size), "threshold": float(threshold)}
    if y.size == 0:
        return out
    pred = (p >= float(threshold)).astype(np.int8)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))

    def _fbeta_from_pr(prec: float, rec: float, beta: float) -> float:
        beta_sq = float(beta) ** 2
        denom = beta_sq * prec + rec
        if denom <= 0.0:
            return 0.0
        return float((1.0 + beta_sq) * prec * rec / denom)

    out.update(
        {
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f0_5": _fbeta_from_pr(precision, recall, beta=0.5),
            "f2": _fbeta_from_pr(precision, recall, beta=2.0),
            "pos_rate": float(np.mean(y)),
            "pred_rate": float(np.mean(pred)),
        }
    )
    try:
        out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        out["logloss"] = float("nan")
    try:
        out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["average_precision"] = (
            float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        )
    except ValueError:
        out["average_precision"] = float("nan")
    return out


def sweep_thresholds(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows = [binary_metrics(y_true, probs, threshold=float(t)) for t in threshold_grid(cfg)]
    return pd.DataFrame(rows)


def choose_threshold(sweep_df: pd.DataFrame, *, objective: str) -> tuple[float, dict[str, float]]:
    if sweep_df.empty:
        return 0.5, {"threshold": 0.5}
    key = str(objective).strip().lower().replace("-", "_")
    aliases = {
        "ap": "average_precision",
        "pr_auc": "average_precision",
        "f0.5": "f0_5",
        "f05": "f0_5",
        "fbeta_0_5": "f0_5",
        "fbeta0_5": "f0_5",
        "fbeta0.5": "f0_5",
    }
    metric = key if key in sweep_df.columns else aliases.get(key, "f1")
    if metric not in sweep_df.columns:
        metric = "f1"

    ascending_metric = metric == "logloss"
    ordered = sweep_df.sort_values(
        [metric, "precision", "recall", "threshold"],
        ascending=[ascending_metric, False, False, False],
    )
    best = ordered.iloc[0].to_dict()
    return float(best["threshold"]), {str(k): (float(v) if isinstance(v, (int, float, np.generic)) and np.isfinite(v) else None) for k, v in best.items()}


def save_train_val_loss_plot(
    *,
    histories: dict[str, dict[str, list[float]] | None],
    save_path: Path,
    title: str,
) -> Path | None:
    valid_items = [(name, hist) for name, hist in histories.items() if hist and hist.get("train_logloss") and hist.get("val_logloss")]
    if not valid_items:
        return None
    fig, axes = plt.subplots(1, len(valid_items), figsize=(10 * len(valid_items), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    for ax, (name, hist) in zip(axes, valid_items):
        train_vals = np.asarray(hist["train_logloss"], dtype=float)
        val_vals = np.asarray(hist["val_logloss"], dtype=float)
        rounds = np.arange(train_vals.size)
        ax.plot(rounds, train_vals, label="train_logloss")
        ax.plot(rounds, val_vals, label="val_logloss")
        ax.set_title(str(name).replace("_", " ").title())
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Logloss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_booster_artifacts(
    *,
    out_dir: Path,
    result: TrainResult,
    feature_cols: list[str],
    oof_name: str,
    full_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.model is not None:
        result.model.save_model(str(out_dir / "xgb_model.json"))
    np.save(out_dir / f"{oof_name}.npy", result.oof_probs)
    np.save(out_dir / f"{full_name}.npy", result.full_probs)
    (out_dir / "feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    meta = {
        "target_col": result.target_col,
        "constant_prob": result.constant_prob,
        "xgb_params": result.params,
        "num_boost_round": int(result.num_boost_round),
        "coverage": float(result.coverage),
        "fold_rows": result.fold_rows,
        "valid_rows": int(np.sum(result.valid_mask)),
        "eval_history": result.eval_history,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_prob_frame(path: Path, *, index: pd.Index, columns: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns, index=index)
    df.to_parquet(path)
    return path


def load_prob_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing probability frame: {path}")
    return pd.read_parquet(path)


def add_position_context_features(
    df: pd.DataFrame,
    *,
    side: str,
    enter_col: str,
    point_exit_col: str,
    atr_col: str,
    use_next_open: bool,
    a_tp: float,
    trail_activate_atr: float,
    trail_atr: float,
    trail_atr_after_tp: float,
    use_tp_to_tighten_trail: bool,
) -> pd.DataFrame:
    side_key = str(side).strip().lower()
    if side_key not in {"long", "short"}:
        raise ValueError("side must be long or short")
    out = df.copy()
    n = len(out)
    prefix = side_key
    active_col = f"in_{prefix}_trade"
    bars_col = f"{prefix}_bars_since_entry"
    mfe_col = f"{prefix}_mfe_atr"
    mae_col = f"{prefix}_mae_atr"
    tp_seen_col = f"{prefix}_tp_seen_run"
    trail_gap_col = f"{prefix}_trail_gap_atr"
    entry_price_col = f"{prefix}_entry_price_ctx"

    active = np.zeros(n, dtype=np.int8)
    bars_since = np.full(n, np.nan, dtype=float)
    mfe_atr = np.full(n, np.nan, dtype=float)
    mae_atr = np.full(n, np.nan, dtype=float)
    tp_seen_run = np.zeros(n, dtype=np.int8)
    trail_gap_atr = np.full(n, np.nan, dtype=float)
    entry_price_ctx = np.full(n, np.nan, dtype=float)

    open_px = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
    high_px = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low_px = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close_px = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(out[atr_col], errors="coerce").to_numpy(dtype=float)
    enter_sig = out[enter_col].fillna(0).astype(int).to_numpy() == 1
    point_exit = out[point_exit_col].fillna(0).astype(int).to_numpy() == 1
    entry_offset = 1 if bool(use_next_open) else 0

    i = 0
    while i < n:
        if not enter_sig[i]:
            i += 1
            continue
        entry_idx = i + entry_offset
        if entry_idx >= n:
            break
        entry = open_px[entry_idx]
        atr_i = atr[i]
        if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0.0:
            i += 1
            continue

        tp = entry + float(a_tp) * atr_i if side_key == "long" else entry - float(a_tp) * atr_i
        trail_dist = float(trail_atr) * atr_i
        trail_dist_tight = max(float(trail_atr_after_tp), 1e-9) * atr_i
        trail_active = False
        tp_seen = False
        favorable_anchor = entry
        adverse_anchor = entry
        j = entry_idx
        while j < n and not point_exit[j]:
            active[j] = 1
            bars_since[j] = float(j - entry_idx)
            entry_price_ctx[j] = entry
            if side_key == "long":
                if np.isfinite(high_px[j]):
                    favorable_anchor = max(favorable_anchor, high_px[j])
                if np.isfinite(low_px[j]):
                    adverse_anchor = min(adverse_anchor, low_px[j])
                if (favorable_anchor - entry) >= float(trail_activate_atr) * atr_i:
                    trail_active = True
                if np.isfinite(high_px[j]) and high_px[j] >= tp:
                    tp_seen = True
                    if use_tp_to_tighten_trail:
                        trail_active = True
                        trail_dist = min(trail_dist, trail_dist_tight)
                mfe_atr[j] = (favorable_anchor - entry) / atr_i
                mae_atr[j] = (entry - adverse_anchor) / atr_i
                trail_level = favorable_anchor - trail_dist if trail_active else np.nan
                trail_gap_atr[j] = (close_px[j] - trail_level) / atr_i if np.isfinite(trail_level) and np.isfinite(close_px[j]) else np.nan
            else:
                if np.isfinite(low_px[j]):
                    favorable_anchor = min(favorable_anchor, low_px[j])
                if np.isfinite(high_px[j]):
                    adverse_anchor = max(adverse_anchor, high_px[j])
                if (entry - favorable_anchor) >= float(trail_activate_atr) * atr_i:
                    trail_active = True
                if np.isfinite(low_px[j]) and low_px[j] <= tp:
                    tp_seen = True
                    if use_tp_to_tighten_trail:
                        trail_active = True
                        trail_dist = min(trail_dist, trail_dist_tight)
                mfe_atr[j] = (entry - favorable_anchor) / atr_i
                mae_atr[j] = (adverse_anchor - entry) / atr_i
                trail_level = favorable_anchor + trail_dist if trail_active else np.nan
                trail_gap_atr[j] = (trail_level - close_px[j]) / atr_i if np.isfinite(trail_level) and np.isfinite(close_px[j]) else np.nan
            tp_seen_run[j] = np.int8(tp_seen)
            j += 1
        i = max(i + 1, j)

    out[active_col] = active
    out[bars_col] = bars_since
    out[mfe_col] = mfe_atr
    out[mae_col] = mae_atr
    out[tp_seen_col] = tp_seen_run
    out[trail_gap_col] = trail_gap_atr
    out[entry_price_col] = entry_price_ctx
    return out
