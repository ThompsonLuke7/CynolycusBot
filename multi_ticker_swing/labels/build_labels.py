"""
Label builder for the multi-ticker 30-minute swing pipeline.

Primary training label — soft swing zone (same scheme as the 30m plots):
  - Fractal pivots detected with sequence_count=3 (3 bars each side)
  - Core label shifted T-1 within the same session (so the model fires
    one bar before the confirmed reversal bar, not 3 bars after)
  - Zone weighting: core=1.0, ±1 bar neighbor=0.75, ambiguous=0.0
  - Session-aware: neighbor/ambiguous window never crosses overnight gap
  - First-in-run filter with price-progression reset

Columns written per ticker:
  soft_long_label   / soft_short_label   — 0/1 zone membership
  soft_long_weight  / soft_short_weight  — sample weight (1.0 / 0.75 / 0.0)
  soft_long_core    / soft_short_core    — 0/1 core (T-1 shifted) bars only
  target             — 0=short  1=neutral  2=long  (hard class from zone membership)
  sample_weight      — combined weight for XGBoost fit_params
  p_long_state_gate / p_short_state_gate — swing state machine gates (analysis)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Features.label_generations import add_pivot_swing_state_machine
from Features.feature_sets.custom_indicators import add_fractal_pivots

# Reuse session-aware helpers and constants directly from the plot script
# so label logic and plots are guaranteed to be identical.
from multi_ticker_swing.plots.generate_soft_swing_30m_plots import (
    AMBIGUOUS_WEIGHT,
    AMBIGUOUS_WINDOW_BARS,
    FIRST_IN_RUN_FILTER,
    FOLLOWTHROUGH_BARS,
    FOLLOWTHROUGH_MIN_ATR,
    LABEL_SHIFT_BARS,
    NEIGHBOR_WEIGHT,
    POSITIVE_WINDOW_BARS,
    _compute_followthrough_mask,
    _shift_pivot_labels_back,
    apply_swing_pivot_zone_weights_session_aware,
    keep_first_same_side_event_session_reset,
)

from multi_ticker_swing.config.pipeline_config import (
    FEATURES_COMBINED,
    LABELS_COMBINED,
    PROCESSED_30M_DIR,
    PROCESSED_LBL_DIR,
    SWING_LABEL_30M_CONFIG,
    SWING_STATE_CONFIG,
    TRAINING_MATRIX,
    UNIVERSE_CSV,
    FEATURE_COLUMNS,
)
from multi_ticker_swing.data.fetch_data import load_universe
from multi_ticker_swing.data.load_data import load_ticker_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft swing zone labels — mirrors generate_soft_swing_30m_plots.py exactly
# ---------------------------------------------------------------------------

def _build_soft_swing_labels(
    df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Compute soft swing zone labels on df (must have a DatetimeIndex and
    pivot_up / pivot_down columns already set).

    Writes columns:
      soft_long_core, soft_short_core      — shifted T-1 core pivot bars
      soft_long_label, soft_short_label    — zone membership (0/1)
      soft_long_weight, soft_short_weight  — sample weights
    """
    timestamps = df.index

    long_core  = _shift_pivot_labels_back(df["pivot_down"], timestamps, LABEL_SHIFT_BARS)
    short_core = _shift_pivot_labels_back(df["pivot_up"],   timestamps, LABEL_SHIFT_BARS)

    long_core_arr  = long_core.to_numpy(dtype=np.int64)
    short_core_arr = short_core.to_numpy(dtype=np.int64)

    if FIRST_IN_RUN_FILTER:
        lows  = df["low"].to_numpy(float)
        highs = df["high"].to_numpy(float)
        long_core_arr, short_core_arr, _, _ = keep_first_same_side_event_session_reset(
            long_core_arr, short_core_arr, timestamps,
            lows=lows, highs=highs,
        )

    # ------------------------------------------------------------------
    # Follow-through filter: zero out cores that didn't follow through.
    # A core with no follow-through has no learning signal — training on
    # it as a directional label adds noise. Its neighbor zone bars are also
    # zeroed by removing the core before zone computation.
    # Requires atr_14 column; silently skip if missing.
    # ------------------------------------------------------------------
    if "atr_14" in df.columns:
        long_ft  = _compute_followthrough_mask(df, long_core_arr.astype(bool),  "long",
                                               min_atr=FOLLOWTHROUGH_MIN_ATR,
                                               max_bars=FOLLOWTHROUGH_BARS)
        short_ft = _compute_followthrough_mask(df, short_core_arr.astype(bool), "short",
                                               min_atr=FOLLOWTHROUGH_MIN_ATR,
                                               max_bars=FOLLOWTHROUGH_BARS)
        n_lc_before = int(long_core_arr.sum())
        n_sc_before = int(short_core_arr.sum())
        long_core_arr[~long_ft]  = 0   # zero non-follow-through long cores
        short_core_arr[~short_ft] = 0  # zero non-follow-through short cores
        logger.info(
            "[%s] follow-through filter: long %d→%d  short %d→%d  "
            "(%.0f%% long  %.0f%% short kept)",
            ticker,
            n_lc_before, int(long_core_arr.sum()),
            n_sc_before, int(short_core_arr.sum()),
            100 * long_ft.mean()  if n_lc_before else 0,
            100 * short_ft.mean() if n_sc_before else 0,
        )
    else:
        logger.warning("[%s] atr_14 not in df — follow-through filter skipped", ticker)

    long_zone, long_w, _  = apply_swing_pivot_zone_weights_session_aware(
        long_core_arr, timestamps,
        positive_window_bars=POSITIVE_WINDOW_BARS,
        ambiguous_window_bars=AMBIGUOUS_WINDOW_BARS,
        neighbor_weight=NEIGHBOR_WEIGHT,
        ambiguous_weight=AMBIGUOUS_WEIGHT,
    )
    short_zone, short_w, _ = apply_swing_pivot_zone_weights_session_aware(
        short_core_arr, timestamps,
        positive_window_bars=POSITIVE_WINDOW_BARS,
        ambiguous_window_bars=AMBIGUOUS_WINDOW_BARS,
        neighbor_weight=NEIGHBOR_WEIGHT,
        ambiguous_weight=AMBIGUOUS_WEIGHT,
    )

    df["soft_long_core"]    = long_core_arr.astype(np.int8)
    df["soft_short_core"]   = short_core_arr.astype(np.int8)
    df["soft_long_label"]   = long_zone.astype(np.int8)
    df["soft_short_label"]  = short_zone.astype(np.int8)
    df["soft_long_weight"]  = long_w.astype(np.float32)
    df["soft_short_weight"] = short_w.astype(np.float32)

    n_lc = int(long_core_arr.sum());   n_lz = int(long_zone.sum())
    n_sc = int(short_core_arr.sum());  n_sz = int(short_zone.sum())
    logger.info(
        "[%s] soft labels: long core=%d zone=%d  short core=%d zone=%d",
        ticker, n_lc, n_lz, n_sc, n_sz,
    )
    return df


# ---------------------------------------------------------------------------
# Target + sample_weight construction
# ---------------------------------------------------------------------------

def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive hard XGBoost target and sample_weight from soft zone columns.

    target:
      2 (long)    if soft_long_label  == 1  (core or neighbor)
      0 (short)   if soft_short_label == 1
      1 (neutral) otherwise
      Conflict (both long + short zone) → neutral, weight=0

    sample_weight:
      long  core → 1.0,  long  neighbor → 0.75
      short core → 1.0,  short neighbor → 0.75
      ambiguous  → 0.0
      neutral    → 1.0
    """
    long_z  = df["soft_long_label"].to_numpy(np.int8)
    short_z = df["soft_short_label"].to_numpy(np.int8)
    long_w  = df["soft_long_weight"].to_numpy(np.float32)
    short_w = df["soft_short_weight"].to_numpy(np.float32)

    n = len(df)
    target = np.ones(n, dtype=np.int8)   # default neutral
    weight = np.ones(n, dtype=np.float32)

    long_only  = (long_z  == 1) & (short_z == 0)
    short_only = (short_z == 1) & (long_z  == 0)
    conflict   = (long_z  == 1) & (short_z == 1)

    target[long_only]  = 2
    target[short_only] = 0
    target[conflict]   = 1

    weight[long_only]  = long_w[long_only]
    weight[short_only] = short_w[short_only]
    weight[conflict]   = 0.0

    df["target"]        = target.astype(int)
    df["sample_weight"] = weight.astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Per-ticker label builder
# ---------------------------------------------------------------------------

def build_ticker_labels(
    ticker: str,
    df_feat: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Compute soft swing zone labels + state gate for one ticker.

    df_feat must have a DatetimeIndex and causal OHLCV + atr_14 columns.
    Returns a DataFrame of label columns aligned to df_feat's index.
    """
    df = df_feat.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure DatetimeIndex (features stage sets this, but be safe)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("timestamp", "time", "date"):
            if col in df.columns:
                df = df.set_index(col)
                break
        df.index = pd.to_datetime(df.index)

    required = {"open", "high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("[%s] missing columns: %s", ticker, missing)
        return None

    # ------------------------------------------------------------------
    # 1. Fractal pivots (lookahead OK — labels look forward by design)
    # ------------------------------------------------------------------
    try:
        df = add_fractal_pivots(df, sequence_count=SWING_LABEL_30M_CONFIG["sequence_count"])
    except Exception as exc:
        logger.warning("[%s] add_fractal_pivots failed: %s", ticker, exc)

    if "pivot_up"   not in df.columns: df["pivot_up"]   = 0
    if "pivot_down" not in df.columns: df["pivot_down"] = 0

    # ------------------------------------------------------------------
    # 2. Soft swing zone labels (mirrors plot script exactly)
    # ------------------------------------------------------------------
    try:
        df = _build_soft_swing_labels(df, ticker)
    except Exception as exc:
        logger.error("[%s] soft swing labels failed: %s", ticker, exc)
        return None

    # ------------------------------------------------------------------
    # 3. XGBoost target + sample_weight
    # ------------------------------------------------------------------
    df = _build_target(df)

    # ------------------------------------------------------------------
    # 4. Swing state machine gate (binary fallback; for future Cat5 features)
    # ------------------------------------------------------------------
    try:
        df = add_pivot_swing_state_machine(
            df,
            atr_col="atr_14",
            atr_length=SWING_STATE_CONFIG.get("atr_length", 14),
            threshold=SWING_STATE_CONFIG["threshold"],
            confirm_threshold=SWING_STATE_CONFIG["confirm_threshold"],
            confirm_mult=SWING_STATE_CONFIG["confirm_mult"],
            pending_max_bars=SWING_STATE_CONFIG["pending_max_bars"],
            tp_mult=SWING_STATE_CONFIG["tp_mult"],
            sl_mult=SWING_STATE_CONFIG["sl_mult"],
            max_holding=SWING_STATE_CONFIG["max_holding"],
            cooldown_bars=SWING_STATE_CONFIG["cooldown_bars"],
            session_open_minutes=SWING_STATE_CONFIG["session_open_minutes"],
            session_late_minutes=SWING_STATE_CONFIG["session_late_minutes"],
            allow_binary_fallback=True,
        )
    except Exception as exc:
        logger.warning("[%s] swing state machine failed: %s", ticker, exc)
        df["p_long_state_gate"]  = np.nan
        df["p_short_state_gate"] = np.nan
        df["p_long_pending"]     = np.nan
        df["p_short_pending"]    = np.nan

    # ------------------------------------------------------------------
    # Return label columns only
    # ------------------------------------------------------------------
    label_cols = [
        "soft_long_core",  "soft_short_core",
        "soft_long_label", "soft_short_label",
        "soft_long_weight","soft_short_weight",
        "target", "sample_weight",
        "p_long_state_gate", "p_short_state_gate",
        "p_long_pending",    "p_short_pending",
        "p_state_id",
    ]
    existing = [c for c in label_cols if c in df.columns]
    return df[existing]


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def build_all_labels(
    *,
    universe_csv: Path | str = UNIVERSE_CSV,
    processed_30m_dir: Path = PROCESSED_30M_DIR,
    processed_lbl_dir: Path = PROCESSED_LBL_DIR,
    combined_path: Path = LABELS_COMBINED,
    force: bool = False,
) -> None:
    if combined_path.exists() and not force:
        logger.info("Combined labels exist at %s — skipping.", combined_path)
        return

    universe = load_universe(universe_csv)
    tickers  = universe["ticker"].tolist()
    processed_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    n = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        logger.info("(%d/%d) Building labels for %s", i, n, ticker)

        per_lbl_path = processed_lbl_dir / f"{ticker}_labels.parquet"
        if per_lbl_path.exists() and not force:
            logger.info("[%s] label cache hit", ticker)
            df_lbl = pd.read_parquet(per_lbl_path)
            df_lbl["ticker"] = ticker
            all_frames.append(df_lbl)
            continue

        df_feat = load_ticker_features(ticker, processed_30m_dir)
        if df_feat is None:
            logger.warning("[%s] feature file missing — skipping", ticker)
            continue

        df_lbl = build_ticker_labels(ticker, df_feat)
        if df_lbl is None:
            continue

        df_lbl.to_parquet(per_lbl_path)
        df_lbl["ticker"] = ticker
        all_frames.append(df_lbl)
        logger.info("[%s] done (%d labelled bars)", ticker, len(df_lbl))

    if not all_frames:
        logger.error("No label frames built.")
        return

    combined = pd.concat(all_frames)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path)
    logger.info("Saved combined labels → %s  (%d rows)", combined_path, len(combined))


# ---------------------------------------------------------------------------
# Training matrix builder
# ---------------------------------------------------------------------------

def build_training_matrix(
    *,
    features_path: Path = FEATURES_COMBINED,
    labels_path: Path = LABELS_COMBINED,
    output_path: Path = TRAINING_MATRIX,
    feature_columns: list[str] = FEATURE_COLUMNS,
    force: bool = False,
) -> pd.DataFrame | None:
    if output_path.exists() and not force:
        logger.info("Training matrix exists at %s — skipping.", output_path)
        return None

    feat = pd.read_parquet(features_path).reset_index()
    lbl  = pd.read_parquet(labels_path).reset_index()

    feat_ts = _find_ts_col(feat)
    lbl_ts  = _find_ts_col(lbl)

    lbl_keep = [lbl_ts, "ticker",
                "target", "sample_weight",
                "soft_long_label", "soft_short_label",
                "soft_long_weight", "soft_short_weight",
                "soft_long_core", "soft_short_core",
                "p_long_state_gate", "p_short_state_gate"]
    lbl_keep = [c for c in lbl_keep if c in lbl.columns]

    merged = pd.merge(
        feat,
        lbl[lbl_keep],
        left_on=[feat_ts, "ticker"],
        right_on=[lbl_ts, "ticker"],
        how="inner",
    )

    available = [c for c in feature_columns if c in merged.columns]
    missing   = set(feature_columns) - set(available)
    if missing:
        logger.warning("Features missing from matrix: %s", sorted(missing))

    keep = [feat_ts, "ticker"] + available + [
        "target", "sample_weight",
        "soft_long_label", "soft_short_label",
        "p_long_state_gate", "p_short_state_gate",
    ]
    keep   = [c for c in keep if c in merged.columns]
    merged = merged[keep]

    before = len(merged)
    merged = merged.dropna(subset=available)
    after  = len(merged)
    logger.info("Dropped %d NaN rows (%d → %d)", before - after, before, after)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info("Saved training matrix → %s  (%d rows, %d features)",
                output_path, after, len(available))
    return merged


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]
