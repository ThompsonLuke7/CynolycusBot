"""
Label builder for the multi-ticker 30-minute swing pipeline.

Labels computed per ticker:
  1. Triple-barrier (multi-day swing horizon, adapted from existing logic)
  2. Swing state machine gate (adapted add_pivot_swing_state_machine)
  3. Final target: -1 → class 0 (short), 0 → class 1 (neutral), 1 → class 2 (long)

Saves per-ticker label parquets, then assembles a combined parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Existing label functions (reused directly)
from Features.label_generations import (
    add_pivot_swing_state_machine,
    add_triple_barrier_labels_atr,
)
from Features.feature_sets.custom_indicators import add_fractal_pivots

from multi_ticker_swing.config.pipeline_config import (
    LABELS_COMBINED,
    PROCESSED_30M_DIR,
    PROCESSED_LBL_DIR,
    SWING_STATE_CONFIG,
    TRAINING_MATRIX,
    TRIPLE_BARRIER_CONFIG,
    UNIVERSE_CSV,
    FEATURE_COLUMNS,
    FEATURES_COMBINED,
)
from multi_ticker_swing.data.fetch_data import load_universe
from multi_ticker_swing.data.load_data import load_ticker_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

_LABEL_MAP = {-1: 0, 0: 1, 1: 2}   # short → 0, neutral → 1, long → 2


def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map tb_label (-1/0/1) to XGBoost multi-class target (0/1/2).
    Any NaN tb_label row becomes class 1 (neutral).
    """
    df["target"] = (
        df["tb_label"].map(_LABEL_MAP).fillna(1).astype(int)
    )
    return df


def _add_swing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add swing metrics based on fractal pivots (from add_fractal_pivots).
    These columns use lookahead and are for label analysis only, not model features.
    """
    # Binary fractal pivot indicators (from add_fractal_pivots)
    pivot_long  = df.get("pivot_up", pd.Series(0, index=df.index))
    pivot_short = df.get("pivot_down", pd.Series(0, index=df.index))

    df["fractal_pivot_long"]  = pivot_long
    df["fractal_pivot_short"] = pivot_short

    # Bars since last fractal pivot
    def _bars_since(flag: pd.Series) -> pd.Series:
        out = []
        count = 999
        for v in flag:
            if v:
                count = 0
            else:
                count = min(count + 1, 999)
            out.append(count)
        return pd.Series(out, index=flag.index, dtype=float)

    df["bars_since_fractal_long"]  = _bars_since(pivot_long > 0)
    df["bars_since_fractal_short"] = _bars_since(pivot_short > 0)

    return df


# ---------------------------------------------------------------------------
# Per-ticker label builder
# ---------------------------------------------------------------------------

def build_ticker_labels(
    ticker: str,
    df_feat: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Given a ticker's feature DataFrame (which is causal), compute:
      - Triple-barrier labels (tb_label, tb_long_label, tb_short_label)
      - Swing state machine gate (p_long_state_gate, p_short_state_gate) using fractal pivots
      - Swing setup/metrics using fractal pivots (for analysis, not model features)
      - Final target column

    Note: Fractal pivots use 3-bar lookahead, which is OK here because:
      1. They're used only for label generation (labels already look forward)
      2. They're used for swing metrics/gates (not fed to the model as features)
      3. Features themselves (in df_feat) are already causal

    Returns a DataFrame with label/gate columns (index-aligned to df_feat).
    """
    df = df_feat.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("[%s] missing columns for labels: %s", ticker, missing)
        return None

    # ------------------------------------------------------------------
    # 1. Add fractal pivots for swing setup/metrics (OK to have lookahead here)
    # ------------------------------------------------------------------
    try:
        df = add_fractal_pivots(df)
    except Exception as exc:
        logger.warning("[%s] add_fractal_pivots failed: %s", ticker, exc)

    # Ensure pivots exist for swing state machine
    if "pivot_up" not in df.columns:
        df["pivot_up"] = 0
    if "pivot_down" not in df.columns:
        df["pivot_down"] = 0

    # ------------------------------------------------------------------
    # 2. Triple-barrier labels (multi-day swing horizon)
    # ------------------------------------------------------------------
    try:
        df = add_triple_barrier_labels_atr(
            df,
            close_col="close",
            high_col="high",
            low_col="low",
            open_col="open",
            atr_col="atr_14",
            atr_length=TRIPLE_BARRIER_CONFIG["atr_length"],
            k_up=TRIPLE_BARRIER_CONFIG["k_up"],
            k_dn=TRIPLE_BARRIER_CONFIG["k_dn"],
            max_holding=TRIPLE_BARRIER_CONFIG["max_holding"],
            chop_atr_mult=TRIPLE_BARRIER_CONFIG["chop_atr_mult"],
            base_label_col="tb_label",
        )
    except Exception as exc:
        logger.error("[%s] add_triple_barrier_labels_atr failed: %s", ticker, exc)
        return None

    # ------------------------------------------------------------------
    # 3. Swing state machine (binary fallback: uses pivot_up/pivot_down)
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
        logger.warning("[%s] swing state machine failed: %s — gate cols will be NaN", ticker, exc)
        df["p_long_state_gate"]  = np.nan
        df["p_short_state_gate"] = np.nan
        df["p_long_pending"]     = np.nan
        df["p_short_pending"]    = np.nan

    # ------------------------------------------------------------------
    # 4. Add swing setup/metrics based on fractal pivots (for analysis)
    # These use lookahead but are only for label metrics, not model features
    # ------------------------------------------------------------------
    df = _add_swing_metrics(df)

    # ------------------------------------------------------------------
    # 5. Build target
    # ------------------------------------------------------------------
    df = _build_target(df)

    # ------------------------------------------------------------------
    # Return label columns + swing metrics
    # ------------------------------------------------------------------
    label_cols = [
        "tb_label",
        "tb_long_label",
        "tb_short_label",
        "tb_entry_price",
        "tb_exit_price",
        "tb_holding_bars",
        "tb_realized_return",
        "p_long_state_gate",
        "p_short_state_gate",
        "p_long_pending",
        "p_short_pending",
        "p_state_id",
        "target",
        # Swing metrics (from fractal pivots - lookahead OK for labels)
        "fractal_pivot_long",
        "fractal_pivot_short",
        "bars_since_fractal_long",
        "bars_since_fractal_short",
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
    """
    Build labels for every universe ticker using per-ticker feature parquets,
    save per-ticker label parquets, then combine.
    """
    if combined_path.exists() and not force:
        logger.info("Combined labels already exist at %s — skipping.", combined_path)
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
            logger.info("[%s] label cache hit — loading", ticker)
            df_lbl = pd.read_parquet(per_lbl_path)
            df_lbl["ticker"] = ticker
            all_frames.append(df_lbl)
            continue

        df_feat = load_ticker_features(ticker, processed_30m_dir)
        if df_feat is None:
            logger.warning("[%s] feature file missing — skipping labels", ticker)
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

    combined = pd.concat(all_frames, axis=0)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path)
    logger.info("Saved combined labels → %s  (%d rows)", combined_path, len(combined))


# ---------------------------------------------------------------------------
# Training matrix builder (merges features + labels, drops NaNs)
# ---------------------------------------------------------------------------

def build_training_matrix(
    *,
    features_path: Path = FEATURES_COMBINED,
    labels_path: Path = LABELS_COMBINED,
    output_path: Path = TRAINING_MATRIX,
    feature_columns: list[str] = FEATURE_COLUMNS,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Inner-join features and labels on (timestamp, ticker), keep only
    feature_columns + target, drop rows with NaN in any feature column.
    """
    if output_path.exists() and not force:
        logger.info("Training matrix already exists at %s — skipping.", output_path)
        return None

    logger.info("Loading features from %s", features_path)
    feat = pd.read_parquet(features_path)
    logger.info("Loading labels from %s", labels_path)
    lbl  = pd.read_parquet(labels_path)

    # Align index structure: both should have (timestamp, ticker) or similar
    feat = feat.reset_index()
    lbl  = lbl.reset_index()

    # Identify join keys
    feat_ts_col  = _find_ts_col(feat)
    lbl_ts_col   = _find_ts_col(lbl)

    merged = pd.merge(
        feat,
        lbl[[ lbl_ts_col, "ticker", "target",
              "tb_label", "tb_long_label", "tb_short_label",
              "tb_realized_return", "tb_holding_bars",
              "p_long_state_gate", "p_short_state_gate" ]],
        left_on=[feat_ts_col, "ticker"],
        right_on=[lbl_ts_col, "ticker"],
        how="inner",
    )

    # Keep only feature columns that exist
    available_feats = [c for c in feature_columns if c in merged.columns]
    missing_feats   = set(feature_columns) - set(available_feats)
    if missing_feats:
        logger.warning("Features missing from combined matrix: %s", sorted(missing_feats))

    keep = [feat_ts_col, "ticker"] + available_feats + [
        "target", "tb_label", "tb_realized_return",
        "p_long_state_gate", "p_short_state_gate",
    ]
    keep = [c for c in keep if c in merged.columns]
    merged = merged[keep]

    before = len(merged)
    merged = merged.dropna(subset=available_feats)
    after  = len(merged)
    logger.info("Dropped %d NaN rows (%d → %d)", before - after, before, after)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info("Saved training matrix → %s  (%d rows, %d features)", output_path, after, len(available_feats))
    return merged


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]
