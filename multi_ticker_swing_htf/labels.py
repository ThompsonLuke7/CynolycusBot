"""4H pivot-anchored swing labels.

The module asks a different question from momentum expansion:

    Did this pivot zone become a clean 5-15 day swing trade?

Forward stats grade the pivot, but the candidate label itself is anchored to
4H pivot-low/pivot-high zones.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from momentum_expansion.config.momentum_config import RAW_4H_DIR
from momentum_expansion.data.load_bars import load_4h
from momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H

from multi_ticker_swing_htf.config import (
    FEATURES_COMBINED,
    LABELS_COMBINED,
    PIVOT_LABEL_CONFIG,
    TRAINING_MATRIX,
)

logger = logging.getLogger(__name__)


def _fractal_pivots(df: pd.DataFrame, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    lows = df["low"]
    highs = df["high"]
    pivot_low = pd.Series(True, index=df.index)
    pivot_high = pd.Series(True, index=df.index)
    for i in range(1, left + 1):
        pivot_low &= lows < lows.shift(i)
        pivot_high &= highs > highs.shift(i)
    for i in range(1, right + 1):
        pivot_low &= lows <= lows.shift(-i)
        pivot_high &= highs >= highs.shift(-i)
    pivot_low.iloc[:left] = False
    pivot_low.iloc[-right:] = False
    pivot_high.iloc[:left] = False
    pivot_high.iloc[-right:] = False
    return pivot_low.astype(bool), pivot_high.astype(bool)


def _shift_core(mask: pd.Series, bars: int) -> pd.Series:
    return mask.shift(-int(bars), fill_value=False).astype(bool)


def _zone(core: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        return core.astype(bool)
    out = core.copy().astype(bool)
    for i in range(1, window + 1):
        out |= core.shift(i, fill_value=False)
        out |= core.shift(-i, fill_value=False)
    return out.astype(bool)


def _forward_extremes(df: pd.DataFrame, min_bars: int, max_bars: int) -> pd.DataFrame:
    c = df["close"]
    highs = pd.concat([df["high"].shift(-i) for i in range(min_bars, max_bars + 1)], axis=1)
    lows = pd.concat([df["low"].shift(-i) for i in range(min_bars, max_bars + 1)], axis=1)
    closes = pd.concat([df["close"].shift(-i) for i in range(1, max_bars + 1)], axis=1)
    return pd.DataFrame(
        {
            "fwd_best_high_return": highs.max(axis=1) / c - 1.0,
            "fwd_worst_low_return": lows.min(axis=1) / c - 1.0,
            "fwd_close_return": closes.iloc[:, -1] / c - 1.0,
            "long_persistence": (closes > c.values[:, None]).mean(axis=1),
            "short_persistence": (closes < c.values[:, None]).mean(axis=1),
        },
        index=df.index,
    )


def _bench_forward_alpha(index: pd.DatetimeIndex, benchmark: str, min_bars: int, max_bars: int) -> tuple[pd.Series, pd.Series]:
    bench_path = RAW_4H_DIR / f"{benchmark}.parquet"
    if not bench_path.exists():
        nan = pd.Series(np.nan, index=index)
        return nan, nan
    bench = load_4h(benchmark).reindex(index, method="ffill")
    ext = _forward_extremes(bench, min_bars, max_bars)
    return ext["fwd_best_high_return"], ext["fwd_worst_low_return"]


def build_ticker_labels_4h(
    *,
    ticker: str,
    df_4h: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame | None:
    cfg = {**PIVOT_LABEL_CONFIG, **(cfg or {})}
    left = int(cfg["pivot_left_bars"])
    right = int(cfg["pivot_right_bars"])
    shift_bars = int(cfg["label_shift_bars"])
    pos_window = int(cfg["positive_window_bars"])
    amb_window = int(cfg["ambiguous_window_bars"])
    min_bars = int(cfg["forward_min_bars"])
    max_bars = int(cfg["forward_max_bars"])

    if df_4h is None or len(df_4h) < max_bars + left + right + 100:
        return None
    df = df_4h.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return None

    pivot_low, pivot_high = _fractal_pivots(df, left, right)
    long_core = _shift_core(pivot_low, shift_bars)
    short_core = _shift_core(pivot_high, shift_bars)
    long_zone = _zone(long_core, pos_window)
    short_zone = _zone(short_core, pos_window)
    ambiguous = _zone(long_core | short_core, pos_window + amb_window) & ~(long_zone | short_zone)

    ext = _forward_extremes(df, min_bars, max_bars)
    bench_best, bench_worst = _bench_forward_alpha(df.index, str(cfg["alpha_benchmark"]), min_bars, max_bars)
    atr_pct = ((df["high"] - df["low"]).rolling(14).mean() / df["close"].replace(0, np.nan)).replace(0, np.nan)

    weights = cfg["composite_weights"]
    long_alpha = ext["fwd_best_high_return"] - bench_best
    long_drawdown = (-ext["fwd_worst_low_return"]).clip(lower=0)
    long_atr_adj = ext["fwd_best_high_return"] / atr_pct
    long_quality = (
        weights["alpha"] * long_alpha
        + weights["atr_adjusted"] * long_atr_adj
        - weights["drawdown"] * long_drawdown
        + weights["persistence"] * ext["long_persistence"]
    )

    short_best = (-ext["fwd_worst_low_return"]).clip(lower=0)
    short_adverse = ext["fwd_best_high_return"].clip(lower=0)
    short_alpha = short_best + bench_worst
    short_atr_adj = short_best / atr_pct
    short_quality = (
        weights["alpha"] * short_alpha
        + weights["atr_adjusted"] * short_atr_adj
        - weights["drawdown"] * short_adverse
        + weights["persistence"] * ext["short_persistence"]
    )

    target = np.ones(len(df), dtype=np.int8)
    target[long_zone.to_numpy() & ~short_zone.to_numpy()] = 2
    target[short_zone.to_numpy() & ~long_zone.to_numpy()] = 0
    conflict = long_zone & short_zone
    target[conflict.to_numpy()] = 1

    sample_weight = np.ones(len(df), dtype=np.float32)
    sample_weight[ambiguous.to_numpy() | conflict.to_numpy()] = 0.0
    sample_weight[long_core.to_numpy() | short_core.to_numpy()] = 1.0
    sample_weight[((long_zone | short_zone) & ~(long_core | short_core)).to_numpy()] = 0.75

    out = pd.DataFrame(
        {
            "pivot_low": pivot_low.astype(np.int8),
            "pivot_high": pivot_high.astype(np.int8),
            "htf_long_core": long_core.astype(np.int8),
            "htf_short_core": short_core.astype(np.int8),
            "htf_long_label": long_zone.astype(np.int8),
            "htf_short_label": short_zone.astype(np.int8),
            "htf_ambiguous": ambiguous.astype(np.int8),
            "target": target,
            "sample_weight": sample_weight,
            "long_swing_quality": long_quality,
            "short_swing_quality": short_quality,
            "fwd_best_high_return": ext["fwd_best_high_return"],
            "fwd_worst_low_return": ext["fwd_worst_low_return"],
            "fwd_close_return": ext["fwd_close_return"],
            "long_persistence": ext["long_persistence"],
            "short_persistence": ext["short_persistence"],
        },
        index=df.index,
    )
    out["htf_swing_score"] = np.where(out["target"].eq(2), out["long_swing_quality"], np.nan)
    out["htf_swing_score"] = np.where(out["target"].eq(0), out["short_swing_quality"], out["htf_swing_score"])
    out.loc[out["sample_weight"] <= 0, "htf_swing_score"] = np.nan
    return out


def build_all_labels_4h(
    *,
    tickers: Iterable[str],
    out_path: Path = LABELS_COMBINED,
    force: bool = False,
    cfg: dict | None = None,
) -> pd.DataFrame | None:
    if out_path.exists() and not force:
        logger.info("Labels already exist at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)
    frames = []
    tickers = list(dict.fromkeys(str(t).upper() for t in tickers))
    for i, ticker in enumerate(tickers, 1):
        try:
            df_4h = load_4h(ticker)
        except FileNotFoundError:
            continue
        labels = build_ticker_labels_4h(ticker=ticker, df_4h=df_4h, cfg=cfg)
        if labels is None:
            continue
        labels["ticker"] = ticker
        frames.append(labels)
        if i % 50 == 0:
            logger.info("(%d/%d) HTF labels built for %s", i, len(tickers), ticker)
    if not frames:
        logger.error("No HTF labels built.")
        return None
    combined = pd.concat(frames).reset_index().rename(columns={"index": "timestamp"})
    ts_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
    combined = combined.set_index([ts_col, "ticker"]).sort_index()
    top_q = float((cfg or PIVOT_LABEL_CONFIG).get("top_quantile", PIVOT_LABEL_CONFIG["top_quantile"]))
    score_rank = combined.groupby(level=0)["htf_swing_score"].rank(pct=True)
    combined["htf_top_swing_target"] = (score_rank >= 1.0 - top_q).astype(float)
    combined.loc[combined["htf_swing_score"].isna(), "htf_top_swing_target"] = np.nan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    logger.info("Saved HTF pivot labels (%d rows) -> %s", len(combined), out_path)
    return combined


def build_training_matrix(
    *,
    features_path: Path = FEATURES_COMBINED,
    labels_path: Path = LABELS_COMBINED,
    out_path: Path = TRAINING_MATRIX,
    force: bool = False,
) -> pd.DataFrame | None:
    if out_path.exists() and not force:
        logger.info("Training matrix exists at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features missing at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels missing at {labels_path}")
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    df = features.join(labels, how="inner")
    label_cols = [
        "target",
        "sample_weight",
        "htf_swing_score",
        "htf_top_swing_target",
        "htf_long_label",
        "htf_short_label",
        "htf_long_core",
        "htf_short_core",
        "long_swing_quality",
        "short_swing_quality",
        "fwd_best_high_return",
        "fwd_worst_low_return",
        "fwd_close_return",
        "long_persistence",
        "short_persistence",
    ]
    keep = [c for c in FEATURE_COLUMNS_4H if c in df.columns] + [c for c in label_cols if c in df.columns]
    df = df[keep].copy()
    feature_cols = [c for c in FEATURE_COLUMNS_4H if c in df.columns]
    df = df.dropna(subset=feature_cols, how="any")
    df = df.dropna(subset=["target", "sample_weight"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("Saved HTF training matrix (%d rows × %d cols) -> %s", len(df), df.shape[1], out_path)
    return df

