"""
Forward expansion labels for momentum_expansion.

For each 4H bar at time T we look forward W bars and compute:
  - fwd_max_return        max(close[T+1..T+W] / close[T] - 1)
  - fwd_max_drawdown      max((close[T] - low[T+1..T+W]) / close[T])  (positive number)
  - fwd_max_alpha         fwd_max_return - SPY equivalent
  - fwd_atr_adj_return    fwd_max_return / atr_pct_14[T]
  - trend_persistence     fraction of bars in [T+1..T+W] with close > close[T]

A composite expansion-quality score combines these via configurable weights.
A binary target (expansion_target) flags the top quantile of composite
within each ticker's rolling temporal window.

ALL future-window slices end at index T+W, never inclusive of T (no leakage
of T's bar into the label of T).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from momentum_expansion.config.momentum_config import (
    CONTEXT_ONLY_TICKERS,
    CORRELATED_FEATURE_REPORT,
    CORRELATION_PRUNE_THRESHOLD,
    FEATURES_COMBINED,
    LABEL_CONFIG,
    LABELS_COMBINED,
    MOMENTUM_CANDIDATE_FILTER_CONFIG,
    PROCESSED_FEAT_DIR,
    RAW_4H_DIR,
    TRAINING_MATRIX_CONFIG,
    TRAINING_MATRIX,
)
from momentum_expansion.data.load_bars import load_4h
from momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H
from momentum_expansion.inference.candidate_filter import momentum_candidate_mask

logger = logging.getLogger(__name__)
_CONTEXT_ONLY_SET = {t.upper() for t in CONTEXT_ONLY_TICKERS}


# ---------------------------------------------------------------------------
# Per-ticker label builder
# ---------------------------------------------------------------------------

def build_ticker_labels_4h(
    *,
    ticker: str,
    df_4h: pd.DataFrame,
    df_bench_4h: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame | None:
    """
    Returns a DataFrame indexed by the ticker's 4H timestamps with label
    columns. NaN for the last W bars where the forward window is incomplete.
    """
    cfg = {**LABEL_CONFIG, **(cfg or {})}
    W = int(cfg["forward_window_4h_bars"])
    persistence_w = int(cfg["trend_persistence_window"])
    weights = cfg["composite_weights"]

    if df_4h is None or len(df_4h) < W + 50:
        return None

    df = df_4h.copy()
    df.columns = [c.lower() for c in df.columns]
    c = df["close"]
    lo = df["low"]
    h = df["high"]

    # Forward max return (using rolling forward window of HIGH for max favorable
    # excursion, since this is the "expansion" we want to capture)
    fwd_high = h.shift(-1).rolling(window=W, min_periods=W).max().shift(-(W - 1))
    fwd_low  = lo.shift(-1).rolling(window=W, min_periods=W).min().shift(-(W - 1))

    fwd_max_return = (fwd_high / c - 1.0)
    fwd_max_drawdown = ((c - fwd_low) / c).clip(lower=0)

    # Forward close path for trend persistence + final return
    closes_fwd = pd.concat(
        [c.shift(-i) for i in range(1, W + 1)], axis=1
    )
    closes_fwd.columns = [f"f{i}" for i in range(1, W + 1)]
    persist_count = (closes_fwd.iloc[:, :persistence_w] > c.values[:, None]).sum(axis=1)
    trend_persistence = persist_count / float(persistence_w)
    fwd_close_return = closes_fwd.iloc[:, -1] / c - 1.0

    # ATR-adjusted return (using ATR pct already in features; compute fallback)
    if "atr_pct_14" in df.columns:
        atr_pct = df["atr_pct_14"].replace(0, np.nan)
    else:
        atr_pct = ((h - lo) / c).rolling(14).mean().replace(0, np.nan)
    fwd_atr_adj_return = fwd_max_return / atr_pct

    # Benchmark forward max return aligned to same timestamps
    bench = df_bench_4h.copy()
    bench.columns = [x.lower() for x in bench.columns]
    bench_close = bench["close"].reindex(df.index, method="ffill")
    bench_high = bench["high"].reindex(df.index, method="ffill")
    bench_fwd_high = bench_high.shift(-1).rolling(window=W, min_periods=W).max().shift(-(W - 1))
    bench_fwd_max_return = (bench_fwd_high / bench_close - 1.0)
    fwd_max_alpha = fwd_max_return - bench_fwd_max_return

    out = pd.DataFrame({
        "fwd_max_return":     fwd_max_return,
        "fwd_max_alpha":      fwd_max_alpha,
        "fwd_atr_adj_return": fwd_atr_adj_return,
        "fwd_max_drawdown":   fwd_max_drawdown,
        "fwd_close_return":   fwd_close_return,
        "trend_persistence":  trend_persistence,
    }, index=df.index)

    # Composite score: cross-sectional rank-normalized within the ticker's
    # rolling window (handled at the combined-pipeline stage where multiple
    # tickers and time history are known). Here we just write raw values.
    return out


# ---------------------------------------------------------------------------
# Combined-pipeline composite + binary target
# ---------------------------------------------------------------------------

def _rolling_quantile_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling-window quantile rank of a series (causal, no future leakage).

    min_periods is ~window/8 (~6 months at 2 4H bars/day) so young listings get a
    binary expansion_target from a partial window instead of NaN — matching the
    ~6-month feature warmup so the two don't gate the universe differently.
    """
    return s.rolling(window, min_periods=max(50, window // 8)).rank(pct=True)


def _cross_sectional_expansion_survival_score(df: pd.DataFrame) -> pd.Series:
    """
    Continuous market-leader target for training.

    This intentionally ranks forward expansion quality across all candidate
    stocks at the same timestamp instead of comparing each ticker only to its
    own history. Higher values mean the row became one of the cleaner/bigger
    forward expansion leaders among that bar's candidate universe.
    """
    ts_level = df.index.names[0] if isinstance(df.index, pd.MultiIndex) else 0

    def _rank(col: str, *, ascending: bool = True) -> pd.Series:
        return df.groupby(level=ts_level)[col].rank(pct=True, ascending=ascending)

    weights = LABEL_CONFIG["composite_weights"]
    alpha_r = _rank("fwd_max_alpha")
    atr_adj_r = _rank("fwd_atr_adj_return")
    persistence_r = _rank("trend_persistence")
    drawdown_good_r = _rank("fwd_max_drawdown", ascending=False)
    return (
        weights["fwd_max_alpha"] * alpha_r
        + weights["fwd_atr_adj_return"] * atr_adj_r
        + weights["trend_persistence"] * persistence_r
        + weights["fwd_max_drawdown"] * drawdown_good_r
    )


def assemble_composite_and_target(
    df_labels: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    df_labels: (timestamp, ticker) MultiIndex frame containing per-ticker
               raw forward stats. Adds:
                 expansion_score  : weighted composite of normalized stats
                 expansion_target : 1 if score is in top binary_top_quantile of
                                    rolling window across (ticker, time), else 0
    """
    cfg = {**LABEL_CONFIG, **(cfg or {})}
    weights = cfg["composite_weights"]
    win = int(cfg["binary_window_bars"])

    df = df_labels.copy()
    required_raw_labels = [
        "fwd_max_return",
        "fwd_max_alpha",
        "fwd_atr_adj_return",
        "fwd_max_drawdown",
        "fwd_close_return",
        "trend_persistence",
    ]
    complete_forward_window = df[required_raw_labels].notna().all(axis=1)

    # Per-ticker rolling rank. This labels each symbol's own best expansion
    # regimes instead of letting high-volatility symbols dominate the target.
    out = df.copy()
    for col in ["fwd_max_alpha", "fwd_atr_adj_return", "trend_persistence", "fwd_max_drawdown"]:
        if col not in df.columns:
            out[f"{col}_r"] = np.nan
            continue
        out[f"{col}_r"] = (
            df.groupby(level="ticker")[col]
              .transform(lambda s: _rolling_quantile_rank(s, win))
        )

    # fwd_max_drawdown is "bad" — invert
    drawdown_term = (1.0 - out["fwd_max_drawdown_r"].fillna(0.5))

    score = (
        weights["fwd_max_alpha"]      * out["fwd_max_alpha_r"].fillna(0.5)
        + weights["fwd_atr_adj_return"] * out["fwd_atr_adj_return_r"].fillna(0.5)
        + weights["trend_persistence"]  * out["trend_persistence_r"].fillna(0.5)
        + weights["fwd_max_drawdown"]   * drawdown_term
    )
    out["expansion_score"] = score

    threshold_q = 1.0 - float(cfg["binary_top_quantile"])
    score_rank = out.groupby(level="ticker")["expansion_score"].transform(
        lambda s: _rolling_quantile_rank(s, win)
    )
    out["expansion_target"] = (score_rank >= threshold_q).astype(float)
    out.loc[score_rank.isna(), "expansion_target"] = np.nan
    out.loc[~complete_forward_window, "expansion_target"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def build_all_labels_4h(
    *,
    tickers: Iterable[str],
    benchmark: str = "SPY",
    out_path: Path = LABELS_COMBINED,
    force: bool = False,
    cfg: dict | None = None,
) -> pd.DataFrame | None:
    if out_path.exists() and not force:
        logger.info("Labels already exist at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)

    bench_4h_path = RAW_4H_DIR / f"{benchmark}.parquet"
    if not bench_4h_path.exists():
        raise FileNotFoundError(f"Benchmark 4H bars missing: {bench_4h_path}")
    df_bench = load_4h(benchmark)

    frames: list[pd.DataFrame] = []
    original_tickers = [str(t).upper() for t in tickers]
    tickers = [t for t in dict.fromkeys(original_tickers) if t not in _CONTEXT_ONLY_SET]
    skipped = sorted(set(original_tickers) - set(tickers))
    if skipped:
        logger.info("Skipping context-only tickers for labels: %s", ", ".join(skipped))
    for i, t in enumerate(tickers, 1):
        try:
            df_4h = load_4h(t)
        except FileNotFoundError:
            continue
        df_lab = build_ticker_labels_4h(
            ticker=t, df_4h=df_4h, df_bench_4h=df_bench, cfg=cfg
        )
        if df_lab is None:
            continue
        df_lab["ticker"] = t
        frames.append(df_lab)
        if i % 25 == 0:
            logger.info("(%d/%d) labels built for %s", i, len(tickers), t)

    if not frames:
        logger.error("No labels built.")
        return None

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    ts_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
    combined = combined.set_index([ts_col, "ticker"]).sort_index()
    combined = assemble_composite_and_target(combined, cfg=cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    logger.info("Saved labels (%d rows) -> %s", len(combined), out_path)
    return combined


# ---------------------------------------------------------------------------
# Build the combined training matrix (features + labels, NaN-pruned)
# ---------------------------------------------------------------------------

def _drop_correlated_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    threshold: float,
    report_path: Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Drop redundant feature columns whose absolute pairwise correlation exceeds
    threshold. Later columns are dropped, preserving FEATURE_COLUMNS_4H order.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        report = pd.DataFrame(columns=["dropped_feature", "kept_feature", "abs_corr"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        return df, [], report

    corr = df[cols].astype("float32").corr().abs()
    to_drop: set[str] = set()
    rows: list[dict[str, object]] = []

    for i, keep_col in enumerate(cols):
        if keep_col in to_drop:
            continue
        for drop_col in cols[i + 1:]:
            if drop_col in to_drop:
                continue
            value = corr.at[keep_col, drop_col]
            if pd.notna(value) and value > threshold:
                to_drop.add(drop_col)
                rows.append({
                    "dropped_feature": drop_col,
                    "kept_feature": keep_col,
                    "abs_corr": float(value),
                })

    dropped = [c for c in cols if c in to_drop]
    report = pd.DataFrame(rows, columns=["dropped_feature", "kept_feature", "abs_corr"])
    if not report.empty:
        report = report.sort_values(["abs_corr", "dropped_feature"], ascending=[False, True])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped, report


def build_training_matrix(
    *,
    features_path: Path = FEATURES_COMBINED,
    labels_path: Path = LABELS_COMBINED,
    out_path: Path = TRAINING_MATRIX,
    corr_report_path: Path = CORRELATED_FEATURE_REPORT,
    corr_threshold: float = CORRELATION_PRUNE_THRESHOLD,
    force: bool = False,
) -> pd.DataFrame | None:
    if out_path.exists() and not force:
        logger.info("Training matrix exists at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features missing at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels missing at {labels_path}")

    feats = pd.read_parquet(features_path)
    labs = pd.read_parquet(labels_path)
    df = feats.join(labs, how="inner")

    keep_cols = [c for c in FEATURE_COLUMNS_4H if c in df.columns] + [
        "fwd_max_return", "fwd_max_alpha", "fwd_atr_adj_return",
        "fwd_max_drawdown", "fwd_close_return", "trend_persistence",
        "expansion_score", "expansion_target",
    ]
    df = df[keep_cols].copy()
    df = df.dropna(subset=["expansion_target"])
    label_cols = [
        "fwd_max_return", "fwd_max_alpha", "fwd_atr_adj_return",
        "fwd_max_drawdown", "fwd_close_return", "trend_persistence",
        "expansion_score",
    ]
    df = df.dropna(subset=[c for c in label_cols if c in df.columns], how="any")
    feature_cols = [c for c in FEATURE_COLUMNS_4H if c in df.columns]
    df = df.dropna(subset=feature_cols, how="any")
    if TRAINING_MATRIX_CONFIG.get("apply_momentum_candidate_filter", False):
        before = len(df)
        df["momentum_candidate"] = momentum_candidate_mask(df).astype(float)
        df = df[df["momentum_candidate"] > 0].copy()
        logger.info(
            "Applied momentum candidate filter: %d -> %d rows (%.2f%% kept)",
            before,
            len(df),
            100.0 * len(df) / max(before, 1),
        )
    df["expansion_survival_score"] = _cross_sectional_expansion_survival_score(df)
    df = df.dropna(subset=["expansion_survival_score"])
    df, dropped, corr_report = _drop_correlated_features(
        df,
        feature_cols=feature_cols,
        threshold=float(corr_threshold),
        report_path=corr_report_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    if dropped:
        logger.info(
            "Dropped %d correlated features at abs(corr) > %.5f. Report -> %s",
            len(dropped),
            float(corr_threshold),
            corr_report_path,
        )
        for row in corr_report.head(25).itertuples(index=False):
            logger.info(
                "  drop %s, keep %s, abs_corr=%.6f",
                row.dropped_feature,
                row.kept_feature,
                row.abs_corr,
            )
    else:
        logger.info(
            "No correlated features exceeded abs(corr) > %.5f. Empty report -> %s",
            float(corr_threshold),
            corr_report_path,
        )
    logger.info("Training matrix: %d rows × %d cols -> %s", len(df), df.shape[1], out_path)
    return df
