"""Metric computation for the ablation harness (WS-E deliverable 1).

All metrics operate on a long (row-per-ticker-per-bar) frame with a
``group_col`` (default ``"timestamp"``) identifying the cross-section a rank
metric is computed within. Every metric is a pure function of
(scores, labels, groups) — no model fitting happens here (D6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rank IC
# ---------------------------------------------------------------------------

def cross_sectional_rank_ic(
    df: pd.DataFrame, *, score_col: str, label_col: str, group_col: str = "timestamp",
    min_group_n: int = 5,
) -> pd.Series:
    """Per-group (per-bar) Spearman rank correlation of score vs label.

    Returns a Series indexed by group value; groups with fewer than
    ``min_group_n`` non-NaN (score, label) pairs, or with zero variance in
    either column, are NaN (never silently dropped from the index, so a
    caller can see how much of the requested window was actually usable).
    """
    sub = df[[group_col, score_col, label_col]].dropna()

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < min_group_n:
            return np.nan
        s, y = g[score_col], g[label_col]
        if s.nunique() < 2 or y.nunique() < 2:
            return np.nan
        return float(s.corr(y, method="spearman"))

    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby(group_col, sort=True).apply(_ic, include_groups=False)


def mean_rank_ic(ic_by_group: pd.Series) -> float:
    valid = ic_by_group.dropna()
    return float(valid.mean()) if len(valid) else float("nan")


def rank_ic_icir(ic_by_group: pd.Series) -> float:
    """IC information ratio: mean(IC) / std(IC) across groups."""
    valid = ic_by_group.dropna()
    if len(valid) < 2 or valid.std(ddof=1) == 0:
        return float("nan")
    return float(valid.mean() / valid.std(ddof=1))


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------

def _dcg_at_k(relevance_in_rank_order: np.ndarray, k: int) -> float:
    rel = relevance_in_rank_order[:k]
    if len(rel) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k_group(score: pd.Series, label: pd.Series, k: int = 10) -> float:
    """NDCG@k for one cross-section using linear gain (label can be negative;
    classic exp2-gain NDCG requires non-negative relevance, which forward
    returns/alpha are not)."""
    sub = pd.DataFrame({"score": score, "label": label}).dropna()
    if len(sub) < 2:
        return np.nan
    by_score = sub.sort_values("score", ascending=False)["label"].to_numpy()
    by_ideal = sub.sort_values("label", ascending=False)["label"].to_numpy()
    dcg = _dcg_at_k(by_score, k)
    idcg = _dcg_at_k(by_ideal, k)
    if idcg == 0:
        return np.nan
    return dcg / idcg


def ndcg_at_k(
    df: pd.DataFrame, *, score_col: str, label_col: str, group_col: str = "timestamp",
    k: int = 10, min_group_n: int = 5,
) -> pd.Series:
    sub = df[[group_col, score_col, label_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def _g(g: pd.DataFrame) -> float:
        if len(g) < min_group_n:
            return np.nan
        return ndcg_at_k_group(g[score_col], g[label_col], k=k)

    return sub.groupby(group_col, sort=True).apply(_g, include_groups=False)


# ---------------------------------------------------------------------------
# Top-N selection metrics
# ---------------------------------------------------------------------------

def top_n_per_group(
    df: pd.DataFrame, *, score_col: str, group_col: str = "timestamp", n: int = 10,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Rows selected by the top-``n`` score within each group. Ties broken by
    original row order (stable sort), matching the house rank-selection
    convention (RANKING_CONFIG) rather than an arbitrary tiebreak."""
    sub = df.dropna(subset=[score_col])
    return (
        sub.sort_values([group_col, score_col], ascending=[True, False], kind="mergesort")
           .groupby(group_col, sort=True)
           .head(n)
    )


def top_n_forward_metrics(
    df: pd.DataFrame, *, score_col: str, label_col: str, group_col: str = "timestamp",
    n: int = 10, win_col: str | None = None,
) -> dict:
    """Mean forward ``label_col`` and win rate of the top-N selection, pooled
    across all selected rows (not averaged-of-averages) — plus per-group means
    for downstream bootstrap/period-level reporting."""
    win_col = win_col or label_col
    picks = top_n_per_group(df, score_col=score_col, group_col=group_col, n=n)
    if picks.empty:
        return {"n_picks": 0, "mean_label": np.nan, "win_rate": np.nan, "per_group_mean": pd.Series(dtype=float)}
    per_group_mean = picks.groupby(group_col)[label_col].mean()
    return {
        "n_picks": int(len(picks)),
        "mean_label": float(picks[label_col].mean()),
        "win_rate": float((picks[win_col] > 0).mean()),
        "per_group_mean": per_group_mean,
    }


def turnover(
    df: pd.DataFrame, *, score_col: str, group_col: str = "timestamp", ticker_col: str = "ticker",
    n: int = 10,
) -> float:
    """Mean bar-over-bar churn of the top-N selection: fraction of the set
    that is NOT shared with the immediately preceding group's top-N set
    (0 = identical picks every bar, 1 = fully turns over every bar)."""
    picks = top_n_per_group(df, score_col=score_col, group_col=group_col, n=n, ticker_col=ticker_col)
    if picks.empty:
        return float("nan")
    sets_by_group = picks.groupby(group_col)[ticker_col].apply(set).sort_index()
    if len(sets_by_group) < 2:
        return float("nan")
    churns = []
    prev = None
    for s in sets_by_group:
        if prev is not None and len(prev) > 0:
            shared = len(s & prev)
            churns.append(1.0 - shared / max(len(prev), 1))
        prev = s
    return float(np.mean(churns)) if churns else float("nan")


# ---------------------------------------------------------------------------
# Return-series risk metrics (Sharpe / MaxDD) on the top-N per-group mean
# return series
# ---------------------------------------------------------------------------

def sharpe_ratio(period_returns: pd.Series, *, periods_per_year: float) -> float:
    r = period_returns.dropna()
    std = r.std(ddof=1) if len(r) >= 2 else 0.0
    if len(r) < 2 or not np.isfinite(std) or std < 1e-12:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(period_returns: pd.Series) -> float:
    """Max drawdown of the cumulative-product equity curve implied by a
    period-return series. Returns a positive fraction (e.g. 0.15 = -15% peak
    to trough)."""
    r = period_returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max())
