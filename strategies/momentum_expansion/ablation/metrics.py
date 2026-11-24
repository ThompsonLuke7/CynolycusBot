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


# ---------------------------------------------------------------------------
# Frozen-evaluator metrics (checklist 0.3). Vectorised over a long frame with a
# bar key; every function takes the frame and column names, never a model.
# ---------------------------------------------------------------------------

def within_bar_rank_ic(df: pd.DataFrame, *, score_col: str, label_col: str, group_col: str = "timestamp",
                       min_group_n: int = 5) -> pd.Series:
    """Per-bar Spearman IC, vectorised (same contract as ``cross_sectional_rank_ic``:
    NaN for bars with < ``min_group_n`` pairs or no variance)."""
    sub = df[[group_col, score_col, label_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    g = sub.groupby(group_col, sort=True)
    rs = g[score_col].rank(method="average")
    ry = g[label_col].rank(method="average")
    t = pd.DataFrame({group_col: sub[group_col], "x": rs, "y": ry})
    t["xy"], t["xx"], t["yy"] = t.x * t.y, t.x * t.x, t.y * t.y
    s = t.groupby(group_col, sort=True).agg(n=("x", "size"), x=("x", "sum"), y=("y", "sum"),
                                            xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"))
    cov = s.xy - s.x * s.y / s.n
    vx, vy = s.xx - s.x ** 2 / s.n, s.yy - s.y ** 2 / s.n
    ic = cov / np.sqrt(vx * vy)
    ic[(s.n < min_group_n) | (vx <= 1e-12) | (vy <= 1e-12)] = np.nan
    return ic.astype(float)


def top_k_mask(df: pd.DataFrame, *, score_col: str, k: int, group_col: str = "timestamp") -> pd.Series:
    """Boolean mask of each bar's top-``k`` scores (stable: ties keep row order)."""
    rank = df.groupby(group_col, sort=False)[score_col].rank(method="first", ascending=False)
    return (rank <= k) & df[score_col].notna()


def per_bar_mean(df: pd.DataFrame, mask: pd.Series, value_col: str, *, group_col: str = "timestamp") -> pd.Series:
    """Mean of ``value_col`` over the masked rows of each bar (bars with no rows dropped)."""
    sub = df.loc[mask & df[value_col].notna(), [group_col, value_col]]
    return sub.groupby(group_col, sort=True)[value_col].mean()


def large_winner_recall(df: pd.DataFrame, mask: pd.Series, *, ret_col: str, group_col: str = "timestamp",
                        top_frac: float = 0.10) -> tuple[pd.Series, pd.Series]:
    """Per bar: share of the bar's top-``top_frac`` forward returns that the selection caught,
    and the random-selection expectation (selected / names) to compare it with."""
    sub = df.loc[df[ret_col].notna(), [group_col, ret_col]].assign(_sel=mask.loc[df[ret_col].notna()].to_numpy())
    pct = sub.groupby(group_col)[ret_col].rank(pct=True, method="first")
    sub["_win"] = pct > 1 - top_frac
    g = sub.groupby(group_col, sort=True)
    caught = g.apply(lambda x: (x["_win"] & x["_sel"]).sum(), include_groups=False)
    winners = g["_win"].sum()
    recall = (caught / winners.replace(0, np.nan)).astype(float)
    random = (g["_sel"].sum() / g.size()).astype(float)
    return recall, random


def calibration_report(prob: pd.Series, outcome: pd.Series, *, bins: int = 10) -> dict:
    """Brier score and expected calibration error of a probability against a 0/1 outcome."""
    d = pd.DataFrame({"p": pd.to_numeric(prob, errors="coerce"), "y": pd.to_numeric(outcome, errors="coerce")}).dropna()
    if d.empty:
        return {"n": 0, "brier": float("nan"), "ece": float("nan"), "bins": []}
    d["bin"] = np.minimum((d.p.clip(0, 1) * bins).astype(int), bins - 1)
    t = d.groupby("bin").agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    ece = float((t.n * (t.p - t.y).abs()).sum() / t.n.sum())
    return {"n": int(len(d)), "brier": float(((d.p - d.y) ** 2).mean()), "ece": ece,
            "bins": t.reset_index().round(4).to_dict(orient="records")}


def residualise_within_bar(df: pd.DataFrame, *, score_col: str, control_cols: list[str],
                           group_col: str = "timestamp") -> pd.Series:
    """The pre-registered neutral score (eval_stage2._residualise): the within-bar OLS
    residual of the score's percentile rank on the controls' percentile ranks, one
    regression per bar solved in one batch. A missing control ranks 0.5."""
    codes, _ = pd.factorize(df[group_col], sort=True)
    n_bars = int(codes.max()) + 1 if len(codes) else 0
    out = np.full(len(df), np.nan)
    rows = np.flatnonzero(np.isfinite(df[score_col].to_numpy(float)))
    if n_bars == 0 or len(rows) == 0:
        return pd.Series(out, index=df.index)
    g = df.groupby(group_col, sort=False)
    X = np.column_stack([np.ones(len(df))] + [g[c].rank(pct=True).fillna(0.5).to_numpy(float)
                                              for c in control_cols])[rows]
    y = df[score_col].iloc[rows].groupby(codes[rows]).rank(method="average", pct=True).to_numpy(float)
    b = codes[rows]
    k = X.shape[1]
    xtx, xty = np.empty((n_bars, k, k)), np.empty((n_bars, k))
    for i in range(k):
        xty[:, i] = np.bincount(b, X[:, i] * y, minlength=n_bars)
        for j in range(i, k):
            xtx[:, i, j] = xtx[:, j, i] = np.bincount(b, X[:, i] * X[:, j], minlength=n_bars)
    beta = np.linalg.solve(xtx + 1e-9 * np.eye(k), xty[..., None])[..., 0]
    out[rows] = y - np.einsum("ij,ij->i", X, beta[b])
    return pd.Series(out, index=df.index)
