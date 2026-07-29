"""Capital-controlled metrics for the pyramiding study.

The point of the study is that pyramiding deploys MORE capital, which in an
up-market mechanically produces more total P&L while looking like better
trading. Every table therefore carries both:

  * ``sharpe_daily`` — annualized Sharpe of the DAILY mark-to-market net-P&L
    series. Sharpe is scale-invariant, so deploying 2x the capital does not
    move it.
  * ``ret_per_dollar_deployed`` — period net P&L / time-average open cost
    basis. This is the per-dollar efficiency number; an arm that beats
    baseline on ``total_pnl_net`` but not on this one is deploying more
    capital, not trading better.

plus the capital-footprint columns (``avg_deployed``, ``peak_deployed``,
``max_concurrent_notional``) needed to see that for yourself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_naive_utc(ts) -> np.datetime64:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return np.datetime64(t)


def window_mask(ts_master: np.ndarray, lo, hi) -> np.ndarray:
    return (ts_master >= _to_naive_utc(lo)) & (ts_master <= _to_naive_utc(hi))


def arm_metrics(
    ts_master: np.ndarray, pnl: np.ndarray, deployed: np.ndarray,
    positions: pd.DataFrame, *, lo=None, hi=None,
) -> dict:
    """Metrics for one arm over one window.

    P&L and deployed capital are windowed on the BAR clock (P&L is attributed
    to the bar that earned it, mark-to-market), while trade-level stats
    (count, win rate, hold, turnover) are windowed on ENTRY timestamp — the
    same convention ``regime_policy/run_study.py`` uses (``signal_ts``).
    """
    m = np.ones(len(ts_master), dtype=bool) if lo is None else window_mask(ts_master, lo, hi)
    ts_w, pnl_w, dep_w = ts_master[m], pnl[m], deployed[m]
    if len(ts_w) == 0:
        return {"trades": 0, "n_bars": 0}

    pos = positions
    if lo is not None and not pos.empty:
        pos = pos[(pos["entry_ts"] >= _to_naive_utc(lo)) & (pos["entry_ts"] <= _to_naive_utc(hi))]

    total_pnl = float(pnl_w.sum())
    avg_dep = float(dep_w.mean())
    peak_dep = float(dep_w.max())

    daily = pd.Series(pnl_w, index=pd.DatetimeIndex(ts_w)).resample("1D").sum()
    daily = daily[daily.index.dayofweek < 5]  # drop weekend buckets (no 4H bars)
    sharpe = (float(daily.mean() / daily.std() * np.sqrt(252))
              if len(daily) > 2 and daily.std() > 0 else float("nan"))

    eq = daily.cumsum()
    dd_dollars = float((eq - eq.cummax()).min()) if len(eq) else 0.0

    fills = float(pos["fill_notional"].sum()) if not pos.empty else 0.0
    return {
        "trades": int(len(pos)),
        "n_bars": int(len(ts_w)),
        "total_pnl_net": round(total_pnl, 2),
        "avg_deployed": round(avg_dep, 2),
        "peak_deployed": round(peak_dep, 2),
        "max_concurrent_notional": round(peak_dep, 2),
        "ret_per_dollar_deployed": round(total_pnl / avg_dep, 6) if avg_dep > 0 else float("nan"),
        "sharpe_daily": round(sharpe, 3) if np.isfinite(sharpe) else float("nan"),
        "max_dd_dollars": round(dd_dollars, 2),
        "max_dd_pct_of_avg_deployed": (round(dd_dollars / avg_dep * 100, 3) if avg_dep > 0 else float("nan")),
        "win_rate": round(float((pos["pnl_net"] > 0).mean()), 4) if len(pos) else float("nan"),
        "avg_trade_pct": round(float(pos["ret_on_initial"].mean()) * 100, 4) if len(pos) else float("nan"),
        "avg_bars_held": round(float(pos["bars_held"].mean()), 2) if len(pos) else float("nan"),
        "turnover_x_avg_deployed": round(fills / avg_dep, 3) if avg_dep > 0 else float("nan"),
        "total_adds": int(pos["n_adds"].sum()) if len(pos) else 0,
        "pct_positions_with_add": (round(float((pos["n_adds"] > 0).mean()) * 100, 2) if len(pos) else float("nan")),
        "avg_adds_per_position": round(float(pos["n_adds"].mean()), 3) if len(pos) else float("nan"),
        "total_fees": round(float(pos["fees"].sum()), 2) if len(pos) else 0.0,
    }


def weekly_pnl(ts_master: np.ndarray, pnl: np.ndarray, *, lo=None, hi=None) -> pd.Series:
    """Weekly net-$ P&L on the bar clock, keyed by ISO week (the same key
    ``ablation.bootstrap.week_of`` produces)."""
    from strategies.momentum_expansion.ablation.bootstrap import week_of

    m = np.ones(len(ts_master), dtype=bool) if lo is None else window_mask(ts_master, lo, hi)
    if not m.any():
        return pd.Series(dtype=float)
    s = pd.Series(pnl[m], index=pd.DatetimeIndex(ts_master[m]))
    keys = week_of(pd.Series(s.index))  # week_of localizes naive input to UTC
    return s.groupby(keys.to_numpy()).sum()


def weekly_diff_bootstrap(base_weekly: pd.Series, arm_weekly: pd.Series,
                          *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Week-block bootstrap CI + two-sided p-value on the mean weekly net-$
    P&L difference (arm - baseline).

    Structurally identical to ``regime_policy/engine.weekly_diff_bootstrap``
    (same reuse of ``week_block_bootstrap_ci``, same union-of-weeks fill-0,
    same bootstrap p-value); only the input series differ (bar-clock weekly
    P&L here vs. exit-date weekly P&L there, because this engine produces a
    true mark-to-market series).
    """
    from strategies.momentum_expansion.ablation.bootstrap import week_block_bootstrap_ci

    weeks = base_weekly.index.union(arm_weekly.index)
    diff = arm_weekly.reindex(weeks, fill_value=0.0) - base_weekly.reindex(weeks, fill_value=0.0)
    if diff.empty:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_weeks": 0,
                "excludes_zero": False, "p_value": np.nan}
    boot = week_block_bootstrap_ci(pd.Series(diff.to_numpy()), pd.Series(diff.index.to_numpy()),
                                   n_boot=n_boot, seed=seed)
    p_value = np.nan
    if boot["n_weeks"] >= 2:
        rng = np.random.default_rng(seed + 1)
        vals = diff.to_numpy()
        boots = np.empty(n_boot)
        for b in range(n_boot):
            boots[b] = rng.choice(vals, size=len(vals), replace=True).mean()
        p_ge0, p_le0 = float(np.mean(boots >= 0)), float(np.mean(boots <= 0))
        p_value = float(min(1.0, 2 * min(p_ge0, p_le0)))
    return {**boot, "p_value": p_value}
