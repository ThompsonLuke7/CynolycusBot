"""Statistical discipline for the ablation harness (WS-E deliverable 3):
week-block bootstrap confidence intervals and BH-FDR.

BH-FDR is imported from ``scripts.confluence_discovery.search.bh_fdr`` and
re-exported here rather than reimplemented (repo convention: reuse existing
project abstractions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.confluence_discovery.search import bh_fdr  # re-exported, not reimplemented

__all__ = ["bh_fdr", "week_block_bootstrap_ci", "week_of"]


def week_of(ts: pd.Series) -> pd.Series:
    """ISO (year, week) key used to block-resample by calendar week — blocking
    respects the autocorrelation of overlapping forward-looking labels (a
    25x4H-bar forward window means adjacent bars' labels are not independent
    observations; a plain per-row bootstrap would understate variance)."""
    t = pd.to_datetime(ts, utc=True)
    iso = t.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def week_block_bootstrap_ci(
    values: pd.Series,
    week: pd.Series,
    *,
    statistic=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.10,
    seed: int = 42,
) -> dict:
    """Week-block bootstrap CI for ``statistic(values)``.

    Resamples whole weeks with replacement (not individual rows), so that
    within-week autocorrelation from overlapping forward windows is preserved
    in each bootstrap draw rather than averaged away. Returns the point
    estimate, the ``alpha``-level CI (default 90%, i.e. q<=0.10 two-sided
    FDR-compatible convention used elsewhere in this repo), the number of
    distinct weeks, and whether the CI excludes zero.
    """
    df = pd.DataFrame({"value": values, "week": week}).dropna(subset=["value"])
    weeks = df["week"].dropna().unique()
    n_weeks = len(weeks)
    if n_weeks == 0 or df.empty:
        return {
            "point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "n_weeks": 0, "n_obs": 0, "excludes_zero": False,
        }

    point = float(statistic(df["value"].to_numpy()))
    if n_weeks < 2:
        # Not enough independent blocks to bootstrap; report the point
        # estimate with an (nan, nan) CI rather than fabricate a spread from
        # one block.
        return {
            "point": point, "ci_lo": np.nan, "ci_hi": np.nan,
            "n_weeks": n_weeks, "n_obs": len(df), "excludes_zero": False,
        }

    rng = np.random.default_rng(seed)
    by_week = {w: g["value"].to_numpy() for w, g in df.groupby("week")}
    week_keys = np.array(list(by_week.keys()))
    boot_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(week_keys, size=n_weeks, replace=True)
        sample = np.concatenate([by_week[w] for w in draw])
        boot_stats[b] = statistic(sample)

    lo = float(np.nanquantile(boot_stats, alpha / 2))
    hi = float(np.nanquantile(boot_stats, 1 - alpha / 2))
    return {
        "point": point, "ci_lo": lo, "ci_hi": hi,
        "n_weeks": n_weeks, "n_obs": len(df),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }
