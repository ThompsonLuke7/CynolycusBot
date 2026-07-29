"""Training-free local screen for the market-regime / sector-context feature
block (WS-E deliverable 2 — plan defect D6: no local training).

METHODOLOGICAL TAXONOMY (read before touching this file)
----------------------------------------------------------
The new regime/sector columns split into three groups that must be evaluated
differently, or a structural zero gets misread as a null result:

1. MARKET-WIDE (same value for every ticker on a given bar):
   ``risk_appetite_z``, ``liquidity_stress_z``, ``credit_risk_z``,
   ``sector_dispersion_21d``, ``market_breadth_z``. Cross-sectional rank IC on these is exactly
   zero by construction (a constant has no cross-sectional variation to
   correlate with) — evaluated instead as regime-conditional expectancy:
   quintile-bucket bars by the feature, and measure the forward outcome of
   the EXISTING deployed model's top-N selection within each bucket
   (``models/expansion_v1/oof_preds.parquet`` — real walk-forward
   out-of-fold scores from the already-trained xgb_classifier seed45
   winner, not a locally-trained model). Question: "does the existing
   signal work better in some regimes than others."

2. SECTOR-LEVEL (constant within a sector, varies across sectors):
   ``sector_excess_21d/63d``, ``sector_rank_21d/63d``, ``sector_rs_accel``.
   Cross-sectional rank IC is valid; also reported de-overlapped to one row
   per (timestamp, sector_etf) to avoid the IC being inflated purely by how
   many tickers happen to share a sector on a given bar.

   CORRECTION TO THE PLAN'S OWN TAXONOMY: the task brief (and a literal
   reading of the WS-E plan prose) lists ``sector_breadth_20``/
   ``sector_breadth_50`` alongside the market-wide group. That is not what
   the WS-B code actually does — ``build_ticker_features_4h`` sources them
   from ``sector_state.parquet``'s ``above_20d``/``above_50d``, joined on
   the TICKER's OWN resolved sector ETF (feature_matrix_4h.py's
   ``sector_join["above_20d"]``), i.e. one row per (date, sector_etf), not
   one row per date. They are empirically NOT constant across tickers on a
   bar (see ``sector_breadth_disagreement_rate`` below) and are grouped
   here with the sector-level features (group 2), not group 1. This module
   treats that as ground truth over the plan prose.

   UPDATE (post-verification): WS-B has since renamed these two columns from
   ``sector_breadth_20``/``sector_breadth_50`` to ``sector_above_20d``/
   ``sector_above_50d`` at the source — the "breadth" name was itself the
   bug this correction flagged (they were never breadth; genuine breadth is
   the market-wide ``breadth_z`` composite in ``daily_regime.parquet``,
   which WS-B had built but never joined). WS-B now also joins that real
   composite as ``market_breadth_z`` (group 1, genuinely market-wide). This
   module's naming and grouping below track that fix; the taxonomy argument
   above is preserved as the historical record of why ``sector_above_20d/50d``
   belong in group 2, not group 1.

3. INTERACTIONS (vary per ticker): ``int_rs_spy_20_x_sector_rank_63d``,
   ``int_ret_20_x_risk_appetite_z``, ``int_breakout_20_x_liquidity_stress_z``.
   Cross-sectional rank IC is valid and is the primary metric.

Data sources (all read-only; nothing here writes to a production path):
  * ``strategies/momentum_expansion/data/processed/training_matrix_4h.parquet``
    — engineered features (rs_spy_20, ret_20, breakout_20, ...) + labels.
  * ``strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet``
    — the deployed model's real walk-forward OOF scores (2022-11-14 ..
    2026-05-14), used ONLY as "the existing signal" for group 1's
    conditional-expectancy analysis. No model is fit here.
  * ``Data/shared/market_regime/{daily_regime,sector_state}.parquet`` (WS-A).

Labels: per the task instruction, ``trend_persistence`` is EXCLUDED (a known
forward-looking/leaky label per LIVING_SUMMARY's meta-breadth retraction).
Primary label is ``fwd_max_alpha`` (the composite's largest weight, 0.40, and
the confluence-discovery module's own primary continuous outcome); auxiliary
labels ``fwd_atr_adj_return``, ``expansion_score``, ``fwd_close_return`` (win
rate), and ``fwd_max_drawdown`` (bucket drawdown reporting) are also used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.config.momentum_config import (
    MODELS_DIR,
    TRAINING_MATRIX,
)
from strategies.momentum_expansion.features import feature_matrix_4h as fm4h
from strategies.momentum_expansion.features.feature_matrix_4h import _asof_join_regime
from signals.market_regime.sector_map import SECTOR_MAP, sector_etf_for

from . import metrics as M
from .bootstrap import bh_fdr, week_block_bootstrap_ci, week_of

logger = logging.getLogger(__name__)

OOF_PREDS_PATH = MODELS_DIR / "oof_preds.parquet"

MARKET_WIDE_FEATURES: list[str] = [
    "risk_appetite_z", "liquidity_stress_z", "credit_risk_z", "sector_dispersion_21d",
    "market_breadth_z",
]
SECTOR_LEVEL_FEATURES: list[str] = [
    "sector_excess_21d", "sector_excess_63d", "sector_rank_21d", "sector_rank_63d",
    "sector_rs_accel", "sector_above_20d", "sector_above_50d",
]
INTERACTION_FEATURES: list[str] = [
    "int_rs_spy_20_x_sector_rank_63d",
    "int_ret_20_x_risk_appetite_z",
    "int_breakout_20_x_liquidity_stress_z",
]

LABEL_COLUMNS = ["fwd_max_return", "fwd_max_alpha", "fwd_atr_adj_return",
                 "fwd_max_drawdown", "fwd_close_return", "expansion_score"]
PRIMARY_LABEL = "fwd_max_alpha"
BANNED_LABEL = "trend_persistence"  # forward-looking / leaky — never used as a target here


# ---------------------------------------------------------------------------
# Loading (read-only; never writes to a production path)
# ---------------------------------------------------------------------------

TRAINING_MATRIX_LOAD_COLUMNS = [
    "rs_spy_20", "ret_20", "breakout_20", "dollar_vol_pctile_252", "low_price_flag", "is_etf",
    *LABEL_COLUMNS,
]


def load_training_matrix(
    *, matrix_path: Path = TRAINING_MATRIX, columns: list[str] | None = None,
) -> pd.DataFrame:
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found at {matrix_path}. This screen reads it "
            "read-only; run the existing momentum_expansion pipeline first."
        )
    cols = columns if columns is not None else TRAINING_MATRIX_LOAD_COLUMNS
    df = pd.read_parquet(matrix_path, columns=cols)
    df = df.reset_index()  # -> timestamp, ticker columns
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_oof_scores(*, oof_path: Path = OOF_PREDS_PATH) -> pd.DataFrame:
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Deployed model OOF predictions not found at {oof_path}. Group-1 "
            "(market-wide regime-conditional) analysis needs a real, already-"
            "trained 'existing signal' to bucket — it does not train one here (D6)."
        )
    df = pd.read_parquet(oof_path, columns=["score", "fwd_max_alpha", "fwd_close_return", "fwd_max_drawdown"])
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.rename(columns={"score": "oof_score"})


# ---------------------------------------------------------------------------
# Regime/sector join (reuses fm4h's loaders + _asof_join_regime; does not
# reimplement the as-of-join logic or the sector resolution policy)
# ---------------------------------------------------------------------------

_SECTOR_STATE_COLS = ["excess_21d", "excess_63d", "rank_21d", "rank_63d", "rs_accel",
                      "above_20d", "above_50d"]
_SECTOR_STATE_RENAME = {
    "excess_21d": "sector_excess_21d", "excess_63d": "sector_excess_63d",
    "rank_21d": "sector_rank_21d", "rank_63d": "sector_rank_63d",
    "rs_accel": "sector_rs_accel", "above_20d": "sector_above_20d", "above_50d": "sector_above_50d",
}


def _resolve_sector_curated_only(ticker: str) -> str | None:
    """Curated-map-only resolution (plan D2: empirical resolver stays off by
    default). Uses ``signals.market_regime.sector_map.sector_etf_for`` with
    ``resolver_enabled=False`` explicitly, rather than relying on the module
    default (belt-and-suspenders for a research script)."""
    return sector_etf_for(ticker, resolver_enabled=False)


def join_market_wide_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add just the market-wide daily_regime columns (``risk_appetite_z``,
    ``liquidity_stress_z``, ``credit_risk_z``, ``sector_dispersion_21d``,
    ``market_breadth_z``, ``regime_available``, ``regime_stale_days``) to
    ``df`` — needs only
    ``timestamp`` (tz-aware UTC); no ``ticker``/sector resolution required,
    since these are ticker-independent by construction (group 1 of the
    taxonomy). Used standalone for the group-1 screen against
    ``oof_preds.parquet``, which doesn't carry the ticker-level raw features
    the sector join / interactions need.
    """
    out = df.copy()
    ts_utc = pd.DatetimeIndex(out["timestamp"])

    daily_regime = fm4h._load_daily_regime_table()
    market_join = _asof_join_regime(
        ts_utc, daily_regime,
        ["risk_appetite_z", "liquidity_stress_z", "credit_risk_z", "sector_dispersion_raw", "breadth_z"],
    )
    out["risk_appetite_z"] = market_join["risk_appetite_z"].to_numpy()
    out["liquidity_stress_z"] = market_join["liquidity_stress_z"].to_numpy()
    out["credit_risk_z"] = market_join["credit_risk_z"].to_numpy()
    out["sector_dispersion_21d"] = market_join["sector_dispersion_raw"].to_numpy()
    out["market_breadth_z"] = market_join["breadth_z"].to_numpy()

    bar_session_date = pd.to_datetime(
        pd.Series(ts_utc.tz_convert("America/New_York").date, index=out.index)
    )
    regime_row_date = pd.to_datetime(market_join["_regime_date"]).reset_index(drop=True)
    out["regime_available"] = regime_row_date.notna().astype(float).to_numpy()
    out["regime_stale_days"] = (bar_session_date.reset_index(drop=True) - regime_row_date).dt.days.astype(float).to_numpy()
    return out


def join_regime_and_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every REGIME_FEATURE_COLUMNS_4H-equivalent column to ``df``
    (needs ``timestamp`` (tz-aware UTC) and ``ticker`` columns, and — for the
    interactions — ``rs_spy_20``/``ret_20``/``breakout_20`` already present).

    Byte-for-byte the same as-of-join semantics as
    ``build_ticker_features_4h`` (WS-B), just batched across the combined
    matrix by sector group instead of looping per ticker.
    """
    out = join_market_wide_features(df)
    ts_utc = pd.DatetimeIndex(out["timestamp"])

    sector_table = fm4h._load_sector_state_table()
    unique_tickers = out["ticker"].unique()
    sector_lookup = {t: _resolve_sector_curated_only(t) for t in unique_tickers}
    resolved_sector = out["ticker"].map(sector_lookup)
    out["sector_etf"] = resolved_sector
    for col in _SECTOR_STATE_RENAME.values():
        out[col] = np.nan
    if sector_table is not None:
        for etf in resolved_sector.dropna().unique():
            mask = (resolved_sector == etf).to_numpy()
            if not mask.any():
                continue
            sub_table = sector_table[sector_table["sector_etf"] == etf]
            bar_index = pd.DatetimeIndex(ts_utc[mask])
            joined = _asof_join_regime(bar_index, sub_table, _SECTOR_STATE_COLS)
            for src, dst in _SECTOR_STATE_RENAME.items():
                out.loc[mask, dst] = joined[src].to_numpy()

    out["int_rs_spy_20_x_sector_rank_63d"] = out["rs_spy_20"] * out["sector_rank_63d"]
    out["int_ret_20_x_risk_appetite_z"] = out["ret_20"] * out["risk_appetite_z"]
    out["int_breakout_20_x_liquidity_stress_z"] = out["breakout_20"] * out["liquidity_stress_z"]
    return out


def sector_breadth_disagreement_rate(sector_table: pd.DataFrame | None = None) -> dict:
    """Empirical check backing the taxonomy correction above: on what fraction
    of (date) rows do the 11 sector ETFs NOT all agree on above_20d/above_50d?
    A near-0% disagreement rate would support treating sector_above_20d/50d as
    effectively market-wide; a high rate confirms they are genuinely
    sector-varying and belong in group 2."""
    sector_table = sector_table if sector_table is not None else fm4h._load_sector_state_table()
    if sector_table is None or sector_table.empty:
        return {"disagreement_rate_20d": np.nan, "disagreement_rate_50d": np.nan, "n_dates": 0}
    g = sector_table.groupby("date")
    disagree_20 = g["above_20d"].apply(lambda s: s.nunique() > 1)
    disagree_50 = g["above_50d"].apply(lambda s: s.nunique() > 1)
    return {
        "disagreement_rate_20d": float(disagree_20.mean()),
        "disagreement_rate_50d": float(disagree_50.mean()),
        "n_dates": int(g.ngroups),
    }


# ---------------------------------------------------------------------------
# Group 1 — market-wide: regime-conditional expectancy of the EXISTING
# deployed model's top-N selection
# ---------------------------------------------------------------------------

def market_wide_conditional_expectancy(
    period_df: pd.DataFrame,
    *,
    feature_col: str,
    score_col: str = "oof_score",
    label_col: str = "fwd_max_alpha",
    win_col: str = "fwd_close_return",
    dd_col: str = "fwd_max_drawdown",
    n_buckets: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
    """Quintile-bucket ``period_df`` by ``feature_col`` (a market-wide constant
    at each timestamp), and report the top-N-by-``score_col`` forward outcome
    within each bucket. Returns one row per bucket plus a final 'spread' row
    (top-bucket minus bottom-bucket mean_label)."""
    sub = period_df.dropna(subset=[feature_col, score_col]).copy()
    if sub.empty or sub[feature_col].nunique() < 2:
        return pd.DataFrame()
    try:
        sub["_bucket"] = pd.qcut(sub[feature_col], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows = []
    for b in sorted(sub["_bucket"].dropna().unique()):
        bucket_df = sub[sub["_bucket"] == b]
        m = M.top_n_forward_metrics(
            bucket_df, score_col=score_col, label_col=label_col,
            group_col="timestamp", n=top_n, win_col=win_col,
        )
        dd = M.top_n_forward_metrics(
            bucket_df, score_col=score_col, label_col=dd_col,
            group_col="timestamp", n=top_n,
        )
        rows.append({
            "bucket": int(b),
            "feature_lo": float(bucket_df[feature_col].min()),
            "feature_hi": float(bucket_df[feature_col].max()),
            "n_bars": int(bucket_df["timestamp"].nunique()),
            "n_picks": m["n_picks"],
            "mean_fwd_alpha": m["mean_label"],
            "win_rate": m["win_rate"],
            "mean_fwd_drawdown": dd["mean_label"],
        })
    out = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    if len(out) >= 2:
        top_val = out.iloc[-1]["mean_fwd_alpha"]
        bot_val = out.iloc[0]["mean_fwd_alpha"]
        spread_row = {c: np.nan for c in out.columns}
        spread_row.update({"bucket": "top_minus_bottom", "mean_fwd_alpha": top_val - bot_val})
        out = pd.concat([out, pd.DataFrame([spread_row])], ignore_index=True)
    return out


_EMPTY_SPREAD_RESULT = {
    "point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_weeks_top": 0, "n_weeks_bot": 0,
    "excludes_zero": False, "p_value": np.nan,
}


def market_wide_spread_bootstrap(
    period_df: pd.DataFrame,
    *,
    feature_col: str,
    score_col: str = "oof_score",
    label_col: str = "fwd_max_alpha",
    n_buckets: int = 5,
    top_n: int = 10,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """Week-block bootstrap CI on the top-bucket-minus-bottom-bucket spread of
    the top-N selection's mean forward label.

    This is an UNPAIRED two-sample block bootstrap, not a paired one: a
    market-wide regime feature (risk_appetite_z, liquidity_stress_z, ...) is
    strongly autocorrelated, so it typically stays in one quintile for
    multi-week stretches — the top-quintile weeks and bottom-quintile weeks
    of a 6-month window are almost always two DISJOINT sets of calendar
    weeks, not a shared set of weeks with two paired observations each.
    Pairing by common week (as an initial version of this function did)
    therefore finds ~0 overlapping weeks and silently produces an
    uninformative empty CI. Instead: resample each bucket's OWN weeks with
    replacement (preserving within-week autocorrelation, per the module's
    statistical-discipline requirement) and bootstrap the difference of the
    two independently-resampled means.
    """
    sub = period_df.dropna(subset=[feature_col, score_col]).copy()
    if sub.empty or sub[feature_col].nunique() < 2:
        return dict(_EMPTY_SPREAD_RESULT)
    try:
        sub["_bucket"] = pd.qcut(sub[feature_col], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return dict(_EMPTY_SPREAD_RESULT)
    top_bucket = sub["_bucket"].max()
    bot_bucket = sub["_bucket"].min()
    if top_bucket == bot_bucket:
        return dict(_EMPTY_SPREAD_RESULT)

    top_picks = M.top_n_per_group(sub[sub["_bucket"] == top_bucket], score_col=score_col, group_col="timestamp", n=top_n)
    bot_picks = M.top_n_per_group(sub[sub["_bucket"] == bot_bucket], score_col=score_col, group_col="timestamp", n=top_n)
    if top_picks.empty or bot_picks.empty:
        return dict(_EMPTY_SPREAD_RESULT)
    top_picks = top_picks.assign(week=week_of(top_picks["timestamp"]))
    bot_picks = bot_picks.assign(week=week_of(bot_picks["timestamp"]))

    top_by_week = {w: g[label_col].to_numpy() for w, g in top_picks.groupby("week")}
    bot_by_week = {w: g[label_col].to_numpy() for w, g in bot_picks.groupby("week")}
    top_weeks, bot_weeks = np.array(list(top_by_week)), np.array(list(bot_by_week))
    n_weeks_top, n_weeks_bot = len(top_weeks), len(bot_weeks)

    point = float(top_picks[label_col].mean() - bot_picks[label_col].mean())
    if n_weeks_top < 2 or n_weeks_bot < 2:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_weeks_top": n_weeks_top, "n_weeks_bot": n_weeks_bot,
                "excludes_zero": False, "p_value": np.nan}

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        top_draw = rng.choice(top_weeks, size=n_weeks_top, replace=True)
        bot_draw = rng.choice(bot_weeks, size=n_weeks_bot, replace=True)
        top_mean = np.mean(np.concatenate([top_by_week[w] for w in top_draw]))
        bot_mean = np.mean(np.concatenate([bot_by_week[w] for w in bot_draw]))
        boot[b] = top_mean - bot_mean
    lo, hi = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    p_ge0 = float(np.mean(boot >= 0))
    p_le0 = float(np.mean(boot <= 0))
    p_value = float(min(1.0, 2 * min(p_ge0, p_le0)))
    return {
        "point": point, "ci_lo": lo, "ci_hi": hi,
        "n_weeks_top": n_weeks_top, "n_weeks_bot": n_weeks_bot,
        "excludes_zero": bool(lo > 0 or hi < 0), "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Group 2/3 — cross-sectional rank IC (valid for sector-level + interactions)
# ---------------------------------------------------------------------------

def rank_ic_with_bootstrap(
    period_df: pd.DataFrame,
    *,
    feature_col: str,
    label_col: str = PRIMARY_LABEL,
    group_col: str = "timestamp",
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """Per-bar rank IC, aggregated to a week-block bootstrap CI + p-value on
    the mean IC across the period's weeks."""
    ic_by_bar = M.cross_sectional_rank_ic(period_df, score_col=feature_col, label_col=label_col, group_col=group_col)
    if ic_by_bar.empty or ic_by_bar.dropna().empty:
        return {"mean_ic": np.nan, "icir": np.nan, "n_bars": 0, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_weeks": 0, "excludes_zero": False, "p_value": np.nan}
    mean_ic = M.mean_rank_ic(ic_by_bar)
    icir = M.rank_ic_icir(ic_by_bar)
    bar_ts = pd.to_datetime(pd.Series(ic_by_bar.index), utc=True)
    weeks = week_of(bar_ts)
    boot = week_block_bootstrap_ci(pd.Series(ic_by_bar.values), weeks, n_boot=n_boot, seed=seed)
    # bootstrap p-value: two-sided, symmetric around the resampled distribution
    p_value = np.nan
    if boot["n_weeks"] >= 2:
        rng = np.random.default_rng(seed + 1)
        df_ic = pd.DataFrame({"value": ic_by_bar.values, "week": weeks.to_numpy()}).dropna(subset=["value"])
        by_week = {w: g["value"].to_numpy() for w, g in df_ic.groupby("week")}
        weeks_arr = np.array(list(by_week.keys()))
        boots = np.empty(n_boot)
        for b in range(n_boot):
            draw = rng.choice(weeks_arr, size=len(weeks_arr), replace=True)
            boots[b] = np.mean(np.concatenate([by_week[w] for w in draw]))
        p_ge0, p_le0 = float(np.mean(boots >= 0)), float(np.mean(boots <= 0))
        p_value = float(min(1.0, 2 * min(p_ge0, p_le0)))
    return {
        "mean_ic": mean_ic, "icir": icir, "n_bars": int(ic_by_bar.notna().sum()),
        "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "n_weeks": boot["n_weeks"],
        "excludes_zero": boot["excludes_zero"], "p_value": p_value,
    }


def sector_level_dedup(period_df: pd.DataFrame, *, feature_col: str, label_col: str = PRIMARY_LABEL) -> pd.DataFrame:
    """Collapse to one row per (timestamp, sector_etf): feature value (already
    sector-day constant) + the MEAN forward label across that sector's
    tickers on that bar. Prevents the IC from being inflated purely by how
    many tickers happen to share a sector on a given bar."""
    sub = period_df.dropna(subset=[feature_col, label_col, "sector_etf"])
    if sub.empty:
        return pd.DataFrame(columns=["timestamp", "sector_etf", feature_col, label_col])
    agg = sub.groupby(["timestamp", "sector_etf"]).agg(
        **{feature_col: (feature_col, "first"), label_col: (label_col, "mean")}
    ).reset_index()
    return agg


def apply_bh_fdr(rows: list[dict], *, p_col: str = "p_value", q: float = 0.10) -> pd.DataFrame:
    """Attach BH-FDR q-values across the full family of tests in ``rows`` and
    a boolean 'survives_fdr' flag at threshold ``q``."""
    df = pd.DataFrame(rows)
    if df.empty or p_col not in df.columns:
        return df
    df["q_fdr"] = bh_fdr(df[p_col])
    df["survives_fdr"] = df["q_fdr"] <= q
    return df
