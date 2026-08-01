"""Step 9 — Generate Meta-Ranker Theme Features.

Produces daily per-ticker features that replace the old theme_expansion theme
context block in build_meta_ranker_matrix.py.

Features:
  primary_theme            highest-membership theme for this ticker
  primary_theme_rank       rank of primary theme by heat score (1=hottest)
  theme_heat_score         avg 5-day return of cluster members
  theme_breadth            fraction of cluster members with return > 0
  theme_acceleration       heat_score[t-5:t] - heat_score[t-10:t-5]
  theme_strength           avg return of top-quartile cluster members
  membership_score         cosine_sim(ticker_embedding, theme_centroid)
  parent_theme_heat        heat score of parent theme
  related_theme_heat       avg heat score of related themes
  related_theme_rank       avg rank of related themes
  theme_age_days           days since this theme first appeared in registry
  theme_newness_score      1 / (1 + theme_age_days / 30)

Output: ticker_theme_features.parquet
Schema : ticker | date | taxonomy_version | <feature columns>
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    BREADTH_THRESHOLD,
    DAILY_BARS_DIR,
    HEAT_PRIOR_WINDOW_DAYS,
    HEAT_WINDOW_DAYS,
    TICKER_THEME_FEATURES_PATH,
    TOP_Q_STRENGTH,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step05_claude_labeling import _load_registry_or_empty
from themes.dynamic_theme.stages.step07_relationships import load_relationships
from themes.dynamic_theme.stages.step08_memberships import get_primary_theme, load_memberships

logger = logging.getLogger(__name__)

THEME_FEATURE_COLUMNS = [
    "primary_theme",
    "primary_theme_rank",
    "theme_heat_score",
    "theme_breadth",
    "theme_acceleration",
    "theme_strength",
    "membership_score",
    "parent_theme_heat",
    "related_theme_heat",
    "related_theme_rank",
    "theme_age_days",
    "theme_newness_score",
]
_FEATURE_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "taxonomy_version",
    *THEME_FEATURE_COLUMNS,
]


def _taxonomy_from_memberships(memberships: pd.DataFrame) -> str:
    attr_value = memberships.attrs.get("taxonomy_version")
    attr_taxonomy: str | None = None
    if attr_value is not None:
        if not pd.api.types.is_scalar(attr_value) or pd.isna(attr_value):
            raise ValueError("membership taxonomy_version metadata must be one scalar value")
        attr_taxonomy = str(attr_value).strip()
        if not attr_taxonomy:
            raise ValueError("membership taxonomy_version metadata must be non-empty")

    column_taxonomy: str | None = None
    if "taxonomy_version" in memberships.columns and not memberships.empty:
        if memberships["taxonomy_version"].isna().any():
            raise ValueError("membership taxonomy_version column contains missing values")
        values = {
            str(value).strip()
            for value in memberships["taxonomy_version"].tolist()
        }
        if "" in values:
            raise ValueError("membership taxonomy_version column contains empty values")
        if len(values) != 1:
            raise ValueError("membership taxonomy_version column is ambiguous")
        column_taxonomy = values.pop()

    if attr_taxonomy is None and column_taxonomy is None:
        raise ValueError("current memberships require exact taxonomy_version evidence")
    if (
        attr_taxonomy is not None
        and column_taxonomy is not None
        and attr_taxonomy != column_taxonomy
    ):
        raise ValueError("conflicting membership taxonomy_version metadata")
    if attr_taxonomy is not None:
        return attr_taxonomy
    assert column_taxonomy is not None
    return column_taxonomy


def _validated_existing_features(path: Path) -> pd.DataFrame:
    existing = pd.read_parquet(path)
    required = {"ticker", "date", "taxonomy_version"}
    missing = sorted(required - set(existing.columns))
    if missing:
        raise ValueError(f"existing theme features missing columns: {missing}")
    if existing.empty:
        return existing
    if existing["taxonomy_version"].isna().any():
        raise ValueError("existing theme features contain missing taxonomy_version")
    existing = existing.copy()
    existing["taxonomy_version"] = existing["taxonomy_version"].map(
        lambda value: str(value).strip()
    )
    if (existing["taxonomy_version"] == "").any():
        raise ValueError("existing theme features contain empty taxonomy_version")
    existing["date"] = pd.to_datetime(existing["date"], errors="raise").dt.normalize()
    return existing


# ── price helpers ─────────────────────────────────────────────────────────────

def _load_daily_returns(
    tickers: list[str],
    *,
    window: int,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load last `window` trading days of daily close returns for given tickers.

    Returns wide DataFrame: index=date, columns=ticker, values=pct_change.
    Missing tickers silently dropped. `as_of` caps the date range to avoid
    lookahead during historical backfill.
    """
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = DAILY_BARS_DIR / f"{ticker}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        df.columns = [c.lower() for c in df.columns]
        time_col = next((c for c in df.columns if c in ("timestamp", "date", "time")), None)
        close_col = next((c for c in df.columns if c == "close"), None)
        if time_col is None or close_col is None:
            continue
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        series = df.set_index(time_col)[close_col].sort_index()
        if as_of is not None:
            series = series[series.index <= as_of]
        frames[ticker] = series

    if not frames:
        return pd.DataFrame()

    wide = pd.DataFrame(frames)
    needed = window + HEAT_PRIOR_WINDOW_DAYS + 2
    wide = wide.tail(needed)
    return wide.pct_change().iloc[1:]  # returns frame


def _compute_theme_heat(
    returns: pd.DataFrame,
    members: list[str],
    *,
    window: int = HEAT_WINDOW_DAYS,
    prior_window_start: int | None = None,
) -> pd.Series:
    """Mean return of cluster members over last `window` days, indexed by end-date."""
    cols = [t for t in members if t in returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = returns[cols]

    if prior_window_start is not None:
        # prior window: [-(window+prior_window_start): -prior_window_start]
        if len(sub) < window + prior_window_start:
            return pd.Series(dtype=float)
        chunk = sub.iloc[-(window + prior_window_start): -prior_window_start if prior_window_start > 0 else None]
    else:
        chunk = sub.tail(window)

    if chunk.empty:
        return pd.Series(dtype=float)
    return chunk.mean(axis=0).rename(None)  # mean across dates, per ticker


# ── main ──────────────────────────────────────────────────────────────────────

def build_meta_features(
    memberships_df: pd.DataFrame | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build ticker-level theme features and write ticker_theme_features.parquet."""
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if memberships_df is None:
        memberships_df = load_memberships(latest_only=True)

    taxonomy_version = _taxonomy_from_memberships(memberships_df)

    if memberships_df.empty:
        logger.warning("No memberships — cannot build meta features")
        empty = pd.DataFrame(columns=_FEATURE_OUTPUT_COLUMNS)
        empty.attrs["taxonomy_version"] = taxonomy_version
        if not TICKER_THEME_FEATURES_PATH.exists():
            empty.to_parquet(TICKER_THEME_FEATURES_PATH, index=False)
        return empty

    registry = _load_registry_or_empty()
    if not registry.empty and "date" in registry.columns:
        latest_reg_date = registry["date"].max()
        registry_today = registry[registry["date"] == latest_reg_date].copy()
    else:
        registry_today = registry.copy()

    relationships = load_relationships(latest_only=True)

    # primary theme per ticker
    primary = get_primary_theme(memberships_df)  # ticker | primary_theme | membership_score

    all_tickers = primary["ticker"].tolist()
    all_themes = memberships_df["theme"].unique().tolist()

    # members per theme — use only primary members (highest membership theme per
    # ticker) so that heat/breadth reflect that cluster's actual movers, not the
    # full market. This prevents all themes sharing identical breadth values.
    theme_members: dict[str, list[str]] = {}
    for theme, grp in primary.groupby("primary_theme"):
        theme_members[str(theme)] = grp["ticker"].tolist()

    # Load price returns for all tickers in the universe
    needed_window = HEAT_WINDOW_DAYS + HEAT_PRIOR_WINDOW_DAYS + 2
    all_member_tickers = list(set(t for members in theme_members.values() for t in members))
    logger.info("Loading daily returns for %d tickers ...", len(all_member_tickers))
    returns = _load_daily_returns(all_member_tickers, window=needed_window, as_of=as_of)

    # ── per-theme aggregates ───────────────────────────────────────────────────
    theme_heat: dict[str, float] = {}
    theme_breadth: dict[str, float] = {}
    theme_accel: dict[str, float] = {}
    theme_strength: dict[str, float] = {}

    for theme, members in theme_members.items():
        cols = [t for t in members if t in returns.columns]
        if not cols or returns.empty:
            theme_heat[theme] = np.nan
            theme_breadth[theme] = np.nan
            theme_accel[theme] = np.nan
            theme_strength[theme] = np.nan
            continue

        sub = returns[cols].tail(HEAT_WINDOW_DAYS + HEAT_PRIOR_WINDOW_DAYS)
        if sub.empty:
            theme_heat[theme] = np.nan
            theme_breadth[theme] = np.nan
            theme_accel[theme] = np.nan
            theme_strength[theme] = np.nan
            continue

        recent = sub.tail(HEAT_WINDOW_DAYS)
        prior = sub.head(HEAT_PRIOR_WINDOW_DAYS)

        # heat: avg return over heat window (mean across tickers and days)
        heat = float(recent.values.mean()) if not recent.empty else np.nan
        theme_heat[theme] = heat

        # breadth: fraction of tickers with avg return > threshold
        if not recent.empty:
            ticker_avgs = recent.mean(axis=0)
            theme_breadth[theme] = float((ticker_avgs > BREADTH_THRESHOLD).mean())
        else:
            theme_breadth[theme] = np.nan

        # acceleration: recent heat - prior heat
        prior_heat = float(prior.values.mean()) if not prior.empty else np.nan
        theme_accel[theme] = heat - prior_heat if not np.isnan(heat) and not np.isnan(prior_heat) else np.nan

        # strength: top-quartile avg return
        if not recent.empty:
            ticker_avgs = recent.mean(axis=0)
            cutoff = ticker_avgs.quantile(TOP_Q_STRENGTH)
            top_tickers = ticker_avgs[ticker_avgs >= cutoff].index.tolist()
            if top_tickers:
                theme_strength[theme] = float(returns[top_tickers].tail(HEAT_WINDOW_DAYS).values.mean())
            else:
                theme_strength[theme] = np.nan
        else:
            theme_strength[theme] = np.nan

    # Rank themes by heat_score (1 = hottest)
    heat_series = pd.Series(theme_heat).dropna()
    heat_ranks = heat_series.rank(ascending=False, method="min")
    theme_heat_rank: dict[str, float] = heat_ranks.to_dict()

    # ── registry metadata ──────────────────────────────────────────────────────
    # theme_age_days: days since theme first appeared in full registry
    theme_first_seen: dict[str, pd.Timestamp] = {}
    if not registry.empty and "date" in registry.columns and "theme_name" in registry.columns:
        for theme_name, grp in registry.groupby("theme_name"):
            theme_first_seen[str(theme_name)] = pd.to_datetime(grp["date"].min())

    # parent theme mapping
    parent_of: dict[str, str] = {}
    if not registry_today.empty and "theme_name" in registry_today.columns and "parent_theme" in registry_today.columns:
        parent_of = dict(zip(registry_today["theme_name"], registry_today["parent_theme"]))

    # related themes from relationship graph
    related_of: dict[str, list[str]] = {t: [] for t in all_themes}
    if not relationships.empty:
        for _, row in relationships.iterrows():
            src, tgt = str(row["source"]), str(row["target"])
            related_of.setdefault(src, []).append(tgt)
            related_of.setdefault(tgt, []).append(src)

    # ── build per-ticker feature row ──────────────────────────────────────────
    rows = []
    for _, prow in primary.iterrows():
        ticker = str(prow["ticker"])
        theme = str(prow["primary_theme"])
        mem_score = float(prow["membership_score"])

        heat = theme_heat.get(theme, np.nan)
        breadth = theme_breadth.get(theme, np.nan)
        accel = theme_accel.get(theme, np.nan)
        strength = theme_strength.get(theme, np.nan)
        rank = float(theme_heat_rank.get(theme, np.nan))

        # parent theme heat
        parent = parent_of.get(theme, "")
        parent_heat = theme_heat.get(parent, np.nan) if parent else np.nan

        # related themes
        related = [r for r in related_of.get(theme, []) if r in theme_heat]
        related_heat = float(np.nanmean([theme_heat[r] for r in related])) if related else np.nan
        related_rank = float(np.nanmean([theme_heat_rank[r] for r in related if r in theme_heat_rank])) if related else np.nan

        # age — clamp to 0 for backfill dates before registry was created
        first_seen = theme_first_seen.get(theme)
        age_days = float(max(0, (as_of - first_seen).days)) if first_seen is not None else np.nan
        newness_score = float(1.0 / (1.0 + age_days / 30.0)) if not np.isnan(age_days) else np.nan

        rows.append(
            {
                "ticker": ticker,
                "date": as_of,
                "taxonomy_version": taxonomy_version,
                "primary_theme": theme,
                "primary_theme_rank": rank,
                "theme_heat_score": heat,
                "theme_breadth": breadth,
                "theme_acceleration": accel,
                "theme_strength": strength,
                "membership_score": mem_score,
                "parent_theme_heat": parent_heat,
                "related_theme_heat": related_heat,
                "related_theme_rank": related_rank,
                "theme_age_days": age_days,
                "theme_newness_score": newness_score,
            }
        )

    out = pd.DataFrame(rows, columns=_FEATURE_OUTPUT_COLUMNS)

    # Replace only the current date/taxonomy evidence. A revised taxonomy on
    # the same represented date remains separate historical evidence.
    if TICKER_THEME_FEATURES_PATH.exists():
        existing = _validated_existing_features(TICKER_THEME_FEATURES_PATH)
        same_evidence = (
            (existing["date"] == as_of)
            & (existing["taxonomy_version"] == taxonomy_version)
        )
        out = pd.concat([existing.loc[~same_evidence], out], ignore_index=True)

    out = out.sort_values(["date", "taxonomy_version", "ticker"]).reset_index(drop=True)
    out.attrs["taxonomy_version"] = taxonomy_version
    out.to_parquet(TICKER_THEME_FEATURES_PATH, index=False)
    current_rows = out[
        (out["date"] == as_of)
        & (out["taxonomy_version"] == taxonomy_version)
    ]
    logger.info(
        "Wrote %s  rows=%d  date=%s taxonomy=%s",
        TICKER_THEME_FEATURES_PATH,
        len(current_rows),
        as_of.date(),
        taxonomy_version,
    )
    return out
