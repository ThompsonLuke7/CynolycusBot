"""
Build the Meta Ranker training matrix + labels.

Unit:   (ticker, 4H bar).  Horizon: momentum's ~25x4H (~10 trading day) window.
Label:  drawdown-penalized forward return (risk-adjusted).

Stacked-generalization design: the base-model signals used as features are the
models' OUT-OF-FOLD predictions (leakage-free), so the meta model never sees an
in-sample base prediction. Daily theme context is joined AS-OF the PRIOR trading
day so an intraday 4H bar cannot see that day's end-of-day theme aggregates.

Feature groups:
  base scores   - momentum / HTF OOF scores
  theme context - dynamic theme features from dynamic_theme module (prior day)
                  Replaces legacy theme_expansion theme context block.
  cross-context - cross-sectional ranks, signal agreement, within-theme rank,
                  theme crowding
  ticker meta   - sector, cap bucket, liquidity, beta, asset type, is_etf
  regime        - SPY trend / ret, VIX z / high (vol regime), treasury yield curve (macro)
  news catalyst - news_catalyst_signal (embedded-news + announcement aggregates, prior day)
  calendar      - earnings / macro-event proximity + treasury rates/spreads/inversion

Base scores use the walk-forward competition-winner OOF (leak-free, 21d embargo).
options_score was removed (all-NaN). catalyst_score is the news_catalyst_signal join.

Outputs: meta_context/meta_ranker/{meta_ranker_matrix.parquet, manifest.json}
Run:     python meta_context/build_meta_ranker_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "signals/meta_context/meta_ranker"

# Walk-forward competition-winner OOF (leak-free, 21d embargo) from the 2026-06-14
# competition bundles — supersedes the older single-model bundle OOF.
MOM_OOF = REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet"
HTF_OOF = REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet"

# Dynamic theme features — history file covers full training window (2022-2026)
# Current-day file is the fallback for any dates not yet in history
DYNAMIC_THEME_FEATURES_HISTORY = REPO / "themes/dynamic_theme/outputs/ticker_theme_features_history.parquet"
DYNAMIC_THEME_FEATURES = REPO / "themes/dynamic_theme/outputs/ticker_theme_features.parquet"

FEATURES_4H = REPO / "strategies/momentum_expansion/data/processed/features_4h.parquet"

NEWS_CATALYST_SIGNAL = REPO / "signals/meta_context/data/processed/news_catalyst_signal.parquet"

# Forward-guidance signal (structured earnings-guidance features from news guidance text).
from signals.meta_context.build_forward_guidance_signal import FG_FEATURE_COLS
FORWARD_GUIDANCE_SIGNAL = REPO / "signals/meta_context/data/processed/forward_guidance_signal.parquet"
# How long an earnings-guidance reading stays "live" before we treat it as stale (one
# quarter + buffer); beyond this the fg_* features are nulled rather than carried forward.
FG_MAX_CARRY_DAYS = 120

# How long a dynamic-theme reading stays "live". The taxonomy is rebuilt weekly by
# scripts/weekly_refresh.sh stage 4, so a healthy feed is at most ~7 days old and 21
# allows two missed cycles before the features are nulled instead of carried forward.
# Unbounded carry-forward is not hypothetical: stage 4 failed on 2026-08-17 and
# 2026-08-24, ticker_theme_features.parquet froze at 2026-08-10, and the merge_asof
# below kept serving those values as current with nothing marking them stale.
# theme_days_since_refresh is the visible counterpart, mirroring fg_days_since_guidance.
THEME_MAX_CARRY_DAYS = 21

EARNINGS_CALENDAR_CANDIDATES = [
    REPO / "signals/news/data/processed/ticker_earnings_calendar.parquet",
    REPO / "signals/news/data/processed/fmp_earnings_calendar.parquet",
    REPO / "drive-download-20260613T045727Z-3-001/fmp_earnings_calendar.parquet",
]
ECONOMIC_CALENDAR_CANDIDATES = [
    REPO / "signals/news/data/processed/economic_calendar.parquet",
    REPO / "signals/news/data/processed/fmp_economic_calendar.parquet",
    REPO / "drive-download-20260613T045727Z-3-001/fmp_economic_calendar.parquet",
]
TREASURY_RATE_CANDIDATES = [
    REPO / "signals/meta_context/data/processed/fmp_treasury_rates.parquet",
    REPO / "signals/news/data/processed/fmp_treasury_rates.parquet",
    REPO / "drive-download-20260613T045727Z-3-001/fmp_treasury_rates.parquet",
]

MAX_EARNINGS_DISTANCE_DAYS = 120
MAX_MACRO_EVENT_DISTANCE_DAYS = 30

# All numeric news catalyst columns to join into the meta-ranker matrix.
# Categorical columns (top_family, top_subtype) are excluded — use news_catalyst_score_* for signal strength.
NEWS_CATALYST_COLS = [
    "news_catalyst_score",
    "news_catalyst_score_mean",
    "news_catalyst_score_std",
    "news_catalyst_count",
    "news_unique_sources",
    "news_high_score_count",
    "news_high_score_source_diversity",
    "news_high_alpha_count",
    "news_breaking_count",
    "news_bull_alignment",
    "news_bear_alignment",
    "news_p_bull_steady",
    "news_p_bull_volatile",
    "news_p_v_bounce",
    "news_p_crash_stayed",
    "news_p_flat",
]

CALENDAR_MACRO_COLS = [
    "days_to_earnings",
    "days_since_earnings",
    "is_earnings_today",
    "is_pre_earnings_3d",
    "is_post_earnings_3d",
    "days_to_macro_event",
    "days_since_macro_event",
    "days_to_high_impact_macro",
    "days_since_high_impact_macro",
    "macro_event_today_count",
    "macro_high_impact_today_count",
    "macro_event_next_3d_count",
    "macro_high_impact_next_3d_count",
    "treasury_3m",
    "treasury_2y",
    "treasury_10y",
    "treasury_30y",
    "treasury_spread_2s10s",
    "treasury_spread_3m10y",
    "treasury_inverted",
    "treasury_10y_change_5d",
]

DRAWDOWN_PENALTY = 1.0   # legacy meta_label = fwd_close_return - penalty * fwd_max_drawdown

# ---- trade_quality (continuous) + meta_good (binary) labels --------------------
# Continuous risk/alpha-aware setup quality. Built only from forward-outcome columns
# carried in the momentum OOF (fwd_max_alpha, trend_persistence, fwd_max_drawdown,
# fwd_max_return, fwd_close_return). fwd_max_drawdown is a POSITIVE magnitude.
#   trade_quality = W_ALPHA*fwd_max_alpha + W_PERSIST*trend_persistence
#                   - W_MAE*fwd_max_drawdown - W_VOL*max(0, fwd_max_return - fwd_close_return)
# The last term penalises round-trip "give-back" volatility (peaked then faded).
TRADE_QUALITY_WEIGHTS = {"alpha": 1.0, "persist": 0.25, "mae": 1.0, "vol": 0.25}

# Binary good-setup flag: a clean, liquid, alpha-positive move within the label window.
GOOD_RETURN_THRESHOLD = 0.12     # forward max return >= +12%
GOOD_DRAWDOWN_MAG_MAX = 0.08     # max adverse excursion <= 8% (fwd_max_drawdown magnitude)
GOOD_LIQUIDITY_PCTILE = 0.40     # dollar_vol_pctile_252 >= 0.40

# Upside variant: bigger raw winners regardless of cleanliness. Drops the drawdown/alpha
# gates so volatile-but-large moves count (what momentum/HTF capture); keeps liquidity so
# the names are tradeable. Trained as a second target so we can compare a "quality" meta
# vs an "upside" meta. Regression upside target is the raw fwd_max_return.
UPSIDE_RETURN_THRESHOLD = 0.15   # forward max return >= +15% (bigger winners)

META_COLS = ["sector_id", "market_cap_bucket", "asset_type", "is_etf",
             "beta_spy_60", "dollar_vol_pctile_252",
             "regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "regime_vix_high"]

# Dynamic theme context feature columns (from dynamic_theme/stages/step09)
DYNAMIC_THEME_CTX = [
    "primary_theme",
    "primary_theme_rank",
    "theme_heat_score",
    "theme_breadth",
    "theme_acceleration",
    "theme_strength",
    "membership_score",
    # parent_theme_heat removed — step09 never populates it (0% non-null coverage)
    "related_theme_heat",
    "related_theme_rank",
    "theme_age_days",
    "theme_newness_score",
]


def _norm_date(ts: pd.Series) -> pd.Series:
    """tz-aware UTC 4H timestamp -> naive midnight date (matches theme daily)."""
    return pd.to_datetime(ts, utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()


def _asof_prior_day(spine: pd.DataFrame, right: pd.DataFrame, by="theme") -> pd.DataFrame:
    """merge_asof on 'date' within each theme, strictly prior day (no lookahead)."""
    s = spine.sort_values("date").reset_index()  # keep original index in a column
    r = right.sort_values("date")
    merged = pd.merge_asof(s, r, on="date", by=by, direction="backward", allow_exact_matches=False)
    return merged.set_index("index").sort_index()


def _asof_prior_day_ticker(spine: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """merge_asof on 'date' within each ticker, strictly prior day (no lookahead)."""
    s = spine.sort_values("date").reset_index()
    r = right.sort_values("date")
    merged = pd.merge_asof(s, r, on="date", by="ticker", direction="backward", allow_exact_matches=False)
    return merged.set_index("index").sort_index()


def join_theme_context(spine: pd.DataFrame, *, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Join dynamic theme context as-of the prior day, nulling stale carry-forward.

    Shared by the full research builder and the live incremental updater
    (``meta_ranker/update_meta_matrix.py``). They each had their own copy of this
    join, and the copies had already drifted: the research path applied a
    staleness cap to the forward-guidance block while live carried it forward
    forever, which is exactly the research/live feature-parity break AGENTS.md
    forbids. One function, two callers, no drift.

    Returns ``(spine, theme_ctx)`` where ``theme_ctx`` leads with ``theme`` (the
    ticker's primary theme, used by the cross-context groupby) followed by the
    numeric context features.
    """
    path = DYNAMIC_THEME_FEATURES_HISTORY if DYNAMIC_THEME_FEATURES_HISTORY.exists() else DYNAMIC_THEME_FEATURES
    if not path.exists():
        if verbose:
            print("  WARNING: dynamic_theme features not found — theme context will be NaN")
        spine["theme"] = np.nan
        numeric = [c for c in DYNAMIC_THEME_CTX if c != "primary_theme"]
        for col in numeric:
            spine[col] = np.nan
        spine["theme_days_since_refresh"] = np.nan
        return spine, ["theme"] + numeric + ["theme_days_since_refresh"]

    if verbose:
        print(f"  loading from {path.name} ...")
    dtf = pd.read_parquet(path)
    dtf["date"] = pd.to_datetime(dtf["date"]).dt.normalize()
    # If both history and current-day exist, append current-day for any missing dates
    if path == DYNAMIC_THEME_FEATURES_HISTORY and DYNAMIC_THEME_FEATURES.exists():
        today_dtf = pd.read_parquet(DYNAMIC_THEME_FEATURES)
        today_dtf["date"] = pd.to_datetime(today_dtf["date"]).dt.normalize()
        missing_dates = set(today_dtf["date"].unique()) - set(dtf["date"].unique())
        if missing_dates:
            dtf = pd.concat([dtf, today_dtf[today_dtf["date"].isin(missing_dates)]], ignore_index=True)
    # Rename primary_theme → theme so cross-context groupby still works
    dtf = dtf.rename(columns={"primary_theme": "theme"})
    dtf["theme_source_date"] = dtf["date"]
    dtf = dtf.sort_values(["ticker", "date"])
    available_ctx = [c for c in DYNAMIC_THEME_CTX if c in dtf.columns and c != "primary_theme"]
    if verbose:
        print(f"  {len(dtf):,} rows, {dtf['date'].nunique()} dates, {len(available_ctx)} ctx features")

    spine = _asof_prior_day_ticker(
        spine,
        dtf[["ticker", "date", "theme", "theme_source_date"] + available_ctx],
    )
    days_since = (spine["date"] - spine["theme_source_date"]).dt.days
    stale = days_since.isna() | (days_since > THEME_MAX_CARRY_DAYS)
    # `theme` is nulled alongside the numerics on purpose: a stale primary_theme
    # would otherwise keep driving within_theme_mom_rank and theme_crowding_frac
    # off a taxonomy that no longer describes the market.
    spine.loc[stale, ["theme"] + available_ctx] = np.nan
    spine["theme_days_since_refresh"] = days_since.where(~stale, np.nan)
    spine = spine.drop(columns=["theme_source_date"])
    if verbose:
        cov = (~stale).mean()
        print(f"  joined {len(available_ctx)} theme ctx columns "
              f"(fresh within {THEME_MAX_CARRY_DAYS}d: {cov:.1%})")
    return spine, ["theme"] + available_ctx + ["theme_days_since_refresh"]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_earnings_events() -> tuple[pd.DataFrame, str | None]:
    path = _first_existing(EARNINGS_CALENDAR_CANDIDATES)
    if path is None:
        return pd.DataFrame(columns=["ticker", "event_date"]), None
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "event_date"]), str(path)

    if {"symbol", "date"}.issubset(df.columns):
        out = df.rename(columns={"symbol": "ticker", "date": "event_date"})[["ticker", "event_date"]].copy()
    elif {"ticker", "next_earnings_date"}.issubset(df.columns):
        out = df.rename(columns={"next_earnings_date": "event_date"})[["ticker", "event_date"]].copy()
    elif {"ticker", "date"}.issubset(df.columns):
        out = df.rename(columns={"date": "event_date"})[["ticker", "event_date"]].copy()
    else:
        print(f"  WARNING: unrecognized earnings calendar schema at {path}")
        return pd.DataFrame(columns=["ticker", "event_date"]), str(path)

    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["ticker", "event_date"]).drop_duplicates(["ticker", "event_date"])
    return out.sort_values(["ticker", "event_date"]).reset_index(drop=True), str(path)


def _join_earnings_features(spine: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    events, source = _load_earnings_events()
    for col in ["days_to_earnings", "days_since_earnings"]:
        spine[col] = np.nan
    for col in ["is_earnings_today", "is_pre_earnings_3d", "is_post_earnings_3d"]:
        spine[col] = 0.0
    if events.empty:
        return spine, source

    left = spine[["ticker", "date"]].sort_values("date").reset_index()
    next_events = pd.merge_asof(
        left,
        events.rename(columns={"event_date": "next_earnings_date"}).sort_values("next_earnings_date"),
        left_on="date",
        right_on="next_earnings_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
    ).set_index("index").sort_index()
    prev_events = pd.merge_asof(
        left,
        events.rename(columns={"event_date": "prev_earnings_date"}).sort_values("prev_earnings_date"),
        left_on="date",
        right_on="prev_earnings_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("index").sort_index()

    days_to = (next_events["next_earnings_date"] - spine["date"]).dt.days
    days_since = (spine["date"] - prev_events["prev_earnings_date"]).dt.days
    days_to = days_to.where(days_to <= MAX_EARNINGS_DISTANCE_DAYS)
    days_since = days_since.where(days_since <= MAX_EARNINGS_DISTANCE_DAYS)
    spine["days_to_earnings"] = days_to.astype(float)
    spine["days_since_earnings"] = days_since.astype(float)
    spine["is_earnings_today"] = (days_to == 0).astype(float)
    spine["is_pre_earnings_3d"] = ((days_to >= 0) & (days_to <= 3)).astype(float)
    spine["is_post_earnings_3d"] = ((days_since >= 0) & (days_since <= 3)).astype(float)
    return spine, source


def _load_economic_events() -> tuple[pd.DataFrame, str | None]:
    path = _first_existing(ECONOMIC_CALENDAR_CANDIDATES)
    if path is None:
        return pd.DataFrame(columns=["event_date", "is_high_impact"]), None
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["event_date", "is_high_impact"]), str(path)

    date_col = "event_date" if "event_date" in df.columns else "date"
    if date_col not in df.columns:
        print(f"  WARNING: unrecognized economic calendar schema at {path}")
        return pd.DataFrame(columns=["event_date", "is_high_impact"]), str(path)
    out = pd.DataFrame({"event_date": pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()})
    impact = df.get("impact", df.get("importance", ""))
    out["impact"] = pd.Series(impact, index=df.index).astype(str).str.lower().values
    country = df.get("country")
    if country is not None:
        out["country"] = country.astype(str).str.upper().values
        out = out[out["country"].isin({"US", "USA", "UNITED STATES"})].copy()
    out["is_high_impact"] = out["impact"].str.contains("high|3", regex=True).astype(float)
    out = out.dropna(subset=["event_date"])
    return out[["event_date", "is_high_impact"]].reset_index(drop=True), str(path)


def _join_economic_features(spine: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    events, source = _load_economic_events()
    if events.empty:
        for col in [
            "days_to_macro_event",
            "days_since_macro_event",
            "days_to_high_impact_macro",
            "days_since_high_impact_macro",
        ]:
            spine[col] = np.nan
        for col in [
            "macro_event_today_count",
            "macro_high_impact_today_count",
            "macro_event_next_3d_count",
            "macro_high_impact_next_3d_count",
        ]:
            spine[col] = 0.0
        return spine, source

    daily = events.groupby("event_date").agg(
        macro_event_today_count=("event_date", "size"),
        macro_high_impact_today_count=("is_high_impact", "sum"),
    ).sort_index()
    coverage_min = daily.index.min()
    coverage_max = daily.index.max()
    all_events = daily.reset_index()[["event_date"]].drop_duplicates()
    high_events = daily[daily["macro_high_impact_today_count"] > 0].reset_index()[["event_date"]]

    unique_dates = pd.DataFrame({"date": pd.Series(spine["date"].drop_duplicates()).sort_values()})
    unique_dates = unique_dates.reset_index(drop=True)
    for label, event_dates in [("macro_event", all_events), ("high_impact_macro", high_events)]:
        if event_dates.empty:
            unique_dates[f"days_to_{label}"] = np.nan
            unique_dates[f"days_since_{label}"] = np.nan
            continue
        next_ev = pd.merge_asof(
            unique_dates,
            event_dates.rename(columns={"event_date": "next_event_date"}).sort_values("next_event_date"),
            left_on="date",
            right_on="next_event_date",
            direction="forward",
            allow_exact_matches=True,
        )
        prev_ev = pd.merge_asof(
            unique_dates,
            event_dates.rename(columns={"event_date": "prev_event_date"}).sort_values("prev_event_date"),
            left_on="date",
            right_on="prev_event_date",
            direction="backward",
            allow_exact_matches=True,
        )
        unique_dates[f"days_to_{label}"] = (next_ev["next_event_date"] - unique_dates["date"]).dt.days.astype(float)
        unique_dates[f"days_since_{label}"] = (unique_dates["date"] - prev_ev["prev_event_date"]).dt.days.astype(float)
        unique_dates[f"days_to_{label}"] = unique_dates[f"days_to_{label}"].where(
            unique_dates[f"days_to_{label}"] <= MAX_MACRO_EVENT_DISTANCE_DAYS
        )
        unique_dates[f"days_since_{label}"] = unique_dates[f"days_since_{label}"].where(
            unique_dates[f"days_since_{label}"] <= MAX_MACRO_EVENT_DISTANCE_DAYS
        )

    counts = daily.rename_axis("date")[["macro_event_today_count", "macro_high_impact_today_count"]]
    daily_features = []
    for dt in unique_dates["date"]:
        window = counts[(counts.index >= dt) & (counts.index <= dt + pd.Timedelta(days=3))]
        today = counts.reindex([dt]).fillna(0.0)
        daily_features.append(
            {
                "date": dt,
                "macro_event_today_count": float(today["macro_event_today_count"].iloc[0]),
                "macro_high_impact_today_count": float(today["macro_high_impact_today_count"].iloc[0]),
                "macro_event_next_3d_count": float(window["macro_event_today_count"].sum()),
                "macro_high_impact_next_3d_count": float(window["macro_high_impact_today_count"].sum()),
            }
        )

    unique_dates = unique_dates.merge(pd.DataFrame(daily_features), on="date", how="left")
    outside_coverage = (unique_dates["date"] < coverage_min) | (unique_dates["date"] > coverage_max)
    count_cols = [
        "macro_event_today_count",
        "macro_high_impact_today_count",
        "macro_event_next_3d_count",
        "macro_high_impact_next_3d_count",
    ]
    unique_dates.loc[outside_coverage, count_cols] = np.nan
    return spine.merge(unique_dates, on="date", how="left"), source


def _join_treasury_features(spine: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    path = _first_existing(TREASURY_RATE_CANDIDATES)
    if path is None:
        for col in [c for c in CALENDAR_MACRO_COLS if c.startswith("treasury_")]:
            spine[col] = np.nan
        return spine, None
    tr = pd.read_parquet(path)
    if tr.empty or "date" not in tr.columns:
        for col in [c for c in CALENDAR_MACRO_COLS if c.startswith("treasury_")]:
            spine[col] = np.nan
        return spine, str(path)
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    tr = tr.sort_values("date")
    col_map = {
        "month3": "treasury_3m",
        "year2": "treasury_2y",
        "year10": "treasury_10y",
        "year30": "treasury_30y",
        "spread_2s10s": "treasury_spread_2s10s",
        "spread_3m10y": "treasury_spread_3m10y",
        "inverted": "treasury_inverted",
    }
    keep = ["date"] + [c for c in col_map if c in tr.columns]
    tr = tr[keep].rename(columns=col_map)
    if "treasury_10y" in tr.columns:
        tr["treasury_10y_change_5d"] = tr["treasury_10y"].diff(5)
    else:
        tr["treasury_10y_change_5d"] = np.nan
    # Treasury rates are treated as prior-day macro state to avoid intraday lookahead.
    merged = pd.merge_asof(
        spine.sort_values("date").reset_index(),
        tr.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("index").sort_index()
    for col in [c for c in CALENDAR_MACRO_COLS if c.startswith("treasury_")]:
        if col in merged.columns:
            spine[col] = merged[col]
        else:
            spine[col] = np.nan
    return spine, str(path)


def _join_calendar_macro_features(spine: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    spine, earnings_source = _join_earnings_features(spine)
    spine, economic_source = _join_economic_features(spine)
    spine, treasury_source = _join_treasury_features(spine)
    return spine, {
        "earnings_calendar": earnings_source,
        "economic_calendar": economic_source,
        "treasury_rates": treasury_source,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- spine: momentum OOF (score + forward outcome columns for the label)
    print("loading momentum OOF (spine) ...")
    mom = pd.read_parquet(MOM_OOF).reset_index()
    mom = mom.rename(columns={mom.columns[0]: "timestamp"})
    mom["timestamp"] = pd.to_datetime(mom["timestamp"], utc=True)
    # base OOF files carry a few duplicate (timestamp,ticker) rows at walk-forward
    # fold boundaries; keep the latest fold's prediction so the spine is unique.
    mom = mom.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
    _mom_label_cols = ["fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return", "trend_persistence"]
    # fwd_max_return / fwd_max_alpha drive the continuous trade_quality + binary meta_good
    # labels; they are present in the momentum OOF but were not previously selected.
    for _opt in ("fwd_max_return", "fwd_max_alpha"):
        if _opt in mom.columns:
            _mom_label_cols.append(_opt)
    spine = mom[["timestamp", "ticker", "score", *_mom_label_cols]].copy()
    spine = spine.rename(columns={"score": "mom_score"})
    spine["date"] = _norm_date(spine["timestamp"])
    print(f"  spine rows: {len(spine):,}  bars: {spine['timestamp'].nunique():,}  "
          f"range {spine['timestamp'].min()} -> {spine['timestamp'].max()}")

    # ---- HTF OOF score
    print("joining HTF OOF score ...")
    htf = pd.read_parquet(HTF_OOF, columns=["score"]).reset_index()
    htf = htf.rename(columns={htf.columns[0]: "timestamp", "score": "htf_score"})
    htf["timestamp"] = pd.to_datetime(htf["timestamp"], utc=True)
    htf = htf.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
    spine = spine.merge(htf[["timestamp", "ticker", "htf_score"]], on=["timestamp", "ticker"], how="left")

    # ---- dynamic theme context (prior day, per ticker) ----------------------
    # Replaces the legacy theme_expansion theme_scores join and theme_map join.
    # primary_theme is the ticker's highest-membership theme from the dynamic
    # taxonomy; all other theme features come from step09_meta_features.
    print("joining dynamic theme context (prior day) ...")
    spine, theme_ctx = join_theme_context(spine)

    # ---- ticker meta + regime (point-in-time, exact 4H join)
    print("joining ticker meta + regime from features_4h ...")
    cut = spine["timestamp"].min().to_pydatetime()
    f4 = pd.read_parquet(FEATURES_4H, columns=META_COLS,
                         filters=[("timestamp", ">=", cut)])
    f4 = f4.reset_index()
    f4 = f4.rename(columns={f4.columns[0]: "timestamp"}) if "timestamp" not in f4.columns else f4
    f4["timestamp"] = pd.to_datetime(f4["timestamp"], utc=True)
    spine = spine.merge(f4, on=["timestamp", "ticker"], how="left")

    # ---- cross-context (per 4H bar)
    print("deriving cross-context features ...")
    g = spine.groupby("timestamp")
    spine["mom_xs_rank"] = g["mom_score"].rank(pct=True)
    spine["htf_xs_rank"] = g["htf_score"].rank(pct=True)
    spine["signal_agreement"] = spine["mom_xs_rank"] * spine["htf_xs_rank"]
    # within-theme momentum rank + crowding (top-quintile share) per bar+theme
    spine["theme"] = spine["theme"].fillna("__unknown__")
    gt = spine.groupby(["timestamp", "theme"])
    spine["within_theme_mom_rank"] = gt["mom_score"].rank(pct=True)
    spine["_hot"] = (spine["mom_xs_rank"] > 0.8).astype(float)
    spine["theme_crowding_frac"] = gt["_hot"].transform("mean")
    spine = spine.drop(columns=["_hot"])
    spine["theme"] = spine["theme"].replace("__unknown__", np.nan)

    # ---- news catalyst signal (prior day, per ticker) -----------------------
    # Replaces the old catalyst_score=np.nan stub. Joined as-of the prior
    # trading day so an intraday 4H bar cannot see that day's news aggregates.
    print("joining news catalyst signal (prior day) ...")
    news_cols_present: list[str] = []
    if NEWS_CATALYST_SIGNAL.exists():
        ncs = pd.read_parquet(NEWS_CATALYST_SIGNAL)
        ncs["date"] = pd.to_datetime(ncs["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        ncs = ncs.sort_values(["ticker", "date"])
        available_news = [c for c in NEWS_CATALYST_COLS if c in ncs.columns]
        spine = _asof_prior_day_ticker(spine, ncs[["ticker", "date"] + available_news])
        news_cols_present = available_news
        print(f"  joined {len(available_news)} news catalyst columns")
    else:
        print("  WARNING: news_catalyst_signal.parquet not found — news features will be NaN")
        for col in NEWS_CATALYST_COLS:
            spine[col] = np.nan
        news_cols_present = NEWS_CATALYST_COLS

    # ---- earnings/economic-calendar/treasury macro context -----------------
    print("joining earnings/economic/treasury calendar context ...")
    spine, calendar_sources = _join_calendar_macro_features(spine)
    print(
        "  sources: "
        f"earnings={calendar_sources['earnings_calendar'] or 'missing'}, "
        f"economic={calendar_sources['economic_calendar'] or 'missing'}, "
        f"treasury={calendar_sources['treasury_rates'] or 'missing'}"
    )

    # ---- forward-guidance signal (prior day, per ticker) --------------------
    # Structured earnings-guidance features (raise/cut, margin, demand, confidence/
    # uncertainty language, overall strength) extracted from the news guidance text.
    # Joined as-of the prior trading day; carried forward up to FG_MAX_CARRY_DAYS
    # (one quarter+buffer) then nulled, with fg_days_since_guidance as a recency feature.
    print("joining forward-guidance signal (prior day) ...")
    fg_features_present: list[str] = []
    if FORWARD_GUIDANCE_SIGNAL.exists():
        fg = pd.read_parquet(FORWARD_GUIDANCE_SIGNAL)
        fg["date"] = pd.to_datetime(fg["date"]).dt.normalize()
        fg["fg_event_date"] = fg["date"]
        fg = fg.sort_values(["ticker", "date"])
        avail_fg = [c for c in FG_FEATURE_COLS if c in fg.columns]
        spine = _asof_prior_day_ticker(spine, fg[["ticker", "date", "fg_event_date"] + avail_fg])
        days_since = (spine["date"] - spine["fg_event_date"]).dt.days
        stale = days_since.isna() | (days_since > FG_MAX_CARRY_DAYS)
        spine.loc[stale, avail_fg] = np.nan
        spine["fg_days_since_guidance"] = days_since.where(~stale, np.nan)
        spine = spine.drop(columns=["fg_event_date"])
        fg_features_present = avail_fg + ["fg_days_since_guidance"]
        cov = spine[avail_fg[0]].notna().mean() if avail_fg else 0.0
        print(f"  joined {len(fg_features_present)} forward-guidance columns (coverage {cov:.1%})")
    else:
        print("  WARNING: forward_guidance_signal.parquet not found — run "
              "build_forward_guidance_signal.py; fg_* features skipped")

    # options_score stub removed — it was an all-NaN placeholder (0% coverage)
    # carrying no signal. Re-add a real options feature here when available.

    # ---- labels --------------------------------------------------------------
    # Legacy drawdown-penalized forward return (kept for continuity / comparison).
    spine["meta_label"] = spine["fwd_close_return"] - DRAWDOWN_PENALTY * spine["fwd_max_drawdown"]

    # Continuous trade_quality + binary meta_good (need fwd_max_return/alpha from MOM_OOF).
    have_quality = {"fwd_max_alpha", "fwd_max_return"}.issubset(spine.columns)
    if have_quality:
        w = TRADE_QUALITY_WEIGHTS
        giveback = (spine["fwd_max_return"] - spine["fwd_close_return"]).clip(lower=0.0)
        spine["trade_quality"] = (
            w["alpha"] * spine["fwd_max_alpha"]
            + w["persist"] * spine["trend_persistence"].fillna(0.0)
            - w["mae"] * spine["fwd_max_drawdown"]
            - w["vol"] * giveback
        )
        liq = spine["dollar_vol_pctile_252"] if "dollar_vol_pctile_252" in spine.columns else 1.0
        spine["meta_good"] = (
            (spine["fwd_max_return"] >= GOOD_RETURN_THRESHOLD)
            & (spine["fwd_max_drawdown"] <= GOOD_DRAWDOWN_MAG_MAX)
            & (spine["fwd_max_alpha"] > 0.0)
            & (liq >= GOOD_LIQUIDITY_PCTILE)
        ).astype(int)
        # Upside variant: bigger raw winners, drop drawdown/alpha gates, keep liquidity.
        spine["meta_upside"] = (
            (spine["fwd_max_return"] >= UPSIDE_RETURN_THRESHOLD)
            & (liq >= GOOD_LIQUIDITY_PCTILE)
        ).astype(int)
    else:
        print("  WARNING: fwd_max_return/fwd_max_alpha absent from MOM_OOF — "
              "trade_quality/meta_good not built (rebuild momentum OOF to enable).")

    # ---- assemble + manifest
    id_cols = ["timestamp", "ticker", "theme", "date"]
    label_cols = ["meta_label", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return", "trend_persistence"]
    if have_quality:
        label_cols += ["trade_quality", "meta_good", "meta_upside", "fwd_max_return", "fwd_max_alpha"]
    base_scores = ["mom_score", "htf_score"]
    numeric_theme_ctx = [c for c in theme_ctx if c != "theme"]
    cross = ["mom_xs_rank", "htf_xs_rank", "signal_agreement", "within_theme_mom_rank", "theme_crowding_frac"]
    news_features = news_cols_present
    calendar_macro_features = CALENDAR_MACRO_COLS
    forward_guidance_features = fg_features_present
    stubs: list[str] = []  # options_score stub removed (was all-NaN)
    feature_cols = (base_scores + numeric_theme_ctx + cross + META_COLS + news_features
                    + calendar_macro_features + forward_guidance_features + stubs)

    out = spine[id_cols + feature_cols + label_cols].copy()
    out = out.dropna(subset=["meta_label"]).set_index(["timestamp", "ticker"]).sort_index()
    out.to_parquet(OUT / "meta_ranker_matrix.parquet")

    cov = {c: float(out[c].notna().mean()) for c in feature_cols}
    # Competition harness labels: regression on continuous trade_quality, classifier /
    # ranker relevance on the binary meta_good flag. Falls back to the legacy meta_label
    # if the quality columns could not be built.
    primary_label = "trade_quality" if have_quality else "meta_label"
    manifest = {
        "unit": "(ticker, 4H bar)",
        "horizon": "momentum 25x4H (~10 trading days)",
        "label_column": primary_label,
        "label_definition": (
            "trade_quality = %.2f*fwd_max_alpha + %.2f*trend_persistence "
            "- %.2f*fwd_max_drawdown - %.2f*max(0, fwd_max_return - fwd_close_return)"
            % (TRADE_QUALITY_WEIGHTS["alpha"], TRADE_QUALITY_WEIGHTS["persist"],
               TRADE_QUALITY_WEIGHTS["mae"], TRADE_QUALITY_WEIGHTS["vol"])
            if have_quality
            else "fwd_close_return - %.1f * fwd_max_drawdown" % DRAWDOWN_PENALTY
        ),
        # colab_competition.py keys
        "target_column": "meta_good" if have_quality else "meta_label",
        "regression_target_column": primary_label,
        "relevance_column": "meta_good" if have_quality else None,
        "meta_good_definition": (
            "fwd_max_return >= %.2f AND fwd_max_drawdown <= %.2f AND fwd_max_alpha > 0 "
            "AND dollar_vol_pctile_252 >= %.2f"
            % (GOOD_RETURN_THRESHOLD, GOOD_DRAWDOWN_MAG_MAX, GOOD_LIQUIDITY_PCTILE)
        ) if have_quality else None,
        "meta_good_positive_rate": float(out["meta_good"].mean()) if have_quality and "meta_good" in out.columns else None,
        "train_frac": 0.6,
        "val_frac": 0.2,
        "rank_group": "timestamp",
        "top_k": 20,
        "walk_forward": {"train_months": 18, "embargo_days": 21, "test_months": 4, "min_train_rows": 50000},
        "id_columns": id_cols,
        "feature_columns": feature_cols,
        "base_score_columns": base_scores,
        "theme_context_columns": numeric_theme_ctx,
        "theme_context_source": "themes/dynamic_theme/outputs/ticker_theme_features.parquet",
        "cross_context_columns": cross,
        "ticker_meta_columns": META_COLS,
        "news_catalyst_columns": news_features,
        "news_catalyst_source": "signals/meta_context/data/processed/news_catalyst_signal.parquet",
        "calendar_macro_columns": calendar_macro_features,
        "calendar_macro_sources": calendar_sources,
        "forward_guidance_columns": forward_guidance_features,
        "forward_guidance_source": "signals/meta_context/data/processed/forward_guidance_signal.parquet",
        "stub_columns": stubs,
        "label_columns": label_cols,
        "leakage_controls": [
            "base scores are out-of-fold (stacked generalization)",
            "dynamic theme context joined as-of PRIOR trading day (per ticker)",
            "news catalyst signal joined as-of PRIOR trading day (per ticker)",
            "earnings calendar features are event-date distances known from the calendar",
            "economic calendar features are event-date distances/counts known from the calendar",
            "treasury rate features are joined as-of PRIOR trading day",
            "forward-guidance signal joined as-of PRIOR trading day (per ticker), nulled beyond carry window",
            "ticker meta/regime are point-in-time 4H features",
        ],
        "n_rows": int(len(out)),
        "n_bars": int(out.index.get_level_values("timestamp").nunique()),
        "n_tickers": int(out.index.get_level_values("ticker").nunique()),
        "date_min": str(out.index.get_level_values("timestamp").min()),
        "date_max": str(out.index.get_level_values("timestamp").max()),
        "feature_coverage_non_null": cov,
        "categorical_columns": ["sector_id", "market_cap_bucket", "asset_type"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # Second target variant: "upside" meta (bigger raw winners). Same features/matrix, just a
    # different label so Colab can train both in one session (set META_MANIFEST=manifest_upside.json).
    if have_quality and "meta_upside" in out.columns:
        upside = dict(manifest)
        upside["label_column"] = "meta_upside"
        upside["target_column"] = "meta_upside"
        upside["relevance_column"] = "meta_upside"
        upside["regression_target_column"] = "fwd_max_return"
        upside["label_definition"] = (
            "meta_upside = fwd_max_return >= %.2f AND dollar_vol_pctile_252 >= %.2f "
            "(bigger raw winners; drops drawdown/alpha cleanliness gates)"
            % (UPSIDE_RETURN_THRESHOLD, GOOD_LIQUIDITY_PCTILE))
        upside["meta_upside_positive_rate"] = float(out["meta_upside"].mean())
        upside["variant"] = "upside"
        (OUT / "manifest_upside.json").write_text(json.dumps(upside, indent=2, default=str))
        print(f"wrote manifest_upside.json  meta_upside positive rate={out['meta_upside'].mean():.4f}")

    print(f"\nwrote {OUT/'meta_ranker_matrix.parquet'}  rows={len(out):,}  features={len(feature_cols)}")
    print("coverage (non-null):")
    for c in feature_cols:
        prefix = "  [news] " if c in news_features else ("  [cal] " if c in calendar_macro_features else "  ")
        print(f"{prefix}{c:36s} {cov[c]:6.1%}")
    print(f"\nlabel meta_label: mean {out['meta_label'].mean():+.4f}  std {out['meta_label'].std():.4f}")


if __name__ == "__main__":
    main()
