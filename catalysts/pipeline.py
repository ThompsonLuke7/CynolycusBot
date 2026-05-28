"""Unified catalyst records, scores, and timestamp features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from catalysts.config import (
    CATALYST_FEATURE_COLUMNS,
    CATALYST_FEATURE_MATRIX_PATH,
    CATALYST_RECORDS_PATH,
    CATALYST_SCORES_PATH,
    ensure_dirs,
)
from events.config import EARNINGS_EVENTS_PATH, EVENT_FEATURES_PATH, MACRO_EVENTS_PATH
from news.config import NEWS_FEATURE_MATRIX_PATH, NEWS_RECORDS_PATH, NEWS_SCORES_PATH
from catalysts.earnings import build_earnings_result_catalysts


def _read(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix.lower() != ".csv" else pd.read_csv(p)


def _clean_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.replace("$", "", regex=False).str.strip()


def _event_type_from_source(source: pd.Series) -> pd.Series:
    """Preserve the actual SEC form (or 'company_news' for press wire) in event_type."""
    s = source.astype(str)
    is_sec = s.str.startswith("sec")
    # Strip the leading 'sec_' so 'sec_10-q' -> '10-q', 'sec_8-k' -> '8-k', etc.
    sec_form = s.str.replace("^sec_", "", regex=True)
    return s.where(~is_sec, "sec_" + sec_form).where(is_sec, "company_news")


def news_to_catalysts(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame()
    source = news["source"].astype(str) if "source" in news.columns else pd.Series([""] * len(news), index=news.index)
    is_sec = source.str.startswith("sec")
    out = pd.DataFrame(
        {
            "catalyst_id": news["record_id"],
            "record_id": news["record_id"],
            "ticker": _clean_ticker(news["ticker"]),
            "timestamp": pd.to_datetime(news["timestamp"], utc=True, errors="coerce"),
            "catalyst_kind": np.where(is_sec, "sec_filing", "news"),
            "event_type": _event_type_from_source(source),
            "headline": news.get("headline", ""),
            "summary": news.get("summary", ""),
            "source": source,
            "url": news.get("url", ""),
            "relation_type": news.get("relation_type", "ambiguous"),
            "impact_role": news.get("impact_role", "unknown"),
            "relation_confidence": news.get("relation_confidence", np.nan),
            "is_direct_catalyst": news.get("is_direct_catalyst", np.nan),
        }
    )
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def scheduled_events_to_catalysts(events: pd.DataFrame, *, default_kind: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    ticker = events["ticker"] if "ticker" in events.columns else ""
    title = events["title"] if "title" in events.columns else events.get("event_type", "")
    event_type = events["event_type"].astype(str).str.lower()
    out = pd.DataFrame(
        {
            "catalyst_id": [
                f"{default_kind}:{str(t).upper()}:{pd.Timestamp(ts).isoformat()}:{et}"
                for t, ts, et in zip(ticker, events["timestamp"], event_type)
            ],
            "record_id": "",
            "ticker": _clean_ticker(pd.Series(ticker, index=events.index)).replace("NAN", ""),
            "timestamp": pd.to_datetime(events["timestamp"], utc=True, errors="coerce"),
            "catalyst_kind": default_kind,
            "event_type": event_type,
            "headline": title,
            "summary": "",
            "source": events["source"] if "source" in events.columns else default_kind,
            "url": events["url"] if "url" in events.columns else "",
            "relation_type": np.where(pd.Series(ticker, index=events.index).astype(str).str.strip().ne(""), "scheduled_ticker_event", "scheduled_macro_event"),
            "impact_role": np.where(event_type.eq("earnings"), "earnings_event", "event_risk"),
            "relation_confidence": 1.0,
            "is_direct_catalyst": np.where(pd.Series(ticker, index=events.index).astype(str).str.strip().ne(""), 1.0, 0.0),
        }
    )
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def build_catalyst_records(
    *,
    news_path: Path | str = NEWS_RECORDS_PATH,
    macro_path: Path | str = MACRO_EVENTS_PATH,
    earnings_path: Path | str = EARNINGS_EVENTS_PATH,
    output_path: Path | str = CATALYST_RECORDS_PATH,
) -> pd.DataFrame:
    ensure_dirs()
    frames = [
        news_to_catalysts(_read(news_path)),
        scheduled_events_to_catalysts(_read(macro_path), default_kind="scheduled_event"),
        scheduled_events_to_catalysts(_read(earnings_path), default_kind="earnings"),
        build_earnings_result_catalysts(),
    ]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["timestamp", "ticker", "event_type"]).reset_index(drop=True)
    out.to_parquet(output_path, index=False)
    return out


def build_catalyst_scores(
    *,
    catalyst_path: Path | str = CATALYST_RECORDS_PATH,
    news_scores_path: Path | str = NEWS_SCORES_PATH,
    output_path: Path | str = CATALYST_SCORES_PATH,
) -> pd.DataFrame:
    ensure_dirs()
    catalysts = _read(catalyst_path)
    if catalysts.empty:
        out = pd.DataFrame()
        out.to_parquet(output_path, index=False)
        return out
    news_scores = _read(news_scores_path)
    out = catalysts[["catalyst_id", "record_id", "ticker", "timestamp", "catalyst_kind", "event_type", "relation_type", "impact_role"]].copy()
    if not news_scores.empty:
        out = out.merge(
            news_scores[["record_id", "ticker", "timestamp", "news_similarity_score", "news_similarity_neighbor_count", "news_similarity_max", "realized_news_score"]],
            on=["record_id", "ticker", "timestamp"],
            how="left",
        )
    else:
        out["news_similarity_score"] = np.nan
        out["news_similarity_neighbor_count"] = np.nan
        out["news_similarity_max"] = np.nan
        out["realized_news_score"] = np.nan
    out["scheduled_event_score"] = 0.0
    out.loc[out["event_type"].isin(["cpi", "fomc_decision", "fomc_minutes", "nfp", "ppi", "gdp"]), "scheduled_event_score"] = -0.20
    out.loc[out["event_type"].eq("opex"), "scheduled_event_score"] = -0.10
    out.loc[out["event_type"].eq("earnings"), "scheduled_event_score"] = -0.05
    out["catalyst_score"] = out["news_similarity_score"].fillna(out["scheduled_event_score"]).clip(-1.0, 1.0)
    out.to_parquet(output_path, index=False)
    return out


def build_catalyst_features(
    timestamps: pd.DataFrame,
    *,
    news_features_path: Path | str = NEWS_FEATURE_MATRIX_PATH,
    event_features_path: Path | str = EVENT_FEATURES_PATH,
    catalyst_scores_path: Path | str = CATALYST_SCORES_PATH,
    output_path: Path | str = CATALYST_FEATURE_MATRIX_PATH,
) -> pd.DataFrame:
    ensure_dirs()
    base = timestamps.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base["ticker"] = _clean_ticker(base["ticker"])
    news_features = _read(news_features_path)
    event_features = _read(event_features_path)
    scores = _read(catalyst_scores_path)
    out = base.copy()
    for frame in (news_features, event_features):
        if frame.empty:
            continue
        cur = frame.copy()
        cur["timestamp"] = pd.to_datetime(cur["timestamp"], utc=True, errors="coerce")
        cur["ticker"] = _clean_ticker(cur["ticker"])
        out = out.merge(cur, on=["timestamp", "ticker"], how="left")
    out["news_catalyst_score"] = out.get("news_similarity_score", pd.Series(np.nan, index=out.index)).fillna(0.0)
    out["event_risk_score"] = 0.0
    if "macro_event_next_24h" in out.columns:
        out["event_risk_score"] -= out["macro_event_next_24h"].fillna(0.0) * 0.20
    if "earnings_next_7d" in out.columns:
        out["event_risk_score"] -= out["earnings_next_7d"].fillna(0.0) * 0.05
    out["scheduled_event_score"] = out["event_risk_score"].clip(-1.0, 1.0)
    out["catalyst_score"] = (out["news_catalyst_score"] + out["scheduled_event_score"]).clip(-1.0, 1.0)
    out["catalyst_count_24h"] = out.get("news_count_24h", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["direct_catalyst_count_24h"] = out.get("direct_news_count_24h", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["hours_since_catalyst"] = out.get("hours_since_news", pd.Series(np.nan, index=out.index))
    out["latest_catalyst_relation_confidence"] = out.get("news_relation_confidence", pd.Series(np.nan, index=out.index))
    out["latest_catalyst_is_direct"] = out.get("news_is_direct_catalyst", pd.Series(np.nan, index=out.index))
    if not scores.empty:
        # Scores are retained in catalyst_scores.parquet; timestamp features use latest 24h news score.
        pass
    for col in CATALYST_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out.to_parquet(output_path, index=False)
    return out
