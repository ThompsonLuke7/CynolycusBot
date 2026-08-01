"""Unified catalyst records, scores, and timestamp features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from signals.catalysts.config import (
    CATALYST_FEATURE_COLUMNS,
    CATALYST_FEATURE_MATRIX_PATH,
    CATALYST_RECORDS_PATH,
    CATALYST_SCORES_PATH,
    ensure_dirs,
)
from signals.events.config import EARNINGS_EVENTS_PATH, EVENT_FEATURES_PATH, MACRO_EVENTS_PATH
from signals.news.config import NEWS_FEATURE_MATRIX_PATH, NEWS_RECORDS_PATH, NEWS_SCORES_PATH
from signals.news.schema import parquet_safe_causal_metadata
from signals.events.forward_guidance.config import FEATURES_PATH as FORWARD_GUIDANCE_FEATURES_PATH
from signals.events.forward_guidance.config import LABELS_PATH as FORWARD_GUIDANCE_LABELS_PATH

from core.nervous_system.contracts.quality import LineageRef
from signals.catalysts.nervous_system_adapter import (
    hindsight_evidence_fields,
    publish_catalyst_states,
)

if TYPE_CHECKING:
    from core.nervous_system.persistence.uow import UnitOfWork


TIMESTAMP_SEMANTICS_VERSION = "catalyst-time@1"


def _series(news: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in news.columns:
        return news[column]
    return pd.Series([default] * len(news), index=news.index)


def _missing_scalar(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, bool) and result


def _strict_causal_timestamp(value: object) -> object:
    """Keep invalid/naive explicit metadata visible to the adapter.

    ``pd.to_datetime(..., utc=True)`` silently localizes naive values.  Causal
    observation and availability fields cannot make that assumption, so an
    invalid scalar is returned unchanged for the adapter to quarantine.
    """

    if _missing_scalar(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return value
    return timestamp.tz_convert("UTC")


def _strict_causal_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(
        [_strict_causal_timestamp(value) for value in _series(frame, column, pd.NaT)],
        index=frame.index,
        dtype=object,
    )


def _fill_missing_values(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    return pd.Series(
        [
            fallback_value if _missing_scalar(value) else value
            for value, fallback_value in zip(primary, fallback)
        ],
        index=primary.index,
        dtype=object,
    )


def _hindsight_markers(frame: pd.DataFrame) -> list[dict[str, object] | None]:
    return [
        (evidence := hindsight_evidence_fields(row.to_dict())) or None
        for _, row in frame.iterrows()
    ]


def _read(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix.lower() != ".csv" else pd.read_csv(p)


def _clean_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.replace("$", "", regex=False).str.strip()


def _clean_optional_ticker(series: pd.Series) -> pd.Series:
    return pd.Series(
        [
            None
            if _missing_scalar(value)
            else (str(value).upper().replace("$", "").strip() or None)
            for value in series
        ],
        index=series.index,
        dtype=object,
    )


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
    record_ids = _series(news, "record_id")
    source_record_ids = _series(news, "source_record_id", "")
    source_record_ids = source_record_ids.astype(object)
    source_record_ids = source_record_ids.where(
        source_record_ids.astype(str).str.strip().ne(""), ""
    )
    timestamp = pd.to_datetime(_series(news, "timestamp", pd.NaT), utc=True, errors="coerce")
    raw_timestamp = _strict_causal_series(news, "timestamp")
    event_time = _strict_causal_series(news, "event_time")
    event_time = _fill_missing_values(event_time, raw_timestamp)
    out = pd.DataFrame(
        {
            "catalyst_id": record_ids,
            "record_id": record_ids,
            "ticker": _clean_ticker(_series(news, "ticker")),
            "timestamp": timestamp,
            "event_time": event_time,
            "published_at": _strict_causal_series(news, "published_at"),
            "observed_at": _strict_causal_series(news, "observed_at"),
            "available_at": _strict_causal_series(news, "available_at"),
            "source_record_id": source_record_ids,
            "source_artifact_hash": _series(news, "source_artifact_hash", None),
            "timestamp_semantics_version": _series(
                news, "timestamp_semantics_version", TIMESTAMP_SEMANTICS_VERSION
            ),
            "catalyst_kind": np.where(is_sec, "sec_filing", "news"),
            "event_type": _event_type_from_source(source),
            "headline": _series(news, "headline"),
            "summary": _series(news, "summary"),
            "source": source,
            "url": _series(news, "url"),
            "relation_type": _series(news, "relation_type", "ambiguous"),
            "impact_role": _series(news, "impact_role", "unknown"),
            "relation_confidence": _series(news, "relation_confidence", np.nan),
            "is_direct_catalyst": _series(news, "is_direct_catalyst", np.nan),
            "directional_score": _series(news, "directional_score", np.nan),
            "raw_score": _series(news, "raw_score", np.nan),
            "catalyst_score": _series(news, "catalyst_score", np.nan),
            "score": _series(news, "score", np.nan),
            "hindsight_evidence": _hindsight_markers(news),
            "ingestion_quarantine_code": _series(news, "ingestion_quarantine_code", None),
            "ingestion_quarantine_message": _series(news, "ingestion_quarantine_message", None),
            "raw_ingestion_fields": _series(news, "raw_ingestion_fields", None),
        }
    )
    # Keep malformed source rows so the adapter can quarantine them with
    # their original evidence instead of silently losing the source record.
    return out.reset_index(drop=True)


def scheduled_events_to_catalysts(events: pd.DataFrame, *, default_kind: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    ticker = _clean_optional_ticker(_series(events, "ticker", None))
    title = _series(events, "title") if "title" in events.columns else _series(events, "event_type", "")
    event_type = _series(events, "event_type", "").astype(str).str.lower()
    timestamp = pd.to_datetime(_series(events, "timestamp", pd.NaT), utc=True, errors="coerce")
    raw_timestamp = _strict_causal_series(events, "timestamp")
    event_time = _fill_missing_values(_strict_causal_series(events, "event_time"), raw_timestamp)
    source = _series(events, "source", default_kind)
    has_ticker = ticker.notna()
    event_identity_values = []
    for t, ts, et in zip(ticker, event_time, event_type):
        try:
            time_key = pd.Timestamp(ts).isoformat()
        except (TypeError, ValueError):
            time_key = str(ts)
        event_identity_values.append(f"{default_kind}:{t or ''}:{time_key}:{et}")
    generated_source_ids = pd.Series(
        event_identity_values,
        index=events.index,
    )
    source_record_ids = _series(events, "source_record_id") if "source_record_id" in events.columns else generated_source_ids
    source_record_ids = source_record_ids.astype(object).where(
        source_record_ids.astype(str).str.strip().ne(""), generated_source_ids
    )
    out = pd.DataFrame(
        {
            "catalyst_id": event_identity_values,
            "record_id": "",
            "ticker": ticker,
            "timestamp": timestamp,
            "event_time": event_time,
            "published_at": _strict_causal_series(events, "published_at"),
            "observed_at": _strict_causal_series(events, "observed_at"),
            "available_at": _strict_causal_series(events, "available_at"),
            "source_record_id": source_record_ids,
            "source_artifact_hash": _series(events, "source_artifact_hash", None),
            "timestamp_semantics_version": _series(
                events, "timestamp_semantics_version", TIMESTAMP_SEMANTICS_VERSION
            ),
            "catalyst_kind": default_kind,
            "event_type": event_type,
            "headline": title,
            "summary": "",
            "source": source,
            "url": _series(events, "url", ""),
            "relation_type": np.where(has_ticker, "scheduled_ticker_event", "scheduled_macro_event"),
            "impact_role": np.where(event_type.eq("earnings"), "earnings_event", "event_risk"),
            "relation_confidence": 1.0,
            "is_direct_catalyst": np.where(has_ticker, 1.0, 0.0),
            "directional_score": _series(events, "directional_score", np.nan),
            "raw_score": _series(events, "raw_score", np.nan),
            "catalyst_score": _series(events, "catalyst_score", np.nan),
            "score": _series(events, "score", np.nan),
            "hindsight_evidence": _hindsight_markers(events),
            "ingestion_quarantine_code": _series(events, "ingestion_quarantine_code", None),
            "ingestion_quarantine_message": _series(events, "ingestion_quarantine_message", None),
            "raw_ingestion_fields": _series(events, "raw_ingestion_fields", None),
        }
    )
    # Keep malformed scheduled rows so the adapter can quarantine them with
    # their original evidence instead of silently losing the source record.
    return out.reset_index(drop=True)


def build_catalyst_records(
    *,
    news_path: Path | str = NEWS_RECORDS_PATH,
    macro_path: Path | str = MACRO_EVENTS_PATH,
    earnings_path: Path | str = EARNINGS_EVENTS_PATH,
    earnings_result_features_path: Path | str | None = FORWARD_GUIDANCE_FEATURES_PATH,
    earnings_result_labels_path: Path | str | None = FORWARD_GUIDANCE_LABELS_PATH,
    output_path: Path | str = CATALYST_RECORDS_PATH,
    unit_of_work: "UnitOfWork | None" = None,
    source_artifact: LineageRef | None = None,
    decision_time: datetime | None = None,
    valid_until: datetime | None = None,
) -> pd.DataFrame:
    ensure_dirs()
    frames = [
        news_to_catalysts(_read(news_path)),
        scheduled_events_to_catalysts(_read(macro_path), default_kind="scheduled_event"),
        scheduled_events_to_catalysts(_read(earnings_path), default_kind="earnings"),
    ]
    # Forward-guidance features/labels contain post-result observations and
    # realized forward returns.  They are retained as research evidence in
    # their own artifacts, but are never eligible catalyst-event states.
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["timestamp", "ticker", "event_type"]).reset_index(drop=True)
    parquet_safe_causal_metadata(out).to_parquet(output_path, index=False)
    if unit_of_work is not None:
        if source_artifact is None:
            raise ValueError(
                "exact source_artifact LineageRef is required for catalyst publication"
            )
        publish_catalyst_states(
            out.to_dict(orient="records"),
            unit_of_work=unit_of_work,
            source_artifact=source_artifact,
            decision_time=decision_time,
            valid_until=valid_until,
        )
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
    # Safe Parquet serialization stores mixed legacy timestamps as strings;
    # normalize only this ordinary join key when reloading downstream.
    if "timestamp" in catalysts.columns:
        catalysts = catalysts.copy()
        catalysts["timestamp"] = pd.to_datetime(catalysts["timestamp"], utc=True, errors="coerce")
    if "timestamp" in news_scores.columns:
        news_scores = news_scores.copy()
        news_scores["timestamp"] = pd.to_datetime(news_scores["timestamp"], utc=True, errors="coerce")
    # Explicitly exclude quarantined/post-result legacy evidence if an older
    # Parquet artifact still contains it.
    hindsight_mask = pd.Series(False, index=catalysts.index)
    for column in ("catalyst_kind", "event_type"):
        if column in catalysts.columns:
            hindsight_mask |= catalysts[column].astype(str).str.lower().str.contains("earnings_result")
    catalysts = catalysts.loc[~hindsight_mask].copy()
    base_columns = [
        "catalyst_id",
        "record_id",
        "ticker",
        "timestamp",
        "event_time",
        "published_at",
        "observed_at",
        "available_at",
        "source_record_id",
        "source_artifact_hash",
        "timestamp_semantics_version",
        "catalyst_kind",
        "event_type",
        "relation_type",
        "impact_role",
    ]
    out = catalysts[[column for column in base_columns if column in catalysts.columns]].copy()
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
    # Keep the producer's directional score on its original scale.  This is
    # not a calibrated probability and is never copied into a probability map.
    out["catalyst_score"] = out["news_similarity_score"].fillna(out["scheduled_event_score"])
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
