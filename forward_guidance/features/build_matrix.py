"""Build cached event-level feature, label, and training matrices."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from forward_guidance.config import (
    FEATURES_PATH,
    LABELS_PATH,
    TRAINING_MATRIX,
    ensure_data_dirs,
)
from forward_guidance.data.ingest_events import load_events, read_event_text
from forward_guidance.data.market_data import load_event_market_window
from forward_guidance.data.schema import EarningsEvent, event_from_record
from forward_guidance.features.market_context import compute_forward_labels, compute_market_context
from forward_guidance.features.nlp import (
    DEFAULT_SENTENCE_MODEL,
    embedding_feature_dict,
    embed_sentence_transformer,
    extract_forward_sections,
    extract_structured_guidance_features,
    score_finbert,
)
from forward_guidance.utils.io import flatten_dict, read_json, write_dataframe

logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = {
    "event_id",
    "ticker",
    "earnings_date",
    "report_time",
    "reaction_date",
    "signal_timestamp",
    "fiscal_period",
    "sector",
    "sector_etf",
    "cik",
    "source_url",
    "source_type",
    "available_at",
}

LABEL_COLUMNS = {
    "fwd_ret_5d",
    "fwd_ret_20d",
    "fwd_ret_60d",
    "fwd_60d_excess_ret_vs_spy",
    "fwd_60d_excess_ret_vs_sector",
    "max_runup",
    "max_drawdown",
    "target",
}


def _metrics_for_event(event: EarningsEvent) -> dict[str, object]:
    from forward_guidance.config import RAW_DIR
    from forward_guidance.data.schema import raw_event_dir

    path = raw_event_dir(RAW_DIR, event) / "metrics.json"
    metrics = read_json(path, default={}) or {}
    flat = flatten_dict(metrics, prefix="metric_")
    out: dict[str, object] = {}
    for key, value in flat.items():
        if isinstance(value, (int, float, np.integer, np.floating)) or value is None:
            out[key] = value
    return out


def _guidance_text_for_event(event: EarningsEvent) -> str:
    guidance = read_event_text(event, "guidance_section")
    if guidance.strip():
        return guidance
    text = read_event_text(event, "press_release") or read_event_text(event, "transcript")
    return extract_forward_sections(text).get("forward_guidance", "")


def build_feature_row(
    event: EarningsEvent,
    *,
    generate_embeddings: bool = False,
    generate_finbert: bool = False,
    embedding_model: str = DEFAULT_SENTENCE_MODEL,
) -> tuple[dict[str, object], dict[str, object]]:
    row: dict[str, object] = event.to_record()
    guidance_text = _guidance_text_for_event(event)
    row.update(extract_structured_guidance_features(guidance_text))
    row.update(_metrics_for_event(event))

    bars = load_event_market_window(event)
    market = compute_market_context(event, bars)
    row.update(market)

    guidance_strength = float(row.get("guidance_strength_score") or 0.0)
    bad_reaction = float(row.get("bad_initial_reaction_flag") or 0.0)
    row["guidance_reaction_disagreement_score"] = guidance_strength * bad_reaction

    if generate_embeddings and guidance_text.strip():
        try:
            embedding = embed_sentence_transformer(guidance_text, model_name=embedding_model, event_id=event.event_id)
            row.update(embedding_feature_dict(embedding, prefix="emb_minilm"))
            row["embedding_available"] = 1.0
        except Exception as exc:
            logger.warning("[%s] embedding failed: %s", event.event_id, exc)
            row["embedding_available"] = 0.0
    else:
        row["embedding_available"] = 0.0

    if generate_finbert and guidance_text.strip():
        try:
            row.update(score_finbert(guidance_text))
            row["finbert_available"] = 1.0
        except Exception as exc:
            logger.warning("[%s] FinBERT failed: %s", event.event_id, exc)
            row["finbert_available"] = 0.0
    else:
        row["finbert_available"] = 0.0

    labels = event.to_record()
    labels.update(compute_forward_labels(event, bars))
    return row, labels


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = IDENTITY_COLUMNS | LABEL_COLUMNS | {"metadata"}
    cols: list[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def build_feature_matrix(
    events: Iterable[EarningsEvent] | pd.DataFrame | None = None,
    *,
    events_path: Path | str | None = None,
    force: bool = False,
    generate_embeddings: bool = False,
    generate_finbert: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_data_dirs()
    if not force and FEATURES_PATH.exists() and LABELS_PATH.exists() and TRAINING_MATRIX.exists():
        return pd.read_parquet(FEATURES_PATH), pd.read_parquet(LABELS_PATH), pd.read_parquet(TRAINING_MATRIX)

    if events is None:
        df_events = load_events(events_path) if events_path is not None else load_events()
        event_list = [event_from_record(row) for _, row in df_events.iterrows()]
    elif isinstance(events, pd.DataFrame):
        event_list = [event_from_record(row) for _, row in events.iterrows()]
    else:
        event_list = list(events)

    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for event in event_list:
        try:
            row, labels = build_feature_row(
                event,
                generate_embeddings=generate_embeddings,
                generate_finbert=generate_finbert,
            )
            feature_rows.append(row)
            label_rows.append(labels)
        except Exception as exc:
            logger.warning("[%s] feature row failed: %s", event.event_id, exc)

    features = pd.DataFrame(feature_rows)
    labels = pd.DataFrame(label_rows)
    write_dataframe(features, FEATURES_PATH)
    write_dataframe(labels, LABELS_PATH)

    if not features.empty and not labels.empty:
        join_cols = ["event_id", "ticker", "earnings_date", "reaction_date", "signal_timestamp"]
        labels_keep = [c for c in labels.columns if c in LABEL_COLUMNS or c in join_cols]
        matrix = features.merge(labels[labels_keep], on=join_cols, how="left")
        if "target" in matrix.columns:
            matrix = matrix.loc[matrix["target"].notna()].reset_index(drop=True)
    else:
        matrix = pd.DataFrame()
    write_dataframe(matrix, TRAINING_MATRIX)
    return features, labels, matrix
