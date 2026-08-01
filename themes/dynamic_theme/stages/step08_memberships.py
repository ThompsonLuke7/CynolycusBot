"""Step 8 — durable ticker/theme membership scores.

The compatibility output remains the latest ``ticker_theme_membership`` view.
The separate history output is append-preserving and keyed by represented date,
ticker, semantic theme ID, and taxonomy version.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    BGE_MODEL,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    SEED_THEMES,
    TICKER_CLUSTERS_PATH,
    TICKER_EMBEDDINGS_PATH,
    TICKER_MEMBERSHIP_PATH,
    TICKER_MEMBERSHIP_HISTORY_PATH,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step02_embed import load_embeddings_matrix
from themes.dynamic_theme.stages.step03_cluster import compute_centroids
from themes.dynamic_theme.stages.step05_claude_labeling import load_registry

logger = logging.getLogger(__name__)

UTC = timezone.utc
PRODUCER_VERSION = "dynamic-theme@1"
_EPHEMERAL_THEME_RE = re.compile(r"^cluster_-?\d+$")
_HISTORY_COLUMNS = [
    "as_of",
    "available_at",
    "generated_at",
    "ticker",
    "theme",
    "membership_score",
    "taxonomy_version",
    "producer_version",
]
_HISTORY_KEY = ["as_of", "ticker", "theme", "taxonomy_version"]
_IGNORED_CANONICAL_KEYS = {
    "path",
    "local_path",
    "filepath",
    "file_path",
    "timestamp",
    "datetime",
    "date",
    "generated_at",
    "available_at",
    "created_at",
}
_CANONICAL_THEME_ID_RE = re.compile(r"^.+--[0-9a-f]{64}$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware_utc(value: object, *, field_name: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field_name} must be timezone-aware")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _canonical_value(value: object, *, key: str | None = None) -> object:
    if key is not None:
        normalized_key = key.lower()
        if (
            normalized_key in _IGNORED_CANONICAL_KEYS
            or normalized_key.endswith("_path")
            or normalized_key.endswith("_timestamp")
            or normalized_key.endswith("_at")
        ):
            return None
    if isinstance(value, Mapping):
        result = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            normalized = _canonical_value(raw_value, key=str(raw_key))
            if normalized is not None:
                result[str(raw_key)] = normalized
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_canonical_value(item) for item in value]
        return sorted(
            values,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("taxonomy inputs must be finite")
        return value
    return str(value)


def _theme_label(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        raise ValueError("theme label must be a non-empty semantic identifier")
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        raise ValueError("theme label must be a non-empty semantic identifier")
    label = str(value).strip()
    if not label:
        raise ValueError("theme label must be a non-empty semantic identifier")
    return label


def canonical_theme_id(value: object) -> str:
    """Return a stable semantic ID while retaining the raw label separately."""

    label = _theme_label(value)
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        unicodedata.normalize("NFKC", label).casefold(),
    ).strip("_")
    if not normalized:
        normalized = "theme"
    if _EPHEMERAL_THEME_RE.fullmatch(normalized):
        raise ValueError(f"ephemeral cluster label is not a durable theme ID: {label}")
    if _CANONICAL_THEME_ID_RE.fullmatch(label):
        return label
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return f"{normalized[:48]}--{digest}"


def _theme_id_map(values: Iterable[object]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for value in values:
        label = _theme_label(value)
        theme_id = canonical_theme_id(label)
        prior = ids.get(theme_id)
        if prior is not None and prior != label:
            raise ValueError(
                f"theme ID collision: {prior!r} and {label!r} map to {theme_id}"
            )
        ids[theme_id] = label
    return ids


def _registry_theme_ids(registry_df: pd.DataFrame | None) -> list[str]:
    if registry_df is None or registry_df.empty:
        return []
    if "theme_name" in registry_df.columns:
        column = "theme_name"
    elif "theme_id" in registry_df.columns:
        column = "theme_id"
    else:
        column = "theme"
    if column not in registry_df.columns:
        raise ValueError("theme registry must contain theme_name")
    return sorted(_theme_id_map(registry_df[column].tolist()))


def _default_seed_members() -> dict[str, list[str]]:
    return {
        canonical_theme_id(seed["theme_name"]): sorted(
            {str(ticker).strip().upper() for ticker in seed.get("anchor_tickers", ())}
        )
        for seed in SEED_THEMES
    }


def _default_clustering_parameters() -> dict[str, object]:
    return {
        "umap": {
            "n_components": UMAP_N_COMPONENTS,
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
        },
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": HDBSCAN_METRIC,
            "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD,
        },
    }


def canonical_taxonomy_json(
    registry_df: pd.DataFrame | None = None,
    *,
    theme_ids: Iterable[object] | None = None,
    seed_members: Mapping[object, Iterable[object]] | None = None,
    embedding_model: str = BGE_MODEL,
    clustering_parameters: Mapping[str, object] | None = None,
) -> str:
    """Return path/time/order-independent canonical taxonomy material."""

    if theme_ids is None:
        ids = _registry_theme_ids(registry_df)
    else:
        ids = sorted(_theme_id_map(theme_ids))

    raw_seed_members = _default_seed_members() if seed_members is None else seed_members
    normalized_seed_members = {
        canonical_theme_id(name): sorted(
            {str(ticker).strip().upper() for ticker in members}
        )
        for name, members in raw_seed_members.items()
    }
    ids = sorted(set(ids) | set(normalized_seed_members))
    clustering = (
        _default_clustering_parameters()
        if clustering_parameters is None
        else clustering_parameters
    )
    payload = {
        "schema": "dynamic-theme-taxonomy@1",
        "theme_ids": ids,
        "seed_members": normalized_seed_members,
        "embedding_model": str(embedding_model),
        "clustering_parameters": _canonical_value(clustering),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_taxonomy_version(
    registry_df: pd.DataFrame | None = None,
    *,
    theme_ids: Iterable[object] | None = None,
    seed_members: Mapping[object, Iterable[object]] | None = None,
    embedding_model: str = BGE_MODEL,
    clustering_parameters: Mapping[str, object] | None = None,
) -> str:
    """Hash semantic taxonomy inputs, never ephemeral cluster identifiers."""

    canonical = canonical_taxonomy_json(
        registry_df,
        theme_ids=theme_ids,
        seed_members=seed_members,
        embedding_model=embedding_model,
        clustering_parameters=clustering_parameters,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"taxonomy:{digest}"


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_score(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("membership_score must be a finite numeric score")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("membership_score must be a finite numeric score") from exc
    if not math.isfinite(result):
        raise ValueError("membership_score must be finite")
    return result


def _taxonomy_from_current(current: pd.DataFrame) -> str:
    column_value = current.get("taxonomy_version")
    if column_value is not None:
        values = {str(value) for value in column_value.dropna().tolist()}
        if len(values) != 1:
            raise ValueError("current memberships must contain one taxonomy_version")
        return values.pop()
    attr_value = current.attrs.get("taxonomy_version")
    if attr_value is not None and str(attr_value).strip():
        return str(attr_value).strip()
    return compute_taxonomy_version(theme_ids=current["theme"].tolist())


def _history_key_tuples(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        (
            "\x1f".join(values)
            for values in frame[_HISTORY_KEY]
            .astype(str)
            .itertuples(index=False, name=None)
        ),
        index=frame.index,
        dtype="string",
    )


def _dedupe_history_rows(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=_HISTORY_COLUMNS)
    keys = _history_key_tuples(frame)
    duplicate_mask = keys.duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask].copy()
        for _, group in duplicates.groupby(_HISTORY_KEY, sort=False, dropna=False):
            if len(group.drop_duplicates(_HISTORY_COLUMNS)) != 1:
                raise ValueError(f"conflicting immutable membership rows in {source_name}")
        frame = frame.loc[~keys.duplicated(keep="first")].copy()
    return frame


def _normalize_existing_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    existing = pd.read_parquet(history_path)
    if list(existing.columns) != _HISTORY_COLUMNS:
        raise ValueError(
            f"membership history schema mismatch: expected {_HISTORY_COLUMNS}, "
            f"got {list(existing.columns)}"
        )
    existing = existing.copy()
    existing["as_of"] = pd.to_datetime(existing["as_of"], errors="raise").dt.date
    for column in ("available_at", "generated_at"):
        values = [
            _aware_utc(value, field_name=f"history.{column}")
            for value in existing[column].tolist()
        ]
        existing[column] = pd.to_datetime(values, utc=True)
    _validate_history_timestamps(existing, source_name=str(history_path))
    existing["ticker"] = existing["ticker"].map(lambda value: str(value).strip().upper())
    existing["theme"] = existing["theme"].map(_theme_label)
    existing["membership_score"] = existing["membership_score"].map(_validate_score)
    existing["taxonomy_version"] = existing["taxonomy_version"].map(str)
    existing["producer_version"] = existing["producer_version"].map(str)
    return _dedupe_history_rows(existing, source_name=str(history_path))


def _validate_history_timestamps(frame: pd.DataFrame, *, source_name: str) -> None:
    for row in frame.itertuples(index=False):
        as_of = row.as_of
        generated_at = row.generated_at.to_pydatetime()
        available_at = row.available_at.to_pydatetime()
        as_of_start = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
        if generated_at < as_of_start:
            raise ValueError(f"generated_at precedes as_of in {source_name}")
        if generated_at > available_at:
            raise ValueError(f"generated_at follows available_at in {source_name}")


def append_membership_history(
    current: pd.DataFrame,
    *,
    history_path: Path,
    as_of: date,
    generated_at: datetime,
) -> pd.DataFrame:
    """Append new immutable membership evidence and atomically replace history."""

    required = {"ticker", "theme", "membership_score"}
    missing = sorted(required - set(current.columns))
    if missing:
        raise ValueError(f"membership frame missing columns: {missing}")
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    generated_at_utc = _aware_utc(generated_at, field_name="generated_at")
    available_at_utc = _aware_utc(_utc_now(), field_name="available_at")
    if generated_at_utc > available_at_utc:
        raise ValueError("generated_at follows available_at")
    as_of_start = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
    if generated_at_utc < as_of_start:
        raise ValueError("generated_at precedes as_of")
    taxonomy_version = _taxonomy_from_current(current)
    producer_version = str(current.attrs.get("producer_version", PRODUCER_VERSION))

    incoming_rows: list[dict[str, object]] = []
    for row in current.itertuples(index=False):
        row_data = row._asdict()
        incoming_rows.append(
            {
                "as_of": as_of,
                "available_at": available_at_utc,
                "generated_at": generated_at_utc,
                "ticker": str(row_data["ticker"]).strip().upper(),
                "theme": _theme_label(row_data["theme"]),
                "membership_score": _validate_score(row_data["membership_score"]),
                "taxonomy_version": taxonomy_version,
                "producer_version": producer_version,
            }
        )
    incoming = pd.DataFrame(incoming_rows, columns=_HISTORY_COLUMNS)
    incoming = _dedupe_history_rows(incoming, source_name="current memberships")
    _validate_history_timestamps(incoming, source_name="current memberships")
    existing = _normalize_existing_history(Path(history_path))

    existing_keys = set(_history_key_tuples(existing).tolist())
    incoming = incoming.loc[~_history_key_tuples(incoming).isin(existing_keys)].copy()
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values(_HISTORY_KEY, kind="mergesort").reset_index(drop=True)
    combined = combined.reindex(columns=_HISTORY_COLUMNS)
    _atomic_write_parquet(combined, Path(history_path))
    return combined


def _cosine_sim_matrix(ticker_matrix: np.ndarray, centroid_matrix: np.ndarray) -> np.ndarray:
    """Return [N_tickers, N_themes] cosine similarity matrix."""

    tn = ticker_matrix / (np.linalg.norm(ticker_matrix, axis=1, keepdims=True) + 1e-10)
    cn = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10)
    return (tn @ cn.T).astype(np.float32)


def compute_memberships(
    embeddings_df: pd.DataFrame | None = None,
    clusters_df: pd.DataFrame | None = None,
    registry_df: pd.DataFrame | None = None,
    *,
    as_of: pd.Timestamp | None = None,
    generated_at: datetime | None = None,
) -> pd.DataFrame:
    """Compute soft scores, publish the latest view, and append history."""

    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)
    generated_at = generated_at or _utc_now()

    if embeddings_df is None:
        tickers, matrix, _date = load_embeddings_matrix()
    else:
        tickers = embeddings_df["ticker"].astype(str).tolist()
        matrix = np.array(embeddings_df["embedding"].tolist(), dtype=np.float32)

    if clusters_df is None:
        clusters_df = pd.read_parquet(TICKER_CLUSTERS_PATH)
    if registry_df is None:
        registry_df = load_registry(latest_only=True)

    taxonomy_version = compute_taxonomy_version(registry_df)
    empty = pd.DataFrame(columns=["ticker", "theme", "membership_score", "date"])
    empty.attrs["taxonomy_version"] = taxonomy_version
    empty.attrs["producer_version"] = PRODUCER_VERSION

    if registry_df.empty:
        logger.warning("Theme registry is empty — cannot compute memberships")
        return empty

    centroids_by_id = compute_centroids(matrix, tickers, clusters_df)
    id_to_theme: dict[object, str] = {}
    theme_ids: dict[str, str] = {}
    for cluster_id, theme_name in zip(registry_df["cluster_id"], registry_df["theme_name"]):
        raw_label = _theme_label(theme_name)
        semantic_id = canonical_theme_id(raw_label)
        prior = theme_ids.get(semantic_id)
        if prior is not None and prior != raw_label:
            raise ValueError(
                f"theme ID collision: {prior!r} and {raw_label!r} map to {semantic_id}"
            )
        theme_ids[semantic_id] = raw_label
        id_to_theme[cluster_id] = raw_label

    from themes.dynamic_theme.seed_themes import seed_centroids

    seed_cents, seed_names = seed_centroids(tickers, matrix)
    existing_names = {canonical_theme_id(name) for name in id_to_theme.values()}
    for cluster_id, name in seed_names.items():
        raw_label = _theme_label(name)
        semantic_name = canonical_theme_id(raw_label)
        if semantic_name in existing_names:
            logger.info("Seed theme '%s' already emerged this run — not injecting seed", name)
            continue
        centroids_by_id[cluster_id] = seed_cents[cluster_id]
        id_to_theme[cluster_id] = raw_label

    valid_ids = sorted(
        (cluster_id for cluster_id in centroids_by_id if cluster_id in id_to_theme),
        key=lambda cluster_id: canonical_theme_id(id_to_theme[cluster_id]),
    )
    if not valid_ids:
        logger.warning("No cluster centroids match current registry")
        _atomic_write_parquet(empty, TICKER_MEMBERSHIP_PATH)
        append_membership_history(
            empty,
            history_path=TICKER_MEMBERSHIP_HISTORY_PATH,
            as_of=as_of.date(),
            generated_at=generated_at,
        )
        return empty

    theme_names = [id_to_theme[cluster_id] for cluster_id in valid_ids]
    centroid_matrix = np.array(
        [centroids_by_id[cluster_id] for cluster_id in valid_ids], dtype=np.float32
    )
    logger.info(
        "Computing membership scores: %d tickers × %d themes",
        len(tickers),
        len(theme_names),
    )
    sim_matrix = _cosine_sim_matrix(matrix, centroid_matrix)
    rows = []
    for i, ticker in enumerate(tickers):
        for j, theme in enumerate(theme_names):
            score = float(sim_matrix[i, j])
            if score > 0.0:
                rows.append(
                    {
                        "ticker": ticker,
                        "theme": theme,
                        "membership_score": score,
                        "date": as_of,
                    }
                )
    out = pd.DataFrame(rows, columns=["ticker", "theme", "membership_score", "date"])
    out.attrs["taxonomy_version"] = taxonomy_version
    out.attrs["producer_version"] = PRODUCER_VERSION
    _atomic_write_parquet(out, TICKER_MEMBERSHIP_PATH)
    append_membership_history(
        out,
        history_path=TICKER_MEMBERSHIP_HISTORY_PATH,
        as_of=as_of.date(),
        generated_at=generated_at,
    )
    logger.info(
        "Wrote %s rows=%d avg_score=%.3f",
        TICKER_MEMBERSHIP_PATH,
        len(out),
        out["membership_score"].mean() if not out.empty else 0.0,
    )
    return out


def load_memberships(*, latest_only: bool = True) -> pd.DataFrame:
    if not TICKER_MEMBERSHIP_PATH.exists():
        return pd.DataFrame(columns=["ticker", "theme", "membership_score", "date"])
    df = pd.read_parquet(TICKER_MEMBERSHIP_PATH)
    if latest_only and not df.empty and "date" in df.columns:
        latest = df["date"].max()
        df = df[df["date"] == latest].copy()
    return df


def get_primary_theme(memberships_df: pd.DataFrame) -> pd.DataFrame:
    """Return {ticker, primary_theme, membership_score} — highest score per ticker."""

    if memberships_df.empty:
        return pd.DataFrame(columns=["ticker", "primary_theme", "membership_score"])
    idx = memberships_df.groupby("ticker")["membership_score"].idxmax()
    top = memberships_df.loc[idx, ["ticker", "theme", "membership_score"]].rename(
        columns={"theme": "primary_theme"}
    )
    return top.reset_index(drop=True)


__all__ = [
    "append_membership_history",
    "canonical_taxonomy_json",
    "canonical_theme_id",
    "compute_memberships",
    "compute_taxonomy_version",
    "get_primary_theme",
    "load_memberships",
]
