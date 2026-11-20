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
    GEOGRAPHIC_THEME_PATTERNS,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HOME_COUNTRY,
    MIN_CLUSTER_LABEL_CONFIDENCE,
    MIN_INDUSTRY_PEER_SHARE,
    MIN_INDUSTRY_PEERS_IN_THEME,
    OFFSHORE_DOMICILES,
    SEED_THEMES,
    TICKER_CLUSTERS_PATH,
    TICKER_PEER_PINS,
    TICKER_EMBEDDINGS_PATH,
    TICKER_MEMBERSHIP_PATH,
    TICKER_MEMBERSHIP_HISTORY_PATH,
    TICKER_PRIMARY_ANCHOR_PATH,
    UMAP_METRIC,
    UNCLASSIFIED_THEME,
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


def assert_publishable_memberships(current: pd.DataFrame) -> pd.DataFrame:
    """Normalize a membership frame, or raise before any of it is published.

    ``_atomic_write_parquet`` only stops a reader seeing a *torn* file; it
    publishes a *wrong* one just as atomically. This check used to live solely
    inside ``append_membership_history``, which ``compute_memberships`` called
    AFTER it had already written ``TICKER_MEMBERSHIP_PATH`` — so on 2026-08-17
    and 2026-08-24 the run raised, reported exit=1, and still left 435,450
    conflicting rows (77% of the file) in the artifact every consumer reads,
    while the immutable history the check was protecting was never written at
    all. Validate first, publish second.

    Returns the normalized, de-duplicated frame so callers do not repeat the
    work. Identical duplicate rows are collapsed; *conflicting* ones raise.
    """
    required = {"ticker", "theme", "membership_score"}
    missing = sorted(required - set(current.columns))
    if missing:
        raise ValueError(f"membership frame missing columns: {missing}")

    validated_rows: list[dict[str, object]] = []
    for row in current.itertuples(index=False):
        row_data = row._asdict()
        validated_rows.append(
            {
                "ticker": str(row_data["ticker"]).strip().upper(),
                "theme": _theme_label(row_data["theme"]),
                "membership_score": _validate_score(row_data["membership_score"]),
            }
        )
    validated = pd.DataFrame(
        validated_rows,
        columns=["ticker", "theme", "membership_score"],
    )
    duplicate_mask = validated.duplicated(["ticker", "theme"], keep=False)
    if duplicate_mask.any():
        duplicates = validated.loc[duplicate_mask]
        for _, group in duplicates.groupby(["ticker", "theme"], sort=False):
            if group["membership_score"].nunique(dropna=False) != 1:
                raise ValueError("conflicting immutable membership rows in current memberships")
        validated = validated.drop_duplicates(["ticker", "theme"], keep="first")
    return validated


def append_membership_history(
    current: pd.DataFrame,
    *,
    history_path: Path,
    as_of: date,
    generated_at: datetime,
) -> pd.DataFrame:
    """Append new immutable membership evidence and atomically replace history.

    ``available_at`` marks when validated current and prior evidence is ready
    for serialization/publication.  It is captured immediately before the new
    immutable rows are materialized; the following same-directory atomic write
    does not infer or rewrite that timestamp from filesystem metadata.
    """

    validated = assert_publishable_memberships(current)
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    generated_at_utc = _aware_utc(generated_at, field_name="generated_at")
    as_of_start = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
    if generated_at_utc < as_of_start:
        raise ValueError("generated_at precedes as_of")
    taxonomy_version = _taxonomy_from_current(current)
    producer_version = str(current.attrs.get("producer_version", PRODUCER_VERSION))

    existing = _normalize_existing_history(Path(history_path))

    # Semantic boundary: all inputs are validated and the record is now ready
    # for serialization/publication.  The atomic Parquet replacement follows.
    available_at_utc = _aware_utc(_utc_now(), field_name="available_at")
    if generated_at_utc > available_at_utc:
        raise ValueError("generated_at follows available_at")
    incoming_rows = [
        {
            "as_of": as_of,
            "available_at": available_at_utc,
            "generated_at": generated_at_utc,
            "ticker": row.ticker,
            "theme": row.theme,
            "membership_score": row.membership_score,
            "taxonomy_version": taxonomy_version,
            "producer_version": producer_version,
        }
        for row in validated.itertuples(index=False)
    ]
    incoming = pd.DataFrame(incoming_rows, columns=_HISTORY_COLUMNS)
    incoming = _dedupe_history_rows(incoming, source_name="current memberships")
    _validate_history_timestamps(incoming, source_name="current memberships")

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


def _collapse_duplicate_themes(
    sim_matrix: np.ndarray, theme_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Merge columns that carry the same theme name into a single theme.

    step05 deliberately lets Claude reuse a name already in play rather than
    mint a near-duplicate for a gray-zone cluster, so two clusters legitimately
    arrive here under one name. They are one theme, and a ticker belongs to it
    if it is close to ANY of its lobes — hence ``max`` rather than a mean, which
    would invent a centroid in the empty space between two genuinely separate
    lobes and understate membership for both.

    Left unmerged, the caller emits one row per (ticker, theme-*instance*): on
    2026-08-24 fifty names were claimed by more than one cluster
    (``mixed_small_cap_value`` by eleven) and 435,450 of 563,182 rows were
    duplicate (ticker, theme) pairs carrying conflicting scores. The 2026-08-17
    exclusivity fix in step05 closed only the carry-forward half of this; the
    Claude relabel path had no such constraint and was masked behind it.
    """
    first_col: dict[str, int] = {}
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for idx, name in enumerate(theme_names):
        key = canonical_theme_id(name)
        if key not in first_col:
            first_col[key] = idx
            order.append(key)
            members[key] = []
        members[key].append(idx)
    if len(order) == len(theme_names):
        return sim_matrix, theme_names

    merged_names = [theme_names[first_col[key]] for key in order]
    collapsed = np.column_stack(
        [sim_matrix[:, members[key]].max(axis=1) for key in order]
    ).astype(np.float32)
    for key in order:
        if len(members[key]) > 1:
            logger.info(
                "  theme '%s' spans %d clusters — merged (max membership)",
                theme_names[first_col[key]],
                len(members[key]),
            )
    logger.info(
        "Collapsed %d cluster columns into %d distinct themes",
        len(theme_names),
        len(merged_names),
    )
    return collapsed, merged_names


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
    # Only names backed by a real cluster centroid count as "emerged". step05
    # gap-fills the seed's own registry row (reserved negative id, no centroid);
    # counting it here skipped injection and left the seed with zero members.
    grab_bags = grab_bag_cluster_ids(registry_df)
    if grab_bags:
        logger.info(
            "Not publishing %d grab-bag cluster(s) as themes (label confidence < %.2f)",
            len(grab_bags & set(centroids_by_id)), MIN_CLUSTER_LABEL_CONFIDENCE,
        )
    existing_names = {
        canonical_theme_id(name)
        for cluster_id, name in id_to_theme.items()
        if cluster_id in centroids_by_id and cluster_id not in grab_bags
    }
    for cluster_id, name in seed_names.items():
        raw_label = _theme_label(name)
        semantic_name = canonical_theme_id(raw_label)
        if semantic_name in existing_names:
            logger.info("Seed theme '%s' already emerged this run — not injecting seed", name)
            continue
        centroids_by_id[cluster_id] = seed_cents[cluster_id]
        id_to_theme[cluster_id] = raw_label

    valid_ids = sorted(
        (
            cluster_id for cluster_id in centroids_by_id
            if cluster_id in id_to_theme and cluster_id not in grab_bags
        ),
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
    sim_matrix, theme_names = _collapse_duplicate_themes(sim_matrix, theme_names)
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
    # Precondition, not a formality: publishing before this check is what put a
    # 77%-duplicate file in front of every consumer on 2026-08-17 and 08-24.
    assert_publishable_memberships(out)
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


def grab_bag_cluster_ids(registry_df: pd.DataFrame | None) -> set:
    """Cluster ids the labeler could not confidently name as one business.

    Empty when the registry carries no confidence column; a missing confidence
    on a row is treated as unknown, not as low.

    Only the LATEST registry date is read. HDBSCAN reuses cluster ids every
    week and the weekly pipeline passes the full multi-date registry, so judging
    ids across history unpublishes nearly everything (2026-10-05: 181 of 191
    clusters dropped, 9 themes published, because 184 ids had been low-confidence
    in SOME past week; the real count that week was 28).
    """
    if registry_df is None or registry_df.empty or "confidence" not in registry_df.columns:
        return set()
    if "date" in registry_df.columns and registry_df["date"].notna().any():
        registry_df = registry_df[registry_df["date"] == registry_df["date"].max()]
    conf = pd.to_numeric(registry_df["confidence"], errors="coerce")
    return set(registry_df.loc[conf < MIN_CLUSTER_LABEL_CONFIDENCE, "cluster_id"].tolist())


def assign_primary_themes(
    memberships_df: pd.DataFrame,
    clusters_df: pd.DataFrame | None,
    registry_df: pd.DataFrame | None,
    industries: Mapping[str, str] | None,
    countries: Mapping[str, str] | None = None,
    peer_pins: Mapping[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """{ticker, primary_theme, membership_score, source} placed by business type.

    * A ticker in a confidently-named cluster (or a seed anchor) keeps its
      nearest published theme -- unless it is the only company of its industry
      there                                                     -> source "cluster"
    * Any other ticker (HDBSCAN noise, grab-bag member, industry singleton) gets
      its best-matching theme among those holding clustered companies of its own
      industry; a home-market ticker counts home-market peers only
                                                                -> source "industry"
    * and is UNCLASSIFIED_THEME when no theme does              -> source "unclassified"
    * ``peer_pins`` (default TICKER_PEER_PINS) then override: the ticker takes
      the theme most of its listed peers were given             -> source "pinned"

    Without cluster assignments this is plain nearest-theme (source "similarity").
    """
    best = get_primary_theme(memberships_df)
    if best.empty or clusters_df is None or clusters_df.empty:
        return best.assign(source="similarity")

    grab_bags = grab_bag_cluster_ids(registry_df)
    hard = clusters_df.assign(ticker=clusters_df["ticker"].astype(str)).set_index("ticker")["cluster_id"]
    seed_anchors = {t for members in _default_seed_members().values() for t in members}
    anchored = set(hard.index[(hard >= 0) & ~hard.isin(grab_bags)]) | seed_anchors

    industry = pd.Series(dict(industries or {}), dtype=object)
    country = pd.Series(dict(countries or {}), dtype=object)
    home = country.eq(HOME_COUNTRY)

    kept = best[best["ticker"].astype(str).isin(anchored)].assign(source="cluster")
    # A clustered ticker's primary theme is its OWN group's theme (2026-11-18). The nearest
    # centroid disagreed for 9% of clustered names -- incl. NVDA and ANET, whose own group
    # is ai_infrastructure, picked as analog semis / memory -- and near-ties made it flip
    # weekly. Nearest centroid remains only when the own group's theme is not published.
    kept, own = _own_group_themes(kept, hard, registry_df, memberships_df)
    # A seed anchor defines its seed theme, so it sits there even when another theme scores a
    # hair higher (11-03: AAPL 0.775 agricultural_chemicals_fertilizers vs 0.760 mega_cap_platforms).
    kept = _place_seed_anchors(kept, memberships_df)
    kept = kept.assign(industry=kept["ticker"].map(industry))
    # An industry singleton has no business-type support in its theme (ORCL in a quantum group,
    # a software name among banks) -- unless the theme is its own group AND that group is
    # cross-industry (no industry holds half its members): trade groups such as
    # ai_infrastructure (NVDA, ANET, NBIS, ... over 6+ industries) hold one chip maker by design.
    n_same = kept.groupby(["primary_theme", "industry"])["ticker"].transform("size")
    in_cross_group = kept["ticker"].isin(own) & kept["ticker"].map(hard).isin(
        _cross_industry_groups(hard, industry))
    alone = (n_same.eq(1) & kept["industry"].notna() & ~kept["ticker"].isin(seed_anchors)
             & ~in_cross_group)
    kept = kept[~alone]
    # A theme named for a place is closed to other countries' companies on every route:
    # a clustered ticker whose nearest theme is one is re-placed with industry peers.
    shut_out = closed_to(kept["primary_theme"], kept["ticker"].map(country))
    kept = kept[~shut_out]

    def _eligible(peers: pd.DataFrame) -> pd.MultiIndex:
        """(industry, theme) pairs where that industry's clustered names sit."""
        counts = peers.groupby(["industry", "primary_theme"]).size()
        share = counts / counts.groupby(level="industry").transform("sum")
        return counts.index[
            (counts >= MIN_INDUSTRY_PEERS_IN_THEME) & (share >= MIN_INDUSTRY_PEER_SHARE)
        ]

    peers = kept.dropna(subset=["industry"])
    eligible_any = _eligible(peers)
    eligible_home = _eligible(peers[peers["ticker"].map(home).eq(True)])

    loose = memberships_df[~memberships_df["ticker"].astype(str).isin(set(kept["ticker"]))]
    loose = loose.assign(industry=loose["ticker"].map(industry))
    pair = pd.MultiIndex.from_frame(loose[["industry", "theme"]])
    is_home = loose["ticker"].map(home).eq(True).to_numpy()
    backed = loose[np.where(is_home, pair.isin(eligible_home), pair.isin(eligible_any))]
    backed = backed[~closed_to(backed["theme"], backed["ticker"].map(country))]
    # A home-market ticker with no home-backed theme may use one backed by foreign peers of
    # its industry, provided at least one home-market peer of that industry sits there too
    # (JPM/WFC: only BAC and C of the US diversified banks were clustered -- too few on their
    # own -- both in global_diversified_banks with HSBC/TD/BMO; XOM with CVX). A theme of
    # foreign peers only stays closed (GOOGL/META into Chinese ADRs, 2026-10-05), as do
    # place-named themes (closed_to).
    home_peers = peers[peers["ticker"].map(home).eq(True)].groupby(["industry", "primary_theme"]).size().index
    homeless = is_home & ~loose["ticker"].isin(set(backed["ticker"])).to_numpy()
    fallback = loose[homeless & pair.isin(eligible_any) & pair.isin(home_peers)]
    fallback = fallback[~closed_to(fallback["theme"], fallback["ticker"].map(country))]
    backed = pd.concat([backed, fallback])
    placed = (
        backed.loc[backed.groupby("ticker")["membership_score"].idxmax(),
                   ["ticker", "theme", "membership_score"]]
        .rename(columns={"theme": "primary_theme"})
        .assign(source="industry")
    )
    settled = set(kept["ticker"]) | set(placed["ticker"])
    rest = best.loc[~best["ticker"].isin(settled), ["ticker"]].assign(
        primary_theme=UNCLASSIFIED_THEME, membership_score=np.nan, source="unclassified"
    )
    out = pd.concat([kept.drop(columns="industry"), placed, rest], ignore_index=True)
    out = _apply_peer_pins(out, memberships_df, TICKER_PEER_PINS if peer_pins is None else peer_pins)
    logger.info(
        "Primary themes: %s (industry singletons re-placed: %d; clustered names shut out of "
        "place-named themes: %d)",
        out["source"].value_counts().to_dict(), int(alone.sum()), int(shut_out.sum()),
    )
    return out.sort_values("ticker").reset_index(drop=True)


def _cross_industry_groups(hard: pd.Series, industry: pd.Series, max_share: float = 0.5) -> set[int]:
    """Cluster ids whose largest industry holds less than ``max_share`` of the known members."""
    df = pd.DataFrame({"cid": hard, "industry": industry.reindex(hard.index)}).dropna()
    df = df[df["cid"] >= 0]
    if df.empty:
        return set()
    top = df.groupby("cid")["industry"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return {int(c) for c in top.index[top < max_share]}


def _own_group_themes(
    kept: pd.DataFrame,
    hard: pd.Series,
    registry_df: pd.DataFrame | None,
    memberships_df: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """Move each clustered ticker to its own group's theme when that theme is published.

    Returns the updated frame and the tickers now sitting in their own group's theme.
    """
    if registry_df is None or registry_df.empty or kept.empty:
        return kept, set()
    reg = registry_df
    if "date" in reg.columns and reg["date"].notna().any():
        reg = reg[reg["date"] == reg["date"].max()]
    id_to_theme = {int(c): str(n) for c, n in zip(reg["cluster_id"], reg["theme_name"])}
    published = set(memberships_df["theme"].astype(str))
    score = memberships_df.set_index(["ticker", "theme"])["membership_score"].to_dict()
    out = kept.copy()
    own: set[str] = set()
    for i, ticker in zip(out.index, out["ticker"].astype(str)):
        cid = hard.get(ticker)
        theme = id_to_theme.get(int(cid)) if cid is not None and cid >= 0 else None
        if theme is None or theme not in published:
            continue
        own.add(ticker)
        if out.at[i, "primary_theme"] != theme:
            out.at[i, "primary_theme"] = theme
            out.at[i, "membership_score"] = float(score.get((ticker, theme), np.nan))
    return out, own


def _place_seed_anchors(kept: pd.DataFrame, memberships_df: pd.DataFrame) -> pd.DataFrame:
    """Put each seed anchor in its seed theme (when that theme was scored this run)."""
    score = memberships_df.set_index(["ticker", "theme"])["membership_score"]
    # Same seed source as the anchored set above; its keys are canonical ids ("name--sha256"),
    # memberships carry the plain name.
    seed_of = {t: str(key).split("--", 1)[0] for key, members in _default_seed_members().items() for t in members}
    out = kept.copy()
    for i, ticker in zip(out.index, out["ticker"].astype(str)):
        theme = seed_of.get(ticker)
        if theme is not None and (ticker, theme) in score.index:
            out.loc[i, ["primary_theme", "membership_score"]] = [theme, float(score.loc[(ticker, theme)])]
    return out


def theme_countries(theme: object) -> frozenset[str] | None:
    """The countries a place-named theme is for (``latam_banks`` -> Latin America); None if no place."""
    name = str(theme or "")
    allowed: set[str] = set()
    for pattern, countries in GEOGRAPHIC_THEME_PATTERNS:
        if re.search(pattern, name):
            allowed |= countries
    return frozenset(allowed) if allowed else None


def closed_to(themes: pd.Series, ticker_country: pd.Series) -> pd.Series:
    """True where a place-named theme is for other countries than the ticker's.

    An unknown country or an offshore holding domicile is never shut out.
    """
    scopes = themes.map(theme_countries)
    closed = [
        scope is not None and isinstance(c, str) and c not in OFFSHORE_DOMICILES and c not in scope
        for scope, c in zip(scopes, ticker_country)
    ]
    return pd.Series(closed, index=themes.index, dtype=bool)


def _apply_peer_pins(
    out: pd.DataFrame,
    memberships_df: pd.DataFrame,
    peer_pins: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    """Move each pinned ticker to the theme most of its peers were given."""
    theme_of = out.set_index("ticker")["primary_theme"]
    score = memberships_df.set_index(["ticker", "theme"])["membership_score"]
    out = out.set_index("ticker")
    for ticker, peer_list in peer_pins.items():
        if ticker not in out.index:
            continue
        votes = theme_of.reindex([p for p in peer_list if p != ticker]).dropna()
        votes = votes[votes != UNCLASSIFIED_THEME]
        if votes.empty:
            logger.warning("Peer pin for %s: none of its peers has a theme -- left as %s",
                           ticker, out.loc[ticker, "primary_theme"])
            continue
        theme = votes.value_counts().index[0]
        out.loc[ticker, ["primary_theme", "membership_score", "source"]] = [
            theme, float(score.get((ticker, theme), np.nan)), "pinned"
        ]
    return out.reset_index()


def anchor_primary_themes(
    primary: pd.DataFrame,
    memberships_df: pd.DataFrame,
    clusters_df: pd.DataFrame | None,
    registry_df: pd.DataFrame | None,
    *,
    anchor_path: Path = TICKER_PRIMARY_ANCHOR_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Freeze each ticker's primary theme for the life of the current re-fit.

    ``primary`` is this week's fresh assignment (``assign_primary_themes``). A ticker
    already anchored under the same re-fit (``clusters_df.refit_id``) keeps the theme
    of its anchored cluster id -- whatever that cluster is called this week -- as long
    as the theme is still published; its source becomes "anchored" when that differs
    from the fresh pick. Anyone else (new ticker, theme gone, unclassified) takes the
    fresh pick, which is anchored from then on (unclassified is re-tried weekly).
    Returns (primary, anchor_to_write); the anchor is None when there is no re-fit id.

    Why: the fresh pick is the nearest centroid, and for ~1 in 5 names the top two
    themes are within 0.01 cosine, so it flipped with every weekly embedding wobble
    (19 of the top 20 by market cap changed theme in five weeks).
    """
    if clusters_df is None or clusters_df.empty or "refit_id" not in clusters_df.columns \
            or registry_df is None or registry_df.empty:
        return primary, None
    refit_id = str(clusters_df["refit_id"].iloc[0])
    grab_bags = grab_bag_cluster_ids(registry_df)
    reg = registry_df[~registry_df["cluster_id"].isin(grab_bags)]
    id_to_theme = {int(c): str(n) for c, n in zip(reg["cluster_id"], reg["theme_name"])}
    theme_to_id: dict[str, int] = {}
    for cid, name in sorted(id_to_theme.items()):
        theme_to_id.setdefault(name, cid)
    published = set(memberships_df["theme"].astype(str))
    score = memberships_df.set_index(["ticker", "theme"])["membership_score"].to_dict()

    anchor = pd.read_parquet(anchor_path) if anchor_path.exists() else pd.DataFrame(
        columns=["ticker", "cluster_id", "refit_id"])
    held = {str(t): int(c) for t, c, r in zip(anchor["ticker"], anchor["cluster_id"], anchor["refit_id"])
            if str(r) == refit_id}

    out = primary.copy()
    out["ticker"] = out["ticker"].astype(str)
    keep_rows, changed = [], 0
    for i, row in out.iterrows():
        cid = held.get(row["ticker"])
        theme = id_to_theme.get(cid) if cid is not None else None
        if theme is not None and theme in published:
            if row["primary_theme"] != theme:
                out.at[i, "primary_theme"] = theme
                out.at[i, "membership_score"] = float(score.get((row["ticker"], theme), np.nan))
                out.at[i, "source"] = "anchored"
                changed += 1
            keep_rows.append((row["ticker"], cid))
            continue
        fresh_id = theme_to_id.get(str(row["primary_theme"]))
        if fresh_id is not None:
            keep_rows.append((row["ticker"], fresh_id))
    logger.info("Primary themes: %d held by the re-fit %s anchor against a different fresh pick", changed, refit_id)
    new_anchor = pd.DataFrame(keep_rows, columns=["ticker", "cluster_id"]).assign(refit_id=refit_id)
    return out, new_anchor


__all__ = [
    "anchor_primary_themes",
    "append_membership_history",
    "assign_primary_themes",
    "grab_bag_cluster_ids",
    "assert_publishable_memberships",
    "canonical_taxonomy_json",
    "canonical_theme_id",
    "compute_memberships",
    "compute_taxonomy_version",
    "get_primary_theme",
    "load_memberships",
]
