"""Dynamic Theme Taxonomy Pipeline.

Daily run  (after market close):
  Step 1  - Build/refresh ticker documents
  Step 2  - Regenerate embeddings
  Step 8  - Recompute soft membership scores
  Step 9  - Generate meta-ranker theme features

Weekly run (Sunday, after the Sunday universe refresh):
  Step 1  - Build ticker documents
  Step 2  - Generate embeddings
  Step 3  - HDBSCAN recluster
  Step 4  - Build cluster summaries
  Step 5  - Claude theme labeling
  Step 6  - Discover new themes
  Step 6b - Dedupe near-duplicate themes
  Step 7  - Update theme relationship graph
  Step 8  - Recompute memberships
  Step 9  - Generate meta features
"""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# Load .env so ANTHROPIC_API_KEY and other secrets are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.nervous_system.contracts.quality import LineageRef
from themes.dynamic_theme.config import (
    THEME_REGISTRY_PATH,
    TICKER_MEMBERSHIP_HISTORY_PATH,
    TICKER_THEME_FEATURES_PATH,
    ensure_outputs,
)
from themes.dynamic_theme.nervous_system_adapter import persist_theme_states
from themes.dynamic_theme.stages.step01_build_documents import build_ticker_documents
from themes.dynamic_theme.stages.step02_embed import generate_embeddings
from themes.dynamic_theme.stages.step03_cluster import cluster_tickers
from themes.dynamic_theme.stages.step04_cluster_summary import build_cluster_summaries
from themes.dynamic_theme.stages.step05_claude_labeling import label_clusters
from themes.dynamic_theme.stages.step06_discovery import discover_new_themes
from themes.dynamic_theme.stages.step06b_theme_dedup import (
    build_canonical_map,
    compute_active_theme_centroids,
    dedupe_registry,
    find_duplicate_groups,
)
from themes.dynamic_theme.stages.step07_relationships import build_relationship_graph
from themes.dynamic_theme.stages.step08_memberships import compute_memberships
from themes.dynamic_theme.stages.step09_meta_features import build_meta_features

logger = logging.getLogger(__name__)
UTC = timezone.utc

if TYPE_CHECKING:
    from core.nervous_system.persistence.uow import UnitOfWork


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


def _represented_date(value: object, *, field_name: str) -> date:
    """Return the calendar date represented by the caller without zone shifting."""

    if isinstance(value, datetime):
        if pd.isna(value):
            raise ValueError(f"{field_name} must be a valid represented date")
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid represented date") from exc
    if pd.isna(parsed):
        raise ValueError(f"{field_name} must be a valid represented date")
    return parsed.date()


def _artifact_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage_for_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    table_name: str,
) -> tuple[LineageRef, ...]:
    """Attach exact post-write artifact hash and original row locators."""

    if not Path(path).exists():
        raise FileNotFoundError(f"published artifact does not exist: {path}")
    artifact_hash = _artifact_sha256(Path(path))
    if frame.empty:
        raise ValueError(f"cannot publish empty {table_name} artifact without row lineage")
    return tuple(
        LineageRef(
            source_id=str(path),
            content_hash=artifact_hash,
            record_locator=f"{table_name}:row:{index}",
        )
        for index in frame.index
    )


def _publish_completed_theme_outputs(
    memberships: pd.DataFrame,
    features: pd.DataFrame,
    *,
    unit_of_work: "UnitOfWork | None",
    valid_until_for: Callable[[datetime], datetime] | None,
    represented_as_of: object,
    feature_completion_at: datetime | None = None,
) -> None:
    if unit_of_work is None:
        return
    if valid_until_for is None:
        raise ValueError("valid_until_for is required when publishing nervous-system states")
    if not TICKER_MEMBERSHIP_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"membership history artifact is required for publication: {TICKER_MEMBERSHIP_HISTORY_PATH}"
        )
    history = pd.read_parquet(TICKER_MEMBERSHIP_HISTORY_PATH)
    if history.empty:
        raise ValueError("cannot publish theme states without membership history rows")
    required_history_columns = {
        "as_of",
        "available_at",
        "ticker",
        "theme",
        "taxonomy_version",
    }
    missing_history_columns = sorted(required_history_columns - set(history.columns))
    if missing_history_columns:
        raise ValueError(
            f"membership history missing publication columns: {missing_history_columns}"
        )
    run_date = _represented_date(represented_as_of, field_name="represented_as_of")
    history["as_of"] = history["as_of"].map(
        lambda value: _represented_date(value, field_name="history.as_of")
    )

    taxonomy_attr = memberships.attrs.get("taxonomy_version")
    if not pd.api.types.is_scalar(taxonomy_attr) or pd.isna(taxonomy_attr):
        raise ValueError(
            "current memberships attrs must contain one taxonomy_version"
        )
    taxonomy_version = str(taxonomy_attr).strip()
    if not taxonomy_version:
        raise ValueError(
            "current memberships attrs must contain one taxonomy_version"
        )
    if "taxonomy_version" in memberships.columns:
        current_taxonomies = {
            str(value).strip()
            for value in memberships["taxonomy_version"].dropna().tolist()
        }
        if current_taxonomies and current_taxonomies != {taxonomy_version}:
            raise ValueError(
                "current memberships contain ambiguous taxonomy_version evidence"
            )

    history = history[
        (history["as_of"] == run_date)
        & (history["taxonomy_version"].astype(str) == taxonomy_version)
    ].copy()
    if history.empty:
        raise ValueError(
            "no membership history evidence for exact represented_as_of "
            f"{run_date.isoformat()} and taxonomy_version {taxonomy_version!r}"
        )
    evidence_key = ["as_of", "ticker", "theme", "taxonomy_version"]
    if history.duplicated(evidence_key, keep=False).any():
        raise ValueError(
            "ambiguous membership history evidence for exact represented_as_of "
            "and taxonomy_version"
        )

    feature_taxonomy_attr: str | None = None
    if "taxonomy_version" in features.attrs:
        raw_feature_taxonomy = features.attrs["taxonomy_version"]
        if (
            not pd.api.types.is_scalar(raw_feature_taxonomy)
            or pd.isna(raw_feature_taxonomy)
            or not str(raw_feature_taxonomy).strip()
        ):
            raise ValueError("ambiguous feature taxonomy_version metadata")
        feature_taxonomy_attr = str(raw_feature_taxonomy).strip()
        if feature_taxonomy_attr != taxonomy_version:
            raise ValueError("conflicting feature taxonomy_version metadata")

    if features.empty:
        run_features = features.copy()
    else:
        required_feature_columns = {"ticker", "date", "taxonomy_version"}
        missing_feature_columns = sorted(
            required_feature_columns - set(features.columns)
        )
        if missing_feature_columns:
            raise ValueError(
                f"current theme features missing columns: {missing_feature_columns}"
            )
        if features["taxonomy_version"].isna().any():
            raise ValueError("current theme features contain missing taxonomy_version")
        feature_taxonomies = features["taxonomy_version"].map(
            lambda value: str(value).strip()
        )
        if (feature_taxonomies == "").any():
            raise ValueError("current theme features contain empty taxonomy_version")
        feature_dates = features["date"].map(
            lambda value: _represented_date(value, field_name="features.date")
        )
        run_features = features.loc[
            (feature_dates == run_date)
            & (feature_taxonomies == taxonomy_version)
        ].copy()
        if run_features.empty:
            raise ValueError(
                "no theme feature evidence for exact represented_as_of "
                f"{run_date.isoformat()} and taxonomy_version {taxonomy_version!r}"
            )
        feature_evidence_key = ["date", "ticker", "taxonomy_version"]
        if run_features.duplicated(feature_evidence_key, keep=False).any():
            raise ValueError(
                "ambiguous theme feature evidence for exact represented_as_of "
                "and taxonomy_version"
            )
    membership_available_at = max(
        _aware_utc(value, field_name="history.available_at")
        for value in history["available_at"].tolist()
    )
    feature_completion_utc = _aware_utc(
        feature_completion_at if feature_completion_at is not None else _utc_now(),
        field_name="feature_completion_at",
    )
    available_at = max(membership_available_at, feature_completion_utc)
    valid_until = valid_until_for(available_at)
    lineage = _lineage_for_frame(
        history,
        path=TICKER_MEMBERSHIP_HISTORY_PATH,
        table_name="ticker_theme_membership_history",
    )
    if not run_features.empty:
        lineage += _lineage_for_frame(
            run_features,
            path=TICKER_THEME_FEATURES_PATH,
            table_name="ticker_theme_features",
        )
    persist_theme_states(
        history,
        run_features,
        unit_of_work=unit_of_work,
        available_at=available_at,
        valid_until=valid_until,
        taxonomy_version=taxonomy_version,
        lineage=lineage,
    )


def _get_tickers() -> list[str]:
    """Load the shared universe ticker list."""
    try:
        from core.shared_universe.universe import shared_tickers
        tickers = shared_tickers(eligible_only=True)
        if tickers:
            return tickers
    except Exception as exc:
        logger.warning("shared_universe not available: %s", exc)

    # Fallback: read from legacy theme_map if shared_universe is unavailable
    try:
        from themes.dynamic_theme.config import REPO_ROOT
        import pandas as pd
        legacy = REPO_ROOT / "themes" / "theme_expansion_legacy" / "data" / "theme_map_v4.csv"
        if legacy.exists():
            df = pd.read_csv(legacy)
            return df["ticker"].astype(str).str.upper().unique().tolist()
    except Exception:
        pass

    raise RuntimeError(
        "No ticker universe available. Ensure shared_universe is populated "
        "or theme_expansion_legacy/data/theme_map_v4.csv exists."
    )


def daily_run(
    *,
    as_of: pd.Timestamp | None = None,
    tickers: list[str] | None = None,
    unit_of_work: "UnitOfWork | None" = None,
    valid_until_for: Callable[[datetime], datetime] | None = None,
) -> None:
    """Run the daily theme feature refresh (no reclustering)."""
    ensure_outputs()
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    tickers = tickers or _get_tickers()

    logger.info("=== Dynamic Theme Daily Run [%s] ===", as_of.date())

    docs = build_ticker_documents(tickers, as_of=as_of)
    embeddings = generate_embeddings(docs, as_of=as_of)
    memberships = compute_memberships(embeddings_df=embeddings, as_of=as_of)
    features = build_meta_features(memberships_df=memberships, as_of=as_of)
    feature_completion_at = _utc_now()
    _publish_completed_theme_outputs(
        memberships,
        features,
        unit_of_work=unit_of_work,
        valid_until_for=valid_until_for,
        represented_as_of=as_of,
        feature_completion_at=feature_completion_at,
    )

    logger.info("=== Daily run complete ===")


def weekly_run(
    *,
    as_of: pd.Timestamp | None = None,
    tickers: list[str] | None = None,
    unit_of_work: "UnitOfWork | None" = None,
    valid_until_for: Callable[[datetime], datetime] | None = None,
) -> None:
    """Run the full weekly theme taxonomy refresh (recluster + Claude labeling)."""
    ensure_outputs()
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    tickers = tickers or _get_tickers()

    logger.info("=== Dynamic Theme Weekly Run [%s] — %d tickers ===", as_of.date(), len(tickers))

    # Step 1 + 2: documents + embeddings (pass as_of for price co-movement lookback)
    docs = build_ticker_documents(tickers, as_of=as_of)
    embeddings = generate_embeddings(docs, as_of=as_of)

    # Step 3: cluster
    clusters = cluster_tickers(embeddings_df=embeddings)

    # Step 4: summaries
    summaries = build_cluster_summaries(clusters, docs_df=docs)

    if not summaries:
        logger.warning("No clusters found — skipping labeling and relationship graph")
        return

    # Step 5: Claude labeling
    registry = label_clusters(summaries, as_of=as_of)

    # Step 6: discover new themes (labels incremental new clusters not in prior week)
    _new_count, registry = discover_new_themes(
        current_clusters_df=clusters,
        current_embeddings_df=embeddings,
        cluster_summaries=summaries,
        as_of=as_of,
    )

    # Step 6b: dedupe near-duplicate themes (e.g. HDBSCAN splitting one sector
    # into several small clusters that each get a slightly different name).
    # Persisted to disk so next week's stability matching (step05) sees the
    # canonical names, not the ones it's collapsing away.
    theme_to_centroid, theme_to_size = compute_active_theme_centroids(registry, embeddings, clusters)
    groups = find_duplicate_groups(theme_to_centroid)
    canonical_map = build_canonical_map(groups, theme_to_size)
    if canonical_map:
        registry = dedupe_registry(registry, canonical_map)
        registry.to_parquet(THEME_REGISTRY_PATH, index=False)
        logger.info("Deduped %d theme(s) into %d canonical group(s)", len(canonical_map), len(groups))

    # Step 7: relationship graph
    build_relationship_graph(registry_df=registry, as_of=as_of)

    # Step 8: memberships
    memberships = compute_memberships(
        embeddings_df=embeddings,
        clusters_df=clusters,
        registry_df=registry,
        as_of=as_of,
    )

    # Step 9: meta features
    features = build_meta_features(memberships_df=memberships, as_of=as_of)
    feature_completion_at = _utc_now()
    _publish_completed_theme_outputs(
        memberships,
        features,
        unit_of_work=unit_of_work,
        valid_until_for=valid_until_for,
        represented_as_of=as_of,
        feature_completion_at=feature_completion_at,
    )

    logger.info("=== Weekly run complete ===")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Dynamic Theme Taxonomy Pipeline")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="weekly")
    parser.add_argument("--as-of", default=None, help="Date override YYYY-MM-DD")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of, tz="UTC") if args.as_of else None
    if args.mode == "weekly":
        weekly_run(as_of=as_of)
    else:
        daily_run(as_of=as_of)
