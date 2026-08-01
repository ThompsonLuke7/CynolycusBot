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
from datetime import datetime, timezone
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
    history["as_of"] = pd.to_datetime(history["as_of"], errors="raise").dt.date
    latest_date = history["as_of"].max()
    taxonomy_version = str(
        memberships.attrs.get(
            "taxonomy_version",
            history.loc[history["as_of"] == latest_date, "taxonomy_version"].iloc[-1],
        )
    )
    history = history[
        (history["as_of"] == latest_date)
        & (history["taxonomy_version"].astype(str) == taxonomy_version)
    ].copy()
    if history.empty:
        raise ValueError("selected taxonomy has no current membership history rows")

    if features.empty:
        latest_features = features.copy()
    else:
        feature_dates = pd.to_datetime(features["date"], errors="raise").dt.date
        latest_features = features.loc[feature_dates == latest_date].copy()
    available_at = (
        pd.to_datetime(history["available_at"], utc=True, errors="raise")
        .max()
        .to_pydatetime()
    )
    valid_until = valid_until_for(available_at)
    lineage = _lineage_for_frame(
        history,
        path=TICKER_MEMBERSHIP_HISTORY_PATH,
        table_name="ticker_theme_membership_history",
    ) + _lineage_for_frame(
        latest_features,
        path=TICKER_THEME_FEATURES_PATH,
        table_name="ticker_theme_features",
    ) if not latest_features.empty else _lineage_for_frame(
        history,
        path=TICKER_MEMBERSHIP_HISTORY_PATH,
        table_name="ticker_theme_membership_history",
    )
    persist_theme_states(
        history,
        latest_features,
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
    _publish_completed_theme_outputs(
        memberships,
        features,
        unit_of_work=unit_of_work,
        valid_until_for=valid_until_for,
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
    _publish_completed_theme_outputs(
        memberships,
        features,
        unit_of_work=unit_of_work,
        valid_until_for=valid_until_for,
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
