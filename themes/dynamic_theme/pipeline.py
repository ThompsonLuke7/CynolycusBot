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
  Step 7  - Update theme relationship graph
  Step 8  - Recompute memberships
  Step 9  - Generate meta features
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import pandas as pd

from themes.dynamic_theme.config import ensure_outputs
from themes.dynamic_theme.stages.step01_build_documents import build_ticker_documents
from themes.dynamic_theme.stages.step02_embed import generate_embeddings
from themes.dynamic_theme.stages.step03_cluster import cluster_tickers
from themes.dynamic_theme.stages.step04_cluster_summary import build_cluster_summaries
from themes.dynamic_theme.stages.step05_claude_labeling import label_clusters
from themes.dynamic_theme.stages.step06_discovery import discover_new_themes
from themes.dynamic_theme.stages.step07_relationships import build_relationship_graph
from themes.dynamic_theme.stages.step08_memberships import compute_memberships
from themes.dynamic_theme.stages.step09_meta_features import build_meta_features

logger = logging.getLogger(__name__)


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
) -> None:
    """Run the daily theme feature refresh (no reclustering)."""
    ensure_outputs()
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    tickers = tickers or _get_tickers()

    logger.info("=== Dynamic Theme Daily Run [%s] ===", as_of.date())

    docs = build_ticker_documents(tickers, as_of=as_of)
    embeddings = generate_embeddings(docs)
    memberships = compute_memberships(embeddings_df=embeddings, as_of=as_of)
    build_meta_features(memberships_df=memberships, as_of=as_of)

    logger.info("=== Daily run complete ===")


def weekly_run(
    *,
    as_of: pd.Timestamp | None = None,
    tickers: list[str] | None = None,
) -> None:
    """Run the full weekly theme taxonomy refresh (recluster + Claude labeling)."""
    ensure_outputs()
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    tickers = tickers or _get_tickers()

    logger.info("=== Dynamic Theme Weekly Run [%s] — %d tickers ===", as_of.date(), len(tickers))

    # Step 1 + 2: documents + embeddings
    docs = build_ticker_documents(tickers, as_of=as_of)
    embeddings = generate_embeddings(docs)

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
    build_meta_features(memberships_df=memberships, as_of=as_of)

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
