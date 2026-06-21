"""Backfill daily theme membership + meta features for the Meta Ranker training window.

Uses the CURRENT cluster structure (today's centroids + registry) and re-computes
membership scores + theme features for each historical trading date. This gives the
Meta Ranker training matrix non-NaN dynamic theme features going back to 2022.

This is a deliberate approximation: we use today's discovered themes to label
historical dates, rather than re-running the full weekly pipeline for every week
in history. In practice this is fine for v1 because:
  - The broad cluster structure (semiconductors, miners, biotech, etc.) is stable
  - The Meta Ranker learns from the features, not the cluster IDs themselves
  - True historical re-clustering would require re-labeling ~200 weeks with Claude

Run once after the weekly pipeline has produced a stable registry:
    python -m themes.dynamic_theme.backfill_theme_features

Outputs: themes/dynamic_theme/outputs/ticker_theme_features_history.parquet
         (appended daily to avoid re-computing already-done dates)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from themes.dynamic_theme.config import (
    DAILY_BARS_DIR,
    OUTPUTS_DIR,
    TICKER_EMBEDDINGS_PATH,
    TICKER_CLUSTERS_PATH,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step02_embed import _build_comovement_vectors, _COMOVEMENT_PCA_COMPONENTS, _COMOVEMENT_WEIGHT
from themes.dynamic_theme.stages.step08_memberships import compute_memberships, _cosine_sim_matrix
from themes.dynamic_theme.stages.step09_meta_features import build_meta_features
from themes.dynamic_theme.stages.step05_claude_labeling import load_registry
from themes.dynamic_theme.stages.step03_cluster import compute_centroids

logger = logging.getLogger(__name__)

HISTORY_PATH = OUTPUTS_DIR / "ticker_theme_features_history.parquet"

# Training window: match momentum OOF start date
BACKFILL_START = pd.Timestamp("2022-11-01")
# Don't backfill today — the main pipeline handles current date
BACKFILL_END = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def _trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return business days in [start, end] that have at least one bar file."""
    idx = pd.bdate_range(start, end, freq="B")
    # Verify against a liquid ticker
    ref = pd.read_parquet(DAILY_BARS_DIR / "SPY.parquet", columns=["timestamp"])
    ref["timestamp"] = pd.to_datetime(ref["timestamp"]).dt.tz_localize(None).dt.normalize()
    valid = set(ref["timestamp"].tolist())
    return [d for d in idx if d in valid]


def _already_done_dates() -> set[pd.Timestamp]:
    if not HISTORY_PATH.exists():
        return set()
    try:
        df = pd.read_parquet(HISTORY_PATH, columns=["date"])
        return set(pd.to_datetime(df["date"]).dt.normalize().unique())
    except Exception:
        return set()


def run_backfill(
    start: pd.Timestamp = BACKFILL_START,
    end: pd.Timestamp = BACKFILL_END,
    skip_existing: bool = True,
) -> None:
    ensure_outputs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load current cluster structure (stable proxy for all history)
    registry_df = load_registry(latest_only=True)
    if registry_df.empty:
        raise RuntimeError("No theme registry found. Run weekly pipeline first.")

    clusters_df = pd.read_parquet(TICKER_CLUSTERS_PATH)

    # Load today's text embeddings (same matrix used for centroid computation)
    text_emb_df = pd.read_parquet(TICKER_EMBEDDINGS_PATH)
    tickers = text_emb_df["ticker"].astype(str).tolist()
    text_matrix = np.array(text_emb_df["embedding"].tolist(), dtype=np.float32)
    text_dim = text_matrix.shape[1] - _COMOVEMENT_PCA_COMPONENTS  # strip behavior suffix
    # Separate text vs behavior portions of the stored blended embedding
    text_only = text_matrix[:, :text_dim]

    # Centroids from the blended (text+behavior) embeddings in the stored matrix
    centroids_by_id = compute_centroids(text_matrix, tickers, clusters_df)
    id_to_theme = dict(zip(registry_df["cluster_id"], registry_df["theme_name"]))
    valid_ids = sorted(cid for cid in centroids_by_id if cid in id_to_theme)
    if not valid_ids:
        raise RuntimeError("No cluster centroids match registry. Re-run weekly pipeline.")

    theme_names = [id_to_theme[cid] for cid in valid_ids]
    centroid_matrix = np.array([centroids_by_id[cid] for cid in valid_ids], dtype=np.float32)
    # Centroid dim must match embedding dim — pad centroids if needed
    if centroid_matrix.shape[1] != text_matrix.shape[1]:
        logger.warning("Centroid dim %d != embedding dim %d — recomputing from stored embeddings",
                       centroid_matrix.shape[1], text_matrix.shape[1])

    trading_days = _trading_days_between(start, end)
    done = _already_done_dates() if skip_existing else set()
    todo = [d for d in trading_days if d not in done]
    logger.info("Backfill: %d trading days to process (%d already done)", len(todo), len(done))

    appended_chunks: list[pd.DataFrame] = []

    # Co-movement vectors come from a 60-day rolling return window, so they
    # barely move day-to-day. Recompute them once per ISO week and reuse within
    # the week — a ~5x speedup with negligible effect on the (already approximate)
    # historical membership scores.
    _behavior_cache: dict = {}
    _behavior_week = None

    for i, day in enumerate(todo):
        if i % 50 == 0:
            logger.info("  [%d/%d] %s", i, len(todo), day.date())

        # Build behavior vectors as-of this historical date (no lookahead),
        # cached per ISO week.
        iso = day.isocalendar()
        wk = (iso[0], iso[1])
        if wk != _behavior_week:
            _behavior_cache = _build_comovement_vectors(tickers, as_of=day)
            _behavior_week = wk
        behavior_by_ticker = _behavior_cache

        # Rebuild blended embeddings for this date
        behavior_dim = _COMOVEMENT_PCA_COMPONENTS
        blended_vecs = []
        for j, ticker in enumerate(tickers):
            tvec = text_only[j]
            if ticker in behavior_by_ticker:
                bvec = behavior_by_ticker[ticker]
                if len(bvec) < behavior_dim:
                    bvec = np.pad(bvec, (0, behavior_dim - len(bvec)))
                combined = np.concatenate([
                    (1.0 - _COMOVEMENT_WEIGHT) * tvec,
                    _COMOVEMENT_WEIGHT * bvec,
                ]).astype(np.float32)
            else:
                combined = np.concatenate([tvec, np.zeros(behavior_dim, dtype=np.float32)]).astype(np.float32)
            norm = np.linalg.norm(combined)
            blended_vecs.append(combined / norm if norm > 0 else combined)

        blended_matrix = np.array(blended_vecs, dtype=np.float32)

        # Recompute centroids with this date's blended matrix
        day_centroids_by_id = compute_centroids(blended_matrix, tickers, clusters_df)
        day_centroid_matrix = np.array(
            [day_centroids_by_id.get(cid, centroid_matrix[k]) for k, cid in enumerate(valid_ids)],
            dtype=np.float32,
        )

        # Membership scores: cosine sim of each ticker to each theme centroid
        sim_matrix = _cosine_sim_matrix(blended_matrix, day_centroid_matrix)

        # Build embeddings_df stub so compute_memberships can use it
        emb_df_day = pd.DataFrame({
            "ticker": tickers,
            "embedding": [blended_vecs[j].tolist() for j in range(len(tickers))],
            "date": day,
        })

        memberships = compute_memberships(
            embeddings_df=emb_df_day,
            clusters_df=clusters_df,
            registry_df=registry_df,
            as_of=day,
        )

        if memberships.empty:
            continue

        # Generate meta features for this day (returns df with date=day)
        features = build_meta_features(memberships_df=memberships, as_of=day)
        if features is not None and not features.empty:
            appended_chunks.append(features)

        # Write incrementally every 20 days to avoid losing progress
        if len(appended_chunks) >= 20:
            _flush(appended_chunks)
            appended_chunks = []

    if appended_chunks:
        _flush(appended_chunks)

    if HISTORY_PATH.exists():
        total = len(pd.read_parquet(HISTORY_PATH, columns=["date"])["date"].unique())
        logger.info("Backfill complete. History file: %d unique dates", total)


def _flush(chunks: list[pd.DataFrame], *, keep_existing: bool = True) -> None:
    new = pd.concat(chunks, ignore_index=True)
    if keep_existing and HISTORY_PATH.exists():
        existing = pd.read_parquet(HISTORY_PATH)
        combined = pd.concat([existing, new], ignore_index=True)
        # keep='last' so re-computed rows overwrite stale ones
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    else:
        combined = new
    combined.to_parquet(HISTORY_PATH, index=False)
    logger.info("Flushed %d rows → %s (total %d)", len(new), HISTORY_PATH, len(combined))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill dynamic theme features for Meta Ranker training")
    parser.add_argument("--start", default=str(BACKFILL_START.date()), help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=str(BACKFILL_END.date()), help="End date YYYY-MM-DD")
    parser.add_argument("--no-skip", action="store_true", help="Recompute already-done dates")
    args = parser.parse_args()
    run_backfill(
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        skip_existing=not args.no_skip,
    )
