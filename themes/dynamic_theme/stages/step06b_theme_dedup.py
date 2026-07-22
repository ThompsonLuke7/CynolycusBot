"""Step 6b — Theme Deduplication.

HDBSCAN's fine-grained leaf clustering (min_cluster_size=5) regularly splits
one economic sector into several small clusters, and step05's Claude labeling
— even with the existing-theme-name list added to its prompt — can still mint
near-duplicate names for them (e.g. "regional_bank_holding_companies",
"community_regional_banks", "small_cap_regional_banks" all describing the
same regional-bank complex). This step finds those near-duplicates among the
*currently active* theme set and collapses each group onto one canonical name.

Merge rule (conservative, two independent signals must both agree):
  1. Raw embedding centroid cosine similarity >= MIN_COSINE
  2. Stemmed snake_case token Jaccard overlap >= MIN_TOKEN_JACCARD
A single signal alone is unreliable — cosine similarity alone catches
economically distinct themes that just happen to share a sparse embedding
neighborhood (e.g. "cloud_cybersecurity" vs "cloud_data_platforms"); token
overlap alone catches unrelated themes that happen to share a generic word.
Requiring both keeps the merge list to only the same actual thing.

Canonical name per group = the member with the most constituent tickers this
run (ties broken alphabetically), so the surviving name is the most
representative one, not an arbitrary pick.

Runs between step06 (discovery) and step07 (relationships) in the weekly
pipeline, operating on the registry only (relationships/memberships/features
are computed after, so they see the deduped set directly — no cleanup logic
needed on their side for a normal weekly run). `remediate_existing_outputs`
below is for patching artifacts that were already written *before* this step
existed in the pipeline.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_COSINE = 0.94
MIN_TOKEN_JACCARD = 0.34


def _stem(tok: str) -> str:
    for suf, rep in [("ies", "y"), ("ing", ""), ("es", ""), ("s", "")]:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + rep
    return tok


def _tokens(name: str) -> set[str]:
    return {_stem(t) for t in name.split("_") if len(t) > 2}


def compute_active_theme_centroids(
    registry_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """(theme_name -> centroid, theme_name -> member ticker count) for the latest registry date.

    Seed themes (config-driven, no cluster_id in this run's clusters_df) are
    excluded — they're hand-pinned and never candidates for merging away.
    """
    from themes.dynamic_theme.stages.step03_cluster import compute_centroids

    tickers = embeddings_df["ticker"].astype(str).tolist()
    matrix = np.array(embeddings_df["embedding"].tolist(), dtype=np.float32)
    centroids_by_id = compute_centroids(matrix, tickers, clusters_df)

    latest_date = registry_df["date"].max()
    latest = registry_df[registry_df["date"] == latest_date]
    id_to_theme = dict(zip(latest["cluster_id"], latest["theme_name"]))

    sizes = clusters_df["cluster_id"].value_counts().to_dict()
    theme_to_centroid = {
        name: centroids_by_id[cid] for cid, name in id_to_theme.items() if cid in centroids_by_id
    }
    theme_to_size = {
        name: int(sizes.get(cid, 0)) for cid, name in id_to_theme.items() if cid in centroids_by_id
    }
    return theme_to_centroid, theme_to_size


def find_duplicate_groups(
    theme_to_centroid: dict[str, np.ndarray],
    *,
    min_cosine: float = MIN_COSINE,
    min_token_jaccard: float = MIN_TOKEN_JACCARD,
) -> list[list[str]]:
    """Connected components of themes that are both embedding-close and name-similar."""
    names = sorted(theme_to_centroid.keys())
    if len(names) < 2:
        return []
    X = np.array([theme_to_centroid[n] for n in names])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    sim = Xn @ Xn.T
    name_tokens = {n: _tokens(n) for n in names}

    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s < min_cosine:
                continue
            ta, tb = name_tokens[names[i]], name_tokens[names[j]]
            union_tok = ta | tb
            jacc = len(ta & tb) / len(union_tok) if union_tok else 0.0
            if jacc >= min_token_jaccard:
                union(names[i], names[j])

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def build_canonical_map(
    groups: list[list[str]],
    theme_to_size: dict[str, int],
) -> dict[str, str]:
    """{old_name: canonical_name} for every non-canonical member of every group."""
    canonical_map: dict[str, str] = {}
    for group in groups:
        # most member tickers wins; ties broken alphabetically for determinism
        canonical = sorted(group, key=lambda n: (-theme_to_size.get(n, 0), n))[0]
        for member in group:
            if member != canonical:
                canonical_map[member] = canonical
        logger.info(
            "Theme dedup: merging %s -> %r (sizes: %s)",
            [m for m in group if m != canonical], canonical,
            {m: theme_to_size.get(m, 0) for m in group},
        )
    return canonical_map


def dedupe_registry(registry_df: pd.DataFrame, canonical_map: dict[str, str]) -> pd.DataFrame:
    """Rename theme_name -> canonical for the latest date's rows only."""
    if not canonical_map:
        return registry_df
    out = registry_df.copy()
    latest_date = out["date"].max()
    mask = out["date"] == latest_date
    out.loc[mask, "theme_name"] = out.loc[mask, "theme_name"].replace(canonical_map)
    return out


def _backup(path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = path.with_suffix(f".pre_dedup_{ts}.parquet")
        shutil.copy2(path, backup_path)
        logger.info("Backed up %s -> %s", path, backup_path)


def remediate_existing_outputs(canonical_map: dict[str, str], *, as_of: pd.Timestamp) -> None:
    """One-time patch for artifacts already written before this step existed.

    Renames theme names to canonical in registry/membership/relationships
    (latest date only), then re-runs step08 (memberships already correct —
    just collapsed) and step09 (meta features) so every derived file is
    consistent with the deduped taxonomy. Every mutated file is backed up
    first (outputs/ is gitignored — there is no other recovery path).
    """
    from themes.dynamic_theme.config import (
        THEME_REGISTRY_PATH,
        THEME_RELATIONSHIPS_PATH,
        TICKER_MEMBERSHIP_PATH,
    )
    from themes.dynamic_theme.stages.step09_meta_features import build_meta_features

    if not canonical_map:
        logger.info("No duplicate groups found — nothing to remediate")
        return

    # --- registry ---
    _backup(THEME_REGISTRY_PATH)
    registry = pd.read_parquet(THEME_REGISTRY_PATH)
    registry = dedupe_registry(registry, canonical_map)
    registry.to_parquet(THEME_REGISTRY_PATH, index=False)
    logger.info("Patched %s", THEME_REGISTRY_PATH)

    # --- membership: remap + collapse duplicate (ticker, theme) rows ---
    _backup(TICKER_MEMBERSHIP_PATH)
    mem = pd.read_parquet(TICKER_MEMBERSHIP_PATH)
    latest_mem_date = mem["date"].max()
    mask = mem["date"] == latest_mem_date
    mem.loc[mask, "theme"] = mem.loc[mask, "theme"].replace(canonical_map)
    today = mem[mask].groupby(["ticker", "theme", "date"], as_index=False)["membership_score"].max()
    mem = pd.concat([mem[~mask], today], ignore_index=True)
    mem.to_parquet(TICKER_MEMBERSHIP_PATH, index=False)
    logger.info("Patched %s  rows=%d", TICKER_MEMBERSHIP_PATH, len(mem))

    # --- relationships: remap, drop self-loops, dedupe pairs ---
    _backup(THEME_RELATIONSHIPS_PATH)
    if THEME_RELATIONSHIPS_PATH.exists():
        rel = pd.read_parquet(THEME_RELATIONSHIPS_PATH)
        latest_rel_date = rel["date"].max()
        rmask = rel["date"] == latest_rel_date
        rel.loc[rmask, "source"] = rel.loc[rmask, "source"].replace(canonical_map)
        rel.loc[rmask, "target"] = rel.loc[rmask, "target"].replace(canonical_map)
        today_rel = rel[rmask]
        today_rel = today_rel[today_rel["source"] != today_rel["target"]]
        pair_key = today_rel.apply(lambda r: tuple(sorted((r["source"], r["target"]))), axis=1)
        today_rel = today_rel.loc[
            today_rel.assign(_strength=today_rel["strength"]).groupby(pair_key)["_strength"].idxmax()
        ]
        rel = pd.concat([rel[~rmask], today_rel], ignore_index=True)
        rel.to_parquet(THEME_RELATIONSHIPS_PATH, index=False)
        logger.info("Patched %s  rows=%d", THEME_RELATIONSHIPS_PATH, len(rel))

    # --- derived features: recompute from the corrected membership, don't hand-patch ---
    from themes.dynamic_theme.config import TICKER_THEME_FEATURES_PATH
    _backup(TICKER_THEME_FEATURES_PATH)
    build_meta_features(as_of=as_of)
    logger.info("Rebuilt %s from deduped membership + relationships", TICKER_THEME_FEATURES_PATH)
