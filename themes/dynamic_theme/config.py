"""Central configuration for the dynamic_theme module."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Source data ──────────────────────────────────────────────────────────────
# Raw daily bars per ticker (shared Alpaca cache)
DAILY_BARS_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "1d"

# News records from the news module
NEWS_RECORDS_PATH = REPO_ROOT / "signals" / "news" / "data" / "processed" / "news_records.parquet"

# Yahoo Finance ticker profiles (from news module)
TICKER_PROFILES_PATH = REPO_ROOT / "signals" / "news" / "data" / "processed" / "ticker_profiles.parquet"
# Theme-owned profiles for universe names the shared table lacks. Kept separate
# because the live news-catalyst scorer reads the shared table as model input.
TICKER_PROFILES_SUPPLEMENT_PATH = (
    REPO_ROOT / "signals" / "news" / "data" / "processed" / "ticker_profiles_theme_supplement.parquet"
)
PROFILE_RETRY_DAYS = 30            # re-request a ticker Yahoo returned nothing for
WEEKLY_PROFILE_FETCH_CAP = 300     # new-name top-up per weekly run (~0.7s each)

# Per-ticker JSON profile cache (from legacy theme_expansion)
TICKER_PROFILES_CACHE_DIR = REPO_ROOT / "themes" / "theme_expansion_legacy" / "data" / "ticker_profiles_cache"

# Forward guidance / events
EVENTS_FG_PATH = REPO_ROOT / "signals" / "events" / "forward_guidance" / "data" / "processed"

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUTS_DIR = REPO_ROOT / "themes" / "dynamic_theme" / "outputs"

TICKER_DOCUMENTS_PATH   = OUTPUTS_DIR / "ticker_documents.parquet"
TICKER_EMBEDDINGS_PATH  = OUTPUTS_DIR / "ticker_embeddings.parquet"
TICKER_CLUSTERS_PATH    = OUTPUTS_DIR / "ticker_clusters.parquet"
THEME_REGISTRY_PATH     = OUTPUTS_DIR / "theme_registry.parquet"
THEME_RELATIONSHIPS_PATH = OUTPUTS_DIR / "theme_relationships.parquet"
TICKER_MEMBERSHIP_PATH  = OUTPUTS_DIR / "ticker_theme_membership.parquet"
TICKER_MEMBERSHIP_HISTORY_PATH = OUTPUTS_DIR / "ticker_theme_membership_history.parquet"
# Descriptive alias for callers that do not need the legacy ticker-prefixed name.
MEMBERSHIP_HISTORY_PATH = TICKER_MEMBERSHIP_HISTORY_PATH
membership_history_path = TICKER_MEMBERSHIP_HISTORY_PATH
TICKER_THEME_FEATURES_PATH = OUTPUTS_DIR / "ticker_theme_features.parquet"
PENDING_THEME_CANDIDATES_PATH = OUTPUTS_DIR / "pending_theme_candidates.parquet"
PENDING_THEME_REGISTRY_PATH = OUTPUTS_DIR / "pending_theme_registry.parquet"
PENDING_THEME_MEMBERSHIP_PATH = OUTPUTS_DIR / "pending_theme_membership.parquet"
PENDING_THEME_HISTORY_PATH = OUTPUTS_DIR / "pending_theme_history.parquet"
PENDING_THEME_NEWS_PATH = OUTPUTS_DIR / "pending_theme_news.parquet"
PENDING_THEME_PROFILES_PATH = OUTPUTS_DIR / "pending_theme_profiles.parquet"

# ── Embedding model ───────────────────────────────────────────────────────────
BGE_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small-en-v1.5

# ── Text-vector composition ───────────────────────────────────────────────────
# The text portion of a ticker's embedding anchors on its Yahoo business
# description ("what the company IS") and blends in a cleaned news mean ("what
# it's DOING"). Without the description anchor, a mega-cap whose feed is full of
# "AAPL is now Fund X's 9th largest position" 13F spam gets embedded as
# institutional-ownership news and lands in a junk holdings cluster instead of
# consumer tech. DESC_BLEND_WEIGHT is the weight on the description vector.
DESC_BLEND_WEIGHT = 0.6

# Headlines/bodies matching these (case-insensitive) are ownership-filing /
# 13F / analyst-coverage spam or HTML scaffolding — dropped before averaging a
# ticker's article embeddings (unless dropping would leave it with no news).
NEWS_SPAM_PATTERNS = [
    r"largest position",
    r"\b13[FDG]\b",
    r"\bSC 13[DG]\b",
    r"\bstake in\b",
    r"holdings in\b",
    r"\bshares (?:of|in)\b.*\b(?:bought|sold|acquired|purchased)\b",
    r"\b(?:boosts|trims|lowers|raises|cuts|reduces|increases|decreases)\b.*\b(?:holdings|stake|position|shares)\b",
    r"\b(?:buys|sells|acquires|purchases)\b.*\bshares\b",
    r"market ?beat",
    r"hedge fund",
    r"institutional (?:investor|ownership|holdings)",
    r"\bhas \$[\d.,]+ (?:million|billion) (?:stock )?(?:holdings|position)",
]

# ── Clustering ───────────────────────────────────────────────────────────────
# UMAP reduces 384-dim embeddings to a low-dim space where HDBSCAN can find
# fine-grained density structure. Without it, HDBSCAN collapses to 2-3 blobs.
UMAP_N_COMPONENTS = 20
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0           # tight packing — best for clustering
UMAP_METRIC = "cosine"        # BGE embeddings are cosine-similarity aligned

# min_cluster_size=2 gives maximum granularity (every pair can be a theme).
# cluster_selection_method='leaf' keeps fine-grained leaf clusters instead of
# merging them upward — produces many more, smaller themes.
# Soft membership (step08) means every ticker scores against every cluster
# centroid, so hard assignment precision matters less than having rich structure.
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_METRIC = "euclidean"  # UMAP output is Euclidean
HDBSCAN_CLUSTER_SELECTION_METHOD = "leaf"

# ── Discovery engine ─────────────────────────────────────────────────────────
# Centroid cosine similarity below this → new theme detected
NEW_THEME_SIMILARITY_THRESHOLD = 0.75

# ── Label stability ───────────────────────────────────────────────────────────
# HDBSCAN cluster ids are NOT stable week to week, so the labeler used to call
# Claude on EVERY cluster every week — re-naming themes that were already correct.
# Instead, match each current cluster to the prior week's theme centroids; if the
# best match's cosine similarity clears this threshold, CARRY FORWARD the prior
# label (no Claude call, no rename). Only clusters with no strong prior match
# (genuinely new, or grown from sub-threshold/noise into a real cluster) get
# (re)labeled by Claude. Set high so only near-identical clusters are reused.
LABEL_STABILITY_THRESHOLD = 0.90

# ── Seeded / anchor themes ────────────────────────────────────────────────────
# Unsupervised HDBSCAN gives no coverage guarantee: a real, coherent group can
# fail to form its own cluster and scatter into neighbours (e.g. the whole
# memory/storage complex landed in semiconductor_capital_equipment / batteries /
# copper_mining). Seed themes are HAND-PINNED: their anchor centroid is the mean
# embedding of the anchor tickers, injected alongside the emergent centroids in
# step08 so the theme ALWAYS exists and any similar name (incl. the anchors) maps
# to it. They survive the weekly recluster because they are config-driven, not
# emergent, and are never sent to Claude for (re)labeling.
#
# Reserved cluster-id range so seeds never collide with HDBSCAN ids (>= 0; -1 is
# HDBSCAN noise). Seed i gets SEED_CLUSTER_ID_BASE - i.
SEED_CLUSTER_ID_BASE = -1000

SEED_THEMES = [
    {
        "theme_name": "memory_storage",
        "parent_theme": "semiconductors",
        "description": (
            "Memory and data-storage hardware makers: DRAM, NAND flash, HBM, "
            "SSDs and hard disk drives (distinct from chip-making capital "
            "equipment and from energy/battery storage)."
        ),
        # Pure-play memory/storage hardware names define the anchor; other names
        # (NTAP, PSTG, SMCI, etc.) are then ASSIGNED to it by similarity.
        "anchor_tickers": ["MU", "WDC", "STX", "SNDK"],
        "related_themes": ["semiconductor_capital_equipment", "ai_infrastructure", "data_centers"],
    },
    {
        # theme_map_v4 gives AAPL no business theme, only this bucket, and nothing
        # embeds near it (best centroid 0.79 vs a 0.90 median). A similarity seed
        # cannot hold several mega caps -- size is not a business likeness (dry-run
        # 2026-10-03: only one name landed in it) -- so it anchors on AAPL alone;
        # MSFT/ORCL/NVDA are placed with their industry peers instead.
        "theme_name": "mega_cap_platforms",
        "parent_theme": "technology",
        "description": (
            "Diversified mega-cap platform companies with no single business-line "
            "peer group (hand-pinned from theme_map_v4)."
        ),
        "anchor_tickers": ["AAPL"],
        "related_themes": [],
    },
]

# ── Business-type assignment ──────────────────────────────────────────────────
# A cluster the labeler could not confidently name as one line of business
# ("heterogeneous cluster ... no single sector theme") is a grab-bag, not a
# theme. Below this label confidence a cluster is not published as a theme.
# 2026-09-28: 46 of 193 clusters / 396 tickers sit under 0.50 (0.18-0.47, e.g.
# mixed_small_cap_value, mid_cap_mixed_services); at 0.52 the clusters are real
# business groups (light metals, EMS, fintech, gaming).
MIN_CLUSTER_LABEL_CONFIDENCE = 0.50
UNCLASSIFIED_THEME = "unclassified"
# A ticker with no confident cluster of its own (HDBSCAN noise, or a grab-bag
# member) is placed in its best-matching theme AMONG those that already hold
# clustered companies from its own Yahoo industry -- at least this many, and at
# least this share of that industry's clustered names. No such theme ->
# UNCLASSIFIED_THEME. Nearest-centroid alone is not trusted for these names.
# Thresholds chosen 2026-10-04 against theme_map_v4 (435 unclustered names on the
# hand-curated map; score = share of the assigned theme's clustered members that
# share a curated theme with the ticker): nearest-centroid 0.32 (0.36 on the
# same tickers); this rule at (3, 0.20) 0.54, level with confidently clustered
# names in their own theme (0.53), placing 74% of unclustered names. (2, 0.10)
# scored 0.47; (3, 0.50) 0.60 but placed only 51%.
MIN_INDUSTRY_PEERS_IN_THEME = 3
MIN_INDUSTRY_PEER_SHARE = 0.20
# Two refinements, both measured the same way on the 2026-10-05 clusters
# (agreement over all placed names 0.52 -> 0.58, unclassified 227 -> 348):
#  * A clustered ticker that is the ONLY company of its industry in its theme is
#    treated as unclustered: such singletons agreed with the curated map 0.15 in
#    their own cluster (0.63 for other members) and 0.40 once re-placed.
#    Industries with two members in a theme showed no gain (0.32 vs 0.33) and stay.
#  * A home-market ticker is placed only with home-market industry peers, so US
#    names are not pulled into country baskets (GOOGL/META/AMZN had landed in
#    chinese_internet_stocks: 7 of the 12 clustered names in their industry are
#    Chinese ADRs). Foreign tickers may join any industry peers.
HOME_COUNTRY = "United States"
# Hand placements that must survive whatever the week's clustering does: the
# ticker takes the theme most of its listed peers land in this run (so the pin
# follows the business group through renames). User decision 2026-10-04: MSFT and
# ORCL go with enterprise software. Needed in practice: on 2026-10-05 HDBSCAN put
# ORCL in a confident "quantum_computing" cluster beside two quantum-software names.
TICKER_PEER_PINS: dict[str, list[str]] = {
    "MSFT": ["CRM", "NOW", "ADBE", "INTU", "WDAY", "SNOW", "DDOG"],
    "ORCL": ["CRM", "NOW", "ADBE", "INTU", "WDAY", "SNOW", "DDOG"],
}

# ── News window ───────────────────────────────────────────────────────────────
NEWS_LOOKBACK_DAYS = 30
MAX_HEADLINES_PER_TICKER = 10

# ── Theme heat features ───────────────────────────────────────────────────────
HEAT_WINDOW_DAYS = 5          # short window for heat / breadth
HEAT_PRIOR_WINDOW_DAYS = 10   # prior window for acceleration base
BREADTH_THRESHOLD = 0.0       # return > 0 = constituent is "up"
TOP_Q_STRENGTH = 0.75         # top-quartile cutoff for theme_strength

# ── Claude ────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Labeling: the JSON itself is ~300 tokens, but "plenty" assumed the response
# is JSON and nothing else. On ambiguous clusters the model reasons through the
# tickers first, and at 512 the budget is gone before the JSON is emitted — the
# response ends mid-sentence with no object to parse, and the cluster falls back
# to an ephemeral `cluster_<id>` name. Same failure the relationship graph hit
# below. Sized for a preamble plus the object.
CLAUDE_LABEL_MAX_TOKENS = 2048
# Relationship graph: ~85 themes × ~5 edges each is a large JSON array. At 8192
# tokens the response truncated mid-object and the whole graph was discarded,
# leaving stale edges that no longer matched the relabeled registry. Sonnet
# supports far larger outputs, so give it ample room.
CLAUDE_RELATIONSHIP_MAX_TOKENS = 64000
# Relationship graph carry-forward: if the fraction of themes that are new
# since last week is at or below this, only ask Claude to place the new
# themes and carry forward last week's edges for everything else. Above it,
# rebuild the full graph in one call (incremental placement stops saving
# much once most of the taxonomy changed).
RELATIONSHIP_FULL_REBUILD_NEW_FRACTION = 0.5
CLAUDE_MAX_TOKENS = 512  # default fallback


def ensure_outputs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
