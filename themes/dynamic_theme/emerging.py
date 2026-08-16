"""Provisional theme discovery for unpromoted ticker candidates.

This layer is intentionally separate from the production theme registry and
meta-ranker features. Pending tickers first map to established theme centroids;
only semantically novel residuals are clustered and labeled as provisional
themes. Repeated runs append a compact history so persistence can be measured
before anything is promoted into the production taxonomy.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from core.shared_universe.universe import PENDING_NOVEL_TICKERS_CSV
from themes.dynamic_theme.config import (
    BGE_MODEL,
    DAILY_BARS_DIR,
    DESC_BLEND_WEIGHT,
    PENDING_THEME_CANDIDATES_PATH,
    PENDING_THEME_HISTORY_PATH,
    PENDING_THEME_MEMBERSHIP_PATH,
    PENDING_THEME_NEWS_PATH,
    PENDING_THEME_PROFILES_PATH,
    PENDING_THEME_REGISTRY_PATH,
    TICKER_CLUSTERS_PATH,
    TICKER_EMBEDDINGS_PATH,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step02_embed import _clean_text
from themes.dynamic_theme.stages.step04_cluster_summary import _top_keywords
from themes.dynamic_theme.stages.step05_claude_labeling import _label_cluster, load_registry
from themes.dynamic_theme.sklearn_compat import load_hdbscan, load_umap

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 300
# Best-centroid similarities are naturally much higher than random ticker-pair
# similarities. 0.82 isolates roughly the weakest semantic quintile without
# manufacturing novelty from clearly established-theme members.
ESTABLISHED_SIMILARITY_THRESHOLD = 0.82
MIN_PROVISIONAL_CLUSTER_SIZE = 5
PROFILE_MAX_AGE_DAYS = 7


def rank_pending_candidates(pending: pd.DataFrame, *, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """Rank theme-ready pending names without treating size alone as novelty."""
    if pending.empty:
        return pending.copy()
    out = pending.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    for col in ("last_price", "avg_dollar_volume_20d", "market_cap", "history_days", "catalyst_mentions"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    for col in ("passes_price", "passes_adv", "passes_history"):
        out[col] = out.get(col, False).fillna(False).astype(bool)

    out = out[
        out["status"].astype(str).eq("pending")
        & out["passes_price"]
        & out["passes_adv"]
        & out["passes_history"]
    ].copy()
    if out.empty:
        return out

    def percentile(series: pd.Series) -> pd.Series:
        return series.rank(pct=True, method="average").fillna(0.0)

    adv_score = percentile(np.log1p(out["avg_dollar_volume_20d"].clip(lower=0)))
    history_score = percentile(out["history_days"])
    catalyst_raw = out["catalyst_mentions"].fillna(0).clip(lower=0)
    catalyst_score = np.log1p(catalyst_raw) / max(1.0, float(np.log1p(catalyst_raw.max())))
    source_bonus = out["source"].astype(str).str.contains("catalyst|both", case=False, regex=True).astype(float)
    known_cap = out["market_cap"].notna().astype(float)
    small_mid_bonus = out["market_cap"].between(100e6, 10e9, inclusive="both").fillna(False).astype(float)

    out["candidate_score"] = (
        0.34 * adv_score
        + 0.20 * history_score
        + 0.23 * catalyst_score
        + 0.10 * source_bonus
        + 0.05 * known_cap
        + 0.08 * small_mid_bonus
    )
    out = out.sort_values(
        ["candidate_score", "catalyst_mentions", "avg_dollar_volume_20d"],
        ascending=False,
    ).head(max(1, int(limit)))
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def _load_cached_profiles() -> pd.DataFrame:
    if not PENDING_THEME_PROFILES_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(PENDING_THEME_PROFILES_PATH)
    except Exception:
        return pd.DataFrame()


def enrich_profiles(tickers: list[str]) -> pd.DataFrame:
    """Refresh missing/stale Yahoo profiles and preserve prior successful rows."""
    from signals.news.sources import fetch_yfinance_profiles

    cached = _load_cached_profiles()
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    fresh: set[str] = set()
    if not cached.empty and "snapshot_date" in cached:
        snapshot = pd.to_datetime(cached["snapshot_date"], errors="coerce").dt.tz_localize(None)
        fresh = set(
            cached.loc[
                snapshot.ge(today - pd.Timedelta(days=PROFILE_MAX_AGE_DAYS))
                & cached["longBusinessSummary"].fillna("").astype(str).str.len().gt(30),
                "ticker",
            ].astype(str)
        )
    needed = [ticker for ticker in tickers if ticker not in fresh]
    logger.info("Pending profiles: %d cached/fresh, %d to fetch", len(tickers) - len(needed), len(needed))
    fetched = fetch_yfinance_profiles(needed, progress_every=25) if needed else pd.DataFrame()
    frames = [frame for frame in (cached, fetched) if not frame.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"ticker": tickers})
    if not merged.empty:
        merged["ticker"] = merged["ticker"].astype(str).str.upper()
        merged = merged.sort_values("snapshot_date").drop_duplicates("ticker", keep="last")
        merged.to_parquet(PENDING_THEME_PROFILES_PATH, index=False)
    return merged[merged["ticker"].isin(tickers)].copy()


def fetch_daily_bars(tickers: list[str], *, lookback_days: int = 420, batch_size: int = 100) -> int:
    """Batch-fetch candidate daily bars into the shared cache using Alpaca IEX."""
    import datetime as dt

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from core.API.Alpaca_API.core.config import AlpacaConfig

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=int(lookback_days))
    needed = list(tickers)

    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(cfg.key_id, cfg.secret_key)
    saved = 0
    DAILY_BARS_DIR.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(needed), batch_size):
        batch = needed[offset : offset + batch_size]
        logger.info("Pending daily bars: batch %d/%d (%d symbols)", offset // batch_size + 1, math.ceil(len(needed) / batch_size), len(batch))
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
            adjustment=Adjustment.RAW,
            feed=DataFeed.IEX,
        )
        try:
            response = client.get_stock_bars(req)
            frame = response.df.reset_index() if response.df is not None and not response.df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Daily-bar batch failed: %s", exc)
            continue
        if frame.empty:
            continue
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        for ticker, group in frame.groupby("symbol"):
            out = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            path = DAILY_BARS_DIR / f"{ticker}.parquet"
            if path.exists():
                try:
                    prior = pd.read_parquet(path)
                    out = (
                        pd.concat([prior, out], ignore_index=True)
                        .sort_values("timestamp")
                        .drop_duplicates("timestamp", keep="last")
                        .reset_index(drop=True)
                    )
                except Exception:
                    pass
            out.to_parquet(path, index=False)
            saved += 1
    return saved


def collect_pending_news(
    tickers: list[str],
    *,
    lookback_days: int = 14,
    sources: Iterable[str] = ("finnhub", "yfinance", "google_news"),
) -> pd.DataFrame:
    """Collect a focused recent-news corpus without polluting production records."""
    from signals.news.pipeline import collect_company_news

    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=int(lookback_days))
    return collect_company_news(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        sources=tuple(sources),
        sec_include_archives=False,
        sec_enrich_ex99=False,
        output_path=PENDING_THEME_NEWS_PATH,
        merge_with_existing=True,
    )


def build_candidate_documents(
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
    news: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    profile_by = profiles.set_index("ticker").to_dict("index") if not profiles.empty else {}
    headlines: dict[str, str] = {}
    if not news.empty:
        news = news.copy()
        news["ticker"] = news["ticker"].astype(str).str.upper()
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
        for ticker, group in news.sort_values("timestamp", ascending=False).groupby("ticker"):
            headlines[ticker] = " | ".join(group["headline"].dropna().astype(str).head(12))

    rows = []
    for row in candidates.itertuples(index=False):
        ticker = str(row.ticker)
        profile = profile_by.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "description": str(profile.get("longBusinessSummary") or "")[:1200],
                "sector": str(profile.get("sector") or profile.get("sectorDisp") or ""),
                "industry": str(profile.get("industry") or profile.get("industryDisp") or ""),
                "quote_type": str(profile.get("quoteType") or ""),
                "recent_news_summary": headlines.get(ticker, ""),
                "candidate_score": float(row.candidate_score),
                "candidate_rank": int(row.candidate_rank),
                "catalyst_mentions": float(row.catalyst_mentions) if pd.notna(row.catalyst_mentions) else 0.0,
                "date": as_of,
            }
        )
    return pd.DataFrame(rows)


def embed_candidate_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """Build semantic-only vectors compatible with production text dimensions."""
    from signals.news.nlp import embed_texts_bge

    descriptions = docs["description"].fillna("").map(_clean_text)
    news = docs["recent_news_summary"].fillna("").map(_clean_text)
    fallback = (
        docs["sector"].fillna("")
        + " "
        + docs["industry"].fillna("")
        + " "
        + docs["ticker"].fillna("")
    ).map(_clean_text)

    desc_text = descriptions.where(descriptions.str.len().gt(20), fallback)
    news_mask = news.str.len().gt(20)
    desc_vecs = np.asarray(embed_texts_bge(desc_text.tolist(), model_name=BGE_MODEL), dtype=np.float32)
    vectors = desc_vecs.copy()
    if news_mask.any():
        news_vecs = np.asarray(embed_texts_bge(news[news_mask].tolist(), model_name=BGE_MODEL), dtype=np.float32)
        vectors[news_mask.to_numpy()] = (
            float(DESC_BLEND_WEIGHT) * desc_vecs[news_mask.to_numpy()]
            + (1.0 - float(DESC_BLEND_WEIGHT)) * news_vecs
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return pd.DataFrame(
        {"ticker": docs["ticker"].tolist(), "embedding": [v.tolist() for v in vectors], "date": docs["date"].tolist()}
    )


def _established_text_centroids() -> tuple[list[str], np.ndarray]:
    registry = load_registry(latest_only=True)
    clusters = pd.read_parquet(TICKER_CLUSTERS_PATH)
    embeddings = pd.read_parquet(TICKER_EMBEDDINGS_PATH)
    registry = registry.sort_values("confidence", ascending=False).drop_duplicates("cluster_id", keep="first")
    id_to_theme = dict(zip(registry["cluster_id"].astype(int), registry["theme_name"].astype(str)))
    vector_by = {
        str(ticker): np.asarray(vector, dtype=np.float32)[:384]
        for ticker, vector in zip(embeddings["ticker"], embeddings["embedding"])
    }
    names, centroids = [], []
    for cluster_id, group in clusters[clusters["cluster_id"].ge(0)].groupby("cluster_id"):
        name = id_to_theme.get(int(cluster_id))
        vectors = [vector_by[str(ticker)] for ticker in group["ticker"] if str(ticker) in vector_by]
        if not name or not vectors:
            continue
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        names.append(name)
        centroids.append(centroid / norm if norm > 0 else centroid)
    return names, np.asarray(centroids, dtype=np.float32)


def map_to_established(embeddings: pd.DataFrame) -> pd.DataFrame:
    names, centroids = _established_text_centroids()
    matrix = np.asarray(embeddings["embedding"].tolist(), dtype=np.float32)
    similarities = matrix @ centroids.T
    best = similarities.argmax(axis=1)
    out = pd.DataFrame(
        {
            "ticker": embeddings["ticker"].astype(str),
            "closest_theme": [names[index] for index in best],
            "theme_similarity": similarities[np.arange(len(matrix)), best],
        }
    )
    out["assignment_type"] = np.where(
        out["theme_similarity"].ge(ESTABLISHED_SIMILARITY_THRESHOLD),
        "extension",
        "novel_residual",
    )
    return out


def cluster_residuals(embeddings: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Cluster only candidates that do not resemble an established theme."""
    residual_tickers = set(assignments.loc[assignments["assignment_type"].eq("novel_residual"), "ticker"])
    residual = embeddings[embeddings["ticker"].isin(residual_tickers)].copy()
    if len(residual) < MIN_PROVISIONAL_CLUSTER_SIZE:
        return pd.DataFrame(columns=["ticker", "provisional_cluster", "cluster_probability"])
    hdbscan = load_hdbscan()
    umap = load_umap()

    matrix = np.asarray(residual["embedding"].tolist(), dtype=np.float32)
    n_neighbors = min(15, max(2, len(residual) - 1))
    n_components = min(12, matrix.shape[1] - 1, len(residual) - 2)
    reduced = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    ).fit_transform(matrix)
    model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_PROVISIONAL_CLUSTER_SIZE,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="leaf",
    ).fit(reduced)
    return pd.DataFrame(
        {
            "ticker": residual["ticker"].tolist(),
            "provisional_cluster": model.labels_.astype(int),
            "cluster_probability": model.probabilities_.astype(float),
        }
    )


def _price_metrics(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        path = DAILY_BARS_DIR / f"{ticker}.parquet"
        if not path.exists():
            continue
        try:
            bars = pd.read_parquet(path).sort_values("timestamp")
            close = pd.to_numeric(bars["close"], errors="coerce").dropna()
            volume = pd.to_numeric(bars["volume"], errors="coerce").dropna()
            rows.append(
                {
                    "ticker": ticker,
                    "return_5d": float(close.pct_change(5).iloc[-1]) if len(close) > 5 else np.nan,
                    "return_20d": float(close.pct_change(20).iloc[-1]) if len(close) > 20 else np.nan,
                    "volume_acceleration": (
                        float(volume.tail(5).mean() / volume.tail(20).mean() - 1.0)
                        if len(volume) >= 20 and volume.tail(20).mean() > 0
                        else np.nan
                    ),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def _fallback_label(cluster_id: int, docs: pd.DataFrame) -> dict:
    texts = (
        docs["sector"].fillna("")
        + " "
        + docs["industry"].fillna("")
        + " "
        + docs["recent_news_summary"].fillna("")
    ).tolist()
    keywords = _top_keywords(texts, top_n=3)
    stem = "_".join(keywords[:2]) or f"cluster_{cluster_id}"
    return {
        "theme_name": f"emerging_{stem}",
        "parent_theme": "emerging_market_theme",
        "description": f"Provisional cluster centered on {', '.join(keywords) or 'newly discovered companies'}.",
        "related_themes": [],
        "confidence": 0.35,
    }


def _prior_provisional_labels() -> list[dict]:
    if not PENDING_THEME_REGISTRY_PATH.exists() or not PENDING_THEME_MEMBERSHIP_PATH.exists():
        return []
    try:
        registry = pd.read_parquet(PENDING_THEME_REGISTRY_PATH)
        membership = pd.read_parquet(PENDING_THEME_MEMBERSHIP_PATH)
    except Exception:
        return []
    rows = []
    for _, row in registry[registry["theme_type"].eq("provisional")].iterrows():
        key = str(row["theme_key"])
        tickers = set(membership.loc[membership["theme_key"].eq(key), "ticker"].astype(str))
        rows.append({"row": row, "tickers": tickers})
    return rows


def build_outputs(
    candidates: pd.DataFrame,
    docs: pd.DataFrame,
    embeddings: pd.DataFrame,
    assignments: pd.DataFrame,
    residual_clusters: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    document_columns = [
        "ticker",
        "description",
        "sector",
        "industry",
        "quote_type",
        "recent_news_summary",
    ]
    enriched = candidates.merge(docs[document_columns], on="ticker", how="left")
    enriched = enriched.merge(assignments, on="ticker", how="left")
    if residual_clusters.empty:
        residual_clusters = pd.DataFrame(
            columns=["ticker", "provisional_cluster", "cluster_probability"]
        )
    enriched = enriched.merge(residual_clusters, on="ticker", how="left")
    price_metrics = _price_metrics(enriched["ticker"].tolist())
    if price_metrics.empty:
        price_metrics = pd.DataFrame(
            columns=["ticker", "return_5d", "return_20d", "volume_acceleration"]
        )
    enriched = enriched.merge(price_metrics, on="ticker", how="left")
    for column in (
        "return_5d",
        "return_20d",
        "volume_acceleration",
        "provisional_cluster",
        "cluster_probability",
    ):
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    registry_rows = []
    membership_rows = []
    prior_provisional = _prior_provisional_labels()
    existing_extensions = enriched[enriched["assignment_type"].eq("extension")]
    for theme, group in existing_extensions.groupby("closest_theme"):
        score = float(
            0.55 * group["candidate_score"].mean()
            + 0.25 * group["theme_similarity"].mean()
            + 0.20 * group["return_5d"].fillna(0).gt(0).mean()
        )
        registry_rows.append(
            {
                "theme_key": f"extension::{theme}",
                "theme_name": str(theme),
                "theme_type": "extension",
                "parent_theme": "",
                "description": f"Pending ticker expansion into established theme {theme}.",
                "closest_theme": str(theme),
                "closest_similarity": float(group["theme_similarity"].mean()),
                "emerging_score": score,
                "ticker_count": int(len(group)),
                "breadth_5d": float(group["return_5d"].fillna(0).gt(0).mean()),
                "avg_return_5d": float(group["return_5d"].mean()) if group["return_5d"].notna().any() else np.nan,
                "avg_volume_acceleration": float(group["volume_acceleration"].mean()) if group["volume_acceleration"].notna().any() else np.nan,
                "tickers": json.dumps(group["ticker"].tolist()),
                "date": as_of,
            }
        )
        for row in group.itertuples(index=False):
            membership_rows.append(
                {
                    "ticker": row.ticker,
                    "theme": theme,
                    "theme_key": f"extension::{theme}",
                    "theme_type": "extension",
                    "membership_score": float(row.theme_similarity),
                    "date": as_of,
                }
            )

    clustered = enriched[enriched["provisional_cluster"].fillna(-1).ge(0)].copy()
    for cluster_id, group in clustered.groupby("provisional_cluster"):
        cluster_id = int(cluster_id)
        group_docs = docs[docs["ticker"].isin(group["ticker"])].copy()
        summary = {
            "cluster_id": cluster_id,
            "tickers": group["ticker"].tolist(),
            "top_keywords": _top_keywords(
                (
                    group_docs["description"].fillna("")
                    + " "
                    + group_docs["recent_news_summary"].fillna("")
                ).tolist()
            ),
            "sample_headlines": [
                headline
                for text in group_docs["recent_news_summary"].fillna("")
                for headline in str(text).split(" | ")[:2]
                if headline
            ][:6],
        }
        current_tickers = set(group["ticker"].astype(str))
        prior_match = None
        prior_overlap = 0.0
        for prior in prior_provisional:
            union = current_tickers | prior["tickers"]
            overlap = len(current_tickers & prior["tickers"]) / len(union) if union else 0.0
            if overlap > prior_overlap:
                prior_overlap = overlap
                prior_match = prior
        if prior_match is not None and prior_overlap >= 0.40:
            prior_row = prior_match["row"]
            label = {
                "theme_name": prior_row["theme_name"],
                "parent_theme": prior_row["parent_theme"],
                "description": prior_row["description"],
                "confidence": prior_row.get("emerging_score", 0.5),
            }
        else:
            try:
                label = _label_cluster(summary)
            except Exception:
                label = _fallback_label(cluster_id, group_docs)
        if str(label.get("theme_name", "")).startswith("cluster_"):
            label = _fallback_label(cluster_id, group_docs)
        closest_row = group.sort_values("theme_similarity", ascending=False).iloc[0]
        breadth = float(group["return_5d"].fillna(0).gt(0).mean())
        score = float(
            0.40 * group["candidate_score"].mean()
            + 0.20 * min(1.0, len(group) / 12.0)
            + 0.15 * group["cluster_probability"].fillna(0).mean()
            + 0.15 * breadth
            + 0.10 * np.clip(group["volume_acceleration"].fillna(0).mean(), -1, 1)
        )
        theme_name = str(label.get("theme_name") or f"emerging_cluster_{cluster_id}")
        theme_key = f"provisional::{theme_name}"
        registry_rows.append(
            {
                "theme_key": theme_key,
                "theme_name": theme_name,
                "theme_type": "provisional",
                "parent_theme": str(label.get("parent_theme") or "emerging_market_theme"),
                "description": str(label.get("description") or ""),
                "closest_theme": str(closest_row["closest_theme"]),
                "closest_similarity": float(group["theme_similarity"].mean()),
                "emerging_score": score,
                "ticker_count": int(len(group)),
                "breadth_5d": breadth,
                "avg_return_5d": float(group["return_5d"].mean()) if group["return_5d"].notna().any() else np.nan,
                "avg_volume_acceleration": float(group["volume_acceleration"].mean()) if group["volume_acceleration"].notna().any() else np.nan,
                "tickers": json.dumps(group["ticker"].tolist()),
                "date": as_of,
            }
        )
        for row in group.itertuples(index=False):
            membership_rows.append(
                {
                    "ticker": row.ticker,
                    "theme": theme_name,
                    "theme_key": theme_key,
                    "theme_type": "provisional",
                    "membership_score": float(row.cluster_probability),
                    "date": as_of,
                }
            )

    registry = pd.DataFrame(registry_rows)
    membership = pd.DataFrame(membership_rows)
    enriched["date"] = as_of
    enriched.to_parquet(PENDING_THEME_CANDIDATES_PATH, index=False)
    registry.to_parquet(PENDING_THEME_REGISTRY_PATH, index=False)
    membership.to_parquet(PENDING_THEME_MEMBERSHIP_PATH, index=False)

    if PENDING_THEME_HISTORY_PATH.exists():
        history = pd.read_parquet(PENDING_THEME_HISTORY_PATH)
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history[history["date"].ne(pd.Timestamp(as_of))]
        history = pd.concat([history, registry], ignore_index=True)
    else:
        history = registry.copy()
    history = history.drop_duplicates(["theme_key", "date"], keep="last").sort_values(["date", "theme_key"])
    history.to_parquet(PENDING_THEME_HISTORY_PATH, index=False)
    return registry, membership


def run(
    *,
    limit: int = DEFAULT_LIMIT,
    enrich: bool = True,
    collect_news: bool = True,
) -> dict:
    ensure_outputs()
    as_of = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    pending = pd.read_csv(PENDING_NOVEL_TICKERS_CSV)
    candidates = rank_pending_candidates(pending, limit=limit)
    tickers = candidates["ticker"].tolist()
    logger.info("Selected %d provisional-theme candidates", len(tickers))

    profiles = enrich_profiles(tickers) if enrich else _load_cached_profiles()
    if enrich:
        saved = fetch_daily_bars(tickers)
        logger.info("Saved daily bars for %d pending candidates", saved)
    news = collect_pending_news(tickers) if collect_news else (
        pd.read_parquet(PENDING_THEME_NEWS_PATH) if PENDING_THEME_NEWS_PATH.exists() else pd.DataFrame()
    )
    docs = build_candidate_documents(candidates, profiles, news, as_of=as_of)
    embeddings = embed_candidate_documents(docs)
    assignments = map_to_established(embeddings)
    residual_clusters = cluster_residuals(embeddings, assignments)
    registry, membership = build_outputs(
        candidates,
        docs,
        embeddings,
        assignments,
        residual_clusters,
        as_of=as_of,
    )
    return {
        "candidates": len(candidates),
        "extensions": int((assignments["assignment_type"] == "extension").sum()),
        "residuals": int((assignments["assignment_type"] == "novel_residual").sum()),
        "provisional_themes": int((registry["theme_type"] == "provisional").sum()) if not registry.empty else 0,
        "extension_themes": int((registry["theme_type"] == "extension").sum()) if not registry.empty else 0,
        "memberships": len(membership),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = run(limit=args.limit, enrich=not args.skip_enrich, collect_news=not args.skip_news)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
