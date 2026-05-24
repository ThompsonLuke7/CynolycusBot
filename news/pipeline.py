"""News collection, embedding, clustering, labels, and features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from news.config import (
    LOSER_LIBRARY_PATH,
    NEWS_EMBEDDINGS_PATH,
    NEWS_FEATURE_COLUMNS,
    NEWS_FEATURE_MATRIX_PATH,
    NEWS_LABELS_PATH,
    NEWS_RECORDS_PATH,
    WINNER_LIBRARY_PATH,
    ensure_data_dirs,
)
from news.dedup import deduplicate_news
from news.nlp import embed_texts_bge, finbert_scores
from news.schema import empty_news_frame, records_from_frame
from news.sources import (
    fetch_alpha_vantage_news,
    fetch_finnhub_company_news,
    fetch_fmp_stock_news,
    fetch_sec_8k_news,
)


def collect_company_news(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    sources: Iterable[str] = ("finnhub", "fmp", "alpha_vantage", "sec_8k"),
    output_path: Path | str = NEWS_RECORDS_PATH,
) -> pd.DataFrame:
    ensure_data_dirs()
    frames = []
    source_set = set(sources)
    if "finnhub" in source_set:
        frames.append(fetch_finnhub_company_news(tickers, start=start, end=end))
    if "fmp" in source_set:
        frames.append(fetch_fmp_stock_news(tickers, start=start, end=end))
    if "alpha_vantage" in source_set:
        frames.append(fetch_alpha_vantage_news(tickers))
    if "sec_8k" in source_set:
        frames.append(fetch_sec_8k_news(tickers, start=start, end=end))
    raw = pd.concat(frames, ignore_index=True) if frames else empty_news_frame()
    out = deduplicate_news(raw)
    out.to_parquet(output_path, index=False)
    return out


def collect_news_from_csv(input_csv: Path | str, *, output_path: Path | str = NEWS_RECORDS_PATH) -> pd.DataFrame:
    ensure_data_dirs()
    out = deduplicate_news(records_from_frame(pd.read_csv(input_csv), source="csv"))
    out.to_parquet(output_path, index=False)
    return out


def build_news_embeddings(
    news_path: Path | str = NEWS_RECORDS_PATH,
    *,
    output_path: Path | str = NEWS_EMBEDDINGS_PATH,
    generate_embeddings: bool = True,
    generate_finbert: bool = True,
) -> pd.DataFrame:
    ensure_data_dirs()
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    if news.empty:
        out = pd.DataFrame(columns=["record_id", "embedding", "embedding_available", "finbert_available"])
        out.to_parquet(output_path, index=False)
        return out

    out = news[["record_id", "ticker", "timestamp", "text"]].copy()
    out["embedding"] = None
    out["embedding_available"] = 0.0
    if generate_embeddings:
        try:
            vectors = embed_texts_bge(out["text"].fillna("").tolist())
            out["embedding"] = [json.dumps(v.astype(float).tolist()) for v in vectors]
            out["embedding_available"] = 1.0
        except ImportError:
            out["embedding_available"] = 0.0

    tone_rows = []
    for text in out["text"].fillna(""):
        if not generate_finbert:
            tone_rows.append({"finbert_positive_score": np.nan, "finbert_negative_score": np.nan, "finbert_neutral_score": np.nan, "finbert_available": 0.0})
            continue
        try:
            scores = finbert_scores(text)
            scores["finbert_available"] = 1.0
            tone_rows.append(scores)
        except ImportError:
            tone_rows.append({"finbert_positive_score": np.nan, "finbert_negative_score": np.nan, "finbert_neutral_score": np.nan, "finbert_available": 0.0})
    out = pd.concat([out, pd.DataFrame(tone_rows)], axis=1)
    out.to_parquet(output_path, index=False)
    return out


def parse_embedding(value: object) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    try:
        return np.asarray(json.loads(str(value)), dtype=np.float32)
    except Exception:
        return None


def cluster_news_embeddings(
    embeddings_path: Path | str = NEWS_EMBEDDINGS_PATH,
    *,
    output_path: Path | str = NEWS_EMBEDDINGS_PATH,
    n_clusters: int = 12,
) -> pd.DataFrame:
    df = pd.read_parquet(embeddings_path) if Path(embeddings_path).exists() else pd.DataFrame()
    if df.empty or "embedding" not in df.columns:
        df["news_cluster_id"] = np.nan
        df.to_parquet(output_path, index=False)
        return df
    vectors = [parse_embedding(v) for v in df["embedding"]]
    valid_idx = [i for i, v in enumerate(vectors) if v is not None]
    df["news_cluster_id"] = np.nan
    if len(valid_idx) >= 2:
        from sklearn.cluster import KMeans

        x = np.vstack([vectors[i] for i in valid_idx])
        k = min(int(n_clusters), len(valid_idx))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
        df.loc[df.index[valid_idx], "news_cluster_id"] = labels.astype(float)
    df.to_parquet(output_path, index=False)
    return df


def label_news_forward_returns(
    news_path: Path | str,
    bars: pd.DataFrame,
    *,
    bars_per_day: int = 13,
    expansion_threshold: float = 0.10,
    output_path: Path | str = NEWS_LABELS_PATH,
) -> pd.DataFrame:
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    if news.empty:
        out = pd.DataFrame()
        out.to_parquet(output_path, index=False)
        return out
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values(["ticker", "timestamp"])
    rows = []
    for rec in news.itertuples(index=False):
        ticker_bars = bars.loc[bars["ticker"].astype(str).str.upper().eq(str(rec.ticker).upper())]
        ts = pd.Timestamp(rec.timestamp)
        future = ticker_bars.loc[ticker_bars["timestamp"] > ts].head(int(10 * bars_per_day))
        if future.empty:
            continue
        entry = float(future.iloc[0]["close"])
        closes = future["close"].astype(float).to_numpy()
        returns = closes / entry - 1.0
        one_day = int(bars_per_day)
        five_day = int(5 * bars_per_day)
        ten_day = int(10 * bars_per_day)
        rows.append(
            {
                "record_id": rec.record_id,
                "ticker": rec.ticker,
                "timestamp": ts,
                "forward_1d_return": float(returns[one_day - 1]) if len(returns) >= one_day else np.nan,
                "forward_5d_return": float(returns[five_day - 1]) if len(returns) >= five_day else np.nan,
                "forward_10d_return": float(returns[ten_day - 1]) if len(returns) >= ten_day else np.nan,
                "max_forward_return": float(np.nanmax(returns)),
                "max_drawdown": float(np.nanmin(returns)),
                "expansion_label": float(np.nanmax(returns) >= expansion_threshold),
            }
        )
    out = pd.DataFrame(rows)
    out.to_parquet(output_path, index=False)
    return out


def build_winner_loser_libraries(
    embeddings_path: Path | str = NEWS_EMBEDDINGS_PATH,
    labels_path: Path | str = NEWS_LABELS_PATH,
    *,
    winner_path: Path | str = WINNER_LIBRARY_PATH,
    loser_path: Path | str = LOSER_LIBRARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb = pd.read_parquet(embeddings_path) if Path(embeddings_path).exists() else pd.DataFrame()
    labels = pd.read_parquet(labels_path) if Path(labels_path).exists() else pd.DataFrame()
    if emb.empty or labels.empty:
        winners = pd.DataFrame()
        losers = pd.DataFrame()
    else:
        lib = emb.merge(labels, on=["record_id", "ticker", "timestamp"], how="inner")
        winners = lib.loc[lib["expansion_label"].eq(1.0)].reset_index(drop=True)
        losers = lib.loc[lib["expansion_label"].eq(0.0)].reset_index(drop=True)
    winners.to_parquet(winner_path, index=False)
    losers.to_parquet(loser_path, index=False)
    return winners, losers


def max_cosine_similarity(vector: np.ndarray | None, library_embeddings: list[np.ndarray]) -> float:
    if vector is None or not library_embeddings:
        return float("nan")
    x = vector / max(float(np.linalg.norm(vector)), 1e-12)
    sims = []
    for item in library_embeddings:
        y = item / max(float(np.linalg.norm(item)), 1e-12)
        sims.append(float(np.dot(x, y)))
    return float(np.nanmax(sims)) if sims else float("nan")


def build_news_features(
    timestamps: pd.DataFrame,
    news_path: Path | str = NEWS_RECORDS_PATH,
    embeddings_path: Path | str = NEWS_EMBEDDINGS_PATH,
    *,
    winner_path: Path | str = WINNER_LIBRARY_PATH,
    loser_path: Path | str = LOSER_LIBRARY_PATH,
    output_path: Path | str = NEWS_FEATURE_MATRIX_PATH,
) -> pd.DataFrame:
    ensure_data_dirs()
    base = timestamps.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base["ticker"] = base["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    emb = pd.read_parquet(embeddings_path) if Path(embeddings_path).exists() else pd.DataFrame()
    if not news.empty:
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    merged = news.merge(emb.drop(columns=["ticker", "timestamp", "text"], errors="ignore"), on="record_id", how="left") if not news.empty and not emb.empty else news

    winners = pd.read_parquet(winner_path) if Path(winner_path).exists() else pd.DataFrame()
    losers = pd.read_parquet(loser_path) if Path(loser_path).exists() else pd.DataFrame()
    winner_vecs = [v for v in (parse_embedding(x) for x in winners.get("embedding", [])) if v is not None]
    loser_vecs = [v for v in (parse_embedding(x) for x in losers.get("embedding", [])) if v is not None]

    rows = []
    for row in base.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        ticker = str(row.ticker).upper()
        prior = merged.loc[
            merged["ticker"].astype(str).str.upper().eq(ticker)
            & merged["timestamp"].between(ts - pd.Timedelta(hours=24), ts)
        ] if not merged.empty else pd.DataFrame()
        latest = prior.sort_values("timestamp").tail(1)
        latest_embedding = parse_embedding(latest.iloc[0]["embedding"]) if not latest.empty and "embedding" in latest.columns else None
        winner_sim = max_cosine_similarity(latest_embedding, winner_vecs)
        loser_sim = max_cosine_similarity(latest_embedding, loser_vecs)
        rows.append(
            {
                "timestamp": ts,
                "ticker": ticker,
                "news_count_24h": float(len(prior)),
                "hours_since_news": float((ts - prior["timestamp"].max()).total_seconds() / 3600.0) if not prior.empty else np.nan,
                "finbert_positive_score": float(latest.get("finbert_positive_score", pd.Series([np.nan])).iloc[0]) if not latest.empty else np.nan,
                "finbert_negative_score": float(latest.get("finbert_negative_score", pd.Series([np.nan])).iloc[0]) if not latest.empty else np.nan,
                "finbert_neutral_score": float(latest.get("finbert_neutral_score", pd.Series([np.nan])).iloc[0]) if not latest.empty else np.nan,
                "news_cluster_id": float(latest.get("news_cluster_id", pd.Series([np.nan])).iloc[0]) if not latest.empty else np.nan,
                "winner_similarity_max": winner_sim,
                "loser_similarity_max": loser_sim,
                "news_edge_score": winner_sim - loser_sim if np.isfinite(winner_sim) and np.isfinite(loser_sim) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    for col in NEWS_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out.to_parquet(output_path, index=False)
    return out
