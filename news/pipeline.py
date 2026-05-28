"""News collection, embedding, clustering, labels, and features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from news.config import (
    LABEL_HORIZON_DAYS,
    LOSER_LIBRARY_PATH,
    NEWS_EMBEDDINGS_PATH,
    NEWS_FEATURE_COLUMNS,
    NEWS_FEATURE_MATRIX_PATH,
    NEWS_LABELS_PATH,
    NEWS_RECORDS_PATH,
    NEWS_SCORES_PATH,
    WINNER_LIBRARY_PATH,
    ensure_data_dirs,
)
from news.catalyst_types import classify_catalyst_types
from news.dedup import deduplicate_news
from news.earnings import enrich_earnings_catalyst_fields
from news.nlp import embed_texts_bge, finbert_scores_batch
from news.relations import classify_news_relations
from news.schema import empty_news_frame, records_from_frame
from news.sources import (
    fetch_sec_alpha_filings,
    fetch_finnhub_company_news,
    fetch_sec_8k_news,
)


def collect_company_news(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    sources: Iterable[str] = ("finnhub", "sec_8k"),
    sec_include_archives: bool = True,
    sec_full_text_limit: int = 0,
    output_path: Path | str = NEWS_RECORDS_PATH,
) -> pd.DataFrame:
    ensure_data_dirs()
    frames = []
    source_set = set(sources)
    if "finnhub" in source_set:
        frames.append(fetch_finnhub_company_news(tickers, start=start, end=end))
    if "sec_8k" in source_set:
        frames.append(
            fetch_sec_8k_news(
                tickers,
                start=start,
                end=end,
                include_archives=sec_include_archives,
                full_text_limit=sec_full_text_limit,
            )
        )
    if "sec_alpha" in source_set:
        frames.append(
            fetch_sec_alpha_filings(
                tickers,
                start=start,
                end=end,
                include_archives=sec_include_archives,
                full_text_limit=sec_full_text_limit,
            )
        )
    raw = pd.concat(frames, ignore_index=True) if frames else empty_news_frame()
    if not raw.empty:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        raw = raw.loc[raw["timestamp"].between(start_ts, end_ts)].copy()
    out = deduplicate_news(raw)
    out = classify_catalyst_types(enrich_earnings_catalyst_fields(classify_news_relations(out)))
    out.to_parquet(output_path, index=False)
    return out


def collect_news_from_csv(input_csv: Path | str, *, output_path: Path | str = NEWS_RECORDS_PATH) -> pd.DataFrame:
    ensure_data_dirs()
    out = classify_catalyst_types(enrich_earnings_catalyst_fields(classify_news_relations(deduplicate_news(records_from_frame(pd.read_csv(input_csv), source="csv")))))
    out.to_parquet(output_path, index=False)
    return out


def classify_existing_news(
    news_path: Path | str = NEWS_RECORDS_PATH,
    *,
    output_path: Path | str = NEWS_RECORDS_PATH,
) -> pd.DataFrame:
    ensure_data_dirs()
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    out = classify_catalyst_types(enrich_earnings_catalyst_fields(classify_news_relations(news)))
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

    keep_cols = ["record_id", "ticker", "timestamp", "text"]
    for col in ("catalyst_family", "catalyst_subtype"):
        if col in news.columns:
            keep_cols.append(col)
    out = news[keep_cols].copy()
    if "earnings_embedding_text" in news.columns:
        enriched_text = news["earnings_embedding_text"].fillna("").astype(str)
        out["text"] = np.where(enriched_text.str.len() > 0, enriched_text, out["text"].fillna("").astype(str))
    out["embedding"] = None
    out["embedding_available"] = 0.0
    if generate_embeddings:
        try:
            vectors = embed_texts_bge(out["text"].fillna("").tolist())
            out["embedding"] = [json.dumps(v.astype(float).tolist()) for v in vectors]
            out["embedding_available"] = 1.0
        except ImportError:
            out["embedding_available"] = 0.0

    if generate_finbert:
        try:
            tone_rows = finbert_scores_batch(out["text"].fillna("").tolist())
            for scores in tone_rows:
                scores["finbert_available"] = 1.0
        except ImportError:
            tone_rows = [
                {
                    "finbert_positive_score": np.nan,
                    "finbert_negative_score": np.nan,
                    "finbert_neutral_score": np.nan,
                    "finbert_available": 0.0,
                }
                for _ in range(len(out))
            ]
    else:
        tone_rows = [
            {
                "finbert_positive_score": np.nan,
                "finbert_negative_score": np.nan,
                "finbert_neutral_score": np.nan,
                "finbert_available": 0.0,
            }
            for _ in range(len(out))
        ]
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
    df["news_cluster_key"] = ""
    if len(valid_idx) >= 2:
        from sklearn.cluster import KMeans

        valid = df.iloc[valid_idx].copy()
        if "catalyst_family" not in valid.columns:
            valid["catalyst_family"] = "all"
        next_cluster_id = 0
        for family, group in valid.groupby("catalyst_family", sort=True):
            group_positions = group.index.tolist()
            if len(group_positions) < 2:
                df.loc[group_positions, "news_cluster_id"] = float(next_cluster_id)
                df.loc[group_positions, "news_cluster_key"] = f"{family}:0"
                next_cluster_id += 1
                continue
            x = np.vstack([vectors[df.index.get_loc(i)] for i in group_positions])
            k = min(max(2, min(int(n_clusters), len(group_positions) // 8 or 2)), len(group_positions))
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
            for local_label in sorted(set(labels)):
                mask = labels == local_label
                idx = list(np.asarray(group_positions)[mask])
                df.loc[idx, "news_cluster_id"] = float(next_cluster_id)
                df.loc[idx, "news_cluster_key"] = f"{family}:{local_label}"
                next_cluster_id += 1
    df.to_parquet(output_path, index=False)
    return df


def label_news_forward_returns(
    news_path: Path | str,
    bars: pd.DataFrame,
    *,
    bars_per_day: int = 13,
    expansion_threshold: float = 0.10,
    max_entry_gap_days: float = 7.0,
    output_path: Path | str = NEWS_LABELS_PATH,
) -> pd.DataFrame:
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    if news.empty:
        out = pd.DataFrame()
        out.to_parquet(output_path, index=False)
        return out
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars["ticker"] = bars["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    bars = bars.sort_values(["ticker", "timestamp"])
    bars_by_ticker: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, ticker_bars in bars.groupby("ticker", sort=False):
        clean_times = (
            ticker_bars["timestamp"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )
        bars_by_ticker[str(ticker)] = (clean_times, ticker_bars["close"].astype(float).to_numpy())
    rows = []
    for rec in news.itertuples(index=False):
        ts = pd.Timestamp(rec.timestamp)
        ticker_data = bars_by_ticker.get(str(rec.ticker).upper())
        if ticker_data is None:
            continue
        times, closes_all = ticker_data
        ts64 = ts.tz_convert("UTC").tz_localize(None).to_datetime64()
        start_idx = int(np.searchsorted(times, ts64, side="right"))
        end_idx = min(start_idx + int(10 * bars_per_day), len(closes_all))
        if start_idx >= end_idx:
            continue
        entry_gap_days = float((times[start_idx] - ts64) / np.timedelta64(1, "D"))
        if entry_gap_days > float(max_entry_gap_days):
            continue
        closes = closes_all[start_idx:end_idx]
        entry = float(closes[0])
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
                "entry_gap_days": entry_gap_days,
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
    scores_path: Path | str = NEWS_SCORES_PATH,
    winner_path: Path | str = WINNER_LIBRARY_PATH,
    loser_path: Path | str = LOSER_LIBRARY_PATH,
    output_path: Path | str = NEWS_FEATURE_MATRIX_PATH,
    label_horizon_days: int = LABEL_HORIZON_DAYS,
) -> pd.DataFrame:
    ensure_data_dirs()
    base = timestamps.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base["ticker"] = base["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    emb = pd.read_parquet(embeddings_path) if Path(embeddings_path).exists() else pd.DataFrame()
    if not news.empty:
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    merged = (
        news.merge(
            emb.drop(columns=["text", "catalyst_family", "catalyst_subtype"], errors="ignore"),
            on=["record_id", "ticker", "timestamp"],
            how="left",
        )
        if not news.empty and not emb.empty
        else news
    )
    scores = pd.read_parquet(scores_path) if Path(scores_path).exists() else pd.DataFrame()
    if not merged.empty and not scores.empty:
        merged = merged.merge(
            scores[
                [
                    "record_id",
                    "ticker",
                    "timestamp",
                    "news_similarity_score",
                    "news_similarity_neighbor_count",
                    "news_similarity_max",
                ]
            ],
            on=["record_id", "ticker", "timestamp"],
            how="left",
        )

    winners = pd.read_parquet(winner_path) if Path(winner_path).exists() else pd.DataFrame()
    losers = pd.read_parquet(loser_path) if Path(loser_path).exists() else pd.DataFrame()
    if not winners.empty and "timestamp" in winners.columns:
        winners["timestamp"] = pd.to_datetime(winners["timestamp"], utc=True, errors="coerce")
    if not losers.empty and "timestamp" in losers.columns:
        losers["timestamp"] = pd.to_datetime(losers["timestamp"], utc=True, errors="coerce")

    def _clean_times(series: pd.Series) -> np.ndarray:
        return series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    def _clean_ts(value: pd.Timestamp) -> np.datetime64:
        return pd.Timestamp(value).tz_convert("UTC").tz_localize(None).to_datetime64()

    horizon = pd.Timedelta(days=int(label_horizon_days))

    def _prepare_library(lib: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (label_ready_times, normalized_matrix) sorted by label_ready_times.

        A record's vector only contributes to a similarity score once its forward-return
        label was already realized. Sorting by ``timestamp + horizon`` and cutting off
        with ``searchsorted(side='right')`` against the prediction timestamp enforces
        that invariant.
        """
        if lib.empty or "timestamp" not in lib.columns or "embedding" not in lib.columns:
            return np.asarray([], dtype="datetime64[ns]"), np.empty((0, 0), dtype=np.float32)
        ordered = lib.copy()
        ordered["__label_ready_ts"] = ordered["timestamp"] + horizon
        ordered = ordered.sort_values("__label_ready_ts")
        vectors = [parse_embedding(value) for value in ordered["embedding"]]
        valid = [(i, vec) for i, vec in enumerate(vectors) if vec is not None]
        if not valid:
            return np.asarray([], dtype="datetime64[ns]"), np.empty((0, 0), dtype=np.float32)
        idx, vecs = zip(*valid)
        mat = np.vstack(vecs).astype(np.float32)
        mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        times = _clean_times(ordered.iloc[list(idx)]["__label_ready_ts"])
        return times, mat

    winner_times, winner_matrix = _prepare_library(winners)
    loser_times, loser_matrix = _prepare_library(losers)
    similarity_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def _library_similarity(vector: np.ndarray | None, cutoff: pd.Timestamp, times: np.ndarray, matrix: np.ndarray) -> float:
        """Max cosine of ``vector`` against library entries whose labels were ready by ``cutoff``.

        ``times`` here are label-ready timestamps (record_ts + horizon), not record
        timestamps, so ``side='right'`` admits entries whose horizon has fully elapsed
        at or before ``cutoff``.
        """
        if vector is None or len(times) == 0 or matrix.size == 0:
            return float("nan")
        end_idx = int(np.searchsorted(times, _clean_ts(cutoff), side="right"))
        if end_idx <= 0:
            return float("nan")
        x = vector.astype(np.float32)
        x = x / max(float(np.linalg.norm(x)), 1e-12)
        return float(np.nanmax(matrix[:end_idx] @ x))

    news_by_ticker: dict[str, dict[str, object]] = {}
    if not merged.empty:
        merged["ticker"] = merged["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
        merged = merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        for ticker, ticker_news in merged.groupby("ticker", sort=False):
            news_by_ticker[str(ticker)] = {
                "frame": ticker_news.reset_index(drop=True),
                "times": _clean_times(ticker_news["timestamp"]),
            }

    rows = []
    for _, base_group in base.sort_values(["ticker", "timestamp"]).groupby("ticker", sort=False):
        ticker = str(base_group.iloc[0]["ticker"]).upper()
        ticker_news_data = news_by_ticker.get(ticker)
        ticker_news = ticker_news_data["frame"] if ticker_news_data else pd.DataFrame()
        news_times = ticker_news_data["times"] if ticker_news_data else np.asarray([], dtype="datetime64[ns]")
        for row in base_group.itertuples(index=False):
            ts = pd.Timestamp(row.timestamp)
            ts64 = _clean_ts(ts)
            start64 = _clean_ts(ts - pd.Timedelta(hours=24))
            start_idx = int(np.searchsorted(news_times, start64, side="left"))
            end_idx = int(np.searchsorted(news_times, ts64, side="left"))
            count = max(end_idx - start_idx, 0)
            latest_row = ticker_news.iloc[end_idx - 1] if count > 0 else None
            if latest_row is not None:
                latest_embedding = parse_embedding(latest_row.get("embedding"))
                latest_ts = pd.Timestamp(latest_row["timestamp"])
                cache_key = (str(latest_row.get("record_id", "")), str(latest_ts))
                if cache_key not in similarity_cache:
                    similarity_cache[cache_key] = (
                        _library_similarity(latest_embedding, latest_ts, winner_times, winner_matrix),
                        _library_similarity(latest_embedding, latest_ts, loser_times, loser_matrix),
                    )
                winner_sim, loser_sim = similarity_cache[cache_key]
                hours_since_news = float((ts - latest_ts).total_seconds() / 3600.0)
            else:
                winner_sim = loser_sim = hours_since_news = np.nan
            rows.append(
                {
                    "timestamp": ts,
                    "ticker": ticker,
                    "news_count_24h": float(count),
                    "direct_news_count_24h": float(ticker_news.iloc[start_idx:end_idx]["is_direct_catalyst"].fillna(0).sum()) if count > 0 and "is_direct_catalyst" in ticker_news.columns else np.nan,
                    "hours_since_news": hours_since_news,
                    "finbert_positive_score": float(latest_row.get("finbert_positive_score", np.nan)) if latest_row is not None else np.nan,
                    "finbert_negative_score": float(latest_row.get("finbert_negative_score", np.nan)) if latest_row is not None else np.nan,
                    "finbert_neutral_score": float(latest_row.get("finbert_neutral_score", np.nan)) if latest_row is not None else np.nan,
                    "news_cluster_id": float(latest_row.get("news_cluster_id", np.nan)) if latest_row is not None else np.nan,
                    "news_relation_confidence": float(latest_row.get("relation_confidence", np.nan)) if latest_row is not None else np.nan,
                    "news_is_direct_catalyst": float(latest_row.get("is_direct_catalyst", np.nan)) if latest_row is not None else np.nan,
                    "news_similarity_score": float(latest_row.get("news_similarity_score", np.nan)) if latest_row is not None else np.nan,
                    "news_similarity_neighbor_count": float(latest_row.get("news_similarity_neighbor_count", np.nan)) if latest_row is not None else np.nan,
                    "news_similarity_max": float(latest_row.get("news_similarity_max", np.nan)) if latest_row is not None else np.nan,
                    "winner_similarity_max": winner_sim,
                    "loser_similarity_max": loser_sim,
                    "news_edge_score": winner_sim - loser_sim if np.isfinite(winner_sim) and np.isfinite(loser_sim) else np.nan,
                    "earnings_catalyst_count_24h": float(ticker_news.iloc[start_idx:end_idx]["is_earnings_catalyst"].fillna(0).sum()) if count > 0 and "is_earnings_catalyst" in ticker_news.columns else np.nan,
                    "earnings_language_score": float(latest_row.get("earnings_language_score", np.nan)) if latest_row is not None else np.nan,
                    "earnings_guidance_score": float(latest_row.get("earnings_guidance_score", np.nan)) if latest_row is not None else np.nan,
                    "earnings_beat_miss_score": float(latest_row.get("earnings_beat_miss_score", np.nan)) if latest_row is not None else np.nan,
                    "earnings_relevance_score": float(latest_row.get("earnings_relevance_score", np.nan)) if latest_row is not None else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    for col in NEWS_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out.to_parquet(output_path, index=False)
    return out
