"""News collection, embedding, clustering, labels, and features."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from signals.news.config import (
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
from signals.news.catalyst_types import classify_catalyst_types, classify_source_quality, refine_catalyst_types_from_clusters
from signals.news.dedup import deduplicate_news
from signals.news.earnings import enrich_earnings_catalyst_fields
from signals.news.nlp import embed_texts_bge, finbert_scores_batch
from signals.news.relations import classify_news_relations
from signals.news.schema import (
    TIMESTAMP_SEMANTICS_VERSION,
    add_observation_metadata,
    empty_news_frame,
    records_from_frame,
    parquet_safe_causal_metadata,
)
from signals.news.sources import (
    enrich_sec_8k_ex99_text,
    fetch_clinicaltrials_updates,
    fetch_fed_press_releases,
    fetch_finnhub_company_news,
    fetch_fmp_earnings_transcripts,
    fetch_google_news_rss,
    fetch_openfda_drug_approvals,
    fetch_sec_8k_news,
    fetch_sec_alpha_filings,
    fetch_yfinance_news,
    fetch_yfinance_unusual_options_activity,
)


def collect_company_news(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    sources: Iterable[str] = ("finnhub", "sec_8k", "sec_alpha", "yfinance", "google_news"),
    sec_include_archives: bool = True,
    sec_full_text_limit: int = 0,
    sec_enrich_ex99: bool = True,
    output_path: Path | str = NEWS_RECORDS_PATH,
    merge_with_existing: bool = True,
    google_news_min_interval_s: float = 0.25,
    google_news_workers: int = 8,
) -> pd.DataFrame:
    """Collect company news across all enabled sources.

    By default merges into the existing ``news_records.parquet`` so multiple
    pulls accumulate rather than overwrite. Set ``merge_with_existing=False``
    for a clean rebuild.
    """
    ensure_data_dirs()
    # One observation timestamp describes the exact collection batch.  It is
    # captured before any source call and is never replaced by event time,
    # publication time, or a file timestamp.
    batch_observed_at = datetime.now(timezone.utc)
    frames = []
    source_set = set(sources)
    if "finnhub" in source_set:
        frames.append(fetch_finnhub_company_news(tickers, start=start, end=end))
    if "sec_8k" in source_set:
        sec_frame = fetch_sec_8k_news(
            tickers,
            start=start,
            end=end,
            include_archives=sec_include_archives,
            full_text_limit=sec_full_text_limit,
        )
        if sec_enrich_ex99 and not sec_frame.empty:
            sec_frame = enrich_sec_8k_ex99_text(sec_frame)
        frames.append(sec_frame)
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
    if "yfinance" in source_set:
        frames.append(fetch_yfinance_news(tickers, start=start, end=end))
    if "google_news" in source_set:
        frames.append(fetch_google_news_rss(
            tickers, start=start, end=end,
            min_interval_s=google_news_min_interval_s, workers=google_news_workers,
        ))
    if "fed_rss" in source_set:
        frames.append(fetch_fed_press_releases(start=start, end=end))
    if "openfda" in source_set:
        frames.append(fetch_openfda_drug_approvals(start=start, end=end))
    if "clinicaltrials" in source_set:
        frames.append(fetch_clinicaltrials_updates(tickers, start=start, end=end))
    if "yf_options_flow" in source_set:
        frames.append(fetch_yfinance_unusual_options_activity(tickers))
    if "fmp_transcripts" in source_set:
        frames.append(fetch_fmp_earnings_transcripts(tickers))
    raw = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in frames
    ) else empty_news_frame()
    if not raw.empty:
        raw = add_observation_metadata(raw, observed_at=batch_observed_at)
    if not raw.empty:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        timestamp_mask = raw["timestamp"].between(start_ts, end_ts)
        # Preserve invalid explicit occurrences for adapter quarantine rather
        # than turning them into absent rows during collection filtering.
        raw = raw.loc[timestamp_mask | raw["timestamp"].isna()].copy()
    if merge_with_existing and Path(output_path).exists():
        existing = pd.read_parquet(output_path)
        if not existing.empty:
            # Keep legacy and causal columns together.  Reusing only the
            # intersection would silently discard metadata from either side.
            raw = pd.concat([existing, raw], ignore_index=True, sort=False)
    out = deduplicate_news(raw)
    out = classify_source_quality(classify_catalyst_types(enrich_earnings_catalyst_fields(classify_news_relations(out))))
    parquet_safe_causal_metadata(out).to_parquet(output_path, index=False)
    return out


def collect_news_from_csv(
    input_csv: Path | str,
    *,
    output_path: Path | str = NEWS_RECORDS_PATH,
    observed_at: datetime | str | None = None,
    collection_time: datetime | str | None = None,
) -> pd.DataFrame:
    """Normalize CSV news while retaining unknown availability explicitly.

    A caller may provide one aware collection time for the batch.  Without
    it, the output keeps ``observed_at``/``available_at`` null so the catalyst
    adapter can quarantine the row instead of inventing causality.
    """

    ensure_data_dirs()
    out = classify_source_quality(
        classify_catalyst_types(
            enrich_earnings_catalyst_fields(
                classify_news_relations(
                    deduplicate_news(
                        records_from_frame(
                            pd.read_csv(input_csv),
                            source="csv",
                            observed_at=observed_at,
                            collection_time=collection_time,
                        )
                    )
                )
            )
        )
    )
    parquet_safe_causal_metadata(out).to_parquet(output_path, index=False)
    return out


def classify_existing_news(
    news_path: Path | str = NEWS_RECORDS_PATH,
    *,
    output_path: Path | str = NEWS_RECORDS_PATH,
) -> pd.DataFrame:
    ensure_data_dirs()
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    out = classify_source_quality(classify_catalyst_types(enrich_earnings_catalyst_fields(classify_news_relations(news))))
    out.to_parquet(output_path, index=False)
    return out


def _as_utc_timestamps(values: pd.Series, *, label: str) -> pd.Series:
    """Converge a causal-metadata column on tz-aware datetimes.

    ``news_records.parquet`` persists these columns as ISO strings — see
    ``schema.parquet_safe_causal_metadata``, which serializes them so naive
    values are not silently localized. The embeddings artifact has always
    stored real datetimes. The incremental append below concatenates the two,
    and an unaligned pair yields an object column holding both ``str`` and
    ``Timestamp``; pyarrow then infers int64 from the pandas metadata and
    fails the write outright. Coerce here rather than at the write so the
    mismatch cannot reach the parquet layer.

    Unparseable values raise instead of becoming NaT: a timestamp that cannot
    be read is a collection defect, and silently nulling it would drop the
    record from every downstream time-windowed join without a trace.
    """
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    newly_null = int(converted.isna().sum() - pd.isna(values).sum())
    if newly_null > 0:
        raise ValueError(
            f"{label}: {newly_null:,} timestamp value(s) are unparseable; "
            "refusing to null them silently"
        )
    return converted


# The embeddings artifact stores each 384-float vector as a JSON string (~8.5 KB
# a record), so loading it into pandas costs ~8 GB of Python strings plus the
# Arrow buffers on read and again on write. The incremental pass used to do that
# in three of its four steps and rewrite the file twice: it peaked at 18.4 GB of
# a 19.5 GB box, and two nightly runs (2026-10-12 21:41, 10-16 06:43) died inside
# it. The helpers below keep the incremental steps to a light column read plus
# batched passes, measured at ~0.26 GB and ~0.8 GB on the 951k-row file.
_STREAM_BATCH_ROWS = 20_000


def _iter_parquet_batches(path: Path | str, *, columns: list[str] | None = None):
    pf = pq.ParquetFile(path, buffer_size=8 << 20, pre_buffer=False)
    yield from pf.iter_batches(batch_size=_STREAM_BATCH_ROWS, columns=columns)


def _rewrite_parquet_in_batches(
    path: Path | str,
    *,
    edit: Callable[[pa.Table], pa.Table] | None = None,
    tail: pa.Table | None = None,
    expected_rows: int | None = None,
) -> int:
    """Rewrite ``path`` one batch at a time with its own schema, then swap it in.

    ``edit`` may modify or filter each batch; ``tail`` is appended after the
    last one. The replacement is atomic and happens only after the row count
    matches ``expected_rows``, so a crash or a miscount leaves the old file.
    """
    path = Path(path)
    schema = pq.read_schema(path)
    tmp = path.with_name(path.name + ".tmp")
    rows = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression="snappy") as writer:
            for batch in _iter_parquet_batches(path):
                table = pa.Table.from_batches([batch]).replace_schema_metadata(schema.metadata)
                if edit is not None:
                    table = edit(table)
                if table.num_rows:
                    writer.write_table(table)
                    rows += table.num_rows
            if tail is not None and tail.num_rows:
                writer.write_table(tail.replace_schema_metadata(schema.metadata))
                rows += tail.num_rows
        if expected_rows is not None and rows != expected_rows:
            raise RuntimeError(f"{path.name}: batched rewrite produced {rows:,} rows, expected {expected_rows:,}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return rows


def _frame_to_schema(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """``frame`` as a table with exactly ``schema``; absent columns become nulls."""
    arrays = []
    for field in schema:
        if field.name in frame.columns:
            arrays.append(pa.array(frame[field.name], type=field.type, from_pandas=True))
        else:
            arrays.append(pa.nulls(len(frame), type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16] if text else ""


def _embed_rows(to_embed: pd.DataFrame, *, generate_embeddings: bool, generate_finbert: bool) -> pd.DataFrame:
    """BGE vectors (as JSON strings), FinBERT tone and the text hash for ``to_embed``."""
    to_embed = to_embed.reset_index(drop=True)
    to_embed["embedding"] = None
    to_embed["embedding_available"] = 0.0
    if generate_embeddings:
        try:
            vectors = embed_texts_bge(to_embed["text"].fillna("").tolist())
            to_embed["embedding"] = [json.dumps(v.astype(float).tolist()) for v in vectors]
            to_embed["embedding_available"] = 1.0
        except ImportError:
            to_embed["embedding_available"] = 0.0

    no_tone = {
        "finbert_positive_score": np.nan,
        "finbert_negative_score": np.nan,
        "finbert_neutral_score": np.nan,
        "finbert_available": 0.0,
    }
    if generate_finbert:
        try:
            tone_rows = finbert_scores_batch(to_embed["text"].fillna("").tolist())
            for scores in tone_rows:
                scores["finbert_available"] = 1.0
        except ImportError:
            tone_rows = [dict(no_tone) for _ in range(len(to_embed))]
    else:
        tone_rows = [dict(no_tone) for _ in range(len(to_embed))]
    to_embed = pd.concat([to_embed, pd.DataFrame(tone_rows)], axis=1)
    # Ensure _text_hash is persisted so future incremental runs can detect changes
    if "_text_hash" not in to_embed.columns:
        to_embed["_text_hash"] = to_embed["text"].fillna("").astype(str).apply(_text_hash)
    return to_embed


def _changed_or_new(full: pd.DataFrame, prior_ids: pd.Series, prior_hashes: dict[str, str]) -> pd.DataFrame:
    """Rows of ``full`` that are new or whose text hash moved, with the usual log line."""
    full = full.copy()
    full["_text_hash"] = full["text"].fillna("").astype(str).apply(_text_hash)
    current_hashes = dict(zip(full["record_id"].astype(str), full["_text_hash"]))
    changed_or_new = [rid for rid, h in current_hashes.items() if prior_hashes.get(rid) != h]
    to_embed = full[full["record_id"].astype(str).isin(set(changed_or_new))].copy()
    new_count = sum(1 for rid in changed_or_new if rid not in prior_hashes)
    print(
        f"  incremental: {len(prior_ids):,} prior, {len(to_embed):,} to embed "
        f"({new_count:,} new + {len(changed_or_new) - new_count:,} text changed)"
    )
    return to_embed


def _prior_hashes(prior: pd.DataFrame) -> dict[str, str]:
    ids = prior["record_id"].astype(str)
    if "_text_hash" in prior.columns:
        return dict(zip(ids, prior["_text_hash"].fillna("").astype(str)))
    # First migration — derive prior hashes from cached text column if present
    if "text" in prior.columns:
        return dict(zip(ids, prior["text"].fillna("").astype(str).apply(_text_hash)))
    return {rid: "<unknown>" for rid in ids}


def _can_stream(path: Path | str, required: set[str]) -> bool:
    try:
        schema = pq.read_schema(path)
    except Exception:
        return False
    if not required <= set(schema.names):
        return False
    if "timestamp" in schema.names:
        ts_type = schema.field("timestamp").type
        if not (pa.types.is_timestamp(ts_type) and ts_type.tz is not None):
            return False  # the full-load path normalizes legacy string timestamps
    return True


def build_news_embeddings(
    news_path: Path | str = NEWS_RECORDS_PATH,
    *,
    output_path: Path | str = NEWS_EMBEDDINGS_PATH,
    generate_embeddings: bool = True,
    generate_finbert: bool = True,
    incremental: bool = True,
) -> pd.DataFrame:
    """Compute BGE + FinBERT for news_records.

    With ``incremental=True`` (default), only records whose ``record_id`` is
    not yet present in the existing embeddings parquet, or whose text changed,
    are embedded; they are appended to the saved file and returned. The prior
    embeddings are never loaded: only their ids and text hashes are read, and
    the append is a batched rewrite (nothing is written when nothing is new).
    With ``incremental=False`` the full parquet is rewritten from scratch and
    returned (the legacy behavior).
    """
    ensure_data_dirs()
    if Path(news_path).exists():
        # Only these feed the embedding text and row identity; the other ~30 columns are not needed.
        wanted = ("record_id", "ticker", "timestamp", "text", "catalyst_family", "catalyst_subtype", "earnings_embedding_text")
        news = pd.read_parquet(news_path, columns=[c for c in wanted if c in pq.read_schema(news_path).names])
    else:
        news = empty_news_frame()
    if news.empty:
        out = pd.DataFrame(columns=["record_id", "embedding", "embedding_available", "finbert_available"])
        out.to_parquet(output_path, index=False)
        return out

    keep_cols = ["record_id", "ticker", "timestamp", "text"]
    for col in ("catalyst_family", "catalyst_subtype"):
        if col in news.columns:
            keep_cols.append(col)
    full = news[keep_cols].copy()
    full["timestamp"] = _as_utc_timestamps(full["timestamp"], label="news_records")
    if "earnings_embedding_text" in news.columns:
        enriched_text = news["earnings_embedding_text"].fillna("").astype(str)
        full["text"] = np.where(enriched_text.str.len() > 0, enriched_text, full["text"].fillna("").astype(str))
    del news

    if incremental and Path(output_path).exists() and _can_stream(output_path, {"record_id"}):
        names = pq.read_schema(output_path).names
        light_cols = ["record_id"] + [c for c in ("_text_hash",) if c in names]
        if "_text_hash" not in names and "text" in names:
            light_cols.append("text")
        prior = pq.read_table(output_path, columns=light_cols).to_pandas()
        if prior["record_id"].astype(str).is_unique:
            return _append_embeddings_streaming(
                full, prior, output_path=output_path,
                generate_embeddings=generate_embeddings, generate_finbert=generate_finbert,
            )

    # Full-load path: first build, legacy layouts, or a prior file with
    # duplicate ids (which the concat + drop_duplicates below also repairs).
    prior = pd.DataFrame()
    to_embed = full
    if incremental and Path(output_path).exists():
        try:
            prior = pd.read_parquet(output_path)
        except Exception:
            prior = pd.DataFrame()
        if not prior.empty and "timestamp" in prior.columns:
            prior["timestamp"] = _as_utc_timestamps(prior["timestamp"], label="news_embeddings")
        if not prior.empty and "record_id" in prior.columns:
            to_embed = _changed_or_new(full, prior["record_id"], _prior_hashes(prior))

    if to_embed.empty:
        print("  nothing new to embed — re-saving prior parquet")
        prior.to_parquet(output_path, index=False)
        return prior

    to_embed = _embed_rows(to_embed, generate_embeddings=generate_embeddings, generate_finbert=generate_finbert)

    if not prior.empty:
        # Align columns; new rows will inherit NaN for cluster fields until refit/predict
        shared_cols = list(set(prior.columns) | set(to_embed.columns))
        for col in shared_cols:
            if col not in to_embed.columns:
                to_embed[col] = np.nan
            if col not in prior.columns:
                prior[col] = np.nan
        out = pd.concat([prior, to_embed[prior.columns]], ignore_index=True)
        out = out.drop_duplicates(subset=["record_id"], keep="last")
    else:
        out = to_embed
    out.to_parquet(output_path, index=False)
    return out


def _append_embeddings_streaming(
    full: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    output_path: Path | str,
    generate_embeddings: bool,
    generate_finbert: bool,
) -> pd.DataFrame:
    """The incremental append without loading the prior vectors (see above)."""
    to_embed = _changed_or_new(full, prior["record_id"], _prior_hashes(prior))
    if to_embed.empty:
        print("  nothing new to embed — prior parquet unchanged")
        return to_embed
    to_embed = _embed_rows(to_embed, generate_embeddings=generate_embeddings, generate_finbert=generate_finbert)
    to_embed = to_embed.drop_duplicates(subset=["record_id"], keep="last").reset_index(drop=True)

    schema = pq.read_schema(output_path)
    extra = [c for c in to_embed.columns if c not in schema.names]
    if extra:
        # A new column needs a schema change; take the full-load route once.
        prior_full = pd.read_parquet(output_path)
        for col in set(prior_full.columns) | set(to_embed.columns):
            if col not in to_embed.columns:
                to_embed[col] = np.nan
            if col not in prior_full.columns:
                prior_full[col] = np.nan
        out = pd.concat([prior_full, to_embed[prior_full.columns]], ignore_index=True)
        out.drop_duplicates(subset=["record_id"], keep="last").to_parquet(output_path, index=False)
        return to_embed

    replaced = pa.array(sorted(set(to_embed["record_id"].astype(str)) & set(prior["record_id"].astype(str))), type=pa.string())

    def _drop_replaced(table: pa.Table) -> pa.Table:
        if len(replaced) == 0:
            return table
        return table.filter(pc.invert(pc.is_in(table["record_id"], value_set=replaced)))

    _rewrite_parquet_in_batches(
        output_path, edit=_drop_replaced, tail=_frame_to_schema(to_embed, schema),
        expected_rows=len(prior) - len(replaced) + len(to_embed),
    )
    return to_embed


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


KMEANS_MODELS_PATH = Path(__file__).resolve().parent / "data" / "processed" / "kmeans_per_family.pkl"


def cluster_news_embeddings(
    embeddings_path: Path | str = NEWS_EMBEDDINGS_PATH,
    *,
    output_path: Path | str = NEWS_EMBEDDINGS_PATH,
    n_clusters: int = 12,
    incremental: bool = True,
    models_path: Path | str = KMEANS_MODELS_PATH,
) -> pd.DataFrame:
    """Cluster BGE embeddings within each catalyst_family.

    With ``incremental=True`` (default), load the saved per-family KMeans
    models and call ``.predict()`` for records that don't yet have a
    cluster_id. Falls back to a full ``.fit_predict()`` if no saved models
    exist or ``incremental=False`` is passed (the legacy weekly refit path).

    Saved models live at ``models_path`` as a pickled dict
    ``{family: (KMeans, base_cluster_id)}``.
    """
    import pickle

    if (
        incremental
        and Path(models_path).exists()
        and Path(embeddings_path) == Path(output_path)
        and _can_stream(embeddings_path, {"record_id", "embedding", "news_cluster_id"})
    ):
        try:
            with open(models_path, "rb") as fh:
                saved_models = pickle.load(fh)
        except Exception:
            saved_models = {}
        if saved_models:
            return _assign_clusters_streaming(embeddings_path, saved_models)

    df = pd.read_parquet(embeddings_path) if Path(embeddings_path).exists() else pd.DataFrame()
    if df.empty or "embedding" not in df.columns:
        df["news_cluster_id"] = np.nan
        df.to_parquet(output_path, index=False)
        return df

    vectors = [parse_embedding(v) for v in df["embedding"]]
    valid_idx = [i for i, v in enumerate(vectors) if v is not None]
    if "news_cluster_id" not in df.columns:
        df["news_cluster_id"] = np.nan
    if "news_cluster_key" not in df.columns:
        df["news_cluster_key"] = ""

    if len(valid_idx) < 2:
        df.to_parquet(output_path, index=False)
        return df

    from sklearn.cluster import KMeans

    valid = df.iloc[valid_idx].copy()
    if "catalyst_family" not in valid.columns:
        valid["catalyst_family"] = "all"

    models_path = Path(models_path)
    saved_models: dict[str, tuple] = {}
    if incremental and models_path.exists():
        try:
            with open(models_path, "rb") as fh:
                saved_models = pickle.load(fh)
        except Exception:
            saved_models = {}

    needs_full_refit = (not incremental) or not saved_models
    if needs_full_refit:
        # Full refit path — re-fit per family from scratch and re-save models
        models_path.parent.mkdir(parents=True, exist_ok=True)
        new_models: dict[str, tuple] = {}
        next_cluster_id = 0
        for family, group in valid.groupby("catalyst_family", sort=True):
            group_positions = group.index.tolist()
            if len(group_positions) < 2:
                df.loc[group_positions, "news_cluster_id"] = float(next_cluster_id)
                df.loc[group_positions, "news_cluster_key"] = f"{family}:0"
                new_models[str(family)] = (None, next_cluster_id)
                next_cluster_id += 1
                continue
            x = np.vstack([vectors[df.index.get_loc(i)] for i in group_positions])
            k = min(max(2, min(int(n_clusters), len(group_positions) // 8 or 2)), len(group_positions))
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
            labels = km.labels_
            base_id = next_cluster_id
            new_models[str(family)] = (km, base_id)
            for local_label in sorted(set(labels)):
                mask = labels == local_label
                idx = list(np.asarray(group_positions)[mask])
                df.loc[idx, "news_cluster_id"] = float(base_id + int(local_label))
                df.loc[idx, "news_cluster_key"] = f"{family}:{local_label}"
            next_cluster_id = base_id + k
        with open(models_path, "wb") as fh:
            pickle.dump(new_models, fh)
        print(f"  cluster: full refit, {len(new_models)} family models saved -> {models_path}")
    else:
        # Incremental path — use saved models to predict for rows missing cluster_id
        unassigned_mask = df["news_cluster_id"].isna() & df.index.isin(valid_idx)
        unassigned_count = int(unassigned_mask.sum())
        if unassigned_count == 0:
            print("  cluster: no unassigned rows, nothing to do")
        else:
            print(f"  cluster: {unassigned_count:,} rows need assignment via saved KMeans")
            for family, group in valid[unassigned_mask.loc[valid.index]].groupby("catalyst_family", sort=True):
                group_positions = group.index.tolist()
                if family not in saved_models:
                    # Family seen for the first time post-refit — assign a placeholder cluster
                    df.loc[group_positions, "news_cluster_id"] = -1.0
                    df.loc[group_positions, "news_cluster_key"] = f"{family}:unassigned"
                    continue
                km, base_id = saved_models[family]
                if km is None:
                    df.loc[group_positions, "news_cluster_id"] = float(base_id)
                    df.loc[group_positions, "news_cluster_key"] = f"{family}:0"
                    continue
                x = np.vstack([vectors[df.index.get_loc(i)] for i in group_positions])
                labels = km.predict(x)
                for local_label in sorted(set(labels)):
                    mask = labels == local_label
                    idx = list(np.asarray(group_positions)[mask])
                    df.loc[idx, "news_cluster_id"] = float(int(base_id) + int(local_label))
                    df.loc[idx, "news_cluster_key"] = f"{family}:{local_label}"

    df.to_parquet(output_path, index=False)
    return df


def _assign_clusters_streaming(embeddings_path: Path | str, saved_models: dict[str, tuple]) -> pd.DataFrame:
    """Incremental cluster assignment that never loads the whole embeddings file.

    Same rules as the in-memory incremental path: only rows with a missing
    ``news_cluster_id`` and a parseable embedding are assigned, grouped by
    ``catalyst_family`` (rows with no family stay unassigned, as ``groupby``
    drops them); a family absent from the saved models gets the ``-1``
    placeholder; a single-row family model maps to its base id. Nothing is
    rewritten when nothing is assigned. Returns the assigned rows' ids.
    """
    names = pq.read_schema(embeddings_path).names
    has_family = "catalyst_family" in names
    light = pq.read_table(embeddings_path, columns=["record_id", "news_cluster_id"]).to_pandas()
    pending = set(light.loc[light["news_cluster_id"].isna(), "record_id"].astype(str))
    if not pending:
        print("  cluster: no unassigned rows, nothing to do")
        return pd.DataFrame(columns=["record_id", "news_cluster_id", "news_cluster_key"])

    # Pass 1: pull only the pending rows' vectors.
    wanted = pa.array(sorted(pending), type=pa.string())
    cols = ["record_id", "embedding"] + (["catalyst_family"] if has_family else [])
    ids: list[str] = []
    families: list[object] = []
    vectors: list[np.ndarray] = []
    for batch in _iter_parquet_batches(embeddings_path, columns=cols):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["record_id"], value_set=wanted))
        if not table.num_rows:
            continue
        frame = table.to_pandas()
        for i, rid in enumerate(frame["record_id"].astype(str)):
            vector = parse_embedding(frame["embedding"].iat[i])
            if vector is None:
                continue
            ids.append(rid)
            families.append(frame["catalyst_family"].iat[i] if has_family else "all")
            vectors.append(vector)

    assigned: dict[str, tuple[float, str]] = {}
    if ids:
        frame = pd.DataFrame({"record_id": ids, "catalyst_family": families, "pos": range(len(ids))})
        print(f"  cluster: {len(ids):,} rows need assignment via saved KMeans")
        for family, group in frame.groupby("catalyst_family", sort=True):
            group_ids = group["record_id"].tolist()
            if family not in saved_models:
                # Family seen for the first time post-refit — assign a placeholder cluster
                assigned.update({rid: (-1.0, f"{family}:unassigned") for rid in group_ids})
                continue
            km, base_id = saved_models[family]
            if km is None:
                assigned.update({rid: (float(base_id), f"{family}:0") for rid in group_ids})
                continue
            labels = km.predict(np.vstack([vectors[i] for i in group["pos"]]))
            assigned.update({
                rid: (float(int(base_id) + int(label)), f"{family}:{label}")
                for rid, label in zip(group_ids, labels)
            })
    if not assigned:
        print("  cluster: no unassigned rows, nothing to do")
        return pd.DataFrame(columns=["record_id", "news_cluster_id", "news_cluster_key"])

    # Pass 2: write the assignments, touching only batches that hold one.
    keys = pa.array(sorted(assigned), type=pa.string())
    schema = pq.read_schema(embeddings_path)

    def _apply(table: pa.Table) -> pa.Table:
        if not pc.any(pc.is_in(table["record_id"], value_set=keys)).as_py():
            return table
        frame = table.to_pandas()
        hit = frame["record_id"].astype(str).map(assigned)
        mask = hit.notna()
        frame.loc[mask, "news_cluster_id"] = [v[0] for v in hit[mask]]
        if "news_cluster_key" in frame.columns:
            frame.loc[mask, "news_cluster_key"] = [v[1] for v in hit[mask]]
        return _frame_to_schema(frame, schema)

    _rewrite_parquet_in_batches(embeddings_path, edit=_apply, expected_rows=len(light))
    return pd.DataFrame(
        [(rid, cid, key) for rid, (cid, key) in assigned.items()],
        columns=["record_id", "news_cluster_id", "news_cluster_key"],
    )


def refine_news_records_from_clusters(
    news_path: Path | str = NEWS_RECORDS_PATH,
    embeddings_path: Path | str = NEWS_EMBEDDINGS_PATH,
    *,
    output_path: Path | str = NEWS_RECORDS_PATH,
    cluster_label_threshold: float = 0.35,
    min_cluster_size: int = 4,
) -> pd.DataFrame:
    """Join cluster ids from embeddings parquet onto news_records and re-derive
    catalyst_family/catalyst_subtype from cluster modal tags.

    This is the post-clustering refinement step that lets a cluster of
    options-flow alerts override the regex-assigned ``earnings_guidance``
    family even though the headlines mention ``earnings``.
    """
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    if news.empty:
        news.to_parquet(output_path, index=False)
        return news
    emb = pd.DataFrame()
    if Path(embeddings_path).exists():
        # Only the id and cluster columns are joined; the vectors are never needed.
        wanted = [c for c in ("record_id", "news_cluster_id", "news_cluster_key") if c in pq.read_schema(embeddings_path).names]
        emb = pd.read_parquet(embeddings_path, columns=wanted)
    if emb.empty or "news_cluster_id" not in emb.columns:
        news.to_parquet(output_path, index=False)
        return news
    keep = [c for c in ("record_id", "news_cluster_id", "news_cluster_key") if c in emb.columns]
    news = news.drop(columns=[c for c in ("news_cluster_id", "news_cluster_key") if c in news.columns])
    news = news.merge(emb[keep], on="record_id", how="left")
    news = refine_catalyst_types_from_clusters(
        news,
        cluster_label_threshold=cluster_label_threshold,
        min_cluster_size=min_cluster_size,
    )
    news.to_parquet(output_path, index=False)
    return news


def _detect_bars_per_day(bars: pd.DataFrame) -> int:
    """Infer bars-per-trading-day from the median intra-ticker interval.

    A 4-hour bar at ~6.5h/day trading yields ~2 bars/day; a 30-minute bar
    yields ~13. Daily bars yield 1.
    """
    if bars.empty or "timestamp" not in bars.columns or "ticker" not in bars.columns:
        return 0
    sample = bars.sort_values(["ticker", "timestamp"]).head(50000).copy()
    sample["timestamp"] = pd.to_datetime(sample["timestamp"], utc=True, errors="coerce")
    sample = sample.dropna(subset=["timestamp"])
    deltas = sample.groupby("ticker")["timestamp"].diff().dropna()
    if deltas.empty:
        return 0
    intraday = deltas[(deltas > pd.Timedelta(seconds=0)) & (deltas < pd.Timedelta(hours=12))]
    if intraday.empty:
        return 1  # daily bars
    median_seconds = float(intraday.median().total_seconds())
    if median_seconds <= 0:
        return 0
    trading_seconds_per_day = 6.5 * 3600
    return max(1, int(round(trading_seconds_per_day / median_seconds)))


def label_news_forward_returns(
    news_path: Path | str,
    bars: pd.DataFrame,
    *,
    bars_per_day: int | None = None,
    expansion_threshold: float = 0.10,
    max_entry_gap_days: float = 7.0,
    output_path: Path | str = NEWS_LABELS_PATH,
    incremental: bool = True,
) -> pd.DataFrame:
    """Label forward returns for news records.

    With ``incremental=True`` (default), skip records that already have a
    label row (by record_id) in the existing news_labels parquet. Appends
    any newly-computed labels for records whose forward window has matured.
    """
    news = pd.read_parquet(news_path) if Path(news_path).exists() else empty_news_frame()
    if news.empty:
        out = pd.DataFrame()
        out.to_parquet(output_path, index=False)
        return out

    prior_labels = pd.DataFrame()
    if incremental and Path(output_path).exists():
        try:
            prior_labels = pd.read_parquet(output_path)
        except Exception:
            prior_labels = pd.DataFrame()
        if not prior_labels.empty and "record_id" in prior_labels.columns:
            already = set(prior_labels["record_id"].dropna().astype(str))
            news = news[~news["record_id"].astype(str).isin(already)].copy()
            print(f"  incremental label: {len(prior_labels):,} already labeled, {len(news):,} new to label")
        if news.empty:
            print("  nothing new to label")
            prior_labels.to_parquet(output_path, index=False)
            return prior_labels

    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars["ticker"] = bars["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    bars = bars.sort_values(["ticker", "timestamp"])
    if bars_per_day is None or int(bars_per_day) <= 0:
        detected = _detect_bars_per_day(bars)
        if detected > 0:
            bars_per_day = detected
        else:
            bars_per_day = 13  # legacy default
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
    if not prior_labels.empty:
        out = pd.concat([prior_labels, out], ignore_index=True)
        out = out.drop_duplicates(subset=["record_id"], keep="last")
    out.to_parquet(output_path, index=False)
    return out


def build_winner_loser_libraries(
    embeddings_path: Path | str = NEWS_EMBEDDINGS_PATH,
    labels_path: Path | str = NEWS_LABELS_PATH,
    *,
    winner_path: Path | str = WINNER_LIBRARY_PATH,
    loser_path: Path | str = LOSER_LIBRARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["record_id", "ticker", "timestamp"]
    labels = pd.read_parquet(labels_path) if Path(labels_path).exists() else pd.DataFrame()
    if (
        not labels.empty
        and Path(embeddings_path).exists()
        and _can_stream(embeddings_path, set(keys))
        and set(keys + ["expansion_label"]) <= set(labels.columns)
    ):
        streamed = _build_libraries_streaming(embeddings_path, labels, keys=keys,
                                              winner_path=winner_path, loser_path=loser_path)
        if streamed is not None:
            return streamed

    emb = pd.read_parquet(embeddings_path) if Path(embeddings_path).exists() else pd.DataFrame()
    if emb.empty or labels.empty:
        winners = pd.DataFrame()
        losers = pd.DataFrame()
    else:
        lib = emb.merge(labels, on=keys, how="inner")
        winners = lib.loc[lib["expansion_label"].eq(1.0)].reset_index(drop=True)
        losers = lib.loc[lib["expansion_label"].eq(0.0)].reset_index(drop=True)
    winners.to_parquet(winner_path, index=False)
    losers.to_parquet(loser_path, index=False)
    return winners, losers


def _build_libraries_streaming(
    embeddings_path: Path | str,
    labels: pd.DataFrame,
    *,
    keys: list[str],
    winner_path: Path | str,
    loser_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Winner/loser libraries one embeddings batch at a time.

    An inner merge keeps the left frame's order, so merging batch by batch and
    appending gives the same rows in the same order as one whole-file merge,
    without holding ~770k embedding strings at once. Returns light frames (ids,
    timestamps and labels, no vectors); the libraries on disk are complete.
    Returns None when the label columns would collide with embedding columns,
    leaving that rare layout to the full-load path.
    """
    emb_schema = pq.read_schema(embeddings_path)
    label_extra = [c for c in labels.columns if c not in keys]
    if any(c in emb_schema.names for c in label_extra):
        return None
    label_schema = pa.Schema.from_pandas(labels[label_extra], preserve_index=False)
    out_schema = pa.schema(list(emb_schema) + list(label_schema))

    outputs = {}
    tmp_paths = {}
    light: dict[str, list[pd.DataFrame]] = {"winner": [], "loser": []}
    try:
        for name, path in (("winner", Path(winner_path)), ("loser", Path(loser_path))):
            tmp_paths[name] = path.with_name(path.name + ".tmp")
            outputs[name] = pq.ParquetWriter(tmp_paths[name], out_schema, compression="snappy")
        for batch in _iter_parquet_batches(embeddings_path):
            frame = batch.to_pandas()
            lib = frame.merge(labels, on=keys, how="inner")
            if lib.empty:
                continue
            for name, value in (("winner", 1.0), ("loser", 0.0)):
                part = lib.loc[lib["expansion_label"].eq(value)]
                if len(part):
                    outputs[name].write_table(_frame_to_schema(part, out_schema))
                    light[name].append(part[keys + label_extra].copy())
        for writer in outputs.values():
            writer.close()
        os.replace(tmp_paths["winner"], winner_path)
        os.replace(tmp_paths["loser"], loser_path)
    finally:
        for writer in outputs.values():
            if writer.is_open:
                writer.close()
        for path in tmp_paths.values():
            path.unlink(missing_ok=True)

    def _light(name: str) -> pd.DataFrame:
        parts = light[name]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keys + label_extra)

    return _light("winner"), _light("loser")


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
