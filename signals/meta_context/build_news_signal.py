"""Build the per-(ticker, timestamp) news catalyst signal that feeds the
meta-context matrix.

Loads all records from news_records.parquet, runs both the binary expansion
classifier AND the multiclass trajectory classifier on each, then aggregates
to a per-(ticker, snapshot_date) summary with:

  Binary (expansion) outputs:
    news_catalyst_score          max binary score across day
    news_catalyst_score_mean     mean binary score across day
    news_catalyst_count          # of records that day
    news_catalyst_top_family     family of the highest-scoring record
    news_catalyst_top_subtype    subtype of the highest-scoring record

  Trajectory (multiclass) outputs — max P() across the day:
    news_p_bull_steady           "clean breakout" probability
    news_p_bull_volatile         "move with shakeout" probability
    news_p_v_bounce              "spike then fade" probability
    news_p_crash_stayed          "real bear catalyst" probability
    news_p_flat                  "nothing happens" probability

Output: meta_context/data/processed/news_catalyst_signal.parquet

This parquet is one of the inputs to ``meta_context.build_matrix.build_from_paths``
via the ``news_path=`` argument; the meta-ranker (Model 6) consumes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from signals.news.config import NEWS_RECORDS_PATH
from signals.news.live_scorer import CatalystScorer, TRAJECTORY_LABELS
from signals.news.nlp import text_hash


OUTPUT_PATH = Path("signals/meta_context/data/processed/news_catalyst_signal.parquet")
PER_RECORD_PATH = Path("signals/meta_context/data/processed/news_catalyst_per_record.parquet")
OOF_BINARY_PATH = Path("signals/meta_context/data/processed/catalyst_oof_target_expansion_10pct.parquet")
OOF_TRAJECTORY_PATH = Path("signals/meta_context/data/processed/catalyst_oof_target_trajectory_code.parquet")
TRAINING_MATRIX_PATH = Path("signals/meta_context/data/processed/catalyst_training_matrix.parquet")

# Thresholds for the new multiplicity / confluence features
HIGH_SCORE_THRESHOLD = 0.70   # catalyst_score >= this counts as "strong-signal"
BULL_ALIGNMENT_THRESHOLD = 0.50  # p_bull_steady + p_bull_volatile >= this counts as bullish
BEAR_ALIGNMENT_THRESHOLD = 0.30  # p_crash_stayed >= this counts as bearish


def _compute_record_cache_hash(df: pd.DataFrame) -> pd.Series:
    def _series(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    parts = [
        _series("ticker").str.upper(),
        pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else pd.Series([pd.NaT] * len(df), index=df.index), utc=True, errors="coerce").astype(str),
        _series("source"),
        _series("headline"),
        _series("summary"),
        _series("body"),
        _series("earnings_forward_guidance_text"),
    ]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + "||" + part
    return combined.map(text_hash)


def _aggregate_from_per_record_df(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["snapshot_date"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    print(f"  {len(df):,} per-record predictions")

    has_traj = all(f"p_{c}" in df.columns for c in TRAJECTORY_LABELS)
    df["_is_high_score"] = (df["catalyst_score"] >= HIGH_SCORE_THRESHOLD).astype("int8")
    if "source_quality" in df.columns:
        df["_is_breaking"] = (df["source_quality"] == "breaking").astype("int8")
        df["_is_high_alpha"] = df["source_quality"].isin(["breaking", "high_alpha"]).astype("int8")
    else:
        df["_is_breaking"] = 0
        df["_is_high_alpha"] = 0
    if has_traj:
        bull_combined = df["p_bull_steady"].fillna(0) + df["p_bull_volatile"].fillna(0)
        df["_is_bullish"] = (bull_combined >= BULL_ALIGNMENT_THRESHOLD).astype("int8")
        df["_is_bearish"] = (df["p_crash_stayed"].fillna(0) >= BEAR_ALIGNMENT_THRESHOLD).astype("int8")
    else:
        df["_is_bullish"] = 0
        df["_is_bearish"] = 0

    df_sorted = df.sort_values("catalyst_score", ascending=False)
    nr = pd.read_parquet(NEWS_RECORDS_PATH, columns=["record_id", "catalyst_family", "catalyst_subtype"])
    df_sorted = df_sorted.merge(nr, on="record_id", how="left")

    agg_kwargs = dict(
        news_catalyst_score=("catalyst_score", "max"),
        news_catalyst_score_mean=("catalyst_score", "mean"),
        news_catalyst_score_std=("catalyst_score", "std"),
        news_catalyst_count=("catalyst_score", "size"),
        news_catalyst_top_family=("catalyst_family", "first"),
        news_catalyst_top_subtype=("catalyst_subtype", "first"),
        news_unique_sources=("source", "nunique"),
        news_high_score_count=("_is_high_score", "sum"),
        news_high_alpha_count=("_is_high_alpha", "sum"),
        news_breaking_count=("_is_breaking", "sum"),
        news_bull_alignment=("_is_bullish", "mean"),
        news_bear_alignment=("_is_bearish", "mean"),
    )
    if has_traj:
        for col in TRAJECTORY_LABELS:
            agg_kwargs[f"news_p_{col}"] = (f"p_{col}", "max")
            agg_kwargs[f"news_p_{col}_mean"] = (f"p_{col}", "mean")
    agg = df_sorted.groupby(["ticker", "snapshot_date"]).agg(**agg_kwargs).reset_index()

    if (df["_is_high_score"] > 0).any():
        high_score_div = (
            df[df["_is_high_score"] == 1]
            .groupby(["ticker", "snapshot_date"])["source"].nunique()
            .rename("news_high_score_source_diversity").reset_index()
        )
        agg = agg.merge(high_score_div, on=["ticker", "snapshot_date"], how="left")
        agg["news_high_score_source_diversity"] = agg["news_high_score_source_diversity"].fillna(0).astype("int16")
    else:
        agg["news_high_score_source_diversity"] = 0
    agg["news_catalyst_score_std"] = agg["news_catalyst_score_std"].fillna(0.0)

    agg["timestamp"] = pd.to_datetime(agg["snapshot_date"], utc=True)
    keep_cols = [
        "timestamp", "ticker",
        "news_catalyst_score", "news_catalyst_score_mean", "news_catalyst_score_std",
        "news_catalyst_count", "news_catalyst_top_family", "news_catalyst_top_subtype",
        "news_unique_sources", "news_high_score_count", "news_high_score_source_diversity",
        "news_high_alpha_count", "news_breaking_count",
        "news_bull_alignment", "news_bear_alignment",
    ]
    if has_traj:
        for col in TRAJECTORY_LABELS:
            keep_cols += [f"news_p_{col}", f"news_p_{col}_mean"]
    out = agg[keep_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"saved -> {output_path}: {len(out):,} (ticker, day) rows, {len(out.columns)} cols")
    return out


def _load_oof_predictions() -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Load OOF predictions and return them keyed by record_id.

    Returns:
        binary_oof: DataFrame with columns [record_id, oof_pred]
        traj_oof:   DataFrame with columns [record_id, oof_p_<class> for each]
        train_record_ids: set of record_ids that are in the train split
    """
    binary_oof = pd.read_parquet(OOF_BINARY_PATH)[["record_id", "oof_pred"]] \
        if OOF_BINARY_PATH.exists() else pd.DataFrame(columns=["record_id", "oof_pred"])
    if OOF_TRAJECTORY_PATH.exists():
        traj_cols = ["record_id"] + [f"oof_p_{c}" for c in TRAJECTORY_LABELS]
        traj_df = pd.read_parquet(OOF_TRAJECTORY_PATH)
        traj_cols = [c for c in traj_cols if c in traj_df.columns]
        traj_oof = traj_df[traj_cols]
    else:
        traj_oof = pd.DataFrame(columns=["record_id"])

    # The set of train record_ids (per the matrix's split column)
    train_record_ids: set = set()
    if TRAINING_MATRIX_PATH.exists():
        m = pd.read_parquet(TRAINING_MATRIX_PATH, columns=["record_id", "split"])
        train_record_ids = set(m.loc[m["split"] == "train", "record_id"].astype(str))

    return binary_oof, traj_oof, train_record_ids


def build(
    records_path: Path = Path(NEWS_RECORDS_PATH),
    output_path: Path = OUTPUT_PATH,
    *,
    batch_size: int = 1024,
    use_finbert: bool = True,
) -> pd.DataFrame:
    if not records_path.exists():
        raise SystemExit(f"news_records not found at {records_path}")

    print(f"loading {records_path}...")
    df = pd.read_parquet(records_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["_record_cache_hash"] = _compute_record_cache_hash(df)
    print(f"  records to score: {len(df):,}")

    # Load OOF predictions for train-slice records — these are leak-free
    binary_oof, traj_oof, train_record_ids = _load_oof_predictions()
    print(f"OOF predictions loaded: binary={len(binary_oof):,}  trajectory={len(traj_oof):,}")
    print(f"train-slice record_ids: {len(train_record_ids):,}")

    scorer = CatalystScorer(use_finbert=use_finbert)
    has_traj = scorer.trajectory_booster is not None
    print(f"production v3 binary classifier loaded; trajectory classifier loaded: {has_traj}")

    # Split records into two groups:
    #   - "train-slice records": predict via OOF parquet (leak-free)
    #   - "non-train records":   predict via the production v3 model (clean OOS)
    df["record_id_str"] = df["record_id"].astype(str)
    needs_production_score = ~df["record_id_str"].isin(train_record_ids)
    n_oof = (~needs_production_score).sum()
    n_prod = needs_production_score.sum()
    print(f"records to score with production model: {n_prod:,}  (OOF lookup for {n_oof:,} train-slice records)")

    pred_chunks: list[pd.DataFrame] = []
    # Score only the non-train slice with the production model
    prod_df = df[needs_production_score].reset_index(drop=True)
    for start in range(0, len(prod_df), batch_size):
        batch = prod_df.iloc[start:start + batch_size].copy()
        cols = ["ticker", "timestamp", "headline", "summary", "body", "source"]
        if "earnings_forward_guidance_text" in df.columns:
            cols.append("earnings_forward_guidance_text")
        batch_records = batch[cols].to_dict(orient="records")
        pred_chunks.append(scorer.score_both(batch_records))
        if (start // batch_size + 1) % 10 == 0:
            print(f"  scored {start + len(batch):,} / {len(prod_df):,}", flush=True)

    prod_pred = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()

    # Map production predictions back to df indices
    df["catalyst_score"] = np.nan
    for col in TRAJECTORY_LABELS:
        df[f"p_{col}"] = np.nan
    prod_idx = df.index[needs_production_score]
    if not prod_pred.empty:
        df.loc[prod_idx, "catalyst_score"] = prod_pred["catalyst_score"].values
        if has_traj:
            for col in TRAJECTORY_LABELS:
                df.loc[prod_idx, f"p_{col}"] = prod_pred[col].values

    # Now overlay OOF predictions for train-slice records
    if not binary_oof.empty:
        oof_lookup = dict(zip(binary_oof["record_id"].astype(str), binary_oof["oof_pred"]))
        df.loc[~needs_production_score, "catalyst_score"] = df.loc[~needs_production_score, "record_id_str"].map(oof_lookup)
    if has_traj and not traj_oof.empty:
        traj_oof_lookup = traj_oof.set_index(traj_oof["record_id"].astype(str))
        for col in TRAJECTORY_LABELS:
            oof_col = f"oof_p_{col}"
            if oof_col in traj_oof_lookup.columns:
                lookup = traj_oof_lookup[oof_col].to_dict()
                df.loc[~needs_production_score, f"p_{col}"] = df.loc[~needs_production_score, "record_id_str"].map(lookup)

    # Drop records that ended up with NaN catalyst_score (train-slice fold-0 records
    # without OOF predictions, or any other gap)
    n_missing = df["catalyst_score"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing:,} records have no catalyst_score (train-slice fold 0?). Dropping for aggregation.")
        df = df.dropna(subset=["catalyst_score"]).copy()

    # Persist per-record predictions so future re-aggregations don't need to re-score
    per_record_cols = ["record_id", "ticker", "timestamp", "source", "catalyst_score", "_record_cache_hash"]
    if "source_quality" in df.columns:
        per_record_cols.append("source_quality")
    for col in TRAJECTORY_LABELS:
        if f"p_{col}" in df.columns:
            per_record_cols.append(f"p_{col}")
    df[per_record_cols].to_parquet(PER_RECORD_PATH, index=False)
    print(f"saved per-record predictions -> {PER_RECORD_PATH}: {len(df):,} rows")

    print("aggregating to per-(ticker, day)...")
    out = _aggregate_from_per_record_df(df[per_record_cols], output_path)
    print(f"score distribution: min={out['news_catalyst_score'].min():.3f} median={out['news_catalyst_score'].median():.3f} max={out['news_catalyst_score'].max():.3f}")
    if has_traj:
        print("trajectory probability distributions (max-per-day):")
        for col in TRAJECTORY_LABELS:
            c = f"news_p_{col}"
            print(f"  {c:<28} min={out[c].min():.3f}  median={out[c].median():.3f}  max={out[c].max():.3f}")
    return out


def build_incremental(
    records_path: Path = Path(NEWS_RECORDS_PATH),
    output_path: Path = OUTPUT_PATH,
    *,
    per_record_path: Path = PER_RECORD_PATH,
    batch_size: int = 1024,
    use_finbert: bool = True,
) -> pd.DataFrame:
    if not records_path.exists():
        raise SystemExit(f"news_records not found at {records_path}")
    if not per_record_path.exists():
        print("per-record cache missing; falling back to full build")
        return build(records_path=records_path, output_path=output_path, batch_size=batch_size, use_finbert=use_finbert)

    print(f"loading {records_path}...")
    df = pd.read_parquet(records_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["record_id_str"] = df["record_id"].astype(str)
    df["_record_cache_hash"] = _compute_record_cache_hash(df)
    print(f"  records in corpus: {len(df):,}")

    binary_oof, traj_oof, train_record_ids = _load_oof_predictions()
    oof_lookup = dict(zip(binary_oof["record_id"].astype(str), binary_oof["oof_pred"])) if not binary_oof.empty else {}
    traj_lookup = (
        traj_oof.assign(record_id_str=traj_oof["record_id"].astype(str)).set_index("record_id_str")
        if not traj_oof.empty else pd.DataFrame()
    )

    cache = pd.read_parquet(per_record_path)
    cache["record_id_str"] = cache["record_id"].astype(str)
    cache = cache.drop_duplicates("record_id_str", keep="last")

    current_ids = set(df["record_id_str"])
    cache = cache[cache["record_id_str"].isin(current_ids)].copy()

    required_pred_cols = ["catalyst_score", *[f"p_{col}" for col in TRAJECTORY_LABELS]]
    cache_cols_present = set(cache.columns)
    cache_has_hash = "_record_cache_hash" in cache_cols_present
    cache_has_traj = all(f"p_{col}" in cache_cols_present for col in TRAJECTORY_LABELS)
    stale_ids: list[str] = []
    cache_by_id = cache.set_index("record_id_str", drop=False) if not cache.empty else pd.DataFrame()

    for rid, rec_hash in zip(df["record_id_str"], df["_record_cache_hash"]):
        if cache.empty or rid not in cache_by_id.index:
            stale_ids.append(rid)
            continue
        row = cache_by_id.loc[rid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if any(col not in cache_cols_present or pd.isna(row.get(col)) for col in required_pred_cols[:1]):
            stale_ids.append(rid)
            continue
        if not cache_has_traj:
            stale_ids.append(rid)
            continue
        if cache_has_hash and row.get("_record_cache_hash") != rec_hash:
            stale_ids.append(rid)

    print(f"cache refresh needed for {len(stale_ids):,} / {len(df):,} records")

    current_meta_cols = ["record_id_str", "record_id", "ticker", "timestamp", "source", "_record_cache_hash"]
    if "source_quality" in df.columns:
        current_meta_cols.append("source_quality")
    current_meta = df[current_meta_cols].copy()

    retained = current_meta[~current_meta["record_id_str"].isin(stale_ids)].merge(
        cache[["record_id_str", *[col for col in required_pred_cols if col in cache.columns]]],
        on="record_id_str",
        how="inner",
    )

    refresh_df = df[df["record_id_str"].isin(stale_ids)].copy()
    refresh_parts: list[pd.DataFrame] = []

    train_refresh = refresh_df[refresh_df["record_id_str"].isin(train_record_ids)].copy()
    if not train_refresh.empty:
        part = train_refresh[current_meta_cols].copy()
        part["catalyst_score"] = part["record_id_str"].map(oof_lookup)
        for col in TRAJECTORY_LABELS:
            oof_col = f"oof_p_{col}"
            if not traj_lookup.empty and oof_col in traj_lookup.columns:
                part[f"p_{col}"] = part["record_id_str"].map(traj_lookup[oof_col].to_dict())
            else:
                part[f"p_{col}"] = np.nan
        refresh_parts.append(part)

    prod_refresh = refresh_df[~refresh_df["record_id_str"].isin(train_record_ids)].reset_index(drop=True)
    if not prod_refresh.empty:
        scorer = CatalystScorer(use_finbert=use_finbert)
        print(f"scoring {len(prod_refresh):,} uncached production records...")
        for start in range(0, len(prod_refresh), batch_size):
            batch = prod_refresh.iloc[start:start + batch_size].copy()
            cols = ["ticker", "timestamp", "headline", "summary", "body", "source"]
            if "earnings_forward_guidance_text" in prod_refresh.columns:
                cols.append("earnings_forward_guidance_text")
            pred = scorer.score_both(batch[cols].to_dict(orient="records")).rename(
                columns={col: f"p_{col}" for col in TRAJECTORY_LABELS}
            )
            refresh_parts.append(pd.concat([batch[current_meta_cols].reset_index(drop=True), pred], axis=1))
            if (start // batch_size + 1) % 10 == 0:
                print(f"  scored {start + len(batch):,} / {len(prod_refresh):,}", flush=True)

    refreshed = pd.concat(refresh_parts, ignore_index=True) if refresh_parts else pd.DataFrame(columns=retained.columns)
    combined = pd.concat([retained, refreshed], ignore_index=True)
    for col in [f"p_{c}" for c in TRAJECTORY_LABELS]:
        if col not in combined.columns:
            combined[col] = np.nan

    n_missing = combined["catalyst_score"].isna().sum() if "catalyst_score" in combined.columns else len(combined)
    if n_missing:
        print(f"WARNING: {n_missing:,} records still have no catalyst_score after cache refresh. Dropping for aggregation.")
        combined = combined.dropna(subset=["catalyst_score"]).copy()

    out_cols = current_meta_cols[1:] + ["catalyst_score", *[f"p_{col}" for col in TRAJECTORY_LABELS]]
    combined[out_cols].to_parquet(per_record_path, index=False)
    print(f"saved refreshed per-record cache -> {per_record_path}: {len(combined):,} rows")

    print("aggregating to per-(ticker, day)...")
    out = _aggregate_from_per_record_df(combined[out_cols], output_path)
    print(f"score distribution: min={out['news_catalyst_score'].min():.3f} median={out['news_catalyst_score'].median():.3f} max={out['news_catalyst_score'].max():.3f}")
    return out


def reaggregate_from_cache(
    per_record_path: Path = PER_RECORD_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Re-aggregate news_catalyst_signal.parquet from the cached per-record
    predictions parquet — no model scoring needed. Use this whenever you just
    want to add/change aggregation columns without paying the 50-min FinBERT cost.
    """
    if not per_record_path.exists():
        raise SystemExit(f"per-record cache not found at {per_record_path}; run build() first")
    print(f"loading per-record predictions from {per_record_path}")
    df = pd.read_parquet(per_record_path)
    return _aggregate_from_per_record_df(df, output_path)


if __name__ == "__main__":
    import sys as _sys
    if "--from-cache" in _sys.argv:
        reaggregate_from_cache()
    elif "--incremental-cache" in _sys.argv:
        build_incremental()
    else:
        build()
