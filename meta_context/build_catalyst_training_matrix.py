"""Build the training feature matrix for the catalyst-classifier (model 2).

Joins:
- winner / loser libraries (BGE embedding + FinBERT scores + label)
- ticker_profiles (sector, marketCap, beta, float)
- finra_short_volume (short_ratio z-score at event time)
- cboe_options_summary (iv30, put_call_volume_ratio at event time)

Output: meta_context/data/processed/catalyst_training_matrix.parquet
        — one row per catalyst record, 384 BGE dims + ~70 hand features +
        time-series split assignment + expansion_label target.

Run: .venv/bin/python -m meta_context.build_catalyst_training_matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news.config import (
    CBOE_OPTIONS_SUMMARY_PATH,
    FINRA_SHORT_VOLUME_PATH,
    LOSER_LIBRARY_PATH,
    TICKER_PROFILE_PATH,
    WINNER_LIBRARY_PATH,
)


OUTPUT_PATH = Path("meta_context/data/processed/catalyst_training_matrix.parquet")

# Train through 2025-09; validate 2025-10 to 2026-02; test 2026-03+
TRAIN_END = pd.Timestamp("2025-10-01", tz="UTC")
VAL_END = pd.Timestamp("2026-03-01", tz="UTC")


def _parse_embedding(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    try:
        return np.asarray(json.loads(str(value)), dtype=np.float32)
    except Exception:
        return np.zeros(384, dtype=np.float32)


def _assign_split(ts: pd.Timestamp) -> str:
    if ts < TRAIN_END:
        return "train"
    if ts < VAL_END:
        return "val"
    return "test"


def _onehot_top_k(series: pd.Series, k: int, prefix: str) -> pd.DataFrame:
    top = series.value_counts().head(k).index.tolist()
    out = {f"{prefix}_{v}": (series == v).astype(np.float32) for v in top}
    out[f"{prefix}_other"] = (~series.isin(top)).astype(np.float32)
    return pd.DataFrame(out)


def build() -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("loading winner/loser libraries...")
    w = pd.read_parquet(WINNER_LIBRARY_PATH).assign(_lbl=1)
    l = pd.read_parquet(LOSER_LIBRARY_PATH).assign(_lbl=0)
    lib = pd.concat([w, l], ignore_index=True)
    lib["timestamp"] = pd.to_datetime(lib["timestamp"], utc=True, errors="coerce")
    lib = lib.dropna(subset=["timestamp", "embedding"])
    print(f"  total labeled records: {len(lib):,}")

    print("expanding BGE embedding (384 dims)...")
    emb = np.stack([_parse_embedding(v) for v in lib["embedding"]])
    bge_cols = [f"bge_{i:03d}" for i in range(emb.shape[1])]
    bge_df = pd.DataFrame(emb, columns=bge_cols, index=lib.index)

    print("one-hot family + subtype (top 8 + top 30)...")
    fam_df = _onehot_top_k(lib["catalyst_family"].fillna("none"), k=8, prefix="fam")
    sub_df = _onehot_top_k(lib["catalyst_subtype"].fillna("none"), k=30, prefix="sub")

    print("joining ticker profile features...")
    prof = pd.read_parquet(TICKER_PROFILE_PATH) if TICKER_PROFILE_PATH.exists() else pd.DataFrame()
    prof_features = pd.DataFrame(index=lib.index)
    if not prof.empty:
        prof_keep = prof[["ticker", "sector", "marketCap", "beta", "floatShares", "averageVolume"]].copy()
        joined = lib[["ticker"]].merge(prof_keep, on="ticker", how="left")
        prof_features["prof_marketCap_log"] = np.log1p(pd.to_numeric(joined["marketCap"], errors="coerce").fillna(0))
        prof_features["prof_float_log"] = np.log1p(pd.to_numeric(joined["floatShares"], errors="coerce").fillna(0))
        prof_features["prof_avg_vol_log"] = np.log1p(pd.to_numeric(joined["averageVolume"], errors="coerce").fillna(0))
        prof_features["prof_beta"] = pd.to_numeric(joined["beta"], errors="coerce").fillna(1.0)
        prof_sector_oh = _onehot_top_k(joined["sector"].fillna("none"), k=11, prefix="sec")
        prof_features = pd.concat([prof_features.reset_index(drop=True), prof_sector_oh.reset_index(drop=True)], axis=1)
        prof_features.index = lib.index

    print("joining FINRA short_ratio z-score at event time...")
    finra_features = pd.DataFrame(index=lib.index)
    if FINRA_SHORT_VOLUME_PATH.exists():
        finra = pd.read_parquet(FINRA_SHORT_VOLUME_PATH)
        # Both sides naive datetime64[ns] for join compatibility
        finra["date"] = pd.to_datetime(finra["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        # FINRA files have one row per (ticker, date, market venue) — collapse to one row per (ticker, date)
        finra = (
            finra.groupby(["ticker", "date"], as_index=False)[["short_volume", "total_volume"]]
            .sum()
        )
        finra["short_ratio"] = finra["short_volume"] / finra["total_volume"].clip(lower=1)
        finra = finra.sort_values(["ticker", "date"])
        grp = finra.groupby("ticker")
        finra["mean_ratio"] = grp["short_ratio"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        finra["std_ratio"] = grp["short_ratio"].transform(lambda s: s.rolling(20, min_periods=5).std())
        finra["short_z"] = (finra["short_ratio"] - finra["mean_ratio"]) / finra["std_ratio"].replace(0, np.nan)
        finra_lookup = finra.set_index(["ticker", "date"])[["short_ratio", "short_z"]]
        # Round event timestamp to date for join — strip tz to match
        event_date = lib["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
        lib_keys = pd.DataFrame({"ticker": lib["ticker"].values, "date": event_date.values}, index=lib.index)
        joined_finra = lib_keys.join(finra_lookup, on=["ticker", "date"])
        finra_features["finra_short_ratio"] = joined_finra["short_ratio"].fillna(0.0).astype(np.float32)
        finra_features["finra_short_z"] = joined_finra["short_z"].fillna(0.0).astype(np.float32)

    # CBOE iv30 is current-snapshot only (no history). For now, leave it for
    # forward-looking inference rather than historical training. Mark a hook.

    # Stitch features
    print("stitching feature matrix...")
    base = lib[
        [
            "record_id",
            "ticker",
            "timestamp",
            "catalyst_family",
            "catalyst_subtype",
            "finbert_positive_score",
            "finbert_negative_score",
            "finbert_neutral_score",
            "max_forward_return",
            "max_drawdown",
            "forward_5d_return",
            "forward_10d_return",
            "_lbl",
        ]
    ].copy().reset_index(drop=True)
    base = base.rename(columns={"_lbl": "expansion_label"})
    base["split"] = base["timestamp"].apply(_assign_split)

    parts = [
        base,
        bge_df.reset_index(drop=True),
        fam_df.reset_index(drop=True),
        sub_df.reset_index(drop=True),
    ]
    if not prof_features.empty:
        parts.append(prof_features.reset_index(drop=True))
    if not finra_features.empty:
        parts.append(finra_features.reset_index(drop=True))

    out = pd.concat(parts, axis=1)
    print(f"  feature matrix shape: {out.shape}")
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"  saved -> {OUTPUT_PATH}")

    print("\nsplit distribution:")
    print(out["split"].value_counts().to_string())
    print("\nlabel rate per split:")
    print(out.groupby("split")["expansion_label"].mean().to_string())
    return out


if __name__ == "__main__":
    build()
