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
MANIFEST_PATH = Path("meta_context/data/processed/catalyst_training_manifest.json")

# Daily bars for per-ticker price-action features
BARS_DAILY_DIR = Path("Data/shared/bars/1d")

# Kaggle macro features dataset
KAGGLE_MARKET_PRICES = Path("/home/luket/.cache/kagglehub/datasets/belbino/financial-news-sentiment-vs-market-2020-present/versions/50/market_prices.csv")

# Chronological splits sized roughly 70K / 20K / 20K against the current
# corpus distribution. Recompute these if the corpus shape changes:
#   .venv/bin/python -c "import pandas as pd; m=pd.read_parquet('meta_context/data/processed/catalyst_training_matrix.parquet'); m=m.sort_values('timestamp').reset_index(drop=True); print(m.iloc[70000]['timestamp'], m.iloc[90000]['timestamp'])"
TRAIN_END = pd.Timestamp("2026-04-29", tz="UTC")
VAL_END = pd.Timestamp("2026-05-17", tz="UTC")


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

    # Source quality + mention frequency (social-buzz / attention-peak signal)
    print("computing source_quality one-hots + per-(ticker, day) mention frequency...")
    quality_features = pd.DataFrame(index=lib.index)
    if "source_quality" in lib.columns:
        for q in ("breaking", "high_alpha", "opinion", "aggregator"):
            quality_features[f"sq_{q}"] = (lib["source_quality"] == q).astype(np.float32).values
    # Per-(ticker, day) mention frequency — counts all records (any source_quality)
    # plus aggregator-only count (the "buzz / peak attention" proxy).
    nr_path = Path("news/data/processed/news_records.parquet")
    if nr_path.exists():
        all_records = pd.read_parquet(nr_path, columns=["ticker", "timestamp", "source_quality"]).copy()
        all_records["timestamp"] = pd.to_datetime(all_records["timestamp"], utc=True, errors="coerce")
        all_records = all_records.dropna(subset=["timestamp"])
        all_records["date"] = all_records["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
        all_records["ticker"] = all_records["ticker"].astype(str).str.upper()
        mention_total = (
            all_records.groupby(["ticker", "date"]).size().rename("mention_total_1d").reset_index()
        )
        mention_agg = (
            all_records[all_records["source_quality"] == "aggregator"]
            .groupby(["ticker", "date"]).size().rename("mention_agg_1d").reset_index()
        )
        # Rolling 5-day and 20-day counts per ticker
        m = mention_total.copy()
        m = m.sort_values(["ticker", "date"])
        m["mention_total_5d"] = m.groupby("ticker")["mention_total_1d"].transform(lambda s: s.rolling(5, min_periods=1).sum())
        m["mention_total_20d"] = m.groupby("ticker")["mention_total_1d"].transform(lambda s: s.rolling(20, min_periods=1).sum())
        ma = mention_agg.copy().sort_values(["ticker", "date"])
        if not ma.empty:
            ma["mention_agg_5d"] = ma.groupby("ticker")["mention_agg_1d"].transform(lambda s: s.rolling(5, min_periods=1).sum())
        else:
            ma["mention_agg_5d"] = 0
        # Join to lib by (ticker, date)
        lib_keys = pd.DataFrame(
            {
                "ticker": lib["ticker"].astype(str).str.upper().values,
                "date": lib["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]").values,
            },
            index=lib.index,
        )
        joined = lib_keys.merge(m[["ticker", "date", "mention_total_1d", "mention_total_5d", "mention_total_20d"]], on=["ticker", "date"], how="left")
        joined = joined.merge(ma[["ticker", "date", "mention_agg_1d", "mention_agg_5d"]], on=["ticker", "date"], how="left")
        for c in ("mention_total_1d", "mention_total_5d", "mention_total_20d", "mention_agg_1d", "mention_agg_5d"):
            quality_features[c] = joined[c].fillna(0.0).astype(np.float32).values
        # buzz acceleration: today vs 20-day baseline (peak attention proxy)
        baseline_20d = (quality_features["mention_total_20d"] / 20.0).clip(lower=0.1)
        quality_features["mention_buzz_ratio"] = (quality_features["mention_total_1d"] / baseline_20d).astype(np.float32)

    print("joining Kaggle market-regime features at event date...")
    macro_features = pd.DataFrame(index=lib.index)
    if KAGGLE_MARKET_PRICES.exists():
        macro = pd.read_csv(KAGGLE_MARKET_PRICES)
        macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        macro_keep = [
            "date", "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_range_pct", "spy_gap_pct",
            "qqq_return_1d", "qqq_return_5d", "qqq_return_20d",
            "vix_close", "vix_pct_chg", "market_direction",
        ]
        macro_keep = [c for c in macro_keep if c in macro.columns]
        macro = macro[macro_keep].sort_values("date")
        # Lookup at event date — forward-fill so events on weekends/holidays take Friday's value
        macro_indexed = macro.set_index("date").astype(float)
        event_dates_naive = lib["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
        joined = macro_indexed.reindex(event_dates_naive, method="ffill")
        for col in macro_indexed.columns:
            macro_features[f"macro_{col}"] = joined[col].fillna(0.0).astype(np.float32).values

    print("joining per-ticker price-action features at event time...")
    price_features = pd.DataFrame(index=lib.index)
    if BARS_DAILY_DIR.exists():
        # Cache bars per ticker as needed (only tickers in lib)
        tickers_in_lib = lib["ticker"].astype(str).str.upper().unique()
        bars_cache: dict[str, pd.DataFrame] = {}
        loaded = 0
        for t in tickers_in_lib:
            p = BARS_DAILY_DIR / f"{t}.parquet"
            if not p.exists():
                continue
            try:
                b = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close", "volume"])
            except Exception:
                continue
            b["date"] = pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert(None).dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
            b = b.sort_values("date").reset_index(drop=True)
            # Pre-compute rolling features
            b["ret_1d"] = b["close"].pct_change(1)
            b["ret_5d"] = b["close"].pct_change(5)
            b["ret_20d"] = b["close"].pct_change(20)
            b["vol_20d"] = b["ret_1d"].rolling(20, min_periods=5).std()
            b["high_20d"] = b["high"].rolling(20, min_periods=5).max()
            b["dist_from_high_20d"] = b["close"] / b["high_20d"] - 1.0
            b["high_252d"] = b["high"].rolling(252, min_periods=20).max()
            b["dist_from_high_252d"] = b["close"] / b["high_252d"] - 1.0
            # Simple RSI(14)
            delta = b["ret_1d"]
            gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
            rs = gain / loss.replace(0, np.nan)
            b["rsi_14"] = 100 - 100 / (1 + rs)
            # Drawdown from rolling-20d high
            b["dd_from_20d_high"] = b["close"] / b["high_20d"] - 1.0
            bars_cache[t] = b.set_index("date")[
                ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "dist_from_high_20d", "dist_from_high_252d", "rsi_14"]
            ]
            loaded += 1
        print(f"  bar files loaded: {loaded}/{len(tickers_in_lib)} tickers")

        feature_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "dist_from_high_20d", "dist_from_high_252d", "rsi_14"]
        for col in feature_cols:
            price_features[f"px_{col}"] = 0.0

        # Vectorize lookup per ticker
        event_dates_naive = lib["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
        lookup_df = pd.DataFrame({"ticker": lib["ticker"].astype(str).str.upper().values,
                                   "date": event_dates_naive.values}, index=lib.index)
        for t, grp in lookup_df.groupby("ticker"):
            bars = bars_cache.get(t)
            if bars is None:
                continue
            # Reindex bars at the event dates, forward-fill so a Saturday event uses Friday's bar
            window = bars.reindex(grp["date"].values, method="ffill")
            for col in feature_cols:
                price_features.loc[grp.index, f"px_{col}"] = window[col].values

        # Fill remaining NaNs (events without bar data) with 0 and downcast
        for col in price_features.columns:
            price_features[col] = price_features[col].fillna(0.0).astype(np.float32)

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

    # Additional targets so we can train multiple model variants from the same
    # matrix. These are derived from the existing forward-return columns:
    #
    #   expansion_10pct  : max_forward_return >= 0.10  (the legacy binary target)
    #   expansion_5pct   : max_forward_return >= 0.05  (lenient threshold)
    #   crash_5pct       : forward_10d_return <= -0.05 OR max_drawdown <= -0.05
    #   fwd_10d_return   : continuous regression target (clipped at ±50%)
    #
    base["target_expansion_10pct"] = (base["max_forward_return"] >= 0.10).astype(int)
    base["target_expansion_5pct"] = (base["max_forward_return"] >= 0.05).astype(int)
    crash_by_fwd = base["forward_10d_return"].fillna(0) <= -0.05
    crash_by_dd = base["max_drawdown"].fillna(0) <= -0.05
    base["target_crash_5pct"] = (crash_by_fwd | crash_by_dd).astype(int)
    base["target_fwd_10d_reg"] = base["forward_10d_return"].fillna(0.0).clip(-0.5, 0.5).astype(float)

    # Multi-class trajectory target: separates "spiked then crashed" from "stayed up" etc.
    # This is what lets a downstream model learn the *kind* of move per catalyst, not
    # just "did it expand or not".
    mfr = base["max_forward_return"].fillna(0.0)
    mdd = base["max_drawdown"].fillna(0.0)
    fwd10 = base["forward_10d_return"].fillna(0.0)

    traj = pd.Series(["flat"] * len(base), index=base.index)
    traj[(mfr >= 0.10) & (mdd > -0.05)] = "bull_steady"
    traj[(mfr >= 0.10) & (mdd <= -0.05) & (fwd10 >= 0)] = "bull_volatile"
    traj[(fwd10 <= -0.05) & (mfr >= 0.10)] = "v_bounce"
    traj[(fwd10 <= -0.05) & (mfr < 0.05)] = "crash_stayed"
    base["target_trajectory"] = traj
    # Also persist an integer encoding so XGBoost can consume it directly
    trajectory_codes = {"flat": 0, "bull_steady": 1, "bull_volatile": 2, "v_bounce": 3, "crash_stayed": 4}
    base["target_trajectory_code"] = base["target_trajectory"].map(trajectory_codes).astype(int)

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
    if not macro_features.empty:
        parts.append(macro_features.reset_index(drop=True))
    if not price_features.empty:
        parts.append(price_features.reset_index(drop=True))
    if not quality_features.empty:
        parts.append(quality_features.reset_index(drop=True))

    out = pd.concat(parts, axis=1)
    print(f"  feature matrix shape: {out.shape}")
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"  saved -> {OUTPUT_PATH}")

    # Save a manifest describing the exact column order + categorical levels
    # so the live scorer can reproduce the feature vector layout one-shot.
    drop_for_features = [
        "record_id", "ticker", "timestamp", "catalyst_family", "catalyst_subtype",
        "source_quality",
        "expansion_label",
        "target_expansion_10pct", "target_expansion_5pct", "target_crash_5pct",
        "target_fwd_10d_reg", "target_trajectory", "target_trajectory_code",
        "max_forward_return", "max_drawdown",
        "forward_5d_return", "forward_10d_return", "split",
    ]
    feature_cols = [c for c in out.columns if c not in drop_for_features]
    manifest = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "family_levels": [c.replace("fam_", "") for c in out.columns if c.startswith("fam_")],
        "subtype_levels": [c.replace("sub_", "") for c in out.columns if c.startswith("sub_")],
        "sector_levels": [c.replace("sec_", "") for c in out.columns if c.startswith("sec_")],
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
        "bge_dim": 384,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {MANIFEST_PATH}")

    print("\nsplit distribution:")
    print(out["split"].value_counts().to_string())
    print("\nlabel rates per split:")
    for tgt in ("target_expansion_10pct", "target_expansion_5pct", "target_crash_5pct"):
        if tgt in out.columns:
            print(f"  {tgt}: {out.groupby('split')[tgt].mean().round(3).to_dict()}")
    if "target_fwd_10d_reg" in out.columns:
        print(f"  target_fwd_10d_reg mean per split: {out.groupby('split')['target_fwd_10d_reg'].mean().round(4).to_dict()}")
    if "target_trajectory" in out.columns:
        print("  target_trajectory distribution per split:")
        for split, grp in out.groupby("split"):
            print(f"    {split}: {grp['target_trajectory'].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    build()
