"""
POC: do SEMANTIC embeddings of earnings forward-guidance text predict forward returns?

Uses data we already have (no network for labels):
  - guidance text: news_records `earnings_forward_guidance_text` (~24.9k events)
  - forward returns: momentum walk-forward OOF (leak-free) fwd_max_return / fwd_close_return,
    attached as-of the FIRST 4H bar at/after the guidance date (return measured AFTER guidance).

Label: target = fwd_max_return >= +10% within the momentum forward window ("guidance that
precedes a +10% move"). Features: 384-dim MiniLM embedding (+ a few structured guidance feats).
Model: xgboost with a time-based train/test split; report AUC + decile lift + top-bucket stats.

Run: .venv/bin/python scripts/fg_news_embedding_poc.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics import roc_auc_score

from signals.news.config import NEWS_RECORDS_PATH
from signals.events.forward_guidance.features.nlp import extract_structured_guidance_features

MOM_OOF = "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet"
EMB_CACHE = Path("/tmp/guidance_minilm_emb.npy")
UP_THRESHOLD = 0.10


def _clean(s):
    return s.astype(str).str.upper().str.replace("$", "", regex=False).str.strip()


def main():
    cols = ["ticker", "timestamp", "earnings_forward_guidance_text", "catalyst_family"]
    have = [c for c in cols if c in pq.read_schema(NEWS_RECORDS_PATH).names]
    df = pq.read_table(NEWS_RECORDS_PATH, columns=have).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["ticker"] = _clean(df["ticker"])
    txt = df["earnings_forward_guidance_text"].astype(str)
    keep = txt.str.strip().str.len() >= 80
    df = df[keep & df["timestamp"].notna()].reset_index(drop=True)
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    print(f"guidance events with usable text: {len(df):,}  tickers={df['ticker'].nunique():,}")

    # ---- forward returns from leak-free momentum OOF (measured AFTER guidance) ----
    oof = pq.read_table(MOM_OOF, columns=["score"]).to_pandas().reset_index()
    oof = pq.read_table(MOM_OOF).to_pandas().reset_index()
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True)
    oof["ticker"] = _clean(oof["ticker"])
    oof = oof[["timestamp", "ticker", "fwd_max_return", "fwd_close_return"]].dropna()
    oof = oof.sort_values("timestamp")
    df = df.sort_values("timestamp")
    # as-of FORWARD: first OOF bar at/after the guidance timestamp, per ticker
    merged = pd.merge_asof(df, oof, on="timestamp", by="ticker", direction="forward",
                           tolerance=pd.Timedelta("7D"))
    merged = merged.dropna(subset=["fwd_max_return"]).reset_index(drop=True)
    merged["target"] = (merged["fwd_max_return"] >= UP_THRESHOLD).astype(int)
    print(f"events joined to forward returns: {len(merged):,}  base +10% rate={merged['target'].mean():.3f}")

    # ---- embeddings (MiniLM, cached) ----
    texts = merged["earnings_forward_guidance_text"].astype(str).tolist()
    if EMB_CACHE.exists() and np.load(EMB_CACHE).shape[0] == len(texts):
        emb = np.load(EMB_CACHE)
        print("loaded cached embeddings", emb.shape)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
        np.save(EMB_CACHE, emb)
        print("embedded", emb.shape)

    # ---- structured guidance features (cheap, for comparison/combo) ----
    struct = pd.DataFrame([extract_structured_guidance_features(t) for t in texts]).reset_index(drop=True)

    # ---- time-based split + xgb ----
    import xgboost as xgb
    order = merged["timestamp"].values.argsort()
    emb, struct, merged = emb[order], struct.iloc[order].reset_index(drop=True), merged.iloc[order].reset_index(drop=True)
    cut = int(len(merged) * 0.8)
    y = merged["target"].values
    embcols = [f"e{i}" for i in range(emb.shape[1])]

    def run(name, X):
        d = xgb.train({"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 4,
                       "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "verbosity": 0},
                      xgb.DMatrix(X[:cut], label=y[:cut]), num_boost_round=250)
        p = d.predict(xgb.DMatrix(X[cut:]))
        auc = roc_auc_score(y[cut:], p) if len(set(y[cut:])) > 1 else float("nan")
        te = merged.iloc[cut:].copy(); te["p"] = p
        te["dec"] = pd.qcut(te["p"], 10, labels=False, duplicates="drop")
        lift = te.groupby("dec")["fwd_max_return"].mean()
        topdec = te[te["dec"] == te["dec"].max()]
        print(f"  {name:22s} test AUC={auc:.4f} | top-decile mean fwd_max={lift.iloc[-1]:+.3f} "
              f"+10%hit={topdec['target'].mean():.2f} (n={len(topdec)}) | bottom-decile={lift.iloc[0]:+.3f}")
        return auc

    Xemb = emb.astype("float32")
    Xstruct = struct.values.astype("float32")
    Xboth = np.hstack([Xemb, Xstruct]).astype("float32")
    print(f"\nsplit: train={cut} test={len(merged)-cut}  (test +10% rate={y[cut:].mean():.3f})")
    run("structured-only", Xstruct)
    run("embeddings-only", Xemb)
    run("embeddings+structured", Xboth)


if __name__ == "__main__":
    main()
