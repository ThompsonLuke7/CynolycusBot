"""
Forward-guidance signal for the Meta Ranker — semantic-embedding model score.

Mirrors the momentum / HTF base scores: a leak-free, walk-forward out-of-fold model
output. We embed each earnings forward-guidance text (MiniLM) + a few structured
guidance features, label it with the leak-free momentum-OOF forward return
(fwd_max_return >= +10% within the forward window), and produce a per-event
``fg_guidance_score`` via walk-forward OOF (each event scored by a model that did not
train on it). Recent unlabeled events get the final-fold model's score for live use.

Guidance text comes from the news records' `earnings_forward_guidance_text` (~24k events,
local, no network). This supersedes the earlier keyword-only block, which carried no edge
(AUC ~0.54) — the embeddings do (AUC ~0.64; top decile ~62% hit of +10% moves).

Output: signals/meta_context/data/processed/forward_guidance_signal.parquet
Run:    PYTHONPATH=. .venv/bin/python -m signals.meta_context.build_forward_guidance_signal
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from signals.news.config import NEWS_RECORDS_PATH
from signals.events.forward_guidance.features.nlp import extract_structured_guidance_features

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "signals/meta_context/data/processed/forward_guidance_signal.parquet"
EMB_CACHE = REPO / "signals/meta_context/data/processed/guidance_minilm_emb.npz"
MOM_OOF = REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet"

UP_THRESHOLD = 0.10          # label: guidance precedes a +10% forward max move
EMBARGO_DAYS = 15            # > forward-label horizon, to keep OOF folds leak-free
N_FOLDS = 5
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Final fg_ columns the meta-ranker consumes (lean: the model score + recency/density).
FG_FEATURE_COLS = ["fg_guidance_score", "fg_guidance_count_90d"]

_STRUCT_KEYS = [
    "guidance_strength_score", "guidance_revenue_raise_cut", "guidance_eps_raise_cut",
    "margin_expansion_score", "ai_demand_mentions", "backlog_order_mentions",
    "uncertainty_language", "confidence_language", "capex_expansion", "hiring_slowdown",
    "guidance_text_chars",
]


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("$", "", regex=False).str.strip()


def _load_guidance_events() -> pd.DataFrame:
    cols = ["ticker", "timestamp", "earnings_forward_guidance_text"]
    have = [c for c in cols if c in pq.read_schema(NEWS_RECORDS_PATH).names]
    df = pq.read_table(NEWS_RECORDS_PATH, columns=have).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["ticker"] = _clean(df["ticker"])
    txt = df["earnings_forward_guidance_text"].astype(str)
    keep = (txt.str.strip().str.len() >= 80) & df["timestamp"].notna() & df["ticker"].ne("")
    return df[keep].sort_values("timestamp").reset_index(drop=True)


def _embed(texts: list[str]) -> np.ndarray:
    # Cache guard: same event count -> reuse (texts are deterministic from the same records).
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        if z["emb"].shape[0] == len(texts):
            print(f"  loaded cached embeddings {z['emb'].shape}")
            return z["emb"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    np.savez(EMB_CACHE, emb=emb)
    return emb


def _walk_forward_oof(X: np.ndarray, y: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Leak-free OOF probs: each time-fold scored by a model trained only on earlier events."""
    import xgboost as xgb
    oof = np.full(len(y), np.nan, dtype="float32")
    edges = np.linspace(0, len(y), N_FOLDS + 1).astype(int)  # data already sorted by ts
    params = dict(objective="binary:logistic", max_depth=4, eta=0.05, subsample=0.8,
                  colsample_bytree=0.6, verbosity=0)
    for k in range(1, N_FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test_start = ts[lo]
        train_mask = ts < (test_start - np.timedelta64(EMBARGO_DAYS, "D"))
        if train_mask.sum() < 500 or len(set(y[train_mask])) < 2:
            continue
        bst = xgb.train(params, xgb.DMatrix(X[train_mask], label=y[train_mask]), num_boost_round=250)
        oof[lo:hi] = bst.predict(xgb.DMatrix(X[lo:hi]))
    return oof


def build(output_path: Path = OUT) -> pd.DataFrame:
    ev = _load_guidance_events()
    print(f"guidance events with usable text: {len(ev):,}  tickers={ev['ticker'].nunique():,}")
    texts = ev["earnings_forward_guidance_text"].astype(str).tolist()
    emb = _embed(texts)
    struct = pd.DataFrame([extract_structured_guidance_features(t) for t in texts])[_STRUCT_KEYS].astype("float32")
    X = np.hstack([emb, struct.values]).astype("float32")

    # ---- leak-free forward-return label from momentum OOF (measured AFTER guidance) ----
    oof = pq.read_table(MOM_OOF).to_pandas().reset_index()
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True)
    oof["ticker"] = _clean(oof["ticker"])
    oof = oof[["timestamp", "ticker", "fwd_max_return"]].dropna().sort_values("timestamp")
    lab = pd.merge_asof(ev[["timestamp", "ticker"]], oof, on="timestamp", by="ticker",
                        direction="forward", tolerance=pd.Timedelta("7D"))
    y_all = (lab["fwd_max_return"] >= UP_THRESHOLD).astype("float32").values
    labeled = lab["fwd_max_return"].notna().values
    print(f"  labeled events: {labeled.sum():,}  base +{int(UP_THRESHOLD*100)}% rate={y_all[labeled].mean():.3f}")

    # ---- walk-forward OOF on labeled events; final model scores the rest (live inference) ----
    ts = ev["timestamp"].values.astype("datetime64[ns]")
    score = np.full(len(ev), np.nan, dtype="float32")
    Xl, yl, tsl = X[labeled], y_all[labeled], ts[labeled]
    score[labeled] = _walk_forward_oof(Xl, yl, tsl)
    import xgboost as xgb
    final = xgb.train(dict(objective="binary:logistic", max_depth=4, eta=0.05, subsample=0.8,
                           colsample_bytree=0.6, verbosity=0),
                      xgb.DMatrix(Xl, label=yl), num_boost_round=250)
    need = np.isnan(score)
    if need.any():
        score[need] = final.predict(xgb.DMatrix(X[need]))
    ev["fg_guidance_score"] = score
    from sklearn.metrics import roc_auc_score
    m = labeled & ~np.isnan(score)
    print(f"  OOF guidance_score AUC (labeled) = {roc_auc_score(y_all[m], score[m]):.4f}")

    # ---- one row per (ticker, event day); trailing-90d guidance density ----
    ev["date"] = ev["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    ev["_abs"] = (ev["fg_guidance_score"] - 0.5).abs()
    ev = (ev.sort_values(["ticker", "date", "_abs"])
            .drop_duplicates(["ticker", "date"], keep="last").drop(columns="_abs"))
    ev = ev.sort_values(["ticker", "date"]).reset_index(drop=True)
    counts = [pd.Series(1, index=pd.DatetimeIndex(g["date"])).rolling("90D").sum().to_numpy()
              for _, g in ev.groupby("ticker", sort=False)]
    ev["fg_guidance_count_90d"] = np.concatenate(counts) if counts else 0.0

    out = ev[["ticker", "date"] + FG_FEATURE_COLS].sort_values(["ticker", "date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"wrote {output_path}  rows={len(out):,}  tickers={out['ticker'].nunique():,}  "
          f"dates {out['date'].min().date()}..{out['date'].max().date()}")
    return out


if __name__ == "__main__":
    build()
