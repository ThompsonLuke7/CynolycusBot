"""
Order-policy experiments + exclusion analysis for the Meta Ranker combo signal.

Runs honest OOF (per-signal) experiments on the scored matrix:
  1. K sweep (concentration vs diversification)
  2. stop/target grid (using MFE/MAE; stop-priority conservative approximation)
  3. eligibility gates (liquidity floor, combo confidence floor)
  4. per-ticker performance blacklist derived on TRAIN, validated on a true HOLDOUT
     -> demonstrates it does NOT generalize; liquidity/score floors do.

  PYTHONPATH=. python signals/meta_context/meta_ranker/analyze_policy.py

Uses /tmp/meta_scored.parquet if present, else scores the matrix (needs the bundles).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "meta_ranker_matrix.parquet"
SCORED_CACHE = Path("/tmp/meta_scored.parquet")
SPLIT = pd.Timestamp("2025-07-01", tz="UTC")
K = 10


def load_scored() -> pd.DataFrame:
    if SCORED_CACHE.exists():
        df = pd.read_parquet(SCORED_CACHE)
    else:
        import xgboost as xgb
        M = HERE / "models"
        mat = pd.read_parquet(MATRIX).reset_index()
        mat["timestamp"] = pd.to_datetime(mat["timestamp"], utc=True)
        cols = ["timestamp", "ticker", "theme", "fwd_close_return", "fwd_max_return",
                "fwd_max_drawdown", "meta_good", "dollar_vol_pctile_252"]
        df = mat[cols].copy()
        for lbl in ("upside", "quality"):
            feats = json.loads((M / lbl / "meta.json").read_text())["feature_names"]
            bst = xgb.Booster(); bst.load_model(str(M / lbl / "meta_ranker_xgb.json"))
            df[f"s_{lbl}"] = bst.predict(xgb.DMatrix(mat[feats].astype("float32").values, feature_names=feats))
        df["s_combo"] = (df.groupby("timestamp")["s_upside"].rank(pct=True)
                         + df.groupby("timestamp")["s_quality"].rank(pct=True)) / 2
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "dollar_vol_pctile_252" not in df.columns:
        liq = pd.read_parquet(MATRIX, columns=["dollar_vol_pctile_252"]).reset_index()
        liq["timestamp"] = pd.to_datetime(liq["timestamp"], utc=True)
        df = df.merge(liq, on=["timestamp", "ticker"], how="left")
    return df.dropna(subset=["s_combo", "fwd_close_return", "fwd_max_return", "fwd_max_drawdown"])


def picks(d, k=K):
    return d.sort_values("s_combo", ascending=False).groupby("timestamp").head(k)


def summ(p, name):
    r = p.fwd_close_return
    print(f"  {name:26} n={len(p):6} mean={r.mean():.4f} win={(r>0).mean():.3f} "
          f"MAE={p.fwd_max_drawdown.mean():.3f} ret/std={r.mean()/r.std():.3f}")


def apply_stop_target(p, stop, target):
    r = p.fwd_close_return.values.copy()
    mae, mfe = p.fwd_max_drawdown.values, p.fwd_max_return.values
    out = r.copy()
    stopped = mae >= stop
    out[stopped] = -stop
    out[(~stopped) & (mfe >= target)] = target
    return out


def main():
    df = load_scored()
    print(f"scored {len(df):,} rows  bars={df.timestamp.nunique()}  "
          f"{df.timestamp.min().date()}..{df.timestamp.max().date()}")

    print("\n=== 1) K sweep (combo top-K per bar) ===")
    for k in (5, 10, 15, 20, 30, 50):
        summ(picks(df, k), f"top-{k}")

    print("\n=== 2) stop/target grid at K=10 (stop-priority) ===")
    p = picks(df)
    for stop in (0.06, 0.08, 0.10, 0.15):
        for target in (0.20, 0.30, 99):
            o = apply_stop_target(p, stop, target)
            tlabel = "none" if target == 99 else f"{target:.2f}"
            print(f"  stop {stop:.2f} target {tlabel:>4}: mean={o.mean():.4f} win={(o>0).mean():.3f} ret/std={o.mean()/o.std():.3f}")

    print("\n=== 3) eligibility gates, TRAIN vs HOLDOUT (generalization) ===")
    train, hold = df[df.timestamp < SPLIT], df[df.timestamp >= SPLIT]
    for tag, d in (("TRAIN", train), ("HOLDOUT", hold)):
        print(f" [{tag}]")
        summ(picks(d), "baseline")
        summ(picks(d[d.dollar_vol_pctile_252 >= 0.6]), "liquidity>=0.6")
        summ(picks(d[d.s_combo >= 0.90]), "combo>=0.90")
        summ(picks(d[(d.dollar_vol_pctile_252 >= 0.6) & (d.s_combo >= 0.90)]), "liq0.6 + combo0.90")

    print("\n=== 4) per-ticker perf blacklist (train) -> HOLDOUT (does NOT generalize) ===")
    tp = picks(train)
    g = tp.groupby("ticker")["fwd_close_return"]
    st = pd.DataFrame({"n": g.size(), "mean": g.mean(), "win": g.apply(lambda s: (s > 0).mean())})
    st = st[st.n >= 8]
    bl = set(st[(st["mean"] < 0) & (st["win"] < 0.45)].index)
    summ(picks(hold), "holdout baseline")
    summ(picks(hold[~hold.ticker.isin(bl)]), f"holdout minus {len(bl)} blacklist")
    removed = picks(hold)[picks(hold).ticker.isin(bl)]
    print(f"  -> blacklisted names' holdout mean when picked: {removed.fwd_close_return.mean():.4f} "
          f"(they reverted; excluding them costs return)")


if __name__ == "__main__":
    main()
