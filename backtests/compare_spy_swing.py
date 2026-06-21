"""Head-to-head backtest: NEW swing-label competition model vs the LIVE-wired
GA-XGB swing model, scored on the SAME held-out time tail of the context matrix.

Both target the swing-pivot labels (long_swing_label / short_swing_label).
Per trading day we pick the top-K rows by model score and earn the realized
swing return (atr_realized_return); short PnL is negated because the column is
raw long-direction return. Outputs a metrics table + equity-curve plots.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "Data/processed/spy/training_export/spy_daytrader_context_matrix.parquet"
FULL_FEATURES = REPO / "Data/processed/spy/datasets/10min/features_X_10min_tree.txt"
NEW = Path("/tmp/spy_new_model")
LIVE = REPO / "Data/models/ga_xgboost/10min"
OUT = REPO / "backtests/20260618_swing_compare"
OUT.mkdir(parents=True, exist_ok=True)

TEST_FRAC = 0.15      # new model's competition held-out tail
K = 5                 # top-K picks per day
RET_COL = "atr_realized_return"


def load_new(side: str):
    """XGBRanker + its selected feature order."""
    lab = f"{side}_swing_label"
    d = NEW / lab / lab
    model = joblib.load(d / "best_model.joblib")
    feats = json.loads((d / f"selected_{lab}.json").read_text())["features"]
    return model, feats


def load_live(side: str):
    """Native booster + the exact masked feature order from live inference."""
    d = LIVE / side / "swing"
    bst = xgb.Booster()
    bst.load_model(str(d / "xgb_model.json"))
    feats = [l.strip() for l in (d / "selected_features.txt").read_text().splitlines() if l.strip()]
    return bst, feats


def score(model, feats, frame: pd.DataFrame, native: bool) -> np.ndarray:
    X = frame[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    if native:
        return model.predict(xgb.DMatrix(X))
    return model.predict(X)


def evaluate(name, side, score_vec, test: pd.DataFrame) -> dict:
    label = f"{side}_swing_label"
    y = (pd.to_numeric(test[label], errors="coerce") == 1).astype(int).to_numpy()
    ret = pd.to_numeric(test[RET_COL], errors="coerce").to_numpy()
    if side == "short":          # raw return is long-direction; short profits when price falls
        ret = -ret
    g = pd.DataFrame({"score": score_vec, "y": y, "ret": ret, "date": test["date"].to_numpy()})
    base = float(y.mean())
    try:
        auc = float(roc_auc_score(y, score_vec))
    except ValueError:
        auc = float("nan")
    prec, picked = [], []
    for _, day in g.groupby("date", sort=True):
        if len(day) < 2:
            continue
        top = day.nlargest(min(K, len(day)), "score")
        prec.append(top["y"].mean())
        picked.append(top["ret"].mean())
    picked = pd.Series(picked).fillna(0.0)
    prec = float(np.nanmean(prec))
    eq = picked.cumsum()
    return dict(
        model=name, side=side, n_test=int(len(test)), base_rate=round(base, 4),
        auc=round(auc, 4), precision_at_k=round(prec, 4),
        lift=round(prec / base, 2) if base else float("nan"),
        mean_ret_topk=round(float(picked.mean()), 6),
        total_ret=round(float(eq.iloc[-1]), 4) if len(eq) else 0.0,
        n_days=int(len(picked)), _equity=eq,
    )


def main():
    full_feats = [l.strip() for l in FULL_FEATURES.read_text().splitlines() if l.strip()]
    need = sorted(set(full_feats) | {"timestamp", RET_COL, "long_swing_label", "short_swing_label"})
    # union with new-model features
    for s in ("long", "short"):
        _, nf = load_new(s)
        need = sorted(set(need) | set(nf))
    df = pd.read_parquet(MATRIX, columns=[c for c in need if c != "timestamp"] )
    ts = pd.read_parquet(MATRIX, columns=["timestamp"])["timestamp"]
    df["timestamp"] = pd.to_datetime(ts.values, utc=True)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date

    n = len(df)
    t0 = int(n * (1 - TEST_FRAC))
    test = df.iloc[t0:].copy()
    print(f"test rows: {len(test)}  span {test['timestamp'].iloc[0]} -> {test['timestamp'].iloc[-1]}")

    rows = []
    equity = {}
    for side in ("long", "short"):
        nm, nf = load_new(side)
        lm, lf = load_live(side)
        r_new = evaluate("new", side, score(nm, nf, test, native=False), test)
        r_live = evaluate("live", side, score(lm, lf, test, native=True), test)
        for r in (r_live, r_new):
            equity[(r["model"], side)] = r.pop("_equity")
            rows.append(r)

    summary = pd.DataFrame(rows)[
        ["side", "model", "n_test", "base_rate", "auc", "precision_at_k", "lift", "mean_ret_topk", "total_ret", "n_days"]
    ]
    print("\n" + summary.to_string(index=False))
    summary.to_csv(OUT / "summary.csv", index=False)

    # equity plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, side in zip(axes, ("long", "short")):
        for model, style in (("live", "--"), ("new", "-")):
            eq = equity[(model, side)]
            ax.plot(eq.values, style, label=f"{model} (tot={eq.iloc[-1]:+.3f})")
        ax.axhline(0, color="k", lw=.6)
        ax.set(title=f"{side.upper()} swing — top-{K}/day cum return", xlabel="trading day", ylabel="cum atr_realized_return")
        ax.legend()
    fig.suptitle("SPY swing: LIVE vs NEW competition model (held-out tail 2025-06 → 2026-04)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_compare.png", dpi=120)
    print("\nwrote", OUT / "equity_compare.png")


if __name__ == "__main__":
    main()
