"""Reconstruct each competition's held-out test split, score the chosen model,
and emit probability-vs-actual diagnostics + an equity curve per strategy.

Test split = same time-ordered tail the colab competition used (train_frac/val_frac).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

OUT = Path("backtests/20260614")
OUT.mkdir(parents=True, exist_ok=True)

# ---- strategy configs -------------------------------------------------------
CFGS = {
    "momentum": dict(
        matrix="strategies/momentum_expansion/data/training_export/training_matrix_4h.parquet",
        manifest="strategies/momentum_expansion/models/expansion_v1/feature_manifest.json",
        model="strategies/momentum_expansion/models/expansion_v1/model_xgb_classifier_seed45.joblib",
        kind="classifier", positive_label=1,
        target_src="fwd_max_return", target_thresh=0.20,
        ret_col="fwd_close_return", group="timestamp", topk=10,
        train_frac=0.60, val_frac=0.20,
        title="Momentum Expansion — xgb_classifier s45",
    ),
    "htf_swing": dict(
        matrix="strategies/multi_ticker_swing_htf/data/training_export/htf_training_matrix_4h.parquet",
        manifest="strategies/multi_ticker_swing_htf/models/feature_manifest.json",
        model="strategies/multi_ticker_swing_htf/models/model_lgbm_classifier_seed46.joblib",
        kind="classifier", positive_label=1,
        target_src="fwd_best_high_return", target_thresh=0.15,
        ret_col="fwd_close_return", group="timestamp", topk=10,
        train_frac=0.60, val_frac=0.20,
        title="HTF Swing — lgbm_classifier s46",
    ),
    "spy_daytrader": dict(
        matrix="Data/processed/spy/training_export/spy_daytrader_context_matrix.parquet",
        manifest="Data/processed/spy/training_export/spy_daytrader_context_manifest.json",
        model="strategies/spy_intraday/Models/tb_label_competition/model_xgb_ranker_seed46.joblib",
        # also score the classifier for a true probability/calibration view
        proba_model="strategies/spy_intraday/Models/tb_label_competition/model_lgbm_classifier_seed46.joblib",
        kind="ranker", positive_label=2,          # encoded: tb_label==+1
        target_col="tb_label", target_pos_value=1.0,
        ret_col="tb_realized_return", group="date", topk=5,
        train_frac=0.70, val_frac=0.15,
        title="SPY Day Trader — xgb_ranker s46 (tb_label +1)",
    ),
}


def load_features(manifest_path: str) -> list[str]:
    d = json.load(open(manifest_path))
    return d["feature_columns"]


def positive_index(model, positive_label):
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(positive_label) if positive_label in classes else len(classes) - 1


def run(name: str, c: dict):
    feats = load_features(c["manifest"])
    need = list(dict.fromkeys(feats + [c["ret_col"], "timestamp"]))
    if "target_src" in c:
        need.append(c["target_src"])
    if "target_col" in c:
        need.append(c["target_col"])
    cols = pd.read_parquet(c["matrix"], columns=[x for x in dict.fromkeys(need)])
    if "timestamp" not in cols.columns:      # some matrices store timestamp as the index
        cols = cols.reset_index()
    cols["timestamp"] = pd.to_datetime(cols["timestamp"], utc=True)

    # binary outcome
    if "target_src" in c:
        y = (pd.to_numeric(cols[c["target_src"]], errors="coerce") >= c["target_thresh"]).astype(int)
    else:
        y = (pd.to_numeric(cols[c["target_col"]], errors="coerce") == c["target_pos_value"]).astype(int)
    cols["_y"] = y
    cols = cols.dropna(subset=["_y"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    n = len(cols)
    t2 = int(n * (c["train_frac"] + c["val_frac"]))
    test = cols.iloc[t2:].copy()

    X = test[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    model = joblib.load(c["model"])

    if c["kind"] == "classifier":
        pi = positive_index(model, c["positive_label"])
        score = model.predict_proba(X)[:, pi]
        prob = score
    else:  # ranker -> raw score; also get a real probability from classifier
        score = model.predict(X)
        pm = joblib.load(c["proba_model"])
        prob = pm.predict_proba(X)[:, positive_index(pm, c["positive_label"])]

    test["score"] = score
    test["prob"] = prob
    test["date"] = test["timestamp"].dt.date
    yv = test["_y"].to_numpy()
    base = float(yv.mean())

    # ---- group top-K precision + equity ----
    gkey = test["date"] if c["group"] == "date" else test["timestamp"]
    test["_g"] = gkey
    K = c["topk"]
    prec_at_k, picked_ret, all_ret = [], [], []
    for _, g in test.groupby("_g", sort=False):
        if len(g) < 2:
            continue
        top = g.nlargest(min(K, len(g)), "score")
        prec_at_k.append(top["_y"].mean())
        picked_ret.append(pd.to_numeric(top[c["ret_col"]], errors="coerce").mean())
        all_ret.append(pd.to_numeric(g[c["ret_col"]], errors="coerce").mean())
    prec = float(np.nanmean(prec_at_k))
    lift = prec / base if base else float("nan")
    try:
        auc = roc_auc_score(yv, score)
    except ValueError:
        auc = float("nan")

    picked_ret = pd.Series(picked_ret).fillna(0.0)
    all_ret = pd.Series(all_ret).fillna(0.0)
    eq_pick = picked_ret.cumsum()
    eq_base = all_ret.cumsum()

    summary = dict(
        n_test=int(len(test)), base_rate=round(base, 4), auc=round(float(auc), 4),
        precision_at_k=round(prec, 4), lift=round(float(lift), 2), K=K,
        mean_ret_topk=round(float(picked_ret.mean()), 5),
        mean_ret_all=round(float(all_ret.mean()), 5),
        total_ret_topk=round(float(eq_pick.iloc[-1]), 3),
        n_groups=int(len(picked_ret)),
    )

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{c['title']}  |  test n={len(test):,}  base={base:.3f}  AUC={auc:.3f}  "
                 f"P@{K}={prec:.3f}  lift={lift:.2f}x", fontsize=13)

    # (1) calibration deciles
    dfp = pd.DataFrame({"prob": prob, "y": yv})
    dfp["bin"] = pd.qcut(dfp["prob"].rank(method="first"), 10, labels=False)
    cal = dfp.groupby("bin").agg(pred=("prob", "mean"), actual=("y", "mean"))
    a = ax[0, 0]
    a.plot(cal["pred"], cal["actual"], "o-", label="model")
    lim = [0, max(cal["pred"].max(), cal["actual"].max()) * 1.1]
    a.plot(lim, lim, "k--", alpha=.5, label="perfect")
    a.axhline(base, color="r", ls=":", label=f"base {base:.3f}")
    a.set(title="Calibration (decile)", xlabel="predicted P(+)", ylabel="actual P(+)"); a.legend()

    # (2) score decile vs realized return (economic lift)
    dfr = pd.DataFrame({"score": score, "ret": pd.to_numeric(test[c["ret_col"]], errors="coerce").to_numpy()})
    dfr["bin"] = pd.qcut(dfr["score"].rank(method="first"), 10, labels=False)
    rr = dfr.groupby("bin")["ret"].mean()
    a = ax[0, 1]
    a.bar(rr.index, rr.values, color=["#d62728" if v < 0 else "#2ca02c" for v in rr.values])
    a.axhline(dfr["ret"].mean(), color="k", ls=":", label=f"avg {dfr['ret'].mean():.4f}")
    a.set(title="Mean realized return by score decile", xlabel="score decile (9=top)", ylabel=c["ret_col"]); a.legend()

    # (3) equity curve
    a = ax[1, 0]
    a.plot(eq_pick.values, label=f"top-{K}/grp")
    a.plot(eq_base.values, label="all (baseline)", alpha=.7)
    a.axhline(0, color="k", lw=.6)
    a.set(title="Cumulative realized return", xlabel="group # (time)", ylabel="cum return"); a.legend()

    # (4) probability vs actual over a sample test window
    a = ax[1, 1]
    s = test.sort_values("timestamp").iloc[-400:].reset_index(drop=True)
    a.plot(s.index, s["prob"], lw=.8, color="#1f77b4", label="P(+)")
    win = s[s["_y"] == 1]
    a.scatter(win.index, win["prob"], c="#2ca02c", s=18, label="actual +", zorder=3)
    los = s[s["_y"] == 0]
    a.scatter(los.index, los["prob"], c="#d62728", s=6, alpha=.3, label="actual -", zorder=2)
    a.axhline(base, color="r", ls=":", alpha=.6)
    a.set(title="Predicted prob vs actual (last 400 test rows)", xlabel="test row", ylabel="P(+)"); a.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f"{name}_backtest.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    summary["plot"] = str(p)
    return summary


if __name__ == "__main__":
    results = {}
    for name, c in CFGS.items():
        print(f"\n=== {name} ===")
        try:
            results[name] = run(name, c)
            print(json.dumps(results[name], indent=2))
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": repr(e)}
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "summary.json")
