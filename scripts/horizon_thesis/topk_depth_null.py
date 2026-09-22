"""Is the top-k DEPTH gradient real, or the same noise tilt? (selectivity check)

`momentum_config.RANKING_CONFIG["top_n"]` was cut 10 -> 3 on the strength of a
DEPTH comparison: top-3 beat top-10 by +1.10/+1.66/+2.82pp across holds. That
rested on a shuffled-SCORE null, which §5 showed is ~0 and therefore only proves
the harness is clean.

`null_tilt_check.py` then found the thing that actually bites: a model trained on
within-bar PERMUTED LABELS scores ~0 rank correlation yet still shows +2.5 to
+8pp top-3 excess on raw returns, because fitting noise still produces a
cross-sectional tilt (its top-3 carried 1.46x universe ATR, 1.62x realised vol,
1.36x beta). That tilt concentrates at the TOP of the ranking, so it inflates a
shallow-vs-deep comparison too -- and the depth decision was never tested
against it.

This fits a real model and a permuted-label model under the grid's exact
protocol and compares their DEPTH gradients, not just their levels:

    real   k3-k10 gradient  vs  null k3-k10 gradient

If the null shows a comparable gradient, `top_n=3` is buying a volatility tilt
and the parameter should go back. If the null's gradient is ~0 while the real
model's is positive, the depth effect is real and the change stands.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from run_horizon_grid import (  # noqa: E402
    EVAL_H, MIN_XS, N_ROUNDS, PARAMS, block_boot, load, per_bar_spearman, split_bars,
)

DEPTHS = (1, 2, 3, 5, 10)
TILTS = ["atr_pct_14", "beta_spy_60", "realized_vol_20", "rvol_20"]


def topk_excess(bar, p, y, k):
    """Per-bar mean of the top-k minus the bar's own cross-sectional mean."""
    d = pd.DataFrame({"b": bar, "p": p, "y": y}).dropna()
    n = d.groupby("b")["y"].transform("size")
    d = d[n >= MIN_XS]
    d["r"] = d.groupby("b")["p"].rank(ascending=False, method="first")
    top = d[d["r"] <= k].groupby("b")["y"].mean()
    return top - d.groupby("b")["y"].mean().reindex(top.index)


def tilt_profile(test: pd.DataFrame, bar, p, k) -> dict:
    d = pd.DataFrame({"b": bar, "p": p}, index=test.index)
    d["r"] = d.groupby("b")["p"].rank(ascending=False, method="first")
    sel = d["r"] <= k
    out = {}
    for c in TILTS:
        if c not in test.columns:
            continue
        uni = test[c].mean()
        out[c] = float(test.loc[sel, c].mean() / uni) if uni and np.isfinite(uni) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = ap.parse_args()

    df, feats = load("mom")
    train, val, test, bounds = split_bars(df)
    print(f"\ntrain/val/test bars: {train.index.get_level_values(0).nunique()}/"
          f"{val.index.get_level_values(0).nunique()}/{test.index.get_level_values(0).nunique()}")
    print(f"test window {bounds['test_start']} .. {bounds['test_end']}\n")

    # Real label: within-bar percentile rank of 10-day MFE, the grid's convention.
    real = df["mfe_20"].groupby(level=0).rank(pct=True)
    rng = np.random.default_rng(1003)
    null = real.groupby(level=0).transform(
        lambda x: pd.Series(rng.permutation(x.to_numpy()), index=x.index))
    labels = {"REAL": real, "NULL_permuted_label": null}

    # CPU: the repo's standing rule is no GPU training.
    params = {**PARAMS, "device": "cpu"}
    bar = test.index.get_level_values(0).to_numpy()
    preds, boot = {}, np.random.default_rng(7)

    for name, y in labels.items():
        dtr = xgb.DMatrix(train[feats], label=y.reindex(train.index))
        model = xgb.train(params, dtr, num_boost_round=args.rounds)
        preds[name] = pd.Series(model.predict(xgb.DMatrix(test[feats])), index=test.index)
        rho = per_bar_spearman(bar, preds[name].to_numpy(),
                               y.reindex(test.index).to_numpy()).mean()
        print(f"{name:22s} mean within-bar rank correlation on test: {rho:+.4f}")

    print("\n" + "=" * 100)
    print("TOP-K EXCESS OVER THE BAR MEAN, on RAW forward returns")
    print("=" * 100)
    rows = []
    for hname, h in EVAL_H.items():
        y = test[f"fwdret_{h}"].to_numpy()
        print(f"\n--- {hname} ({h} bars) ---")
        print(f"{'model':22s} {'k':>3s} {'excess%':>9s} {'CI':>19s} {'p':>7s}")
        series = {}
        for name in labels:
            for k in DEPTHS:
                s = topk_excess(bar, preds[name].to_numpy(), y, k)
                series[(name, k)] = s
                lo, hi, p = block_boot(s, h, boot)
                print(f"{name:22s} {k:3d} {s.mean()*100:9.2f} "
                      f"[{lo*100:7.2f},{hi*100:7.2f}] {p:7.3f}")
        print(f"\n  {'DEPTH GRADIENT (k=3 minus k=10)':40s}")
        for name in labels:
            g = (series[(name, 3)] - series[(name, 10)]).dropna()
            lo, hi, p = block_boot(g, h, boot)
            rows.append({"hold": hname, "model": name, "gradient_pp": g.mean() * 100,
                         "lo": lo * 100, "hi": hi * 100, "p": p})
            print(f"  {name:22s} {g.mean()*100:+7.2f}pp [{lo*100:+7.2f},{hi*100:+7.2f}] p={p:.3f}")

    print("\n" + "=" * 100)
    print("WHAT THE TOP-K ACTUALLY HOLDS (ratio to universe mean)")
    print("=" * 100)
    print(f"{'model':22s} {'k':>3s} " + " ".join(f"{c:>22s}" for c in TILTS))
    for name in labels:
        for k in (3, 10):
            t = tilt_profile(test, bar, preds[name].to_numpy(), k)
            print(f"{name:22s} {k:3d} " + " ".join(f"{t.get(c, float('nan')):22.2f}" for c in TILTS))

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    out = pd.DataFrame(rows)
    for hname in EVAL_H:
        r = out[out["hold"] == hname].set_index("model")
        if "REAL" not in r.index or "NULL_permuted_label" not in r.index:
            continue
        rg, ng = r.loc["REAL", "gradient_pp"], r.loc["NULL_permuted_label", "gradient_pp"]
        null_sig = r.loc["NULL_permuted_label", "p"] < 0.05
        print(f"  {hname}: real {rg:+.2f}pp vs null {ng:+.2f}pp "
              f"-> {'NULL ALSO POSITIVE, depth gradient is contaminated' if null_sig and ng > 0 else 'null flat; real gradient survives'}")
    out.to_csv(REPO / "research/execution_quality/data/topk_depth_null.csv", index=False)


if __name__ == "__main__":
    main()
