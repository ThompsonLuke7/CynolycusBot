"""Why does a model trained on SCRAMBLED labels still show a positive top-3 excess?

In the horizon grid the permuted-label nulls score 0.00-0.02 rank correlation
(correctly ~nothing) yet their top-3 still beats the universe by +2.5 to +8pp on
raw % returns. If that is real, the top-3-excess metric has a positive noise floor
for any MODEL-based score, and only the rank correlation can be read at face value.

The hypothesis: a model fitted to noise still produces a systematic cross-sectional
tilt -- it ranks on whatever features dominate the split structure (volatility,
beta, liquidity) -- and in a rising test period a high-beta tilt beats the mean.

This fits one real label and one permuted label under the grid's exact protocol,
then reports what their top-3 picks LOOK like: realised volatility, beta, price
level and dollar volume, against the universe average.
"""
from __future__ import annotations

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

from run_horizon_grid import N_ROUNDS, PARAMS, load, per_bar_topk_excess, split_bars  # noqa: E402

TILTS = ["atr_pct_14", "beta_spy_60", "realized_vol_20", "dollar_vol_pctile_252", "rvol_20"]


def main() -> None:
    df, feats = load("mom")
    train, val, test, bounds = split_bars(df)
    print(f"test bars {test.index.get_level_values(0).nunique()}  rows {len(test):,}")
    labels = {"mfe_10d": df["mfe_20"].groupby(level=0).rank(pct=True)}
    rng = np.random.default_rng(1003)
    labels["NULL_perm"] = labels["mfe_10d"].groupby(level=0).transform(
        lambda x: pd.Series(rng.permutation(x.to_numpy()), index=x.index))

    bars = test.index.get_level_values(0).to_numpy()
    uni = {c: float(test[c].mean()) for c in TILTS if c in test.columns}
    print("\nuniverse mean:", {k: round(v, 4) for k, v in uni.items()})
    for name, lab in labels.items():
        ytr, yva = lab.reindex(train.index), lab.reindex(val.index)
        bst = xgb.train(PARAMS, xgb.DMatrix(train[feats], label=ytr), num_boost_round=N_ROUNDS,
                        evals=[(xgb.DMatrix(val[feats], label=yva), "v")],
                        early_stopping_rounds=30, verbose_eval=False)
        p = bst.predict(xgb.DMatrix(test[feats]))
        d = pd.DataFrame({"bar": bars, "p": p}, index=test.index)
        d["r"] = d.groupby("bar")["p"].rank(ascending=False, method="first")
        top = d["r"] <= 3
        ex20 = per_bar_topk_excess(bars, p, test["fwdret_40"].to_numpy())
        prof = {c: round(float(test.loc[top.to_numpy(), c].mean()), 4) for c in TILTS if c in test.columns}
        ratio = {c: round(prof[c] / uni[c], 2) for c in prof if uni.get(c)}
        print(f"\n{name}: top-3 excess on 20d return = {ex20.mean() * 100:+.2f}pp")
        print(f"  top-3 profile {prof}")
        print(f"  vs universe   {ratio}  (x universe mean)")


if __name__ == "__main__":
    main()
