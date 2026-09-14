"""Horizon thesis: train on one forward horizon, score at the 2-6 week thesis horizon.

The Label Horizon Paradox (arXiv 2602.03395) says the best TRAINING horizon can
be shorter than the horizon you are graded on. The thesis being tested is a
momentum/gamma expansion that plays out over ~10-30 trading days. So:

  train labels   forward MFE-in-ATR and clean (MFE - MAE) over 2/5/10/15/20/30
                 trading days (4..60 x 4H bars), each ranked within its bar
  eval targets   forward MFE/clean at 10/20/30 days, plus the TRADEABLE number:
                 share return from the next bar's open to the close H bars later

Everything else is held fixed: features (FEATURE_COLUMNS_4H), rows, split,
hyperparameters. The winner for each eval target is chosen on VALIDATION and
reported on TEST, so the choice itself is out-of-sample. Two pre-specified
stacks (all MFE horizons; 2d+30d) test the "stack them" idea.

Nulls on the identical pipeline (the falsification check): the 20-bar label
permuted within each bar, several seeds. Any "edge" these show is a leak or a
selection artefact, and it sets the noise floor for everything else.

Leakage controls:
  * embargo counted in DECISION BARS, not hours. run_bakeoff.py used
    Timedelta(hours=4*60) = 10 calendar days ~ 14 bars, not the 60 it names.
  * ATR from the bar before the decision; forward windows start at t+1.
  * every row whose 30-day forward window touches a non-organic corporate-action
    flag (unadjusted bar cache) is dropped before anything is fit.
  * CIs are moving-block bootstraps over test bars (block = eval horizon),
    because forward windows of neighbouring bars overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H  # noqa: E402

DATA = REPO / "research/execution_quality/data"
BARS_4H = REPO / "Data/shared/bars/4h"
FLAGS = DATA / "corporate_action_flags.parquet"
TRAIN_H = {"2d": 4, "5d": 10, "10d": 20, "15d": 30, "20d": 40, "30d": 60}
EVAL_H = {"10d": 20, "20d": 40, "30d": 60}
MAX_H = max(max(TRAIN_H.values()), max(EVAL_H.values()))
EMBARGO_BARS = MAX_H + 10
MIN_XS = 50
TOP_K = 3
N_TICKERS = 500
SEED = 17
N_PERM = 5
PARAMS = dict(max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20, tree_method="hist", device="cuda",
              reg_lambda=2.0, objective="reg:squarederror", seed=SEED)
N_ROUNDS = 300
MOM_MATRIX = REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet"
META_MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet"
META_DROP = {"meta_label", "trade_quality", "meta_good", "meta_upside", "meta_rank_quality",
             "fwd_max_return", "fwd_max_alpha", "fwd_close_return", "fwd_max_drawdown",
             "fwd_atr_adj_return", "trend_persistence", "theme", "date", "timestamp", "ticker",
             "__index_level_0__", "__index_level_1__"}


# ---------------------------------------------------------------- labels

def bar_labels(ticker: str) -> pd.DataFrame | None:
    path = BARS_4H / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    if len(d) < 3 * MAX_H:
        return None
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").reset_index(drop=True)
    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=8).mean().shift(1)
    ref, nxt_open = d["close"], d["open"].shift(-1)
    good = atr.notna() & (atr > 0) & (ref > 0)
    out = {"timestamp": d["timestamp"], "ticker": ticker}
    for h in sorted(set(TRAIN_H.values()) | set(EVAL_H.values())):
        hi = d["high"].shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1))
        lo = d["low"].shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1))
        mfe = np.where(good, (hi - ref) / atr, np.nan)
        mae = np.where(good, (ref - lo) / atr, np.nan)
        out[f"mfe_{h}"] = mfe
        out[f"clean_{h}"] = mfe - mae
        with np.errstate(divide="ignore", invalid="ignore"):
            # `fwdret_` and not `ret_`: ret_5/ret_10/ret_20 are PAST-return features.
            out[f"fwdret_{h}"] = np.where(nxt_open > 0, d["close"].shift(-h) / nxt_open - 1.0, np.nan)
    return pd.DataFrame(out)


def ca_contaminated(idx: pd.MultiIndex) -> np.ndarray:
    flags = pd.read_parquet(FLAGS)
    flags = flags[~flags["organic"]]
    day = idx.get_level_values(0).tz_convert("America/New_York").normalize().tz_localize(None).to_numpy()
    tick = idx.get_level_values(1).to_numpy()
    span = np.timedelta64(int(np.ceil(MAX_H / 2 * 7 / 5)) + 4, "D")
    bad = np.zeros(len(idx), bool)
    for t, g in flags.groupby("ticker"):
        dates = np.sort(pd.to_datetime(g["date"]).to_numpy())
        m = tick == t
        if m.any():
            st = day[m]
            bad[m] = np.searchsorted(dates, st + span, side="right") > np.searchsorted(dates, st, side="left")
    return bad


# ---------------------------------------------------------------- metrics

def _bars(df: pd.DataFrame) -> np.ndarray:
    return df.index.get_level_values(0).to_numpy()


def per_bar_spearman(bar, p, y) -> pd.Series:
    d = pd.DataFrame({"b": bar, "p": p, "y": y}).dropna()
    g = d.groupby("b")
    d["rp"] = g["p"].rank()
    d["ry"] = g["y"].rank()
    d["cp"] = d["rp"] - d.groupby("b")["rp"].transform("mean")
    d["cy"] = d["ry"] - d.groupby("b")["ry"].transform("mean")
    s = d.assign(xy=d["cp"] * d["cy"], xx=d["cp"] ** 2, yy=d["cy"] ** 2).groupby("b")[["xy", "xx", "yy"]].sum()
    n = d.groupby("b").size()
    r = s["xy"] / np.sqrt(s["xx"] * s["yy"])
    return r[(n >= MIN_XS) & (s["xx"] > 0) & (s["yy"] > 0)]


def per_bar_topk_excess(bar, p, y, k=TOP_K) -> pd.Series:
    d = pd.DataFrame({"b": bar, "p": p, "y": y}).dropna()
    n = d.groupby("b")["y"].transform("size")
    d = d[n >= MIN_XS]
    d["r"] = d.groupby("b")["p"].rank(ascending=False, method="first")
    top = d[d["r"] <= k].groupby("b")["y"].mean()
    return top - d.groupby("b")["y"].mean().reindex(top.index)


def block_boot(s: pd.Series, block: int, rng, n=2000) -> tuple[float, float, float]:
    v = s.sort_index().to_numpy()
    if len(v) < 2 * block:
        return np.nan, np.nan, np.nan
    starts = np.arange(len(v) - block + 1)
    nb = int(np.ceil(len(v) / block))
    means = np.empty(n)
    for i in range(n):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(block)).ravel()[: len(v)]
        means[i] = v[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(2 * min((means <= 0).mean(), (means >= 0).mean()))


# ---------------------------------------------------------------- data

def load(module: str) -> tuple[pd.DataFrame, list[str]]:
    if module == "mom":
        path, schema = MOM_MATRIX, set(pq.ParquetFile(MOM_MATRIX).schema.names)
        feats = [c for c in FEATURE_COLUMNS_4H if c in schema]
        extra = ["fwd_max_alpha", "fwd_atr_adj_return", "trend_persistence", "fwd_max_drawdown"]
    else:
        path = META_MATRIX
        feats = [c for c in pq.ParquetFile(META_MATRIX).schema.names if c not in META_DROP]
        extra = []
    df = pd.read_parquet(path, columns=feats + extra)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    tk = pd.Series(sorted(df.index.get_level_values(1).unique()))
    if len(tk) > N_TICKERS:
        keep = set(tk.sample(N_TICKERS, random_state=SEED))
        df = df[df.index.get_level_values(1).isin(keep)]
    t0 = time.time()
    labs = [x for x in (bar_labels(t) for t in sorted(df.index.get_level_values(1).unique())) if x is not None]
    lab = pd.concat(labs, ignore_index=True).set_index(["timestamp", "ticker"])
    for c in lab.columns:
        lab[c] = lab[c].astype("float32")
    df = df.join(lab, how="inner")
    print(f"  labels built in {time.time() - t0:.0f}s; joined rows {len(df):,}", flush=True)
    bad = ca_contaminated(df.index)
    print(f"  corporate-action guard dropped {int(bad.sum()):,} rows ({bad.mean():.3%})", flush=True)
    df = df[~bad]
    need = [f"mfe_{MAX_H}", f"fwdret_{MAX_H}"]
    before = len(df)
    df = df.dropna(subset=need)
    print(f"  rows without a complete {MAX_H}-bar forward window dropped: {before - len(df):,}", flush=True)
    df = df.dropna(subset=feats, how="all")
    return df.sort_index(), feats


def split_bars(df: pd.DataFrame):
    uniq = np.array(sorted(df.index.get_level_values(0).unique()))
    n = len(uniq)
    i_tr, i_va = int(n * 0.60), int(n * 0.78)
    tr_end, va_start = uniq[i_tr], uniq[i_tr + EMBARGO_BARS]
    va_end, te_start = uniq[i_va], uniq[i_va + EMBARGO_BARS]
    ts = df.index.get_level_values(0)
    return (df[ts <= tr_end], df[(ts >= va_start) & (ts <= va_end)], df[ts >= te_start],
            dict(train_end=str(tr_end), val=(str(va_start), str(va_end)), test_start=str(te_start),
                 test_end=str(uniq[-1])))


def rank_in_bar(df: pd.DataFrame, col: pd.Series) -> pd.Series:
    return col.groupby(level=0).rank(pct=True)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["mom", "meta"], default="mom")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    print(f"=== horizon grid: {args.module} ===", flush=True)
    df, feats = load(args.module)
    train, val, test, bounds = split_bars(df)
    print(f"  features={len(feats)} rows train={len(train):,} val={len(val):,} test={len(test):,}")
    print(f"  bounds {bounds}  (embargo {EMBARGO_BARS} decision bars each side)", flush=True)

    labels: dict[str, pd.Series] = {}
    for fam in ("mfe", "clean"):
        for name, h in TRAIN_H.items():
            labels[f"{fam}_{name}"] = rank_in_bar(df, df[f"{fam}_{h}"])
    if args.module == "mom":
        r = lambda c, asc=True: df[c].groupby(level=0).rank(pct=True, ascending=asc)  # noqa: E731
        labels["M0_current_25bar"] = (0.40 * r("fwd_max_alpha") + 0.25 * r("fwd_atr_adj_return")
                                      + 0.20 * r("trend_persistence") + 0.15 * r("fwd_max_drawdown", False))
    base = labels["mfe_10d"]
    for s in range(N_PERM):
        perm_rng = np.random.default_rng(1000 + s)
        labels[f"NULL_perm_{s}"] = base.groupby(level=0).transform(
            lambda x, g=perm_rng: pd.Series(g.permutation(x.to_numpy()), index=x.index))

    preds_val, preds_test, iters = {}, {}, {}
    Xtr, Xva, Xte = train[feats], val[feats], test[feats]
    dte, dva_x = xgb.DMatrix(Xte), xgb.DMatrix(Xva)
    t0 = time.time()
    for name, lab in labels.items():
        ytr, yva = lab.reindex(train.index), lab.reindex(val.index)
        ok, okv = ytr.notna().to_numpy(), yva.notna().to_numpy()
        bst = xgb.train(PARAMS, xgb.DMatrix(Xtr[ok], label=ytr[ok]), num_boost_round=N_ROUNDS,
                        evals=[(xgb.DMatrix(Xva[okv], label=yva[okv]), "v")],
                        early_stopping_rounds=30, verbose_eval=False)
        preds_val[name], preds_test[name] = bst.predict(dva_x), bst.predict(dte)
        iters[name] = int(bst.best_iteration or 0)
        print(f"  fit {name:18s} best_iter={iters[name]:3d}  ({time.time() - t0:.0f}s)", flush=True)

    def _stack(preds, frame, names):
        b = _bars(frame)
        return np.mean([pd.Series(preds[n]).groupby(b).rank(pct=True).to_numpy() for n in names], axis=0)

    for store, frame in ((preds_val, val), (preds_test, test)):
        store["STACK_mfe_all"] = _stack(store, frame, [f"mfe_{n}" for n in TRAIN_H])
        store["STACK_mfe_2d_30d"] = _stack(store, frame, ["mfe_2d", "mfe_30d"])

    evals = {f"{fam}_{e}": (f"{fam}_{h}", h) for fam in ("mfe", "clean", "fwdret") for e, h in EVAL_H.items()}
    rows = []
    for cand in preds_test:
        for ename, (col, h) in evals.items():
            rv = per_bar_spearman(_bars(val), preds_val[cand], val[col].to_numpy())
            rt = per_bar_spearman(_bars(test), preds_test[cand], test[col].to_numpy())
            kt = per_bar_topk_excess(_bars(test), preds_test[cand], test[col].to_numpy())
            kv = per_bar_topk_excess(_bars(val), preds_val[cand], val[col].to_numpy())
            lo, hi, p = block_boot(kt, h, rng, n=1000)
            rows.append(dict(candidate=cand, eval=ename, val_rho=float(rv.mean()), test_rho=float(rt.mean()),
                             val_top3=float(kv.mean()), test_top3=float(kt.mean()), top3_lo=lo, top3_hi=hi,
                             top3_p=p, test_bars=int(len(kt))))
    res = pd.DataFrame(rows)

    # Paired, block-bootstrapped comparisons on the tradeable target.
    paired = []
    for ename, (col, h) in evals.items():
        if not ename.startswith("fwdret_"):
            continue
        sub = res[(res["eval"] == ename) & ~res["candidate"].str.startswith("NULL")]
        best_val = sub.sort_values("val_rho", ascending=False).iloc[0]["candidate"]
        matched = {20: "mfe_10d", 40: "mfe_20d", 60: "mfe_30d"}[h]
        ref = "M0_current_25bar" if args.module == "mom" else matched
        for a, b in ((ref, best_val), (matched, best_val), (ref, "STACK_mfe_all"), (ref, "mfe_2d")):
            if a == b:
                continue
            ka = per_bar_topk_excess(_bars(test), preds_test[a], test[col].to_numpy())
            kb = per_bar_topk_excess(_bars(test), preds_test[b], test[col].to_numpy())
            ra = per_bar_spearman(_bars(test), preds_test[a], test[col].to_numpy())
            rb = per_bar_spearman(_bars(test), preds_test[b], test[col].to_numpy())
            dk = (kb - ka).dropna()
            dr = (rb - ra).dropna()
            klo, khi, kp = block_boot(dk, h, rng, n=1000)
            rlo, rhi, rp = block_boot(dr, h, rng, n=1000)
            paired.append(dict(eval=ename, a=a, b=b, d_top3=float(dk.mean()), top3_ci=(klo, khi), top3_p=kp,
                               d_rho=float(dr.mean()), rho_ci=(rlo, rhi), rho_p=rp))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for ename in evals:
        sub = res[res["eval"] == ename].sort_values("val_rho", ascending=False)
        print(f"\n--- eval {ename}  (winner picked on VAL rho; top{TOP_K} excess is test, "
              f"{'% share return' if ename.startswith('fwdret') else 'ATR'}) ---")
        show = sub.copy()
        if ename.startswith("fwdret"):
            for c in ("val_top3", "test_top3", "top3_lo", "top3_hi"):
                show[c] = show[c] * 100
        print(show.drop(columns=["eval"]).round(4).to_string(index=False))
    print("\n=== paired comparisons on the tradeable target (b - a), block bootstrap ===")
    for r_ in paired:
        print(f"  {r_['eval']:7s} {r_['b']:18s} - {r_['a']:18s}: top3 {r_['d_top3'] * 100:+.2f}pp "
              f"[{r_['top3_ci'][0] * 100:+.2f},{r_['top3_ci'][1] * 100:+.2f}] p={r_['top3_p']:.3f} | "
              f"rho {r_['d_rho']:+.4f} [{r_['rho_ci'][0]:+.4f},{r_['rho_ci'][1]:+.4f}] p={r_['rho_p']:.3f}")
    out = DATA / f"horizon_grid_{args.module}.json"
    out.write_text(json.dumps(dict(bounds=bounds, features=len(feats), iters=iters, results=rows,
                                   paired=paired), indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
