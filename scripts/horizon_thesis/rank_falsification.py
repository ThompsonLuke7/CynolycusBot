"""Falsification + factor audit of the momentum ranking, on 3.5 years of OOF scores.

Three questions about the same top-3 excess number, at holds of 5-30 sessions:

  NULL      shuffle the score within each bar. The pipeline, the universe, the
            days and the drift are all unchanged, so whatever excess survives is
            the noise floor of the measurement rather than signal. (arXiv
            2604.15531's falsification idea, applied to the evaluation itself.)

  FACTORS   regress each bar's forward returns cross-sectionally on exposures
            known before entry -- 20/60/120d past return, 20d realised vol, log
            dollar volume, 60d beta, distance to the 252d high -- and re-measure
            the excess on the RESIDUALS. The same paper's finding is that ML
            cross-sectional models often reproduce known factors: significant
            gross, no alpha. If the excess dies here, the ranker is a repackaged
            factor bet and the factor sort is the cheaper way to own it.

  BASELINES the one-line sorts themselves (past 20d return, past 60d return,
            nearest 252d high) measured identically, so "is the ML worth it"
            has a number rather than an opinion.

CIs are moving-block bootstraps over decision bars (block = the hold), because
forward windows of neighbouring bars overlap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from build_decision_panel import load_panel  # noqa: E402
from run_horizon_grid import block_boot  # noqa: E402

DATA = REPO / "research/execution_quality/data"
HOLDS = [5, 10, 15, 20, 30]
TOP_K = 3
MIN_XS = 200
N_NULL = 10
FACTORS = ["past_ret_20", "past_ret_60", "past_ret_120", "past_vol_20",
           "log_dollar_vol_20", "beta_60", "dist_252_high"]
SEED = 11


def topk_excess(df: pd.DataFrame, score: str, target: str, k: int = TOP_K) -> pd.Series:
    d = df[[score, target]].copy()
    d["bar"] = df["timestamp"].to_numpy()
    d = d.dropna()
    n = d.groupby("bar")[target].transform("size")
    d = d[n >= MIN_XS]
    d["r"] = d.groupby("bar")[score].rank(ascending=False, method="first")
    top = d[d["r"] <= k].groupby("bar")[target].mean()
    return (top - d.groupby("bar")[target].mean().reindex(top.index)).dropna()


def residualise(df: pd.DataFrame, target: str) -> pd.Series:
    """Cross-sectional OLS residuals of `target` on FACTORS, one regression per bar."""
    cols = FACTORS + [target]
    sub = df[["timestamp"] + cols].dropna()
    out = np.full(len(sub), np.nan)
    x_all = sub[FACTORS].to_numpy(float)
    y_all = sub[target].to_numpy(float)
    bars = sub["timestamp"].to_numpy()
    order = np.argsort(bars, kind="stable")
    starts = np.searchsorted(bars[order], np.unique(bars))
    bounds = list(starts) + [len(order)]
    for i in range(len(bounds) - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        if len(idx) < len(FACTORS) + 20:
            continue
        x = np.column_stack([np.ones(len(idx)), x_all[idx]])
        y = y_all[idx]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[idx] = y - x @ beta
    res = pd.Series(out, index=sub.index, name=f"resid_{target}")
    return res


def main() -> None:
    rng = np.random.default_rng(SEED)
    panel = load_panel(require=["mom_score"])
    panel = panel[panel["xs_size"] >= MIN_XS]
    print(f"panel rows {len(panel):,}  bars {panel['timestamp'].nunique():,}  "
          f"tickers {panel['ticker'].nunique():,}  "
          f"({panel['entry_session'].min().date()} .. {panel['entry_session'].max().date()})")
    print(f"factors: {FACTORS}\n")

    rows = []
    for h in HOLDS:
        tgt = f"fwdret_{h}"
        sub = panel.dropna(subset=[tgt]).copy()
        raw = topk_excess(sub, "mom_score", tgt)
        lo, hi, p = block_boot(raw, h, rng, n=1000)

        resid = residualise(sub, tgt)
        sub_r = sub.loc[resid.index].assign(**{f"resid_{h}": resid.to_numpy()})
        res_ex = topk_excess(sub_r, "mom_score", f"resid_{h}")
        rlo, rhi, rp = block_boot(res_ex, h, rng, n=1000)

        nulls = []
        for s in range(N_NULL):
            g = np.random.default_rng(500 + s)
            shuffled = sub.groupby("timestamp")["mom_score"].transform(
                lambda x, gg=g: gg.permutation(x.to_numpy()))
            nulls.append(float(topk_excess(sub.assign(null_score=shuffled), "null_score", tgt).mean()))
        base = {b: float(topk_excess(sub, b, tgt).mean()) for b in
                ("past_ret_20", "past_ret_60", "dist_252_high")}

        print(f"--- hold {h:2d} sessions  ({len(raw):,} bars, {len(sub):,} rows) ---")
        print(f"  mom_score top{TOP_K} excess      {raw.mean() * 100:+6.2f}%  "
              f"[{lo * 100:+6.2f},{hi * 100:+6.2f}] p={p:.3f}")
        print(f"  ... on factor RESIDUALS        {res_ex.mean() * 100:+6.2f}%  "
              f"[{rlo * 100:+6.2f},{rhi * 100:+6.2f}] p={rp:.3f}")
        print(f"  NULL shuffled score            {np.mean(nulls) * 100:+6.2f}%  "
              f"[{np.percentile(nulls, 2.5) * 100:+6.2f},{np.percentile(nulls, 97.5) * 100:+6.2f}] "
              f"({N_NULL} seeds)")
        for b, v in base.items():
            print(f"  baseline sort {b:16s} {v * 100:+6.2f}%")
        print()
        rows.append(dict(hold=h, bars=int(len(raw)), rows=int(len(sub)),
                         raw=float(raw.mean()), raw_ci=[lo, hi], raw_p=p,
                         resid=float(res_ex.mean()), resid_ci=[rlo, rhi], resid_p=rp,
                         null_mean=float(np.mean(nulls)),
                         null_ci=[float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))],
                         baselines=base))
    out = DATA / "rank_falsification.json"
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
