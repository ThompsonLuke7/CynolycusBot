"""Backtest: does gating entries on rank RE-IMPROVEMENT (+ a quality floor) beat
just holding whatever is top-K each bar?

Motivation (CAR vs MU):
  * CAR kept a 0.99 combo even AT its parabolic top — "still ranked" re-entries
    would have pyramided into the blow-off. The quality head, not the combo,
    de-ranked it. So re-entry should require a *fresh* rank improvement + quality.
  * MU rode for months at mid-pack rank, so the question is whether the moments it
    crosses back into the top tier ("re-acceleration") are the good entries.

Cohorts (each = a set of (bar, ticker) entry events), scored with the matrix's
own forward labels (fwd_close_return / fwd_max_return):
  A) top-K           : every bar with rank <= K            (naive "always hold")
  B) cross-in        : rank <= K AND prior bar rank > K     (fresh entry/re-accel)
  C) cross-in + qual : B AND s_quality >= floor             (the proposed gate)

  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/reentry_gate_backtest.py --k 20 --since 2025-09-01 --qual 0.5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from signals.meta_context.meta_ranker.score import score_frame, DEFAULT_MATRIX


def _summ(name: str, d: pd.DataFrame) -> dict:
    return {
        "cohort": name,
        "n": len(d),
        "med_fwd_max": d["fwd_max_return"].median(),
        "med_fwd_close": d["fwd_close_return"].median(),
        "mean_fwd_close": d["fwd_close_return"].mean(),
        "hit_10pct_close": (d["fwd_close_return"] >= 0.10).mean(),
        "loss_rate_close": (d["fwd_close_return"] < 0).mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="top-K rank band")
    ap.add_argument("--qual", type=float, default=0.5, help="s_quality floor for the gate")
    ap.add_argument("--since", default="2025-09-01")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    args = ap.parse_args()

    m = pd.read_parquet(args.matrix).reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    m = m[m["timestamp"] >= args.since].copy()
    # keep only bars with realized forward labels
    m = m[m["fwd_close_return"].notna() & m["fwd_max_return"].notna()]
    print(f"scoring {m['timestamp'].nunique()} labeled bars since {args.since} "
          f"({len(m):,} rows) ...")
    sc = score_frame(m)
    sc["rank"] = sc.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    sc = sc.sort_values(["ticker", "timestamp"])
    sc["prev_rank"] = sc.groupby("ticker")["rank"].shift(1)

    K, Q = args.k, args.qual
    topk = sc[sc["rank"] <= K]
    crossin = topk[(topk["prev_rank"] > K) | (topk["prev_rank"].isna())]
    gated = crossin[crossin["s_quality"] >= Q]

    # re-entries only: a cross-in that had previously been top-K earlier in history
    seen_before = sc[sc["rank"] <= K].groupby("ticker")["timestamp"].min().rename("first_topk")
    ci = crossin.merge(seen_before, on="ticker", how="left")
    reentries = ci[ci["timestamp"] > ci["first_topk"]]
    reentries_gated = reentries[reentries["s_quality"] >= Q]

    rows = [
        _summ(f"A) top-{K} (all bars)", topk),
        _summ(f"B) cross-in (fresh)", crossin),
        _summ(f"C) cross-in + qual>={Q}", gated),
        _summ("   re-entries only", reentries),
        _summ(f"   re-entries + qual>={Q}", reentries_gated),
    ]
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for c in ["med_fwd_max", "med_fwd_close", "mean_fwd_close"]:
        out[c] = (out[c] * 100).round(1)
    for c in ["hit_10pct_close", "loss_rate_close"]:
        out[c] = (out[c] * 100).round(0)
    print("\n=== forward outcomes by entry rule (returns in %, hold = label horizon) ===")
    print(out.to_string(index=False))
    print("\nReading: higher med/mean fwd_close and hit_10pct, lower loss_rate = better entries.")


if __name__ == "__main__":
    main()
