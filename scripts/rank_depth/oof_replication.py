"""Replicate the top-k finding on 3.5 years instead of two months.

The live study had 67-72 decision bars because ranked signals were reconstructed
from live order history, which starts 2026-07-01. But the Meta matrix is BUILT
from momentum's and HTF's walk-forward out-of-fold predictions -- `mom_score`
and `htf_score`, leak-free with a 21-day embargo
(`build_meta_ranker_matrix.py:23,46-49`). Those are exactly the historical ranks
the live study lacked, they cost nothing to use, and they span 2022-11-14 to
2026-05-14.

What this measures is the SCORE's ordering power. It is not a full replay of the
live module: the live runner also applies a candidate filter and a 1H entry
trigger, neither of which is reproduced here. So it answers "does the ranking
order forward returns, out of sample, over 3.5 years" -- which is the claim the
`top_n` change rests on -- and not "what would the module have returned".

Entry/exit/control are identical to the live study: entry at the open of the
first session after the bar, exit at the close H trading days later, reported as
excess over an equal-weight draw from that bar's own scored cross-section.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet"
BARS = REPO / "Data/shared/bars/1d"
HOLDS = [5, 8, 10, 15]
DEPTHS = [1, 2, 3, 5, 10]
MIN_XS = 200          # bars with a thinner cross-section cannot support a top-10


def forward_table(tickers: list[str], holds: list[int]) -> pd.DataFrame:
    """Long frame of (ticker, session_date, ret_h...) for a merge_asof join.

    Built once and merged rather than masked per ticker: a per-ticker boolean
    mask over the full matrix is O(tickers x rows x holds) and does not finish.
    `session_date` is the ENTRY session -- each row says "entering at this
    session's open and exiting h sessions later returns ret_h".
    """
    frames = []
    for i, t in enumerate(tickers, 1):
        p = BARS / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["timestamp", "open", "close"])
        if len(d) < 40:
            continue
        sess = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
            "America/New_York").dt.normalize().dt.tz_localize(None)
        o = d["open"].to_numpy(float)
        c = d["close"].to_numpy(float)
        out = {"ticker": t, "session_date": sess.to_numpy()}
        for h in holds:
            ex = np.full(len(d), np.nan)
            if len(d) >= h:
                ex[:len(d) - h + 1] = c[h - 1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"ret_{h}"] = np.where(o > 0, ex / o - 1.0, np.nan)
        frames.append(pd.DataFrame(out))
        if i % 300 == 0:
            print(f"  forward table {i}/{len(tickers)}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["mom", "htf"], default="mom")
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()
    col = f"{args.module}_score"

    m = pd.read_parquet(MATRIX, columns=[col]).reset_index()
    m = m.dropna(subset=[col])
    size = m.groupby("timestamp")["ticker"].transform("size")
    m = m[size >= MIN_XS].copy()
    m["decision_day"] = m["timestamp"].dt.tz_convert("America/New_York").dt.normalize(
    ).dt.tz_localize(None)
    print(f"{col}: {len(m):,} rows over {m['timestamp'].nunique():,} bars "
          f"({m['decision_day'].min().date()} .. {m['decision_day'].max().date()})")

    tickers = sorted(m["ticker"].unique())
    print(f"building forward table for {len(tickers)} tickers ...")
    ft = forward_table(tickers, HOLDS)
    print(f"  forward rows: {len(ft):,} over {ft['ticker'].nunique()} tickers")

    # Join each decision to the FIRST session strictly after it. merge_asof with
    # direction="forward" on (decision_day + 1 day) does exactly that: no
    # look-ahead, and it lands on the next trading session across weekends and
    # holidays without needing a calendar.
    m["_key"] = m["decision_day"] + pd.Timedelta(days=1)
    m = m.sort_values("_key")
    ft = ft.sort_values("session_date")
    m = pd.merge_asof(m, ft, left_on="_key", right_on="session_date",
                      by="ticker", direction="forward",
                      tolerance=pd.Timedelta(days=7))
    m = m.drop(columns=["_key"])
    m["rank"] = m.groupby("timestamp")[col].rank(ascending=False, method="first")
    m.to_parquet(DATA / f"oof_rank_depth_{args.module}.parquet")
    print(f"  joined: ret_10 present on {m['ret_10'].notna().mean():.1%} of rows\n")

    rng = np.random.default_rng(53)
    print(f"{'hold':>4s} {'k':>3s} {'n':>7s} {'bars':>6s} {'topk%':>7s} "
          f"{'ctrl%':>7s} {'excess%':>8s} {'CI':>17s} {'p':>6s} {'perm p':>8s}")
    for h in HOLDS:
        rc = f"ret_{h}"
        s = m[m[rc].notna()]
        if len(s) < 500:
            continue
        ctrl = s.groupby("timestamp")[rc].mean()      # that bar's own cross-section
        pool = s[s["rank"] <= 10]
        groups = [g[rc].to_numpy() for _t, g in pool.groupby("timestamp")
                  if len(g) >= 4]
        for k in DEPTHS:
            top = s[s["rank"] <= k]
            if len(top) < 100:
                continue
            per_bar = top.groupby("timestamp")[rc].mean()
            aligned = ctrl.reindex(per_bar.index)
            diff = (per_bar - aligned).dropna()
            obs = float(diff.mean())
            draws = np.array([rng.choice(diff.values, len(diff), replace=True).mean()
                              for _ in range(3000)])
            lo, hi = np.percentile(draws, [2.5, 97.5])
            p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
            # Permutation: is the top-k better than a random draw from the SAME
            # bar's top-10 pick set? Holds pick set, days and drift fixed.
            obs_top = float(pool[pool["rank"] <= k].groupby(
                "timestamp")[rc].mean().mean())
            nd = np.empty(args.perm)
            for i in range(args.perm):
                nd[i] = float(np.mean([rng.permutation(g)[:k].mean()
                                       for g in groups]))
            pp = (np.sum(nd >= obs_top) + 1) / (args.perm + 1)
            star = "*" if p < 0.05 else " "
            pstar = "*" if pp < 0.05 else " "
            print(f"{h:4d} {k:3d} {len(top):7d} {len(diff):6d} "
                  f"{per_bar.mean() * 100:7.2f} {aligned.mean() * 100:7.2f} "
                  f"{obs * 100:8.2f} [{lo * 100:6.2f},{hi * 100:6.2f}] "
                  f"{p:6.3f}{star} {pp:7.4f}{pstar}")
        print()


if __name__ == "__main__":
    main()
