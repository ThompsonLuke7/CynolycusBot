"""Do backward-looking "price recap" catalysts predict anything?

Question raised 2026-08-03: a large share of live catalyst records only restate
a move that already happened ("Shares Skyrocket", "Stock Price Up 10.5%",
"Gains 2.45% Today"). If those carry no forward information they are noise in
the catalyst score, and the honest answer is to measure rather than to guess.

Method (leakage-controlled):
  * entry  = close of the FIRST daily bar strictly AFTER the record timestamp,
             so nothing is priced off the bar the news landed in;
  * forward return measured at +1, +3 and +5 trading days from that entry;
  * records are bucketed as price_recap / forward_looking by headline pattern;
  * a same-period, same-universe baseline of ALL records is reported alongside,
    because absolute forward returns over a 7-week window say little on their own.

Run:
  PYTHONPATH=. .venv/bin/python research/news_catalysts/measure_backward_looking_catalysts.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signals.news.information_direction import classify_information_direction

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "signals/news/data/processed/live_catalyst_records.parquet"
BARS_1D = REPO / "Data/shared/bars/1d"

def bucket(headline: str) -> str:
    """Delegates to the shipped classifier so research and live agree."""
    return classify_information_direction(headline)


def load_bars(ticker: str) -> pd.DataFrame | None:
    p = BARS_1D / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["timestamp", "close"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def forward_returns(df: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    """Attach +N trading-day returns from the first bar strictly after the record."""
    out = {f"fwd_{h}d": [] for h in horizons}
    cache: dict[str, pd.DataFrame | None] = {}
    for tk, ts in zip(df["ticker"], df["timestamp"]):
        if tk not in cache:
            cache[tk] = load_bars(tk)
        bars = cache[tk]
        if bars is None or bars.empty:
            for h in horizons:
                out[f"fwd_{h}d"].append(np.nan)
            continue
        after = bars.index[bars.index > pd.Timestamp(ts)]
        if len(after) == 0:
            for h in horizons:
                out[f"fwd_{h}d"].append(np.nan)
            continue
        i = bars.index.get_loc(after[0])
        entry = float(bars["close"].iloc[i])
        for h in horizons:
            j = i + h
            if entry > 0 and j < len(bars):
                out[f"fwd_{h}d"].append(float(bars["close"].iloc[j]) / entry - 1.0)
            else:
                out[f"fwd_{h}d"].append(np.nan)
    for k, v in out.items():
        df[k] = v
    return df


def main() -> int:
    df = pd.read_parquet(LEDGER)
    df["bucket"] = [bucket(h) for h in df["headline"]]
    print(f"records: {len(df):,}   span {df.timestamp.min()} -> {df.timestamp.max()}")
    print("\nbucket mix:")
    for b, c in df["bucket"].value_counts().items():
        print(f"   {b:16s} {c:6,d}  ({100 * c / len(df):5.1f}%)")

    df = forward_returns(df)
    have = df.dropna(subset=["fwd_1d"])
    print(f"\nwith forward bars: {len(have):,} / {len(df):,}")

    print("\nforward return by bucket (mean / median / win-rate / n):")
    header = f"   {'bucket':16s} {'+1d mean':>9s} {'+1d med':>9s} {'+1d win':>8s} "
    header += f"{'+3d mean':>9s} {'+5d mean':>9s} {'n':>7s}"
    print(header)
    for b in ["price_recap", "forward_looking", "mixed", "other"]:
        s = have[have["bucket"] == b]
        if s.empty:
            continue
        print(f"   {b:16s} {s.fwd_1d.mean() * 100:8.3f}% {s.fwd_1d.median() * 100:8.3f}% "
              f"{(s.fwd_1d > 0).mean() * 100:7.1f}% {s.fwd_3d.mean() * 100:8.3f}% "
              f"{s.fwd_5d.mean() * 100:8.3f}% {len(s):7,d}")
    b = have
    print(f"   {'ALL (baseline)':16s} {b.fwd_1d.mean() * 100:8.3f}% {b.fwd_1d.median() * 100:8.3f}% "
          f"{(b.fwd_1d > 0).mean() * 100:7.1f}% {b.fwd_3d.mean() * 100:8.3f}% "
          f"{b.fwd_5d.mean() * 100:8.3f}% {len(b):7,d}")

    # Does the catalyst SCORE separate winners inside each bucket? If a bucket's
    # score has no monotone relationship with forward return, that bucket is not
    # contributing usable ranking information.
    print("\ncatalyst_score decile -> +1d mean return (does the score rank inside the bucket?):")
    for bkt in ["price_recap", "forward_looking"]:
        s = have[have["bucket"] == bkt].copy()
        if len(s) < 200:
            continue
        s["decile"] = pd.qcut(s["catalyst_score"], 10, labels=False, duplicates="drop")
        g = s.groupby("decile")["fwd_1d"].agg(["mean", "count"])
        top, bot = g["mean"].iloc[-1], g["mean"].iloc[0]
        corr = s[["catalyst_score", "fwd_1d"]].corr(method="spearman").iloc[0, 1]
        print(f"   {bkt:16s} bottom {bot * 100:+7.3f}%  top {top * 100:+7.3f}%  "
              f"spread {(top - bot) * 100:+7.3f}pp  spearman {corr:+.4f}  n={len(s):,}")

    # A high-score subset is what actually reaches the modules.
    print("\nhigh-conviction slice (catalyst_score >= 0.70):")
    hi = have[have["catalyst_score"] >= 0.70]
    for bkt in ["price_recap", "forward_looking", "mixed", "other"]:
        s = hi[hi["bucket"] == bkt]
        if len(s) < 30:
            continue
        print(f"   {bkt:16s} +1d {s.fwd_1d.mean() * 100:+7.3f}%  "
              f"win {(s.fwd_1d > 0).mean() * 100:5.1f}%  n={len(s):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
