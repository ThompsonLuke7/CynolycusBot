"""Balance-envelope edge fade vs matched controls on 1-minute RTH history (research sketch).

Causal at every bar i (bar i is complete; entry is bar i+1's open):
  envelope  hi/lo = max high / min low of bars [i-W, i-1]          (excludes bar i)
  balance   Kaufman efficiency |c[i-1]-c[i-W]| / sum|dc| < ER_MAX  and
            width (hi-lo)/mid in [W_MIN, W_MAX]                    and
            both hi and lo were set >= AGE bars ago (edges established, not the live impulse)
  FADE long : low[i] <= lo + EDGE*w, close[i] >= lo, close[i] > open[i]
  FADE short: high[i] >= hi - EDGE*w, close[i] <= hi, close[i] < open[i]
  bracket   stop = lo - STOP*w (long) / hi + STOP*w (short); target = lo + TGT*w / hi - TGT*w
  flat 15:55 ET; one position at a time; entries 10:30-15:30 ET.
Controls (same balance gate, same bracket distances measured from the actual entry):
  RANDOM  : at a random balance bar, random side, same stop/target distances in price units
  BREAKOUT: same signal bars as FADE, opposite side, distances mirrored
Prices: Alpaca 1m bars (Data/raw/qqq/qqq_intraday_1min.parquet). Not a profitability claim.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

W, ER_MAX, W_MIN, W_MAX, AGE, EDGE, STOP = 60, 0.30, 0.0030, 0.0150, 10, 0.20, 0.25
MIN_TOUCH = 0  # --min-touch: >=N prior separate visits to the tested edge (and >=1 to the other); 0 = off


def visits(h, l, lo, hi, w, side):
    """Separate visits to the tested edge zone; a new visit counts only after price returned past the midline."""
    mid = lo + 0.5 * w
    n, armed = 0, True
    for hh, ll in zip(h, l):
        in_zone = ll <= lo + EDGE * w if side == 1 else hh >= hi - EDGE * w
        past_mid = hh >= mid if side == 1 else ll <= mid
        if in_zone and armed:
            n += 1
            armed = False
        elif past_mid:
            armed = True
    return n


def sim_exit(h, l, c, k0, side, stop, target, last):
    for k in range(k0, last + 1):
        if side == 1:
            if l[k] <= stop:
                return stop, k, "stop"
            if h[k] >= target:
                return target, k, "target"
        else:
            if h[k] >= stop:
                return stop, k, "stop"
            if l[k] <= target:
                return target, k, "target"
    return c[last], last, "eod"


def run(df: pd.DataFrame, tgt: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for d, day in df.groupby(df.index.date):
        day = day.between_time("09:30", "15:59")
        if len(day) < 300:
            continue
        o, h, l, c = (day[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        tod = day.index.time
        n = len(day)
        last = max(k for k in range(n) if tod[k] <= pd.Timestamp("15:55").time())
        busy = {"fade": -1, "breakout": -1, "random": -1}
        for i in range(W, n - 1):
            if not (pd.Timestamp("10:30").time() <= tod[i] <= pd.Timestamp("15:30").time()):
                continue
            hw, lw = h[i - W:i], l[i - W:i]
            hi, lo = hw.max(), lw.min()
            mid = (hi + lo) / 2
            w = hi - lo
            width = w / mid
            path = np.abs(np.diff(c[i - W:i])).sum()
            er = abs(c[i - 1] - c[i - W]) / path if path > 0 else 1.0
            age_hi = W - 1 - int(np.argmax(hw))
            age_lo = W - 1 - int(np.argmin(lw))
            if not (er < ER_MAX and W_MIN <= width <= W_MAX and age_hi >= AGE and age_lo >= AGE):
                continue
            side = 0
            if l[i] <= lo + EDGE * w and c[i] >= lo and c[i] > o[i]:
                side, stop, target = 1, lo - STOP * w, lo + tgt * w
            elif h[i] >= hi - EDGE * w and c[i] <= hi and c[i] < o[i]:
                side, stop, target = -1, hi + STOP * w, hi - tgt * w
            entry = o[i + 1]
            if side and MIN_TOUCH and not (visits(hw, lw, lo, hi, w, side) >= MIN_TOUCH and visits(hw, lw, lo, hi, w, -side) >= 1):
                side = 0
            if side and i + 1 > busy["fade"]:
                if (side == 1 and stop < entry < target) or (side == -1 and target < entry < stop):
                    px, k, why = sim_exit(h, l, c, i + 1, side, stop, target, last)
                    out.append(dict(date=d, arm="fade", side=side, t=str(tod[i]), entry=entry, exit=px, why=why, width=width,
                                    bps=side * (px - entry) / entry * 1e4, bars=k - i))
                    busy["fade"] = k
                    # breakout control: opposite side, mirrored distances
                    if i + 1 > busy["breakout"]:
                        s2 = -side
                        st2 = entry - s2 * abs(target - entry)
                        tg2 = entry + s2 * abs(entry - stop)
                        px2, k2, why2 = sim_exit(h, l, c, i + 1, s2, st2, tg2, last)
                        out.append(dict(date=d, arm="breakout", side=s2, t=str(tod[i]), entry=entry, exit=px2, why=why2, width=width,
                                        bps=s2 * (px2 - entry) / entry * 1e4, bars=k2 - i))
                        busy["breakout"] = k2
            # random control: ~ same frequency, at any balance bar, random side, fade-like bracket distances
            if i + 1 > busy["random"] and rng.random() < 0.03:
                s3 = 1 if rng.random() < 0.5 else -1
                st3 = entry - s3 * (EDGE / 2 + STOP) * w
                tg3 = entry + s3 * (tgt - EDGE / 2) * w
                px3, k3, why3 = sim_exit(h, l, c, i + 1, s3, st3, tg3, last)
                out.append(dict(date=d, arm="random", side=s3, t=str(tod[i]), entry=entry, exit=px3, why=why3, width=width,
                                bps=s3 * (px3 - entry) / entry * 1e4, bars=k3 - i))
                busy["random"] = k3
    return pd.DataFrame(out)


def summarize(tr: pd.DataFrame, costs=(2.0, 12.0)) -> pd.DataFrame:
    rows = []
    tr = tr.assign(year=pd.to_datetime(tr.date).dt.year)
    for (arm, yr), g in list(tr.groupby(["arm", "year"])) + [((a, "ALL"), g) for a, g in tr.groupby("arm")]:
        sd = g.bps.std(ddof=1)
        rows.append(dict(arm=arm, year=yr, n=len(g), days=g.date.nunique(), win=(g.bps > 0).mean(),
                         gross_bps=g.bps.mean(), se=sd / np.sqrt(len(g)), mde80=2.8 * sd / np.sqrt(len(g)),
                         **{f"net{c:g}": g.bps.mean() - c for c in costs},
                         stop_rate=(g.why == "stop").mean(), med_bars=g.bars.median()))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="Data/raw/qqq/qqq_intraday_1min.parquet")
    ap.add_argument("--extra-jsonl", default=None, help="append one day of archived 1m bars")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--tgt", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-touch", type=int, default=0)
    a = ap.parse_args()
    global MIN_TOUCH
    MIN_TOUCH = a.min_touch
    df = pd.read_parquet(a.parquet)
    df = df[df.symbol == a.symbol] if "symbol" in df else df
    df["t"] = pd.to_datetime(df.timestamp, utc=True).dt.tz_convert("America/New_York")
    df = df.drop_duplicates("t").set_index("t").sort_index()[["open", "high", "low", "close"]]
    if a.extra_jsonl:
        x = pd.DataFrame([json.loads(r) for r in open(a.extra_jsonl)])
        x = x[x.symbol == a.symbol]
        x["t"] = pd.to_datetime(x.timestamp, utc=True).dt.tz_convert("America/New_York")
        x = x.drop_duplicates("t").set_index("t").sort_index()[["open", "high", "low", "close"]]
        df = pd.concat([df[df.index < x.index.min()], x])
    tr = run(df, a.tgt)
    pd.set_option("display.width", 220)
    print(f"symbol={a.symbol} tgt={a.tgt} min_touch={MIN_TOUCH} sessions={len(set(df.index.date))} {df.index.min().date()}..{df.index.max().date()}")
    print(summarize(tr).round(3).to_string(index=False))
    if a.out:
        tr.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
