"""Steps 2 and 3 of the intraday_structure audit.

Step 1 (`matched_control.py`) retracted "negative selection": paired against a
time-of-day and run-up matched control, CONFIRMED setups are indistinguishable
from comparable moments, while DECLINED ones are WORSE than theirs. So the gate
does discriminate; what it does not do is produce setups with an edge.

STEP 2 -- the gate. 98.8% of abstentions are `invalidation_wider_than_max_atr`,
i.e. the stop the structure implied was wider than `max_invalidation_atr` (2.0).
The proposal was to widen the cap and SIZE DOWN instead of refusing, which is
only right if the declined setups still pay per unit of risk.

  INSTRUMENTATION GAP: `proposed_invalidation`, `proposed_target` and
  `reward_risk` are 0% populated on these events -- the engine abstains before
  it computes them, so the width its most common decision turns on is never
  recorded. `room_to_support_atr` (long) / `room_to_resistance_atr` (short) is
  used as a PROXY: the invalidation sits beyond that level, so it is a lower
  bound on the implied stop. Everything in step 2 inherits that proxy.

STEP 3 -- entry timing. The exhaustion hypothesis: the engine confirms after the
move is already made. Measured as forward excursion against the run-up already
in hand at the decision minute.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/intraday_audit"))

from matched_control import BARS, CLOSED, EVENTS, covariates, excursion  # noqa: E402

HORIZONS = (30, 60)
WIDTH_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]
TRAIL_BUCKETS = [(-1e9, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.0), (1.0, 1e9)]


def load_declined(have: set[str], sample: int) -> list[dict]:
    seen, out = set(), []
    for line in EVENTS.open():
        if '"setup_abstention"' not in line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event_type") != "setup_abstention":
            continue
        p = r.get("payload") or {}
        if p.get("no_trade_reason") != "invalidation_wider_than_max_atr":
            continue
        t = r.get("ticker")
        if t not in have:
            continue
        ts = pd.Timestamp(r["event_time"])
        key = (p.get("setup_id"), t, str(ts.date()))
        if key in seen:
            continue
        seen.add(key)
        d = str(p.get("direction") or "").lower()
        width = p.get("room_to_support_atr") if d == "long" else p.get("room_to_resistance_atr")
        out.append({"ticker": t, "when": ts, "direction": d, "width": width,
                    "confidence": p.get("confidence"), "cand": p.get("candidate_score")})
    random.Random(7).shuffle(out)
    return out[:sample]


def load_confirmed(have: set[str]) -> list[dict]:
    out = []
    for line in CLOSED.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, et = r.get("ticker"), r.get("entry_time") or r.get("confirmed_at")
        if t in have and et:
            out.append({"ticker": t, "when": pd.Timestamp(et),
                        "direction": str(r.get("direction") or "").lower(),
                        "width": None, "confidence": None, "cand": None})
    return out


def enrich(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        cov = covariates(r["ticker"], r["when"], r["direction"])
        if cov is None:
            continue
        rec = {**r, **cov}
        for h in HORIZONS:
            e = excursion(r["ticker"], r["when"], r["direction"], h)
            rec[f"mfe_{h}"] = e["mfe"] if e else np.nan
            rec[f"mae_{h}"] = e["mae"] if e else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def bucket_table(df: pd.DataFrame, col: str, buckets, label: str) -> None:
    print(f"\n{label}")
    print(f"{'bucket':>14s} {'n':>5s} " + " ".join(
        f"{f'med MFE {h}m':>12s} {f'med R {h}m':>10s}" for h in HORIZONS))
    for lo, hi in buckets:
        s = df[(df[col] >= lo) & (df[col] < hi)]
        s = s[s[f"mfe_{HORIZONS[0]}"].notna()]
        if len(s) < 15:
            continue
        cells = []
        for h in HORIZONS:
            mfe = s[f"mfe_{h}"]
            r = (s[f"mfe_{h}"] / s["width"]) if col == "width" else pd.Series(np.nan, index=s.index)
            cells.append(f"{mfe.median():12.3f} {r.median():10.3f}")
        name = f"{lo:g}..{hi:g}" if hi < 1e8 else f">{lo:g}"
        print(f"{name:>14s} {len(s):5d} " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    have = {p.stem for p in BARS.glob("*.parquet")}

    dec = enrich(load_declined(have, args.sample))
    con = enrich(load_confirmed(have))
    dec_w = dec[dec["width"].notna() & (dec["width"] > 0)]
    print(f"declined (wide-invalidation, in bar window): {len(dec)}  with width proxy: {len(dec_w)}")
    print(f"confirmed (in bar window)                  : {len(con)}")

    print("\n" + "=" * 96)
    print("STEP 2 — does the implied stop width predict anything? (width = PROXY, see docstring)")
    print("=" * 96)
    print(f"declined width proxy (ATR): median {dec_w['width'].median():.2f}  "
          f"p25 {dec_w['width'].quantile(.25):.2f}  p75 {dec_w['width'].quantile(.75):.2f}  "
          f"share > 2.0 cap: {(dec_w['width'] > 2).mean():.0%}")
    bucket_table(dec_w, "width", WIDTH_BUCKETS, "DECLINED, by implied stop width")
    for h in HORIZONS:
        s = dec_w[dec_w[f"mfe_{h}"].notna()]
        if len(s) < 30:
            continue
        rho_m = spearmanr(s["width"], s[f"mfe_{h}"]).statistic
        rho_r = spearmanr(s["width"], s[f"mfe_{h}"] / s["width"]).statistic
        print(f"  {h}m: rho(width, MFE) = {rho_m:+.3f}   rho(width, MFE/width) = {rho_r:+.3f}  n={len(s)}")
    print("\n  MFE alone rising with width is mechanical — a wider stop sits on a bigger")
    print("  structure. MFE/width is the number that decides 'size down instead of refuse'.")
    for h in HORIZONS:
        s = dec_w[dec_w[f"mfe_{h}"].notna()]
        c = con[con[f"mfe_{h}"].notna()]
        if len(s) < 30 or len(c) < 20:
            continue
        print(f"  {h}m: declined median MFE/width {(s[f'mfe_{h}'] / s['width']).median():.3f} "
              f"vs confirmed median MFE {c[f'mfe_{h}'].median():.3f} (confirmed width not recorded)")

    print("\n" + "=" * 96)
    print("STEP 3 — entry timing: is the engine confirming after the move?")
    print("=" * 96)
    for name, df in (("CONFIRMED", con), ("DECLINED", dec)):
        sub = df[df[f"mfe_{HORIZONS[0]}"].notna()]
        if len(sub) < 30:
            print(f"{name}: only {len(sub)} usable rows — not reported")
            continue
        bucket_table(sub, "trail", TRAIL_BUCKETS,
                     f"{name}, by run-up already in hand at the decision (ATR, signed to the setup)")
        for h in HORIZONS:
            s = sub[sub[f"mfe_{h}"].notna()]
            if len(s) < 30:
                continue
            rho = spearmanr(s["trail"], s[f"mfe_{h}"]).statistic
            print(f"  {name} {h}m: rho(run-up, forward MFE) = {rho:+.3f}  n={len(s)}")
    print("\n  Negative rho = the more the move has already run when the engine acts, the")
    print("  less is left afterwards, i.e. it is confirming into exhaustion.")


if __name__ == "__main__":
    main()
