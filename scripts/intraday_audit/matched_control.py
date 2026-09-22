"""Does intraday_structure really select worse-than-random moments? (step 1)

`20_recall_test.md` found CONFIRMED setups reach 0.501 ATR of favourable
excursion against 0.758 for a random minute on the same ticker, and called that
negative selection. Its own stated limit was that the control was NOT matched:
the engine fires at specific moments -- breakouts, pullback completions, opening
range -- while the control sampled any RTH minute. If the engine systematically
acts AFTER a move, lower forward excursion is partly mechanical, not a selection
failure, and the original comparison cannot tell those apart.

This rebuilds the control matched on the two covariates that confound it:

  * TIME OF DAY   -- within +/-`--tod-tol` minutes. Intraday excursion is
    strongly time-dependent (open >> midday), and the engine is not uniform
    across the session.
  * TRAILING MOVE -- the move over the prior `--trail-min` minutes, expressed in
    ATR and SIGNED in the setup's own direction, within +/-`--trail-tol`. This is
    the exhaustion question: a minute that has already run 1.5 ATR in your
    direction has different forward odds than one that has not.

Control minutes are drawn from the same ticker on a DIFFERENT day, so the
control's forward window cannot overlap the event's own.

A balance table is printed first. A matched control that is not actually
balanced proves nothing, so the match is evidence before the result is.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = DATA / "bars_1m"
EVENTS = REPO / "Data/inference/intraday_structure/decision_events.jsonl"
CLOSED = REPO / "Data/inference/intraday_structure/closed_setups.jsonl"
HORIZONS = (15, 30, 60)
ET = "America/New_York"
_c: dict[str, pd.DataFrame | None] = {}


def bars(t: str):
    """1m bars with the matching covariates precomputed once per ticker."""
    if t not in _c:
        p = BARS / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p)
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            d = d.sort_values("timestamp").set_index("timestamp")
            if len(d) >= 500:
                # Same ATR convention as the original test: 14-bar mean true
                # range on 1m bars, scaled to a ~30-minute horizon.
                atr = (d["high"] - d["low"]).rolling(14).mean() * np.sqrt(30.0)
                d["atr30"] = atr.where(atr > 0)
                loc = d.index.tz_convert(ET)
                d["tod"] = loc.hour * 60 + loc.minute
                d["day"] = loc.normalize().tz_localize(None)
                d["_c30"] = d["close"].shift(30)
            else:
                d = None
        _c[t] = d
    return _c[t]


def trail_atr(d: pd.DataFrame, direction: str) -> pd.Series:
    """Prior-30-minute move in ATR units, signed in the setup's direction."""
    sign = 1.0 if str(direction).lower() == "long" else -1.0
    return sign * (d["close"] - d["_c30"]) / d["atr30"]


#: The 1m cache spans a fixed window (2026-07-08..08-28) while the ledgers run
#: months past it. Without this guard `prior.index[-1]` silently returns the LAST
#: BAR IN THE FILE for any later event -- which is how the first run of this
#: script produced a median decision time of 16:28 (the cache's final bar) and
#: failed to match 89% of events.
STALE_TOLERANCE = pd.Timedelta(minutes=2)


def covariates(t: str, when, direction: str):
    d = bars(t)
    if d is None:
        return None
    prior = d.loc[d.index <= when]
    if prior.empty:
        return None
    i = prior.index[-1]
    if when - i > STALE_TOLERANCE:
        return None            # event lies outside the cached bar window
    row = d.loc[i]
    tr = trail_atr(d.loc[[i]], direction).iloc[0]
    if not np.isfinite(row.get("atr30", np.nan)) or not np.isfinite(tr):
        return None
    return {"tod": float(row["tod"]), "trail": float(tr), "day": row["day"]}


def excursion(t: str, when, direction: str, minutes: int):
    d = bars(t)
    if d is None:
        return None
    prior = d.loc[d.index <= when]
    if prior.empty:
        return None
    a = float(prior["atr30"].iloc[-1]) if np.isfinite(prior["atr30"].iloc[-1]) else None
    if not a:
        return None
    w = d.loc[(d.index > when) & (d.index <= when + timedelta(minutes=minutes * 3))]
    if len(w) < 5:
        return None
    w = w.iloc[:minutes]
    ref = float(prior["close"].iloc[-1])
    sign = 1 if str(direction).lower() == "long" else -1
    fav = (float(w["high"].max()) - ref) if sign > 0 else (ref - float(w["low"].min()))
    adv = (ref - float(w["low"].min())) if sign > 0 else (float(w["high"].max()) - ref)
    return {"mfe": fav / a, "mae": adv / a}


def matched_control(ev: dict, rng: random.Random, *, tod_tol: int, trail_tol: float):
    """A minute on the same ticker, another day, similar time and similar run-up."""
    d = bars(ev["ticker"])
    if d is None:
        return None
    tr = trail_atr(d, ev["direction"])
    ok = (
        (d["day"] != ev["day"])
        & (d["tod"].sub(ev["tod"]).abs() <= tod_tol)
        & (tr.sub(ev["trail"]).abs() <= trail_tol)
        & d["atr30"].notna()
        & tr.notna()
        # leave room for the longest forward window
        & (d["tod"] <= 16 * 60 - max(HORIZONS) - 5)
    )
    idx = d.index[ok.fillna(False).to_numpy()]
    if len(idx) == 0:
        return None
    when = idx[rng.randrange(len(idx))]
    cov = covariates(ev["ticker"], when, ev["direction"])
    if cov is None:
        return None
    return {"ticker": ev["ticker"], "when": when, "direction": ev["direction"], **cov}


def load_events(have: set[str], sample: int):
    seen, declined = set(), []
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
        t = r.get("ticker")
        if t not in have:
            continue
        ts = pd.Timestamp(r["event_time"])
        key = (p.get("setup_id"), t, str(ts.date()))
        if key in seen:
            continue
        seen.add(key)
        declined.append({"ticker": t, "when": ts, "direction": p.get("direction"),
                         "reason": p.get("no_trade_reason")})
    random.Random(7).shuffle(declined)
    declined = declined[:sample]

    confirmed = []
    for line in CLOSED.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, et = r.get("ticker"), r.get("entry_time") or r.get("confirmed_at")
        if t in have and et:
            confirmed.append({"ticker": t, "when": pd.Timestamp(et),
                              "direction": r.get("direction")})
    return declined, confirmed


def boot_median_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n=5000):
    """Bootstrap CI for median(a) - median(b); the original reported no interval."""
    d = np.empty(n)
    for i in range(n):
        d[i] = (np.median(rng.choice(a, len(a), replace=True))
                - np.median(rng.choice(b, len(b), replace=True)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.median(a) - np.median(b)), float(lo), float(hi), float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--tod-tol", type=int, default=15, help="time-of-day tolerance, minutes")
    ap.add_argument("--trail-tol", type=float, default=0.25, help="trailing-move tolerance, ATR")
    args = ap.parse_args()

    have = {p.stem for p in BARS.glob("*.parquet")}
    declined, confirmed = load_events(have, args.sample)

    groups: dict[str, list] = {"DECLINED": [], "CONFIRMED": []}
    for name, rows in (("DECLINED", declined), ("CONFIRMED", confirmed)):
        for r in rows:
            cov = covariates(r["ticker"], r["when"], r["direction"])
            if cov:
                groups[name].append({**r, **cov})
    print(f"declined (one per setup per day, with covariates): {len(groups['DECLINED']):,}")
    print(f"confirmed setups with covariates                 : {len(groups['CONFIRMED']):,}")

    rng = random.Random(11)
    # UNMATCHED control, as the original built it, kept so the two are comparable.
    unmatched = []
    for ev in groups["DECLINED"]:
        d = bars(ev["ticker"])
        if d is None:
            continue
        when = d.index[rng.randrange(200, len(d) - 200)]
        cov = covariates(ev["ticker"], when, ev["direction"])
        if cov:
            unmatched.append({"ticker": ev["ticker"], "when": when,
                              "direction": ev["direction"], **cov})

    # PAIRED: keep the event alongside its own control, so the comparison is
    # event-vs-its-match. Comparing the full CONFIRMED set against a control
    # built only from the matchable subset compares different populations --
    # the first run of this script did exactly that.
    pairs = {"CONFIRMED": [], "DECLINED": []}
    misses = 0
    for name in ("CONFIRMED", "DECLINED"):
        for ev in groups[name]:
            m = matched_control(ev, rng, tod_tol=args.tod_tol, trail_tol=args.trail_tol)
            if m is None:
                misses += 1
            else:
                pairs[name].append((ev, m))
    print(f"matched pairs: confirmed {len(pairs['CONFIRMED']):,}, "
          f"declined {len(pairs['DECLINED']):,} ({misses} events had no match)\n")

    groups["CONTROL_random"] = unmatched
    groups["CONFIRMED_matched_subset"] = [e for e, _ in pairs["CONFIRMED"]]
    groups["CONTROL_matched_confirmed"] = [m for _, m in pairs["CONFIRMED"]]
    groups["DECLINED_matched_subset"] = [e for e, _ in pairs["DECLINED"]]
    groups["CONTROL_matched_declined"] = [m for _, m in pairs["DECLINED"]]

    print("=" * 92)
    print("BALANCE — the match has to be shown before the result can be read")
    print("=" * 92)
    print(f"{'group':30s} {'n':>6s} {'median tod':>11s} {'median trail(ATR)':>18s} "
          f"{'p25..p75 trail':>20s}")
    for name, rows in groups.items():
        if not rows:
            continue
        tod = np.array([r["tod"] for r in rows])
        tr = np.array([r["trail"] for r in rows])
        hh, mm = divmod(int(np.median(tod)), 60)
        print(f"{name:30s} {len(rows):6d} {f'{hh:02d}:{mm:02d}':>11s} "
              f"{np.median(tr):18.3f} {f'{np.percentile(tr,25):.2f}..{np.percentile(tr,75):.2f}':>20s}")

    print("\n" + "=" * 92)
    print("FORWARD EXCURSION, ATR units, from the decision minute")
    print("=" * 92)
    print(f"{'group':30s} {'horizon':>8s} {'n':>6s} {'med MFE':>9s} {'med MAE':>9s} "
          f"{'MFE-MAE':>9s} {'MFE>1':>7s}")
    res: dict = {}
    for name, rows in groups.items():
        for h in HORIZONS:
            vals = [excursion(r["ticker"], r["when"], r["direction"], h) for r in rows]
            vals = [v for v in vals if v]
            if len(vals) < 20:
                continue
            mfe = np.array([v["mfe"] for v in vals])
            mae = np.array([v["mae"] for v in vals])
            res[(name, h)] = mfe
            print(f"{name:30s} {h:7d}m {len(vals):6d} {np.median(mfe):9.3f} "
                  f"{np.median(mae):9.3f} {np.median(mfe - mae):9.3f} {np.mean(mfe > 1):6.0%}")
        print()

    print("=" * 92)
    print("UNPAIRED: CONFIRMED (all) vs the random control — the original comparison")
    print("=" * 92)
    g = np.random.default_rng(5)
    for h in HORIZONS:
        if ("CONFIRMED", h) in res and ("CONTROL_random", h) in res:
            d, lo, hi, p = boot_median_diff(res[("CONFIRMED", h)], res[("CONTROL_random", h)], g)
            v = "confirmed WORSE" if hi < 0 else ("confirmed better" if lo > 0 else "no difference")
            print(f"  {h:2d}m  {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}  {v}")

    print("\n" + "=" * 92)
    print("PAIRED: each event against ITS OWN matched control (same ticker, +/-"
          f"{args.tod_tol}m, +/-{args.trail_tol} ATR run-up)")
    print("=" * 92)
    for name in ("CONFIRMED", "DECLINED"):
        for h in HORIZONS:
            diffs = []
            for ev, ctl in pairs[name]:
                a = excursion(ev["ticker"], ev["when"], ev["direction"], h)
                b = excursion(ctl["ticker"], ctl["when"], ctl["direction"], h)
                if a and b:
                    diffs.append(a["mfe"] - b["mfe"])
            if len(diffs) < 20:
                print(f"  {name:10s} {h:2d}m  only {len(diffs)} usable pairs — not reported")
                continue
            v = np.array(diffs)
            bs = np.array([np.median(g.choice(v, len(v), replace=True)) for _ in range(5000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            verdict = ("event WORSE than its match" if hi < 0
                       else "event better than its match" if lo > 0 else "no difference")
            print(f"  {name:10s} {h:2d}m  n={len(v):4d}  median paired MFE diff "
                  f"{np.median(v):+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3f}  {verdict}")
        print()


if __name__ == "__main__":
    main()
