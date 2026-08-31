"""Analyse and report Study A.

Structure is deliberate and the order matters:

1. **Base rates** -- what happens at an ordinary strike. Without this, any
   rejection rate at a wall sounds impressive when it may just be what price
   does at every strike.
2. **Confirmatory** -- the one pre-declared comparison, with session-clustered
   intervals. This is the result.
3. **Exploratory** -- regime splits. Labelled as exploratory throughout because
   they are many comparisons on one sample; they generate hypotheses and do not
   confirm them.

Nothing in the exploratory section may be reported as a finding without an
out-of-sample confirmation on later sessions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.gamma_levels.study_a_level_reaction import cluster_bootstrap

DATA = Path(__file__).resolve().parent / "data"
DECISIVE = ("rejection", "penetration")


def _decisive(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["outcome"].isin(DECISIVE)].copy()


def _rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    return float((frame["outcome"] == "rejection").mean())


def _line(name: str, treated: pd.Series, frame: pd.DataFrame) -> dict:
    work = _decisive(frame)
    mask = treated.reindex(work.index).fillna(False).astype(bool)
    gap, low, high, clusters = cluster_bootstrap(frame, treated)
    return {
        "comparison": name,
        "n_treated": int(mask.sum()),
        "n_control": int((~mask).sum()),
        "rate_treated": _rate(work[mask]),
        "rate_control": _rate(work[~mask]),
        "gap_pp": gap * 100 if pd.notna(gap) else float("nan"),
        "ci_low_pp": low * 100 if pd.notna(low) else float("nan"),
        "ci_high_pp": high * 100 if pd.notna(high) else float("nan"),
        "sessions": clusters,
    }


def _fmt(rows: list[dict]) -> str:
    out = [
        "| comparison | n treated | n control | rej% treated | rej% control | gap (pp) | 95% CI | sessions |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        ci = (
            f"[{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]"
            if pd.notna(r["ci_low_pp"])
            else "n/a"
        )
        sig = " **" if pd.notna(r["ci_low_pp"]) and (r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0) else ""
        out.append(
            f"| {r['comparison']}{sig} | {r['n_treated']} | {r['n_control']} | "
            f"{r['rate_treated']*100:.1f} | {r['rate_control']*100:.1f} | "
            f"{r['gap_pp']:+.1f} | {ci} | {r['sessions']} |"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DATA / "study_a_events.parquet")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "STUDY_A_RESULTS.md")
    args = parser.parse_args()

    ev = pd.read_parquet(args.events)
    ev["hour"] = pd.to_datetime(ev["captured_at"]).dt.hour
    dec = _decisive(ev)

    lines: list[str] = []
    add = lines.append

    add("# Study A results — does price react at gamma levels?\n")
    add(f"Generated from `{args.events.name}`. "
        f"**{len(ev):,} arrival events**, of which **{len(dec):,} resolved** "
        f"(rejection or penetration) across **{ev['session'].nunique()} sessions** "
        f"and **{ev['symbol'].nunique()} symbols**.\n")
    add("An event is price arriving within 10bps of an option strike. Every event is "
        "the same kind of event; the strikes differ only in how much gamma sits on them. "
        "Outcome thresholds are symmetric (20bps each way) so neither outcome is "
        "favoured by the measurement.\n")

    # --- 1. base rates -----------------------------------------------------
    add("## 1. Base rates — what happens at an ordinary strike\n")
    add("Read this first. It is the number every result below has to beat.\n")
    base = pd.DataFrame(
        {
            "events": ev.groupby("symbol").size(),
            "resolved": dec.groupby("symbol").size(),
            "rejection_rate_%": (dec.groupby("symbol")["outcome"]
                                 .apply(lambda s: (s == "rejection").mean() * 100).round(1)),
            "sessions": ev.groupby("symbol")["session"].nunique(),
        }
    )
    add("| symbol | events | resolved | rejection % | sessions |")
    add("|---|---:|---:|---:|---:|")
    for sym, r in base.iterrows():
        add(f"| {sym} | {int(r['events'])} | {int(r['resolved'])} | "
            f"{r['rejection_rate_%']:.1f} | {int(r['sessions'])} |")
    add("")
    add(f"Pooled resolution rate: **{len(dec)/len(ev):.0%}** of arrivals resolve within 30 minutes; "
        f"the rest stay inside the band and are reported as `neither`, never dropped.\n")
    add(f"Pooled rejection rate at **any** strike: **{_rate(dec)*100:.1f}%**.\n")

    # --- 2. confirmatory ---------------------------------------------------
    add("## 2. Confirmatory — the pre-declared comparison\n")
    add("Session-clustered bootstrap, 2,000 draws. `**` marks an interval excluding zero. "
        "The registered decision threshold is **>= 8pp with an interval excluding zero**.\n")
    rows = [
        _line("any gamma level vs plain strike", ev["any_level"], ev),
        _line("call wall vs everything else", ev["is_call_wall"], ev),
        _line("put wall vs everything else", ev["is_put_wall"], ev),
        _line("magnet vs everything else", ev["is_magnet"], ev),
    ]
    add(_fmt(rows))
    add("")

    # --- 2b. de-confounded ------------------------------------------------
    add("### 2b. Walls only — removing the magnet's circularity\n")
    add("Gamma peaks at the money, so the magnet is the strike nearest spot "
        "31-70% of the time (SPY 70%). \"Price arrived at the magnet\" is therefore "
        "partly the statement \"price is where price is\", and any arm containing "
        "magnets inherits that circularity.\n")
    add("The walls do not have this problem: the call wall sits within 10bps of spot "
        "only 14% of the time and the put wall 12%, because they are defined on the "
        "far side of spot. This is the clean test.\n")
    no_magnet = ev[~ev["is_magnet"]].copy()
    walls = no_magnet["is_call_wall"] | no_magnet["is_put_wall"]
    add(_fmt([_line("wall (call or put) vs plain strike, magnets excluded", walls, no_magnet)]))
    add("")

    # --- 3. exploratory ----------------------------------------------------
    add("## 3. Exploratory — regime splits\n")
    add("> **These are exploratory.** Many comparisons on one sample. They generate "
        "hypotheses; they do not confirm them. Nothing here should be wired into a "
        "strategy without confirmation on later, unseen sessions.\n")

    ex: list[dict] = []

    # gamma regime
    pos = ev["dealer_imbalance"] > 0.05
    neg = ev["dealer_imbalance"] < -0.05
    ex.append(_line("gamma level, positive-gamma regime", ev["any_level"] & pos, ev[pos | ~pos]))
    ex.append(_line("gamma level, negative-gamma regime", ev["any_level"] & neg, ev[neg | ~neg]))

    # time of day (UTC: 13:30 open, 20:00 close)
    early = ev["hour"] < 15
    late = ev["hour"] >= 19
    ex.append(_line("gamma level, first 90 min", ev["any_level"] & early, ev))
    ex.append(_line("gamma level, last hour", ev["any_level"] & late, ev))

    # persistence: has the level held its strike?
    persist = ev[["persist_call_wall", "persist_put_wall", "persist_magnet"]].max(axis=1)
    sticky = ev["any_level"] & (persist >= persist.median())
    ex.append(_line("gamma level that has held >= median duration", sticky, ev))

    # concentration
    share = ev[["call_wall_share", "put_wall_share", "magnet_share"]].max(axis=1)
    heavy = ev["any_level"] & (share >= share.quantile(0.75))
    ex.append(_line("gamma level in top-quartile concentration", heavy, ev))

    add(_fmt(ex))
    add("")

    add("### Per-symbol, gamma level vs plain strike\n")
    per = []
    for sym, grp in ev.groupby("symbol"):
        per.append(_line(f"{sym}", grp["any_level"], grp))
    add(_fmt(per))
    add("")

    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
