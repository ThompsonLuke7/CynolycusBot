#!/usr/bin/env python
"""D3: does an intraday-structure confirmation add to the call-wall edge?

Study A established that price rejects at a call wall +9.3pp more often than at
an ordinary strike, and said plainly that the level alone is not the whole story:
"a trader running at 90% is adding selection on top." The intraday structure
engine is a selection layer. This asks whether its selection is worth anything
on top of the level.

Registered at
``docs/superpowers/plans/2026-08-26-intraday-structure-step-d-preregistration.md``
section 4.

**This script enforces its own registration.** Below the registered minimum it
reports arm sizes and stops, without reading a single outcome. That is not
politeness -- looking at a 2-event arm and then deciding what to do next is
exactly how a pre-registration gets spent.

    python -m research.gamma_levels.d3_confirmation_overlap
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from research.gamma_levels.study_a_level_reaction import cluster_bootstrap

REPO = Path(__file__).resolve().parents[2]
EVENTS = Path(__file__).resolve().parent / "data" / "study_a_events.parquet"
TRANSITIONS = REPO / "Data/inference/intraday_structure/transitions.jsonl"

ETFS = ("SPY", "QQQ", "IWM", "GLD", "SLV")

# Registered window. Asymmetric on purpose: a confirmation that lands after the
# outcome window has opened was not available to a trader at the touch.
LOOKBACK_MIN = 15
LOOKAHEAD_MIN = 5

#: A call wall sits above spot, so "rejection" is price turning back down.
REJECTION_DIRECTION = "short"

# Registered minimum, section 4. All three must hold.
MIN_RESOLVED = 120
MIN_SMALLER_ARM = 40
MIN_SESSIONS = 30

logger = logging.getLogger("d3")


def load_confirmations(path: Path = TRANSITIONS) -> pd.DataFrame:
    """Every CONFIRMED transition on the five ETFs, with its direction."""
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "ts", "direction"])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("to_state") != "CONFIRMED":
            continue
        ticker = record.get("ticker")
        if ticker not in ETFS:
            continue
        parts = str(record.get("setup_id", "")).split(":")
        rows.append({
            "symbol": ticker,
            "ts": pd.Timestamp(record["timestamp"]),
            "direction": parts[1] if len(parts) > 1 else "unknown",
        })
    frame = pd.DataFrame(rows, columns=["symbol", "ts", "direction"])
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def build_overlap(events: pd.DataFrame, confirmations: pd.DataFrame) -> pd.DataFrame:
    """Call-wall arrivals inside the window where BOTH datasets exist.

    The engine only started on 2026-07-21, well after Study A's window opens, so
    the overlap is a strict subset and the effective n must be reported on it --
    not on Study A's full session count.
    """
    events = events.copy()
    events["captured_at"] = pd.to_datetime(events["captured_at"], utc=True)
    if confirmations.empty:
        return events.iloc[0:0].assign(confirmed=pd.Series(dtype=bool))
    low, high = confirmations["ts"].min(), events["captured_at"].max()
    window = events[(events["captured_at"] >= low) & (events["captured_at"] <= high)]
    call_wall = window[window["is_call_wall"]].copy()

    lookback = pd.Timedelta(minutes=LOOKBACK_MIN)
    lookahead = pd.Timedelta(minutes=LOOKAHEAD_MIN)
    flags = []
    for row in call_wall.itertuples():
        near = confirmations[
            (confirmations["symbol"] == row.symbol)
            # Registered as "the direction consistent with rejection". At a call
            # wall -- resistance above spot -- rejection means price turns back
            # DOWN, so the rejection-consistent setup is a short. Counting longs
            # as treated would score the engine as right when it said the
            # opposite of what happened.
            & (confirmations["direction"] == REJECTION_DIRECTION)
            & (confirmations["ts"] >= row.captured_at - lookback)
            & (confirmations["ts"] <= row.captured_at + lookahead)
        ]
        flags.append(len(near) > 0)
    call_wall["confirmed"] = flags
    return call_wall


def feasibility(call_wall: pd.DataFrame) -> dict:
    """Arm sizes and whether the registered minimum is met. Reads no outcomes."""
    resolved = call_wall[call_wall["outcome"].isin(["rejection", "penetration"])]
    treated = int(resolved["confirmed"].sum())
    control = int(len(resolved) - treated)
    sessions = int(resolved["session"].nunique()) if not resolved.empty else 0
    return {
        "call_wall_arrivals": int(len(call_wall)),
        "resolved": int(len(resolved)),
        "treated": treated,
        "control": control,
        "smaller_arm": min(treated, control),
        "sessions": sessions,
        "minimum_met": bool(
            len(resolved) >= MIN_RESOLVED
            and min(treated, control) >= MIN_SMALLER_ARM
            and sessions >= MIN_SESSIONS
        ),
    }


def run_study(call_wall: pd.DataFrame) -> dict:
    """The registered contrast. Only reachable once the minimum is met."""
    gap, low, high, clusters = cluster_bootstrap(call_wall, call_wall["confirmed"])
    resolved = call_wall[call_wall["outcome"].isin(["rejection", "penetration"])]
    treated = resolved[resolved["confirmed"]]
    control = resolved[~resolved["confirmed"]]
    rate = lambda f: float((f["outcome"] == "rejection").mean()) if len(f) else float("nan")
    if gap >= 0.08 and low > 0:
        decision = "GRADUATE — confirmation adds to the level"
    elif gap > 0:
        decision = "RECORD ONLY — real but below the +8pp threshold; do not wire"
    else:
        decision = "NEGATIVE — confirmation subtracts; the engine's filters may be removing good touches"
    return {
        "rejection_rate_confirmed": rate(treated),
        "rejection_rate_not_confirmed": rate(control),
        "gap_pp": 100 * gap,
        "ci_low_pp": 100 * low,
        "ci_high_pp": 100 * high,
        "session_clusters": clusters,
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--transitions", type=Path, default=TRANSITIONS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    events = pd.read_parquet(args.events)
    confirmations = load_confirmations(args.transitions)
    call_wall = build_overlap(events, confirmations)
    status = feasibility(call_wall)
    status["etf_confirmations_available"] = int(len(confirmations))

    if status["minimum_met"]:
        status.update(run_study(call_wall))

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    print("=" * 74)
    print("D3 — intraday-structure confirmation at call-wall touches")
    print("=" * 74)
    print(f"  ETF confirmations in the transition log : {status['etf_confirmations_available']}")
    print(f"  call-wall arrivals in overlap           : {status['call_wall_arrivals']}")
    print(f"  ...resolved within 30 min               : {status['resolved']}")
    print(f"  sessions in overlap                     : {status['sessions']}")
    print()
    print(f"  ARMS   treated (confirmed) = {status['treated']}   control = {status['control']}")
    print()
    print(f"  registered minimum: >={MIN_RESOLVED} resolved, >={MIN_SMALLER_ARM} in the smaller arm, >={MIN_SESSIONS} sessions")
    if not status["minimum_met"]:
        print("  MINIMUM NOT MET — no outcome was read. Nothing here is a result.")
        print()
        print("  The binding shortfall is the treated arm: the engine rarely has an")
        print("  opinion at the moment a call wall is touched, so the two signals")
        print("  barely co-occur. That is a fact about the engine, not about the level.")
        return 0
    print("  MINIMUM MET — registered contrast follows.")
    print()
    print(f"  rejection | confirmed     : {100*status['rejection_rate_confirmed']:.1f}%")
    print(f"  rejection | not confirmed : {100*status['rejection_rate_not_confirmed']:.1f}%")
    print(f"  gap                       : {status['gap_pp']:+.1f}pp  "
          f"[{status['ci_low_pp']:+.1f}, {status['ci_high_pp']:+.1f}]  "
          f"({status['session_clusters']} clusters)")
    print(f"  DECISION: {status['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
