"""Thesis test, step 5: the same replay with MEASURED transaction costs.

Every cost number here is measured from live fills in this repo. Nothing is
assumed, and the two things that could not be measured are named as such.

WHAT COULD NOT BE MEASURED
--------------------------
Historical option QUOTES for expired contracts are not available on this
subscription: `/v1beta1/options/quotes` returns 404 and `/quotes/latest` returns
empty for an expired symbol. Only trades and daily bars are served. So a true
quote-by-quote replay is impossible, and this applies a measured cost HAIRCUT to
a trade-price replay instead. That is a weaker construction and is labelled as
one.

WHAT WAS MEASURED
-----------------
ENTRY, n=94 option entries with both a realized broker fill and the mid quoted at
order time (`order_audits.mid_price`), from the same four 4H modules this study
replays:
    median +4.0% above mid   (per module: meta +1.5%, HTF +2.3%,
                              momentum +3.3%, dealer +7.9%)

EXIT, n=566 filled sells from the 30m swing module's session audits, which are
the only records in the repo carrying a quote at fill time (`close_quote`):
    median +12.2% worse than mid, 97% filled worse than mid
    (quoted spread at those moments: median 25.1% of mid)

CAVEAT ON THE EXIT NUMBER, stated because it matters: it comes from a DIFFERENT
module whose contracts were short-dated and frequently far OTM by exit time,
which is where option spreads are widest. It is therefore likely to be
PESSIMISTIC for the 35-45 DTE monthlies replayed here. The sensitivity band below
exists so the conclusion does not depend on that one number.

A further conservatism: the bar close is itself a TRADE print, already struck
somewhere inside the spread, so charging a full half-spread on top of it
double-counts part of the cost. Every option figure below is therefore a lower
bound on performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/thesis_test"))
from validate_and_simulate import PATHS, shares_return, validate  # noqa: E402

MEASURED_ENTRY = 0.040      # +4.0%  realized fill vs mid, n=94, these modules
MEASURED_EXIT = 0.122       # +12.2% realized fill vs mid, n=566, 30m module
EQUITY_COST = 0.0005        # 5 bps round trip; shares are commission-free here
                            # and these are liquid names. Named, not hidden.


def sim_cost(row, hold_days, prem_stop, entry_c, exit_c):
    """Replay with costs. Entry pays up; exit receives less; the stop is checked
    on the intraday LOW of the option bar, which is the best available proxy for
    an intraday stop touch on daily data."""
    bars = row["bars"]
    raw_entry = float(bars[0]["c"])
    if raw_entry <= 0:
        return None
    entry = raw_entry * (1.0 + entry_c)
    path = bars[1:hold_days + 1]
    if not path:
        return None
    for b in path:
        if prem_stop is not None and float(b["l"]) <= raw_entry * (1.0 - prem_stop):
            # Stopped: exit near the stop level, still paying the exit cost.
            exit_px = raw_entry * (1.0 - prem_stop) * (1.0 - exit_c)
            return exit_px / entry - 1.0
    exit_px = float(path[-1]["c"]) * (1.0 - exit_c)
    return exit_px / entry - 1.0


def main() -> None:
    rows = [json.loads(l) for l in PATHS.open() if l.strip()]
    good = [r for r in rows if r.get("n_bars") and validate(r)[0]]
    print(f"usable contracts: {len(good)}\n")

    scenarios = [
        ("GROSS (no cost)", 0.0, 0.0),
        ("MEASURED (+4.0% in / +12.2% out)", MEASURED_ENTRY, MEASURED_EXIT),
        ("PESSIMISTIC (+10% in / +20% out)", 0.10, 0.20),
    ]
    holds = [5, 8, 10, 13, 15, 20]
    stops = [("-39% (live)", 0.39), ("-60%", 0.60), ("none", None)]

    for name, ec, xc in scenarios:
        print("=" * 96)
        print(f"{name}   — total $ per $1,000 deployed")
        print("=" * 96)
        print(f"{'hold':>6s} " + " ".join(f"{s:>14s}" for s, _ in stops) + f" {'SHARES':>12s}")
        for h in holds:
            cells = []
            for _, s in stops:
                v = [sim_cost(g, h, s, ec, xc) for g in good]
                v = np.array([x for x in v if x is not None and np.isfinite(x)])
                cells.append(f"{1000 * v.sum():14,.0f}" if len(v) else f"{'-':>14s}")
            sh = [shares_return(g, h) for g in good]
            sh = np.array([x for x in sh if x is not None and np.isfinite(x)])
            sh = sh - EQUITY_COST
            print(f"{h:5d}d " + " ".join(cells) + f" {1000 * sh.sum():12,.0f}")
        print()

    print("Reminders that bound every number above:")
    print("  * n = 39 contracts. Cell-level differences are not reliable.")
    print("  * Entries are the ones the CURRENT ranker chose.")
    print("  * Historical option quotes do not exist on this plan, so these are")
    print("    trade-price replays with a measured cost haircut, not quote replays.")
    print("  * The exit cost is borrowed from the 30m module and is likely")
    print("    pessimistic for 35-45 DTE monthlies.")
    print("  * Daily bars approximate intraday stop touches by the daily low.")


if __name__ == "__main__":
    main()
