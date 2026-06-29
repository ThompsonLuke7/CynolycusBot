"""
Catch up shared bars (1H -> 4H, 1D) for the meta-ranker universe via Alpaca REST.

Incremental: fetch_one only pulls bars after each ticker's cached last timestamp,
so a daily/4H run is cheap. No websocket needed (REST handles the whole universe).

  PYTHONPATH=. python scripts/catchup_shared_bars.py --workers 6
  PYTHONPATH=. python scripts/catchup_shared_bars.py --tickers NVDA MU   # subset

Writes a progress line to /tmp/catchup_bars_status.txt every 25 tickers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from strategies.momentum_expansion.data import bars  # noqa: E402

UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
STATUS = Path("/tmp/catchup_bars_status.txt")


def _one(ticker: str, end: str, do_1d: bool) -> tuple[str, int, int, str]:
    n1 = n4 = 0
    err = ""
    try:
        d1 = bars.fetch_one(ticker=ticker, kind="1h", end=end)
        n1 = 0 if d1 is None else len(d1)
        if do_1d:
            bars.fetch_one(ticker=ticker, kind="1d", end=end)
        d4 = bars.build_4h_for(ticker, force=False)
        n4 = 0 if d4 is None else len(d4)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    return ticker, n1, n4, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-1d", action="store_true")
    # Fetch through NOW minus a delay buffer. Two reasons:
    #   * A date-only end ("2026-06-23") parses to 00:00 UTC — the START of today —
    #     so today's RTH bars (13:30-20:00 UTC) are excluded and a same-day catch-up
    #     silently fetches nothing new. Use a full timestamp instead.
    #   * The SIP subscription rejects the most recent ~15 min ("subscription does
    #     not permit querying recent SIP data"), so back off 20 min to stay legal
    #     while still pulling the latest completed bar.
    _SIP_DELAY_MIN = 20
    _default_end = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=_SIP_DELAY_MIN))
    ap.add_argument("--end", default=_default_end.isoformat())
    ap.add_argument("--eligible-only", action="store_true",
                    help="Only refresh is_eligible names (the tradeable set). Much faster "
                         "for a same-day pre-close catch-up; the full universe is backfilled "
                         "by the nightly readiness job.")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        u = pd.read_csv(args.universe)
        if args.eligible_only and "is_eligible" in u.columns:
            mask = u["is_eligible"].astype(str).str.lower().isin(["true", "1", "1.0"])
            u = u[mask]
        tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique())
    t0 = time.time()
    print(f"catch-up {len(tickers)} tickers through {args.end} (workers={args.workers})")
    done = errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, t, args.end, not args.no_1d): t for t in tickers}
        for fut in as_completed(futs):
            ticker, n1, n4, err = fut.result()
            done += 1
            if err:
                errs += 1
                if errs <= 20:
                    print(f"  ! {ticker}: {err}")
            if done % 25 == 0 or done == len(tickers):
                msg = f"{done}/{len(tickers)} done, {errs} errors, {time.time()-t0:.0f}s elapsed"
                STATUS.write_text(msg + "\n")
                print(msg, flush=True)
    print(f"\nDONE: {done} tickers, {errs} errors in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
