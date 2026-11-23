"""
Catch up shared bars (1H -> 4H, 1D) for the meta-ranker universe via Alpaca REST.

Incremental: fetch_one only pulls bars after each ticker's cached last timestamp,
so a daily/4H run is cheap. No websocket needed (REST handles the whole universe).

  PYTHONPATH=. python scripts/catchup_shared_bars.py --workers 6
  PYTHONPATH=. python scripts/catchup_shared_bars.py --tickers NVDA MU   # subset

Writes a progress line to /tmp/catchup_bars_status.txt every 25 tickers.
"""
from __future__ import annotations

from collections.abc import Sequence

import argparse
import datetime as dt
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.live_readiness import write_intraday_bars_stamp  # noqa: E402
from strategies.momentum_expansion.data import bars  # noqa: E402

UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
STATUS = Path("/tmp/catchup_bars_status.txt")


def _executor(workers: int, *, threads: bool):
    if threads:
        return ThreadPoolExecutor(max_workers=workers)
    # spawn, not fork: the parent holds Alpaca client threads by the time workers start.
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _one(ticker: str, end: str, do_1d: bool, fetch_1h: bool = True) -> tuple[str, int, int, str]:
    """1D (optional) + 4H rebuild for one ticker; its 1H too unless the batch already did it."""
    n1 = n4 = 0
    err = ""
    try:
        if fetch_1h:
            d1 = bars.fetch_one(ticker=ticker, kind="1h", end=end)
            n1 = 0 if d1 is None else len(d1)
        if do_1d:
            bars.fetch_one(ticker=ticker, kind="1d", end=end)
        d4 = bars.build_4h_for(ticker, force=False)
        n4 = 0 if d4 is None else len(d4)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    return ticker, n1, n4, err


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-1d", action="store_true")
    ap.add_argument(
        "--max-error-rate",
        type=float,
        default=0.10,
        help="Exit non-zero when more than this fraction of tickers error. A few "
             "delistings are normal; a systemic API failure is not, and returning "
             "zero for it let the 4H loop trade on bars that were never fetched.",
    )
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
    ap.add_argument("--threads", action="store_true",
                    help="Run per-ticker work in threads instead of a process pool (tests, debugging).")
    ap.add_argument("--per-ticker", action="store_true",
                    help="Fetch 1H one request per ticker (the pre-2026-11-22 path) instead of "
                         "multi-symbol batches. Batches take ~3 s per 1,000 symbols; one by one "
                         "the eligible set took ~800 s at the Alpaca rate limit.")
    args = ap.parse_args(argv)

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
    one_by_one = set(tickers)
    done = errs = 0
    # Per-ticker work (merge, 1D, 4H rebuild) is GIL-bound pandas: a spawn process pool
    # runs it ~4x faster than threads on 6 cores (1,000 tickers' 4H: 10 s vs 45 s).
    with _executor(args.workers, threads=args.threads) as ex:
        if not args.per_ticker:
            batch_status = bars.fetch_intraday_batched(
                tickers, kind="1h", end=args.end, workers=args.workers,
                executor=None if args.threads else ex)
            # Uncached names need their full history and failed batches a retry: fetch_one.
            one_by_one = {t for t, s in batch_status.items() if s in ("uncached", "error")}
            counts = pd.Series(list(batch_status.values()), dtype=object).value_counts().to_dict()
            print(f"batched 1H in {time.time() - t0:.0f}s: {counts}", flush=True)
        futs = {ex.submit(_one, t, args.end, not args.no_1d, t in one_by_one): t for t in tickers}
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
    elapsed = time.time() - t0
    print(f"\nDONE: {done} tickers, {errs} errors in {elapsed:.0f}s")

    # A zero exit here used to mean "the process ended", not "the bars are
    # there". The 4H loop now depends on this status, so a systemic failure
    # must be reported rather than swallowed.
    attempted = len(tickers)
    rate = (errs / attempted) if attempted else 0.0
    STATUS.write_text(
        f"{done}/{attempted} done, {errs} errors, {elapsed:.0f}s elapsed\n"
    )
    if attempted == 0:
        print("  ! no tickers were attempted", flush=True)
        return 1
    if rate > args.max_error_rate:
        print(
            f"  ! error rate {rate:.1%} exceeds the {args.max_error_rate:.1%} limit",
            flush=True,
        )
        return 1
    if not args.tickers:   # a hand-picked subset proves nothing about the universe
        end_ts = pd.Timestamp(args.end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        write_intraday_bars_stamp(
            end=end_ts.to_pydatetime(), tickers=attempted, errors=errs,
            scope="eligible" if args.eligible_only else "universe",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
