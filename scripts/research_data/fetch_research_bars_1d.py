"""Build the RESEARCH daily-bar cache: SIP volume, split+dividend adjusted, from 2016.

Separate from the live cache (Data/shared/bars/1d), which mixes IEX-era and SIP
volume and carries unadjusted corporate actions from an older writer. Live code
never reads this directory.

    .venv/bin/python -m scripts.research_data.fetch_research_bars_1d            # universe + ETFs
    .venv/bin/python -m scripts.research_data.fetch_research_bars_1d --tickers META NVDA
    .venv/bin/python -m scripts.research_data.fetch_research_bars_1d --tickers-file syms.txt --out-dir Data/research/bars_1d_sip_adj_extra

Output: Data/research/bars_1d_sip_adj/{TICKER}.parquet + _manifest.json
Adjustment "all" means historic prices change whenever a new split/dividend
happens, so each file is a snapshot as of `fetched_at_utc` (recorded in the manifest).
Survivorship: the ticker list is today's universe; delisted names are absent.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "Data" / "research" / "bars_1d_sip_adj"
UNIVERSE_CSV = REPO / "Data" / "shared" / "universe" / "shared_universe.csv"
START = "2016-01-01T00:00:00Z"
FEED, ADJUSTMENT = "sip", "all"

# Broad + thematic ETFs for the long-horizon sleeve and as benchmarks.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "RSP", "MDY", "VUG", "SCHG", "MGK",
    "XLK", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLU",
    "SMH", "SOXX", "IGV", "SKYY", "CIBR", "HACK", "BOTZ", "ROBO", "AIQ", "ARKK",
    "ARKG", "ARKQ", "XBI", "IBB", "ICLN", "TAN", "LIT", "URA", "NLR", "GRID",
    "PAVE", "ITA", "UFO", "FINX", "IPAY", "BLOK", "QTUM", "DRIV", "KWEB", "EEM",
    "VEA", "EFA", "INDA", "GLD", "SLV", "COPX", "TLT", "IEI", "HYG",
]

logger = logging.getLogger("fetch_research_bars_1d")


def _us_trading_day_hours(now: datetime | None = None) -> bool:
    """Weekdays 08:00-17:00 New York: when the live stack needs the same Alpaca quota."""
    ny = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    return ny.weekday() < 5 and 8 <= ny.hour < 17


def throttle_http(per_minute: int, market_per_minute: int | None = None) -> None:
    """Space every HTTP request in this process >= 60/per_minute s apart, shared by all threads
    (``market_per_minute`` instead on weekdays 08:00-17:00 ET, so the live stack keeps headroom).

    Alpaca allows 200 data requests a minute per account, and a 1Hour history comes back
    ~2,000 bars a page (~20 requests per symbol). Unthrottled workers trip 429s; the SDK
    then gives up on the symbol and the retry restarts it from page one, so adding workers
    LOWERED throughput (2026-11-08: 12 workers fetched 1.75 symbols/min against 5 with 5).
    """
    import requests

    lock, last = threading.Lock(), [0.0]
    original = requests.Session.request

    def request(self, *args, **kwargs):
        rpm = market_per_minute if market_per_minute and _us_trading_day_hours() else per_minute
        with lock:
            wait = last[0] + 60.0 / rpm - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
        return original(self, *args, **kwargs)

    requests.Session.request = request


def universe_tickers() -> list[str]:
    u = pd.read_csv(UNIVERSE_CSV)
    return sorted(set(u["ticker"].dropna().astype(str).str.upper()) | set(ETFS))


def fetch_ticker(ticker: str, end: str, force: bool, out_dir: Path = OUT_DIR, start: str = START,
                 timeframe: str = "1Day") -> tuple[str, int | str]:
    out = out_dir / f"{ticker}.parquet"
    if out.exists() and not force:
        return ticker, "cached"
    for attempt in range(1, 6):
        try:
            # skip_delisted=False: dead symbols are the point of a point-in-time cache.
            # limit=None: alpaca-py treats `limit` as a cap on the TOTAL bars and paginates
            # internally, so limit=10_000 silently returned only the first ~2.5 years of
            # hourly bars (2026-11-08 probe). None fetches every page.
            df = fetch_intraday(ticker=ticker, start=start, end=end, timeframe=timeframe,
                                limit=None, adjustment=ADJUSTMENT, feed=FEED, save_path="",
                                skip_delisted=False)
            break
        except Exception as exc:  # noqa: BLE001 - classify then retry or report
            msg = str(exc).lower()
            if ("429" in msg or "too many" in msg or "rate" in msg) and attempt < 5:
                time.sleep(min(60, 2 ** attempt))
                continue
            return ticker, f"error: {exc}"[:200]
    if df is None or df.empty:
        return ticker, "empty"
    df = df.drop_duplicates(["symbol", "timestamp"]).sort_values("timestamp")
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return ticker, len(df)


def main(*, description: str = __doc__.splitlines()[0], default_out: Path = OUT_DIR,
         default_tickers=universe_tickers, timeframe: str = "1Day") -> None:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--tickers-file", type=Path, help="one symbol per line (e.g. delisted names from build_pit_universe)")
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="refetch files that already exist")
    ap.add_argument("--max-rpm", type=int, default=170, help="HTTP requests per minute for this process "
                    "(Alpaca's account limit is 200)")
    ap.add_argument("--max-rpm-market", type=int, default=60,
                    help="the same on weekdays 08:00-17:00 ET, leaving the live stack most of the quota")
    args = ap.parse_args()
    throttle_http(args.max_rpm, args.max_rpm_market)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("core").setLevel(logging.WARNING)

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tickers_file:
        tickers = [t.strip().upper() for t in args.tickers_file.read_text().split() if t.strip()]
    else:
        tickers = [t.upper() for t in args.tickers] if args.tickers else default_tickers()
    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    results: dict[str, int | str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_ticker, t, end, args.force, out_dir, START, timeframe) for t in tickers]
        for i, fut in enumerate(as_completed(futs), 1):
            t, r = fut.result()
            results[t] = r
            if i % 200 == 0:
                logger.info("%d/%d done", i, len(tickers))

    ok = [t for t, r in results.items() if isinstance(r, int) or r == "cached"]
    bad = {t: r for t, r in results.items() if t not in ok}
    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    manifest["runs"].append({
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": START, "end": end, "timeframe": timeframe, "feed": FEED, "adjustment": ADJUSTMENT,
        "requested": len(tickers), "ok": len(ok), "failed": bad,
    })
    manifest_path.write_text(json.dumps(manifest, indent=1))
    logger.info("ok %d / %d; failed %d (see %s)", len(ok), len(tickers), len(bad), manifest_path)


if __name__ == "__main__":
    main()
