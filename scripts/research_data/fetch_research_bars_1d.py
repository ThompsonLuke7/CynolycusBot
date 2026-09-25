"""Build the RESEARCH daily-bar cache: SIP volume, split+dividend adjusted, from 2016.

Separate from the live cache (Data/shared/bars/1d), which mixes IEX-era and SIP
volume and carries unadjusted corporate actions from an older writer. Live code
never reads this directory.

    .venv/bin/python -m scripts.research_data.fetch_research_bars_1d            # universe + ETFs
    .venv/bin/python -m scripts.research_data.fetch_research_bars_1d --tickers META NVDA

Output: Data/research/bars_1d_sip_adj/{TICKER}.parquet + _manifest.json
Adjustment "all" means historic prices change whenever a new split/dividend
happens, so each file is a snapshot as of `fetched_at_utc` (recorded in the manifest).
Survivorship: the ticker list is today's universe; delisted names are absent.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

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


def universe_tickers() -> list[str]:
    u = pd.read_csv(UNIVERSE_CSV)
    return sorted(set(u["ticker"].dropna().astype(str).str.upper()) | set(ETFS))


def fetch_ticker(ticker: str, end: str, force: bool) -> tuple[str, int | str]:
    out = OUT_DIR / f"{ticker}.parquet"
    if out.exists() and not force:
        return ticker, "cached"
    for attempt in range(1, 6):
        try:
            df = fetch_intraday(ticker=ticker, start=START, end=end, timeframe="1Day",
                                limit=10_000, adjustment=ADJUSTMENT, feed=FEED, save_path="")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="refetch files that already exist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("core").setLevel(logging.WARNING)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [t.upper() for t in args.tickers] if args.tickers else universe_tickers()
    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    results: dict[str, int | str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_ticker, t, end, args.force) for t in tickers]
        for i, fut in enumerate(as_completed(futs), 1):
            t, r = fut.result()
            results[t] = r
            if i % 200 == 0:
                logger.info("%d/%d done", i, len(tickers))

    ok = [t for t, r in results.items() if isinstance(r, int) or r == "cached"]
    bad = {t: r for t, r in results.items() if t not in ok}
    manifest_path = OUT_DIR / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    manifest["runs"].append({
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": START, "end": end, "feed": FEED, "adjustment": ADJUSTMENT,
        "requested": len(tickers), "ok": len(ok), "failed": bad,
    })
    manifest_path.write_text(json.dumps(manifest, indent=1))
    logger.info("ok %d / %d; failed %d (see %s)", len(ok), len(tickers), len(bad), manifest_path)


if __name__ == "__main__":
    main()
