"""Probe Alpaca for every symbol that has traded since 2016, including delisted ones.

Why: today's universe is survivor-shaped. Alpaca's asset list omits most delisted names (TWTR,
SIVB, LK are absent), but its bar endpoint still serves their history under the old symbol.
So the reliable way to enumerate them is to ask for bars. This requests MONTHLY raw SIP bars
for every 1-4 letter A-Z symbol, plus any extra candidates (5-letter and dotted symbols from
the Alpaca asset list and Tiingo's supported_tickers list). Unknown symbols return nothing.
Renamed tickers (FB -> META) return nothing under the old name, so there are no duplicates.

    .venv/bin/python -m scripts.research_data.probe_symbol_history

Output: Data/research/pit_universe/probe_parts/part_#####.parquet (one per batch; resumable)
        Data/research/pit_universe/symbol_months.parquet (symbol, month, close, volume, vwap, trade_count)
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import logging
import string
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest

from core.API.Alpaca_API.core.config import AlpacaConfig

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "Data" / "research" / "pit_universe"
PARTS = OUT_DIR / "probe_parts"
TIINGO_DIR = REPO / "Data" / "raw" / "tiingo"
START = dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc)
BATCH = 400
LISTED = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "NYSE ARCA", "NYSE MKT", "NYSE NAT"}

logger = logging.getLogger("probe_symbol_history")


def asset_table(cfg: AlpacaConfig) -> pd.DataFrame:
    """Alpaca's asset master (active + inactive US equities). It gives names for ETF screening."""
    tc = TradingClient(cfg.key_id, cfg.secret_key, paper=True)
    rows = [{"symbol": a.symbol, "name": a.name, "exchange": str(a.exchange.value), "status": str(a.status.value)}
            for a in tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))]
    return pd.DataFrame(rows)


def tiingo_table() -> pd.DataFrame:
    zips = sorted(TIINGO_DIR.glob("supported_tickers_*.zip"))
    if not zips:
        return pd.DataFrame(columns=["ticker", "exchange", "assetType", "startDate", "endDate"])
    with zipfile.ZipFile(zips[-1]) as z:
        t = pd.read_csv(z.open(z.namelist()[0]))
    t["ticker"] = t["ticker"].astype(str).str.upper().str.replace("-", ".", regex=False)
    return t


def candidates(assets: pd.DataFrame, tiingo: pd.DataFrame) -> list[str]:
    letters = string.ascii_uppercase
    brute = ["".join(p) for n in range(1, 5) for p in itertools.product(letters, repeat=n)]
    extra = set(assets.loc[assets["exchange"].isin(LISTED), "symbol"])
    extra |= set(tiingo.loc[tiingo["exchange"].isin(LISTED), "ticker"])
    extra = {s for s in extra if isinstance(s, str) and 0 < len(s) <= 6 and s.replace(".", "").isalpha()}
    return brute + sorted(extra - set(brute))


def fetch_batch(client: StockHistoricalDataClient, i: int, syms: list[str], end: dt.datetime) -> tuple[int, int]:
    out = PARTS / f"part_{i:05d}.parquet"
    if out.exists():
        return i, -1
    for attempt in range(1, 7):
        try:
            resp = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=syms, timeframe=TimeFrame.Month, start=START, end=end,
                adjustment=Adjustment.RAW, feed=DataFeed.SIP))
            break
        except Exception as exc:  # noqa: BLE001 - retry rate limits, report the rest
            msg = str(exc).lower()
            if attempt < 6 and ("429" in msg or "too many" in msg or "rate" in msg or "timed out" in msg or "50" in msg):
                time.sleep(min(60, 2 ** attempt))
                continue
            raise
    df = resp.df.reset_index() if resp.data else pd.DataFrame(columns=["symbol", "timestamp", "close", "volume", "vwap", "trade_count"])
    df = df[["symbol", "timestamp", "close", "volume", "vwap", "trade_count"]]
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return i, len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    PARTS.mkdir(parents=True, exist_ok=True)
    cfg = AlpacaConfig.from_env()
    assets = asset_table(cfg)
    assets.to_parquet(OUT_DIR / "alpaca_assets.parquet", index=False)
    syms = candidates(assets, tiingo_table())
    batches = [syms[k:k + BATCH] for k in range(0, len(syms), BATCH)]
    logger.info("%d candidate symbols, %d batches", len(syms), len(batches))
    end = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_batch, client, i, b, end) for i, b in enumerate(batches)]
        for n, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if n % 100 == 0:
                logger.info("%d/%d batches", n, len(batches))
    allm = pd.concat([pd.read_parquet(f) for f in sorted(PARTS.glob("part_*.parquet"))], ignore_index=True)
    allm = allm.rename(columns={"timestamp": "month"})
    allm.to_parquet(OUT_DIR / "symbol_months.parquet", index=False)
    logger.info("wrote symbol_months: %d rows, %d symbols, %s..%s", len(allm), allm["symbol"].nunique(),
                allm["month"].min(), allm["month"].max())


if __name__ == "__main__":
    main()
