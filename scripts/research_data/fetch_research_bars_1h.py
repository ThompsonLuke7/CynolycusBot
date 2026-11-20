"""Build the RESEARCH hourly-bar cache: SIP, split+dividend adjusted, from 2016, point-in-time.

The symbols are every stock in the point-in-time universe (delisted names included;
Alpaca still serves their history under the old symbol) plus the 4H context ETFs.
One file per SYMBOL: a reused ticker's lives arrive together and are separated later
by the life windows in `Data/research/pit_universe/securities.parquet`. Live code never
reads this directory. Pre-registered in research/execution_quality/27_stage3_preregistration.md.

    .venv/bin/python -m scripts.research_data.fetch_research_bars_1h              # PIT stocks + context ETFs
    .venv/bin/python -m scripts.research_data.fetch_research_bars_1h --tickers SPY BBBY --out-dir /tmp/probe

Output: Data/research/bars_1h_sip_adj/{SYMBOL}.parquet + _manifest.json
Bars are filtered to the live RTH window by fetch_intraday (no 09:00 bar, the 16:00 bar
kept), so 4H bars built from them match the live feature pipeline.
"""
from __future__ import annotations

import pandas as pd

from scripts.research_data.fetch_research_bars_1d import REPO, main
from strategies.momentum_expansion.config.momentum_config import CONTEXT_TICKERS, SECTOR_ETFS

OUT_DIR = REPO / "Data" / "research" / "bars_1h_sip_adj"
SECURITIES = REPO / "Data" / "research" / "pit_universe" / "securities.parquet"


def pit_symbols() -> list[str]:
    sec = pd.read_parquet(SECURITIES, columns=["symbol", "kind"])
    stocks = set(sec.loc[sec["kind"] == "stock", "symbol"].astype(str).str.upper())
    return sorted(stocks | set(CONTEXT_TICKERS) | set(SECTOR_ETFS))


if __name__ == "__main__":
    main(description=__doc__.splitlines()[0], default_out=OUT_DIR, default_tickers=pit_symbols,
         timeframe="1Hour")
