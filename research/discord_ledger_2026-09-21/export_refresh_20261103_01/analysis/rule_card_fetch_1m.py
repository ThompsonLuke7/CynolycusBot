"""Cache SIP 1-minute RTH bars for the rule-card backtest universe (research only, read-only API use).

Universe = every symbol in the caller's published EM tables (SPX excluded: no equity bars) plus the
SPDR sector ETFs (his 10/23 scanner card picked XLF/XLV/XLK). One multi-symbol request per trading
day over 09:30-16:00 ET, so extended-hours bars are never paged. Split-adjusted so prior-day levels
and same-day prices share a basis. Saved per month; a month already on disk is skipped (resumable).
Usage: python rule_card_fetch_1m.py <repo_root> <out_dir> [start] [end]
"""
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

ROOT, OUT = sys.argv[1], Path(sys.argv[2])
START = dt.date.fromisoformat(sys.argv[3] if len(sys.argv) > 3 else '2022-01-03')
END = dt.date.fromisoformat(sys.argv[4] if len(sys.argv) > 4 else '2026-10-30')
sys.path.insert(0, ROOT)
from alpaca.data.enums import Adjustment, DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from core.API.Alpaca_API.core.config import AlpacaConfig  # noqa: E402
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402

EM_TABLE_NAMES = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'SMH', 'AAPL', 'TSLA', 'AMZN', 'NVDA', 'AMD', 'META',
                  'MSFT', 'AVGO', 'GOOG', 'GME', 'DJT', 'SOFI', 'PLTR', 'JPM', 'AMC']
SECTOR_ETFS = ['XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLC', 'XLRE']
UNIVERSE = EM_TABLE_NAMES + SECTOR_ETFS
ET = 'America/New_York'

cfg = AlpacaConfig.from_env()
client = StockHistoricalDataClient(cfg.key_id, cfg.secret_key)
OUT.mkdir(parents=True, exist_ok=True)

days = [d.date() for d in pd.date_range(START, END) if is_trading_day(d.date())]
for month, group in pd.Series(days).groupby(lambda i: days[i].strftime('%Y-%m')):
    path = OUT / f'{month}.parquet'
    if path.exists():
        continue
    frames, t0 = [], time.time()
    for d in group:
        req = StockBarsRequest(symbol_or_symbols=UNIVERSE, timeframe=TimeFrame.Minute,
                               start=pd.Timestamp(f'{d} 09:30', tz=ET).tz_convert('UTC').to_pydatetime(),
                               end=pd.Timestamp(f'{d} 16:00', tz=ET).tz_convert('UTC').to_pydatetime(),
                               feed=DataFeed.SIP, adjustment=Adjustment.SPLIT)
        for attempt in range(5):
            try:
                df = client.get_stock_bars(req).df
                break
            except Exception as exc:  # rate limit / transient network: back off, then fail loudly
                if attempt == 4:
                    raise
                print(f'{d} retry {attempt + 1}: {exc}', flush=True)
                time.sleep(5 * (attempt + 1))
        if len(df):
            frames.append(df.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    print(f'{month}: {len(group)} days, {len(out):,} bars, {out.symbol.nunique()} symbols, {time.time() - t0:.0f}s', flush=True)
print('done')
