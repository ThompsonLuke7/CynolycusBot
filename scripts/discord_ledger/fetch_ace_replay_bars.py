"""Fetch immutable, explicitly requested historical bars for the ACE replay panel.

Uses existing Alpaca credentials/entitlement. No subscription/account/order
changes. SIP is requested explicitly, with no silent IEX fallback. Adds each
entry's next trading session for a two-session path, plus October 6-8 screenshot
case symbols. A response's row coverage is recorded rather than inferred from
HTTP success. Original IEX research data is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import time

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

from core.API.Alpaca_API.core.config import AlpacaConfig
from core.calendar.us_market_calendar import next_trading_day
from scripts.discord_ledger.build import digest
from scripts.discord_ledger.run_ace_sequence_replay import source_labels, session_bounds, utc


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    raw_dir=args.out/'raw_session_bars';raw_dir.mkdir()
    entries,management,_=source_labels()
    panels=defaultdict(set)
    for r in entries:
        panels[r['session_date_et']].add(r['symbol'])
        panels[next_trading_day(date.fromisoformat(r['session_date_et'])).isoformat()].add(r['symbol'])
    for r in management:panels[r['session_date_et']].add(r['symbol'])
    for day in ('2026-10-06','2026-10-07','2026-10-08'):
        panels[day].update({'ARM','INTC','NBIS','ALAB','COHR','MU','SPY'})
    cfg=AlpacaConfig.from_env()
    client=StockHistoricalDataClient(cfg.key_id,cfg.secret_key)
    manifest=[]
    for day,symbols in sorted(panels.items()):
        start,end=session_bounds(day)
        requested=sorted(symbols)
        limit=10000
        if len(requested)*int((end-start).total_seconds()/60)>limit:
            raise ValueError('split this session request; refuse a potentially truncated response')
        raw=client.get_stock_bars(StockBarsRequest(symbol_or_symbols=requested,start=start.to_pydatetime(),
            end=end.to_pydatetime(),timeframe=TimeFrame.Minute,feed=DataFeed.SIP,adjustment=Adjustment.RAW,limit=limit))
        frame=raw.df.reset_index()
        if frame.empty:
            frame=pd.DataFrame(columns=['symbol','timestamp','open','high','low','close','volume','trade_count','vwap'])
        if frame.duplicated(['symbol','timestamp']).any():raise ValueError(f'duplicate source observations {day}')
        p=raw_dir/f'{day}.parquet';frame.to_parquet(p,index=False)
        counts=frame.groupby('symbol').size().to_dict() if len(frame) else {}
        manifest.append({'session_date_et':day,'requested_symbols':requested,'rows':len(frame),
                         'symbol_minute_counts':{s:int(counts.get(s,0)) for s in requested},
                         'missing_requested_symbols':[s for s in requested if not counts.get(s)],
                         'expected_minutes_per_symbol':int((end-start).total_seconds()/60),
                         'path':str(p),'sha256':digest(p),'feed_requested':'SIP','adjustment_requested':'raw',
                         'retrieved_at_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        print(json.dumps({'day':day,'requested':len(requested),'rows':len(frame),'missing':manifest[-1]['missing_requested_symbols']}),flush=True)
        time.sleep(.35)  # bounded to well below the existing data request rate limit
    (args.out/'manifest.json').write_text(json.dumps({'schema':'ace_historical_sip_capture_v1',
        'feed':'SIP','adjustment':'raw','calendar':'explicit NYSE 2025/2026 holidays and half days',
        'subscription_modified':False,'vendor_publication_time_certified':False,'sessions':manifest},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(args.out),'sessions':len(manifest),'rows':sum(r['rows'] for r in manifest)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
