"""Prior-session features and underlying-only execution diagnostics."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.profit_investigation.build_study import DATA, END, read, ts
from core.calendar.us_market_calendar import is_trading_day

OUT=DATA.parent/'analysis'
manifest={}

@lru_cache(maxsize=1000)
def daily(ticker):
    p=ROOT/f'Data/shared/bars/1d/{ticker}.parquet'
    if not p.exists():return None
    manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    d=pd.read_parquet(p).sort_values('timestamp')
    assert not d.timestamp.duplicated().any()
    d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
    d['date']=d.timestamp.dt.tz_convert('America/New_York').dt.date
    d=d[d.timestamp<END].copy()
    d['r1']=d.close.pct_change(fill_method=None)
    prev=d.close.shift(1)
    d['tr']=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    return d.reset_index(drop=True)

def prior_features(ticker,when):
    d=daily(ticker);day=ts(when).tz_convert('America/New_York').date()
    if d is None:return {}
    p=d[d.date<day].tail(65)
    if len(p)<21 or (day-p.date.iloc[-1]).days>5:return {}
    last=p.iloc[-1];cl=p.close
    suspect=bool(((p.open/p.close.shift(1)>1.8)|(p.open/p.close.shift(1)<.55)).tail(21).any())
    feats={'prior_close':last.close,'prior_return_5d':cl.iloc[-1]/cl.iloc[-6]-1,
        'prior_return_20d':cl.iloc[-1]/cl.iloc[-21]-1,
        'prior_volatility_20d':p.r1.tail(20).std(),'prior_atr_pct':p.tr.tail(14).mean()/last.close,
        'prior_dollar_volume_20d':(p.close*p.volume).tail(20).median(),
        'prior_volume_ratio':last.volume/p.volume.iloc[-21:-1].mean(),
        'prior_range_position_20d':(last.close-p.low.tail(20).min())/(p.high.tail(20).max()-p.low.tail(20).min()),
        'prior_gap_flag':suspect,'feature_last_session':str(last.date)}
    spy=daily('SPY');sp=spy[spy.date<day]
    if len(sp)>=21:
        feats['prior_spy_return_5d']=sp.close.iloc[-1]/sp.close.iloc[-6]-1
        feats['prior_relative_spy_5d']=feats['prior_return_5d']-feats['prior_spy_return_5d']
    if suspect:
        # Preserve the price flag; refuse derived return/volatility features
        # around discontinuities rather than infer a corporate-action factor.
        for k in list(feats):
            if k not in ['prior_close','prior_gap_flag','feature_last_session']:feats[k]=None
    return feats

def load_minutes(tickers):
    chunks=defaultdict(list)
    for ticker in tickers:
        p=ROOT/f'research/execution_quality/data/bars_1m/{ticker}.parquet'
        if p.exists():
            manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            d=pd.read_parquet(p);d['source_kind']='historical_cache';chunks[ticker].append(d)
    # Archive observations replace cached bars at an identical symbol/minute;
    # first arrival wins within archive duplicates. They are outcome prices,
    # never features presumed available before their recorded arrival.
    rows=defaultdict(list)
    for p in sorted((ROOT/'Data/archive/intraday_1m').glob('bars_2026*.jsonl')):
        h=hashlib.sha256()
        for line in p.open('rb'):
            h.update(line);r=json.loads(line)
            if r.get('symbol') in tickers:
                rows[r['symbol']].append(r)
        manifest[str(p.relative_to(ROOT))]=h.hexdigest()
    output={};dedup=0
    for ticker in tickers:
        if rows[ticker]:
            d=pd.DataFrame(rows[ticker]);d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
            d=d.sort_values('arrival_at');dedup+=int(d.timestamp.duplicated().sum())
            d=d.drop_duplicates('timestamp',keep='first');d['source_kind']='archive';chunks[ticker].append(d)
        if not chunks[ticker]:continue
        d=pd.concat(chunks[ticker],ignore_index=True);d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
        d=d.drop_duplicates('timestamp',keep='last').sort_values('timestamp')
        et=d.timestamp.dt.tz_convert('America/New_York');minute=et.dt.hour*60+et.dt.minute
        d=d[(minute>=570)&(minute<960)&(d.timestamp<END)].set_index('timestamp')
        output[ticker]=d
    return output,dedup

def observed_price(d,t):
    # A left-labelled one-minute close becomes an outcome price one minute
    # after its stamp. Never use the still-forming bar's eventual close.
    before=d[d.index+pd.Timedelta(minutes=1)<=t]
    if before.empty:return None
    last=before.iloc[-1]
    if (t-before.index[-1]).total_seconds()>180:return None
    return float(last.close)

def expected_minutes(lo,hi):
    total=0
    for day in pd.date_range(lo.tz_convert('America/New_York').normalize(),hi.tz_convert('America/New_York').normalize(),freq='D'):
        if is_trading_day(day.date()):
            start=max(lo,day+pd.Timedelta(hours=9,minutes=30));end=min(hi,day+pd.Timedelta(hours=16))
            total+=max(0,(end-start).total_seconds()/60)
    return total

def path_features(r,d):
    if d is None or not r.get('exit_time'):return {}
    entry=ts(r['entry_time']);exit=ts(r['exit_time']);direction=-1 if r.get('opt_right')=='P' else 1
    ep=observed_price(d,entry);xp=observed_price(d,exit)
    ret={}
    if ep and xp:
        ret['underlying_return']=direction*(xp/ep-1)
    if r.get('available_at'):
        ap=observed_price(d,ts(r['available_at']))
        if ap and ep:ret['underlying_entry_drift']=direction*(ep/ap-1)
    # Fully contained bars only. Reject paths with missing sessions/minutes.
    w=d[(d.index>=entry)&(d.index+pd.Timedelta(minutes=1)<=exit)]
    expected=expected_minutes(entry,exit)
    ret['path_coverage']=min(1,len(w)/max(1,expected))
    if not ep or not xp or expected<5 or len(w)<5 or ret['path_coverage']<.80:return ret
    if direction==1:
        mfe=w.high.max()/ep-1;mae=w.low.min()/ep-1;peak=w.high.idxmax()
    else:
        mfe=1-w.low.min()/ep;mae=1-w.high.max()/ep;peak=w.low.idxmin()
    ret.update({'underlying_mfe':max(0,mfe),'underlying_mae':min(0,mae),
        'underlying_giveback':max(0,mfe)-ret['underlying_return'],
        'peak_fraction_of_hold':(peak-entry).total_seconds()/max(1,(exit-entry).total_seconds())})
    after=d[(d.index>=exit)&(d.index<exit+pd.Timedelta(minutes=60))]
    if len(after)>=45:
        ret['postexit_60m_return']=direction*(after.close.iloc[-1]/xp-1)
    return ret

def signal_outcomes(signals,cycles):
    traded={(c['module'],c['ticker'],str(ts(c['signal_bar']))) for c in cycles if c.get('signal_bar')}
    out=[]
    for r in signals:
        d=daily(r['ticker'])
        if d is None:continue
        day=ts(r['available_at']).tz_convert('America/New_York').date()
        future=d[d.date>day].reset_index(drop=True)
        rr={**r,'was_traded':(r['module'],r['ticker'],str(ts(r['signal_bar']))) in traded,
            **prior_features(r['ticker'],r['available_at'])}
        for h in [1,5,10]:
            if len(future)<h:continue
            w=future.iloc[:h];entry=float(w.open.iloc[0])
            if not entry>0 or ((w.open/w.close.shift(1)>1.8)|(w.open/w.close.shift(1)<.55)).any():continue
            rr[f'forward_{h}d']=float(w.close.iloc[-1])/entry-1
        out.append(rr)
    return out

def main():
    OUT.mkdir(exist_ok=False)
    cycles=read(DATA.parent/'run3/lifecycles.jsonl');signals=read(DATA.parent/'run3/signals.jsonl')
    print('daily feature build',len(cycles),flush=True)
    for c in cycles:c.update(prior_features(c['ticker'],c.get('available_at') or c['entry_time']))
    minutes,dup=load_minutes({r['ticker'] for r in cycles})
    print('minute paths',len(minutes),'archive duplicates',dup,flush=True)
    for c in cycles:c.update(path_features(c,minutes.get(c['ticker'])))
    outcomes=signal_outcomes(signals,cycles)
    for name,rows in [('trades',cycles),('signal_outcomes',outcomes)]:
        pd.DataFrame(rows).drop(columns=['entries','exits','owners'],errors='ignore').to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'price_source_hashes.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'coverage.json').write_text(json.dumps({'archive_duplicate_symbol_minutes':dup,
        'paths':sum('underlying_mfe' in r for r in cycles),'endpoint_returns':sum('underlying_return' in r for r in cycles),
        'signals':len(outcomes),'feature_available':sum('prior_close' in r for r in cycles)},indent=2))
    print('enrichment complete',flush=True)

if __name__=='__main__':main()
