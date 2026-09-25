"""EXPERIMENT A — stop-width sweep on REAL underlying paths.

Bars only. No option marks, so the 2026-07 retraction does not touch this.

Question: our option trades hold a median 3 days and stopped trades a median
4.5 days, while the thesis move takes a median ~10 days (doc 13). Does a wider
stop let the hold reach the move, and what does that do to the UNDERLYING P&L?

Method: replay each real entry (ticker + entry bar) forward on daily bars.
Exit on the first of: stop at k*ATR below entry (intraday LOW), or horizon H.
Reports realized hold, underlying return, and survival to day 10.
"""
import json, glob, os
import numpy as np, pandas as pd

BARS='Data/shared/bars/1d'; cache={}
def load(t):
    if t in cache: return cache[t]
    p=f'{BARS}/{t}.parquet'
    if not os.path.exists(p): cache[t]=None; return None
    df=pd.read_parquet(p,columns=['timestamp','high','low','close']).sort_values('timestamp').reset_index(drop=True)
    c,h,lo=df['close'],df['high'],df['low']; pc=c.shift(1)
    tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    df['atr14']=tr.ewm(alpha=1/14,adjust=False).mean()
    df['date']=df['timestamp'].dt.tz_convert('UTC').dt.date
    cache[t]=df; return df

entries=[]
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    mod=os.path.basename(os.path.dirname(f))
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        t=d.get('ticker'); eb=d.get('entry_bar')
        if not t or not eb: continue
        try: ts=pd.Timestamp(eb)
        except Exception: continue
        if ts.tzinfo is None: ts=ts.tz_localize('UTC')
        entries.append((mod,t,ts.tz_convert('UTC').date(),d.get('route')))
# de-dup: one replay per (ticker, entry date)
seen=set(); uniq=[]
for e in entries:
    k=(e[1],e[2])
    if k in seen: continue
    seen.add(k); uniq.append(e)
print(f"{len(uniq)} unique (ticker, entry-date) replays from {len(entries)} closed trades\n")

H=20
STOPS=[1.0,1.5,2.0,3.0,4.0,None]
rows=[]
for mod,t,d0,route in uniq:
    df=load(t)
    if df is None: continue
    i=df.index[df['date']<=d0]
    if len(i)==0: continue
    i=i[-1]
    if i+H>=len(df): continue
    atr=df.at[i,'atr14']; e=df.at[i,'close']
    if not np.isfinite(atr) or atr<=0 or e<=0: continue
    fut=df.iloc[i+1:i+1+H]
    for k in STOPS:
        stop=None if k is None else e-k*atr
        exit_i=None
        if stop is not None:
            hit=np.where(fut['low'].values<=stop)[0]
            if len(hit): exit_i=int(hit[0])
        if exit_i is None:
            ret=(fut['close'].values[-1]/e)-1.0; hold=H; stopped=False
        else:
            ret=(stop/e)-1.0; hold=exit_i+1; stopped=True
        rows.append(dict(module=mod,ticker=t,k=('none' if k is None else k),
                         ret=ret,hold=hold,stopped=stopped,route=route))
R=pd.DataFrame(rows)
pd.set_option('display.width',210)
print("=== STOP-WIDTH SWEEP (underlying, 20-day horizon) ===")
g=R.groupby('k').agg(n=('ret','size'),mean_ret=('ret',lambda s:s.mean()*100),
    med_ret=('ret',lambda s:s.median()*100),win=('ret',lambda s:(s>0).mean()*100),
    stopped_pct=('stopped',lambda s:s.mean()*100),med_hold=('hold','median'),
    mean_hold=('hold','mean'),survive_10d=('hold',lambda s:(s>=10).mean()*100)).round(2)
print(g.to_string())
print("\nmed_hold/mean_hold in trading days. survive_10d = share still open at day 10,")
print("i.e. still in the trade when the thesis move typically arrives.")
print("\n=== same, OPTION-routed entries only ===")
O=R[R.route=='option']
print(O.groupby('k').agg(n=('ret','size'),mean_ret=('ret',lambda s:s.mean()*100),
    win=('ret',lambda s:(s>0).mean()*100),stopped_pct=('stopped',lambda s:s.mean()*100),
    med_hold=('hold','median'),survive_10d=('hold',lambda s:(s>=10).mean()*100)).round(2).to_string())
R.to_csv('/tmp/claude-1001/-home-luket-repos-CynolycusBot/af61862f-c78b-48b5-980d-a075380fec90/scratchpad/exp_a.csv',index=False)
