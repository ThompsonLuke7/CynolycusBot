"""If stops are too tight for these names, the underlying should RECOVER right
after we get stopped out. Measure the underlying's forward return from the stop
date, and compare it to the same-ticker base rate."""
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
    df['atr_pct']=tr.ewm(alpha=1/14,adjust=False).mean()/c
    for n in (5,10,20): df[f'f{n}']=c.shift(-n)/c-1.0
    df['date']=df['timestamp'].dt.tz_convert('UTC').dt.date
    cache[t]=df; return df

rows=[]
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    mod=os.path.basename(os.path.dirname(f))
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        r=str(d.get('exit_reason') or ''); t=d.get('ticker')
        if 'stop' not in r or not t: continue
        try: ts=pd.Timestamp(d['ts'])
        except Exception: continue
        df=load(t)
        if df is None: continue
        m=df[df['date']<=ts.tz_convert('UTC').date()]
        if len(m)<30: continue
        i=m.index[-1]
        rr=df.loc[i]
        rows.append(dict(module=mod,ticker=t,reason=('option -39%' if '39' in r else
            'underlying -1.5ATR' if 'atr' in r.lower() else 'other stop'),
            atr_pct=float(rr['atr_pct']),
            f5=rr['f5'],f10=rr['f10'],f20=rr['f20'],
            pnl=float(d.get('realized_pnl') or 0)))
S=pd.DataFrame(rows).dropna(subset=['f10'])
pd.set_option('display.width',200)
print(f"=== {len(S)} stopped trades with forward data ===\n")
print("UNDERLYING return AFTER we were stopped out:")
g=S.groupby('reason').agg(n=('f10','size'),
    med_atr=('atr_pct','median'),
    fwd5=('f5',lambda s:s.mean()*100), fwd10=('f10',lambda s:s.mean()*100),
    fwd20=('f20',lambda s:s.mean()*100),
    up10=('f10',lambda s:(s>0).mean()*100), cost=('pnl','sum')).round(2)
print(g.to_string())
print()
allf=S[['f5','f10','f20']].mean()*100
print(f"ALL stops pooled: +{allf.f5:.2f}% (5d)  +{allf.f10:.2f}% (10d)  +{allf.f20:.2f}% (20d) "
      f"| {(S.f10>0).mean()*100:.1f}% higher 10d later")
print(f"median ATR of the stopped names: {S.atr_pct.median()*100:.2f}% per day")
print(f"\n-> a 10d move of +{allf.f10:.2f}% is {allf.f10/ (S.atr_pct.median()*100):.2f} daily ATRs of recovery we did not hold for.")
from scipy import stats
t_,p_=stats.ttest_1samp(S.f10.dropna(),0.0)
print(f"   t-test vs 0 on 10d forward: t={t_:+.2f} p={p_:.4g}  (n={S.f10.notna().sum()})")
