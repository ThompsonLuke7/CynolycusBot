"""EXPERIMENT C — the GAMMA SQUEEZE question: does short DTE own the right tail?

Mean return (Experiment B) says longer DTE is better. But a strategy whose payoff
is a fat right tail is not judged on the mean -- the question is whether short
DTE produces the +200%/+500% outcomes that pay for everything else.

Same Black-Scholes machinery and the same disclosed limits as Experiment B, PLUS
one that cuts specifically against this test:
  * IV is held CONSTANT. Real gamma squeezes come with IV EXPANSION, which pays
    a short-dated contract extra through vega. So this model UNDERSTATES the
    short-DTE tail. Any short-DTE tail advantage it finds is a LOWER BOUND, and
    a short-DTE tail DISadvantage is the weaker claim.
"""
import json, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from research.options_lab.pricing import bsm_price

BARS='Data/shared/bars/1d'; cache={}
def load(t):
    if t in cache: return cache[t]
    p=f'{BARS}/{t}.parquet'
    if not os.path.exists(p): cache[t]=None; return None
    df=pd.read_parquet(p,columns=['timestamp','open','high','low','close']).sort_values('timestamp').reset_index(drop=True)
    c,h,lo,o=df['close'],df['high'],df['low'],df['open']
    gk=(0.5*np.log(h/lo)**2-(2*np.log(2)-1)*np.log(c/o)**2).clip(lower=1e-10)
    df['rv20']=np.sqrt(gk.rolling(20).mean()*252)
    df['date']=df['timestamp'].dt.tz_convert('UTC').dt.date
    cache[t]=df; return df

entries=[]; seen=set()
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        t=d.get('ticker'); eb=d.get('entry_bar')
        if not t or not eb or d.get('route')!='option': continue
        try: ts=pd.Timestamp(eb)
        except Exception: continue
        if ts.tzinfo is None: ts=ts.tz_localize('UTC')
        k=(t,ts.tz_convert('UTC').date())
        if k in seen: continue
        seen.add(k); entries.append(k)

R=0.04
DTES=[2,7,15,30,45,60]
HOLDS=[1,2,3,5,10]
MONEY=[('ATM',1.00),('5% OTM',1.05),('10% OTM',1.10)]
rows=[]
for t,d0 in entries:
    df=load(t)
    if df is None: continue
    idx=df.index[df['date']<=d0]
    if len(idx)==0: continue
    i=idx[-1]
    if i+max(HOLDS)>=len(df): continue
    S0=df.at[i,'close']; rv=df.at[i,'rv20']
    if not (np.isfinite(S0) and S0>0 and np.isfinite(rv) and 0.05<rv<4.0): continue
    for mname,mult in MONEY:
        K=S0*mult
        for dte in DTES:
            p0=float(bsm_price(S0,K,dte/365.0,R,0.0,rv,'c'))
            if p0<=0.02: continue
            for h in HOLDS:
                if h>=dte: continue          # expired; not a comparable cell
                S1=df.at[i+h,'close']
                p1=float(bsm_price(S1,K,max((dte-h)/365.0,0.0),R,0.0,rv,'c'))
                rows.append(dict(money=mname,dte=dte,hold=h,ret=p1/p0-1.0))
D=pd.DataFrame(rows)
pd.set_option('display.width',230)
print(f"{len(entries)} entries | {len(D):,} repriced cells\n")

for h in (2,3,5):
    sub=D[(D.hold==h)&(D.money=='ATM')]
    if sub.empty: continue
    print(f"=== ATM, hold {h}d — TAIL vs MEAN by DTE ===")
    g=sub.groupby('dte').agg(n=('ret','size'),mean=('ret',lambda s:s.mean()*100),
        p90=('ret',lambda s:s.quantile(.90)*100),p99=('ret',lambda s:s.quantile(.99)*100),
        pct_gt100=('ret',lambda s:(s>1.0).mean()*100),
        pct_gt200=('ret',lambda s:(s>2.0).mean()*100)).round(1)
    print(g.to_string()); print()

print("=== OTM strikes, hold 3d — where squeezes actually pay (pct of cells > +100%) ===")
for mname,_ in MONEY:
    sub=D[(D.hold==3)&(D.money==mname)]
    if sub.empty: continue
    g=sub.groupby('dte').ret.agg(n='size',mean=lambda s:s.mean()*100,
        gt100=lambda s:(s>1.0).mean()*100, gt300=lambda s:(s>3.0).mean()*100).round(1)
    print(f"\n-- {mname} --"); print(g.to_string())
