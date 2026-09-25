"""EXPERIMENT B — DTE x HOLD counterfactual, Black-Scholes repriced.

MODEL-BASED. Read the limits before quoting any number:
  * Option values come from `research/options_lab/pricing.bsm_price` on REAL
    underlying paths, NOT from historical option bars. This is deliberately a
    DIFFERENT method from the 2026-07 study that was retracted for marking
    positions off stale trade prints. Its weakness is different and disclosed:
  * IV is ASSUMED (trailing realised vol x a VRP factor) and held CONSTANT over
    the hold. So this isolates THETA and DELTA. It cannot see vega/vol-path
    effects, IV crush after a catalyst, or the smile.
  * No spread or commission is charged, so every cell is optimistic in level.
    The comparison ACROSS DTE for the same path and hold is the output; the
    absolute returns are not.
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
print(f"{len(entries)} unique option entries replayed\n")

R=0.04
DTES=[7,15,30,45,60]
HOLDS=[3,5,10,20]
VRP=[1.0,1.25]
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
    K=round(S0)                      # ~ATM, as the live selector targets
    for vf in VRP:
        sig=rv*vf
        for dte in DTES:
            T0=dte/365.0
            p0=float(bsm_price(S0,K,T0,R,0.0,sig,'c'))
            if p0<=0.01: continue
            for h in HOLDS:
                S1=df.at[i+h,'close']
                T1=max((dte-h)/365.0,0.0)
                p1=float(bsm_price(S1,K,T1,R,0.0,sig,'c'))
                rows.append(dict(ticker=t,vrp=vf,dte=dte,hold=h,
                                 opt_ret=p1/p0-1.0, u_ret=S1/S0-1.0))
D=pd.DataFrame(rows)
pd.set_option('display.width',220)
print(f"{len(D):,} repriced cells\n")
for vf in VRP:
    sub=D[D.vrp==vf]
    print(f"=== MEAN OPTION RETURN (%), IV = trailing RV x {vf:g}  [theta+delta only] ===")
    piv=sub.pivot_table(index='dte',columns='hold',values='opt_ret',aggfunc='mean').mul(100).round(1)
    piv.columns=[f'hold {c}d' for c in piv.columns]
    print(piv.to_string())
    print()
print("=== BEST DTE for each hold (IV = trailing RV, vrp 1.0) ===")
s=D[D.vrp==1.0].groupby(['hold','dte']).opt_ret.mean().reset_index()
for h in HOLDS:
    r=s[s.hold==h].sort_values('opt_ret',ascending=False)
    print(f"  hold {h:>2}d -> best DTE {int(r.iloc[0].dte):>2} ({r.iloc[0].opt_ret*100:+.1f}%), "
          f"worst DTE {int(r.iloc[-1].dte):>2} ({r.iloc[-1].opt_ret*100:+.1f}%)")
print(f"\nunderlying mean return by hold: "
      f"{ {h: round(D[(D.vrp==1.0)&(D.hold==h)].u_ret.mean()*100,2) for h in HOLDS} }")
