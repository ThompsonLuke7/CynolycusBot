"""Paired A/B with bootstrap CIs. Same signals, different DTE/hold/stop policy."""
import json, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from research.options_lab.pricing import bsm_price
HS=0.08; COMM=0.0065
BARS='Data/shared/bars/1d'; cache={}
def load(t):
    if t in cache: return cache[t]
    p=f'{BARS}/{t}.parquet'
    if not os.path.exists(p): cache[t]=None; return None
    df=pd.read_parquet(p,columns=['timestamp','open','high','low','close']).sort_values('timestamp').reset_index(drop=True)
    c,h,lo,o=df['close'],df['high'],df['low'],df['open']
    gk=(0.5*np.log(h/lo)**2-(2*np.log(2)-1)*np.log(c/o)**2).clip(lower=1e-10)
    df['rv20']=np.sqrt(gk.rolling(20).mean()*252)
    pc=c.shift(1)
    tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    df['atr14']=tr.ewm(alpha=1/14,adjust=False).mean()
    df['date']=df['timestamp'].dt.tz_convert('UTC').dt.date
    cache[t]=df; return df
GROUPS={'30m_swing':{'multi_ticker_swing'},
        '4H_modules':{'momentum_expansion','multi_ticker_swing_htf','meta_ranker'}}
ent={g:[] for g in GROUPS}; seen=set()
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    mod=os.path.basename(os.path.dirname(f))
    grp=next((g for g,s in GROUPS.items() if mod in s),None)
    if grp is None: continue
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        t=d.get('ticker'); eb=d.get('entry_bar')
        if not t or not eb: continue
        try: ts=pd.Timestamp(eb)
        except Exception: continue
        if ts.tzinfo is None: ts=ts.tz_localize('UTC')
        k=(grp,t,ts.tz_convert('UTC').date())
        if k in seen: continue
        seen.add(k); ent[grp].append((t,ts.tz_convert('UTC').date()))
def arm(entries,dte,hold,stop_atr):
    out={}
    for t,d0 in entries:
        df=load(t)
        if df is None: continue
        idx=df.index[df['date']<=d0]
        if len(idx)==0: continue
        i=idx[-1]
        if i+hold>=len(df): continue
        S0=df.at[i,'close']; rv=df.at[i,'rv20']; atr=df.at[i,'atr14']
        if not (np.isfinite(S0) and S0>0 and np.isfinite(rv) and 0.05<rv<4.0): continue
        if not (np.isfinite(atr) and atr>0): continue
        p0=float(bsm_price(S0,S0,dte/365.0,0.04,0.0,rv,'c'))
        if p0<=0.05: continue
        fut=df.iloc[i+1:i+1+hold]
        hit=np.where(fut['low'].values<=S0-stop_atr*atr)[0]
        off=int(hit[0]) if len(hit) else hold-1
        S1=fut['close'].values[off]; days=off+1
        p1=float(bsm_price(S1,S0,max((dte-days)/365.0,0.0),0.04,0.0,rv,'c'))
        out[(t,d0)]=max(p1-HS-COMM,0.0)/(p0+HS+COMM)-1.0
    return out
rng=np.random.default_rng(3)
for grp,entries in ent.items():
    A=arm(entries,3,2,1.5)
    Bp=arm(entries,30,10,2.5) if grp=="30m_swing" else arm(entries,45,20,3.0)
    CUR=arm(entries,24,1,1.5) if grp=='30m_swing' else arm(entries,17,4,1.5)
    keys=sorted(set(A)&set(Bp)&set(CUR))
    if len(keys)<15:
        print(f"{grp}: only {len(keys)} paired — skipped"); continue
    a=np.array([A[k] for k in keys]); b=np.array([Bp[k] for k in keys]); c=np.array([CUR[k] for k in keys])
    print(f"\n=== {grp}: {len(keys)} PAIRED entries (identical signals, different policy) ===")
    for lbl,x in (('A  short-DTE scalp (3/2d)',a),('B+ long-DTE hold',b),('CURRENT',c)):
        print(f"  {lbl:<28} mean {x.mean()*100:>7.1f}%  median {np.median(x)*100:>7.1f}%"
              f"  win {(x>0).mean()*100:>5.1f}%  >+100% {(x>1).mean()*100:>5.1f}%")
    for lbl,x in (('A  - CURRENT',a),('B+ - CURRENT',b)):
        d=x-c; idx=rng.integers(0,len(d),(4000,len(d))); s=d[idx].mean(axis=1)
        lo,hi=np.quantile(s,.025),np.quantile(s,.975)
        print(f"  {lbl}: {d.mean()*100:+7.1f}pp   95% CI [{lo*100:+.1f}, {hi*100:+.1f}]"
              f"   -> {'SIGNIFICANT' if (lo>0 or hi<0) else 'NOT distinguishable from 0'}")
