"""EXPERIMENT D — A/B: short-DTE+scalp  vs  long-DTE+hold, WITH spread charged.

Experiments B and C charged NO transaction cost, which quietly biased them toward
short DTE: spread is roughly a FIXED number of cents, so it is punitive on cheap
short-dated contracts and mild on expensive long-dated ones. From this repo's own
real-fill calibration (research/options_experiment/11_option_cost_reference.md,
575 live round-trips, the only option data that survived the retraction):
    ~8-cent HALF spread  ->  58% round-trip on a $0.28 premium, 2% on a $9.52 one.
Ignoring that makes a 3-DTE lottery ticket look free. This run charges it.

Still model-based (BS on real underlying paths, IV = trailing RV held constant,
no vol-path/vega). The ARMS are compared under identical assumptions.
"""
import json, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from research.options_lab.pricing import bsm_price

HALF_SPREAD = 0.08          # dollars/share, per side (repo's real-fill median)
COMMISSION  = 0.65/100.0    # $0.65 per contract -> per share

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

def run_arm(entries, dte, hold, stop_atr, money=1.0):
    """Replay one (DTE, hold, stop) policy. Returns net per-trade returns."""
    out=[]
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
        K=S0*money
        p0=float(bsm_price(S0,K,dte/365.0,0.04,0.0,rv,'c'))
        if p0<=0.05: continue                       # unpriceable / sub-penny
        fut=df.iloc[i+1:i+1+hold]
        stop=S0-stop_atr*atr if stop_atr else None
        exit_off=hold-1
        if stop is not None:
            hit=np.where(fut['low'].values<=stop)[0]
            if len(hit): exit_off=int(hit[0])
        S1=fut['close'].values[exit_off]
        days=exit_off+1
        p1=float(bsm_price(S1,K,max((dte-days)/365.0,0.0),0.04,0.0,rv,'c'))
        # pay the spread + commission on BOTH sides
        buy =p0+HALF_SPREAD+COMMISSION
        sell=max(p1-HALF_SPREAD-COMMISSION,0.0)
        out.append(sell/buy-1.0)
    return np.array(out)

ARMS={
 'A  short DTE + scalp   (3 DTE, 2d hold, 1.5ATR stop)':  dict(dte=3,  hold=2,  stop_atr=1.5),
 'A+ short DTE + scalp   (7 DTE, 3d hold, 1.5ATR stop)':  dict(dte=7,  hold=3,  stop_atr=1.5),
 'B  long DTE + hold     (30 DTE, 10d hold, 2.5ATR stop)':dict(dte=30, hold=10, stop_atr=2.5),
 'B+ long DTE + hold     (45 DTE, 20d hold, 3ATR stop)':  dict(dte=45, hold=20, stop_atr=3.0),
 'C  CURRENT 30m swing   (24 DTE, 1d hold, 1.5ATR stop)': dict(dte=24, hold=1,  stop_atr=1.5),
 'D  CURRENT 4H modules  (17 DTE, 4d hold, 1.5ATR stop)': dict(dte=17, hold=4,  stop_atr=1.5),
}
pd.set_option('display.width',230)
for grp,entries in ent.items():
    print(f"\n{'='*96}\n=== {grp}  ({len(entries)} unique entries) — NET of 8c half-spread + commission ===")
    print(f"{'arm':<56}{'n':>5}{'mean':>9}{'median':>9}{'win%':>7}{'>+100%':>8}{'p99':>9}")
    for name,cfg in ARMS.items():
        r=run_arm(entries,**cfg)
        if len(r)<20: continue
        print(f"{name:<56}{len(r):>5}{r.mean()*100:>8.1f}%{np.median(r)*100:>8.1f}%"
              f"{(r>0).mean()*100:>6.1f}%{(r>1).mean()*100:>7.1f}%{np.quantile(r,.99)*100:>8.0f}%")
