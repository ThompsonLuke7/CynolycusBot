"""Is P&L CONDITIONAL on extension actually different, or does the cap just
make us trade less? Total loss falls under any rule that refuses trades when
mean P&L is negative -- that is not evidence. The test is whether the extended
cohort's PER-TRADE outcome is worse than the non-extended cohort's."""
import json, glob, os
import numpy as np, pandas as pd
from scipy import stats

BARS='Data/shared/bars/1d'; cache={}
def load(t):
    if t in cache: return cache[t]
    p=f'{BARS}/{t}.parquet'
    if not os.path.exists(p): cache[t]=None; return None
    df=pd.read_parquet(p,columns=['timestamp','high','low','close']).sort_values('timestamp').reset_index(drop=True)
    c,h,lo=df['close'],df['high'],df['low']; pc=c.shift(1)
    tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean(); df['atr_pct']=atr/c
    e100=c.ewm(span=100,adjust=False).mean()
    df['ema100_atr']=(c-e100)/atr.replace(0,np.nan)
    df['date']=df['timestamp'].dt.tz_convert('UTC').dt.date
    cache[t]=df; return df

rows=[]
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    mod=os.path.basename(os.path.dirname(f))
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        t=d.get('ticker'); eb=d.get('entry_bar'); pnl=d.get('realized_pnl')
        if not t or not eb or pnl is None: continue
        try: ts=pd.Timestamp(eb)
        except Exception: continue
        if ts.tzinfo is None: ts=ts.tz_localize('UTC')
        df=load(t)
        if df is None: continue
        m=df[df['date']<=ts.tz_convert('UTC').date()]
        if len(m)<130: continue
        r=m.iloc[-1]
        if not np.isfinite(r['ema100_atr']): continue
        # trade RETURN, not dollars: removes the sizing confound entirely
        g=d.get('fill_gain')
        rows.append(dict(module=mod,ema100_atr=float(r['ema100_atr']),
            pnl=float(pnl), gain=float(g) if g is not None else np.nan,
            route=d.get('route')))
T=pd.DataFrame(rows)
print(f"{len(T)} trades; {T.gain.notna().sum()} carry fill_gain (trade RETURN)\n")

for thr in (2.0,3.0,4.0):
    ext=T[T.ema100_atr>thr]; norm=T[T.ema100_atr<=thr]
    print(f"=== threshold {thr:g} ATR above EMA100 ===")
    print(f"  extended : n={len(ext):>3}  mean P&L={ext.pnl.mean():>+9,.0f}  "
          f"mean return={ext.gain.mean()*100 if ext.gain.notna().any() else float('nan'):>+7.2f}%  win={ (ext.pnl>0).mean()*100:>5.1f}%")
    print(f"  the rest : n={len(norm):>3}  mean P&L={norm.pnl.mean():>+9,.0f}  "
          f"mean return={norm.gain.mean()*100 if norm.gain.notna().any() else float('nan'):>+7.2f}%  win={ (norm.pnl>0).mean()*100:>5.1f}%")
    t1,p1=stats.ttest_ind(ext.pnl,norm.pnl,equal_var=False)
    print(f"  P&L    difference: t={t1:+.2f} p={p1:.3g}")
    a,b=ext.gain.dropna(),norm.gain.dropna()
    if len(a)>10 and len(b)>10:
        t2,p2=stats.ttest_ind(a,b,equal_var=False)
        print(f"  RETURN difference: t={t2:+.2f} p={p2:.3g}   <-- the sizing-free test")
    print()

print("=== momentum_expansion only (where extension actually lives) ===")
M=T[T.module=='momentum_expansion']
for thr in (3.0,4.0):
    e,n=M[M.ema100_atr>thr],M[M.ema100_atr<=thr]
    if len(e)<8 or len(n)<8: continue
    ge,gn=e.gain.dropna(),n.gain.dropna()
    print(f"  >{thr:g} ATR: n={len(e)} mean_ret={ge.mean()*100:+.2f}%  |  "
          f"<= : n={len(n)} mean_ret={gn.mean()*100:+.2f}%", end='')
    if len(ge)>5 and len(gn)>5:
        t3,p3=stats.ttest_ind(ge,gn,equal_var=False); print(f"   t={t3:+.2f} p={p3:.3g}")
    else: print()
