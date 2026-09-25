"""Test TWO candidate changes against the flat-$5,000 sizing we actually run.

Sizing today: core/live_4h_exec.py:75 target_notional=5000 for EVERY entry,
with no volatility term. A 15%-ATR name therefore carries ~7x the risk of a
2%-ATR name for the same dollars.

TEST 1 (sizing): inverse-volatility weights, mean-normalised so the SAME average
capital is deployed. A trade's RETURN is unchanged by its size at our notional
(no market impact in liquid names), so rescaling realized P&L is a valid
first-order counterfactual -- this is how Barroso & Santa-Clara evaluate it.
TEST 2 (selection): refuse entries more than N ATR above the EMA100.

NOT a backtest: it cannot capture changed portfolio interaction or buying-power
paths. It answers 'would this have helped on the trades we actually took?'
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
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    df['atr_pct']=atr/c
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
        try: ts=pd.Timestamp(eb); xt=pd.Timestamp(d['ts'])
        except Exception: continue
        if ts.tzinfo is None: ts=ts.tz_localize('UTC')
        df=load(t)
        if df is None: continue
        m=df[df['date']<=ts.tz_convert('UTC').date()]
        if len(m)<130: continue
        r=m.iloc[-1]
        if not np.isfinite(r['atr_pct']) or r['atr_pct']<=0: continue
        if not np.isfinite(r['ema100_atr']): continue
        rows.append(dict(module=mod,ticker=t,exit_ts=xt,
            atr_pct=float(r['atr_pct']),ema100_atr=float(r['ema100_atr']),
            pnl=float(pnl)))
T=pd.DataFrame(rows).sort_values('exit_ts').reset_index(drop=True)
print(f"{len(T)} closed trades with entry structure + volatility\n")

def stats(pnl_series, label):
    p=np.asarray(pnl_series,dtype=float)
    cum=np.cumsum(p); peak=np.maximum.accumulate(cum)
    dd=(cum-peak).min()
    sharpe=p.mean()/p.std()*np.sqrt(len(p)) if p.std()>0 else 0.0
    return dict(label=label,n=len(p),total=p.sum(),mean=p.mean(),
                med=np.median(p),win=(p>0).mean()*100,maxdd=dd,tstat=sharpe)

def show(rs):
    print(f"{'variant':<46} {'n':>5} {'total P&L':>13} {'mean':>9} {'win%':>6} {'maxDD':>13} {'t-stat':>7}")
    for r in rs:
        print(f"{r['label']:<46} {r['n']:>5} {r['total']:>13,.0f} {r['mean']:>9,.0f} "
              f"{r['win']:>6.1f} {r['maxdd']:>13,.0f} {r['tstat']:>7.2f}")

# ---------- TEST 1: inverse-vol sizing ----------
print("=== TEST 1: inverse-volatility sizing vs flat $5,000 (same avg capital) ===")
res=[stats(T.pnl,'BASELINE flat notional (what we run)')]
for cap in (2.0, 3.0, 4.0):
    w=(1.0/T.atr_pct).clip(upper=(1.0/T.atr_pct).median()*cap)
    w=w/w.mean()                                  # same average deployment
    res.append(stats(T.pnl*w, f'inverse-vol sizing (weight cap {cap:g}x median)'))
show(res)

print("\n  per-module effect of inverse-vol sizing (cap 3x):")
w=(1.0/T.atr_pct).clip(upper=(1.0/T.atr_pct).median()*3.0); w=w/w.mean()
T['pnl_vs']=T.pnl*w
g=T.groupby('module').agg(n=('pnl','size'),flat=('pnl','sum'),vol_scaled=('pnl_vs','sum')).round(0)
g['delta']=(g.vol_scaled-g.flat).round(0)
print(g.to_string())

# ---------- TEST 2: extension cap ----------
print("\n=== TEST 2: refuse entries more than N ATR above the EMA100 ===")
res2=[stats(T.pnl,'BASELINE no cap (what we run)')]
for n in (6.0, 4.0, 3.0, 2.0):
    k=T[T.ema100_atr<=n]
    res2.append(stats(k.pnl, f'cap: skip entries >{n:g} ATR above EMA100'))
show(res2)
print("\n  trades REFUSED by each cap, and what they made:")
for n in (6.0,4.0,3.0,2.0):
    d=T[T.ema100_atr>n]
    print(f"   >{n:g} ATR: {len(d):>3} trades refused, their P&L = {d.pnl.sum():>+12,.0f}")

# ---------- TEST 3: both ----------
print("\n=== TEST 3: both together ===")
res3=[stats(T.pnl,'BASELINE')]
for n in (4.0,3.0):
    k=T[T.ema100_atr<=n].copy()
    w2=(1.0/k.atr_pct).clip(upper=(1.0/k.atr_pct).median()*3.0); w2=w2/w2.mean()
    res3.append(stats(k.pnl*w2, f'inverse-vol (3x cap) + skip >{n:g} ATR'))
show(res3)
print("\nt-stat = mean/std * sqrt(n) on per-trade P&L. |t|<2 means not distinguishable from zero.")
