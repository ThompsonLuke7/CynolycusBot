"""Split entries by the two axes that define a trading regime:
   (1) is price above or below the 100 EMA?
   (2) is the 100 EMA rising or falling?
Those four quadrants are four DIFFERENT strategies with different edges."""
import json, glob, os
import numpy as np, pandas as pd

BARS='Data/shared/bars/1d'; cache={}
def load(t):
    if t in cache: return cache[t]
    p=f'{BARS}/{t}.parquet'
    if not os.path.exists(p): cache[t]=None; return None
    df=pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
    c,h,lo=df['close'],df['high'],df['low']; pc=c.shift(1)
    tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    df['atr14']=tr.ewm(alpha=1/14,adjust=False).mean()
    e100=c.ewm(span=100,adjust=False).mean(); df['ema100']=e100
    # slope: 20d change in the EMA, normalised by ATR -> scale-free trend
    df['ema100_slope']=(e100-e100.shift(20))/df['atr14'].replace(0,np.nan)
    rh,rl=h.rolling(20).max(),lo.rolling(20).min()
    df['range_pos_20']=((c-rl)/(rh-rl).replace(0,np.nan)).clip(0,1)
    df['ema100_atr']=(c-e100)/df['atr14'].replace(0,np.nan)
    df['date']=df['timestamp'].dt.tz_convert('UTC').dt.date
    cache[t]=df; return df

rows=[]
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    mod=os.path.basename(os.path.dirname(f))
    for line in open(f):
        try: d=json.loads(line)
        except: continue
        eb,t=d.get('entry_bar'),d.get('ticker')
        if not eb or not t: continue
        try: ts=pd.Timestamp(eb)
        except Exception: continue
        if ts.tzinfo is None: ts=ts.tz_localize('UTC')
        df=load(t)
        if df is None: continue
        m=df[df['date']<=ts.tz_convert('UTC').date()]
        if len(m)<130: continue
        r=m.iloc[-1]
        if not all(np.isfinite([r['range_pos_20'],r['ema100_atr'],r['ema100_slope']])): continue
        rows.append(dict(module=mod,ticker=t,
            above=r['ema100_atr']>0, rising=r['ema100_slope']>0,
            ema100_atr=float(r['ema100_atr']), slope=float(r['ema100_slope']),
            range_pos=float(r['range_pos_20']), pnl=float(d.get('realized_pnl') or 0.0)))
R=pd.DataFrame(rows)
R['quadrant']=np.where(R.rising,
    np.where(R.above,'UPTREND / extended above','UPTREND / pulled back below'),
    np.where(R.above,'DOWNTREND / bounce above','DOWNTREND / broken below'))
pd.set_option('display.width',230)

print("=== WHERE EACH MODULE ENTERS (% of its trades per quadrant) ===")
ct=pd.crosstab(R.module,R.quadrant,normalize='index').mul(100).round(1)
print(ct.to_string()); print()
print("=== P&L BY QUADRANT (all modules pooled) ===")
q=R.groupby('quadrant').agg(n=('pnl','size'),total_pnl=('pnl','sum'),
    median_pnl=('pnl','median'),win_rate=('pnl',lambda s:(s>0).mean()*100)).round(1)
print(q.sort_values('total_pnl',ascending=False).to_string()); print()
print("=== the user's setup: pullback to a RISING ema100, price still ABOVE it ===")
tgt=R[(R.rising)&(R.above)&(R.range_pos<=0.40)]
print(f"trades matching (rising EMA100, price above it, range_pos<=0.40): {len(tgt)} of {len(R)} ({len(tgt)/len(R)*100:.1f}%)")
if len(tgt): print(tgt.groupby('module').agg(n=('pnl','size'),pnl=('pnl','sum')).to_string())
