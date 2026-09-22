"""User's hypothesis: the breakout architecture was built for flat/slightly-up
regimes but fires on jumpy/spiky names where vol reverses the trend at once.
Test it on momentum_expansion's OWN closed trades."""
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
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    df['atr_pct']=atr/c
    df['atr_pct_ratio']=df['atr_pct']/df['atr_pct'].rolling(100).mean()   # vol EXPANSION
    e100=c.ewm(span=100,adjust=False).mean()
    df['ema100_atr']=(c-e100)/atr.replace(0,np.nan)
    df['ret5']=c.pct_change(5); df['ret20']=c.pct_change(20)
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
        if not np.isfinite(r['atr_pct_ratio']): continue
        rows.append(dict(module=mod,ticker=t,
            atr_pct=float(r['atr_pct']), vol_ratio=float(r['atr_pct_ratio']),
            ema100_atr=float(r['ema100_atr']),
            ret5=float(r['ret5']) if np.isfinite(r['ret5']) else np.nan,
            pnl=float(d.get('realized_pnl') or 0.0),
            gain=d.get('fill_gain'), reason=str(d.get('exit_reason') or '')))
R=pd.DataFrame(rows)
pd.set_option('display.width',210)

M=R[R.module=='momentum_expansion'].copy()
print(f"=== momentum_expansion: {len(M)} entries ===")
print(f"median daily ATR as % of price at entry : {M.atr_pct.median()*100:.2f}%   "
      f"(whole-sample stock-day norm is ~3%)")
print(f"median vol EXPANSION ratio (ATR% vs its own 100d mean): {M.vol_ratio.median():.2f}x")
print(f"share entering while vol > 1.25x its own norm: {(M.vol_ratio>1.25).mean()*100:.1f}%")
print(f"share entering after a >+10% 5-day move    : {(M.ret5>0.10).mean()*100:.1f}%")
print(f"share entering after a >+20% 5-day move    : {(M.ret5>0.20).mean()*100:.1f}%")

print("\n=== momentum outcomes by VOLATILITY STATE at entry ===")
M['vol_bucket']=pd.cut(M.vol_ratio,[0,1.0,1.25,1.6,99],
                       labels=['calm (<1.0x)','normal (1.0-1.25x)','elevated (1.25-1.6x)','spiking (>1.6x)'])
g=M.groupby('vol_bucket',observed=True).agg(n=('pnl','size'),total=('pnl','sum'),
    med=('pnl','median'),win=('pnl',lambda s:(s>0).mean()*100)).round(1)
print(g.to_string())

print("\n=== momentum outcomes by HOW EXTENDED above the 100 EMA ===")
M['ext']=pd.cut(M.ema100_atr,[-99,2,4,6,99],
                labels=['<2 ATR','2-4 ATR','4-6 ATR','>6 ATR (parabolic)'])
g2=M.groupby('ext',observed=True).agg(n=('pnl','size'),total=('pnl','sum'),
    med=('pnl','median'),win=('pnl',lambda s:(s>0).mean()*100)).round(1)
print(g2.to_string())

print("\n=== how momentum trades END ===")
print(M.reason.value_counts().head(8).to_string())
