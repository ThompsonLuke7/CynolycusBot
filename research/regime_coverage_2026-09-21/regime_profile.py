"""Empirical entry-structure profile per module.

For every closed trade with an entry_bar, ask: where did this module BUY,
relative to the ticker's own recent structure, on the entry date?
Daily bars. NOTE: Data/shared/bars/1d is UNADJUSTED and survivor-shaped —
fine for 'position within recent range' over a 3-month window, but a split
inside the window would distort; flagged per-ticker below.
"""
import json, glob, os
from collections import defaultdict
import numpy as np, pandas as pd

BARS = 'Data/shared/bars/1d'
cache = {}

def load(t):
    if t in cache: return cache[t]
    p = f'{BARS}/{t}.parquet'
    if not os.path.exists(p):
        cache[t] = None; return None
    df = pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)
    c, h, lo = df['close'], df['high'], df['low']
    pc = c.shift(1)
    tr = pd.concat([h-lo, (h-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['ema100'] = c.ewm(span=100, adjust=False).mean()
    rh, rl = h.rolling(20).max(), lo.rolling(20).min()
    df['range_pos_20'] = ((c-rl)/(rh-rl).replace(0, np.nan)).clip(0, 1)
    df['ema100_atr'] = (c - df['ema100'])/df['atr14'].replace(0, np.nan)
    df['ret_20'] = c.pct_change(20)
    df['ret_5'] = c.pct_change(5)
    df['date'] = df['timestamp'].dt.tz_convert('UTC').dt.date
    # split guard: any >40% single-day close move in the sampled window
    df['jump'] = c.pct_change().abs() > 0.40
    cache[t] = df; return df

rows = []
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    mod = os.path.basename(os.path.dirname(f))
    for line in open(f):
        try: d = json.loads(line)
        except: continue
        eb, t = d.get('entry_bar'), d.get('ticker')
        if not eb or not t: continue
        try: ts = pd.Timestamp(eb)
        except Exception: continue
        if ts.tzinfo is None: ts = ts.tz_localize('UTC')
        edate = ts.tz_convert('UTC').date()
        df = load(t)
        if df is None: continue
        m = df[df['date'] <= edate]
        if len(m) < 120: continue
        r = m.iloc[-1]
        if not np.isfinite(r['range_pos_20']) or not np.isfinite(r['ema100_atr']): continue
        rows.append(dict(module=mod, ticker=t, date=str(edate),
                         range_pos=float(r['range_pos_20']),
                         ema100_atr=float(r['ema100_atr']),
                         ret_20=float(r['ret_20']) if np.isfinite(r['ret_20']) else np.nan,
                         ret_5=float(r['ret_5']) if np.isfinite(r['ret_5']) else np.nan,
                         pnl=float(d.get('realized_pnl') or 0.0),
                         jump=bool(m.tail(20)['jump'].any())))

R = pd.DataFrame(rows)
R.to_csv('/tmp/claude-1001/-home-luket-repos-CynolycusBot/af61862f-c78b-48b5-980d-a075380fec90/scratchpad/entries.csv', index=False)
print(f"{len(R)} entries profiled across {R.module.nunique()} modules "
      f"({R.jump.sum()} on tickers with a >40% single-day move in the prior 20d — possible unadjusted split)\n")

pd.set_option('display.width', 220)
g = R.groupby('module').agg(
    n=('range_pos', 'size'),
    range_pos_med=('range_pos', 'median'),
    pct_top_third=('range_pos', lambda s: (s >= 0.667).mean()*100),
    pct_bot_third=('range_pos', lambda s: (s <= 0.333).mean()*100),
    ema100_atr_med=('ema100_atr', 'median'),
    pct_above_ema100=('ema100_atr', lambda s: (s > 0).mean()*100),
    ret20_med=('ret_20', 'median'),
    ret5_med=('ret_5', 'median'),
    pnl=('pnl', 'sum'),
).round(3)
print(g.to_string())
print("\nrange_pos = position within trailing 20d high/low (1.0 = at the high)")
print("ema100_atr = (close - EMA100) / ATR14  (how far ABOVE the 100 EMA, in ATRs)")
