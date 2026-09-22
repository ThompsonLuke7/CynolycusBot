"""Does 'pullback to a rising 100 EMA' have forward edge in OUR data?

Screen, not a validated edge. Two controls, because the repo has already been
burned by an uncontrolled top-k number (see research/execution_quality/25_*).
  CONTROL A: all stock-days in the same universe/date range (the base rate).
  CONTROL B: same-DAY, same ATR%-decile draws (kills the vol/beta tilt that
             faked +2-8pp excess for a noise-trained model).
CAVEAT: Data/shared/bars/1d is UNADJUSTED and SURVIVOR-shaped. Levels are
inflated; the setup-vs-control CONTRAST is the only thing read here.
"""
import glob, os
import numpy as np, pandas as pd

FILES = sorted(glob.glob('Data/shared/bars/1d/*.parquet'))
START, END = pd.Timestamp('2026-01-01').date(), pd.Timestamp('2026-08-20').date()
recs = []
for p in FILES:
    t = os.path.basename(p)[:-8]
    try: df = pd.read_parquet(p, columns=['timestamp','open','high','low','close','volume'])
    except Exception: continue
    if len(df) < 260: continue
    df = df.sort_values('timestamp').reset_index(drop=True)
    c, h, lo, v = df['close'], df['high'], df['low'], df['volume']
    if (c.tail(120) * v.tail(120)).median() < 5e6: continue       # liquidity floor
    pc = c.shift(1)
    tr = pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    e100 = c.ewm(span=100, adjust=False).mean()
    rh, rl = h.rolling(20).max(), lo.rolling(20).min()
    d = pd.DataFrame({
        'ticker': t,
        'date': df['timestamp'].dt.tz_convert('UTC').dt.date,
        'close': c,
        'atr_pct': atr / c,
        'ema_atr': (c - e100) / atr.replace(0, np.nan),
        'slope': (e100 - e100.shift(20)) / atr.replace(0, np.nan),
        'range_pos': ((c - rl) / (rh - rl).replace(0, np.nan)).clip(0,1),
        'ret20': c.pct_change(20),
        'fwd10': c.shift(-10)/c - 1.0,
        'fwd20': c.shift(-20)/c - 1.0,
    })
    d = d[(d.date >= START) & (d.date <= END)]
    recs.append(d)

U = pd.concat(recs, ignore_index=True).dropna(subset=['ema_atr','slope','range_pos','fwd10','fwd20','atr_pct'])
print(f"universe: {U.ticker.nunique()} tickers, {len(U):,} stock-days, {U.date.min()}..{U.date.max()}")
U['atr_decile'] = U.groupby('date')['atr_pct'].transform(lambda s: pd.qcut(s, 10, labels=False, duplicates='drop'))

# THE SETUP: rising 100 EMA, price AT it (not extended), pulled back in its 20d range
setup = U[(U.slope > 0) & (U.ema_atr.between(-0.5, 1.5)) & (U.range_pos <= 0.35) & (U.ret20 > -0.10)]
print(f"\nSETUP 'pullback to rising 100 EMA': {len(setup):,} stock-days "
      f"({len(setup)/len(U)*100:.2f}% of all) on {setup.ticker.nunique()} tickers, "
      f"{setup.date.nunique()} distinct dates")

def line(name, s):
    return (f"  {name:<34} n={len(s):>7,}  fwd10={s.fwd10.mean()*100:>6.2f}%  "
            f"fwd20={s.fwd20.mean()*100:>6.2f}%  win20={(s.fwd20>0).mean()*100:>5.1f}%")

print("\n=== SETUP vs CONTROLS ===")
print(line('SETUP', setup))
print(line('CONTROL A: whole universe', U))

# CONTROL B: same date + same ATR decile, sampled to the setup's own composition
rng = np.random.default_rng(7)
keys = setup.groupby(['date','atr_decile']).size()
picks = []
for (dt, dec), n in keys.items():
    pool = U[(U.date == dt) & (U.atr_decile == dec)]
    if len(pool) == 0: continue
    picks.append(pool.sample(n=min(n*5, len(pool)), random_state=int(rng.integers(1e6))))
CB = pd.concat(picks, ignore_index=True)
print(line('CONTROL B: same-day, same ATR decile', CB))

ex10 = (setup.fwd10.mean() - CB.fwd10.mean())*100
ex20 = (setup.fwd20.mean() - CB.fwd20.mean())*100
print(f"\n  EXCESS over the matched control:  10d {ex10:+.2f}pp   20d {ex20:+.2f}pp")

from scipy import stats
for hz in ('fwd10','fwd20'):
    t, p = stats.ttest_ind(setup[hz], CB[hz], equal_var=False)
    print(f"  {hz}: Welch t={t:+.2f}  p={p:.4g}")

# stability: does the sign hold month by month? (one big month can carry it)
print("\n=== MONTH BY MONTH (fwd20 excess vs matched control, pp) ===")
setup = setup.copy(); CB = CB.copy()
setup['m'] = pd.to_datetime(setup.date).values.astype('datetime64[M]')
CB['m'] = pd.to_datetime(CB.date).values.astype('datetime64[M]')
for m in sorted(setup.m.unique()):
    a, b = setup[setup.m==m], CB[CB.m==m]
    if len(a) < 20 or len(b) < 20: continue
    print(f"  {str(m)[:7]}  n={len(a):>5}  excess={((a.fwd20.mean()-b.fwd20.mean())*100):+6.2f}pp")
