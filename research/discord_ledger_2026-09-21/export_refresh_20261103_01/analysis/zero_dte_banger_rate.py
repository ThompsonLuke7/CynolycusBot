"""How often does a 0DTE SPY option print +100% after entry, with no skill at all? (research, expired contracts)

Every session 2025-01-02..2026-10-30: the ATM call and put (strike = nearest dollar to the SPY price at
the entry time) and the 2-dollar-OTM call and put, entered at the close of the 1-minute option bar at
09:45, 11:00 and 13:00 ET. "Banger" = any later print >= 2x entry the same day (the caller's recap
metric is the peak print after entry). Expiry value = intrinsic at the SPY 15:59 close. Option bars
are Alpaca trade bars (research.options_lab.chain_cache, cached); SPY 0DTE ATM contracts trade every
minute, and the sign-adjusted correlation of option vs SPY 1-minute returns is checked before use.
Usage: python zero_dte_banger_rate.py <repo_root> <bars_1m_dir> <out_csv>
"""
import glob, sys
import numpy as np, pandas as pd

ROOT, BARS, OUT = sys.argv[1:4]
sys.path.insert(0, ROOT)
from research.options_lab.chain_cache import fetch_bars  # noqa: E402

ET = 'America/New_York'
ENTRIES = {'09:45': 15, '11:00': 90, '13:00': 210}  # minutes after 09:30
spy = pd.concat([pd.read_parquet(f, filters=[('symbol', '==', 'SPY')]) for f in sorted(glob.glob(f'{BARS}/2025-*.parquet')) +
                 sorted(glob.glob(f'{BARS}/2026-*.parquet'))], ignore_index=True)
spy['ts'] = spy['timestamp'].dt.tz_convert(ET)
spy['date'] = spy['ts'].dt.date
spy['minute'] = (spy['ts'].dt.hour - 9) * 60 + spy['ts'].dt.minute - 30
spy = spy[(spy.minute >= 0) & (spy.minute < 390)]


def osi(d, right, strike):
    return f"SPY{d:%y%m%d}{right}{int(round(strike * 1000)):08d}"


rows, corr = [], []
for d, s in spy.groupby('date'):
    s = s.set_index('minute')
    if 15 not in s.index or 389 not in s.index:
        continue
    spx_close = s.loc[389, 'close']
    picks = {}
    for label, mnt in ENTRIES.items():
        k = round(s.loc[mnt, 'close'])
        for right, strike, tag in (('C', k, 'atm'), ('P', k, 'atm'), ('C', k + 2, 'otm2'), ('P', k - 2, 'otm2')):
            picks[(label, right, tag)] = osi(d, right, strike)
    syms = sorted(set(picks.values()))
    start = pd.Timestamp(f'{d} 09:30', tz=ET).tz_convert('UTC').isoformat()
    end = pd.Timestamp(f'{d} 16:00', tz=ET).tz_convert('UTC').isoformat()
    bars = fetch_bars(syms, '1Min', start, end)
    if bars.empty:
        continue
    bars['ts'] = pd.to_datetime(bars['t'], utc=True).dt.tz_convert(ET)
    bars['minute'] = (bars['ts'].dt.hour - 9) * 60 + bars['ts'].dt.minute - 30
    for (label, right, tag), sym in picks.items():
        b = bars[(bars.osi_symbol == sym) & (bars.minute >= 0) & (bars.minute < 390)].set_index('minute').sort_index()
        mnt = ENTRIES[label]
        at = b.loc[mnt:mnt + 4]
        if at.empty:
            continue
        e_min = at.index[0]; entry = at['c'].iloc[0]
        after = b.loc[e_min + 1:]
        if entry <= 0 or after.empty:
            continue
        strike = int(sym[-8:]) / 1000
        intrinsic = max(0.0, spx_close - strike) if right == 'C' else max(0.0, strike - spx_close)
        rows.append({'date': d, 'entry_time': label, 'right': right, 'moneyness': tag, 'symbol': sym, 'entry': entry,
                     'peak_mult': after['h'].max() / entry, 'expiry_mult': intrinsic / entry, 'bars_after': len(after),
                     'minutes_after': 389 - e_min})
        if label == '09:45' and tag == 'atm' and len(b) > 100:
            j = b.join(s[['close']].rename(columns={'close': 'spy'}), how='inner')
            rr = np.log(j['c']).diff(); ru = np.log(j['spy']).diff()
            corr.append({'right': right, 'corr': rr.corr(ru), 'bars': len(b)})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
cc = pd.DataFrame(corr)
print(f"sessions {df.date.nunique()}, contract-entries {len(df)}")
print('sanity: corr(option 1-min return, SPY 1-min return), ATM 09:45, median by right:',
      cc.groupby('right')['corr'].median().round(2).to_dict(), '; median bars/contract', int(cc.bars.median()))
df['peak2x'] = df.peak_mult >= 2
df['peak3x'] = df.peak_mult >= 3
df['dies'] = df.expiry_mult <= 0.1
print('\nshare of entries whose LATER same-day high is >= 2x entry ("100% banger"), >= 3x, and that expire <= 10% of entry:')
print(df.groupby(['entry_time', 'moneyness', 'right'])[['peak2x', 'peak3x', 'dies']].mean().round(3).to_string())
print('\nmean expiry multiple (what holding actually returned), by entry time/moneyness/right:')
print(df.groupby(['entry_time', 'moneyness', 'right']).expiry_mult.mean().sub(1).round(3).to_string())
atm = df[(df.moneyness == 'atm')]
both = atm.pivot_table(index=['date', 'entry_time'], columns='right', values='peak2x').dropna()
print(f"\nATM: sessions where the call AND the put both print >= 2x after entry: "
      f"{both.groupby('entry_time').apply(lambda x: (x.C.astype(bool) & x.P.astype(bool)).mean()).round(3).to_dict()}")
print(f"ATM: sessions where at least one side prints >= 2x: "
      f"{both.groupby('entry_time').apply(lambda x: (x.C.astype(bool) | x.P.astype(bool)).mean()).round(3).to_dict()}")
