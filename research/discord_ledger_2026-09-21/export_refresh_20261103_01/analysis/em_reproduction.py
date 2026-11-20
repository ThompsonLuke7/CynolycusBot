"""Test candidate expected-move formulas against every ACE EM table with a same-date IV capture.

Input: em_tables.json (OCR-parsed, visually checked tables with base close date and horizon).
For each published row: center = (upper+lower)/2, half = (upper-lower)/2, and each candidate
model half-width from our Schwab capture of the base date's close. A correct formula family gives
a near-constant observed/model ratio across names and dates (dispersion is the test statistic).
Usage: python em_reproduction.py <repo_root> <em_tables.json>
"""
import json, sys
from datetime import date
import numpy as np, pandas as pd

ROOT, TABLES = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402


def sessions_between(a: str, b: str) -> int:
    """Trading sessions in (a, b]."""
    return sum(is_trading_day(d.date()) for d in pd.date_range(a, b)[1:])


def atm(chain: pd.DataFrame, expiry: str, spot: float) -> tuple[float, float] | None:
    e = chain[chain.expiration == expiry]
    if e.empty:
        return None
    k = e.strike.iloc[(e.strike - spot).abs().argsort()].iloc[0]
    a = e[e.strike == k]
    if set(a.option_type.str.upper().str[0]) != {'C', 'P'}:
        return None
    return a.volatility_pct.mean() / 100, a.mark.sum()


rows = []
for t in json.load(open(TABLES)):
    cap = f"{ROOT}/Data/dealer_positioning/iv_surface/{t['base_date'].replace('-', '')}"
    try:
        chain = pd.read_parquet(f'{cap}/iv_contracts.parquet')
        feats = pd.read_parquet(f'{cap}/iv_surface_features.parquet').set_index('symbol')
    except FileNotFoundError:
        continue
    chain = chain[~chain.non_standard.astype(bool)].copy()
    chain['expiration'] = pd.to_datetime(chain['expiration']).dt.date.astype(str)
    cal = (date.fromisoformat(t['target_end']) - date.fromisoformat(t['base_date'])).days
    trd = sessions_between(t['base_date'], t['target_end'])
    for sym, (up, dn) in t['rows'].items():
        s = chain[chain.symbol == sym]
        if s.empty:
            continue
        spot = float(s.spot.iloc[0]); c = (up + dn) / 2; h = (up - dn) / 2
        exps = sorted(e for e in s.expiration.unique() if e >= t['target_end'])
        near = atm(s, exps[0], spot) if exps else None
        r = {'table': t['label'], 'kind': t['kind'], 'symbol': sym, 'center_vs_spot_pct': 100 * (c / spot - 1),
             'half_pct': 100 * h / c, 'near_expiry': exps[0] if exps else None, 'cal_days': cal, 'trd_days': trd}
        if near:
            iv, strad = near
            r.update({'iv_near': iv, 'r_near_cal': h / (c * iv * np.sqrt(cal / 365)), 'r_near_trd': h / (c * iv * np.sqrt(trd / 252)),
                      'r_near_1cal': h / (c * iv * np.sqrt(1 / 365)), 'r_straddle_near': h / strad})
        if sym in feats.index and pd.notna(feats.loc[sym, 'iv_atm_30']):
            iv30 = float(feats.loc[sym, 'iv_atm_30'])
            r.update({'iv30': iv30, 'r_iv30_cal': h / (c * iv30 * np.sqrt(cal / 365)), 'r_iv30_trd': h / (c * iv30 * np.sqrt(trd / 252)),
                      'r_iv30_1cal': h / (c * iv30 * np.sqrt(1 / 365))})
        rows.append(r)

df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.precision', 3); pd.set_option('display.max_rows', 200)
print(df[['table', 'symbol', 'center_vs_spot_pct', 'half_pct', 'near_expiry', 'cal_days', 'trd_days', 'r_near_cal', 'r_near_trd', 'r_iv30_cal', 'r_straddle_near']].to_string(index=False))
cols = [c for c in df.columns if c.startswith('r_')]
for kind, g in df.groupby('kind'):
    print(f'\n=== {kind}: {len(g)} rows from {g.table.nunique()} tables; |center-spot| median {g.center_vs_spot_pct.abs().median():.3f}%')
    for c in cols:
        x = g[c].dropna()
        if len(x):
            print(f'  {c:<16} n={len(x):3d} median {x.median():.3f}  IQR {x.quantile(.25):.3f}-{x.quantile(.75):.3f}  CV {x.std() / x.mean():.3f}'
                  + '  per-table medians ' + ', '.join(f'{k}:{v:.3f}' for k, v in g.groupby('table')[c].median().items()))
df.to_csv(TABLES.replace('.json', '_reproduction.csv'), index=False)
