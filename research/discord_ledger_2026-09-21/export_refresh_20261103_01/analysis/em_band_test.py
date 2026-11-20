"""Do the caller's published expected-move bands act as support/resistance? (daily bars, all tables)

For every OCR-parsed EM table: horizon from caption (daily / weekly / monthly; caption-less tables
classified by band width in ATR units), target window = next session(s) after the post, base = prior
close. Per row and per scale f in SCALES, the level center +/- f*half is tested on daily bars:
touch = window high >= upper (or low <= lower); rejection = touched and the window's final close is
back inside that level. If the published edge (f=1) is special, rejection there should exceed the
smooth trend through the neighbouring control scales. Rows are clustered by table (one market path),
so intervals are table-cluster bootstraps. Daily bars are unadjusted; SPX has no bars and is skipped.
Usage: python em_band_test.py <repo_root> <image_evidence.jsonl> <out_prefix>
"""
import json, sys
import numpy as np, pandas as pd

ROOT, IMG, OUT = sys.argv[1:4]
sys.path.insert(0, ROOT)
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402

SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
ET = 'America/New_York'
_bars = {}


def bars(sym):
    if sym not in _bars:
        try:
            d = pd.read_parquet(f'{ROOT}/Data/shared/bars/1d/{sym}.parquet')
        except FileNotFoundError:
            _bars[sym] = None; return None
        d['date'] = pd.to_datetime(d['timestamp']).dt.tz_convert(ET).dt.date
        d = d.set_index('date').sort_index()
        d['atr'] = (d['high'] - d['low']).rolling(14).mean().shift(1)
        _bars[sym] = d
    return _bars[sym]


def next_sessions(start, n=None, until=None):
    out, d = [], start
    while (n is None or len(out) < n) and (until is None or d <= until):
        if is_trading_day(d):
            out.append(d)
        d += pd.Timedelta(days=1)
    return out


def caption_kind(c):
    c = (c or '').lower()
    if 'week' in c:
        return 'weekly'
    if 'month' in c or 'october moves' in c or 'jan 2026' in c:
        return 'monthly'
    if any(w in c for w in ('daily', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', "daily’s", 'index move')):
        return 'daily'
    return None


tables = []
for r in map(json.loads, open(IMG)):
    if r['kind'] != 'em_table':
        continue
    posted = pd.Timestamp(r['posted_at_utc']).tz_convert(ET)
    day = posted.date()
    # Posted after the 09:30 open (or on a non-session day): the target is the next session.
    start = day if is_trading_day(day) and posted.time() < pd.Timestamp('09:30').time() else next_sessions(day + pd.Timedelta(days=1), n=1)[0]
    rows = {x['symbol']: (x['upper'], x['lower']) for x in r['em_rows'] if 'upper' in x and x['symbol'] != 'SPX'}
    tables.append({'id': r['attachment_id'], 'posted': posted, 'start': start, 'caption': r['caption'], 'kind': caption_kind(r['caption']), 'rows': rows})

recs = []
for t in tables:
    base = max(d for d in bars('SPY').index if d < t['start'])
    width_atr = []
    for sym, (u, l) in t['rows'].items():
        b = bars(sym)
        if b is not None and base in b.index and pd.notna(b.loc[base, 'atr']):
            width_atr.append((u - l) / 2 / b.loc[base, 'atr'])
    kind = t['kind'] or ('daily' if np.median(width_atr) < 1.2 else 'weekly' if np.median(width_atr) < 3 else 'monthly')
    if kind == 'daily':
        window = [t['start']]
    elif kind == 'weekly':
        friday = t['start'] + pd.Timedelta(days=(4 - t['start'].weekday()) % 7)
        window = next_sessions(t['start'], until=friday)
    else:
        eom = (pd.Timestamp(t['start']) + pd.offsets.MonthEnd(0)).date()
        window = next_sessions(t['start'], until=eom)
    for sym, (u, l) in t['rows'].items():
        b = bars(sym)
        if b is None or base not in b.index or any(d not in b.index for d in window):
            continue
        c, h = (u + l) / 2, (u - l) / 2
        w = b.loc[window]
        rec = {'table': t['id'], 'kind': kind, 'kind_from_caption': t['kind'] is not None, 'symbol': sym, 'start': str(t['start']),
               'center_vs_base_close_pct': 100 * (c / b.loc[base, 'close'] - 1), 'half_pct': 100 * h / c,
               'final_inside': l <= w['close'].iloc[-1] <= u}
        for f in SCALES:
            U, L = c + f * h, c - f * h
            tu, tl = w['high'].max() >= U, w['low'].min() <= L
            rec[f'touch_{f}'] = float(tu or tl)
            rec[f'rej_up_{f}'] = float(w['close'].iloc[-1] < U) if tu else np.nan
            rec[f'rej_dn_{f}'] = float(w['close'].iloc[-1] > L) if tl else np.nan
        recs.append(rec)

df = pd.DataFrame(recs)
df.to_csv(OUT + '_rows.csv', index=False)
print(f"{len(tables)} tables, {df.table.nunique()} usable, {len(df)} rows; kinds {df.groupby('kind').table.nunique().to_dict()} "
      f"(caption-labelled tables {df[df.kind_from_caption].table.nunique()})")
print('\ncenter vs prior close (%): median |diff| by kind', df.groupby('kind').center_vs_base_close_pct.apply(lambda x: round(x.abs().median(), 3)).to_dict(),
      '; share within 0.05%:', round((df.center_vs_base_close_pct.abs() <= 0.05).mean(), 3))
rng = np.random.default_rng(0)
tabs = df.table.unique()


def boot(fn, n=2000):
    vals = [fn(df[df.table.isin(rng.choice(tabs, len(tabs)))]) for _ in range(n)]
    return np.nanpercentile(vals, [2.5, 97.5])


def rej(d, f):
    x = pd.concat([d[f'rej_up_{f}'], d[f'rej_dn_{f}']]).astype(float).dropna()
    return x.mean() if len(x) else np.nan


for kind in ['all'] + sorted(df.kind.unique()):
    d = df if kind == 'all' else df[df.kind == kind]
    print(f'\n=== {kind}: {len(d)} rows / {d.table.nunique()} tables;  final close inside band {d.final_inside.mean():.1%} (1-sigma ~68%)')
    for f in SCALES:
        n_t = int(pd.concat([d[f"rej_up_{f}"], d[f"rej_dn_{f}"]]).notna().sum())
        print(f'  scale {f:<4}  touch {d[f"touch_{f}"].mean():6.1%}   rejection|touch {rej(d, f):6.1%}  (n touches {n_t})')
    neighbours = np.interp(1.0, [0.75, 1.25], [rej(d, 0.75), rej(d, 1.25)])
    print(f'  edge excess vs interpolated neighbours: {100 * (rej(d, 1.0) - neighbours):+.1f}pp')
lo, hi = boot(lambda d: rej(d, 1.0) - np.interp(1.0, [0.75, 1.25], [rej(d, 0.75), rej(d, 1.25)]))
print(f'\nall kinds: edge excess 95% table-cluster CI [{100 * lo:+.1f}, {100 * hi:+.1f}]pp; a CI half-width this size is roughly the detectable effect')
