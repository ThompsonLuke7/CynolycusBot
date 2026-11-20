"""Intraday reaction at expected-move edges: fade vs continuation (SIP 1-minute, underlying, research).

Pre-registered before the first run (2026-11-05):
  Event   first trade 1 tick through a band level during the session (09:30-15:59 ET); one event per
          name-session (the first level touched); sessions that open outside the level are skipped.
  Trade   fade: limit at the level (fills only when traded 1 tick through), stop 0.25h beyond the level
          (1 tick slippage), target = band center = prior close (needs 1 tick through), flat at the
          close (half tick). Conservative bar ordering: a stop in the entry bar counts, a target there
          does not, a bar touching both is a stop.
  Levels  center +/- f*h for f in (0.75, 1.0, 1.25). f = 1.0 is the published edge; 0.75 and 1.25 are
          control levels with the same stop width and target. A driftless path reaches the target
          before the stop with probability 0.25 / (0.25 + f): 25.0%, 20.0%, 16.7%.
  Reaction  move from the level to the close 30 and 60 minutes after the event, in h units, signed
          so that positive = reversal toward the center (negative = continuation).
  Cohorts A his published daily EM tables (caption says daily); B his weekly tables (first touch
          within the target week, traded that session); C proxy daily bands 2022-01..2026-10 with
          h = k * daily EWMA sigma * prior close, k = 0.966 fitted to 241 rows of his daily tables in
          rule_card_backtest.py (levels only, never outcomes). Published rows need a center within
          0.25% of our prior close so the session mapping is right.
  Primary C at f = 1.0: mean R after fills > 0, and above the interpolation of f = 0.75 / 1.25.
Inference clusters by session date (C) or table (A, B).
Usage: python em_edge_fade.py <repo_root> <bars_1m_dir> <image_evidence.jsonl> <out_dir> [k]
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, BARS, IMG, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
K = float(sys.argv[5]) if len(sys.argv) > 5 else 0.966
sys.path.insert(0, ROOT)
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402

ET, TICK, STOP_H = 'America/New_York', 0.01, 0.25
SCALES = (0.75, 1.0, 1.25)
ETFS = {'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'SMH', 'XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLC', 'XLRE'}
rng = np.random.default_rng(11)
OUT.mkdir(parents=True, exist_ok=True)

m = pd.concat([pd.read_parquet(f, columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close'])
               for f in sorted(glob.glob(f'{BARS}/*.parquet'))], ignore_index=True)
ts = m.pop('timestamp').dt.tz_convert(ET).dt.tz_localize(None)
m['date'] = ts.dt.normalize()
m['minute'] = ((ts - m['date']).dt.total_seconds() // 60 - 570).astype(int)
m = m[(m.minute >= 0) & (m.minute < 390)].sort_values(['symbol', 'date', 'minute'])
day = m.groupby(['symbol', 'date']).agg(open=('open', 'first'), close=('close', 'last'), first_minute=('minute', 'min'),
                                        n=('minute', 'size')).reset_index().sort_values(['symbol', 'date'])
g = day.groupby('symbol')
day['prev_close'] = g['close'].shift()
day['_lr'] = np.log(day['close'] / day['prev_close'])
day['sigma'] = day.groupby('symbol')['_lr'].transform(lambda s: np.sqrt((s ** 2).ewm(alpha=0.06, adjust=False).mean()).shift())
day = day[(day.first_minute == 0) & (day.n >= 200)]
arr = {k: (v['open'].to_numpy(), v['high'].to_numpy(), v['low'].to_numpy(), v['close'].to_numpy())
       for k, v in m.groupby(['symbol', 'date'])}
del m
print(f'{len(day):,} name-sessions, {day.symbol.nunique()} symbols, {day.date.min().date()}..{day.date.max().date()}')


def first(mask):
    if not mask.size:
        return None
    i = int(np.argmax(mask))
    return i if mask[i] else None


def fade(bars, center, half, f):
    """First touch of center +/- f*half in one session; returns the fade trade and reaction, or a skip reason."""
    o, h, l, c = bars
    up, dn = center + f * half, center - f * half
    if o[0] >= up or o[0] <= dn:
        return 'opened_outside'
    i_up, i_dn = first(h >= up + TICK), first(l <= dn - TICK)
    if i_up is None and i_dn is None:
        return 'no_touch'
    if i_up is not None and i_dn is not None and i_up == i_dn:
        return 'both_same_bar'
    short = i_dn is None or (i_up is not None and i_up < i_dn)
    e, level = (i_up, up) if short else (i_dn, dn)
    s = -1 if short else 1  # trade direction: short the upper edge, buy the lower edge
    # mirror into a long so one code path handles both sides
    O, H, L, C = (o, h, l, c) if s == 1 else (-o, -l, -h, -c)
    entry, stop, target = s * level, s * level - STOP_H * half, s * center
    i_stop = first(L[e:] <= stop)
    i_stop = None if i_stop is None else i_stop + e
    i_tgt = first(H[e + 1:] >= target + TICK)
    i_tgt = None if i_tgt is None else i_tgt + e + 1
    if i_stop is not None and (i_tgt is None or i_stop <= i_tgt):
        px = O[i_stop] if i_stop > e and O[i_stop] < stop else stop
        pnl, tag = px - TICK - entry, 'stop'
    elif i_tgt is not None:
        pnl, tag = (max(target, O[i_tgt]) - entry), 'target'
    else:
        pnl, tag = C[-1] - TICK / 2 - entry, 'eod'
    r = STOP_H * half
    fwd = {f'rev{n}': (C[min(e + n, len(C) - 1)] - entry) / half for n in (30, 60)}
    return {'minute': e, 'side': 'fade_upper' if short else 'fade_lower', 'exit': tag, 'R_mult': pnl / r,
            'bps': 1e4 * pnl / abs(entry), **fwd}


def cluster_ci(df, col, cluster):
    sums = df.groupby(cluster)[col].sum(); counts = df.groupby(cluster)[col].size()
    s, n = sums.to_numpy(), counts.to_numpy()
    if len(s) < 3:
        return np.nan, np.nan
    b = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(1000))]
    return tuple(np.percentile(b, [2.5, 97.5]))


def report(rows, label, cluster):
    df = pd.DataFrame(rows)
    if df.empty or 'R_mult' not in df:
        print(f'{label}: no events'); return df
    print(f'\n=== {label}')
    out = []
    for f in SCALES:
        d = df[df.f == f]
        lo, hi = cluster_ci(d, 'R_mult', cluster)
        rlo, rhi = cluster_ci(d, 'rev30', cluster)
        hit = (d.exit == 'target').mean(); drift = STOP_H / (STOP_H + f)
        out.append({'f': f, 'events': len(d), 'clusters': d[cluster].nunique(), 'mean_R': d.R_mult.mean(), 'R_lo': lo, 'R_hi': hi,
                    'target_first': hit, 'driftless': drift, 'stop': (d.exit == 'stop').mean(), 'rev30_h': d.rev30.mean(),
                    'rev30_lo': rlo, 'rev30_hi': rhi, 'rev60_h': d.rev60.mean(), 'bps': d.bps.mean()})
        o = out[-1]
        print(f"  f={f:<4} events {o['events']:>6} ({o['clusters']} clusters)  fade {o['mean_R']:+.3f}R [{lo:+.3f}, {hi:+.3f}] {o['bps']:+.1f} bps  "
              f"target-first {hit:.1%} vs driftless {drift:.1%}  reversal 30m {o['rev30_h']:+.3f}h [{rlo:+.3f}, {rhi:+.3f}], 60m {o['rev60_h']:+.3f}h")
    o = pd.DataFrame(out).set_index('f')
    interp = (o.loc[0.75, 'mean_R'] + o.loc[1.25, 'mean_R']) / 2
    print(f"  edge vs neighbours: fade R at f=1.0 minus mean of f=0.75/1.25 = {o.loc[1.0, 'mean_R'] - interp:+.3f}R; "
          f"30m reversal excess {o.loc[1.0, 'rev30_h'] - (o.loc[0.75, 'rev30_h'] + o.loc[1.25, 'rev30_h']) / 2:+.3f}h")
    return df


# ---------------- C: proxy bands, full history ----------------
rows, skips = [], {}
for r in day.dropna(subset=['prev_close', 'sigma']).itertuples():
    half = K * r.sigma * r.prev_close
    for f in SCALES:
        res = fade(arr[(r.symbol, r.date)], r.prev_close, half, f)
        if isinstance(res, str):
            skips[(f, res)] = skips.get((f, res), 0) + 1; continue
        rows.append({'symbol': r.symbol, 'date': r.date, 'f': f, **res})
print('skips (f, reason):', {k: v for k, v in sorted(skips.items())})
c = report(rows, 'C: proxy daily bands, 33 names, 2022-01..2026-10 (cluster = session date)', 'date')
c.to_csv(OUT / 'fade_proxy_events.csv', index=False)
p = c[c.f == 1.0].copy()
p['year'] = p.date.dt.year
print('\n  f=1.0 breakdowns (mean R, n):')
for name, grp in [('year', p.groupby('year')), ('side', p.groupby('side'))]:
    print('   ', name, {k: (round(v.R_mult.mean(), 3), len(v)) for k, v in grp})
print('    ETF', round(p[p.symbol.isin(ETFS)].R_mult.mean(), 3), ' stocks', round(p[~p.symbol.isin(ETFS)].R_mult.mean(), 3))
early = p.minute < 30
print(f"    touch in first 30 min {p[early].R_mult.mean():+.3f}R (n={early.sum()}), later {p[~early].R_mult.mean():+.3f}R (n={(~early).sum()})")
se = (cluster_ci(p, 'R_mult', 'date')[1] - cluster_ci(p, 'R_mult', 'date')[0]) / (2 * 1.96)
print(f"    MDE (80% power) ~ {2.8 * se:.3f}R per event")

# ---------------- A/B: his published bands ----------------


def caption_kind(cap):
    cap = (cap or '').lower()
    if 'week' in cap:
        return 'weekly'
    if 'month' in cap or 'october moves' in cap or 'jan 2026' in cap:
        return 'monthly'
    if any(w in cap for w in ('daily', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'index move')):
        return 'daily'
    return None


dix = day.set_index(['symbol', 'date'])
pub = {'daily': [], 'weekly': []}
for t in map(json.loads, open(IMG)):
    kind = caption_kind(t['caption']) if t['kind'] == 'em_table' else None
    if kind not in pub:
        continue
    posted = pd.Timestamp(t['posted_at_utc']).tz_convert(ET)
    d0 = posted.date()
    if not (is_trading_day(d0) and posted.time() < pd.Timestamp('09:30').time()):
        d0 += pd.Timedelta(days=1)
        while not is_trading_day(d0):
            d0 += pd.Timedelta(days=1)
    start = pd.Timestamp(d0)
    sessions = [start] if kind == 'daily' else [s for s in pd.date_range(start, start + pd.Timedelta(days=(4 - start.weekday()) % 7))
                                               if is_trading_day(s.date())]
    for x in t['em_rows']:
        if 'upper' not in x or (x['symbol'], start) not in dix.index:
            continue
        center, half = (x['upper'] + x['lower']) / 2, (x['upper'] - x['lower']) / 2
        if abs(center / dix.loc[(x['symbol'], start), 'prev_close'] - 1) > 0.0025:
            continue
        for f in SCALES:
            for s in sessions:
                if (x['symbol'], s) not in arr:
                    break
                res = fade(arr[(x['symbol'], s)], center, half, f)
                if res == 'no_touch':
                    continue
                if isinstance(res, dict):
                    pub[kind].append({'table': t['attachment_id'], 'symbol': x['symbol'], 'date': s, 'f': f, **res})
                break  # first touch (or an outside open) ends the search for this level
for kind, rows in pub.items():
    df = report(rows, f"{'A' if kind == 'daily' else 'B'}: his published {kind} bands (cluster = table)", 'table')
    df.to_csv(OUT / f'fade_published_{kind}_events.csv', index=False)
