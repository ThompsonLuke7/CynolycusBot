"""Backtest the caller's published breakout card on SIP 1-minute bars (underlying only, research).

Card (10/23 post, geometry verified on all six rows): calls trigger on a break of the prior-day high,
puts on the prior-day low; T2 = expected-move edge (prior close +/- h); stop = trigger -/+ 0.25h (R);
T1 = trigger +/- 2R (1R when 2R would reach T2). Rules: only enter on a break in the first 30 minutes,
skip if the open gaps past the trigger, half off at T1, trail the rest (stop to entry) toward T2, flat
at the close. Candidates ("scanner picks") = prior close within 0.5h of the trigger with >= 1R of room.

h has no history before our IV captures (2026-09), so it is a realised-volatility proxy scaled to the
caller's own published daily EM tables (scale fitted on levels he posted, never on outcomes; +/-25%
sensitivity reported). Everything at decision time uses data through the prior close plus today's open.

Fills: buy-stop at max(trigger, bar open) + 1 tick; stops fill 1 tick through (or at a gapped open);
targets need a trade 1 tick through; half-tick on the closing exit. Conservative bar ordering: a stop
in the entry bar counts, targets in the entry bar do not, and a bar touching stop and target is a stop
(the optimistic bound is reported too). Inference clusters by date (one market path per day).
Controls: (1) a driftless price path makes the -1R stop come before a +2R T1 two times in three, so a
trigger with no information shows ~1/3 T1-first; (2) every gap-filtered candidate side, triggered or not,
entered at a random minute of the window with the same R/T1/T2 distances (no conditioning on the break;
a control drawn only from triggered days would condition on the future and is not used).
Usage: python rule_card_backtest.py <repo_root> <bars_1m_dir> <image_evidence.jsonl> <out_dir>
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, BARS, IMG, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, ROOT)
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402

ET, TICK = 'America/New_York', 0.01
ETFS = {'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'SMH', 'XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLC', 'XLRE'}
rng = np.random.default_rng(7)

# ---------------- bars ----------------
cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close']
m = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f'{BARS}/*.parquet'))], ignore_index=True)
ts = m.pop('timestamp').dt.tz_convert(ET).dt.tz_localize(None)
m['date'] = ts.dt.normalize()
m['minute'] = ((ts - m['date']).dt.total_seconds() // 60 - 570).astype(int)
m = m[(m.minute >= 0) & (m.minute < 390)].sort_values(['symbol', 'date', 'minute'])
day = m.groupby(['symbol', 'date']).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'),
                                        first_minute=('minute', 'min'), n=('minute', 'size')).reset_index()
day = day.sort_values(['symbol', 'date'])
g = day.groupby('symbol')
day['prev_close'], day['prev_high'], day['prev_low'] = g['close'].shift(), g['high'].shift(), g['low'].shift()
lr = np.log(day['close'] / day['prev_close'])
tr = pd.concat([day['high'] - day['low'], (day['high'] - day['prev_close']).abs(), (day['low'] - day['prev_close']).abs()], axis=1).max(axis=1)
day['_lr'], day['_tr'] = lr, tr
g = day.groupby('symbol')
# proxies known at the prior close (shifted one session), in dollars at the prior close
day['rv20'] = g['_lr'].transform(lambda s: s.rolling(20).std().shift()) * day['prev_close']
day['ewma'] = g['_lr'].transform(lambda s: np.sqrt((s ** 2).ewm(alpha=0.06, adjust=False).mean()).shift()) * day['prev_close']
day['atr14'] = g['_tr'].transform(lambda s: s.rolling(14).mean().shift())
print(f'bars: {len(m):,} RTH minutes, {day.symbol.nunique()} symbols, {day.date.min().date()}..{day.date.max().date()}')
bad = day[(day.first_minute > 0) | (day.n < 300)]
print(f'sessions with a late first bar or < 300 minutes (excluded): {len(bad)} of {len(day)} (half-days included in this count)')
day = day[(day.first_minute == 0) & (day.n >= 200)]

# ---------------- calibrate h on the caller's daily EM tables ----------------


def caption_kind(c):
    c = (c or '').lower()
    if 'week' in c:
        return 'weekly'
    if 'month' in c or 'october moves' in c or 'jan 2026' in c:
        return 'monthly'
    if any(w in c for w in ('daily', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'index move')):
        return 'daily'
    return None


cal = []
dix = day.set_index(['symbol', 'date'])
for r in map(json.loads, open(IMG)):
    if r['kind'] != 'em_table':
        continue
    posted = pd.Timestamp(r['posted_at_utc']).tz_convert(ET)
    d = posted.date()
    if not (is_trading_day(d) and posted.time() < pd.Timestamp('09:30').time()):
        d += pd.Timedelta(days=1)
        while not is_trading_day(d):
            d += pd.Timedelta(days=1)
    for x in r['em_rows']:
        key = (x['symbol'], pd.Timestamp(d))
        if 'upper' not in x or key not in dix.index:
            continue
        row = dix.loc[key]
        half, center = (x['upper'] - x['lower']) / 2, (x['upper'] + x['lower']) / 2
        cal.append({'table': r['attachment_id'], 'kind': caption_kind(r['caption']), 'symbol': x['symbol'], 'date': d, 'half': half,
                    'center_err_pct': 100 * (center / row['prev_close'] - 1),
                    **{p: half / row[p] for p in ('rv20', 'ewma', 'atr14')}})
cal = pd.DataFrame(cal)
dcal = cal[(cal.kind == 'daily') & (cal.center_err_pct.abs() < 0.25)]
print(f"\nEM calibration: {len(dcal)} rows from {dcal.table.nunique()} daily-captioned tables (center within 0.25% of our prior close)")
disp = {p: float((np.log(dcal[p]) - np.log(dcal[p]).median()).abs().median()) for p in ('rv20', 'ewma', 'atr14')}
PROXY = min(disp, key=disp.get)
K = float(dcal[PROXY].median())
print('  median |log(published / proxy) - median| by proxy:', {k: round(v, 3) for k, v in disp.items()}, '-> using', PROXY)
print(f"  scale k = {K:.3f} (published half-width = k x {PROXY}); per-table medians range "
      f"{dcal.groupby('table')[PROXY].median().min():.2f}..{dcal.groupby('table')[PROXY].median().max():.2f}")
cal.to_csv(OUT / 'em_calibration_rows.csv', index=False)

# ---------------- minute arrays ----------------
arr = {k: (v['open'].to_numpy(), v['high'].to_numpy(), v['low'].to_numpy(), v['close'].to_numpy(), v['minute'].to_numpy())
       for k, v in m.groupby(['symbol', 'date'])}
del m


def first(mask):
    if not mask.size:
        return None
    i = int(np.argmax(mask))
    return i if mask[i] else None


def simulate(o, h, l, c, e, side, entry_px, stop, t1, t2, optimistic=False, reclaim_stop=None):
    """Return price P&L per share (long-positive for calls, short-positive for puts) for half/half exits."""
    s = 1 if side == 'call' else -1
    O, H, L, C = (o, h, l, c) if s == 1 else (-o, -l, -h, -c)  # mirror puts into longs
    entry, stop_, t1_, t2_ = s * entry_px, s * stop, s * t1, s * t2
    n = len(O)
    st0 = e + 1 if optimistic else e
    stop_hits = L[st0:] <= stop_
    if reclaim_stop is not None:  # cut on the first 1-minute close back through the trigger
        stop_hits = stop_hits | (C[st0:] < s * reclaim_stop)
    i_stop = first(stop_hits)
    i_t1 = first(H[e + (0 if optimistic else 1):] >= t1_ + TICK)
    i_stop = None if i_stop is None else i_stop + st0
    i_t1 = None if i_t1 is None else i_t1 + e + (0 if optimistic else 1)
    if i_stop is not None and (i_t1 is None or i_stop <= i_t1):
        px = stop_ - TICK if L[i_stop] <= stop_ else C[i_stop] - TICK / 2
        if i_stop > e and O[i_stop] < stop_:
            px = O[i_stop] - TICK
        return px - entry, 'stop'
    if i_t1 is None:
        return C[n - 1] - TICK / 2 - entry, 'eod'
    leg1 = max(t1_, O[i_t1]) - entry if i_t1 > e else t1_ - entry
    rest_l, rest_h = L[i_t1 + (1 if optimistic else 0):], H[i_t1:]
    i_be = first(rest_l <= entry)
    i_be = None if i_be is None else i_be + i_t1 + (1 if optimistic else 0)
    i_t2 = first(rest_h >= t2_ + TICK)
    i_t2 = None if i_t2 is None else i_t2 + i_t1
    if i_be is not None and (i_t2 is None or i_be <= i_t2):
        leg2, tag = min(entry, O[i_be]) - TICK - entry if i_be > i_t1 else -TICK, 't1_be'
    elif i_t2 is not None:
        leg2, tag = max(t2_, O[i_t2]) - entry if i_t2 > i_t1 else t2_ - entry, 't2'
    else:
        leg2, tag = C[n - 1] - TICK / 2 - entry, 't1_eod'
    return 0.5 * leg1 + 0.5 * leg2, tag


def build(day, k_scale=1.0, prox=0.5, window=30):
    rows = []
    for r in day.dropna(subset=['prev_close', PROXY]).itertuples():
        hh = K * k_scale * getattr(r, PROXY)
        if not hh > 0:
            continue
        R = 0.25 * hh
        for side, trig, t2 in (('call', r.prev_high, r.prev_close + hh), ('put', r.prev_low, r.prev_close - hh)):
            s = 1 if side == 'call' else -1
            room = s * (t2 - trig)
            if room < R:
                continue
            t1 = trig + s * 2 * R if room > 2 * R + TICK else trig + s * R
            dist = s * (trig - r.prev_close) / hh
            rows.append({'symbol': r.symbol, 'date': r.date, 'side': side, 'trigger': trig, 'stop': trig - s * R, 't1': t1, 't2': t2,
                         'R': R, 'h': hh, 'dist_h': dist, 'open': r.open, 'prev_close': r.prev_close,
                         'gap_skip': s * (r.open - trig) >= 0, 'candidate': 0 <= dist <= prox})
    return pd.DataFrame(rows)


def run(cands, window=30, optimistic=False, reclaim=False, extra_cost_bps=0.0):
    out = []
    for (sym, d), grp in cands.groupby(['symbol', 'date']):
        o, h, l, c, mi = arr[(sym, d)]
        win = mi < window
        trig_at = {}
        for r in grp.itertuples():
            s = 1 if r.side == 'call' else -1
            hit = first(((h if s == 1 else -l) >= s * r.trigger) & win)
            if hit is not None:
                trig_at[r.side] = hit
        if not trig_at:
            continue
        # one trade per name-day: the first side to trigger (both in the same bar = ambiguous, skipped)
        side = min(trig_at, key=trig_at.get)
        if len(trig_at) == 2 and trig_at['call'] == trig_at['put']:
            continue
        r = grp[grp.side == side].iloc[0]; e = trig_at[side]; s = 1 if side == 'call' else -1
        entry_px = (max(r.trigger, o[e]) if s == 1 else min(r.trigger, o[e])) + s * TICK
        pnl, tag = simulate(o, h, l, c, e, side, entry_px, r.stop, r.t1, r.t2, optimistic, r.trigger if reclaim else None)
        cost = extra_cost_bps * 1e-4 * entry_px * 2
        rec = {'symbol': sym, 'date': d, 'side': side, 'entry_minute': int(mi[e]), 'entry': entry_px, 'R': r.R, 'h': r.h,
               'trigger': r.trigger, 'stop': r.stop, 't1': r.t1, 't2': r.t2, 't1_R': abs(r.t1 - r.trigger) / r.R,
               'dist_h': r.dist_h, 'exit': tag, 'pnl': pnl - cost, 'R_mult': (pnl - cost) / r.R, 'bps': 1e4 * (pnl - cost) / entry_px}
        out.append(rec)
    return pd.DataFrame(out)


def run_control(cands, window=30, draws=5):
    """Random-minute entries on every candidate side (triggered or not), same bracket distances."""
    out = []
    for r in cands.itertuples():
        o, h, l, c, mi = arr[(r.symbol, r.date)]
        win_idx = np.flatnonzero(mi < window)
        s = 1 if r.side == 'call' else -1
        vals = []
        for _ in range(draws):
            j = int(rng.choice(win_idx)); ep = o[j] + s * TICK; off = ep - (r.trigger + s * TICK)
            pnl, _ = simulate(o, h, l, c, j, r.side, ep, r.stop + off, r.t1 + off, r.t2 + off)
            vals.append(pnl / r.R)
        out.append({'symbol': r.symbol, 'date': r.date, 'side': r.side, 'R_mult': float(np.mean(vals)), 'bps': np.nan})
    return pd.DataFrame(out)


def summarize(t, label, col='R_mult'):
    if t.empty:
        print(f'{label:<46} no trades'); return {}
    by_day = t.groupby('date')[col].sum()
    days = by_day.index.to_numpy(); n_days = len(days)
    per_day_n = t.groupby('date').size().reindex(days).to_numpy(); per_day_sum = by_day.to_numpy()
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n_days, n_days)
        boots.append(per_day_sum[idx].sum() / per_day_n[idx].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = {'label': label, 'trades': len(t), 'days': n_days, 'mean': t[col].mean(), 'ci_lo': lo, 'ci_hi': hi, 'median': t[col].median(),
           'win': (t[col] > 0).mean(), 'mean_bps': t['bps'].mean() if col == 'R_mult' else np.nan}
    print(f"{label:<46} n={len(t):>5} days={n_days:>4}  mean {res['mean']:+.3f}R [{lo:+.3f}, {hi:+.3f}]  median {res['median']:+.2f}R  "
          f"win {res['win']:.0%}" + (f"  {res['mean_bps']:+.1f} bps" if col == 'R_mult' else ''))
    return res


OUT.mkdir(parents=True, exist_ok=True)
base = build(day)
cand = base[base.candidate & ~base.gap_skip]
print(f"\ncandidate sides: {len(base):,} with >=1R room; within 0.5h of trigger {int(base.candidate.sum()):,}; "
      f"after gap-skip {len(cand):,}; gap-skipped {int((base.candidate & base.gap_skip).sum()):,}")

print('\n=== PRIMARY: 0.5h proximity, first 30 min, conservative fills, card geometry')
prim = run(cand)
prim.to_csv(OUT / 'trades_primary.csv', index=False)
res = [summarize(prim, 'primary (rule)')]
ctl = run_control(cand)
res.append(summarize(ctl, 'control: random minute, all candidate sides'))
TICK = 0.0
res.append(summarize(run(cand), 'primary, frictionless (no tick slippage)'))
TICK = 0.01
print('exit mix:', prim.exit.value_counts(normalize=True).round(3).to_dict())
two_r = prim[(prim.t1_R > 1.5)] if 't1_R' in prim else prim
resolved = two_r[two_r.exit != 'eod']
t1_first = resolved.exit.isin(['t2', 't1_be', 't1_eod']).mean()
print(f"T1 (2R) reached before the 1R stop on {t1_first:.1%} of resolved trades; a driftless path gives 33.3%")
print('sample trades (for hand-checking against the bars):')
print(prim.sample(6, random_state=1)[['symbol', 'date', 'side', 'entry_minute', 'entry', 'R', 'exit', 'R_mult']].to_string(index=False))
print(f"1R = {(prim.R / prim.entry * 1e4).median():.0f} bps of price (median); h = {(prim.h / prim.entry * 1e4).median():.0f} bps")

print('\n=== breakdowns (primary)')
prim['year'] = prim.date.dt.year
for y, t in prim.groupby('year'):
    summarize(t, f'  {y}')
for s, t in prim.groupby('side'):
    summarize(t, f'  {s}s')
summarize(prim[prim.symbol.isin(ETFS)], '  ETFs')
summarize(prim[~prim.symbol.isin(ETFS)], '  single stocks')
for lo_, hi_ in ((0, 0.2), (0.2, 0.35), (0.35, 0.5)):
    summarize(prim[(prim.dist_h >= lo_) & (prim.dist_h < hi_)], f'  prior close {lo_:.2f}-{hi_:.2f}h from trigger')

print('\n=== variants')
res.append(summarize(run(cand, optimistic=True), 'optimistic bar ordering'))
res.append(summarize(run(cand, window=15), 'first 15 min only'))
res.append(summarize(run(cand, window=390), 'any time of day (no 30-min rule)'))
res.append(summarize(run(cand, reclaim=True), 'stop = first 1-min close back through trigger'))
res.append(summarize(run(cand, extra_cost_bps=2), 'plus 2 bps/side extra cost'))
top3 = cand.sort_values('dist_h').groupby(['date', 'side']).head(3)
res.append(summarize(run(top3), 'top 3 calls + top 3 puts per day (closest)'))
alls = base[~base.gap_skip]
res.append(summarize(run(alls), 'no proximity filter (every name-day with room)'))
for ks in (0.75, 1.25):
    b2 = build(day, k_scale=ks)
    res.append(summarize(run(b2[b2.candidate & ~b2.gap_skip]), f'EM scale x{ks}'))
pd.DataFrame(res).to_csv(OUT / 'summary.csv', index=False)

se = (res[0]['ci_hi'] - res[0]['ci_lo']) / (2 * 1.96)
print(f"\nMDE (80% power, two-sided 5%) on the primary mean ~ {2.8 * se:.3f}R per trade (day-clustered SE {se:.3f}R)")

print('\n=== the 10/23 card itself (2026-10-23, rows in our universe; the card used option-implied h, here the proxy h is shown)')
card = {'XLF': ('call', 56.70, 56.59, 56.93, 57.10, 0.45), 'XLV': ('call', 182.25, 181.76, 182.74, 183.24, 1.94),
        'QQQ': ('put', 701.85, 704.38, 696.78, 694.34, 10.13), 'XLK': ('put', 181.50, 182.34, 179.83, 179.00, 3.34)}
d = pd.Timestamp('2026-10-23')
for sym, (side, trig, stop, t1, t2, card_h) in card.items():
    if (sym, d) not in arr:
        continue
    o, h, l, c, mi = arr[(sym, d)]
    s = 1 if side == 'call' else -1
    hit = first(((h if s == 1 else -l) >= s * trig) & (mi < 30))
    gap = s * (o[0] - trig) >= 0
    proxy_h = base[(base.symbol == sym) & (base.date == d)].h
    if hit is None or gap:
        print(f"  {sym} {side}: open {o[0]:.2f}  {'gapped past trigger (skip)' if gap else 'no trigger in first 30 min'}; day H/L {h.max():.2f}/{l.min():.2f}")
        continue
    ep = trig + s * TICK
    pnl, tag = simulate(o, h, l, c, hit, side, ep, stop, t1, t2)
    print(f"  {sym} {side}: triggered {9 + (30 + int(mi[hit])) // 60}:{(30 + int(mi[hit])) % 60:02d} ET, exit {tag}, {pnl / abs(trig - stop):+.2f}R "
          f"(card h {card_h:.2f}; our proxy h {proxy_h.iloc[0] if len(proxy_h) else float('nan'):.2f})")
