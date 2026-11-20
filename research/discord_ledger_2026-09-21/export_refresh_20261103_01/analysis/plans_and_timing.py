"""Score the caller's published underlying plans and cycle-timing calls on daily bars (descriptive, small n).

Plans (ace_plans_and_timing_*.json): fill per the entry kind within 5 sessions of the first actionable
session (posted before 09:30 ET on a session day = that day, else the next). After the fill, targets and
the fail level are checked on the following sessions (the fill session too for next_open fills); a
session touching both is 'ambiguous' and counted against the plan. Each T1 gets a base rate from the
same name's prior 250 sessions: P(max high over the next 10 sessions >= close * (1 + T1 distance)).
Fib test: targets vs extension/retracement levels of trailing N-session (low, high) for N in 5..150,
with a jittered-target null. Timing: a call hits if a pivot (lowest low / highest high within +/-k
sessions) falls in its window; base rate = share of same-length windows over the symbol's prior 500
sessions that contain such a pivot. Daily bars are Data/shared/bars/1d (unadjusted; split breaks cut
the history). Momentum: 63-session return percentile among the 1,000 highest 20-day median
dollar-volume names on the last session before the post.
Usage: python plans_and_timing.py <repo_root> <plans.json> <out_prefix>
"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT, PLANS, OUT = sys.argv[1:4]
sys.path.insert(0, ROOT)
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402

ET = 'America/New_York'
FIBS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.272, 1.414, 1.618, 2.0, 2.272, 2.414, 2.618]
spec = json.load(open(PLANS))
_bars = {}


def bars(sym):
    if sym not in _bars:
        d = pd.read_parquet(f'{ROOT}/Data/shared/bars/1d/{sym}.parquet')
        d['date'] = pd.to_datetime(d['timestamp']).dt.tz_convert(ET).dt.date
        d = d.drop_duplicates('date').set_index('date').sort_index()[['open', 'high', 'low', 'close', 'volume']]
        jump = (d['close'] / d['close'].shift() - 1).abs() > 0.45  # split/reverse-split break in unadjusted bars
        if jump.any():
            d = d.loc[d.index[jump.to_numpy()][-1]:]
        c = d['close']
        d['sma5'], d['sma34'], d['sma55'] = c.rolling(5).mean(), c.rolling(34).mean(), c.rolling(55).mean()
        d['ema21'] = c.ewm(span=21, adjust=False).mean()
        _bars[sym] = d
    return _bars[sym]


def first_session(posted):
    t = pd.Timestamp(posted, tz='UTC').tz_convert(ET)
    d = t.date()
    if is_trading_day(d) and t.time() < pd.Timestamp('09:30').time():
        return d, False
    return d, True  # True: the post day itself is not actionable from the open


def sessions_from(b, d, after):
    idx = b.index
    pos = idx.searchsorted(d, side='right' if after else 'left')
    return pos


def fill(plan, b):
    e = plan['entry']; d, after = first_session(plan['posted_utc'])
    if e['kind'] == 'hit':
        pos = idx_pos = b.index.searchsorted(d)
        level = b['ema21'].iloc[idx_pos - 1] if e['level'] == 'ema21' else e['level']
        return pos, float(level), 'hit', False
    pos0 = sessions_from(b, d, after)
    prior_close = b['close'].iloc[pos0 - 1]
    kind = e['kind']
    if kind == 'auto':
        kind = 'zone' if e['level'] < prior_close else 'above'
        e = {'kind': kind, 'lo': e['level'], 'hi': e['level'], 'level': e['level']}
    for pos in range(pos0, min(pos0 + 5, len(b))):
        o, h, l, c = b[['open', 'high', 'low', 'close']].iloc[pos]
        if kind == 'next_open':
            return pos, float(o), kind, True
        if kind == 'zone' and l <= e['hi'] and h >= e['lo']:
            return pos, float(min(o, e['hi']) if o >= e['lo'] else o), kind, False
        if kind == 'above' and h >= e['level']:
            return pos, float(max(o, e['level'])), kind, False
        if kind == 'close_above' and c > e['level']:
            return pos, float(c), kind, False
    return None, None, kind, False


def base_rate_hit(b, end_pos, dist, horizon=10, lookback=250):
    c, h = b['close'].to_numpy(), b['high'].to_numpy()
    lo = max(0, end_pos - lookback)
    hits = [h[i + 1:i + 1 + horizon].max() >= c[i] * (1 + dist) for i in range(lo, end_pos - horizon)]
    return float(np.mean(hits)) if hits else np.nan


def poisson_binomial_p(ps, k):
    """P(X >= k) for X = sum of independent Bernoulli(ps)."""
    dist = np.zeros(len(ps) + 1); dist[0] = 1
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p; dist[0] *= (1 - p)
    return float(dist[k:].sum())


# ---------------- plans ----------------
rows = []
for p in spec['plans']:
    b = bars(p['symbol'])
    pos, px, kind, include_fill_day = fill(p, b)
    post_pos = b.index.searchsorted(first_session(p['posted_utc'])[0])
    r = {'id': p['id'], 'symbol': p['symbol'], 'posted': p['posted_utc'][:10], 'kind': kind, 'filled': pos is not None}
    prior = b.iloc[post_pos - 1]
    r['prior_close'] = round(prior['close'], 2)
    if pos is None:
        rows.append(r); continue
    r.update({'fill_date': str(b.index[pos]), 'fill': round(px, 2)})
    ma = {k: prior[k] for k in ('sma5', 'ema21', 'sma34', 'sma55')}
    near_k = min(ma, key=lambda k: abs(px / ma[k] - 1))
    r['nearest_ma'], r['nearest_ma_dist_pct'] = near_k, round(100 * (px / ma[near_k] - 1), 2)
    r['entry_vs_prior_close_pct'] = round(100 * (px / prior['close'] - 1), 2)
    path = b.iloc[pos if include_fill_day else pos + 1:pos + 11]
    r['sessions_seen'] = len(path)
    t1 = p['targets'][0] if p['targets'] else None
    fail = p.get('fail')
    outcome = 'open'
    for date, bar in path.iterrows():
        hit_t = t1 is not None and bar['high'] >= t1
        hit_f = fail is not None and bar['low'] <= fail
        if hit_t and hit_f:
            outcome = 'ambiguous'; break
        if hit_f:
            outcome = 'fail'; break
        if hit_t:
            outcome = 't1'; break
    r['outcome_10'] = outcome if t1 is not None or fail is not None else 'no_levels'
    if t1 is not None:
        r['t1_dist_pct'] = round(100 * (t1 / px - 1), 2)
        r['t1_hit_10'] = bool(len(path) and path['high'].max() >= t1)
        r['t1_base_rate'] = round(base_rate_hit(b, post_pos, t1 / px - 1), 3)
        r['all_targets_hit_10'] = int(sum(path['high'].max() >= t for t in p['targets'])) if len(path) else 0
    if len(path):
        r['mfe_10_pct'] = round(100 * (path['high'].max() / px - 1), 2)
        r['mae_10_pct'] = round(100 * (path['low'].min() / px - 1), 2)
        r['ret_last_pct'] = round(100 * (path['close'].iloc[-1] / px - 1), 2)
        spy = bars('SPY'); sp = spy.loc[b.index[pos]:path.index[-1], 'close']
        r['spy_ret_pct'] = round(100 * (sp.iloc[-1] / spy['close'].shift().loc[b.index[pos]] - 1), 2) if len(sp) else np.nan
    rows.append(r)
plans = pd.DataFrame(rows)
plans.to_csv(OUT + '_plans.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('=== plans (daily bars; 10 sessions after fill; data end', bars('SPY').index[-1], ')')
print(plans.drop(columns=['posted']).to_string(index=False))
f = plans[plans.filled]
t = f[f.t1_hit_10.notna()]
done = t[t.sessions_seen >= 10]
print(f"\nfilled {len(f)}/{len(plans)}; with T1 {len(t)}; with full 10-session path {len(done)}")
if len(done):
    k = int(done.t1_hit_10.sum()); ps = done.t1_base_rate.fillna(0.5).to_list()
    print(f"T1 hit within 10 sessions: {k}/{len(done)}; expected from each name's own prior-year base rate {sum(ps):.1f}; "
          f"P(X<={k}) = {1 - poisson_binomial_p(ps, k + 1):.3f} (independence assumed; plans cluster on a few dates)")
    print(f"mean T1 distance {done.t1_dist_pct.mean():.1f}%; median MFE {done.mfe_10_pct.median():.1f}%, median MAE {done.mae_10_pct.median():.1f}%")
    print(f"mean return over path {done.ret_last_pct.mean():+.2f}% vs SPY {done.spy_ret_pct.mean():+.2f}% "
          f"(median excess {(done.ret_last_pct - done.spy_ret_pct).median():+.2f}%)")
print('outcome counts (first of T1 / fail):', f.outcome_10.value_counts().to_dict())
pull = f[(f.entry_vs_prior_close_pct <= 0)]
print(f"entries at/below prior close: {len(pull)}; within 1.5% of an MA: {(pull.nearest_ma_dist_pct.abs() <= 1.5).sum()} "
      f"({pull.nearest_ma.value_counts().to_dict()})")

# MA-proximity null: random pullback levels uniform in [0.90, 1.00] x prior close
rng = np.random.default_rng(0)
null = []
for _, r in pull.iterrows():
    prior = bars(r['symbol']).iloc[bars(r['symbol']).index.searchsorted(pd.Timestamp(r['fill_date']).date()) - 1]
    lv = prior['close'] * rng.uniform(0.90, 1.00, 2000)
    mas = np.array([prior[k] for k in ('sma5', 'ema21', 'sma34', 'sma55')])
    null.append((np.abs(lv[:, None] / mas[None, :] - 1).min(axis=1) <= 0.015).mean())
if null:
    print(f"null: a random pullback level lands within 1.5% of one of the four MAs {np.mean(null):.0%} of the time "
          f"(expected {sum(null):.1f} of {len(null)})")

# ---------------- fib reconstruction ----------------
print('\n=== target levels vs fib levels of trailing N-session (low, high), N = 5..150, tolerance 0.25%')


def fib_best(b, end_pos, targets, tol=0.0025):
    best = (0, None)
    lo_s, hi_s = b['low'].to_numpy()[:end_pos], b['high'].to_numpy()[:end_pos]
    for n in range(5, 151):
        if n > len(lo_s):
            break
        L, H = lo_s[-n:].min(), hi_s[-n:].max()
        levels = L + np.array(FIBS) * (H - L)
        m = sum(np.abs(levels / t - 1).min() <= tol for t in targets)
        if m > best[0]:
            best = (m, (n, round(L, 2), round(H, 2)))
    return best


fib_rows = []
for p in spec['plans']:
    if len(p['targets']) < 2:
        continue
    b = bars(p['symbol']); end = b.index.searchsorted(first_session(p['posted_utc'])[0])
    m, anchor = fib_best(b, end, p['targets'])
    jit = [fib_best(b, end, [t * rng.uniform(0.93, 1.07) for t in p['targets']])[0] for _ in range(40)]
    fib_rows.append({'id': p['id'], 'symbol': p['symbol'], 'n_targets': len(p['targets']), 'matched': m,
                     'anchor_N_L_H': anchor, 'null_mean_matched': round(float(np.mean(jit)), 2)})
fib = pd.DataFrame(fib_rows)
print(fib.to_string(index=False))
print(f"targets matched {fib.matched.sum()}/{fib.n_targets.sum()} vs jittered-target null {fib.null_mean_matched.sum():.1f}")
g = bars('GEV'); gend = g.index.searchsorted(pd.Timestamp('2026-10-07').date())
for n in (20, 34, 40, 45, 50, 55, 60):
    print(f"  GEV trailing {n}: low {g['low'].iloc[gend - n:gend].min():.2f} high {g['high'].iloc[gend - n:gend].max():.2f}  (chart: 868.25 / 1099.66)")

# ---------------- timing calls ----------------
print('\n=== cycle-timing calls: pivot inside the window vs the symbol\'s own base rate')


def trading_shift(d, n):
    d = pd.Timestamp(d).date(); step = 1 if n >= 0 else -1; k = 0
    while k < abs(n):
        d = d + pd.Timedelta(days=step)
        if is_trading_day(d):
            k += 1
    return d


def pivots(b, kind, k):
    x = b['low' if kind == 'low' else 'high'].to_numpy(); n = len(x); out = np.zeros(n, bool)
    for i in range(k, n - k):
        w = x[i - k:i + k + 1]
        out[i] = x[i] == (w.min() if kind == 'low' else w.max())
    return out


trows = []
for c in spec['timing_calls']:
    b = bars(c['symbol'])
    if 'window' in c:
        a, z = (pd.Timestamp(x).date() for x in c['window'])
    else:
        center = trading_shift(c['panel_as_of'], c['due_sessions']) if 'panel_as_of' in c else pd.Timestamp(c['center']).date()
        if not is_trading_day(center):
            center = trading_shift(center, 1)
        a, z = trading_shift(center, -c['tol']), trading_shift(center, c['tol'])
    for k in (5, 10):
        piv = pivots(b, c['kind'], k)
        idx = b.index
        ia, iz = idx.searchsorted(a), idx.searchsorted(z, side='right') - 1
        width = iz - ia + 1
        if iz + k >= len(idx):
            trows.append({'id': c['id'], 'symbol': c['symbol'], 'kind': c['kind'], 'k': k, 'window': f'{a}..{z}', 'hit': 'pending'}); continue
        hit = bool(piv[ia:iz + 1].any())
        post = idx.searchsorted(pd.Timestamp(c['posted_utc']).date())
        starts = range(max(k, post - 500), post - width - k)
        base = float(np.mean([piv[s:s + width].any() for s in starts]))
        piv_dates = idx[piv]
        nearest = min(piv_dates[(piv_dates >= idx[max(0, ia - 15)]) & (piv_dates <= idx[min(len(idx) - 1, iz + 15)])],
                      key=lambda d: min(abs(idx.get_loc(d) - ia), abs(idx.get_loc(d) - iz)), default=None)
        trows.append({'id': c['id'], 'symbol': c['symbol'], 'kind': c['kind'], 'k': k, 'window': f'{a}..{z}', 'width': width,
                      'hit': hit, 'base_rate': round(base, 2), 'nearest_pivot': str(nearest)})
tim = pd.DataFrame(trows)
tim.to_csv(OUT + '_timing.csv', index=False)
print(tim.to_string(index=False))
for k in (5, 10):
    d = tim[(tim.k == k) & (tim.hit != 'pending')]
    hits = int(d.hit.astype(bool).sum())
    print(f"k={k}: {hits}/{len(d)} windows contain a pivot; expected {d.base_rate.sum():.1f}; P(X>={hits}) = {poisson_binomial_p(d.base_rate.to_list(), hits):.3f}")

# ---------------- momentum rank of the names he picks ----------------
print('\n=== 63-session return percentile among the 1,000 most liquid names (last session before the post)')
files = glob.glob(f'{ROOT}/Data/shared/bars/1d/*.parquet')
panel_c, panel_dv = {}, {}  # 2024-09 start leaves 252 sessions of history before the first post
for fpath in files:
    try:
        d = pd.read_parquet(fpath, columns=['timestamp', 'close', 'volume'])
    except Exception:
        continue
    d['date'] = pd.to_datetime(d['timestamp']).dt.tz_convert(ET).dt.date
    d = d[d.date >= pd.Timestamp('2024-09-01').date()].drop_duplicates('date').set_index('date')
    if len(d) < 70:
        continue
    s = os.path.basename(fpath)[:-8]
    panel_c[s], panel_dv[s] = d['close'], d['close'] * d['volume']
C, DV = pd.DataFrame(panel_c).sort_index(), pd.DataFrame(panel_dv).sort_index()
liq = DV.rolling(20).median().rank(axis=1, ascending=False) <= 1000
jumps = (C / C.shift() - 1).abs()
rets = {h: (C / C.shift(h) - 1).mask(jumps.rolling(h).max() > 0.45) for h in (21, 63, 126, 252)}  # split artefacts
mom = []
for p in spec['plans']:
    d = first_session(p['posted_utc'])[0]
    day = C.index[C.index.searchsorted(d) - 1]
    s = p['symbol']; r = {'id': p['id'], 'symbol': s, 'asof': str(day), 'in_top1000': bool(liq.loc[day].get(s, False))}
    for h, R in rets.items():
        univ = R.loc[day][liq.loc[day]].dropna()
        r[f'pct{h}'] = round(100 * (univ < R.loc[day, s]).mean(), 1) if s in univ.index else np.nan
    mom.append(r)
mom = pd.DataFrame(mom)
print(mom.to_string(index=False))
u = mom.drop_duplicates('symbol')
for h in (21, 63, 126, 252):
    m = u[f'pct{h}'].dropna()
    print(f"{h:>3}-session return: {len(m)} names, median percentile {m.median():.0f}; top decile {int((m >= 90).sum())}; "
          f"top quintile {int((m >= 80).sum())}; bottom half {int((m < 50).sum())}")
