"""Reverse-engineer the private CycleORB panel from daily bars (exploratory, multiple hypotheses).

1. MA check: SMA5 / EMA21 / SMA34 / SMA55 of close at the panel's as-of bar.
2. Cycle: a pivot low has the lowest low within ``left`` bars before and ``right`` bars after.
   Cycle length L = mean/median of the last k pivot-low spacings (trading days); due = last pivot
   low + L - as-of (trading days). A random hypothesis matches one panel within +/-1 on both
   fields ~0.2% of the time, so matches across many panels are not chance even after the grid.
3. Bias (for the best cycle hypotheses): BULL if the highest high since the last pivot low falls
   after the cycle midpoint (right translation, Hurst convention).
Daily bars are Data/shared/bars/1d (unadjusted; none of these names split in the window).
The SPY panel is a 5-minute chart and is reported separately, not scored.
Usage: python cycle_reverse_engineer.py <repo_root> <panels.json>
"""
import itertools, json, sys
import numpy as np, pandas as pd

ROOT, PANELS = sys.argv[1], sys.argv[2]
panels = json.load(open(PANELS))['panels']


def bars(sym: str, as_of: str) -> pd.DataFrame:
    d = pd.read_parquet(f'{ROOT}/Data/shared/bars/1d/{sym}.parquet')
    d['date'] = pd.to_datetime(d['timestamp']).dt.tz_convert('America/New_York').dt.date.astype(str)
    return d[d.date <= as_of].reset_index(drop=True)


def pivot_lows(low: np.ndarray, left: int, right: int) -> list[int]:
    n = len(low)
    return [i for i in range(left, n - right) if low[i] == low[i - left:i + right + 1].min()
            and (i == 0 or low[i] < low[i - 1] or low[i - 1] != low[i])]


print('=== MA reproduction (TradingView-style; EMA seeded on full history) ===')
ma_rows = []
for p in panels:
    b = bars(p['symbol'], p['as_of']); c = b['close']
    calc = {'sma5': c.tail(5).mean(), 'ema21': c.ewm(span=21, adjust=False).mean().iloc[-1],
            'sma34': c.tail(34).mean(), 'sma55': c.tail(55).mean()}
    for k, v in calc.items():
        ma_rows.append({'symbol': p['symbol'], 'as_of': p['as_of'], 'ma': k, 'panel': p[k], 'ours': round(v, 2),
                        'diff_pct': 100 * (v / p[k] - 1), 'chart': p.get('chart', '1D')})
ma = pd.DataFrame(ma_rows)
daily = ma[ma.chart == '1D']
print(f"daily-chart panels: {(daily.diff_pct.abs() <= 0.15).sum()}/{len(daily)} MAs within 0.15%; "
      f"worst {daily.loc[daily.diff_pct.abs().idxmax(), ['symbol', 'ma', 'panel', 'ours']].to_dict()}")
print(ma[ma.diff_pct.abs() > 0.15].to_string(index=False))

print('\n=== D EM center vs prior close ===')
for p in panels:
    if 'd_em' in p and 'change' in p:
        prior = p['price'] - p['change']; c = sum(p['d_em']) / 2
        print(f"{p['symbol']:<5} {p['as_of']} prior close {prior:.2f}  D EM center {c:.2f}  diff {c - prior:+.2f}")

print('\n=== Cycle hypothesis grid ===')
scored = [p for p in panels if p.get('chart', '1D') == '1D']
data = {p['symbol'] + p['as_of']: bars(p['symbol'], p['as_of']) for p in panels}
results = []
for left, right, k, agg in itertools.product([2, 3, 4, 5, 6, 8, 10, 13, 15, 20], [1, 2, 3, 5, 8, 10], [1, 2, 3, 4, 5], ['mean', 'median']):
    if right > left:
        continue
    hits = both = 0; detail = []
    for p in scored:
        b = data[p['symbol'] + p['as_of']]; low = b['low'].to_numpy()
        piv = pivot_lows(low, left, right)
        if len(piv) < k + 1:
            detail.append(None); continue
        sp = np.diff(piv[-(k + 1):])
        L = int(round(sp.mean() if agg == 'mean' else np.median(sp)))
        due = piv[-1] + L - (len(b) - 1)
        okL, okD = abs(L - p['cycle_days']) <= 1, abs(due - p['due_low_days']) <= 1
        hits += okL; both += okL and okD; detail.append((L, due))
    results.append({'left': left, 'right': right, 'k': k, 'agg': agg, 'L_match': hits, 'L_and_due_match': both, 'detail': detail})
res = pd.DataFrame(results).sort_values(['L_and_due_match', 'L_match'], ascending=False)
print(f'{len(res)} hypotheses over {len(scored)} daily panels; best:')
print(res.head(8).drop(columns='detail').to_string(index=False))
best = res.iloc[0]
print('\nbest hypothesis per panel (panel L/due -> ours L/due):')
for p, d in zip(scored, best['detail']):
    print(f"  {p['symbol']:<5} {p['as_of']}  {p['cycle_days']:>3}d / {p['due_low_days']:>4}  ->  {d}")
print('\ndistribution of max matches: L-only', res.L_match.max(), '/', len(scored), '; L and due', res.L_and_due_match.max(), '/', len(scored))

print('\n=== Bias vs translation (best hypothesis) ===')
agree = 0
for p, d in zip(scored, best['detail']):
    if d is None:
        continue
    b = data[p['symbol'] + p['as_of']]; piv = pivot_lows(b['low'].to_numpy(), best['left'], best['right'])
    since = b.iloc[piv[-1]:]; peak = int(since['high'].to_numpy().argmax())
    right_t = peak / max(d[0], 1) > 0.5
    pred = 'BULL' if right_t else 'BEAR'; agree += pred == p['bias']
    print(f"  {p['symbol']:<5} peak at {peak} of L={d[0]} -> {pred} (panel {p['bias']})")
print(f'translation agreement {agree}/{len(scored)}')
