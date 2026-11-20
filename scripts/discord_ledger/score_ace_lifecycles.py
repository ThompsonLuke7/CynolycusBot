"""Exit-based scoring of every ACE option entry lifecycle against a same-contract random-entry control.

Research only (year-two checklist 3.2 gate): Alpaca historical option 1-minute trade bars through the
options_lab cache, SIP underlying minutes, and the unadjusted daily bar cache. No orders or live state.

Population: every ACE / Zach_ace ``entry_claim_or_instruction`` option lifecycle in a built ledger that
names symbol, strike, right and expiry. Lifecycles without a full contract are counted, not priced.

Rules, fixed before any outcome was computed (2026-11-06):
- entry: open of the first option bar starting >= the entry's availability + 60 s, within 30 minutes;
  later first prints are ``no_timely_entry_print`` (a follower could not have paid that price);
- stated_exits: each linked ``exit`` sells everything left and each ``trim`` half of what is left, at the
  first print >= its availability + 60 s; whatever is left is valued at expiry;
- hold_to_expiry; flat_same_session (last print of the entry session); double_or_expiry (a later print
  >= 2x entry fills a limit at 2x, else expiry value);
- expiry value: intrinsic at the last raw SIP minute close by 16:00 ET on the expiry date (equity/ETF
  options) or the last print by 16:15 ET (index options, no index bars); unexpired contracts are censored
  for expiry-dependent policies. (First run used the daily bar cache, but its CVNA file is split-adjusted for
  the 2026-05-08 5:1 split while option strikes are raw, so the source was replaced before any result was used.)
  Each intrinsic is checked against the contract's own last print in the final 30 minutes;
- duplicate lifecycles (same contract and alert time, annotated twice in the ledger) are scored once;
- control: up to 20 random entry minutes on the same contract and session (09:31-15:30 ET, >= 5 minutes
  from the real entry), identical exit policies; it measures entry timing on his chosen contract;
- fitness: sign-adjusted correlation of option vs underlying 1-minute returns on the entry session;
  ``fit`` needs >= 0.5 over >= 30 shared minutes (stale prints otherwise);
- gate: pass only if the stated-exit win rate beats the control with the 95% date-clustered CI of the
  paired difference above zero AND the mean stated-exit return CI is above zero.
Returns are gross of spread and commissions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
import numpy as np
import pandas as pd

from core.API.Alpaca_API.core.config import AlpacaConfig
from research.options_lab.chain_cache import fetch_bars
from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.identities import ACE_IDS
from scripts.discord_ledger.score_ace_contract_alerts import OSI_ROOT, sign_adjusted_corr, underlying_bars

ET = 'America/New_York'
ENTRY_DELAY = pd.Timedelta(seconds=60)
MAX_ENTRY_WAIT = pd.Timedelta(minutes=30)
CONTROL_DRAWS, CONTROL_GAP = 20, pd.Timedelta(minutes=5)
INDEX_ROOTS = {'SPX', 'SPXW', 'NDX', 'NDXP'}
UNDERLYING_PROXY = {'SPX': 'SPY', 'SPXW': 'SPY', 'NDX': 'QQQ', 'NDXP': 'QQQ', 'BRK.B': 'BRK.B'}
POLICIES = ('stated_exits', 'hold_to_expiry', 'flat_same_session', 'double_or_expiry')


def osi_for(row: dict) -> str:
    exp = pd.Timestamp(row['expiry'])
    root = 'SPXW' if row['symbol'] == 'SPX' else OSI_ROOT.get(row['symbol'], row['symbol'])
    return f"{root}{exp:%y%m%d}{row['option_type'][0].upper()}{round(float(row['strike']) * 1000):08d}"


def session_close(t: pd.Timestamp, index: bool) -> pd.Timestamp:
    d = t.tz_convert(ET).date()
    return pd.Timestamp(f"{d} {'16:15' if index else '16:00'}", tz=ET).tz_convert('UTC')


def intrinsic(right: str, strike: float, spot: float) -> float:
    return max(0.0, spot - strike) if right == 'call' else max(0.0, strike - spot)


def first_print_after(opt: pd.DataFrame, t: pd.Timestamp, until: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    later = opt[(opt.index >= t) & (opt.index <= until)]
    return (later.index[0], float(later['open'].iloc[0])) if len(later) else None


def policy_values(opt: pd.DataFrame, entry_t: pd.Timestamp, entry: float, management: list[tuple[str, pd.Timestamp]],
                  expiry_value: float | None, expiry_t: pd.Timestamp, index: bool) -> dict[str, float | None]:
    """Exit value per unit for each policy; None when the policy needs an expiry that has not happened."""
    path = opt[(opt.index > entry_t) & (opt.index <= expiry_t)]
    remaining, value = 1.0, 0.0
    for kind, t in sorted(management, key=lambda m: m[1]):
        if t <= entry_t or remaining == 0:
            continue
        hit = first_print_after(opt, t + ENTRY_DELAY, expiry_t)
        if hit is None:
            continue
        sold = remaining if kind == 'exit' else remaining / 2
        value += sold * hit[1]
        remaining -= sold
    if remaining == 0:
        stated = value
    else:
        stated = None if expiry_value is None else value + remaining * expiry_value
    # Every print on the entry date from the entry bar on (ETF options trade to 16:15, so never empty).
    same = opt[(opt.index >= entry_t) & (opt.index <= expiry_t)
               & (opt.index.tz_convert(ET).date == entry_t.tz_convert(ET).date())]
    doubled = bool(len(path)) and float(path['high'].max()) >= 2 * entry
    return {'stated_exits': stated,
            'hold_to_expiry': expiry_value,
            'flat_same_session': float(same['close'].iloc[-1]),
            'double_or_expiry': 2 * entry if doubled else expiry_value}


def control_entries(opt: pd.DataFrame, entry_t: pd.Timestamp, rng: np.random.Generator) -> list[tuple[pd.Timestamp, float]]:
    local = opt.index.tz_convert(ET)
    day = entry_t.tz_convert(ET).date()
    mask = ((local.date == day) & (local.time >= pd.Timestamp('09:31').time()) & (local.time <= pd.Timestamp('15:30').time())
            & (abs(opt.index - entry_t) >= CONTROL_GAP))
    pool = opt[mask]
    if len(pool) < 5:
        return []
    picks = rng.choice(len(pool), size=min(CONTROL_DRAWS, len(pool)), replace=False)
    return [(pool.index[i], float(pool['open'].iloc[i])) for i in sorted(picks)]


def expiry_check(opt: pd.DataFrame, expiry_t: pd.Timestamp, value: float) -> str:
    """Compare an intrinsic expiry value with the contract's own last print in the final 30 minutes."""
    late = opt[(opt.index <= expiry_t) & (opt.index >= expiry_t - pd.Timedelta(minutes=30))]
    if late.empty:
        return 'unchecked_no_late_print'
    last = float(late['close'].iloc[-1])
    return 'consistent' if abs(value - last) <= max(0.25, 0.5 * last) else 'mismatch'


def score(row: dict, management: list[tuple[str, pd.Timestamp]], opt: pd.DataFrame, und: pd.DataFrame | None,
          now: pd.Timestamp, rng: np.random.Generator, expiry_spot: float | None = None) -> dict:
    sym, right, strike = row['symbol'], row['option_type'], float(row['strike'])
    index = sym in INDEX_ROOTS
    expiry_t = session_close(pd.Timestamp(f"{row['expiry']} 12:00", tz=ET).tz_convert('UTC'), index)
    out = {'trade_id': row['trade_id'], 'author_id': row['author_id'], 'author_name': row['author_name'], 'symbol': sym,
           'option_type': right, 'strike': strike, 'expiry': row['expiry'], 'osi': osi_for(row),
           'entry_available_at_utc': row['entry_available_at_utc'], 'reported_entry_price': row.get('reported_entry_price'),
           'linked_management': [[k, t.isoformat()] for k, t in management]}
    avail = pd.Timestamp(row['entry_available_at_utc'])
    hit = first_print_after(opt, avail + ENTRY_DELAY, min(expiry_t, now))
    if hit is None:
        return {**out, 'status': 'no_timely_entry_print'}
    entry_t, entry = hit
    # The wait is measured from the later of the follower's earliest action and that session's open,
    # so a premarket or overnight alert is entered at the open rather than rejected.
    session_open = pd.Timestamp(f"{entry_t.tz_convert(ET).date()} 09:30", tz=ET).tz_convert('UTC')
    if entry_t - max(avail + ENTRY_DELAY, session_open) > MAX_ENTRY_WAIT:
        return {**out, 'status': 'no_timely_entry_print', 'first_print_utc': entry_t.isoformat()}
    if entry <= 0:
        return {**out, 'status': 'zero_entry_print'}
    expired = expiry_t <= now
    if not expired:
        expiry_value, basis = None, 'censored_unexpired'
    elif index:
        last = opt[(opt.index <= expiry_t) & (opt.index.tz_convert(ET).date == expiry_t.tz_convert(ET).date())]
        expiry_value, basis = (float(last['close'].iloc[-1]), 'last_index_option_print') if len(last) else (None, 'no_expiry_print')
    elif expiry_spot is not None:
        expiry_value, basis = intrinsic(right, strike, expiry_spot), 'intrinsic_at_sip_last_rth_minute'
    else:
        expiry_value, basis = None, 'no_expiry_spot'
    check = expiry_check(opt, expiry_t, expiry_value) if expiry_value is not None and not index else None
    vals = policy_values(opt, entry_t, entry, management, expiry_value, expiry_t, index)
    out.update({'status': 'scored', 'entry_time_utc': entry_t.isoformat(), 'entry_delay_s': (entry_t - avail).total_seconds(),
                'entry_proxy': entry, 'expiry_basis': basis, 'expiry_spot': expiry_spot, 'expiry_check': check,
                **{f'{p}_return_pct': (100 * (v / entry - 1) if v is not None else None) for p, v in vals.items()}})
    if und is not None and len(und):
        day = entry_t.tz_convert(ET).date()
        u_day, o_day = und[und.index.tz_convert(ET).date == day], opt[opt.index.tz_convert(ET).date == day]
        corr, n = sign_adjusted_corr(o_day, u_day, right)
        u_after = u_day[u_day.index >= entry_t]
        sign = 1 if right == 'call' else -1
        out.update({'sign_adjusted_corr': corr, 'corr_minutes': n,
                    'underlying_dir_return_bps': (1e4 * sign * (u_after['close'].iloc[-1] / u_after['open'].iloc[0] - 1)
                                                  if len(u_after) else None)})
    else:
        out.update({'sign_adjusted_corr': None, 'corr_minutes': 0, 'underlying_dir_return_bps': None})
    out['fit'] = bool(out['sign_adjusted_corr'] is not None and out['sign_adjusted_corr'] >= 0.5 and out['corr_minutes'] >= 30)
    controls = control_entries(opt, entry_t, rng)
    out['control_draws'] = len(controls)
    for p in POLICIES:
        rets = []
        for t, px in controls:
            v = policy_values(opt, t, px, management, expiry_value, expiry_t, index)[p]
            if v is not None and px > 0:
                rets.append(100 * (v / px - 1))
        out[f'control_{p}_return_pct'] = float(np.mean(rets)) if rets else None
        out[f'control_{p}_win'] = float(np.mean([r > 0 for r in rets])) if rets else None
    return out


def summarize(scored: list[dict], rng: np.random.Generator) -> dict:
    df = pd.DataFrame([r for r in scored if r['status'] == 'scored'])
    if df.empty:
        return {}
    df['date'] = pd.to_datetime(df['entry_time_utc']).dt.tz_convert(ET).dt.date
    out = {}
    for subset, d0 in (('all', df), ('fit', df[df['fit']])):
        for p in POLICIES:
            d = d0.dropna(subset=[f'{p}_return_pct', f'control_{p}_win'])
            if len(d) < 5:
                continue
            ret, win, cwin = d[f'{p}_return_pct'], (d[f'{p}_return_pct'] > 0).astype(float), d[f'control_{p}_win']
            g = pd.DataFrame({'date': d['date'], 'ret': ret, 'win': win, 'cwin': cwin}).groupby('date').agg(['sum', 'size'])
            n, s_ret, s_win, s_cwin = (g[('ret', 'size')].to_numpy(), g[('ret', 'sum')].to_numpy(),
                                       g[('win', 'sum')].to_numpy(), g[('cwin', 'sum')].to_numpy())
            dates = g.index
            boots = {'mean': [], 'win_minus_control': []}
            for _ in range(2000):  # resample whole dates with replacement (one market path per date)
                i = rng.integers(0, len(n), len(n))
                boots['mean'].append(s_ret[i].sum() / n[i].sum())
                boots['win_minus_control'].append((s_win[i].sum() - s_cwin[i].sum()) / n[i].sum())
            ci = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in boots.items()}
            out[f'{subset}:{p}'] = {'n': int(len(d)), 'dates': int(len(dates)), 'win_rate': float(win.mean()),
                                    'control_win_rate': float(cwin.mean()), 'win_minus_control_ci': ci['win_minus_control'],
                                    'mean_return_pct': float(ret.mean()), 'mean_ci': ci['mean'], 'median_return_pct': float(ret.median()),
                                    'control_mean_return_pct': float(d[f'control_{p}_return_pct'].mean())}
    s = out.get('all:stated_exits')
    out['gate'] = {'passes': bool(s and s['win_minus_control_ci'][0] > 0 and s['mean_ci'][0] > 0),
                   'rule': 'stated-exit win rate above control (paired CI > 0) and mean stated-exit return CI > 0'}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True, help='built ledger directory (managed_trades.jsonl, events.jsonl)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    now = pd.Timestamp.now(tz='UTC')
    rng = np.random.default_rng(args.seed)
    events = {e['event_id']: e for e in read_jsonl(args.ledger / 'events.jsonl')}
    trades = [t for t in read_jsonl(args.ledger / 'managed_trades.jsonl')
              if t['author_id'] in ACE_IDS and t['origin'] == 'entry_claim_or_instruction' and t['instrument_type'] == 'option']
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(cfg.key_id, cfg.secret_key)
    rows, seen = [], set()
    for t in trades:
        if not (t.get('expiry') and t.get('strike') is not None and t.get('option_type') in ('call', 'put') and t.get('symbol')):
            rows.append({'trade_id': t['trade_id'], 'symbol': t.get('symbol'), 'status': 'no_full_contract'})
            continue
        management = [(events[i]['event_type'], pd.Timestamp(events[i]['available_at_utc']))
                      for i in t['event_ids'] if i != t['entry_event_id'] and events[i]['event_type'] in ('exit', 'trim')]
        key = (osi_for(t), t['entry_available_at_utc'])
        if key in seen:
            rows.append({'trade_id': t['trade_id'], 'osi': key[0], 'status': 'duplicate_lifecycle'})
            continue
        seen.add(key)
        avail = pd.Timestamp(t['entry_available_at_utc'])
        index = t['symbol'] in INDEX_ROOTS
        expiry_t = session_close(pd.Timestamp(f"{t['expiry']} 12:00", tz=ET).tz_convert('UTC'), index)
        if expiry_t < avail:
            rows.append({'trade_id': t['trade_id'], 'symbol': t['symbol'], 'status': 'expiry_before_alert'})
            continue
        try:
            opt = fetch_bars([osi_for(t)], '1Min', (avail - pd.Timedelta(minutes=5)).isoformat(), min(expiry_t, now).isoformat())
        except Exception as exc:  # a vendor refusal (e.g. unexpired contract) is a coverage fact, not a zero
            rows.append({'trade_id': t['trade_id'], 'osi': osi_for(t), 'status': 'option_fetch_error', 'error': str(exc)[:160]})
            continue
        if opt.empty:
            rows.append({'trade_id': t['trade_id'], 'osi': osi_for(t), 'status': 'no_option_bars'})
            continue
        opt = (opt.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
               .assign(t=lambda d: pd.to_datetime(d['t'], utc=True)).set_index('t').sort_index())
        first = opt[opt.index >= avail + ENTRY_DELAY]
        und = None
        if len(first):
            day = first.index[0].tz_convert(ET).date()
            usym = UNDERLYING_PROXY.get(t['symbol'], t['symbol'])
            try:
                und = underlying_bars(client, usym, pd.Timestamp(f'{day} 09:30', tz=ET).tz_convert('UTC'),
                                      pd.Timestamp(f'{day} 16:00', tz=ET).tz_convert('UTC'))
            except Exception:
                und = None
        spot = None
        if not index and expiry_t <= now:
            usym, eday = UNDERLYING_PROXY.get(t['symbol'], t['symbol']), expiry_t.tz_convert(ET).date()
            try:
                close = underlying_bars(client, usym, pd.Timestamp(f'{eday} 15:30', tz=ET).tz_convert('UTC'), expiry_t)
                close = close[close.index < expiry_t]
                spot = float(close['close'].iloc[-1]) if len(close) else None
            except Exception:
                spot = None
        rows.append(score(t, management, opt, und, now, rng, spot))
    write_jsonl(args.out / 'scored_lifecycles.jsonl', rows)
    summary = summarize(rows, rng)
    status = pd.Series([r['status'] for r in rows]).value_counts().to_dict()
    (args.out / 'summary.json').write_text(json.dumps({
        'schema': 'ace_lifecycle_exit_scoring_v1', 'ledger': str(args.ledger), 'scored_at_utc': now.isoformat(), 'seed': args.seed,
        'population': len(trades), 'status_counts': status, 'option_source': 'alpaca_v1beta1_options_bars_1Min_trade_prints',
        'costs': 'gross_no_spread_no_commission', 'results': summary}, indent=2, default=str) + '\n')
    print(json.dumps({'population': len(trades), 'status_counts': status, 'gate': summary.get('gate')}, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
