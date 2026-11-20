"""Score extracted ACE-successor contract alerts on Alpaca option trade prints.

Read-only: Alpaca historical option 1-minute bars (through the options_lab
cache) and SIP underlying minute bars. No orders, subscriptions or live state.

Option bars are trade prints, not marks, and historical quotes are unavailable,
so every alert carries its own fitness evidence: minutes with prints and the
sign-adjusted correlation of option vs underlying 1-minute log returns (calls
should be strongly positive, puts strongly negative before adjustment).

Rules were fixed before any outcome was inspected:
- entry: open of the first option bar starting >= post time + 60 s;
- MFE: max later bar high over entry (the "max gain" convention the caller's
  recap cards use) and over the stated premium;
- rule A: stated stop vs +100% first touch, the stop assumed first when both
  print in one bar; otherwise expiry intrinsic from the underlying's last RTH
  minute; unexpired contracts are censored at the last print;
- rule B: stated stop only, otherwise expiry intrinsic.
Returns are gross of spread and commissions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import numpy as np
import pandas as pd

from core.API.Alpaca_API.core.config import AlpacaConfig
from research.options_lab.chain_cache import fetch_bars
from scripts.discord_ledger.build import read_jsonl, write_jsonl

ET = 'America/New_York'
ENTRY_DELAY = pd.Timedelta(seconds=60)
OSI_ROOT = {'BRK.B': 'BRKB', 'SPX': 'SPXW'}
NO_UNDERLYING_BARS = {'SPX'}


def osi(alert: dict) -> str:
    exp = pd.Timestamp(alert['expiry'])
    return f"{OSI_ROOT.get(alert['symbol'], alert['symbol'])}{exp:%y%m%d}{alert['right'][0].upper()}{round(alert['strike'] * 1000):08d}"


def expiry_close(alert: dict) -> pd.Timestamp:
    return pd.Timestamp(f"{alert['expiry']} 16:00", tz=ET).tz_convert('UTC')


def intrinsic(alert: dict, spot: float) -> float:
    return max(0.0, spot - alert['strike']) if alert['right'] == 'call' else max(0.0, alert['strike'] - spot)


def first_touch(path: pd.DataFrame, stop: float | None, target: float | None) -> tuple[str, float | None, str | None]:
    for t, bar in path.iterrows():
        hit_stop = stop is not None and bar['low'] <= stop
        hit_target = target is not None and bar['high'] >= target
        if hit_stop:  # also the same-bar tie: assume the adverse print came first
            return ('stop_same_bar_as_target' if hit_target else 'stop'), min(stop, bar['open']), t.isoformat()
        if hit_target:
            return 'target', max(target, bar['open']), t.isoformat()
    return 'none', None, None


def sign_adjusted_corr(opt: pd.DataFrame, und: pd.DataFrame, right: str) -> tuple[float | None, int]:
    both = pd.concat({'o': np.log(opt['close']), 'u': np.log(und['close'])}, axis=1, join='inner').diff().dropna()
    both = both[(both['o'] != 0) | (both['u'] != 0)]
    if len(both) < 10:
        return None, len(both)
    c = float(both['o'].corr(both['u']))
    return (c if right == 'call' else -c), len(both)


def score(alert: dict, opt: pd.DataFrame, und: pd.DataFrame | None, now: pd.Timestamp) -> dict:
    posted = pd.Timestamp(alert['posted_at_utc'])
    end = min(expiry_close(alert), now)
    out = {k: alert[k] for k in ('alert_id', 'symbol', 'strike', 'right', 'expiry', 'premium', 'stop', 'flags', 'posted_at_et')}
    out['osi'] = osi(alert)
    expired = expiry_close(alert) <= now
    out['status'] = 'expired' if expired else 'censored_unexpired'
    opt = opt[(opt.index >= posted) & (opt.index <= end)]
    later = opt[opt.index >= posted + ENTRY_DELAY]
    out['option_minutes_with_prints'] = int(len(opt))
    if later.empty:
        return {**out, 'status': 'no_option_prints_after_post'}
    entry_t, entry = later.index[0], float(later['open'].iloc[0])
    path = later.iloc[1:]
    out.update({'entry_time_utc': entry_t.isoformat(), 'entry_delay_s': (entry_t - posted).total_seconds(),
                'entry_proxy': entry, 'entry_vs_stated_pct': 100 * (entry / alert['premium'] - 1)})
    peak = float(path['high'].max()) if len(path) else entry
    out['mfe_pct_vs_entry'] = 100 * (peak / entry - 1)
    out['mfe_pct_vs_stated'] = 100 * (peak / alert['premium'] - 1)
    out['mae_pct_vs_entry'] = 100 * (float(path['low'].min()) / entry - 1) if len(path) else 0.0
    if und is not None and len(und):
        rth = und[(und.index < expiry_close(alert)) & (und.index.tz_convert(ET).time >= pd.Timestamp('09:30').time())]
        corr, n = sign_adjusted_corr(opt, und[(und.index >= posted) & (und.index <= end)], alert['right'])
        out.update({'sign_adjusted_corr_option_vs_underlying': corr, 'corr_minutes': n})
        final_spot = float(rth['close'].iloc[-1]) if expired and len(rth) else None
    else:
        out.update({'sign_adjusted_corr_option_vs_underlying': None, 'corr_minutes': 0})
        final_spot = None
    final = intrinsic(alert, final_spot) if final_spot is not None else float(opt['close'].iloc[-1])
    out['final_basis'] = 'expiry_intrinsic_from_underlying' if final_spot is not None else 'last_option_print'
    out['final_value'] = final
    out['final_spot'] = final_spot
    out['hold_to_final_return_pct'] = 100 * (final / entry - 1)
    if alert['stop'] is not None and entry <= alert['stop']:
        # A follower's first fill is already through the stated stop (e.g. a post one minute before the close).
        return {**out, 'rule_scoring': 'unscorable_entry_at_or_below_stated_stop'}
    out['rule_scoring'] = 'scored'
    for rule, target in (('a', 2 * entry), ('b', None)):
        how, px, when = first_touch(path, alert['stop'], target)
        exit_px = px if px is not None else final
        out[f'rule_{rule}_exit'] = how if px is not None else out['final_basis']
        out[f'rule_{rule}_exit_time_utc'] = when
        out[f'rule_{rule}_return_pct'] = 100 * (exit_px / entry - 1)
    return out


def underlying_bars(client: StockHistoricalDataClient, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, start=start.to_pydatetime(), end=end.to_pydatetime(),
                                                 timeframe=TimeFrame.Minute, feed=DataFeed.SIP, adjustment=Adjustment.RAW, limit=10000))
    df = raw.df.reset_index()
    if len(df) >= 10000:
        raise ValueError(f'{symbol} underlying request may be truncated; split the window')
    return df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']] if len(df) else df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alerts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    now = pd.Timestamp.now(tz='UTC')
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(cfg.key_id, cfg.secret_key)
    rows, raw_dir = [], args.out / 'raw'
    raw_dir.mkdir()
    for alert in (a for a in read_jsonl(args.alerts) if a['kind'] == 'alert' and a['expiry']):
        posted = pd.Timestamp(alert['posted_at_utc'])
        end = min(expiry_close(alert) + pd.Timedelta(minutes=15), now)
        try:
            opt = fetch_bars([osi(alert)], '1Min', posted.isoformat(), end.isoformat())
        except Exception as exc:  # recorded per alert; a vendor refusal is a coverage fact, not a zero
            rows.append({'alert_id': alert['alert_id'], 'osi': osi(alert), 'status': 'option_fetch_error', 'error': str(exc)[:200]})
            continue
        if opt.empty:
            rows.append({'alert_id': alert['alert_id'], 'osi': osi(alert), 'status': 'no_option_bars'})
            continue
        opt = (opt.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'trades', 'vw': 'vwap'})
               .assign(t=lambda d: pd.to_datetime(d['t'], utc=True)).set_index('t').sort_index())
        opt.to_parquet(raw_dir / f'{osi(alert)}.parquet')
        und = None
        if alert['symbol'] not in NO_UNDERLYING_BARS:
            und = underlying_bars(client, alert['symbol'], posted - pd.Timedelta(minutes=5), end)
            und.to_parquet(raw_dir / f"{osi(alert)}_underlying_{alert['symbol']}.parquet")
        rows.append(score(alert, opt, und, now))
    write_jsonl(args.out / 'scored_alerts.jsonl', rows)
    (args.out / 'manifest.json').write_text(json.dumps({
        'schema': 'ace_successor_alert_scoring_v1', 'alerts_path': str(args.alerts), 'scored_at_utc': now.isoformat(),
        'option_source': 'alpaca_v1beta1_options_bars_1Min_trade_prints', 'underlying_source': 'alpaca_sip_raw_1Min',
        'entry_delay_seconds': ENTRY_DELAY.total_seconds(), 'costs': 'gross_no_spread_no_commission',
        'status_counts': pd.Series([r['status'] for r in rows]).value_counts().to_dict()}, indent=2) + '\n')
    print(json.dumps({'scored': len(rows), 'out': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
