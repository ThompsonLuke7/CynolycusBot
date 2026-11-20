"""Check recap-card entry/max claims against Alpaca option trade prints (expired contracts only).

The VIP alert times are not visible, so this tests number honesty, not selection: did the claimed
entry print within the three sessions before the card, and did the claimed max print after it?
Usage: python recap_claims_check.py <repo_root> <screenshot_evidence.json>
"""
import json, sys
import pandas as pd

ROOT, EVIDENCE = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
from research.options_lab.chain_cache import fetch_bars  # noqa: E402
from core.calendar.us_market_calendar import is_trading_day  # noqa: E402

ROOTS = {'SPX': 'SPXW'}
now = pd.Timestamp.now(tz='UTC')


def parse(contract: str, card_date: str):
    parts = contract.replace('  ', ' ').split()
    sym, sk = parts[0], parts[1]
    right, strike = sk[-1].upper(), float(sk[:-1])
    exp = card_date if '0DTE' in contract else f"2026-{int(parts[-1].split('/')[0]):02d}-{int(parts[-1].split('/')[1]):02d}"
    return sym, right, strike, exp


def window_start(card_date: str, sessions: int = 3) -> pd.Timestamp:
    d, n = pd.Timestamp(card_date), 0
    while n < sessions - 1:
        d -= pd.Timedelta(days=1); n += is_trading_day(d.date())
    return pd.Timestamp(f'{d.date()} 09:30', tz='America/New_York').tz_convert('UTC')


out = []
for img in json.load(open(EVIDENCE))['images']:
    if img['type'] != 'premium_recap_card':
        continue
    posted = pd.Timestamp(img['posted_at_et'], tz='America/New_York').tz_convert('UTC')
    for contract, entry, claimed_max, claimed_pct in img['rows']:
        sym, right, strike, exp = parse(contract, img['date'])
        osi = f"{ROOTS.get(sym, sym)}{pd.Timestamp(exp):%y%m%d}{right}{round(strike * 1000):08d}"
        r = {'card': img['date'], 'contract': contract, 'osi': osi, 'entry': entry, 'claimed_max': claimed_max}
        if pd.Timestamp(f'{exp} 16:15', tz='America/New_York') > now:
            out.append({**r, 'status': 'unexpired_not_served'}); continue
        try:
            # A 0DTE card row was entered on the card's own session; earlier sessions priced it as a longer-dated option.
            start = window_start(img['date'], 1 if '0DTE' in contract else 3)
            b = fetch_bars([osi], '1Min', start.isoformat(), posted.isoformat())
        except Exception as exc:
            out.append({**r, 'status': f'fetch_error {str(exc)[:60]}'}); continue
        if b.empty:
            out.append({**r, 'status': 'no_prints'}); continue
        b = b.assign(t=pd.to_datetime(b['t'], utc=True)).set_index('t').sort_index()
        at_entry = b[b['l'] <= entry * 1.02]
        if at_entry.empty:
            out.append({**r, 'status': 'entry_never_printed', 'window_low': float(b['l'].min()), 'window_max': float(b['h'].max())}); continue
        t0 = at_entry.index[0]
        peak = float(b.loc[t0:, 'h'].max())
        out.append({**r, 'status': 'checked', 'first_entry_print_et': t0.tz_convert('America/New_York').strftime('%m-%d %H:%M'),
                    'observed_max_after_entry': peak, 'claimed_vs_observed_max_pct': 100 * (claimed_max / peak - 1),
                    'minutes_with_prints': len(b)})

df = pd.DataFrame(out)
pd.set_option('display.width', 220); pd.set_option('display.precision', 2)
print(df.drop(columns=['osi']).to_string(index=False))
c = df[df.status == 'checked']
print(f"\nchecked {len(c)}/{len(df)}; claimed max within ±10% of observed print max: {(c.claimed_vs_observed_max_pct.abs() <= 10).sum()}/{len(c)}")
