"""Are the caller's Robinhood quote screenshots contemporaneous with real market data?

For each parsed ``option_quote`` image: the phone clock is Pacific (12-hour, no AM/PM), resolved to the
latest PT time at or before the Discord post. The displayed underlying price is checked against SIP
1-minute bars (all sessions) within +/-3 minutes; the displayed option mark against Alpaca option trade
prints within +/-3 minutes when the contract's expiry is identifiable from the caller's own alert or
recap rows and the contract has expired. A quote page proves a price, not a position.
Usage: python quote_authenticity.py <repo_root> <image_evidence.jsonl> <alerts.jsonl> <screenshot_evidence.json> <out.csv>
"""
import json, sys
import pandas as pd

ROOT, IMG, ALERTS, EVID, OUT = sys.argv[1:6]
sys.path.insert(0, ROOT)
from alpaca.data.enums import Adjustment, DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from core.API.Alpaca_API.core.config import AlpacaConfig  # noqa: E402
from research.options_lab.chain_cache import fetch_bars  # noqa: E402

PT, ET, WIN = 'America/Los_Angeles', 'America/New_York', pd.Timedelta(minutes=3)
now = pd.Timestamp.now(tz='UTC')
cfg = AlpacaConfig.from_env()
client = StockHistoricalDataClient(cfg.key_id, cfg.secret_key)

# Known contracts: (symbol, strike, right) -> [(first_seen_utc, expiry)]
known = {}
for a in map(json.loads, open(ALERTS)):
    if a['expiry']:
        known.setdefault((a['symbol'], a['strike'], a['right']), []).append((pd.Timestamp(a['posted_at_utc']), a['expiry']))
for img in json.load(open(EVID))['images']:
    if img['type'] == 'premium_recap_card':
        for contract, *_ in img['rows']:
            p = contract.split(); sk = p[1]
            exp = img['date'] if '0DTE' in contract else f"2026-{int(p[-1].split('/')[0]):02d}-{int(p[-1].split('/')[1]):02d}"
            known.setdefault((p[0], float(sk[:-1]), 'call' if sk[-1].lower() == 'c' else 'put'), []).append(
                (pd.Timestamp(img['date'], tz=ET) - pd.Timedelta(days=4), exp))


def resolve_clock(clock: str, posted: pd.Timestamp) -> pd.Timestamp | None:
    if not clock:
        return None
    h, m = map(int, clock.split(':'))
    p = posted.tz_convert(PT)
    cands = [pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh % 24, minute=m, tz=PT)
             for d in (p - pd.Timedelta(days=1), p) for hh in {h % 12, h % 12 + 12}]
    cands = [c for c in cands if c <= p + pd.Timedelta(minutes=1)]
    return max(cands).tz_convert('UTC') if cands else None


day_cache = {}
def sip_day(sym: str, day) -> pd.DataFrame:
    key = (sym, day)
    if key not in day_cache:
        start = pd.Timestamp(f'{day} 04:00', tz=ET); end = pd.Timestamp(f'{day} 20:00', tz=ET)
        raw = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, start=start.to_pydatetime(), end=end.to_pydatetime(),
                                                     timeframe=TimeFrame.Minute, feed=DataFeed.SIP, adjustment=Adjustment.RAW, limit=10000)).df
        day_cache[key] = raw.reset_index().set_index('timestamp') if len(raw) else pd.DataFrame()
    return day_cache[key]


rows = []
for rec in map(json.loads, open(IMG)):
    if rec['kind'] != 'option_quote' or rec['author_id'] != '1557910141632782409':
        continue
    q = rec['quote']; posted = pd.Timestamp(rec['posted_at_utc'])
    shot = resolve_clock(q['phone_clock'], posted)
    r = {'attachment_id': rec['attachment_id'], 'posted_et': posted.tz_convert(ET).strftime('%m-%d %H:%M'),
         'shot_et': shot.tz_convert(ET).strftime('%m-%d %H:%M') if shot is not None else None,
         'contract': f"{q['underlying']} {q['strike']:g}{q['right'][0].upper()}", 'underlying_shown': q['underlying_price'],
         'mark_shown': q['mark'], 'session_tag': q['session_tag']}
    if shot is None:
        rows.append({**r, 'underlying_check': 'no_phone_clock'}); continue
    r['post_minus_shot_min'] = round((posted - shot).total_seconds() / 60, 1)
    if q['underlying'] != 'SPX':
        bars = sip_day(q['underlying'], shot.tz_convert(ET).date())
        w = bars[(bars.index >= shot - WIN) & (bars.index <= shot + WIN)] if len(bars) else bars
        if len(w):
            lo, hi = float(w['low'].min()), float(w['high'].max())
            r.update({'sip_low': lo, 'sip_high': hi, 'underlying_check': 'inside' if lo * 0.9995 <= q['underlying_price'] <= hi * 1.0005 else 'outside'})
        else:
            r['underlying_check'] = 'no_sip_bars'
    else:
        r['underlying_check'] = 'spx_not_on_alpaca'
    exps = [e for t0, e in known.get((q['underlying'], q['strike'], q['right']), []) if t0 <= posted and e >= str(shot.tz_convert(ET).date())]
    if exps:
        exp = min(exps); root = 'SPXW' if q['underlying'] == 'SPX' else q['underlying']
        osi = f"{root}{pd.Timestamp(exp):%y%m%d}{q['right'][0].upper()}{round(q['strike'] * 1000):08d}"
        r['expiry'] = exp
        if pd.Timestamp(f'{exp} 16:15', tz=ET) <= now:
            b = fetch_bars([osi], '1Min', (shot - WIN).isoformat(), (shot + WIN).isoformat())
            if len(b):
                lo, hi = float(b['l'].min()), float(b['h'].max())
                r.update({'opt_low': lo, 'opt_high': hi, 'option_check': 'inside' if lo * 0.97 <= q['mark'] <= hi * 1.03 else 'outside'})
            else:
                r['option_check'] = 'no_prints_in_window'
        else:
            r['option_check'] = 'unexpired_not_served'
    else:
        r['option_check'] = 'expiry_unknown'
    rows.append(r)

df = pd.DataFrame(rows).sort_values('posted_et')
df.to_csv(OUT, index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(df.to_string(index=False))
print('\nunderlying:', df.underlying_check.value_counts().to_dict())
print('option:', df.option_check.value_counts().to_dict())
print('post minus screenshot (min): median', df.post_minus_shot_min.median(), 'max', df.post_minus_shot_min.max())
