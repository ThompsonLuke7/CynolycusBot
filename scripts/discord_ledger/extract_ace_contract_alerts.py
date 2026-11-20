"""Extract source-cited option-contract alerts from ACE's successor account.

From 2026-10-12 ACE posts as ``Zach_ace`` in a new ``ace-free-alerts``
channel; the original account's last message is 2026-10-08 16:54 UTC and the
new account was created 2026-10-09 00:18 UTC. The alias rests on that timing,
ownership of the same alert/profit channels and members addressing him as Ace.
It is recorded as inferred, never authenticated.

A row is an ``alert`` only when one line names underlying, strike, right and a
premium. Lines without a premium are ``idea`` rows. Management is linked by
Discord reply ID to the alert message only, never by symbol proximity. Lines
quoting a percentage are result claims, not alerts, and are skipped.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from core.calendar.us_market_calendar import next_trading_day
from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.identities import ACE_ID, ACE_SUCCESSOR_ID
from scripts.discord_ledger.identities import SUCCESSOR_ALERTS_CHANNEL_ID as NEW_ALERTS_CHANNEL_ID

ET = ZoneInfo('America/New_York')
ALIAS = {'author_id': ACE_SUCCESSOR_ID, 'author_name': 'Zach_ace', 'alias_of': f'ACE ({ACE_ID})',
         'alias_status': 'inferred_from_account_creation_timing_channel_ownership_and_member_address_not_authenticated'}
CONTRACT = re.compile(r'(?<![\w.])\$?(?P<sym>[A-Za-z]{1,5}(?:\.[A-Za-z])?)\s+(?P<strike>\d+(?:\.\d+)?)\s*'
                      r'(?P<right>calls?|puts?|c|p)\b(?P<rest>[^\n]*)', re.I)
NOT_TICKERS = {'AT', 'TO', 'THE', 'AND', 'OR', 'IN', 'ON', 'OF', 'ITM', 'SL', 'TP', 'BE', 'IS', 'ARE', 'MY', 'UP',
               'OUR', 'ALL', 'FOR', 'WITH', 'YOUR', 'THESE', 'THOSE'}
# Fills/reductions on an existing position ("Sold half of MRK 155 calls at 2.02") are management, not new alerts.
MANAGEMENT = re.compile(r'\b(?:sold|selling|trim\w*|took|taking|stc|closing|closed|out)\b', re.I)
STOP = re.compile(r'\b(?:SL|stop)\s*(?:at\s*)?(\d*\.\d+|\d+)', re.I)
PREMIUM = re.compile(r'(?<![\d/.])(\d*\.\d+)(?![\d/%])')
EXPIRY = re.compile(r'(?<![\d.])(\d{1,2})/(\d{1,2})(?![\d/])')
DTE = re.compile(r'\b([01])\s*DTE\b', re.I)
STANDALONE_STOP = re.compile(r'^\s*(?:SL|stop)\s*(\d*\.\d+)\s*$', re.I)


def et(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(ET)


def resolve_expiry(rest: str, posted: datetime) -> tuple[str | None, str | None]:
    """Literal M/D is resolved to the posting year; 0DTE/1DTE to the posting session."""
    if m := DTE.search(rest):
        session = posted.date()
        return (session if m.group(1) == '0' else next_trading_day(session)).isoformat(), f'{m.group(1)}DTE_relative_to_post_date'
    if m := EXPIRY.search(rest):
        month, day = int(m.group(1)), int(m.group(2))
        year = posted.year + (1 if month < posted.month - 6 else 0)
        return date(year, month, day).isoformat(), 'literal_month_day_year_inferred_from_post_date'
    return None, None


def parse_line(line: str, posted: datetime, context: str | None = None) -> list[dict]:
    """``context`` is the whole message, so flags on another line of it still apply."""
    rows = []
    if '%' in line or MANAGEMENT.search(line):
        return rows
    # "CRM 247.5 call 10/23 OR CRM 245 call 10/23" names two contracts on one line.
    for segment in re.split(r'\s(?:OR|&)\s', line, flags=re.I):
        m = next((c for c in CONTRACT.finditer(segment) if c.group('sym').upper() not in NOT_TICKERS), None)
        if m is None:
            continue
        sym, rest = m.group('sym').upper(), m.group('rest')
        stop = STOP.search(rest)
        without_stop = STOP.sub(' ', rest)
        without_dates = EXPIRY.sub(' ', without_stop)
        premium = PREMIUM.search(without_dates)
        expiry, basis = resolve_expiry(rest, posted)
        lower = (context or line).lower()
        rows.append({'symbol': sym, 'strike': float(m.group('strike')),
                     'right': 'call' if m.group('right').lower().startswith('c') else 'put',
                     'expiry': expiry, 'expiry_basis': basis,
                     'premium': float(premium.group(1)) if premium else None,
                     'stop': float(stop.group(1)) if stop else None,
                     'flags': [f for f in ('lotto', 'size for zero', 'super lotto', 'swing', 'hedge') if f in lower],
                     'kind': 'alert' if premium else 'idea', 'raw_line': line.strip()})
    return rows


def extract(messages: list[dict]) -> list[dict]:
    own = sorted((m for m in messages if m['author_id'] == ACE_SUCCESSOR_ID), key=lambda m: (m['timestamp_utc'], m['message_id']))
    replies: dict[str, list[dict]] = {}
    for m in own:
        if m.get('reply_to_message_id'):
            replies.setdefault(m['reply_to_message_id'], []).append(m)
    out = []
    for i, m in enumerate(own):
        if m['channel_id'] != NEW_ALERTS_CHANNEL_ID or not m.get('text'):
            continue
        posted = et(m['timestamp_utc'])
        for line in m['text'].splitlines():
            for row in parse_line(line, posted, m['text']):
                sources = [m['message_id']]
                if row['kind'] == 'alert' and row['stop'] is None:
                    # A stop posted alone within two minutes ("Stop 1.5") belongs to the alert just above it.
                    for nxt in own[i + 1:i + 4]:
                        if (et(nxt['timestamp_utc']) - posted).total_seconds() > 120:
                            break
                        if s := STANDALONE_STOP.match(nxt.get('text') or ''):
                            row['stop'] = float(s.group(1)); sources.append(nxt['message_id']); break
                row.update({'alert_id': f"{m['message_id']}:{row['symbol']}:{row['strike']:g}{row['right'][0].upper()}",
                            'message_id': m['message_id'], 'channel_id': m['channel_id'],
                            'posted_at_utc': m['timestamp_utc'], 'posted_at_et': posted.isoformat(),
                            'source_message_ids': sources, **ALIAS,
                            'management': [{'message_id': r['message_id'], 'posted_at_utc': r['timestamp_utc'],
                                            'channel_id': r['channel_id'], 'text': r.get('text')}
                                           for r in replies.get(m['message_id'], [])]})
                out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--messages', type=Path, required=True, help='Staged source/messages.jsonl')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite {args.out}')
    rows = extract(read_jsonl(args.messages))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(json.dumps({'rows': len(rows), 'alerts': sum(r['kind'] == 'alert' for r in rows),
                      'ideas': sum(r['kind'] == 'idea' for r in rows)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
