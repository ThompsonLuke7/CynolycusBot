"""Classify caller screenshots from OCR and parse the structured ones.

OCR is a machine reading of preserved attachments, so every parsed value keeps
its attachment ID, source message and OCR confidence; nothing here asserts a
fill. Types:

- ``em_table``: "Stock Ticker | Upside Range | Downside Range" expected-move tables;
- ``watchlist_card``: cards carrying "GEX King"/"VEX King" per contract;
- ``recap_card``: "PREMIUM ALERTS RECAP" max-gain cards;
- ``cycle_panel``: TradingView charts with the private "CYCLE ORB" panel;
- ``option_quote``: Robinhood option quote pages (phone clock, underlying, contract, mark);
  these show a price at a moment, not that the caller holds the position;
- ``position``: position detail, fill notification or account screens;
- ``gex_tool``: third-party GEX/VEX level screens (king, call/put wall, zero gamma);
- ``broker_screen``: other brokerage screens;
- ``other``: charts and anything else, left for visual review.

Rows are rebuilt from OCR box geometry (shared vertical band), not reading order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from scripts.discord_ledger.build import read_jsonl, write_jsonl

MONEY = re.compile(r'^\$?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)$')
TICKER = re.compile(r'^\$?([A-Z]{1,5}(?:\.[A-Z])?)$')
CONTRACT = re.compile(r'\b([A-Z]{1,5})\s+(\d+(?:\.\d+)?)\s*([CcPp])\b\s*(\d{1,2}/\d{1,2})?')
BROKER_WORDS = ('avg cost', 'average cost', 'total return', 'market value', "today's return", 'contracts', 'equity',
                'avg price', 'unrealized', 'p/l', 'filled', 'limit', 'buy to open', 'sell to close', 'positions')
PANEL_KEYS = {'cycle / trend': 'cycle_trend', 'due low': 'due_low', 'due high': 'due_high', 'today': 'today',
              '5-day sma': 'sma5', '21 ema': 'ema21', '55 sma': 'sma55', '34 sma': 'sma34',
              '5 min divergence': 'div_5m', '30 min divergence': 'div_30m', '2hr divergence': 'div_2h',
              'daily divergence': 'div_d', 'weekly divergence': 'div_w', 'd em': 'd_em'}


def classify(text: str) -> str:
    low = text.lower()
    squeezed = re.sub(r'\s+', '', low)  # OCR often merges or splits header words
    if 'upsiderange' in squeezed and 'downsiderange' in squeezed:
        return 'em_table'
    if any(w in low for w in ('call wall', 'put wall', 'net gex', 'net vex')) or re.search(r'\bgex\b.*\bchart\b', low):
        return 'gex_tool'
    if 'gex king' in low or 'vex king' in low:
        return 'watchlist_card'
    if 'alerts recap' in low or 'max gain' in low:
        return 'recap_card'
    if 'cycle orb' in low or 'cycle / trend' in low or 'divergence' in low and 'sma' in low:
        return 'cycle_panel'
    if parse_option_quote(text):
        return 'option_quote'
    if any(w in low for w in ('average cost', 'date bought', 'total return up', 'buying power', 'has gained more than')):
        return 'position'
    if sum(w in low for w in BROKER_WORDS) >= 2:
        return 'broker_screen'
    return 'other'


def rows_by_band(lines: list[dict], tolerance: float = 0.6) -> list[list[dict]]:
    """Group OCR boxes whose vertical centers fall within ``tolerance`` box heights."""
    boxes = []
    for line in lines:
        ys = [p[1] for p in line['box']]
        xs = [p[0] for p in line['box']]
        boxes.append({**line, 'cy': sum(ys) / 4, 'h': max(ys) - min(ys), 'x': min(xs)})
    boxes.sort(key=lambda b: b['cy'])
    rows: list[list[dict]] = []
    for b in boxes:
        if rows and abs(rows[-1][-1]['cy'] - b['cy']) <= tolerance * max(b['h'], rows[-1][-1]['h'], 1):
            rows[-1].append(b)
        else:
            rows.append([b])
    return [sorted(r, key=lambda b: b['x']) for r in rows]


def money(text: str) -> float | None:
    t = text.strip().replace(' ', '')
    if t[:1] in ('$', 'S') and t[1:2].isdigit():  # OCR often reads "$" as "S"
        t = t[1:]
    m = MONEY.match(t)
    return float(m.group(1).replace(',', '')) if m else None


def parse_em_table(lines: list[dict]) -> list[dict]:
    out = []
    for row in rows_by_band(lines):
        tick = TICKER.match(row[0]['text'].strip())
        if not tick or tick.group(1) in {'STOCK', 'TICKER'}:
            continue
        values = [(money(b['text']), b['confidence']) for b in row[1:]]
        values = [(v, c) for v, c in values if v is not None]
        bad = [b['text'] for b in row[1:] if money(b['text']) is None]
        if len(values) == 2 and values[0][0] > values[1][0]:
            out.append({'symbol': tick.group(1), 'upper': values[0][0], 'lower': values[1][0],
                        'min_confidence': min(row[0]['confidence'], *(c for _, c in values))})
        else:
            out.append({'symbol': tick.group(1), 'unparsed_cells': [b['text'] for b in row[1:]], 'error_cells': bad})
    return out


def parse_watchlist_card(lines: list[dict]) -> list[dict]:
    out = []
    for row in rows_by_band(lines):
        joined = ' '.join(b['text'] for b in row)
        m = CONTRACT.search(joined.upper())
        g = re.search(r'GEX\s*King\s*\$?(\d+(?:\.\d+)?)', joined, re.I)
        v = re.search(r'VEX\s*King\s*\$?(\d+(?:\.\d+)?)', joined, re.I)
        if m and (g or v):
            out.append({'symbol': m.group(1), 'strike': float(m.group(2)), 'right': 'call' if m.group(3).upper() == 'C' else 'put',
                        'expiry_md': m.group(4), 'gex_king': float(g.group(1)) if g else None,
                        'vex_king': float(v.group(1)) if v else None, 'row_text': joined})
    return out


QUOTE_UNDERLYING = re.compile(r'\b([A-Z]{1,5})\s*\$?\s*([\d,]+\.\d{2})\s*\(\s*([+-]?[\d.]+)%\)')
QUOTE_CONTRACT = re.compile(r'\$?\s*([\d,]+(?:\.\d+)?)\s*(Call|Put)\s*\|?\s*\$\s*([\d,]*\.\d{2})', re.I)
QUOTE_CLOCK = re.compile(r'^\s*(\d{1,2}):(\d{2})\b')


def parse_option_quote(text: str) -> dict | None:
    """Robinhood option page: '6:32 | View SPY | SPY $767.92 (+0.61%) | SPY $768 Call | $2.54 | $0.92 (56.79%)'.

    The contract symbol is taken from the underlying line (OCR often garbles it), the expiry is
    not on the page, and the unsigned day change is not used.
    """
    flat = text.replace('\n', ' | ')
    und, con = QUOTE_UNDERLYING.search(flat), QUOTE_CONTRACT.search(flat)
    if not (und and con) or con.start() < und.start():
        return None
    clock = QUOTE_CLOCK.match(flat)
    low = flat.lower()
    return {'underlying': und.group(1), 'underlying_price': float(und.group(2).replace(',', '')),
            'underlying_change_pct': float(und.group(3)), 'strike': float(con.group(1).replace(',', '')),
            'right': con.group(2).lower(), 'mark': float(con.group(3).replace(',', '')),
            'phone_clock': f'{clock.group(1)}:{clock.group(2)}' if clock else None,
            'session_tag': next((t for t in ('pre-market', 'overnight', 'extended hours') if t in low), 'regular_or_unlabeled')}


def parse_cycle_panel(lines: list[dict]) -> dict:
    panel = {}
    for row in rows_by_band(lines):
        label = row[0]['text'].strip().lower()
        for key, name in PANEL_KEYS.items():
            if label == key and len(row) > 1:
                panel[name] = ' '.join(b['text'] for b in row[1:])
    return panel


def extract(ocr_rows: list[dict], messages: dict[str, dict]) -> list[dict]:
    out, seen = [], set()
    for o in ocr_rows:
        # Reused and fresh OCR can both cover one attachment; keep the first reading.
        if 'error' in o or o['attachment_id'] in seen:
            continue
        seen.add(o['attachment_id'])
        text = o.get('ocr_text') or ''
        kind = classify(text)
        msg = messages.get(o['message_id'], {})
        rec = {'attachment_id': o['attachment_id'], 'message_id': o['message_id'], 'author_id': o.get('author_id'),
               'channel_id': o.get('channel_id'), 'posted_at_utc': msg.get('timestamp_utc'), 'caption': msg.get('text'),
               'local_path': o['local_path'], 'sha256': o.get('sha256'), 'kind': kind,
               'interpretation': 'ocr_parse_of_caller_image_not_a_fill'}
        lines = o.get('ocr_lines') or []
        if kind == 'em_table':
            rec['em_rows'] = parse_em_table(lines)
        elif kind == 'watchlist_card':
            rec['king_rows'] = parse_watchlist_card(lines)
        elif kind == 'cycle_panel':
            rec['panel'] = parse_cycle_panel(lines)
        elif kind == 'option_quote':
            rec['quote'] = parse_option_quote(text)
        out.append(rec)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr', type=Path, required=True)
    parser.add_argument('--messages', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite {args.out}')
    messages = {m['message_id']: m for m in read_jsonl(args.messages)}
    rows = extract(read_jsonl(args.ocr), messages)
    write_jsonl(args.out, rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r['kind']] = counts.get(r['kind'], 0) + 1
    print(json.dumps({'images': len(rows), 'kinds': counts}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
