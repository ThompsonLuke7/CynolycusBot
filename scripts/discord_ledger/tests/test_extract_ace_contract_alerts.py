from __future__ import annotations

from datetime import datetime

from scripts.discord_ledger.extract_ace_contract_alerts import ACE_SUCCESSOR_ID, NEW_ALERTS_CHANNEL_ID, ET, extract, parse_line

POSTED = datetime(2026, 10, 29, 14, 41, tzinfo=ET)


def fields(rows):
    return [(r['symbol'], r['strike'], r['right'], r['expiry'], r['premium'], r['stop'], r['kind']) for r in rows]


def test_parses_premium_stop_and_literal_expiry_in_any_order() -> None:
    assert fields(parse_line('IWM 280 p 1.2 SL 1.0 11/4', POSTED)) == [('IWM', 280.0, 'put', '2026-11-04', 1.2, 1.0, 'alert')]
    assert fields(parse_line('spy 765 p 11/2 2.2 SL 2.0', POSTED)) == [('SPY', 765.0, 'put', '2026-11-02', 2.2, 2.0, 'alert')]
    assert fields(parse_line('BRK.b 550 c 10/30 .64 SL .5', POSTED)) == [('BRK.B', 550.0, 'call', '2026-10-30', 0.64, 0.5, 'alert')]


def test_relative_dte_resolves_to_the_posting_session() -> None:
    rows = parse_line('SPY 770 c 0DTE .55 lotto size SL .45', POSTED)
    assert fields(rows) == [('SPY', 770.0, 'call', '2026-10-29', 0.55, 0.45, 'alert')]
    assert rows[0]['flags'] == ['lotto']
    assert parse_line('GOOGL 352.5 call 1DTE 1.5', POSTED)[0]['expiry'] == '2026-10-30'


def test_plans_without_premium_are_ideas_and_both_or_legs_are_kept() -> None:
    rows = parse_line('CRM 247.5 call 10/23 OR CRM 245 call 10/23', POSTED)
    assert fields(rows) == [('CRM', 247.5, 'call', '2026-10-23', None, None, 'idea'),
                            ('CRM', 245.0, 'call', '2026-10-23', None, None, 'idea')]


def test_result_claims_and_prose_prices_are_not_alerts() -> None:
    assert parse_line('spy 760 put went 160%!!! called it beautifully', POSTED) == []
    assert parse_line('LITE looks primed looking at 950 calls for 10/23', POSTED) == []
    assert parse_line('tmw when market opens you getting more on our 4 put swings.', POSTED) == []


def test_fills_on_an_existing_position_are_management_not_new_alerts() -> None:
    assert parse_line('Sold half of MRK 155 calls at 2.02 SL set at 1.8 will add more', POSTED) == []


def test_flags_on_another_line_of_the_same_message_apply() -> None:
    text = 'GOOGL 352.5 call 1DTE 1.5\nSIZE FOR ZERO @everyone'
    assert parse_line(text.splitlines()[0], POSTED, text)[0]['flags'] == ['size for zero']


def message(mid, ts, text, reply=None):
    return {'message_id': mid, 'author_id': ACE_SUCCESSOR_ID, 'channel_id': NEW_ALERTS_CHANNEL_ID,
            'timestamp_utc': ts, 'text': text, 'reply_to_message_id': reply}


def test_links_standalone_stop_and_reply_management_by_message_id_only() -> None:
    rows = extract([message('1', '2026-10-29T18:41:05Z', 'QQQ 700 p 1.8 11/2'),
                    message('2', '2026-10-29T18:41:15Z', 'Stop 1.5'),
                    message('3', '2026-10-29T20:10:57Z', 'QQQ put swings over $500 AH', reply='1'),
                    message('4', '2026-10-29T20:11:00Z', 'QQQ puts gone')])
    assert len(rows) == 1
    assert rows[0]['stop'] == 1.5 and rows[0]['source_message_ids'] == ['1', '2']
    assert [m['message_id'] for m in rows[0]['management']] == ['3']
    assert rows[0]['alias_status'].startswith('inferred')
