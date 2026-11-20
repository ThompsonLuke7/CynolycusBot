from __future__ import annotations

from scripts.discord_ledger.extract_image_evidence import classify, parse_cycle_panel, parse_em_table, parse_watchlist_card


def box(text: str, x: float, y: float, conf: float = 0.99) -> dict:
    return {'text': text, 'confidence': conf, 'box': [[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]]}


def test_classifies_each_image_family() -> None:
    assert classify('Stock Ticker Upside Range Downside Range SPY') == 'em_table'
    assert classify('StockTicker UpsideRange Downside\nRange SPY') == 'em_table'
    assert classify('IBM 260C 10/30 GEX King 260 VEX King 260') == 'watchlist_card'
    assert classify('PREMIUM ALERTS RECAP Max Gain') == 'recap_card'
    assert classify('CYCLE ORB Cycle / trend 26d BEAR') == 'cycle_panel'
    assert classify('Avg cost 1.32 Total return +45%') == 'broker_screen'
    assert classify('a chart') == 'other'


def test_em_rows_are_rebuilt_from_geometry_and_dollar_s_misreads() -> None:
    lines = [box('Stock Ticker', 0, 0), box('Upside Range', 200, 2), box('Downside Range', 400, 1),
             box('$750.61', 400, 41), box('SPY', 0, 40), box('S775.89', 200, 39),
             box('SMH', 0, 80), box('#VALUE!', 200, 80), box('#VALUE!', 400, 80)]
    rows = parse_em_table(lines)
    assert rows[0] == {'symbol': 'SPY', 'upper': 775.89, 'lower': 750.61, 'min_confidence': 0.99}
    assert rows[1]['symbol'] == 'SMH' and rows[1]['error_cells'] == ['#VALUE!', '#VALUE!']


def test_option_quote_pages_are_parsed_without_trusting_the_garbled_contract_symbol() -> None:
    text = '12:58\n36\nView IBM\nIBM $258.29(+2.00%)\nIBMS\n$257.5 Call\n$2.48\n$1.06 (74.65%)\nToday'
    assert classify(text) == 'option_quote'
    from scripts.discord_ledger.extract_image_evidence import parse_option_quote
    assert parse_option_quote(text) == {'underlying': 'IBM', 'underlying_price': 258.29, 'underlying_change_pct': 2.0, 'strike': 257.5,
                                        'right': 'call', 'mark': 2.48, 'phone_clock': '12:58', 'session_tag': 'regular_or_unlabeled'}
    spx = '9:02\nView SPX\nSPX 7,691.75 (-0.68%)\nSPXW 7,670 Put\n$5.85\n$2.15 (58.11%)\nExtended hours'
    q = parse_option_quote(spx)
    assert (q['underlying_price'], q['strike'], q['right'], q['mark'], q['session_tag']) == (7691.75, 7670.0, 'put', 5.85, 'extended hours')


def test_position_and_gex_tool_screens() -> None:
    assert classify('Average cost\n$1.53\nDate bought\n10/15') == 'position'
    assert classify('NET GEX NET VEX KING CALL WALL PUT WALL ZERO') == 'gex_tool'


def test_watchlist_and_panel_rows() -> None:
    card = [box('IBM 275C 11/20', 0, 0), box('GEX King', 200, 0), box('275', 260, 0), box('VEX King', 320, 1), box('275', 380, 1)]
    assert parse_watchlist_card(card)[0] == {'symbol': 'IBM', 'strike': 275.0, 'right': 'call', 'expiry_md': '11/20',
                                             'gex_king': 275.0, 'vex_king': 275.0, 'row_text': 'IBM 275C 11/20 GEX King 275 VEX King 275'}
    panel = parse_cycle_panel([box('Due low', 0, 0), box('15d', 200, 0), box('D EM', 0, 40), box('157.24 / 146.34', 200, 40)])
    assert panel == {'due_low': '15d', 'd_em': '157.24 / 146.34'}
