"""An unpriced close is settled only by what the broker did with its order."""
from __future__ import annotations

from scripts.repair_unpriced_closes import classify

OID = "1ebea68e-0000-4000-8000-000000000001"
COVERED = ("2026-09-14T13:35:00Z", "2026-10-05T19:57:00Z")


def _row(**kw):
    base = {"ts": "2026-09-28T14:51:18+00:00", "order_symbol": "SPY260928C00765000",
            "qty": 1.0, "entry_avg_price": 1.08, "exit_fill_price": None,
            "realized_pnl": None, "order_id": OID}
    return {**base, **kw}


def _fill(qty, price, side="sell", symbol="SPY260928C00765000"):
    return {"order_id": OID, "symbol": symbol, "side": side, "qty": str(qty), "price": str(price)}


def test_an_order_that_filled_later_prices_the_row():
    verdict, fixed = classify(_row(), [_row()], [_fill(1, 1.17)], COVERED)

    assert verdict == "price"
    assert (fixed["exit_fill_price"], fixed["realized_pnl"]) == (1.17, 9.0)


def test_a_share_row_is_priced_per_share():
    row = _row(order_symbol="METC", qty=100.0, entry_avg_price=11.36)

    _, fixed = classify(row, [row], [_fill(100, 10.72, symbol="METC")], COVERED)

    assert fixed["realized_pnl"] == -64.0


def test_an_order_that_never_filled_is_a_duplicate_of_the_later_close():
    row = _row(ts="2026-10-01T14:10:15+00:00")
    later = _row(ts="2026-10-01T14:38:35+00:00", realized_pnl=-78.0, order_id="other")

    assert classify(row, [row, later], [], COVERED) == ("duplicate", None)


def test_an_unfilled_order_with_no_later_close_is_left_alone():
    row = _row()

    assert classify(row, [row], [], COVERED) == ("leave", None)


def test_rows_outside_the_journal_or_without_a_real_order_id_are_left_alone():
    old = _row(ts="2026-09-02T14:58:20+00:00")
    no_id = _row(order_id="?")

    assert classify(old, [old], [], COVERED)[0] == "leave"
    assert classify(no_id, [no_id], [], COVERED)[0] == "leave"


def test_fills_that_do_not_match_the_row_are_not_used():
    row = _row()

    assert classify(row, [row], [_fill(2, 1.17)], COVERED)[0] == "leave"          # quantity
    assert classify(row, [row], [_fill(1, 1.17, side="buy")], COVERED)[0] == "leave"
    assert classify(row, [row], [_fill(1, 1.17, symbol="SPY")], COVERED)[0] == "leave"


def test_a_priced_row_is_never_touched():
    row = _row(realized_pnl=9.0, exit_fill_price=1.17)

    assert classify(row, [row], [_fill(1, 2.00)], COVERED)[0] == "leave"
