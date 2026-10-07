"""A position is resized only when the broker's own records prove its size."""
from __future__ import annotations

from scripts.repair_position_sizes import orders_from_fills, proven_size


def _fill(order_id, side, qty, when, price=4.0):
    return {"order_id": order_id, "side": side, "qty": str(qty), "price": str(price),
            "transaction_time": when}


def _size(st, fills, broker_qty):
    return proven_size(st, orders_from_fills(fills), broker_qty)


def test_a_partial_first_reading_is_proven_from_the_entry_order():
    # MYGN 2026-09-22: ordered 1,264, state kept the 51 read a second later.
    st = {"route": "equity", "symbol": "MYGN", "shares": 51.0, "remaining_qty": 51.0,
          "entry_filled_qty": 51.0, "entry_submitted_at": "2026-09-22T18:25:09.561415932Z"}
    fills = [_fill("o1", "buy", 51, "2026-09-22T18:25:10Z"),
             _fill("o1", "buy", 1213, "2026-09-22T18:25:14Z")]

    size, evidence = _size(st, fills, 1264.0)

    assert size == 1264.0 and "filled 1264" in evidence


def test_a_planned_size_with_no_entry_clock_is_proven_from_the_only_buy():
    # ACVA: Meta planned 480, the governed path filled 180.
    st = {"route": "equity", "symbol": "ACVA", "shares": 480,
          "entry_bar": "2026-09-11 18:00:00+00:00"}

    assert _size(st, [_fill("o1", "buy", 180, "2026-09-14T13:35:40Z")], 180.0)[0] == 180.0


def test_a_sell_before_the_entry_belongs_to_an_earlier_position():
    # BW: a prior position was closed on 09-23, then 847 bought on 09-28.
    st = {"route": "equity", "symbol": "BW", "shares": 455.0, "remaining_qty": 455.0,
          "entry_filled_qty": 455.0, "entry_submitted_at": "2026-09-28T18:25:13Z"}
    fills = [_fill("old", "sell", 536, "2026-09-23T18:25:00Z"),
             _fill("o1", "buy", 847, "2026-09-28T18:25:14Z")]

    assert _size(st, fills, 847.0)[0] == 847.0


def test_a_sell_after_the_entry_comes_off_the_size():
    # ABCL: Meta bought 133; another module then sold 123 of them.
    st = {"route": "equity", "symbol": "ABCL", "shares": 355,
          "entry_bar": "2026-09-28 18:00:00+00:00"}
    fills = [_fill("o1", "buy", 133, "2026-09-29T13:35:00Z"),
             _fill("x", "sell", 123, "2026-09-29T18:32:00Z")]

    assert _size(st, fills, 10.0)[0] == 10.0


def test_an_entry_older_than_the_journal_uses_the_recorded_fill():
    # FEIM: bought 76 before the journal began, trimmed 12, state said 52.
    st = {"route": "equity", "symbol": "FEIM", "shares": 52.0, "remaining_qty": 52.0,
          "entry_filled_qty": 76.0, "entry_submitted_at": "2026-09-08T18:33:11Z"}

    assert _size(st, [_fill("t", "sell", 12, "2026-09-17T13:43:00Z")], 64.0)[0] == 64.0


def test_arithmetic_that_does_not_reach_the_broker_quantity_is_refused():
    st = {"route": "equity", "symbol": "MYGN", "shares": 51.0, "remaining_qty": 51.0,
          "entry_filled_qty": 51.0, "entry_submitted_at": "2026-09-22T18:25:09Z"}

    size, why = _size(st, [_fill("o1", "buy", 1264, "2026-09-22T18:25:10Z")], 900.0)

    assert size is None and "!= broker 900" in why


def test_a_second_buy_after_the_entry_is_refused():
    st = {"route": "equity", "symbol": "MYGN", "shares": 51.0, "remaining_qty": 51.0,
          "entry_filled_qty": 51.0, "entry_submitted_at": "2026-09-22T18:25:09Z"}
    fills = [_fill("o1", "buy", 600, "2026-09-22T18:25:10Z"),
             _fill("o2", "buy", 664, "2026-09-25T18:25:10Z")]

    assert _size(st, fills, 1264.0) == (None, "another buy follows the entry")


def test_a_legacy_entry_with_two_candidate_buys_is_refused():
    st = {"route": "equity", "symbol": "ACVA", "shares": 480,
          "entry_bar": "2026-09-11 18:00:00+00:00"}
    fills = [_fill("o1", "buy", 100, "2026-09-14T13:35:40Z"),
             _fill("o2", "buy", 80, "2026-09-15T13:35:40Z")]

    assert _size(st, fills, 180.0)[0] is None


def test_a_partial_reading_with_no_journal_entry_cannot_be_proven():
    st = {"route": "equity", "symbol": "OLD", "shares": 51.0, "remaining_qty": 51.0,
          "entry_filled_qty": 51.0, "entry_submitted_at": "2026-09-01T18:25:09Z"}

    assert _size(st, [], 1264.0)[0] is None


# --- symbols two books claim -------------------------------------------------

import pytest

from scripts.repair_position_sizes import parse_assignments, verify_assignment

HTF, META = "multi_ticker_swing_htf", "meta_ranker"


def _orders(**by_id):
    return lambda order_id: by_id.get(order_id)


def _buy(symbol, qty):
    return {"symbol": symbol, "side": "buy", "filled_qty": str(qty), "status": "filled"}


def test_both_claimants_keep_what_their_own_orders_filled():
    # TTAN: HTF bought 91, Meta's governed order was cut to 34. Broker holds 125.
    shares = parse_assignments([f"TTAN={HTF}:91@o-htf,{META}:34@o-meta"])["TTAN"]

    verify_assignment("TTAN", shares, [META, HTF], 125.0,
                      _orders(**{"o-htf": _buy("TTAN", 91), "o-meta": _buy("TTAN", 34)}))


def test_a_claim_with_no_order_behind_it_can_be_given_nothing():
    # TECX: Meta's entry was logged `id=?`; the 164 shares are HTF's.
    shares = parse_assignments([f"TECX={HTF}:164@o-htf,{META}:0"])["TECX"]

    assert shares[META] == (0.0, None)
    verify_assignment("TECX", shares, [META, HTF], 164.0, _orders(**{"o-htf": _buy("TECX", 164)}))


@pytest.mark.parametrize("spec, held, orders, why", [
    (f"TECX={HTF}:164@o-htf", 164.0, {"o-htf": _buy("TECX", 164)}, "name every claimant"),
    (f"TECX={HTF}:100@o-htf,{META}:0", 164.0, {"o-htf": _buy("TECX", 100)}, "do not add up"),
    (f"TECX={HTF}:164@o-htf,{META}:0", 164.0, {"o-htf": _buy("TECX", 120)}, "not a filled buy"),
    (f"TECX={HTF}:164@o-htf,{META}:0", 164.0, {"o-htf": _buy("EH", 164)}, "not a filled buy"),
    (f"TECX={HTF}:164@o-htf,{META}:0", 164.0, {}, "not a filled buy"),
])
def test_an_assignment_the_broker_does_not_bear_out_is_refused(spec, held, orders, why):
    shares = parse_assignments([spec])["TECX"]

    with pytest.raises(SystemExit, match=why):
        verify_assignment("TECX", shares, [META, HTF], held, _orders(**orders))


def test_a_quantity_with_no_order_named_is_refused():
    with pytest.raises(SystemExit, match="no entry order"):
        parse_assignments([f"TECX={HTF}:164,{META}:0"])
