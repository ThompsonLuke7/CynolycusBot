"""Saved exit evidence is only re-attached to the position it was saved from.

intraday_structure keys its book by setup id and the same setup trades again
another day. Its records carry no `entry_bar`, and recovery compared only that,
so None == None passed for any two trades under one id. On 2026-10-05 a
BAC261009C00054000 call bought at 11:36 was handed the 2026-10-01 exit of
BAC261002C00054000, declared closed and dropped from the book while the account
still held it. GOOGL and both QQQ setups went the same way that week, and the
30m swing then adopted and sold contracts that looked unowned.
"""
from __future__ import annotations

from core.order_reconciliation import recover_pending_exits, same_position, save_evidence

SETUP = "BAC:long:vwap_reclaim_continuation"


def _evidence(root, module, key, state, *, order_id, complete=True):
    saved = {**state, "exit_pending": {"order_id": order_id, "qty": 20.0, "status": "filled"}}
    save_evidence(root, module, order_id, {"owner": key, "symbol": state.get("occ"),
                                           "state": saved, "complete": complete})


def test_an_earlier_trade_under_the_same_setup_id_is_not_recovered(tmp_path):
    _evidence(tmp_path, "intraday_structure", SETUP,
              {"occ": "BAC261002C00054000", "entry_order_id": "entry-oct-1", "qty": 20.0},
              order_id="exit-oct-1")
    todays = {"occ": "BAC261009C00054000", "entry_order_id": "entry-oct-5", "qty": 14.0}
    managed = {SETUP: todays}

    recover_pending_exits(tmp_path, "intraday_structure", managed)

    assert "exit_pending" not in managed[SETUP]


def test_an_exit_saved_from_this_position_is_recovered(tmp_path):
    position = {"occ": "BAC261009C00054000", "entry_order_id": "entry-oct-5", "qty": 14.0}
    _evidence(tmp_path, "intraday_structure", SETUP, position, order_id="exit-oct-5",
              complete=False)
    managed = {SETUP: dict(position)}

    recover_pending_exits(tmp_path, "intraday_structure", managed)

    assert managed[SETUP]["exit_pending"]["order_id"] == "exit-oct-5"


def test_a_4h_position_is_recovered_by_its_entry_bar(tmp_path):
    position = {"route": "equity", "symbol": "MYGN", "shares": 1264.0,
                "entry_bar": "2026-09-22 14:00:00+00:00"}
    _evidence(tmp_path, "multi_ticker_swing_htf", "MYGN", position, order_id="exit-1",
              complete=False)
    managed = {"MYGN": dict(position)}

    recover_pending_exits(tmp_path, "multi_ticker_swing_htf", managed)

    assert managed["MYGN"]["exit_pending"]["order_id"] == "exit-1"


def test_a_re_entry_on_a_later_bar_is_a_different_position(tmp_path):
    _evidence(tmp_path, "momentum_expansion", "MXL",
              {"route": "option", "occ": "MXL261016C00095000",
               "entry_bar": "2026-09-21 14:00:00+00:00"}, order_id="exit-sep-21")
    managed = {"MXL": {"route": "option", "occ": "MXL261016C00095000",
                       "entry_bar": "2026-09-25 14:00:00+00:00"}}

    recover_pending_exits(tmp_path, "momentum_expansion", managed)

    assert "exit_pending" not in managed["MXL"]


def test_two_records_with_no_identity_at_all_do_not_match():
    assert same_position({"occ": "X261009C00001000"}, {"occ": "X261009C00001000"}) is False


def test_an_exit_already_reconciled_is_not_attached_again(tmp_path):
    position = {"occ": "BAC261009C00054000", "entry_order_id": "entry-oct-5"}
    _evidence(tmp_path, "intraday_structure", SETUP, position, order_id="exit-oct-5")
    managed = {SETUP: {**position, "exit_reconciled_ids": ["exit-oct-5"]}}

    recover_pending_exits(tmp_path, "intraday_structure", managed)

    assert "exit_pending" not in managed[SETUP]
