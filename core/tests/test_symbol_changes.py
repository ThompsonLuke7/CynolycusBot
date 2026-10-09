"""A position follows its symbol through a broker name change.

HTF bought 529 PSKY on 2026-10-02. On 2026-10-06 the broker processed a name
change to SKYD (same CUSIP): the shares stayed in the account under the new
symbol and PSKY stopped resolving. The module looked for PSKY, found nothing,
and was one pass from dropping the claim with no ledger row, while the risk
pass reported 529 SKYD as an orphan no module managed.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from core.live_4h_exec import ExecPolicy, build_mixed_plan
from core.live_risk_pass import RiskPassConfig, evaluate_risk_exits
from core.symbol_changes import follow_symbol_changes

TODAY = dt.date(2026, 10, 7)
NOW = dt.datetime(2026, 10, 7, 9, 33, tzinfo=ZoneInfo("America/New_York"))
SKYD = {"SKYD": {"qty": 529, "avg_entry": 9.43, "current": 8.90}}
RENAME = {"id": "2bcccba6", "old_symbol": "PSKY", "new_symbol": "SKYD",
          "old_cusip": "69932A204", "new_cusip": "69932A204", "process_date": "2026-10-06"}


class _Broker:
    def __init__(self, name_changes=(RENAME,), fail=False):
        self._name_changes = list(name_changes)
        self._fail = fail
        self.asked: list[dict] = []

    def get_corporate_actions(self, **params):
        self.asked.append(params)
        if self._fail:
            raise OSError("timed out")
        return {"corporate_actions": {"name_changes": self._name_changes}}


def _psky(**extra):
    return {"route": "equity", "symbol": "PSKY", "shares": 529, "runs_held": 4,
            "bars_out": 4, "entry_bar": "2026-10-01 18:00:00+00:00",
            "entry_avg_price": 9.43, "last_mark_price": 9.86, "u_entry": 9.455,
            "u_atr": 0.2811, **extra}


def test_the_claim_moves_to_the_new_symbol():
    managed = {"PSKY": _psky(), "MYGN": {"route": "equity", "symbol": "MYGN", "shares": 10}}
    broker = _Broker()

    followed = follow_symbol_changes(broker, managed, {**SKYD, "MYGN": {"qty": 10}},
                                     module="multi_ticker_swing_htf", today=TODAY)

    assert set(managed) == {"MYGN", "SKYD"}
    st = managed["SKYD"]
    assert st["symbol"] == "SKYD" and st["shares"] == 529
    assert st["entry_bar"] == "2026-10-01 18:00:00+00:00"      # the position is unchanged
    assert st["symbol_change"] == {"old_symbol": "PSKY", "new_symbol": "SKYD",
                                   "process_date": "2026-10-06", "corporate_action_id": "2bcccba6"}
    assert followed == {"PSKY": st["symbol_change"]}
    # Asked only about the position that is missing, and only since its entry.
    assert broker.asked == [{"symbols": "PSKY", "types": "name_change",
                             "start": "2026-10-01", "end": "2026-10-07"}]


def test_a_claim_is_not_moved_onto_shares_the_broker_does_not_hold():
    for pos_info in ({}, {"SKYD": {"qty": 500}}):       # absent, or fewer than recorded
        managed = {"PSKY": _psky()}
        assert follow_symbol_changes(_Broker(), managed, pos_info,
                                     module="m", today=TODAY) == {}
        assert set(managed) == {"PSKY"} and managed["PSKY"]["symbol"] == "PSKY"


def test_a_claim_is_not_moved_onto_a_symbol_the_book_already_holds():
    managed = {"PSKY": _psky(), "SKYD": {"route": "equity", "symbol": "SKYD", "shares": 529}}
    assert follow_symbol_changes(_Broker(), managed, SKYD, module="m", today=TODAY) == {}
    assert managed["PSKY"]["symbol"] == "PSKY"


def test_an_ambiguous_or_unanswered_lookup_moves_nothing():
    other = {**RENAME, "id": "x", "new_symbol": "ELSE"}
    for broker in (_Broker(name_changes=[RENAME, other]), _Broker(name_changes=[]),
                   _Broker(fail=True), object()):
        managed = {"PSKY": _psky()}
        assert follow_symbol_changes(broker, managed, SKYD, module="m", today=TODAY) == {}
        assert set(managed) == {"PSKY"}


def test_the_broker_is_asked_once_per_position_per_day():
    managed = {"PSKY": _psky()}
    broker = _Broker(name_changes=[])

    for _ in range(5):                                   # five risk passes, same day
        follow_symbol_changes(broker, managed, {}, module="m", today=TODAY)
    assert len(broker.asked) == 1

    follow_symbol_changes(broker, managed, {}, module="m", today=TODAY + dt.timedelta(days=1))
    assert len(broker.asked) == 2


def test_positions_that_are_held_or_in_flight_or_options_are_never_looked_up():
    managed = {
        "MYGN": {"route": "equity", "symbol": "MYGN", "shares": 10},
        "NEW": {"route": "equity", "symbol": "NEW", "shares": 5, "pending_fill": True},
        "OUT": {"route": "equity", "symbol": "OUT", "shares": 5, "exit_pending": {"order_id": "o"}},
        "IOVA": {"route": "option", "occ": "IOVA261016C00015000", "contracts": 42},
    }
    broker = _Broker()
    follow_symbol_changes(broker, managed, {"MYGN": {"qty": 10}}, module="m", today=TODAY)
    assert broker.asked == []


def test_the_4h_pass_manages_a_renamed_position_instead_of_dropping_it():
    res = build_mixed_plan(
        _Broker(), targets=[], managed={"PSKY": _psky()}, pos_info=dict(SKYD),
        bar="2026-10-07 14:00:00+00:00", signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: ("skip", None, "n/a"), ref_price_fn=lambda _t: None,
        module="multi_ticker_swing_htf", verbose=False)

    assert res.dropped == {}
    assert res.new_managed["SKYD"]["symbol"] == "SKYD"
    assert res.new_managed["SKYD"]["runs_held"] == 5          # aged like any held position
    assert res.new_managed["SKYD"]["last_mark_price"] == 8.90


def test_the_risk_pass_puts_a_renamed_position_back_under_its_stop():
    res = evaluate_risk_exits(
        client=_Broker(), module="multi_ticker_swing_htf", managed={"PSKY": _psky()},
        pos_info=dict(SKYD), policy=ExecPolicy(target_notional=5000.0), now_et=NOW,
        cfg=RiskPassConfig(hard_stop=False, expiry_flatten=False),
        underlying_fn=lambda _t, at=None: (8.90, 0.28),
    )

    assert list(res.new_managed) == ["SKYD"]
    assert res.new_managed["SKYD"]["symbol"] == "SKYD"


def test_a_position_that_is_simply_gone_is_still_dropped_and_now_says_so(caplog):
    with caplog.at_level("ERROR", logger="core.live_4h_exec"):
        res = build_mixed_plan(
            _Broker(name_changes=[]), targets=[], managed={"PSKY": _psky()}, pos_info={},
            bar="2026-10-07 14:00:00+00:00", signal_audits={}, policy=ExecPolicy(),
            route_fn=lambda *a, **k: ("skip", None, "n/a"), ref_price_fn=lambda _t: None,
            module="multi_ticker_swing_htf", verbose=False)

    assert res.dropped["PSKY"]["status"] == "not_found"
    assert "NO ledger row" in caplog.text and "PSKY" in caplog.text
