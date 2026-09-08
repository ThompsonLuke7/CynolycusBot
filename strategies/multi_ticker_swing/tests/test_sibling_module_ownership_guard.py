"""Regression: Swing's broker reconciliation must never adopt or flatten a
position a SIBLING module's own managed state currently claims.

2026-07-21 incident: HTF Swing bought 100 FIG shares as a real, intentional
entry. Swing's own "assigned equity" detector found those same shares in the
shared broker account, misdiagnosed them as an accidental option-exercise
assignment (Swing's universe happens to also include FIG), and auto-sold
them -- closing HTF's legitimate position without HTF's own exit logic ever
being consulted. Separately, Swing's broker-position-adoption path (no
ownership check at all) could just as easily have adopted a sibling's open
OPTION position as its own. Both paths now consult every sibling module's own
persisted `managed` state (that's what it exists for) before touching
anything.

Ownership is read through ``core.orphan_positions.claimed_symbols`` rather than
a list kept here, because a list kept here drifted: intraday_structure was never
added to it, and on 2026-09-01 Swing adopted six of that module's option
positions and force-liquidated four as `restored_unknown_expiring` within three
minutes of entry. One reader means a module registered once is honoured by every
sibling's reconcile and by the orphan scan together.
"""
from __future__ import annotations

import json

import pytest

import core.orphan_positions as op_module
import strategies.multi_ticker_swing.live.position_manager as pm_module
from strategies.multi_ticker_swing.live.position_manager import SwingPositionManager
from strategies.multi_ticker_swing.live.universe import TickerConfig


@pytest.fixture
def sibling_books(tmp_path, monkeypatch):
    """Point every claim book the shared reader consults at an empty tmp_path.

    Without this a unit test silently reads the live repo's real state files.
    Returns a callable that installs the sibling `managed` states under test.
    """

    monkeypatch.setattr(op_module, "MANAGED_STATE_PATHS", {})
    monkeypatch.setattr(op_module, "SWING_BOOK_PATH", tmp_path / "no_swing.json")
    monkeypatch.setattr(op_module, "SPY_LIVE_RUNS_ROOT", tmp_path / "no_runs")
    monkeypatch.setattr(op_module, "INTRADAY_STRUCTURE_BOOK_PATH", tmp_path / "no_intraday.json")

    def _install(**modules):
        monkeypatch.setattr(op_module, "MANAGED_STATE_PATHS", dict(modules))

    return _install


class _FakeClient:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return self._positions


def _universe(*tickers: str) -> dict:
    return {t: None for t in tickers}


def _ticker_config(ticker: str) -> TickerConfig:
    return TickerConfig(ticker=ticker, tier=2, entry_threshold=0.60, sl_atr=4.0,
                        np_n_bars=None, np_mfe_atr=None, avg_win_pct=0.30,
                        avg_loss_pct=-0.20, profit_factor=1.4, sharpe=0.9)


def _write_sibling_state(path, managed: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"managed": managed}))


def test_assigned_equity_never_flags_a_symbol_a_sibling_currently_manages(tmp_path, sibling_books):
    htf_state = tmp_path / "htf_live_state.json"
    _write_sibling_state(htf_state, {"FIG": {"route": "equity", "symbol": "FIG", "shares": 100}})
    sibling_books(multi_ticker_swing_htf=htf_state)

    client = _FakeClient([{"symbol": "FIG", "qty": 100, "side": "long", "avg_entry_price": 21.92}])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    # Even if Swing's OWN history looks like a match (e.g. a stale cache
    # entry), the sibling veto must still win.
    detected = swing._broker_assigned_equity_positions(
        _universe("FIG"), owned_tickers={"FIG"}, sibling_owned=pm_module._sibling_module_owned_symbols(),
    )
    assert detected == []


def test_reconcile_does_not_adopt_an_option_position_a_sibling_owns(tmp_path, sibling_books):
    dealer_state = tmp_path / "dealer_ranker_live_state.json"
    _write_sibling_state(dealer_state, {"FIG": {"route": "option", "occ": "FIG260724C00022000", "contracts": 1}})
    sibling_books(dealer_ranker=dealer_state)

    client = _FakeClient([
        {"symbol": "FIG260724C00022000", "qty": 1, "side": "long", "avg_entry_price": 0.97},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.reconcile_with_broker(
        universe=_universe("FIG"),
        price_lookup=lambda t: 22.0,
        atr_lookup=lambda t: 1.0,
    )

    assert result["restored"] == 0
    assert "FIG" not in swing._positions
    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "owned_by_other_module" in reasons


def test_sync_from_broker_does_not_adopt_an_option_position_a_sibling_owns(tmp_path, sibling_books):
    meta_state = tmp_path / "meta_live_state.json"
    _write_sibling_state(meta_state, {"FIG": {"route": "option", "occ": "FIG260724C00022000", "contracts": 1}})
    sibling_books(meta_ranker=meta_state)

    client = _FakeClient([
        {"symbol": "FIG260724C00022000", "qty": 1, "side": "long", "avg_entry_price": 0.97},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.sync_from_broker(
        universe=_universe("FIG"),
        price_lookup=lambda t: 22.0,
        atr_lookup=lambda t: 1.0,
    )

    assert result["restored"] == 0
    assert "FIG" not in swing._positions
    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "owned_by_other_module" in reasons


def test_sibling_owned_symbols_is_missing_file_tolerant(tmp_path, sibling_books):
    sibling_books(meta_ranker=tmp_path / "does_not_exist.json")
    assert pm_module._sibling_module_owned_symbols() == set()


def test_sibling_owned_symbols_collects_both_equity_and_option_keys(tmp_path, sibling_books):
    state_a = tmp_path / "a.json"
    state_b = tmp_path / "b.json"
    _write_sibling_state(state_a, {"FIG": {"route": "equity", "symbol": "FIG"}})
    _write_sibling_state(state_b, {"SNDK": {"route": "option", "occ": "SNDK260814C01700000"}})
    sibling_books(meta_ranker=state_a, dealer_ranker=state_b)

    owned = pm_module._sibling_module_owned_symbols()
    assert owned == {"FIG", "SNDK260814C01700000"}


def test_reconcile_does_not_adopt_an_intraday_structure_contract(tmp_path, sibling_books,
                                                                 monkeypatch):
    """The 2026-09-01 regression, end to end.

    intraday_structure bought QQQ260901P00708000 at 10:27:05 ET. Swing's
    reconcile adopted it 31 seconds later and sold it at 10:30:08 ET as
    `restored_unknown_expiring`, booking -$18 to Swing's ledger and leaving
    intraday_structure retrying an exit for a contract it no longer held.
    """
    book = tmp_path / "open_option_positions.json"
    book.write_text(json.dumps({"open": {
        "QQQ:short:vwap_reclaim_continuation": {
            "occ": "QQQ260901P00708000", "ticker": "QQQ", "qty": 9,
            "expiry": "2026-09-01",
        },
    }}))
    monkeypatch.setattr(op_module, "INTRADAY_STRUCTURE_BOOK_PATH", book)

    client = _FakeClient([
        {"symbol": "QQQ260901P00708000", "qty": 9, "side": "long", "avg_entry_price": 1.02},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.reconcile_with_broker(
        universe=_universe("QQQ"),
        price_lookup=lambda t: 708.0,
        atr_lookup=lambda t: 4.0,
    )

    assert result["restored"] == 0
    assert "QQQ" not in swing._positions
    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "owned_by_other_module" in reasons


def test_a_sibling_contract_claim_does_not_block_a_different_contract(tmp_path,
                                                                      sibling_books,
                                                                      monkeypatch):
    """Claims are per-CONTRACT, so siblings do not fence each other off a ticker.

    Swing's veto tests the option symbol *and* the bare ticker, so if
    intraday_structure claimed the underlying "QQQ" rather than the contract it
    holds, one intraday QQQ position would make every QQQ option unadoptable by
    Swing — including a QQQ contract Swing itself opened and then lost the claim
    to. Per-contract claims are what keep the shared account from behaving like
    a per-ticker lock.
    """
    book = tmp_path / "open_option_positions.json"
    book.write_text(json.dumps({"open": {
        "QQQ:short:vwap": {"occ": "QQQ260901P00708000", "ticker": "QQQ", "qty": 9},
    }}))
    monkeypatch.setattr(op_module, "INTRADAY_STRUCTURE_BOOK_PATH", book)

    # A DIFFERENT QQQ contract, which is nobody's.
    client = _FakeClient([
        {"symbol": "QQQ260918C00720000", "qty": 4, "side": "long", "avg_entry_price": 3.10},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    # A real config: unlike the vetoed cases above, this one is expected to be
    # adopted, so it reaches the restore path that reads the ticker's params.
    result = swing.reconcile_with_broker(
        universe={"QQQ": _ticker_config("QQQ")},
        price_lookup=lambda t: 715.0,
        atr_lookup=lambda t: 4.0,
    )

    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "owned_by_other_module" not in reasons
    assert result["restored"] == 1, "Swing must still adopt a contract nobody claims"


def test_reconcile_never_inflates_qty_to_a_netted_sibling_position(tmp_path, sibling_books):
    """Alpaca nets option positions by symbol across the shared account.

    If a 4H module is also long a contract Swing holds, the broker reports the
    combined size. Adopting that total would make Swing's exit sell the
    sibling's contracts too. Swing never scales into a position, so its own qty
    is the ceiling.
    """
    client = _FakeClient([
        {"symbol": "TGT260911C00157500", "qty": 9, "side": "long", "avg_entry_price": 4.79},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    from strategies.multi_ticker_swing.live.position_manager import SwingPosition
    from datetime import datetime, timezone
    swing._positions["TGT"] = SwingPosition(
        ticker="TGT", direction=1, entry_price=157.0,
        entry_time=datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc),
        atr_at_entry=2.0, qty=4, option_symbol="TGT260911C00157500",
        config=_ticker_config("TGT"),
    )

    result = swing.reconcile_with_broker(
        universe={"TGT": _ticker_config("TGT")},
        price_lookup=lambda t: 158.0,
        atr_lookup=lambda t: 2.0,
    )

    assert swing._positions["TGT"].qty == 4, "Swing absorbed a sibling's 5 contracts"
    assert result["qty_updates"] == []


def test_reconcile_still_shrinks_qty_when_the_broker_holds_fewer(tmp_path, sibling_books):
    """Capping up must not break the case the sync exists for."""
    client = _FakeClient([
        {"symbol": "TGT260911C00157500", "qty": 2, "side": "long", "avg_entry_price": 4.79},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    from strategies.multi_ticker_swing.live.position_manager import SwingPosition
    from datetime import datetime, timezone
    swing._positions["TGT"] = SwingPosition(
        ticker="TGT", direction=1, entry_price=157.0,
        entry_time=datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc),
        atr_at_entry=2.0, qty=6, option_symbol="TGT260911C00157500",
        config=_ticker_config("TGT"),
    )

    swing.reconcile_with_broker(
        universe={"TGT": _ticker_config("TGT")},
        price_lookup=lambda t: 158.0,
        atr_lookup=lambda t: 2.0,
    )
    assert swing._positions["TGT"].qty == 2


# "Could not read a sibling's book" must stop the adoption, not licence it
# (2026-09-02 11:50: open_positions.json read mid-write).


def test_an_unreadable_sibling_book_blocks_adoption_rather_than_allowing_it(
    tmp_path, sibling_books
):
    """The most dangerous answer this guard can give is a confident empty set.

    It used to swallow every failure and return one, which reads as "no module
    owns anything" — and the very next thing the reconcile does with an unowned
    option position is adopt it, then force-exit a same-day expiry minutes
    later. That is how six Intraday Structure contracts were taken and four
    liquidated on 2026-09-01.
    """

    meta_state = tmp_path / "meta_live_state.json"
    meta_state.parent.mkdir(parents=True, exist_ok=True)
    meta_state.write_text("{not json")  # a read that landed mid-write
    sibling_books(meta_ranker=meta_state)

    assert pm_module._sibling_module_owned_symbols() is None

    client = _FakeClient([
        {"symbol": "FIG260724C00022000", "qty": 1, "side": "long", "avg_entry_price": 0.97},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.sync_from_broker(
        universe=_universe("FIG"),
        price_lookup=lambda t: 22.0,
        atr_lookup=lambda t: 1.0,
    )

    assert result["restored"] == 0
    assert "FIG" not in swing._positions
    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "sibling_ownership_unknown" in reasons


def test_a_readable_empty_book_still_permits_adoption(tmp_path, sibling_books):
    """Failing closed must not mean never adopting anything.

    An absent book is a real answer — the module has written no positions — and
    swing still restores its own untracked holdings from it.
    """

    sibling_books(meta_ranker=tmp_path / "does_not_exist.json")

    client = _FakeClient([
        {"symbol": "FIG260724C00022000", "qty": 1, "side": "long", "avg_entry_price": 0.97},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.sync_from_broker(
        universe={"FIG": _ticker_config("FIG")},
        price_lookup=lambda t: 22.0,
        atr_lookup=lambda t: 1.0,
    )

    assert result["restored"] == 1
    assert "FIG" in swing._positions
