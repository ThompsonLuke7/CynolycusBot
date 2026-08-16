"""Profit-protecting exits must protect actual profit, and must not cross any spread.

Both guards regress the CALM 2026-08-03 exit. CALM was a short restored at
option 2.35 basis and marked 4.54 (+93%, +$3,335 unrealized). The UNDERLYING
trail fired at 09:46 with the short still +3.56% in our favour, but by then the
put had decayed to 2.20 — the "profit-protecting" exit realized -6.38% (-$345),
and it crossed a bid 2.23 / ask 4.12 quote (59.5% of mid) to do it.

Mandatory exits (hard stop, expiry, restored-unknown, option-value) must still
fire immediately in every case below.
"""
from __future__ import annotations

from typing import Any

import pytest

from strategies.multi_ticker_swing.live import position_manager as pm


class _Pos:
    """Minimal stand-in for SwingPosition covering what the guards read."""

    def __init__(self, *, option_entry_price=2.35, option_last_price=2.20,
                 ticker="CALM", option_symbol="CALM260821P00090000"):
        self.ticker = ticker
        self.option_symbol = option_symbol
        self.option_entry_price = option_entry_price
        self.option_last_price = option_last_price

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "option_symbol": self.option_symbol}


class _Mgr(pm.SwingPositionManager):
    """Manager with the broker and event sink stubbed out."""

    def __init__(self, spread_pct: float | None = 0.05):
        self._positions = {}
        self._last_close_failure_wall = {}
        self._pending_close_orders = {}
        self._assigned_flatten_last_attempt = {}
        self._exit_spread_deferrals = {}
        self._deferred_trail_cache = {}
        self._worthless_close_abandoned = set()
        self._position_state_cache = {}
        self._sink = None
        self._dry_run = True
        self._spread_pct = spread_pct
        self.events: list[tuple[str, dict]] = []

    def _emit(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))

    def _get_contract_quote_context(self, *, symbol: str) -> dict[str, Any]:
        return {"spread_pct_mid": self._spread_pct}

    def event_kinds(self) -> list[str]:
        return [k for k, _ in self.events]


# --- option_leg_gain_pct -------------------------------------------------------

def test_option_leg_gain_pct_reads_the_option_not_the_underlying():
    assert _Mgr().option_leg_gain_pct(_Pos(option_entry_price=2.35, option_last_price=2.20)) == \
        pytest.approx(-0.0638, abs=1e-4)
    assert _Mgr().option_leg_gain_pct(_Pos(option_entry_price=0.53, option_last_price=0.59)) == \
        pytest.approx(0.1132, abs=1e-4)


@pytest.mark.parametrize("entry,last", [(None, 2.2), (2.35, None), (0.0, 2.2), (2.35, 0.0)])
def test_option_leg_gain_pct_is_none_when_unpriced(entry, last):
    assert _Mgr().option_leg_gain_pct(_Pos(option_entry_price=entry, option_last_price=last)) is None


# --- guard 1: veto a profit exit when the option leg is at a loss --------------

def test_trail_is_vetoed_when_the_option_leg_is_underwater():
    """The CALM case."""
    mgr = _Mgr()
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=2.35, option_last_price=2.20), "trail") is True
    assert "profit_exit_vetoed_option_at_loss" in mgr.event_kinds()


def test_trail_proceeds_when_the_option_leg_is_in_profit():
    """The NOK case — option +11.3%, trail should exit normally."""
    mgr = _Mgr()
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=0.53, option_last_price=0.59), "trail") is False
    assert mgr.event_kinds() == []


def test_unpriced_option_leg_never_vetoes_the_exit():
    mgr = _Mgr()
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=2.35, option_last_price=None), "trail") is False


@pytest.mark.parametrize("reason", sorted(pm._DISCRETIONARY_PROFIT_EXITS))
def test_every_discretionary_reason_is_gated(reason):
    mgr = _Mgr()
    assert mgr._gate_discretionary_exit(_Pos(), reason) is True


@pytest.mark.parametrize("reason", [
    "sl", "no_quote", "option_take_profit", "option_profit_trail",
    "expiring_itm", "expiring_before_closure", "restored_unknown_max_loss",
])
def test_mandatory_exits_are_never_gated(reason):
    """A losing option leg AND a 60%-wide spread must not delay a mandatory exit."""
    mgr = _Mgr(spread_pct=0.595)
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=2.35, option_last_price=2.20), reason) is False


# --- guard 2: exit spread gate -------------------------------------------------

def test_wide_spread_defers_a_profitable_discretionary_exit():
    """CALM's 59.5%-of-mid quote, with the option leg in profit so guard 1 passes."""
    mgr = _Mgr(spread_pct=0.595)
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=2.35, option_last_price=4.54), "trail") is True
    assert "exit_deferred_wide_spread" in mgr.event_kinds()


def test_tight_spread_lets_the_exit_through():
    mgr = _Mgr(spread_pct=0.05)
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=2.35, option_last_price=4.54), "trail") is False


def test_missing_spread_does_not_block_the_exit():
    mgr = _Mgr(spread_pct=None)
    assert mgr._gate_discretionary_exit(_Pos(option_entry_price=2.35, option_last_price=4.54), "trail") is False


def test_spread_deferral_is_bounded_so_a_wide_book_cannot_trap_a_position():
    mgr = _Mgr(spread_pct=0.595)
    pos = _Pos(option_entry_price=2.35, option_last_price=4.54)
    for _ in range(pm._MAX_EXIT_SPREAD_DEFERRALS):
        assert mgr._gate_discretionary_exit(pos, "trail") is True
    assert mgr._gate_discretionary_exit(pos, "trail") is False
    assert "exit_spread_gate_exhausted" in mgr.event_kinds()


def test_deferral_counter_resets_once_the_spread_tightens():
    mgr = _Mgr(spread_pct=0.595)
    pos = _Pos(option_entry_price=2.35, option_last_price=4.54)
    for _ in range(3):
        mgr._gate_discretionary_exit(pos, "trail")
    assert mgr._exit_spread_deferrals[pos.ticker] == 3
    mgr._spread_pct = 0.05
    assert mgr._gate_discretionary_exit(pos, "trail") is False
    assert pos.ticker not in mgr._exit_spread_deferrals


def test_veto_takes_precedence_over_the_spread_gate():
    """A losing leg vetoes outright; it should not consume a spread deferral."""
    mgr = _Mgr(spread_pct=0.595)
    pos = _Pos(option_entry_price=2.35, option_last_price=2.20)
    assert mgr._gate_discretionary_exit(pos, "trail") is True
    assert pos.ticker not in mgr._exit_spread_deferrals
    assert "profit_exit_vetoed_option_at_loss" in mgr.event_kinds()
    assert "exit_deferred_wide_spread" not in mgr.event_kinds()
