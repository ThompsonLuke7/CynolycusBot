"""A restored position must not report a P&L it cannot know.

When a position is rebuilt from a broker snapshot with no local state, there is
no recoverable underlying entry price, so `_restore_from_broker` stamps the
restore-time mark as `entry_price`. Any P&L derived from that measures "move
since we noticed the position", not "move since entry".

2026-08-07: UMC and TKR were both cut by `restored_unknown_loss_cut` with their
option legs at -45.5% and -45.1%, while the audit recorded `pnl_pct: 0.0` for
both — the two positions were restored and cut on the same bar, so the synthetic
basis equalled the mark exactly. The broker's real figure was in
`option_entry_meta` the whole time and went unreported.
"""
from __future__ import annotations

from datetime import datetime

from strategies.multi_ticker_swing.live.position_manager import SwingPosition
from strategies.multi_ticker_swing.live.universe import TickerConfig

SYMBOL = "UMC260918C00021000"


def _cfg() -> TickerConfig:
    return TickerConfig(
        ticker="UMC", tier=1, entry_threshold=0.5, sl_atr=2.0, np_n_bars=None,
        np_mfe_atr=None, avg_win_pct=0.1, avg_loss_pct=-0.05, profit_factor=1.5, sharpe=1.0,
    )


def _position(**kwargs) -> SwingPosition:
    base = dict(
        ticker="UMC", direction=1, entry_price=19.275, entry_time=datetime.now(),
        atr_at_entry=1.0, option_symbol=SYMBOL, qty=31, config=_cfg(),
    )
    base.update(kwargs)
    return SwingPosition(**base)


def test_normal_position_still_reports_pnl():
    pos = _position()
    pos.last_price = 20.0
    row = pos.to_dict()
    assert row["entry_price_is_synthetic"] is False
    assert row["pnl_pct_source"] == "entry_price"
    assert row["pnl_pct"] == (20.0 - 19.275) / 19.275


def test_broker_restored_position_reports_unknown_not_zero():
    pos = _position(
        restored_from_broker=True,
        restore_source="broker_snapshot",
        option_entry_price=1.65,
        option_entry_meta={"broker_unrealized_plpc": -0.4545, "restore_source": "broker_snapshot"},
    )
    row = pos.to_dict()
    assert row["entry_price_is_synthetic"] is True
    # The bug: this used to be exactly 0.0 while the position was -45%.
    assert row["pnl_pct"] is None
    assert row["pnl_pct_source"] == "restore_time_mark_unusable"
    # The authoritative number is carried instead of being dropped.
    assert row["broker_unrealized_plpc"] == -0.4545


def test_position_restored_from_local_state_keeps_its_real_basis():
    # `local_state` restores carry the ORIGINAL entry price, so their P&L is real.
    pos = _position(restored_from_broker=True, restore_source="local_state")
    pos.last_price = 17.0
    row = pos.to_dict()
    assert row["entry_price_is_synthetic"] is False
    assert row["pnl_pct"] == (17.0 - 19.275) / 19.275
    assert row["pnl_pct_source"] == "entry_price"


def test_missing_broker_plpc_is_null_not_zero():
    pos = _position(
        restored_from_broker=True,
        restore_source="broker_snapshot",
        option_entry_meta={"restore_source": "broker_snapshot"},
    )
    row = pos.to_dict()
    assert row["pnl_pct"] is None
    assert row["broker_unrealized_plpc"] is None
