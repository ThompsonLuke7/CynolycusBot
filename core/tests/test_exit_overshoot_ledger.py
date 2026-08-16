"""The closed-trade ledger must record where a stop actually fired and filled.

`exit_reason` names the POLICY ("stop_-39%") and says nothing about where the
position was when the runner looked, or where it filled. On 2026-08-06/07 nine
option stops carrying that label realized a mean of -57.1% (worst -88.5%), and
nothing in the ledger separated "gapped past the level between runs" from
"filled badly once we sold". These fields make that decomposition measurable.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.live_4h_exec import _threshold_from_exit_reason, record_exit_realized_pnl


class _Client:
    """Reports the exit order as filled at `fill`."""

    def __init__(self, fill: float | None) -> None:
        self.fill = fill

    def get_order(self, _order_id):
        return {"status": "filled", "filled_avg_price": self.fill, "filled_qty": "1"}


def _read_only_row(root: Path, module: str) -> dict:
    lines = [l for l in (root / module / "closed_trades.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    return json.loads(lines[0])


def _record(tmp_path: Path, *, reason: str, entry: float, fill: float,
            decision_gain: float | None, route: str = "option") -> dict:
    record_exit_realized_pnl(
        _Client(fill),
        module="htf",
        item=("WDC260821C00590000", "sell", 2, reason, route),
        resp={"id": "abc", "status": "filled", "filled_avg_price": fill, "filled_qty": "2"},
        entry_state={"ticker": "WDC", "entry_bar": "2026-07-30 18:00:00+00:00",
                     "runs_held": 6, "unrealized_gain": decision_gain},
        pos_lookup={"WDC260821C00590000": {"avg_entry": entry}},
        bar="2026-08-06 14:00:00+00:00",
        ledger_root=str(tmp_path),
    )
    return _read_only_row(tmp_path, "htf")


def test_threshold_parsed_only_from_stop_reasons():
    assert _threshold_from_exit_reason("stop_-39%") == 0.39
    assert _threshold_from_exit_reason("stop_-50%") == 0.50
    assert _threshold_from_exit_reason("take_profit_+30%") is None
    assert _threshold_from_exit_reason("trail_-35%") is None
    assert _threshold_from_exit_reason("horizon") is None
    assert _threshold_from_exit_reason(None) is None


def test_gap_through_stop_is_recorded_as_overshoot(tmp_path):
    # The real 2026-08-06 WDC trade: 31.75 -> 3.65 on a -39% label.
    row = _record(tmp_path, reason="stop_-39%", entry=31.75, fill=3.65, decision_gain=-0.87)
    assert row["fill_gain"] == -0.885039
    assert row["decision_gain"] == -0.87
    # Filled 49.5 points of return below where the policy said to exit.
    assert round(row["stop_overshoot"], 3) == -0.495


def test_stop_that_fires_near_its_threshold_shows_little_overshoot(tmp_path):
    row = _record(tmp_path, reason="stop_-39%", entry=1.00, fill=0.60, decision_gain=-0.40)
    assert round(row["stop_overshoot"], 3) == -0.01


def test_non_stop_exits_have_no_overshoot(tmp_path):
    row = _record(tmp_path, reason="take_profit_+30%", entry=1.00, fill=1.35, decision_gain=0.35)
    assert row["stop_overshoot"] is None
    assert row["fill_gain"] == 0.35


def test_missing_decision_gain_is_null_not_zero(tmp_path):
    # State written before this change carries no `unrealized_gain`; reporting
    # 0.0 there would read as "exited at breakeven".
    row = _record(tmp_path, reason="stop_-39%", entry=2.00, fill=1.00, decision_gain=None)
    assert row["decision_gain"] is None
    assert row["fill_gain"] == -0.5


def test_unknown_fill_leaves_derived_fields_null(tmp_path):
    row = _record(tmp_path, reason="stop_-39%", entry=2.00, fill=None, decision_gain=-0.55)
    assert row["fill_gain"] is None
    assert row["stop_overshoot"] is None
    assert row["decision_gain"] == -0.55
