"""Momentum run_pass drives the shared 4H engine: 1H-gated top-N -> option/share
routing -> execute -> persisted managed state (same machine as Meta/HTF)."""
from __future__ import annotations

import json

import pytest

import strategies.momentum_expansion.live.runner as mr


class _FakeClient:
    def __init__(self):
        self.orders = []

    def get_positions(self):
        return []

    def submit_option_order(self, **k):
        self.orders.append(("option", k))
        return {"id": "opt1"}

    def submit_order(self, **k):
        self.orders.append(("equity", k))
        return {"id": "eq1"}


def test_run_pass_routes_and_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("CYNOLYCUS_READINESS_REQUIRED", "0")
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)  # submit path, not the after-close defer
    monkeypatch.setattr(mr, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(mr, "DEFAULT_SIGNAL_AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(mr, "_ref_price_4h", lambda t: 50.0)

    def fake_route(client, t, px, **_):
        if t == "OPTNAME":
            return "option", {
                "occ": "OPTNAME260717C00050000", "limit": 1.6, "mid": 1.5, "delta": 0.45,
                "strike": 50.0, "expiry": "2026-07-17", "open_interest": 900, "volume": 300,
                "spread": 0.1,
            }, "ok"
        return "equity", None, "underlying_lt_10"

    monkeypatch.setattr(mr, "route_option_or_shares", fake_route)

    runner = mr.MomentumLiveRunner(auto_trade=True)

    def fake_eval(bar_ts=None):
        runner._last_gate = {
            "targets": ["OPTNAME", "SHARENAME", "NOTRIG"],
            "entry_ok": {"OPTNAME": True, "SHARENAME": True, "NOTRIG": False},
            "signal_audits": {"OPTNAME": {"score": 0.5}, "SHARENAME": {"score": 0.5}},
            "bar": "2026-07-02T18:00:00Z",
        }
        return []

    monkeypatch.setattr(runner, "evaluate_now", fake_eval)

    client = _FakeClient()
    res = runner.run_pass(client, submit=True)

    # OPTNAME -> 10 contracts, SHARENAME -> 100 shares, NOTRIG -> gated out (no trigger)
    assert res["orders"] == 2
    assert sorted(o[0] for o in client.orders) == ["equity", "option"]
    opt = next(k for r, k in client.orders if r == "option")
    eq = next(k for r, k in client.orders if r == "equity")
    assert opt["qty"] == 10 and eq["qty"] == 100

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["managed"]["OPTNAME"]["route"] == "option"
    assert state["managed"]["SHARENAME"]["route"] == "equity"
    assert "NOTRIG" not in state["managed"]


def test_run_pass_no_targets(monkeypatch, tmp_path):
    monkeypatch.setattr(mr, "STATE_PATH", tmp_path / "state.json")
    runner = mr.MomentumLiveRunner(auto_trade=False)
    monkeypatch.setattr(runner, "evaluate_now", lambda bar_ts=None: [])  # leaves _last_gate None
    runner._last_gate = None
    assert runner.run_pass(_FakeClient(), submit=False) == {"orders": 0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
