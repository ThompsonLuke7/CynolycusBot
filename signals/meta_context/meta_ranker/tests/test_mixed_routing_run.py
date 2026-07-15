"""_run_options builds a MIXED option+share plan and manages both instrument
types with the one hold-based exit/scale-out machine."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import signals.meta_context.meta_ranker.live_runner as lr

OPT_HELD = "AAA260717C00050000"
NEW_OPT_OCC = "BBB260717C00050000"


def _args():
    return SimpleNamespace(
        scale_frac=0.5, take_profit=0.2, horizon_bars=25, grace_bars=3, stop_loss=0.50, trail_stop=0.50,
        submit=False, mode="options", shares=100, contracts=10,
        signal_audit_log="", roll_trading_days=5,
    )


def test_mixed_manage_and_entry(monkeypatch):
    monkeypatch.setattr(lr, "_ref_price", lambda t: 50.0)

    def fake_route(client, t, px, **_):
        if t == "NEWOPT":
            return "option", {
                "occ": NEW_OPT_OCC, "limit": 1.6, "mid": 1.5, "delta": 0.45,
                "strike": 50.0, "expiry": "2026-07-17", "open_interest": 900, "volume": 300,
                "spread": 0.1,
            }, "ok"
        return "equity", None, "underlying_lt_10"  # NEWEQ -> shares

    monkeypatch.setattr(lr, "route_option_or_shares", fake_route)

    managed = {
        # held option, still in targets, +30% -> take-profit trim (sell 5 of 10)
        "AAA": {"route": "option", "occ": OPT_HELD, "runs_held": 0, "bars_out": 0, "trimmed": False},
        # held shares, dropped out of top-K beyond grace -> full exit (100 sh)
        "EQOLD": {"route": "equity", "symbol": "EQOLD", "runs_held": 2, "bars_out": 5, "trimmed": False},
    }
    pos_info = {
        OPT_HELD: {"qty": 10, "avg_entry": 1.0, "current": 1.3},
        "EQOLD": {"qty": 100, "avg_entry": 50.0, "current": 50.5},
    }
    state = {}
    targets = ["AAA", "NEWOPT", "NEWEQ"]
    entry_ok = {"AAA": True, "NEWOPT": True, "NEWEQ": True}

    captured = {}
    monkeypatch.setattr(lr, "_execute", lambda *a, **k: captured.update(
        plan=a[2], new_managed=a[4]))

    lr._run_options(_args(), object(), targets, state, managed, pos_info,
                    "2026-07-02T18:00:00Z", entry_ok=entry_ok, signal_audits={})

    plan = {p[0]: p for p in captured["plan"]}
    nm = captured["new_managed"]

    # held option trimmed 5 (route option); held equity fully exited 100 (route equity)
    assert plan[OPT_HELD] == (OPT_HELD, "sell", 5, plan[OPT_HELD][3], "option")
    assert "take_profit" in plan[OPT_HELD][3]
    assert plan["EQOLD"][1:] == ("sell", 100, plan["EQOLD"][3], "equity")

    # new names: one option (10 contracts), one shares (100)
    assert plan[NEW_OPT_OCC] == (NEW_OPT_OCC, "buy", 10, "entry", "option")
    assert plan["NEWEQ"] == ("NEWEQ", "buy", 100, "entry", "equity")

    # managed state carries route + instrument key; exited name dropped
    assert nm["AAA"]["route"] == "option" and nm["AAA"]["trimmed"] is True
    assert nm["NEWOPT"]["route"] == "option" and nm["NEWOPT"]["occ"] == NEW_OPT_OCC
    assert nm["NEWEQ"]["route"] == "equity" and nm["NEWEQ"]["symbol"] == "NEWEQ"
    assert "EQOLD" not in nm


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
