"""A name exited earlier in the session is not re-entered on the next 4H bar.

2026-08-21, HTF Swing. The risk pass stopped CRDO260918C00250000 out at 13:50
ET for underlying_stop_-1.5atr — 2 contracts, 27.70 -> 15.80, -42.96%, -$2,380.
Thirty-five minutes later the 14:25 4H run ranked CRDO back into the top ten and
bought three contracts of the same contract at 17.00, 7.6% ABOVE the price it
had just sold at.

The exit came from live_risk_pass, a separate process, so the 4H run's managed
state carried no memory of it at all. closed_trades.jsonl is the only shared
record, which is why the cooldown reads that.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from core.live_4h_exec import _tickers_exited_this_session, build_mixed_plan, ExecPolicy

BAR = pd.Timestamp("2026-08-21 18:00:00+00:00")   # 14:00 ET, the 14:25 run's bar


def _ledger(root, module, rows):
    d = root / module
    d.mkdir(parents=True, exist_ok=True)
    (d / "closed_trades.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_reads_an_exit_from_the_same_session(tmp_path):
    _ledger(tmp_path, "htf", [{
        "ts": "2026-08-21T17:50:15.476000+00:00",     # 13:50 ET
        "ticker": "CRDO", "exit_reason": "underlying_stop_-1.5atr",
    }])
    found = _tickers_exited_this_session("htf", BAR, ledger_root=str(tmp_path))
    assert found == {"CRDO": "underlying_stop_-1.5atr"}


def test_ignores_a_previous_session(tmp_path):
    _ledger(tmp_path, "htf", [{
        "ts": "2026-08-20T17:50:15+00:00", "ticker": "CRDO", "exit_reason": "stop",
    }])
    assert _tickers_exited_this_session("htf", BAR, ledger_root=str(tmp_path)) == {}


def test_an_unreadable_ledger_does_not_stop_trading(tmp_path):
    d = tmp_path / "htf"
    d.mkdir(parents=True)
    (d / "closed_trades.jsonl").write_text("{not json\n")
    assert _tickers_exited_this_session("htf", BAR, ledger_root=str(tmp_path)) == {}


def test_missing_ledger_is_empty(tmp_path):
    assert _tickers_exited_this_session("htf", BAR, ledger_root=str(tmp_path)) == {}


def test_entry_is_refused_for_a_name_exited_this_session(tmp_path):
    """The CRDO case end to end: ranked, routable, and still not bought."""
    _ledger(tmp_path, "htf", [{
        "ts": "2026-08-21T17:50:15+00:00",
        "ticker": "CRDO", "exit_reason": "underlying_stop_-1.5atr",
    }])

    def route_fn(client, ticker, px, **_k):
        return "option", {"occ": f"{ticker}260918C00250000", "limit": 17.0,
                          "mid": 16.5, "delta": 0.45, "strike": 250.0,
                          "expiry": "2026-09-18", "open_interest": 900,
                          "volume": 300, "spread": 0.4}, "ok"

    out = build_mixed_plan(
        client=None, targets=["CRDO", "ENHA"], managed={}, pos_info={}, bar=BAR,
        signal_audits={}, policy=ExecPolicy(target_notional=5000.0),
        route_fn=route_fn, ref_price_fn=lambda _t: 229.71,
        verbose=False, module="htf", ledger_root=str(tmp_path),
    )

    bought = {row[0] for row in out.plan if row[1] == "buy"}
    assert not any(sym.startswith("CRDO") for sym in bought)
    assert out.contract_selection["CRDO"]["reason"] == (
        "exited_this_session:underlying_stop_-1.5atr"
    )
    # The cooldown is per name, not a halt: a different name still routes.
    assert any(sym.startswith("ENHA") for sym in bought)


def test_no_second_contract_while_an_unconfirmed_entry_may_still_fill(tmp_path):
    """momentum, 2026-08-21. ALM260918C00017500 x24 was submitted from the 09:43
    pre-open flush, never filled, and dropped not_found at 14:31 — and the same
    run then bought ALM260918C00020000 x34. managed is keyed by ticker, so the
    17.5 entry was overwritten in state while its order could still have been
    resting at the broker. It cost nothing only because that order never filled.
    """
    def route_fn(client, ticker, px, **_k):
        return "option", {"occ": f"{ticker}260918C00020000", "limit": 1.45,
                          "mid": 1.40, "delta": 0.45, "strike": 20.0,
                          "expiry": "2026-09-18", "open_interest": 4071,
                          "volume": 300, "spread": 0.05}, "ok"

    managed = {
        "ALM": {"route": "option", "occ": "ALM260918C00017500", "contracts": 24,
                "runs_held": 0, "bars_out": 0, "trimmed": False,
                "pending_fill": True, "entry_order_id": "a3688ab8"},
    }

    out = build_mixed_plan(
        client=None, targets=["ALM"], managed=managed,
        pos_info={},                      # broker reports nothing: never filled
        bar=BAR, signal_audits={}, policy=ExecPolicy(target_notional=5000.0),
        route_fn=route_fn, ref_price_fn=lambda _t: 17.9,
        verbose=False, module="momentum_expansion", ledger_root=str(tmp_path),
    )

    assert out.dropped["ALM"]["was_unconfirmed_entry"] is True
    assert not any(row[1] == "buy" for row in out.plan)
    assert out.contract_selection["ALM"]["reason"] == "unconfirmed_entry_may_still_fill"
    assert out.contract_selection["ALM"]["prior_symbol"] == "ALM260918C00017500"


def test_a_confirmed_flat_position_can_be_re_entered(tmp_path):
    """Only an UNCONFIRMED entry blocks. A position the broker confirmed closed
    leaves no working order behind, so the name is free to trade again."""
    def route_fn(client, ticker, px, **_k):
        return "option", {"occ": f"{ticker}260918C00020000", "limit": 1.45,
                          "mid": 1.40, "delta": 0.45, "strike": 20.0,
                          "expiry": "2026-09-18", "open_interest": 4071,
                          "volume": 300, "spread": 0.05}, "ok"

    managed = {
        "ALM": {"route": "option", "occ": "ALM260918C00017500", "contracts": 24,
                "runs_held": 3, "bars_out": 0, "trimmed": False},   # no pending_fill
    }
    out = build_mixed_plan(
        client=None, targets=["ALM"], managed=managed,
        pos_info={"ALM260918C00017500": {"qty": 0}},   # present and flat
        bar=BAR, signal_audits={}, policy=ExecPolicy(target_notional=5000.0),
        route_fn=route_fn, ref_price_fn=lambda _t: 17.9,
        verbose=False, module="momentum_expansion", ledger_root=str(tmp_path),
    )
    assert out.dropped["ALM"]["status"] == "confirmed_flat"
    assert any(row[1] == "buy" for row in out.plan)
