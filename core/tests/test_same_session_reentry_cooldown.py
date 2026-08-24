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
