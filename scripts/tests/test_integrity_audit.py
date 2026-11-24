"""The integrity audit must find the 2026-11-23 defects from their raw traces."""
from __future__ import annotations

import json
from datetime import date

import scripts.integrity_audit as audit

DAY = date(2026, 11, 23)


def test_lines_for_date_reads_only_the_requested_session_from_a_long_tail(tmp_path):
    path = tmp_path / "log.jsonl"
    with path.open("w") as fh:
        for i in range(20_000):  # older sessions, larger than the first tail window
            fh.write(json.dumps({"timestamp": "2026-11-20T15:00:00+00:00", "pad": "x" * 500, "i": i}) + "\n")
        for i in range(3):
            fh.write(json.dumps({"timestamp": f"2026-11-23T15:0{i}:00+00:00", "i": i}) + "\n")
    rows = audit._lines_for_date(path, DAY)
    assert [r["i"] for r in rows] == [0, 1, 2]


def test_carried_pre_entry_setups_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "IS_DIR", tmp_path)
    (tmp_path / "state.json").write_text(json.dumps({"setups": [
        {"setup_id": "QQQ:short:gamma_structural_level_rejection", "ticker": "QQQ", "state": "ARMED",
         "state_entered_at": "2026-11-12T16:24:00+00:00"},
        {"setup_id": "QQQ:long:breakout_continuation", "ticker": "QQQ", "state": "ARMED",
         "state_entered_at": "2026-11-23T16:00:00+00:00"},
        {"setup_id": "SPY:long:breakout_continuation", "ticker": "SPY", "state": "WATCHING",
         "state_entered_at": "2026-07-01T16:00:00+00:00"},
    ]}))
    got = audit.check_carried_setups(DAY)
    assert got.status == "FAIL" and "1 pre-entry setups" in got.detail and "2026-11-12" in got.detail


def test_a_far_otm_pick_warns_even_when_it_never_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "SERVER_LOGS", tmp_path)
    monkeypatch.setattr(audit, "IS_DIR", tmp_path)
    (tmp_path / "server_20261123.log").write_text(
        "2026-11-23 09:34:29,001 INFO [strategies.intraday_structure.execution] intraday execution: selected "
        "QQQ261123P00590000 for QQQ:short:vwap_reclaim_continuation (spot 652.13, strike 590.00, otm +9.5%, "
        "pool nearest_strikes_no_greeks)\n"
        "2026-11-23 09:34:30,284 INFO [strategies.intraday_structure.execution] intraday execution: entry for "
        "QQQ261123P00590000 did not fill; not claiming it\n"
    )
    got = audit.check_option_moneyness(DAY)
    assert got.status == "WARN" and "QQQ261123P00590000 (+9.5%)" in got.detail


def test_decision_context_flags_index_decisions_without_dealer_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "IS_DIR", tmp_path)
    monkeypatch.setattr(audit, "CONTEXT_DIR", tmp_path / "decision_context")
    (tmp_path / "decision_context").mkdir()
    rows = [{"kind": "confirm", "bar_ts": "2026-11-23T15:00:00+00:00", "instrument_class": "index_etf",
             "options": {"source": "none"}, "daily_regime": {"date": "2026-11-20"}} for _ in range(3)]
    rows.append({"kind": "manage", "bar_ts": "2026-11-23T15:01:00+00:00", "instrument_class": "index_etf",
                 "options": {"source": "dealer_positioning_static"}, "daily_regime": None})
    (tmp_path / "decision_context" / "20261123.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = audit.check_decision_context(DAY)
    assert got.status == "WARN" and "3/4 index-ETF RTH decisions had NO dealer levels" in got.detail
    assert "1 records without a daily regime row" in got.detail


def test_missing_decision_context_with_activity_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "IS_DIR", tmp_path)
    monkeypatch.setattr(audit, "CONTEXT_DIR", tmp_path / "decision_context")
    (tmp_path / "transitions.jsonl").write_text(json.dumps({"timestamp": "2026-11-23T15:00:00+00:00"}) + "\n")
    assert audit.check_decision_context(DAY).status == "WARN"
