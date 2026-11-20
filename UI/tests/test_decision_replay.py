"""Decision replay: bars, decisions, orders and fills lined up without look-ahead.

The contract under test (see ``UI/decision_replay.py``): replay frame i means
"bar i has closed", and nothing may appear on a frame before it existed. An
archived decision sits on the bar containing its wall-clock ``decision_ts``.
An audit row, which logs only the bar it scored, sits where that bar ended. A
later run re-scoring the same bar sits on the next bar. Fills sit on the bar
during which they executed.
"""
from __future__ import annotations

import json
import threading
import types
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from core import decision_state_archive as dsa
from UI import decision_replay as dr
from UI import trade_library

POLICY = types.SimpleNamespace(stop_loss=0.39, take_profit=0.30, scale_frac=0.16, trail_stop=None,
                               horizon_bars=53, underlying_stop_atr=1.5)
R1, R2, R3 = "a" * 32, "b" * 32, "c" * 32
DAYS = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC") if pd.Timestamp(s).tzinfo is None else pd.Timestamp(s).tz_convert("UTC")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Synthetic 4H bars, ledger, audit log and decision archive for ticker AAA."""
    bars_dir = tmp_path / "bars4h"
    bars_dir.mkdir()
    stamps = [pd.Timestamp(f"{d}T{h}:00:00Z") for d in DAYS for h in ("15", "19")]
    px = np.linspace(10.0, 8.0, len(stamps))
    pd.DataFrame({"timestamp": stamps, "open": px, "high": px + 0.2, "low": px - 0.2,
                  "close": px, "volume": 1000.0}).to_parquet(bars_dir / "AAA.parquet")
    monkeypatch.setattr(trade_library, "BARS_4H_DIR", bars_dir)

    ledger = tmp_path / "inference"
    (ledger / "meta_ranker").mkdir(parents=True)
    (ledger / "meta_ranker" / "closed_trades.jsonl").write_text(json.dumps({
        "ts": "2026-11-19T19:30:00+00:00", "module": "meta_ranker", "ticker": "AAA",
        "order_symbol": "AAA", "route": "equity", "qty": 100, "exit_reason": "stop_-39%",
        "entry_avg_price": 10.0, "exit_fill_price": 6.1, "realized_pnl": -390.0,
        "entry_bar": "2026-11-16 19:00:00+00:00", "entry_filled_at": "2026-11-17T14:45:00.123456789Z",
        "entry_fill_price": 10.0, "u_entry": 10.0, "u_atr": 0.4, "runs_held": 7,
        "decision_gain": -0.39, "fill_gain": -0.39, "exit_submitted_at": "2026-11-19T19:29:00+00:00",
        "order_id": "x1"}) + "\n")
    monkeypatch.setattr(trade_library, "LEDGER_ROOT", ledger)
    monkeypatch.setattr(trade_library, "LIVE_STATES", {})
    monkeypatch.setattr(dr, "AUDIT_ROOT", ledger)
    monkeypatch.setattr(dr, "_policy", lambda module: POLICY)

    audit = [
        # run 0: scores the 11-16 15:00 bar, AAA in the targets (pre-archive, no run id)
        {"event": "signal_decision", "module": "meta_ranker", "bar": "2026-11-16T15:00:00+00:00",
         "targets": ["AAA", "BBB"], "signal_audits": {"AAA": {"score": 0.9, "rank": 1, "rank_pct": 0.99}}},
        # run 1: a later run re-scoring the same bar, AAA not in the targets
        {"event": "signal_decision", "module": "meta_ranker", "bar": "2026-11-16T15:00:00+00:00",
         "targets": ["BBB"], "signal_audits": {}},
        # run 2: the archived run R1 (deduplicated), and its order plan
        {"event": "signal_decision", "module": "meta_ranker", "bar": "2026-11-16T19:00:00+00:00",
         "targets": ["AAA"], "signal_audits": {"AAA": {"score": 0.95, "decision_run_id": R1}}},
        {"event": "order_plan", "module": "meta_ranker", "bar": "2026-11-16T19:00:00+00:00",
         "signal_audits": {"AAA": {"decision_run_id": R1}},
         "contract_selection": {"AAA": {"action": "equity", "reason": "underlying_lt_10", "ref_price": 10.0}},
         "planned": [{"symbol": "AAA", "side": "buy", "qty": 100, "reason": "entry", "route": "equity",
                      "disposition": "submitted"}]},
    ]
    (ledger / "meta_ranker" / "live_signal_audit.jsonl").write_text(
        "\n".join(json.dumps(e) for e in audit) + "\n")

    tickers = [f"T{i:02d}" for i in range(30)] + ["AAA"]

    def run(run_id, decision_ts, bar_ts, aaa_selected):
        cand = pd.DataFrame({"ticker": tickers, "score": np.linspace(0.0, 1.0, len(tickers)),
                             "f1": np.arange(len(tickers), dtype=float),
                             "f2": np.r_[np.full(len(tickers) - 1, 1.0), 5.0]})
        cand.loc[cand.ticker == "AAA", "f1"] = 15.0 if run_id != R1 else 3.0
        sel = ["AAA"] if aaa_selected else ["T29"]
        dsa.archive_decision_state(
            module="meta_ranker", strict=True, run_id=run_id, candidates=cand, bar_ts=bar_ts,
            decision_ts=decision_ts, score_col="score", feature_cols=["f1", "f2"],
            gates={"floor": cand.ticker.isin(sel + ["T28"]).tolist(), "top_k": cand.ticker.isin(sel).tolist()},
            selected=sel, extra_cols=[], inputs_asof={"matrix_bar": bar_ts}, model_version="m@1")

    run(R1, "2026-11-17T14:35:00Z", "2026-11-16T19:00:00Z", True)    # 09:35 ET re-score -> entry
    run(R2, "2026-11-17T19:20:00Z", "2026-11-17T15:00:00Z", False)   # 14:20 ET run
    run(R3, "2026-11-17T21:20:00Z", "2026-11-17T19:00:00Z", False)   # 16:20 ET run
    return {"stamps": stamps, "pid": "meta_ranker:AAA:2026-11-16T19:00:00+00:00"}


def _frame(stamps, iso):
    return stamps.index(pd.Timestamp(iso))


# --------------------------------------------------------------------------- placement

def test_place_uses_the_containing_bar_or_the_next_one_never_an_earlier_one():
    times = np.array([pd.Timestamp(f"2026-11-17T{h}:00:00Z").value for h in ("15", "19")], dtype="int64")
    four_h = int(4 * 3600 * 1e9)
    assert dr.place(times, "2026-11-17T16:30:00Z", four_h) == 0          # inside bar 0
    assert dr.place(times, "2026-11-17T19:00:00Z", four_h) == 1          # a bar's open belongs to it
    assert dr.place(times, "2026-11-17T14:45:00Z", four_h) == 0          # before the first bar opens -> next bar
    assert dr.place(times, "2026-11-17T23:30:00Z", four_h) is None       # after the last bar ended
    assert dr.place(times, None, four_h) is None


def test_timeframes_never_shorter_than_the_bar_a_module_decides_on():
    assert dr.timeframe_options("meta_ranker") == ["4h", "1d"]           # no 1h: would split the scored bar
    assert dr.timeframe_options("multi_ticker_swing") == ["1h", "4h", "1d"]
    assert dr.timeframe_options("intraday_structure") == ["1m", "1h", "4h", "1d"]
    assert "1m" not in dr.timeframe_options("dealer_ranker")
    assert dr.default_timeframe("intraday_structure") == "1m"


# --------------------------------------------------------------------------- end-to-end replay

def test_replay_places_every_decision_no_earlier_than_it_existed(world):
    r = dr.replay(position_id=world["pid"])
    st = world["stamps"]
    assert r["error"] is None and r["subject"]["tf"] == "4h"
    span = pd.Timedelta(hours=4)
    arch = {d["run_id"]: d for d in r["decisions"] if d["source"] == "archive"}
    # wall-clock placement: each archived decision is shown at the close of the bar it was made in
    assert arch[R1]["frame"] == _frame(st, "2026-11-17T15:00:00Z")   # 09:35 ET falls before the 10:00 bar
    assert arch[R1]["scored_frame"] == _frame(st, "2026-11-16T19:00:00Z")
    assert arch[R2]["frame"] == _frame(st, "2026-11-17T19:00:00Z")
    assert arch[R3]["frame"] == _frame(st, "2026-11-17T19:00:00Z")
    for d in arch.values():
        assert st[d["frame"]] + span > _ts(d["decision_ts"])
        assert d["frame"] >= d["scored_frame"]
    audit = [d for d in r["decisions"] if d["source"] == "audit"]
    assert len(audit) == 2                                   # the archived run R1 is not repeated
    first, rescore = audit
    assert first["in_targets"] is True and first["score"] == 0.9
    assert first["frame"] == _frame(st, "2026-11-16T19:00:00Z")      # where the scored 15:00 bar ended
    assert rescore["in_targets"] is False
    assert rescore["frame"] == _frame(st, "2026-11-17T15:00:00Z")    # a later run of the same bar: next bar
    for d in audit:
        assert d["frame"] > d["scored_frame"]


def test_replay_lines_up_orders_fills_and_the_entry_decision(world):
    r = dr.replay(position_id=world["pid"])
    st = world["stamps"]
    entry = _frame(st, "2026-11-17T15:00:00Z")
    assert r["entry_frame"] == entry
    assert r["exit_frame"] == _frame(st, "2026-11-19T19:00:00Z")
    assert r["entry_decision"] == f"archive:{R1}" and r["entry_run_id"] == R1
    (order,) = [e for e in r["events"] if e["kind"] == "order_plan"]
    assert order["frame"] == entry                            # the plan sits with its run, not its scored bar
    assert order["detail"]["disposition"] == "submitted"
    assert order["detail"]["routing"]["reason"] == "underlying_lt_10"
    assert order["detail"]["route"] == "equity"
    (exit_ev,) = [e for e in r["events"] if e["kind"] == "exit"]
    assert exit_ev["decision_gain"] == -0.39 and exit_ev["on_chart_scale"] is True
    assert r["coverage"]["archive"]["rows_in_window"] == 3


def test_equity_levels_prefer_the_threshold_that_fired_and_label_the_rest(world):
    r = dr.replay(position_id=world["pid"])
    lv = {x["key"]: x for x in r["levels"]}
    assert lv["stop"]["price"] == pytest.approx(6.1) and "from exit reason" in lv["stop"]["label"]
    assert lv["take_profit"]["price"] == pytest.approx(13.0)
    assert "current risk-pass policy" in lv["take_profit"]["label"]
    assert r["ref"] == {"price": 10.0, "basis": "entry fill price (shares)"}
    assert lv["stop"]["from_frame"] == r["entry_frame"] and lv["stop"]["to_frame"] == r["exit_frame"]


def test_option_levels_use_the_recorded_underlying_anchor_and_never_replay_premium(monkeypatch):
    monkeypatch.setattr(dr, "_policy", lambda module: POLICY)
    pos = {"module": "meta_ranker", "route": "option", "reasons": [], "entry_price": 3.2,
           "entry_anchor": {"u_entry": 100.0, "u_atr": 2.0}}
    levels, ref, policy, notes = dr.position_levels(pos)
    assert {x["key"]: x["price"] for x in levels} == {"u_entry": 100.0, "u_stop": 97.0}
    assert ref["price"] == 100.0 and "anchor" in ref["basis"]
    assert any("premium is not replayed" in n for n in notes)
    levels, *_ = dr.position_levels({**pos, "reasons": ["underlying_stop_-2atr"]})
    assert {x["key"]: x["price"] for x in levels}["u_stop"] == 96.0


def test_no_policy_means_no_invented_levels(monkeypatch):
    monkeypatch.setattr(dr, "_policy", lambda module: None)
    levels, ref, policy, _ = dr.position_levels({"module": "spy_daytrader", "route": "equity",
                                                 "reasons": ["close"], "entry_fill_price": 5.0})
    assert [x["key"] for x in levels] == ["entry"] and policy is None


# --------------------------------------------------------------------------- run detail

def test_run_detail_reports_features_percentiles_and_drift_since_entry(world):
    out = dr.run_detail("meta_ranker", R2, "AAA", ref_run_id=R1)
    feats = {f["name"]: f for f in out["features"]}
    assert feats["f1"]["value"] == 15.0
    assert feats["f1"]["pct"] == pytest.approx(16 / 31)       # mid-rank: 15 below, 2 tied (T15 and AAA)
    assert feats["f1"]["ref_value"] == 3.0                     # value at the entry decision
    assert feats["f2"]["pct"] == pytest.approx(30.5 / 31)
    assert out["row"]["gates"] == {"floor": False, "top_k": False}
    assert out["row"]["gate_reason"] == "floor"
    assert out["picks"][0]["ticker"] == "T29"
    assert out["gate_pass_rates"]["top_k"] == pytest.approx(1 / 31, abs=1e-4)
    assert out["meta"]["inputs_asof"][0]["age_h"] == pytest.approx(4.33, abs=0.01)
    assert out["ref"]["run_id"] == R1


def test_run_detail_refuses_bad_ids_and_reports_unscored_tickers(world):
    assert "error" in dr.run_detail("meta_ranker", "../../etc/passwd", "AAA")
    assert "error" in dr.run_detail("not_a_module", R1, "AAA")
    out = dr.run_detail("meta_ranker", R1, "ZZZ")
    assert out["row"] is None and "not scored" in out["note"]


def test_whole_universe_runs_report_a_missing_ticker_as_not_scored(world):
    rows, warnings = dr.archive_decisions("meta_ranker", "ZZZ", _ts("2026-11-16"), _ts("2026-11-21"))
    assert len(rows) == 3 and all(r["scored"] is False for r in rows) and not warnings
    assert dr.archive_decisions("multi_ticker_swing", "ZZZ", _ts("2026-11-16"), _ts("2026-11-21"))[0] == []


# --------------------------------------------------------------------------- audit segmentation

def test_momentum_audit_runs_group_per_ticker_events_until_the_order_plan(tmp_path):
    path = tmp_path / "audit.jsonl"
    a, b = "2026-11-17T15:00:00+00:00", "2026-11-17T19:00:00+00:00"
    events = [
        {"event": "signal_decision", "bar": a, "ticker": "X", "entry_triggered": True,
         "signal_audit": {"score": 0.6, "rank": 1}},
        {"event": "signal_decision", "bar": a, "ticker": "Y", "entry_triggered": False, "skip_reason": "no_trigger"},
        {"event": "order_plan", "bar": a, "planned": [], "dispositions": {"Z": "deferred_exit_opg_unavailable"},
         "order_audits": {"Z": {"side": "sell", "qty": 5, "instrument": "equity"}}},
        {"event": "signal_decision", "bar": a, "ticker": "X", "entry_triggered": False},   # next run, same bar
        {"event": "signal_decision", "bar": a, "ticker": None, "skip_reason": "stale_signal_bar"},
        {"event": "signal_decision", "bar": b, "ticker": "Y", "entry_triggered": False},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    idx = dr._build_audit_index(path, "momentum_expansion")
    assert [sorted(r["targets"]) for r in idx["runs"]] == [["X", "Y"], ["X"], ["Y"]]
    assert idx["runs"][1]["skip"] == "stale_signal_bar"
    assert idx["tickers"]["X"][0]["entry_triggered"] is True and idx["tickers"]["X"][0]["score"] == 0.6
    (order,) = idx["tickers"]["Z"][0]["orders"]
    assert order["side"] == "sell" and order["disposition"] == "deferred_exit_opg_unavailable"


# --------------------------------------------------------------------------- free mode / positions

def test_free_mode_validates_its_inputs(world):
    assert "unknown module" in dr.replay(module="nope", ticker="AAA", start="2026-11-16", end="2026-11-18")["error"]
    assert "bad ticker" in dr.replay(module="meta_ranker", ticker="a;b", start="2026-11-16", end="2026-11-18")["error"]
    assert "YYYY-MM-DD" in dr.replay(module="meta_ranker", ticker="AAA", start="16/11", end="2026-11-18")["error"]
    assert "too long" in dr.replay(module="meta_ranker", ticker="AAA", start="2025-01-01", end="2026-11-18")["error"]
    r = dr.replay(module="meta_ranker", ticker="AAA", start="2026-11-17", end="2026-11-17")
    assert [b["t"][:10] for b in r["bars"]] == ["2026-11-17", "2026-11-17"]   # exactly the window asked for
    assert r["position"] is None and r["entry_frame"] is None


def test_open_positions_without_a_sell_are_replayable(world, tmp_path, monkeypatch):
    state = tmp_path / "meta_state.json"
    state.write_text(json.dumps({"managed": {"BBB": {
        "route": "equity", "entry_bar": "2026-11-17 15:00:00+00:00", "entry_avg_price": 9.0,
        "entry_fill_price": 9.0, "entry_filled_at": "2026-11-17T19:25:00Z", "u_entry": 9.0, "u_atr": 0.3,
        "runs_held": 4, "last_mark_price": 8.5, "unrealized_gain": -0.0556, "peak_gain": 0.02}}}))
    monkeypatch.setattr(trade_library, "LIVE_STATES", {"meta_ranker": state})
    stamps = [pd.Timestamp(f"{d}T{h}:00:00Z") for d in DAYS for h in ("15", "19")]
    pd.DataFrame({"timestamp": stamps, "open": 9.0, "high": 9.1, "low": 8.9, "close": 9.0,
                  "volume": 1.0}).to_parquet(trade_library.BARS_4H_DIR / "BBB.parquet")
    listed = dr.positions({"status": "open"})["positions"]
    (pos,) = [p for p in listed if p["ticker"] == "BBB"]
    assert pos["id"].startswith("open:meta_ranker:BBB:") and pos["evidence"] == "archive"
    r = dr.replay(position_id=pos["id"])
    assert r["position"]["unsold"] is True and r["exit_frame"] is None
    assert r["entry_frame"] == _frame(stamps, "2026-11-17T19:00:00Z")
    assert r["ref"]["price"] == 9.0


# --------------------------------------------------------------------------- intraday sources

def test_1m_bars_keep_the_later_arrival_and_report_missing_days(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "BARS_1M_DIR", tmp_path)
    rows = [
        {"symbol": "UNH", "timestamp": "2026-11-18T15:00:00+00:00", "open": 1, "high": 1, "low": 1, "close": 1,
         "volume": 1, "arrival_at": "2026-11-18T15:01:00+00:00"},
        {"symbol": "UNH", "timestamp": "2026-11-18T15:00:00+00:00", "open": 2, "high": 2, "low": 2, "close": 2,
         "volume": 2, "arrival_at": "2026-11-18T15:01:30+00:00"},
        {"symbol": "UNHX", "timestamp": "2026-11-18T15:00:00+00:00", "open": 9, "high": 9, "low": 9, "close": 9},
    ]
    (tmp_path / "bars_20261118.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows))
    bars, err = dr._bars_1m("UNH", _ts("2026-11-18T14:00:00"), _ts("2026-11-18T21:00:00"))
    assert err is None and [b["c"] for b in bars] == [2]
    bars, err = dr._bars_1m("UNH", _ts("2026-11-19T14:00:00"), _ts("2026-11-19T21:00:00"))
    assert bars == [] and "2026-11-19" in err


def test_transitions_are_matched_by_exact_ticker_and_flag_the_position_setup(tmp_path):
    log = tmp_path / "transitions.jsonl"
    rows = [
        {"ticker": "UNH", "setup_id": "UNH:long:a", "from_state": "ARMED", "to_state": "CONFIRMED",
         "timestamp": "2026-11-18T20:35:00+00:00", "reason": "ok", "evidence": ["e"], "spot": 1.0},
        {"ticker": "UNH", "setup_id": "UNH:long:b", "from_state": "WATCHING", "to_state": "CLOSED",
         "timestamp": "2026-11-18T20:36:00+00:00"},
        {"ticker": "UNHX", "setup_id": "UNHX:long:a", "to_state": "CONFIRMED", "timestamp": "2026-11-18T20:35:00+00:00"},
        {"ticker": "UNH", "setup_id": "UNH:long:a", "to_state": "CLOSED", "timestamp": "2026-11-19T20:35:00+00:00"},
    ]
    log.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows))
    out = dr.transition_decisions("UNH", _ts("2026-11-18"), _ts("2026-11-18T23:59:00"), "UNH:long:a", path=log)
    assert [(d["setup_id"], d["to_state"]) for d in out] == [("UNH:long:a", "CONFIRMED"), ("UNH:long:b", "CLOSED")]
    assert out[0]["selected"] and out[0]["position_setup"] and not out[1]["position_setup"]


# --------------------------------------------------------------------------- dashboard + hub

@pytest.fixture
def server(world):
    from UI.decision_replay_dashboard import make_server
    srv = make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_dashboard_endpoints(server, world):
    status, page = _get(server, "/")
    assert status == 200 and "Decision Replay" in page and '"meta_ranker"' in page and "__CONFIG__" not in page
    status, body = _get(server, "/api/state")
    assert status == 200 and json.loads(body)["positions"] == 1
    status, body = _get(server, "/api/positions?module=meta_ranker")
    assert [p["ticker"] for p in json.loads(body)["positions"]] == ["AAA"]
    status, body = _get(server, "/api/replay?id=" + urllib.request.quote(world["pid"], safe=""))
    payload = json.loads(body)
    assert status == 200 and payload["entry_run_id"] == R1 and "NaN" not in body
    status, body = _get(server, f"/api/run?module=meta_ranker&run_id={R2}&ticker=AAA&ref={R1}")
    assert status == 200 and json.loads(body)["row"]["action"] == "gated"
    assert "error" in json.loads(_get(server, "/api/replay?module=bogus&ticker=AAA")[1])
    assert _get(server, "/nope")[0] == 404


def test_non_finite_floats_become_null():
    from UI.decision_replay_dashboard import _finite
    assert _finite({"a": [float("nan"), 1.0, float("inf")], "b": "x"}) == {"a": [None, 1.0, None], "b": "x"}


def test_hub_lists_the_replay_page():
    from UI import hub_dashboard as hub
    app = hub.HubDashboardApp(port_replay=8779)
    assert any(d.key == "replay" and d.port == 8779 for d in app.dashboards)
    assert hub._adapt_replay({"positions": 12, "archive": {"a": "2026-11-18", "b": None}}) == {
        "state": "ready", "detail": "12 positions · decision archive through 2026-11-18"}
    assert hub._adapt_replay({})["state"] == "idle"


# --------------------------------------------------------------------------- contracts, setups, overlays

def test_contract_is_read_from_the_traded_occ_symbol():
    assert dr.parse_occ("SPY261105P00764000") == {"occ": "SPY261105P00764000", "underlying": "SPY",
                                                  "expiry": "2026-11-05", "right": "put", "strike": 764.0}
    assert dr.parse_occ("SPY") is None
    pos = {"route": "option", "ticker": "SPY", "entry_submitted_at": "2026-11-05T14:36:07Z",
           "entry_fill_price": 2.78, "entry_anchor": {"u_entry": 762.16, "u_atr": 2.0},
           "legs": [{"symbol": "SPY261105P00764000", "price": 11.05, "qty": 3.0}]}
    c = dr.contract_info(pos)
    assert c["right"] == "put" and c["strike"] == 764.0 and c["dte_at_entry"] == 0
    assert c["otm_pct_at_entry"] == pytest.approx(1 - 764 / 762.16, abs=1e-5)   # negative = in the money
    assert c["premium_return"] == pytest.approx(11.05 / 2.78 - 1, abs=1e-4)
    assert c["cost"] == pytest.approx(834.0)
    assert dr.contract_info({**pos, "route": "equity"}) is None
    assert dr.contract_info({"route": "option", "ticker": "SPY", "legs": []})["occ"] is None


def test_implied_levels_invert_the_momentum_feature_definitions_and_skip_clipped_values():
    c, atr, ema20, hi20, lo20 = 100.0, 2.0, 96.0, 104.0, 90.0
    feats = {"close": c, "atr_14": atr, "ema_dist_10": 1.0, "ema_dist_20": (c - ema20) / atr,
             "ema_dist_50": 10.0, "ema_dist_100": -3.0, "dist_20bar_high_atr": (hi20 - c) / atr,
             "range_pos_20": (c - lo20) / (hi20 - lo20), "dist_to_52w_high_atr": 50.0, "dist_to_52w_low_atr": 5.0}
    out = dr.implied_levels(feats)
    assert out["ema_20"] == pytest.approx(ema20) and out["high_20"] == pytest.approx(hi20)
    assert out["low_20"] == pytest.approx(lo20) and out["ema_100"] == pytest.approx(106.0)
    assert "ema_50" not in out and "high_52w" not in out        # at their clip bounds: no level implied
    assert out["low_52w"] == pytest.approx(90.0)
    assert dr.implied_levels({"close": 1.0, "atr_14": None}) == {}


def test_technical_overlays_use_the_momentum_formulas_on_full_history(world):
    bars = trade_library.price_series("AAA", timeframe="4h", days=0)["bars"]
    out = dr.technical_series("momentum_expansion", "AAA", "4h", bars[-4:])
    closes = pd.Series([b["c"] for b in bars])
    expect = closes.ewm(span=20, adjust=False).mean().iloc[-4:].round(6).tolist()
    assert out["series"]["ema_20"] == pytest.approx(expect)      # warmed up on the whole history
    assert out["meta"]["ema_20"]["model_input"] is True
    assert dr.technical_series("meta_ranker", "AAA", "4h", bars)["meta"]["ema_20"]["model_input"] is False
    assert dr.technical_series("momentum_expansion", "AAA", "1d", [])["series"] == {}


def test_setup_record_picks_the_entry_nearest_the_fill_and_draws_its_levels(tmp_path):
    log = tmp_path / "closed_setups.jsonl"
    rows = [{"setup_id": "SPY:short:x", "entry_time": "2026-11-04T15:00:00+00:00", "pivot": 1.0},
            {"setup_id": "SPY:short:x", "entry_time": "2026-11-05T14:35:00+00:00", "pivot": 763.76,
             "initial_invalidation": 764.16, "final_invalidation": 757.44, "targets": [755.3, 754.5],
             "setup_type": "gamma_structural_level_rejection"},
            {"setup_id": "SPY:long:y", "entry_time": "2026-11-05T14:35:00+00:00", "pivot": 9.0}]
    log.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows))
    rec = dr.setup_record("SPY:short:x", "2026-11-05T14:36:07Z", path=log)
    assert rec["pivot"] == 763.76
    keys = {lv["key"]: lv["price"] for lv in dr.setup_levels(rec)}
    assert keys == {"pivot": 763.76, "invalidation": 764.16, "target_1": 755.3, "target_2": 754.5}
    assert dr.setup_record("SPY:short:x", "2026-12-01T14:00:00Z", path=log) is None   # nothing within a day


def _write_1m(path, day, n, base, *, step=0.1, start_min=0):
    rows = []
    for i in range(n):
        t = pd.Timestamp(f"{day}T14:30:00Z") + pd.Timedelta(minutes=start_min + i)
        px = base + step * i
        rows.append({"symbol": "ZZZ", "timestamp": t.isoformat(), "open": px, "high": px + 0.05,
                     "low": px - 0.05, "close": px, "volume": 100.0, "arrival_at": t.isoformat()})
    with path.open("a") as fh:
        fh.write("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")


def test_engine_view_is_causal_and_matches_the_per_bar_series(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "BARS_1M_DIR", tmp_path)
    _write_1m(tmp_path / "bars_20261116.jsonl", "2026-11-16", 60, 50.0)            # prior session
    today = tmp_path / "bars_20261117.jsonl"
    _write_1m(today, "2026-11-17", 40, 60.0)
    at = "2026-11-17T14:59:00+00:00"                                               # bar 30 of the session
    view = dr.engine_view(ticker="ZZZ", at=at, setup=None, direction="long")
    assert view["prior_session"] == "2026-11-16"
    assert view["features"]["prior_day_high"] == pytest.approx(50.0 + 0.1 * 59 + 0.05)
    assert view["spot"] == pytest.approx(60.0 + 0.1 * 29)
    assert view["levels"] and all("price" in lv and "type" in lv for lv in view["levels"])
    bars, _ = dr.load_bars("ZZZ", "1m", _ts("2026-11-17T14:30:00"), _ts("2026-11-17T21:00:00"))
    series = dr.engine_series("ZZZ", bars)["series"]
    i = [b["t"] for b in bars].index(at)
    assert series["session_vwap"][i] == pytest.approx(view["features"]["session_vwap"], abs=1e-6)
    assert series["ema_9"][i] == pytest.approx(view["features"]["ema_9"], abs=1e-6)
    # later bars arriving must not change what the engine saw at that bar
    _write_1m(today, "2026-11-17", 30, 99.0, start_min=40)
    assert dr.engine_view(ticker="ZZZ", at=at, setup=None, direction="long")["features"] == view["features"]


def test_feature_series_reads_one_archived_feature_and_validates_names(world):
    out = dr.feature_series("meta_ranker", "AAA", "2026-11-16T00:00:00Z", "2026-11-21T00:00:00Z", "f1")
    assert sorted(r["value"] for r in out["rows"]) == [3.0, 15.0, 15.0]
    assert "error" in dr.feature_series("meta_ranker", "AAA", "2026-11-16T00:00:00Z", "2026-11-21T00:00:00Z", "f1;drop")
    assert "f1" in dr.archive_feature_names("meta_ranker", _ts("2026-11-16"), _ts("2026-11-21"))


def test_new_endpoints_and_the_static_page(server, world):
    from UI.decision_replay_dashboard import render_page
    page = render_page().decode()
    assert "__CONFIG__" not in page and "__NAV__" not in page and '"modules"' in page
    status, body = _get(server, "/api/feature_series?module=meta_ranker&ticker=AAA&name=f1"
                                "&start=2026-11-16T00:00:00Z&end=2026-11-21T00:00:00Z")
    assert status == 200 and len(json.loads(body)["rows"]) == 3
    status, body = _get(server, "/api/engine?ticker=AAA&at=2026-11-17T15:00:00Z")
    assert status == 200 and "error" in json.loads(body)          # no 1m archive in the fixture
    payload = json.loads(_get(server, "/api/replay?id=" + urllib.request.quote(world["pid"], safe=""))[1])
    assert set(payload["overlays"]["series"]) >= {"ema_20", "high_20", "low_20", "rsi_14"}
    assert payload["archive_features"] == ["f1", "f2"]
