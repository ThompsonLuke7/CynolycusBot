"""decision_state_v1: one immutable row per scored candidate, never breaking a run."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core import decision_state_archive as dsa

BAR = pd.Timestamp("2026-11-12 19:00", tz="UTC")
NOW = pd.Timestamp("2026-11-12 21:05", tz="UTC")


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["aaa", "BBB", "CCC", "DDD"],
        "score": [0.9, 0.4, 0.7, np.nan],
        "f1": [1.0, 2.0, 3.0, np.nan],
        "f2": [10, 20, 30, 40],
        "liq": [0.9, 0.2, 0.8, 0.9],
        "label": ["x", "y", "z", "w"],
    })


def _archive(tmp_path, **kw):
    cand = kw.pop("candidates", _candidates())
    args = dict(
        candidates=cand, bar_ts=BAR, score_col="score", feature_cols=["f1", "f2", "absent"],
        gates={"complete": cand["score"].notna(), "liquidity": cand["liq"] >= 0.5},
        selected=["AAA"], extra_cols=["label"], decision_ts=NOW, inference_start=NOW,
        inputs_asof={"matrix": "2026-11-12T20:00:00+00:00"}, run_meta={"pass_kind": "scheduled"},
    )
    args.update(kw)
    return dsa.archive_decision_state(module="unit", root=tmp_path, strict=True, **args)


def test_one_row_per_candidate_with_gates_rank_and_features(tmp_path):
    res = _archive(tmp_path)
    df = pd.read_parquet(res.path)
    assert len(df) == 4 and res.rows == 4
    assert df["ticker"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    # Gates are recorded per row; the first failing gate is the reason.
    assert df.loc[1, "gate_reason"] == "liquidity" and df.loc[3, "gate_reason"] == "complete"
    assert pd.isna(df.loc[0, "gate_reason"]) and pd.isna(df.loc[2, "gate_reason"])
    assert df["eligible"].tolist() == [True, False, True, False]
    # Rank is among ELIGIBLE rows only, best score first.
    assert df.loc[0, "rank"] == 1 and df.loc[2, "rank"] == 2
    assert df.loc[[1, 3], "rank"].isna().all()
    assert df["selected"].tolist() == [True, False, False, False]
    assert df["action"].tolist() == ["selected", "gated", "eligible", "gated"]
    # Full feature vector as float32; a model feature the frame lacks is kept as NaN
    # and named in run_meta, not silently dropped.
    assert df["f__f1"].dtype == np.float32 and df["f__absent"].isna().all()
    assert json.loads(df.loc[0, "run_meta_json"])["missing_features"] == ["absent"]
    assert df.loc[0, "x__label"] == "x"
    assert df.loc[0, "schema_version"] == "decision_state_v1"
    assert df["run_id"].nunique() == 1 and df["bar_ts"].iloc[0] == BAR


def test_partition_layout_is_module_then_et_date(tmp_path):
    res = _archive(tmp_path)
    rel = res.path.relative_to(tmp_path)
    assert rel.parts[0] == "unit" and rel.parts[1] == "2026-11-12"
    assert rel.name.startswith("160500_")  # 21:05 UTC = 16:05 ET


def test_files_are_never_overwritten(tmp_path):
    res = _archive(tmp_path)
    frame = pd.read_parquet(res.path)
    with pytest.raises(FileExistsError):
        dsa.write_immutable(frame, res.path)
    assert not list(res.path.parent.glob(".*.tmp"))


def test_naive_or_future_timestamps_are_refused(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        _archive(tmp_path, bar_ts=datetime(2026, 11, 12, 19, 0))
    with pytest.raises(ValueError, match="future bar"):
        _archive(tmp_path, bar_ts=NOW + pd.Timedelta(hours=1))


def test_failure_is_logged_not_raised_without_strict(tmp_path, caplog):
    bad = pd.DataFrame({"ticker": ["A", "A"], "score": [1.0, 2.0]})
    out = dsa.archive_decision_state(module="unit", root=tmp_path, candidates=bad, bar_ts=BAR,
                                     score_col="score", decision_ts=NOW)
    assert out is None
    assert "archiving FAILED" in caplog.text
    assert not list(tmp_path.rglob("*.parquet"))


def test_switch_disables_writes(tmp_path, monkeypatch):
    monkeypatch.setenv(dsa.ENV_SWITCH, "0")
    assert _archive(tmp_path) is None
    assert not list(tmp_path.rglob("*.parquet"))


def test_env_root_is_used_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv(dsa.ENV_ROOT, str(tmp_path / "env_root"))
    res = dsa.archive_decision_state(module="unit", strict=True, candidates=_candidates(),
                                     bar_ts=BAR, score_col="score", decision_ts=NOW)
    assert str(res.path).startswith(str(tmp_path / "env_root"))


def test_hashes_track_content(tmp_path):
    m = tmp_path / "model.json"
    m.write_text("a")
    h1 = dsa.model_hash([m])
    m.write_text("b")
    assert dsa.model_hash([m]) != h1
    assert dsa.model_hash([tmp_path / "nope.json"]) is not None  # missing is recorded
    assert dsa.universe_hash(["b", "A"]) == dsa.universe_hash(["a", "B"])


def test_stamp_run_id_reaches_every_signal_audit():
    audits = {"AAA": {"ticker": "AAA"}, "BBB": {"ticker": "BBB"}}
    dsa.stamp_run_id(audits, "run123")
    assert all(a["decision_run_id"] == "run123" for a in audits.values())


def test_load_decision_state_unions_runs(tmp_path):
    _archive(tmp_path)
    _archive(tmp_path, decision_ts=NOW + pd.Timedelta(minutes=1))
    df = dsa.load_decision_state(tmp_path, module="unit", start="2026-11-12", end="2026-11-12")
    assert len(df) == 8 and df["run_id"].nunique() == 2
    assert dsa.load_decision_state(tmp_path, module="unit", start="2026-11-13").empty


def test_intraday_structure_events_map_to_the_same_schema(tmp_path):
    path = tmp_path / "decision_events.jsonl"
    events = [
        {"event_type": "setup_abstention", "event_id": "e1", "recorded_at": "2026-11-11T21:01:59+00:00",
         "event_time": "2026-11-11T21:01:00+00:00",
         "payload": {"ticker": "SMH", "direction": "long", "confidence": 0.6, "atr": 0.5,
                     "no_trade_reason": "invalidation_wider_than_max_atr", "trapped_between_levels": False,
                     "context_regime": "TRENDING_UP", "evidence": ["a"], "timestamp": "2026-11-11T21:01:00+00:00",
                     "setup_id": "SMH:long:x"}},
        {"event_type": "setup_transition", "event_id": "e2", "recorded_at": "2026-11-11T21:06:59+00:00",
         "event_time": "2026-11-11T21:06:00+00:00",
         "payload": {"ticker": "SNDK", "setup_id": "SNDK:long:vwap", "from_state": "ARMED",
                     "to_state": "CONFIRMED", "spot": 982.0, "timestamp": "2026-11-11T21:06:00+00:00"}},
        {"event_type": "candidate_refreshed", "event_id": "e3", "payload": {}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    df = dsa.intraday_structure_decision_state(path)
    assert len(df) == 2
    ab, tr = df.iloc[0], df.iloc[1]
    assert ab["action"] == "abstain" and ab["gate_reason"] == "invalidation_wider_than_max_atr"
    assert ab["f__atr"] == 0.5 and ab["f__trapped_between_levels"] == 0.0 and not ab["selected"]
    assert tr["action"] == "transition:CONFIRMED" and tr["selected"] and tr["side"] == "long"
    assert dsa.intraday_structure_decision_state(path, start="2026-11-12").empty


def test_order_plan_actions_join_back_by_run_id(tmp_path):
    log = tmp_path / "audit.jsonl"
    events = [
        {"event": "order_plan", "module": "meta_ranker", "bar": "2026-11-12T19:00:00+00:00", "submit": True,
         "signal_audits": {"AAA": {"decision_run_id": "run1"}, "BBB": {"decision_run_id": "run1"}},
         "contract_selection": {
             "AAA": {"action": "option", "occ": "AAA261218C00050000", "ref_price": 50.0,
                     "quote": {"bid": 1.0, "ask": 1.2}},
             "BBB": {"action": "skip", "reason": "no_ref_price"}},
         "planned": [{"symbol": "AAA261218C00050000", "side": "buy", "qty": 3, "disposition": "submitted"}]},
        {"event": "order_plan", "module": "meta_ranker", "signal_audits": {"OLD": {}},
         "contract_selection": {"OLD": {"action": "equity"}}},  # pre-archive row: no run id
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    df = dsa.order_plan_actions(log).set_index("ticker")
    assert sorted(df.index) == ["AAA", "BBB"]
    assert df.loc["AAA", "route_action"] == "option" and df.loc["AAA", "planned_qty"] == 3
    assert df.loc["AAA", "disposition"] == "submitted" and json.loads(df.loc["AAA", "quote_json"])["ask"] == 1.2
    assert df.loc["BBB", "route_reason"] == "no_ref_price" and pd.isna(df.loc["BBB", "planned_qty"])
