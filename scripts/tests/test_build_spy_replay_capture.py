import json

import pandas as pd

from scripts.build_spy_replay_capture import materialize_capture


def test_materialize_capture_deduplicates_latest_rows(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "live_1m_bars.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "one_minute_bar", "symbol": "SPY", "bar": {"timestamp": "2026-07-28T13:30:00Z", "close": 737.0}}),
            json.dumps({"event": "one_minute_bar", "symbol": "SPY", "bar": {"timestamp": "2026-07-28T13:30:00Z", "close": 738.0}}),
        ]) + "\n"
    )
    (capture / "phase4_decisions.jsonl").write_text(json.dumps({
        "event": "phase4_decision_input", "symbol": "SPY",
        "bar": {"timestamp": "2026-07-28T13:30:00Z", "open": 737.0, "high": 739.0, "low": 736.0, "close": 738.0},
        "probs": {"p_enter_long": .8, "p_enter_short": .2}, "thresholds": {"enter_long": .7},
        "raw_action": 1.0, "exec_pos": 1, "gate_status": "meta_direct",
    }) + "\n")
    out = tmp_path / "out"
    assert materialize_capture(capture, out) == {"one_minute_bars": 1, "phase4_decisions": 1}
    bars = pd.read_parquet(out / "spy_intraday_1min.parquet")
    decisions = pd.read_parquet(out / "phase4_signal_frame.parquet")
    assert bars.loc[0, "close"] == 738.0
    assert decisions.loc[0, "p_enter_long"] == .8
    assert decisions.loc[0, "thr_enter_long"] == .7
