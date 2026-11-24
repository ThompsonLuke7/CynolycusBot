"""Parity compare: float32 archive vs research file, NaN handled, mismatches counted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.decision_state.parity_check import compare_vectors

BAR = pd.Timestamp("2026-11-12 19:00", tz="UTC")


def test_compare_vectors_counts_real_disagreements_only():
    archived = pd.DataFrame({
        "bar_ts": [BAR] * 4, "ticker": ["A", "B", "C", "D"],
        "f__x": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
        "f__y": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "f__only_live": [1.0, 1.0, 1.0, 1.0],
    })
    reference = pd.DataFrame({
        "bar_ts": [BAR] * 3, "ticker": ["A", "B", "C"],
        "x": [1.0, 2.5, np.nan],                    # B disagrees, C NaN on both sides
        "y": [0.1, 0.2, 0.3 + 1e-9],                # float32 rounding is not a mismatch
    })
    table, cov = compare_vectors(archived, reference, keys=["bar_ts", "ticker"])
    t = table.set_index("feature")
    assert cov["matched_rows"] == 3 and cov["features_not_in_reference"] == ["only_live"]
    assert t.loc["x", "mismatch"] == 1 and t.loc["y", "mismatch"] == 0
    assert np.isclose(t.loc["x", "share"], 1 / 3)


def _swing_rows(stamps, tickers, *, synced, selected=None):
    import json
    return pd.DataFrame({
        "decision_ts": [pd.Timestamp("2026-11-24 15:40", tz="UTC")] * len(tickers),
        "bar_ts": [pd.Timestamp(s, tz="America/New_York").tz_convert("UTC") for s in stamps],
        "ticker": tickers, "selected": selected or [False] * len(tickers),
        "run_meta_json": [json.dumps({"bar_sync": {}} if synced else {})] * len(tickers),
        "f__ret_1": np.float32(0.01),
    })


def test_swing_30m_keys_map_live_stamps_to_bucket_start():
    from scripts.decision_state.parity_check import normalize_keys
    rows = _swing_rows(["2026-11-24 10:29", "2026-11-24 10:27", "2026-11-24 10:00"], ["A", "B", "C"], synced=True)
    out = normalize_keys("multi_ticker_swing_30m", rows)
    assert set(out["bar_ts"]) == {pd.Timestamp("2026-11-24 10:00", tz="America/New_York").tz_convert("UTC")}
    assert normalize_keys("meta_ranker", rows)["bar_ts"].equals(rows["bar_ts"])


def test_swing_30m_sample_keeps_selected_and_is_reproducible():
    from scripts.decision_state.parity_check import sample_tickers
    tick = [f"T{i:03d}" for i in range(200)]
    rows = _swing_rows(["2026-11-24 10:29"] * 200, tick, synced=True,
                       selected=[t in ("T150", "T199") for t in tick])
    a, b = sample_tickers(rows, 10, seed=1), sample_tickers(rows, 10, seed=1)
    assert a == b and len(a) == 10 and {"T150", "T199"} <= set(a)
    assert sample_tickers(rows, 0) == tick


def test_swing_30m_pre_fix_rows_are_excluded_not_compared(tmp_path, monkeypatch):
    from scripts.decision_state import parity_check as pc
    rows = pd.concat([_swing_rows(["2026-11-24 10:29"], ["OLD"], synced=False),
                      _swing_rows(["2026-11-24 10:29"], ["NEW"], synced=True)], ignore_index=True)
    monkeypatch.setattr(pc, "load_decision_state", lambda root, module, start: rows.copy())
    seen = {}

    def ref_for(module, archived):
        seen["tickers"] = sorted(archived["ticker"])
        r = archived[["bar_ts", "ticker"]].assign(ret_1=0.01)
        return r, ["bar_ts", "ticker"], "stub"

    monkeypatch.setattr(pc, "reference_for", ref_for)
    (tmp_path / "multi_ticker_swing_30m").mkdir()
    report = pc.run(days=7, root=tmp_path, max_mismatch=0.01, modules=["multi_ticker_swing_30m"])
    m = report["modules"]["multi_ticker_swing_30m"]
    assert seen["tickers"] == ["NEW"] and m["excluded_pre_fix_rows"] == 1 and m["status"] == "PASS"
