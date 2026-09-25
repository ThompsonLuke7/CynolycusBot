from __future__ import annotations

import json

import pandas as pd

from scripts.discord_ledger.capture_ace_candidate_tape import build_candidates, capture_once


def _bars() -> pd.DataFrame:
    times = pd.date_range("2026-09-25T13:30:00Z", periods=32, freq="min")
    return pd.DataFrame({"symbol": "SPY", "timestamp": times, "open": range(100, 132),
                         "high": range(101, 133), "low": range(99, 131), "close": range(100, 132),
                         "volume": [100] * 32, "vwap": range(100, 132)})


def test_build_candidates_emits_two_directions_and_never_imputes_discord_context(tmp_path) -> None:
    snapshot = tmp_path / "universe.csv.gz"
    snapshot.write_bytes(b"fixture")
    rows = build_candidates(bars=_bars(), selected_symbols=[("SPY", 8), ("QQQ", 2)],
                            decision=pd.Timestamp("2026-09-25T14:02:00Z"), capture_at=pd.Timestamp("2026-09-25T14:02:01Z"),
                            run_id="run", snapshot_taken=pd.Timestamp("2026-09-25T14:00:00Z"), snapshot_path=snapshot,
                            snapshot_hash="hash", eligible_symbols={"SPY", "QQQ"})
    assert len(rows) == 4
    spy = [row for row in rows if row["symbol"] == "SPY"]
    assert {row["direction_sign"] for row in spy} == {-1, 1}
    assert spy[0]["discord_context_status"] == "not_captured_in_collector"
    assert spy[0]["observed_ace_alert"] is None
    assert {row["feature_status"] for row in rows if row["symbol"] == "QQQ"} == {"missing_symbol_bars"}


def test_capture_once_persists_immutable_run_without_network(tmp_path) -> None:
    alerts = tmp_path / "alerts.jsonl"
    alerts.write_text(json.dumps({"symbol": "SPY", "direction_sign": 1}) + "\n", encoding="utf-8")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    universe = snapshots / "shared_universe_20260925T130000Z.csv.gz"
    pd.DataFrame({"ticker": ["SPY"], "is_eligible": [True]}).to_csv(universe, index=False, compression="gzip")
    result = capture_once(now=pd.Timestamp("2026-09-25T14:02:00Z"), alerts_path=alerts, snapshot_dir=snapshots,
                          out_root=tmp_path / "out", max_symbols=1, env_file=None,
                          fetcher=lambda *_args, **_kwargs: _bars())
    assert result["status"] == "captured"
    path = tmp_path / "out" / "runs"
    directories = list(path.iterdir())
    assert len(directories) == 1
    assert (directories[0] / "candidates.jsonl").is_file()
    assert (directories[0] / "iex_bars.parquet").is_file()
