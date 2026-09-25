from __future__ import annotations

import pandas as pd
import pytest

from scripts.discord_ledger.audit_ace_candidate_tape_coverage import coverage_record, summarize


def _alert() -> dict:
    return {
        "decision_at_utc": "2026-09-17T15:00:00Z", "symbol": "SPY", "direction_sign": 1,
        "source_message_ids": ["m1"], "source_session_file": "fixture.parquet",
    }


def test_partial_bars_cannot_be_called_full_candidate_tape(tmp_path) -> None:
    snapshot = tmp_path / "shared_universe_20260917T140000Z.csv.gz"
    pd.DataFrame({"ticker": ["SPY", "QQQ"], "is_eligible": [True, True]}).to_csv(snapshot, index=False, compression="gzip")
    row = coverage_record(_alert(), snapshot=(pd.Timestamp("2026-09-17T14:00:00Z"), snapshot), bars={"SPY"})
    assert row["eligible_bar_coverage"] == pytest.approx(0.5)
    assert not row["full_two_direction_candidate_tape_possible"]
    assert summarize([row])["conclusion"] == "retrospective_full_candidate_tape_not_available_from_current_inputs"


def test_complete_universe_requires_all_eligible_symbols(tmp_path) -> None:
    snapshot = tmp_path / "shared_universe_20260917T140000Z.csv.gz"
    pd.DataFrame({"ticker": ["SPY"], "is_eligible": [True]}).to_csv(snapshot, index=False, compression="gzip")
    row = coverage_record(_alert(), snapshot=(pd.Timestamp("2026-09-17T14:00:00Z"), snapshot), bars={"SPY", "OTHER"})
    assert row["full_two_direction_candidate_tape_possible"]
