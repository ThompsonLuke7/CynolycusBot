"""Dealer archives every ranked name with how far its tradeable scan reached."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import strategies.dealer_positioning.live_ranked_options as dl
from core.decision_state_archive import load_decision_state


def _rankings(n=12) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [f"S{i:02d}" for i in range(n)],
        "snapshot_date": "2026-11-12",
        "dealer_swing_potential_score": np.linspace(1.0, 0.1, n),
        "dealer_swing_rank": np.arange(1, n + 1),
        "avg_vacuum_component": np.linspace(0, 1, n),
        "dealer_direction": "bullish",
        "scope_scores_json": "{}",
    })


def test_scan_window_and_tradeable_gates():
    cand, gates, actions = dl.dealer_cross_section(_rankings(), kept=["S00", "S03"], skipped=["S01", "S04"])
    assert gates["scan_reached"].tolist() == [True] * 5 + [False] * 7
    assert gates["tradeable_screen"].tolist() == [True, False, False, True] + [False] * 8
    assert actions["S01"] == "not_optionable" and actions["S03"] == "target"


def test_archive_dealer_decision(tmp_path):
    args = SimpleNamespace(side_mode="call", submit=False, route="equity", top_k=2, min_dte=2,
                           max_dte=21, refresh_chain=False)
    dl.archive_dealer_decision(args, rankings=_rankings(), ranking_path=tmp_path / "r.parquet",
                               kept=["S00", "S03"], skipped=["S01"], spot_map={"S00": 10.0},
                               bar=pd.Timestamp.now(tz="UTC").isoformat(), run_id="d" * 32,
                               inference_start=pd.Timestamp.now(tz="UTC"))
    df = load_decision_state(module="dealer_ranker")
    assert len(df) == 12 and sorted(df.loc[df["selected"], "ticker"]) == ["S00", "S03"]
    assert df.loc[df["ticker"] == "S02", "gate_reason"].iloc[0] == "tradeable_screen"
    assert df.loc[df["ticker"] == "S09", "gate_reason"].iloc[0] == "scan_reached"
    assert df.loc[df["ticker"] == "S00", "x__spot"].iloc[0] == 10.0
    assert "f__avg_vacuum_component" in df.columns and "f__dealer_swing_rank" not in df.columns
