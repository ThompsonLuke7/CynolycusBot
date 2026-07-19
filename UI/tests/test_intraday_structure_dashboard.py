from __future__ import annotations

import queue
from dataclasses import replace

from UI.intraday_structure_dashboard import IntradayStructureDashboardApp
from strategies.intraday_structure.config import IntradayStructureConfig


def test_intraday_structure_dashboard_state_and_manual_candidate(tmp_path) -> None:
    base = IntradayStructureConfig()
    config = replace(
        base,
        state_path=str(tmp_path / "state.json"),
        signal_path=str(tmp_path / "signals.json"),
        transition_log_path=str(tmp_path / "transitions.jsonl"),
        manual_watchlist=(),
    )
    app = IntradayStructureDashboardApp(config, queue.Queue())
    try:
        state = app.snapshot()
        assert state["paper_only"] is True
        added = app.add_candidate({"ticker": "MU", "direction": "long", "score": 0.8})
        assert added["ok"] is True
        assert added["candidate"]["ticker"] == "MU"
    finally:
        app.stop()
