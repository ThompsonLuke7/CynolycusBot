from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.collect_news_scope as scope


pytestmark = pytest.mark.safe


def test_priority_scope_bounds_momentum_by_composite(monkeypatch, tmp_path):
    swing = tmp_path / "swing.json"
    swing.write_text(json.dumps({"AAA": {}, "BBB": {}}))
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    pd.DataFrame(
        {
            "ticker": ["LOW", "TOP", "MID"],
            "composite": [0.1, 0.9, 0.5],
        }
    ).to_csv(snapshots / "universe_2026-07-17.csv", index=False)

    monkeypatch.setattr(scope, "SWING_UNIVERSE", swing)
    monkeypatch.setattr(scope, "MOMENTUM_SNAPSHOT_DIR", snapshots)

    assert scope._priority_universe(momentum_limit=2) == ["AAA", "BBB", "MID", "TOP"]
