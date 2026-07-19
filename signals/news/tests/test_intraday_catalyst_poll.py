from __future__ import annotations

import json

import pytest

from scripts.intraday_catalyst_poll import load_universe


pytestmark = pytest.mark.safe


def test_load_universe_supports_curated_json_mapping(tmp_path):
    path = tmp_path / "universe.json"
    path.write_text(json.dumps({"msft": {"tier": 1}, "AAPL": {"tier": 2}}))

    assert load_universe(path) == ["AAPL", "MSFT"]
