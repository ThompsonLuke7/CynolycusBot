"""Split restatement of a managed claim (scripts/apply_split_to_claim.py)."""
from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "apply_split_to_claim", Path(__file__).resolve().parents[2] / "scripts" / "apply_split_to_claim.py"
)
split = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(split)

ENHA_ACTIONS = {"corporate_actions": {"reverse_splits": [
    {"ex_date": "2026-10-29", "id": "941197c9", "new_rate": 1, "old_rate": 10, "symbol": "ENHA"},
]}}


def test_the_enha_reverse_split_is_found_after_the_last_mark() -> None:
    found = split.splits_after(ENHA_ACTIONS, "ENHA", date(2026, 10, 28))
    assert len(found) == 1 and found[0]["ratio"] == pytest.approx(0.1)
    assert split.splits_after(ENHA_ACTIONS, "ENHA", date(2026, 10, 29)) == []
    assert split.splits_after(ENHA_ACTIONS, "OTHER", date(2026, 10, 1)) == []


def test_restate_scales_shares_and_prices_and_keeps_ratios() -> None:
    claim = {
        "route": "equity", "shares": 3436, "entry_avg_price": 1.28, "u_entry": 1.455,
        "u_atr": 0.0667, "last_mark_price": 1.19, "unrealized_gain": -0.0703,
        "peak_gain": -0.03125, "entry_bar": "2026-10-12 18:00:00+00:00",
        "mark_anomaly": {"ratio": 6.15},
    }
    out = split.restate(claim, 0.1)
    assert out["shares"] == pytest.approx(343.6)
    assert out["entry_avg_price"] == pytest.approx(12.8)
    assert out["u_entry"] == pytest.approx(14.55) and out["u_atr"] == pytest.approx(0.667)
    assert out["last_mark_price"] == pytest.approx(11.9)
    assert out["unrealized_gain"] == claim["unrealized_gain"]
    assert out["peak_gain"] == claim["peak_gain"]
    assert "mark_anomaly" not in out
    assert claim["shares"] == 3436  # input untouched
