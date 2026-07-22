"""Tests for the combined-server hub dashboard."""
from __future__ import annotations

import json

import pytest

from UI.hub_dashboard import HubDashboardApp, _Dash, _json_safe
from UI.ui_chrome import NAV_HTML

pytestmark = pytest.mark.safe


def test_shared_navigation_includes_intraday_structure_dashboard():
    assert '"Intraday Structure",8774' in NAV_HTML


def test_json_safe_replaces_nonfinite_values():
    payload = {
        "ok": True,
        "bad": float("nan"),
        "nested": [1.0, float("inf"), {"x": float("-inf")}],
    }

    encoded = json.dumps(_json_safe(payload), allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded == {"ok": True, "bad": None, "nested": [1.0, None, {"x": None}]}


def test_snapshot_with_nonfinite_account_values_remains_valid_json(monkeypatch):
    app = HubDashboardApp()
    app.dashboards = [
        _Dash(
            "fake",
            "Fake Module",
            1,
            startable=True,
            stoppable=True,
            tradeable=True,
            start_path="/api/start",
            start_body=lambda live: {},
            adapt=lambda state: {
                "state": "ready",
                "detail": "fixture",
                "account_type": "paper",
                "acct": {
                    "equity": float("nan"),
                    "n_positions": 1,
                    "upl": float("inf"),
                    "account_n_positions": 2,
                    "account_upl": float("-inf"),
                },
            },
        )
    ]
    monkeypatch.setattr(app, "_request", lambda *args, **kwargs: {"ok": True})

    snapshot = app.snapshot()
    encoded = json.dumps(_json_safe(snapshot), allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["dashboards"][0]["up"] is True
    assert decoded["dashboards"][0]["state"] == "ready"
    assert decoded["totals"]["equity"] is None
    assert decoded["totals"]["unrealized_pl"] == 0.0


def test_default_hub_includes_amethyst_dashboard():
    app = HubDashboardApp()

    keys = {dash.key for dash in app.dashboards}

    assert "amethyst" in keys


def test_start_all_skips_one_shot_4h_loops(monkeypatch):
    app = HubDashboardApp()
    started: list[str] = []
    monkeypatch.setattr(app, "start_one", lambda key, live: started.append(key) or {"key": key, "ok": True})

    result = app.start_all({})

    assert "spy" in started
    assert "swing" in started
    assert "dealer" in started
    assert "meta" not in started
    assert "momentum" not in started
    skipped = {r["key"]: r for r in result["results"] if r.get("skipped")}
    assert skipped["meta"]["reason"] == "scheduled_4h_loop"
    assert skipped["momentum"]["reason"] == "scheduled_4h_loop"
