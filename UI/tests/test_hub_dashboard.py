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
    # Momentum is skipped because its 4H pass is scheduled. Meta is absent for
    # a stronger reason: it is no longer startable from the hub at all, since
    # its execution is owned by the governed path. A "skipped" entry would
    # imply the hub could start it under other circumstances.
    assert skipped["momentum"]["reason"] == "scheduled_4h_loop"
    assert "meta" not in skipped


@pytest.mark.parametrize("request_path", [
    "/architecture/../local/index.html",
    "/architecture/%2e%2e/local/index.html",
    "/architecture/%2fetc/passwd",
    "/architecture/missing.js",
    "/architecture/leaked.html",
])
def test_atlas_asset_never_exposes_files_outside_public_build(tmp_path, monkeypatch, request_path):
    from UI import hub_dashboard as hub

    public = tmp_path / "public"
    public.mkdir()
    local = tmp_path / "local"
    local.mkdir()
    private = local / "index.html"
    private.write_text("local-only evidence")
    (public / "leaked.html").symlink_to(private)
    monkeypatch.setattr(hub, "_ATLAS_PUBLIC", public)

    assert hub._atlas_asset(request_path) is None


def test_atlas_asset_resolves_public_home_and_nested_fonts(tmp_path, monkeypatch):
    from UI import hub_dashboard as hub

    (tmp_path / "fonts").mkdir()
    (tmp_path / "index.html").write_text("public architecture")
    font = tmp_path / "fonts/SpaceGrotesk.woff2"
    font.write_bytes(b"fixture")
    monkeypatch.setattr(hub, "_ATLAS_PUBLIC", tmp_path)

    assert hub._atlas_asset("/architecture/") == tmp_path / "index.html"
    assert hub._atlas_asset("/architecture/fonts/SpaceGrotesk.woff2") == font
