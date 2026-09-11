"""Optional offline browser checks. Requires Playwright and its Chromium browser.

Run: python -m pytest UI/tests/test_hub_ui_browser.py -q
All snapshots and control responses are fixtures; downstream requests are intercepted.
Set CYNO_UI_SCREENSHOTS to a directory to save fixture-only visual review images.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from UI import hub_dashboard as hub
from scripts.build_architecture_atlas import build_atlas


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as manager:
        browser = manager.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def atlas(tmp_path_factory):
    output = tmp_path_factory.mktemp("browser-atlas") / "dist"
    build_atlas(repo_root=Path(__file__).resolve().parents[2], output_dir=output)
    return output / "public"


@pytest.fixture
def workspace(browser, atlas, monkeypatch):
    app = hub.HubDashboardApp(port_intraday_structure=8774)
    commands = []
    dashboards = [{"key": d.key, "name": d.name, "port": d.port,
                   "startable": d.startable, "stoppable": d.stoppable,
                   "tradeable": d.tradeable, "up": d.key != "dealer",
                   "state": "running" if d.startable else "ready",
                   "account_type": "paper" if d.tradeable else None,
                   "live_available": d.tradeable,
                   "detail": "Fixture snapshot · 4 open positions" if d.tradeable else "Fixture snapshot · research service",
                   "error": "Fixture endpoint unavailable" if d.key == "dealer" else None,
                   "performance": {"tracked_pnl": -345.5 if d.key == "spy" else 1240.25,
                                   "closed_trades": 18, "win_rate": 55.6}}
                  for d in app.dashboards]
    snapshot = {"dashboards": dashboards, "totals": {"equity": 125000, "open_positions": 16,
                "unrealized_pl": -245.5, "account_positions": 18}}
    monkeypatch.setattr(app, "snapshot", lambda: snapshot)
    monkeypatch.setattr(app, "start_one", lambda key, live: commands.append(("start", key, live)) or {"ok": True})
    monkeypatch.setattr(app, "stop_one", lambda key: commands.append(("stop", key)) or {"ok": True})
    monkeypatch.setattr(hub, "_ATLAS_PUBLIC", atlas)
    server = hub.HubHTTPServer(("127.0.0.1", 0), hub.HubHandler)
    server.app = app
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    context = browser.new_context(viewport={"width": 1512, "height": 1100}, reduced_motion="reduce")
    # No browser request can reach the real combined server, broker, or internet.
    context.route("**/*", lambda route: route.continue_() if urlsplit(route.request.url).netloc == urlsplit(base).netloc
                  else route.fulfill(status=200, content_type="text/html", body="Fixture preview"))
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    yield page, base, snapshot, commands, errors
    context.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def ready(page, base):
    page.goto(base)
    playwright.expect(page.locator("#connection")).to_contain_text("Connected")


def screenshot(page, name):
    if output := os.environ.get("CYNO_UI_SCREENSHOTS"):
        Path(output).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(output) / name), full_page=True)


def test_hub_filters_metrics_and_opt_in_previews(workspace):
    page, base, snapshot, _, errors = workspace
    ready(page, base)
    playwright.expect(page.locator("#equity")).to_have_text("$125,000")
    playwright.expect(page.locator("#upl")).to_have_text("−$245.50")
    assert page.locator("iframe").count() == 0
    screenshot(page, "hub-desktop.png")
    page.locator('[data-filter="attention"]').click()
    assert page.locator(".module-card:visible").count() == 1
    page.locator('[data-filter="all"]').click()
    page.locator("#module-search").fill("meta")
    assert page.locator(".module-card:visible").count() == 1
    assert not page.locator('[data-key="meta"] .start').is_visible()
    page.locator("#module-search").fill("does not exist")
    playwright.expect(page.locator("#empty")).to_be_visible()
    page.locator("#reset-filters").click()
    page.locator("#previews-toggle").check()
    assert page.locator("iframe").count() == len(snapshot["dashboards"])
    page.locator("#previews-toggle").uncheck()
    assert page.locator("iframe").count() == 0
    assert not errors


def test_hub_stale_snapshot_and_unavailable_account(workspace):
    page, base, snapshot, _, errors = workspace
    ready(page, base)
    page.route("**/api/state", lambda route: route.fulfill(status=503))
    page.locator("#refresh").click()
    playwright.expect(page.locator("#stale-notice")).to_contain_text("Snapshot is stale")
    playwright.expect(page.locator("#equity")).to_have_text("$125,000")
    playwright.expect(page.locator("#start-all")).to_be_disabled()
    page.unroute("**/api/state")
    for d in snapshot["dashboards"]:
        d["up"] = False
    snapshot["totals"] = {"open_positions": 0, "unrealized_pl": 0}
    page.locator("#refresh").click()
    playwright.expect(page.locator("#stale-notice")).to_be_hidden()
    playwright.expect(page.locator("#positions")).to_have_text("—")
    playwright.expect(page.locator("#upl")).to_have_text("—")
    assert not errors


def test_hub_controls_confirm_real_intent_and_preserve_focus(workspace):
    page, base, _, commands, errors = workspace
    ready(page, base)
    spy = page.locator('[data-key="spy"]')
    spy.locator(".start").click()
    playwright.expect(page.locator("#msg")).to_contain_text("request completed")
    assert commands == [("start", "spy", False)]
    spy.locator(".intent input").check()
    spy.locator(".intent input").focus()
    page.evaluate("document.getElementById('refresh').click()")
    playwright.expect(spy.locator(".intent input")).to_be_focused()
    page.once("dialog", lambda dialog: dialog.dismiss())
    spy.locator(".start").click()
    assert len(commands) == 1
    page.once("dialog", lambda dialog: dialog.accept())
    spy.locator(".start").click()
    playwright.expect(page.locator("#msg")).to_contain_text("request completed")
    assert commands[-1] == ("start", "spy", True)
    spy.locator(".intent input").uncheck()
    page.locator("#start-all").click()
    playwright.expect(page.locator("#msg")).to_contain_text("scheduled passes skipped")
    assert not any(command[1] in ("meta", "momentum", "dealer_ranker") for command in commands)
    assert not errors


@pytest.mark.parametrize("width", [390, 768, 1024])
def test_hub_responsive_layout(workspace, width):
    page, base, _, _, errors = workspace
    page.set_viewport_size({"width": width, "height": 844})
    ready(page, base)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    playwright.expect(page.locator("h1")).to_be_visible()
    assert page.locator(".module-card:visible").count() == 11
    if width == 390:
        screenshot(page, "hub-mobile.png")
    assert not errors


def test_atlas_navigation_search_filters_and_deep_link(workspace):
    page, base, _, _, errors = workspace
    page.goto(base + "/architecture/")
    playwright.expect(page.locator("#hud-nodes")).to_have_text(str(page.evaluate("ATLAS_DATA.datasets.public.nodes.filter(n => n.parent_id === 'system').length")).zfill(3))
    playwright.expect(page.locator("#graph-fallback")).to_be_hidden()
    assert page.locator("body").evaluate("el => el.classList.contains('holo-muted')")
    screenshot(page, "atlas-desktop.png")
    page.locator("#search-open").click()
    page.locator("#search-input").fill("Governance")
    page.locator(".search-result button").first.click()
    playwright.expect(page.locator("#inspector")).to_have_class("inspector open")
    title = page.locator("#inspector-title").text_content()
    url = page.url
    page.reload()
    playwright.expect(page.locator("#inspector-title")).to_have_text(title)
    playwright.expect(page.locator("#inspector")).to_have_class("inspector open")
    assert page.url == url
    screenshot(page, "atlas-detail.png")
    page.locator("#enter-domain").click()
    playwright.expect(page.locator("#back-button")).to_be_visible()
    page.locator("#back-button").click()
    page.locator("#filters-open").click()
    for checkbox in page.locator("#edge-filters input").all():
        checkbox.uncheck()
    playwright.expect(page.locator("#hud-routes")).to_have_text("000")
    page.locator("#filters-reset").click()
    page.keyboard.press("Escape")
    page.locator("#presentation-toggle").click()
    playwright.expect(page.locator("#presentation-toggle")).to_have_attribute("aria-pressed", "true")
    page.locator("#holo-toggle").click()
    assert not page.locator("body").evaluate("el => el.classList.contains('holo-muted')")
    assert not errors


def test_atlas_mobile_outline_and_inspector(workspace):
    page, base, _, _, errors = workspace
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(base + "/architecture/")
    playwright.expect(page.locator("#outline-panel")).to_have_class("outline-panel open")
    page.locator(".outline-node").nth(1).click()
    playwright.expect(page.locator("#inspector")).to_have_class("inspector open")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    screenshot(page, "atlas-mobile.png")
    page.locator("#inspector-close").click()
    playwright.expect(page.locator("#outline-panel")).to_be_visible()
    assert not errors


def test_hub_failed_control_does_not_retry_or_erase_feedback(workspace):
    page, base, _, commands, errors = workspace
    ready(page, base)
    requests = []

    def fail_control(route):
        requests.append(route.request.post_data_json)
        route.fulfill(status=503)

    page.route("**/api/start", fail_control)
    page.locator('[data-key="spy"] .start').click()
    playwright.expect(page.locator("#msg")).to_contain_text("may have reached")
    page.locator("#refresh").click()
    playwright.expect(page.locator("#msg")).to_contain_text("may have reached")
    assert requests == [{"key": "spy", "live": False}]
    assert not commands
    assert not errors


def test_atlas_resize_keeps_graph_inside_canvas(workspace):
    page, base, _, _, errors = workspace
    page.goto(base + "/architecture/#/system?selected=domain.governance")
    playwright.expect(page.locator("#inspector")).to_have_class("inspector open")
    page.set_viewport_size({"width": 1024, "height": 800})
    page.wait_for_function("""() => {
        const el = document.getElementById('cy'), cy = el._cyreg.cy;
        const b = cy.nodes().renderedBoundingBox();
        return b.x1 >= 0 && b.y1 >= 0 && b.x2 <= el.clientWidth && b.y2 <= el.clientHeight;
    }""")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors
