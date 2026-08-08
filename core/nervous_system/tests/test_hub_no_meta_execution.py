"""The hub must not be able to start or fund Meta execution (Task 25).

A dashboard button is the least governed path into a trading system: it has no
snapshot, no policy evaluation, and no audit record. Meta's execution is now
owned by DecisionCoordinator -> ExecutionGateway, so the hub's job is to
*show* state, not to start it or choose which account it runs against.

Static assertions on purpose. A runtime test only proves the routes it happens
to exercise; a source scan proves the capability is absent.
"""

from __future__ import annotations

import ast
from pathlib import Path


HUB = Path(__file__).resolve().parents[3] / "UI/hub_dashboard.py"
SOURCE = HUB.read_text(encoding="utf-8")


def _dashboard_flags(key: str) -> dict[str, object]:
    """Read the _Dash(...) declaration for one module out of the source."""

    tree = ast.parse(SOURCE, filename=str(HUB))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_Dash"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != key:
            continue
        flags: dict[str, object] = {}
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant):
                flags[keyword.arg] = keyword.value.value
            elif isinstance(keyword.value, ast.Name) and keyword.value.id in {"None"}:
                flags[keyword.arg] = None
        return flags
    raise AssertionError(f"no _Dash declaration for {key}")


def test_the_hub_declares_a_meta_panel_at_all() -> None:
    """Guards the assertions below: a missing panel would pass them vacuously."""

    assert _dashboard_flags("meta")


def test_meta_cannot_be_started_from_the_hub() -> None:
    """Starting a 4H order pass from a button bypasses every governed check."""

    assert _dashboard_flags("meta").get("startable") is False


def test_meta_is_not_marked_tradeable_in_the_hub() -> None:
    """`tradeable` is what renders the account chooser; Meta has no choice to
    make any more.
    """

    assert _dashboard_flags("meta").get("tradeable") is False


def test_the_hub_never_sets_a_live_account_for_meta() -> None:
    """`/api/set-live` was removed from the Meta dashboard in the cutover, so
    calling it for Meta is both dead and dangerous.

    Momentum and dealer_ranker still serve that endpoint and are not part of
    this cutover, so the scan is scoped to the Meta branch rather than banning
    the string outright — banning it would have severed two working modules.
    """

    tree = ast.parse(SOURCE, filename=str(HUB))
    start_one = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "start_one"
    )
    guarded = [
        node for node in ast.walk(start_one)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "key"
        and any(
            isinstance(c, ast.Constant) and c.value == "meta" for c in node.comparators
        )
    ]
    assert guarded, "start_one must refuse meta before any account call"

    body = ast.get_source_segment(SOURCE, start_one) or ""
    meta_guard_index = body.index('d.key == "meta"')
    set_live_index = body.index("/api/set-live")
    assert meta_guard_index < set_live_index, "meta must be refused first"


def test_the_meta_panel_offers_no_account_toggle() -> None:
    """The real-money toggle is rendered only for `tradeable` panels, so Meta
    not being tradeable is what removes it.

    Scoped to Meta on purpose: the other modules are not part of this cutover,
    and silently changing their account controls would be a change nobody asked
    for.
    """

    assert _dashboard_flags("meta").get("tradeable") is False
    assert "Run + SUBMIT" not in SOURCE


def test_the_meta_panel_never_advertises_a_live_account() -> None:
    import re

    body = re.search(r"def _adapt_meta.*?\n\n\ndef ", SOURCE, re.S)
    assert body is not None
    assert '"real money"' not in body.group(0)
    assert "live_available\": False" in body.group(0).replace("'", '"')


def test_the_meta_panel_reports_the_governed_mode() -> None:
    """An operator looking at the hub should be able to tell whether the policy
    is merely observing or actually enforcing.

    Asserted on the emitted payload key, not on the word appearing somewhere in
    the file: a local variable of the same name would satisfy that and show the
    operator nothing.
    """

    import re

    body = re.search(r"def _adapt_meta.*?\n\n\ndef ", SOURCE, re.S)
    assert body is not None
    assert '"policy_mode": policy_mode' in body.group(0)
    assert "policy_mode" in body.group(0).split("out = {")[1]


def test_production_live_is_shown_as_blocked_rather_than_available() -> None:
    assert "BLOCKED BY MVP POLICY" in SOURCE


def test_no_dashboard_declaration_starts_meta_execution() -> None:
    tree = ast.parse(SOURCE, filename=str(HUB))
    started = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_Dash"
        for keyword in node.keywords
        if keyword.arg == "start_path" and isinstance(keyword.value, ast.Constant)
    }

    assert "/api/run-loop" not in started or _dashboard_flags("meta").get("startable") is False
