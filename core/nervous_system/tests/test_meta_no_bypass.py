"""No-bypass enforcement for the Meta Ranker path (Task 23).

These are static scans on purpose. A runtime test only proves the paths it
happens to execute; an AST scan proves the capability is absent from the
source, which is the actual guarantee wanted here: Meta code must not be able
to reach a live account at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]

# Files the Meta path owns or can reach from a dashboard button.
META_RUNNER = REPO / "signals/meta_context/meta_ranker/live_runner.py"
META_LOOP = REPO / "signals/meta_context/meta_ranker/run_4h_loop.py"
META_DASHBOARD = REPO / "UI/meta_ranker_dashboard.py"
COMBINED_SERVER = REPO / "UI/combined_server.py"


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def string_constants(tree: ast.Module) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


# --------------------------------------------------------------------------
# The live route is gone
# --------------------------------------------------------------------------


def test_the_meta_dashboard_has_no_live_env_or_toggle() -> None:
    """A dashboard button must not be able to create a live route."""

    source = META_DASHBOARD.read_text(encoding="utf-8")

    assert "live_env_file" not in source
    assert "set_live" not in source
    assert "/api/set-live" not in source


def test_the_meta_dashboard_defines_no_set_live_method() -> None:
    tree = parse(META_DASHBOARD)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "set_live" not in names


def test_the_meta_dashboard_never_passes_live_to_the_loop() -> None:
    assert '"--live"' not in META_DASHBOARD.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [META_RUNNER, META_LOOP])
def test_meta_entry_points_reject_rather_than_forward_live(path: Path) -> None:
    """--live must be refused explicitly, not quietly accepted or ignored."""

    source = path.read_text(encoding="utf-8")

    assert "--live" in source, "the flag is kept so an old command fails loudly"
    assert "not accepted" in source
    assert "argparse.SUPPRESS" in source, "it must not be advertised in --help"


def test_the_meta_runner_selects_the_paper_profile_unconditionally() -> None:
    source = META_RUNNER.read_text(encoding="utf-8")

    assert 'profile = "PAPER"' in source
    assert 'profile = "LIVE" if args.live else "PAPER"' not in source


def test_the_combined_server_never_launches_meta_live() -> None:
    source = COMBINED_SERVER.read_text(encoding="utf-8")

    # The scheduled and pre-open Meta passes are pinned to live=False.
    assert "live=meta_ranker_live" not in source
    assert "--meta-ranker-live is not supported" in source


def test_the_combined_server_builds_the_meta_dashboard_without_a_live_env() -> None:
    source = COMBINED_SERVER.read_text(encoding="utf-8")
    start = source.index("meta_app = MetaRankerDashboardApp(")
    block = source[start : start + 200]

    assert "live_env_file" not in block
    assert "submit=False" in block, "the Meta page defaults to read-only"


# --------------------------------------------------------------------------
# Meta code must not construct a raw broker client for submission
# --------------------------------------------------------------------------


def broker_constructions(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "AlpacaOptionsClient":
                lines.append(node.lineno)
    return lines


def submit_calls(tree: ast.Module) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None)
            if name in {"submit_order", "submit_option_order", "submit_multileg_order"}:
                found.append((name, node.lineno))
    return found


def test_the_meta_loop_constructs_no_broker_client() -> None:
    """The governed loop orchestrates subprocesses; it never trades directly."""

    assert broker_constructions(parse(META_LOOP)) == []
    assert submit_calls(parse(META_LOOP)) == []


def test_the_meta_dashboard_submits_nothing_directly() -> None:
    """The page may read the broker, but must never place an order."""

    assert submit_calls(parse(META_DASHBOARD)) == []


def test_the_remaining_direct_submit_sites_are_known() -> None:
    """Pin the cutover surface still to be routed through the gateway.

    This is a deliberate inventory, not an approval: it fails when a new
    direct submit appears, and must shrink to empty as Task 23 completes.
    """

    pending = {
        # live_runner reached zero: every order now goes through
        # DecisionCoordinator -> ExecutionGateway.
        REPO / "signals/meta_context/meta_ranker/live_runner.py": 0,
        REPO / "core/live_4h_exec.py": 6,
    }
    for path, expected in pending.items():
        actual = len(submit_calls(parse(path)))
        assert actual == expected, (
            f"{path.name} has {actual} direct submit calls, expected {expected}; "
            "route new ones through DecisionCoordinator -> ExecutionGateway"
        )
