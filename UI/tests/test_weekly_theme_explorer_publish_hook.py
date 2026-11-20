"""The emerging-themes + theme-explorer stage runs weekly (moved from the nightly 2026-11-17)."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.safe

REPO_ROOT = Path(__file__).resolve().parents[2]


def _theme_block() -> str:
    source = (REPO_ROOT / "scripts/weekly_refresh.sh").read_text(encoding="utf-8")
    start = source.index('echo "[$(ts)] 6/7 emerging themes + theme explorer"')
    end = source.index("  # 7) Momentum shadow ledger", start)
    return source[start:end]


def test_weekly_runs_emerging_then_builds_before_publishing() -> None:
    block = _theme_block()
    emerging = '"$PYTHON" -u -m themes.dynamic_theme.emerging --limit 300'
    builder = '"$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_explorer'
    publisher = '"$PYTHON" -u scripts/publish_theme_explorer.py'

    assert block.index(emerging) < block.index(builder) < block.index(publisher)
    assert 'echo "[$(ts)] theme explorer build exit=$rc"' in block
    # publication only after a successful build
    assert block.index('echo "[$(ts)] theme explorer build exit=$rc"') < block.index('if [ "$rc" -eq 0 ]; then\n      timeout')


def test_theme_stage_failure_is_recorded_but_not_promoted_to_run_status() -> None:
    block = _theme_block()
    assert 'record_stage "emerging_themes" "$rc"' in block
    assert 'record_stage "theme_explorer" "$rc"' in block
    assert "STATUS=" not in block


def test_nightly_no_longer_runs_the_theme_stage() -> None:
    nightly = (REPO_ROOT / "scripts/nightly_market_data.sh").read_text(encoding="utf-8")
    assert "themes.dynamic_theme.emerging" not in nightly
    assert "build_theme_explorer" not in nightly
