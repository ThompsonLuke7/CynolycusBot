from pathlib import Path

import pytest

pytestmark = pytest.mark.safe

REPO_ROOT = Path(__file__).resolve().parents[2]


def _theme_block() -> str:
    source = (REPO_ROOT / "scripts/nightly_market_data.sh").read_text(encoding="utf-8")
    start = source.index('echo "[$(ts)] themes — pending ticker enrichment')
    end = source.index("  # 9) Earnings enrichment", start)
    return source[start:end]


def test_nightly_captures_explorer_build_status_before_publication() -> None:
    block = _theme_block()
    builder = '"$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_explorer'
    publisher = '"$PYTHON" -u scripts/publish_theme_explorer.py'

    assert builder in block
    assert "explorer_build_exit=$?" in block
    assert publisher in block
    assert block.index("explorer_build_exit=$?") < block.index(publisher)
    assert 'if [ "$explorer_build_exit" -eq 0 ]; then' in block


def test_publication_failure_is_logged_but_not_promoted_to_trading_status() -> None:
    block = _theme_block()
    assert "explorer_publish_exit=$?" in block
    assert 'echo "[$(ts)] theme explorer publication exit=$explorer_publish_exit"' in block
    assert 'STATUS="$explorer_publish_exit"' not in block
