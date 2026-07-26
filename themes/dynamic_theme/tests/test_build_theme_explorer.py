from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "themes/dynamic_theme/viz/build_theme_explorer.py"
ARTIFACT = REPO_ROOT / "themes/dynamic_theme/viz/theme_explorer.html"


def test_reset_view_control_restores_the_default_camera() -> None:
    for document in (BUILDER, ARTIFACT):
        source = document.read_text(encoding="utf-8")

        assert 'title="Clear selections and reset camera"' in source
        assert "⌂ Reset view" in source
        assert "function defaultCameraDistance()" in source
        assert "function resetView()" in source
        assert "cam.position.set(0, 0, defaultCameraDistance());" in source
        assert "resetView();" in source
