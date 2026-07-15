"""
Regression tests for the capstone figure set.

Guards that the committed figure directory, its README claim map, and the
generator stay in sync:
  1. every figure the generator defines exists on disk and is non-trivial,
  2. every figure on disk is documented in README.md (claim map row),
  3. every README-referenced figure actually exists,
  4. the generator's palette is the validated one (fixed entity→color map).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "research" / "capstone" / "figures"
README = FIG_DIR / "README.md"

EXPECTED_FIGS = [
    "fig01_equity_curves.png",
    "fig02_drawdown.png",
    "fig03_rolling_sharpe.png",
    "fig04_selection_bias_correction.png",
    "fig05_regime_performance.png",
    "fig06_trade_return_distributions.png",
    "fig07_hold_times.png",
    "fig08_oof_decile_lift.png",
    "fig09_meta_calibration.png",
    "fig10_feature_importance.png",
    "fig11_meta_exit_policy.png",
    "fig12_paper_sessions.png",
]


def test_all_expected_figures_exist_and_are_nontrivial():
    missing = [f for f in EXPECTED_FIGS if not (FIG_DIR / f).exists()]
    assert not missing, f"figures missing from {FIG_DIR}: {missing}"
    tiny = [f for f in EXPECTED_FIGS if (FIG_DIR / f).stat().st_size < 20_000]
    assert not tiny, f"suspiciously small figure files (broken render?): {tiny}"


def test_readme_documents_every_figure_on_disk():
    text = README.read_text()
    on_disk = sorted(p.name for p in FIG_DIR.glob("fig*.png"))
    undocumented = [f for f in on_disk if f not in text]
    assert not undocumented, f"figures on disk but not in README claim map: {undocumented}"


def test_readme_references_only_existing_figures():
    text = README.read_text()
    referenced = set(re.findall(r"fig\d+[a-z0-9_]*\.png", text))
    ghosts = [f for f in sorted(referenced) if not (FIG_DIR / f).exists()]
    assert not ghosts, f"README references figures that do not exist: {ghosts}"


def test_generator_uses_validated_entity_palette():
    from scripts.capstone import make_figures as mf

    assert mf.COLORS["momentum"] == "#2a78d6"
    assert mf.COLORS["htf_swing"] == "#199e70"
    assert mf.COLORS["swing"] == "#4a3aa7"
    assert mf.COLORS["meta"] == "#008300"
    # benchmark/neutral chrome must stay out of the categorical slots
    assert mf.COLORS["spy"] not in {mf.COLORS[k] for k in ("momentum", "htf_swing", "swing", "meta")}
