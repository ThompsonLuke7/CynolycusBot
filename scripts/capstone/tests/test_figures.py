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
    "fig13_scaleout_grid.png",
    "fig14_baseline_comparison.png",
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


def test_trainers_ask_lightgbm_for_gain_not_split():
    """LGBM's feature_importances_ is split count, not gain.

    The trainers must consult booster_ (gain) BEFORE feature_importances_,
    otherwise LightGBM winners silently report split counts and calendar
    features like week_of_year dominate for the wrong reason.
    """
    trainers = [
        REPO / "strategies/multi_ticker_swing_htf/data/training_export/colab_competition.py",
        REPO / "strategies/momentum_expansion/data/training_export/colab_competition.py",
        REPO / "signals/meta_context/meta_ranker/colab_competition.py",
    ]
    for t in trainers:
        src = t.read_text()
        block = src.split("def feature_importance(")[1].split("\ndef ")[0]
        # compare the hasattr GUARDS (code), not prose mentions in comments
        i_boost = block.find('hasattr(model, "booster_")')
        i_sk = block.find('hasattr(model, "feature_importances_")')
        assert 'importance_type="gain"' in block, f"{t.name}: no explicit gain call"
        assert i_boost != -1 and i_sk != -1, f"{t.name}: expected both hasattr guards"
        assert i_boost < i_sk, (
            f"{t.name}: feature_importances_ is checked before booster_ — "
            "LightGBM will report split counts again"
        )


def test_htf_panel_recomputes_gain_from_booster():
    """fig10's HTF panel must not read the stale split-count CSV."""
    from scripts.capstone import make_figures as mf

    csv = REPO / "strategies/multi_ticker_swing_htf/models/feature_importance_lgbm_classifier_seed46.csv"
    model = REPO / "strategies/multi_ticker_swing_htf/models/model_lgbm_classifier_seed46.joblib"
    if not model.exists():
        import pytest

        pytest.skip("HTF lgbm model artifact not present")
    gain = mf._gain_importance(csv, model)
    top = gain.nlargest(1, "share").iloc[0]
    # by gain the HTF winner is an ATR/volatility model, not a seasonality model
    assert top.feature == "daily_atr_pct", f"expected daily_atr_pct to lead by gain, got {top.feature}"
    assert top.share > 20, f"daily_atr_pct gain share collapsed to {top.share:.1f}% (split counts leaking back?)"
    wk = gain[gain.feature == "week_of_year"]["share"].iloc[0]
    assert wk < 6, f"week_of_year at {wk:.1f}% of gain — looks like split counts, not gain"


def test_exit_policy_grid_matches_results_lock():
    """The committed grid must still reproduce the locked meta_exit_policy numbers."""
    import json

    grid_path = REPO / "research" / "capstone" / "exit_policy_grid.csv"
    if not grid_path.exists():
        import pytest

        pytest.skip("exit_policy_grid.csv not generated")
    import csv as _csv

    with open(grid_path) as fh:
        grid = {r["policy"]: r for r in _csv.DictReader(fh)}
    lock = json.loads((REPO / "research" / "capstone" / "results_lock.json").read_text())
    locked = {m["metric"]: m for m in lock["metrics"] if m["model"] == "meta_exit_policy"}
    pairs = [
        ("rank drop-out g=0 (old live)", "current_live_dropout_g0"),
        ("target +20% full exit", "target20_full_exit"),
        ("scale 50%@+20% + horizon25", "scaleout50_at20_horizon25"),
    ]
    for policy, prefix in pairs:
        assert policy in grid, f"{policy} missing from exit_policy_grid.csv"
        for metric in ("mean", "win"):
            got = float(grid[policy][metric])
            want = locked[f"{prefix}_{metric}"]["value"]
            assert abs(got - want) < 5e-4, f"{policy} {metric}: grid {got:.5f} vs lock {want:.5f}"
        assert int(grid[policy]["n"]) == locked[f"{prefix}_mean"]["n"]


def test_generator_uses_validated_entity_palette():
    from scripts.capstone import make_figures as mf

    assert mf.COLORS["momentum"] == "#2a78d6"
    assert mf.COLORS["htf_swing"] == "#199e70"
    assert mf.COLORS["swing"] == "#4a3aa7"
    assert mf.COLORS["meta"] == "#008300"
    assert mf.COLORS["baseline_ew"] == "#c1440e"
    assert mf.COLORS["baseline_random"] == "#7a3aa0"
    assert mf.COLORS["baseline_oracle"] == "#a8720a"
    # benchmark/neutral chrome must stay out of the categorical slots
    categorical = {mf.COLORS[k] for k in
                  ("momentum", "htf_swing", "swing", "meta", "baseline_ew", "baseline_random", "baseline_oracle")}
    assert mf.COLORS["spy"] not in categorical
    assert len(categorical) == 7, "categorical slot collision — two entities share a hex"
