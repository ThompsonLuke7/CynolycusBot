"""Regression tests for evaluation bugs the project has already shipped (checklist 0.3)."""
import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import evaluator as ev
from strategies.momentum_expansion.ablation import manifest as mf
from strategies.momentum_expansion.ablation.tests.test_evaluator import make_manifest, make_panel


def test_embargo_counts_bars_not_hours():
    """run_bakeoff once purged by wall-clock hours; a weekend made 60 'bars' far fewer."""
    from strategies.model_training.colab_competition import purge_before
    stamps = [pd.Timestamp(f"{d.date()} {h}", tz="UTC") for d in pd.bdate_range("2026-01-05", periods=10)
              for h in ("15:00", "19:00")]
    df = pd.DataFrame({"timestamp": stamps, "ticker": "AAA"})
    boundary = pd.Timestamp("2026-01-19 15:00", tz="UTC")                  # Monday after a weekend
    kept = purge_before(df, boundary, embargo_bars=3)
    assert len(kept) == len(df[df.timestamp < boundary]) - 3                # exactly 3 bars, across the weekend


def test_competition_choose_best_ignores_val_log_loss():
    """Every competition bundle once picked by val_log_loss, which rankers do not even report."""
    from strategies.model_training.colab_competition import choose_best, primary_metric_name
    results = pd.DataFrame({"family": ["xgb_clf", "xgb_ranker"], "val_log_loss": [0.40, np.nan],
                            "val_ndcg_at_10": [0.51, 0.58], "val_precision_at_10": [0.30, 0.31],
                            "test_ndcg_at_10": [0.70, 0.40]})
    assert primary_metric_name(results) == "val_ndcg_at_10"
    assert choose_best(results)["family"] == "xgb_ranker"
    with pytest.raises(ValueError):
        primary_metric_name(results[["family", "val_log_loss", "test_ndcg_at_10"]])  # no test fallback


def test_htf_oof_selected_on_the_future_is_refused():
    """The HTF OOF kept only swing-zone rows (48.8%), defined by 3 future bars."""
    from strategies.momentum_expansion.ablation.run_feature_screen import assert_oof_unfiltered
    matrix = pd.Series(pd.date_range("2025-01-01", periods=1000, freq="4h", tz="UTC"))
    with pytest.raises(ValueError, match="filtered sample"):
        assert_oof_unfiltered(matrix.iloc[::2], matrix, model="htf")
    assert assert_oof_unfiltered(matrix, matrix, model="htf") == 1.0


def test_outcomes_from_the_prior_close_are_refused():
    """MFE measured from the decision bar's own close counted the move that made the signal."""
    with pytest.raises(mf.ManifestError, match="executable entry"):
        mf.validate(make_manifest(label_entry="prior_close"), check_git=False)


def test_stray_thin_timestamps_are_dropped_and_counted():
    """A few bars with a handful of names (data gaps) once dominated per-bar statistics."""
    p = make_panel(days=5)
    thin = p[p.timestamp == p.timestamp.iloc[0]].head(3).assign(
        timestamp=pd.Timestamp("2025-03-05 17:00", tz="UTC"))
    df, counts = ev.prepare(pd.concat([p, thin]), make_manifest(), split="validation")
    assert counts["thin_bars"] == 1 and counts["rows_in_thin_bars"] == 3
    assert pd.Timestamp("2025-03-05 17:00", tz="UTC") not in set(df.timestamp)


def test_nan_in_an_unused_column_does_not_drop_rows():
    """A drop-any over every column once removed every row with a NaN earnings field."""
    from strategies.momentum_expansion.features.feature_matrix_4h import _bars_since_true
    p = make_panel(days=3)
    assert p["days_to_earnings"].isna().all()
    df, counts = ev.prepare(p, make_manifest(), split="validation")
    assert counts["rows_missing_required"] == 0 and len(df) == len(p)
    assert _bars_since_true(pd.Series([False, False, True, False])).tolist() == [252.0, 252.0, 0.0, 1.0]


def test_manifest_file_round_trip_and_unknown_fields(tmp_path):
    import json
    from dataclasses import asdict
    m = make_manifest()
    d = asdict(m)
    d["git_commit"] = mf.git_head()[0]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(d))
    assert mf.load_manifest(path).experiment_id == "unit"
    path.write_text(json.dumps({**d, "secret_tweak": 1}))
    with pytest.raises(mf.ManifestError, match="unknown manifest fields"):
        mf.load_manifest(path)
    with pytest.raises(mf.ManifestError, match="not in this checkout"):
        mf.validate(make_manifest(git_commit="0" * 40))


def test_overlapping_labels_get_horizon_length_blocks():
    """Calendar-week blocks on a 10-20 session label treated overlapping outcomes as
    independent: the Stage 3 reproduction's CI was 30% too narrow until blocks followed h."""
    ts = pd.Series(pd.date_range("2025-01-06", periods=40, freq="7D", tz="UTC"))
    assert ev.horizon_blocks(ts, 1).nunique() == 40
    assert ev.horizon_blocks(ts, 20).nunique() == 10          # ceil(20/5) = 4 weeks per block
    with pytest.raises(mf.ManifestError, match="label_horizon_sessions"):
        mf.validate(make_manifest(label_horizon_sessions=0), check_git=False)
