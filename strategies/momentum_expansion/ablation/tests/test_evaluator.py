"""Frozen evaluator: no manifest no run, sealed test gate, the scorecard on a known signal."""
import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import evaluator as ev
from strategies.momentum_expansion.ablation import manifest as mf


def make_manifest(**over):
    base = dict(experiment_id="unit", hypothesis="score ranks label", dataset_cutoff="2026-11-20",
                universe_version="u1", feature_schema_hash="f1", target_version="t1", label_horizon_sessions=1,
                label_entry="next_bar_open",
                model_family="synthetic", hyperparameters={}, seed=7, calibration="none",
                cost_model="equity_10bps_v1", execution_policy="top-k equal weight", git_commit="HEAD",
                primary_metric="ic", decision_rule="CI above 0", top_k=(1, 5), min_names_per_bar=200,
                hold_sessions=2)
    base.update(over)
    return mf.Manifest(**base)


def make_panel(start="2025-03-03", days=60, names=220, ic=0.3, seed=1):
    rng = np.random.default_rng(seed)
    stamps = [pd.Timestamp(f"{d.date()} {h}", tz="America/New_York").tz_convert("UTC")
              for d in pd.bdate_range(start, periods=days) for h in ("10:00", "14:00")]
    rows = len(stamps) * names
    sig = rng.normal(size=rows)
    df = pd.DataFrame({"timestamp": np.repeat(stamps, names),
                       "ticker": np.tile([f"T{i:03d}" for i in range(names)], len(stamps)),
                       "score": sig + rng.normal(scale=np.sqrt(1 / ic**2 - 1), size=rows),
                       "label": sig, "perm_score": rng.normal(size=rows),
                       "beta": rng.normal(size=rows), "adv_rank": rng.uniform(size=rows),
                       "days_to_earnings": np.nan})
    df["fwd_ret"] = 0.01 * df["label"]
    return df


def test_refuses_without_manifest():
    with pytest.raises(ev.ManifestRequired):
        ev.evaluate(make_panel(days=2), None)


def test_validation_input_in_sealed_window_is_refused():
    p = make_panel(start="2026-05-12", days=5)
    with pytest.raises(ev.SealedTestError):
        ev.evaluate(p, make_manifest(), check_git=False)


def test_test_split_needs_explicit_logged_open(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "TEST_READ_LOG", tmp_path / "reads.jsonl")
    p = make_panel(start="2026-05-18", days=10)
    m = make_manifest()
    with pytest.raises(ev.SealedTestError):
        ev.evaluate(p, m, split="test", check_git=False)
    with pytest.raises(mf.ManifestError):
        ev.evaluate(p, m, split="test", open_test=True, check_git=False)          # no reason
    r1 = ev.evaluate(p, m, split="test", open_test=True, test_read_reason="confirm", check_git=False)
    r2 = ev.evaluate(p, m, split="test", open_test=True, test_read_reason="again", check_git=False)
    assert r1["test_read_number"] == 1 and "warning" not in r1
    assert r2["test_read_number"] == 2 and "only the first read is confirmatory" in r2["warning"]
    assert len(mf.test_reads("unit", log_path=tmp_path / "reads.jsonl")) == 2


def test_scorecard_recovers_a_known_signal():
    r = ev.evaluate(make_panel(), make_manifest(), check_git=False)
    p = r["pooled"]
    assert 0.25 < p["ic"]["point"] < 0.35 and p["ic"]["excludes_zero"]
    assert p["perm_delta_ic"]["ci_lo"] > 0 and np.isfinite(p["ic"]["mde"])
    assert p["neutral_ic"]["controls"] == ["beta", "adv_rank"]
    t5 = p["top_k"]["5"]
    assert t5["net_ret"]["point"] > 0 and t5["precision"]["point"] > 0.5
    assert t5["large_winner_recall"]["recall"] > t5["large_winner_recall"]["random"]
    assert set(r["by_block"]) == {"2025_chop"}


def test_finite_capital_curve_marks_daily():
    p = make_panel(days=30, names=220)
    dates = pd.bdate_range("2025-03-03", periods=30)
    closes = pd.DataFrame(100.0, index=dates, columns=sorted(p.ticker.unique()))
    closes.iloc[10:] = 90.0                                   # a 10% drop mid-run, marked daily
    m = make_manifest(top_k=(5,), cost_model="zero_v1")
    out = ev.evaluate(p, m, daily_close=closes, check_git=False)["finite_capital"]["5"]
    assert out["days"] == 30 and np.isclose(out["max_drawdown_daily_marked"], 0.10, atol=1e-9)
