"""Momentum archives its whole scored cross-section without changing what it picks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strategies.momentum_expansion.inference.ranker as rk
import strategies.momentum_expansion.live.runner as mr
from core.decision_state_archive import load_decision_state


class _FakeRanker:
    def __init__(self, cols=("f1", "f2"), booster_path=Path("missing_booster.json")):
        self.feature_columns = list(cols)
        self.booster_path = booster_path

    def score(self, X: pd.DataFrame) -> pd.Series:
        # Rounded so ties are common: tie order is where a refactor would drift.
        s = np.round(X["f1"].astype(float).to_numpy() * 0.1, 1).astype(np.float32)
        return pd.Series(s, index=X.index, name="expansion_score")


def _legacy_top_n_at(*, bar_ts, features, ranker, cfg=None):
    """top_n_at exactly as it was before score_cross_section existed."""
    cfg = {**rk.RANKING_CONFIG, **(cfg or {})}
    rows = features.xs(bar_ts, level=features.index.names[0])
    snapshot = rk.load_snapshot_for(bar_ts)
    if not snapshot.empty:
        rows = rows[rows.index.isin(set(snapshot["ticker"].astype(str)))]
    if cfg.get("use_momentum_candidate_filter", True):
        rows = rk.filter_momentum_candidates(rows)
    rows = rows.dropna(subset=[c for c in ranker.feature_columns if c in rows.columns])
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "expansion_score", "rank"])
    scores = ranker.score(rows)
    out = pd.DataFrame({"ticker": rows.index, "expansion_score": scores.values})
    out = out.sort_values("expansion_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    n_eff = min(int(cfg["top_n"]), max(1, int(cfg["top_pct"] * len(out))))
    keep = out.head(n_eff)
    return keep[keep["expansion_score"] >= float(cfg["min_score"])].reset_index(drop=True)


def _panel(n: int, seed: int, ts: pd.Timestamp) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    f1 = rng.integers(0, 10, n).astype(float)
    f1[rng.random(n) < 0.1] = np.nan
    df = pd.DataFrame({
        "f1": f1,
        "f2": rng.normal(size=n),
        "dollar_vol_pctile_252": rng.random(n),
        "range_pos_20": rng.random(n),
    }, index=pd.MultiIndex.from_product([[ts], tickers], names=["timestamp", "ticker"]))
    return df


@pytest.mark.parametrize("seed", range(8))
def test_top_n_at_is_unchanged_by_the_refactor(monkeypatch, seed):
    ts = pd.Timestamp("2026-11-12 19:00", tz="UTC")
    feats = _panel(300, seed, ts)
    universe = [f"T{i:03d}" for i in range(0, 300, 2)] if seed % 2 else []
    monkeypatch.setattr(rk, "load_snapshot_for", lambda bar_ts: pd.DataFrame({"ticker": universe}))
    cfg = {"top_n": 25, "top_pct": 0.5, "min_score": 0.2}
    ranker = _FakeRanker()
    new = rk.top_n_at(bar_ts=ts, features=feats, ranker=ranker, cfg=cfg)
    old = _legacy_top_n_at(bar_ts=ts, features=feats, ranker=ranker, cfg=cfg)
    pd.testing.assert_frame_equal(new.reset_index(drop=True), old.reset_index(drop=True),
                                  check_dtype=False)
    xs = rk.score_cross_section(bar_ts=ts, features=feats, ranker=ranker, cfg=cfg)
    assert len(xs) == 300 and int(xs["selected"].sum()) == len(new)
    assert not xs.loc[feats.xs(ts)["f1"].isna().to_numpy(), "complete_features"].any()


class _FakeResult:
    def __init__(self, panel):
        self.panel = panel
        self.tickers_built = panel.index.get_level_values(1).nunique()
        self.tickers_requested = self.tickers_built
        self.build_seconds = 0.1


class _Trig:
    rule, close, suggested_stop_atr, note = "orb", 10.0, 1.5, ""


def _wire(monkeypatch, tmp_path, panel):
    names = list(panel.index.get_level_values(1))
    monkeypatch.setattr(mr, "load_snapshot_for", lambda bar_ts: pd.DataFrame({"ticker": names}))
    monkeypatch.setattr(rk, "load_snapshot_for", lambda bar_ts: pd.DataFrame({"ticker": names}))
    monkeypatch.setattr(mr, "build_live_feature_panel_4h", lambda **k: _FakeResult(panel))
    monkeypatch.setattr(mr, "assert_manifest_coverage", lambda **k: None)
    monkeypatch.setattr(mr, "DEFAULT_SIGNAL_AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(mr, "_emit_alert", lambda payload, **k: None)
    monkeypatch.setattr(mr, "_load_context_4h", lambda: {})
    monkeypatch.setattr(mr, "load_1h", lambda t: pd.DataFrame({"close": [1.0]}))
    # Only the best name prints a trigger.
    monkeypatch.setattr(mr, "evaluate_entry", lambda df_1h: [])
    return names


def test_scheduled_pass_archives_every_scored_name(monkeypatch, tmp_path):
    ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).floor("h")
    panel = _panel(40, 3, ts)
    _wire(monkeypatch, tmp_path, panel)
    runner = mr.MomentumLiveRunner(ranker=_FakeRanker(), auto_trade=False)
    first = {"done": False}

    def one_trigger(df_1h):
        if first["done"]:
            return []
        first["done"] = True
        return [_Trig()]

    monkeypatch.setattr(mr, "evaluate_entry", one_trigger)
    runner.evaluate_now(archive=True)

    df = load_decision_state(module="momentum_expansion")
    assert len(df) == 40 and df["run_id"].nunique() == 1
    run_id = df["run_id"].iloc[0]
    gate = runner._last_gate
    targets = gate["targets"]
    # Every gate is recorded; picks are the triggered top names only.
    assert {"g__in_universe", "g__candidate_filter", "g__complete_features", "g__top_n",
            "g__min_score", "g__entry_trigger"} <= set(df.columns)
    picked = df.loc[df["selected"], "ticker"].tolist()
    assert picked == [t for t, ok in gate["entry_ok"].items() if ok] and len(picked) == 1
    assert set(df.loc[df["ticker"].isin(targets), "action"]) <= {"entry_trigger:orb", "no_entry_trigger"}
    assert not df.loc[df["f__f1"].isna(), "g__complete_features"].any()
    assert df.loc[df["f__f1"].isna(), "score"].isna().all()
    # The full panel, not just the model's two features.
    assert {"f__f1", "f__f2", "f__dollar_vol_pctile_252", "f__range_pos_20"} <= set(df.columns)
    assert json.loads(df["run_meta_json"].iloc[0])["model_features"] == ["f1", "f2"]
    # Signal audits carry the run id, so order audits join back.
    assert all(a.get("decision_run_id") == run_id for a in gate["signal_audits"].values())
    logged = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert all(r["signal_audit"]["decision_run_id"] == run_id for r in logged if r.get("signal_audit"))


def test_preview_scans_do_not_archive(monkeypatch, tmp_path):
    ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).floor("h")
    _wire(monkeypatch, tmp_path, _panel(20, 1, ts))
    runner = mr.MomentumLiveRunner(ranker=_FakeRanker(), auto_trade=False)
    runner.evaluate_now()
    assert load_decision_state(module="momentum_expansion").empty
