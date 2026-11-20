"""Tests for the Stage 2 evaluator on a synthetic world with a planted signal,
a vol-only ranker and noise twins (no model is trained)."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import eval_stage2 as E

TEST_START = "2026-05-15"
N_TICKERS = 60


def _arms() -> list[dict]:
    out = []
    for name, h in (("L0", 13), ("L2_10", 10), ("L2_20", 20)):
        for suffix in ("", "__perm"):
            out.append({"name": name + suffix, "target": f"y_{name}{suffix}", "features": "base",
                        "horizon_sessions": h, "embargo_days": 21, "control": bool(suffix)})
    return out


def _world(tmp_path: Path, arms: list[dict] | None = None) -> tuple[Path, Path]:
    """(manifest path, results folder). L2_10/L2_20 arms carry real signal on
    their yardsticks; L0 ranks volatility only; every __perm arm is noise."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2023-06-01", "2026-09-30") + pd.Timedelta(hours=15)
    ts = pd.DatetimeIndex(np.repeat(days, N_TICKERS)).tz_localize("UTC")
    n = len(ts)
    df = pd.DataFrame({"timestamp": ts, "ticker": np.tile([f"T{i:02d}" for i in range(N_TICKERS)], len(days))})
    sig10, sig20 = rng.normal(size=n), rng.normal(size=n)
    df["daily_atr_pct"] = rng.uniform(1, 6, n)
    df["beta_spy_60"] = rng.normal(1, 0.3, n)
    df["adv_rank"] = rng.uniform(size=n)
    df["px"] = rng.uniform(3, 200, n)
    for h, s in ((10, sig10), (20, sig20), (40, sig20)):
        df[f"L2_{h}"] = 0.02 * s + 0.05 * rng.normal(size=n)
        df[f"fret_{h}"] = df[f"L2_{h}"] + 0.001 * df["daily_atr_pct"]
        df[f"spy_fret_{h}"] = 0.0
        df[f"y_L2_{h}"] = (df.groupby("timestamp")[f"L2_{h}"].rank(pct=True) > 0.8).astype(float)
    df["fmfe_10"] = 0.01 * df["daily_atr_pct"] + 0.01 * rng.uniform(size=n)
    df.loc[df["timestamp"] > ts.max() - pd.Timedelta(days=14), ["L2_10", "fret_10", "y_L2_10"]] = np.nan
    for c in df.columns[2:]:
        df[c] = df[c].astype("float32")
    df.to_parquet(tmp_path / "matrix.parquet", index=False)

    arms = arms or _arms()
    manifest = {"matrix": "matrix.parquet", "arms": arms}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    res = tmp_path / "results"
    res.mkdir()
    oof = df[df["timestamp"] >= pd.Timestamp("2024-01-01", tz="UTC")]
    planted = {"L2_10": sig10, "L2_20": sig20}
    for a in arms:
        name = a["name"]
        idx = oof.index.to_numpy()
        if name in planted:
            score = planted[name][idx] + rng.normal(size=len(idx))
        elif name == "L0":
            score = oof["daily_atr_pct"].to_numpy() + 0.1 * rng.normal(size=len(idx))
        else:
            score = rng.normal(size=len(idx))
        pd.DataFrame({"timestamp": oof["timestamp"].to_numpy(), "ticker": oof["ticker"].to_numpy(),
                      "score": score.astype("float32"),
                      "fold": (oof["timestamp"].dt.year - 2024).to_numpy().astype("int16")}
                     ).to_parquet(res / f"oof_{name}.parquet", index=False)
    return tmp_path / "manifest.json", res


def _run(tmp_path, res, manifest, **kw):
    return E.run(results=res, manifest_path=manifest, out_dir=tmp_path / "out", n_boot=60,
                 test_start=TEST_START, **kw)


# --- per-bar statistics ------------------------------------------------------

def test_bar_ic_matches_pandas_spearman_with_ties():
    rng = np.random.default_rng(0)
    bar = np.repeat(np.arange(4), 80)
    score = rng.normal(size=len(bar))
    y = (rng.random(len(bar)) < 0.3).astype(float)          # binary label: heavy ties
    score[5] = np.nan
    got = E.bar_ic(bar, 4, score, y)
    df = pd.DataFrame({"b": bar, "s": score, "y": y}).dropna()
    want = df.groupby("b").apply(lambda g: g["s"].corr(g["y"], method="spearman"), include_groups=False)
    np.testing.assert_allclose(got, want.to_numpy(), atol=1e-10)


def test_bar_ic_skips_thin_bars():
    bar = np.r_[np.zeros(E.MIN_PAIRS - 1, int), np.ones(E.MIN_PAIRS, int)]
    x = np.arange(len(bar), dtype=float)
    ic = E.bar_ic(bar, 2, x, x)
    assert np.isnan(ic[0]) and ic[1] == pytest.approx(1.0)


def _panel(score_fn) -> E.Panel:
    rng = np.random.default_rng(3)
    nb, n = 3, 60
    bar = np.repeat(np.arange(nb), n)
    col = {"daily_atr_pct": rng.uniform(1, 6, nb * n), "beta_spy_60": rng.normal(1, .3, nb * n),
           "adv_rank": rng.uniform(size=nb * n), "px": rng.uniform(5, 50, nb * n)}
    col["fret_10"] = rng.normal(0, 0.05, nb * n)
    col["L2_10"] = col["fret_10"].copy()
    col["spy_fret_10"] = np.zeros(nb * n)
    col["y_L2_10"] = (rng.random(nb * n) < 0.2).astype(float)
    col["cost"] = np.full(nb * n, 0.001)
    ts = pd.date_range("2025-01-02 15:00", periods=nb, freq="D", tz="UTC")
    return E.Panel(bar=bar, bar_ts=ts, col=col, score={"A": score_fn(col)}, fold={"A": np.zeros(nb, np.int16)})


def _ranks(p):
    return {c: np.nan_to_num(E._group_rank(p.bar, p.col[c], pct=True), nan=0.5) for c in E.RESID_CONTROLS}


def test_topk_excess_net_and_precision():
    p = _panel(lambda c: c["fret_10"])                         # oracle score
    out = E.topk_bars(p, p.score["A"], 10, _ranks(p))
    for b in range(3):
        rows = p.bar == b
        r = p.col["fret_10"][rows]
        assert out["ex_k1"][b] == pytest.approx(r.max() - r.mean(), rel=1e-6)
        top3 = np.sort(r)[-3:]
        assert out["net_k3"][b] == pytest.approx(top3.mean() - 0.001, rel=1e-6)


def test_residualising_removes_a_pure_vol_tilt():
    p = _panel(lambda c: c["daily_atr_pct"])                   # score IS the vol control
    out = E.topk_bars(p, p.score["A"], 10, _ranks(p))
    assert np.all(out["atr_k3"] > 0.9)                         # raw picks: the most volatile names
    resid = E._residualise(p.bar, p.n_bars, p.score["A"], np.ones(len(p.bar), bool), _ranks(p))
    assert np.abs(resid).max() < 1e-6                          # nothing left once vol is regressed out


def test_partitions_purge_the_outcome_window():
    ts = pd.date_range("2026-03-01", "2026-07-01", freq="D", tz="UTC")
    t0 = pd.Timestamp(TEST_START, tz="UTC")
    m = E.partitions(ts, 40, t0)
    assert ts[m["validation"]].max() < t0 - pd.Timedelta(days=E.embargo_days(40))
    assert ts[m["test"]].min() == t0
    assert not (m["validation"] & m["test"]).any()


def test_select_applies_fdr_across_the_slot_candidates():
    arms = [{"name": f"A{i}", "target": f"y_A{i}", "control": False} for i in range(9)] + \
           [{"name": f"A{i}__perm", "target": f"y_A{i}__perm", "control": True} for i in range(9)]
    ic, paired = [], []
    for i in range(9):
        for y in E.SLOTS.values():
            ic.append({"arm": f"A{i}", "yardstick": y, "partition": "validation", "mean": 0.02, "ci_lo": 0.001,
                       "ci_hi": 0.04, "mde": 0.01, "folds_pos": 6, "n_folds": 9})
            p = 0.04 if i == 0 else 0.9                       # A0 alone: nominally significant
            paired.append({"arm": f"A{i}", "versus": f"A{i}__perm", "yardstick": y, "partition": "validation",
                           "mean": 0.01, "ci_lo": 0.0005 if i == 0 else -0.01, "ci_hi": 0.02, "p": p})
    picks = E.select(pd.DataFrame(ic), pd.DataFrame(paired), arms)
    a0 = next(c for c in picks["momentum"]["candidates"] if c["arm"] == "A0")
    assert a0["vs_perm_q"] == pytest.approx(0.36)               # 0.04 x 9 tests
    assert not a0["qualifies"] and picks["momentum"]["pick"] is None


# --- end to end --------------------------------------------------------------

def test_validation_selects_the_planted_arm_and_never_reads_the_test(tmp_path):
    manifest, res = _world(tmp_path)
    out = _run(tmp_path, res, manifest)
    assert out["picks"]["momentum"]["pick"] == "L2_10"
    assert out["picks"]["htf"]["pick"] == "L2_20"
    l0 = next(c for c in out["picks"]["momentum"]["candidates"] if c["arm"] == "L0")
    assert not l0["qualifies"]                                 # a vol ranker has no L2 skill
    assert set(out["ic"]["partition"]) == {"validation"}
    # validation stops a full embargo before the test start
    v = out["ic"][(out["ic"]["arm"] == "L2_10") & (out["ic"]["yardstick"] == "L2_10")]
    assert int(v["n_bars"].iloc[0]) < len(pd.bdate_range("2024-01-01", TEST_START))
    sel = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert sel["complete"] and sel["picks"]["momentum"]["pick"] == "L2_10"
    assert (tmp_path / "out" / "validation_report.md").exists()
    assert not (tmp_path / "out" / "final_test_access.jsonl").exists()
    # the vol pseudo-arm and the noise twin both appear, so raw levels can be read against them
    assert {"VOL", "L2_10__perm"} <= set(out["ic"]["arm"])


def test_final_test_gate(tmp_path):
    manifest, res = _world(tmp_path)
    with pytest.raises(RuntimeError, match="validation pass first"):
        _run(tmp_path, res, manifest, open_final_test=True)
    _run(tmp_path, res, manifest)
    out = _run(tmp_path, res, manifest, open_final_test=True)
    assert set(out["ic"]["partition"]) == {"test"}
    assert (tmp_path / "out" / "test_report.md").exists()
    log = (tmp_path / "out" / "final_test_access.jsonl").read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["picks"]["momentum"] == "L2_10"
    # a selection edited after the read is refused
    sel_path = tmp_path / "out" / "selection.json"
    sel = json.loads(sel_path.read_text())
    sel["selection_hash"] = "edited"
    sel_path.write_text(json.dumps(sel))
    with pytest.raises(RuntimeError, match="may not change"):
        _run(tmp_path, res, manifest, open_final_test=True)


def test_final_test_needs_every_arm(tmp_path):
    manifest, res = _world(tmp_path)
    (res / "oof_L2_20__perm.parquet").unlink()
    out = _run(tmp_path, res, manifest)                       # a partial run still evaluates
    l2_20 = next(c for c in out["picks"]["htf"]["candidates"] if c["arm"] == "L2_20")
    assert "no noise twin" in l2_20["why_not"]
    with pytest.raises(RuntimeError, match="every arm"):
        _run(tmp_path, res, manifest, open_final_test=True)


def test_oof_rows_outside_the_matrix_are_rejected(tmp_path):
    manifest, res = _world(tmp_path)
    p = res / "oof_L0.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "ticker"] = "NOT_IN_MATRIX"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="not in matrix.parquet"):
        _run(tmp_path, res, manifest)


def test_unpack_never_overwrites_a_different_result(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "oof_A.parquet").write_bytes(b"one")
    tar_path = tmp_path / "stage2_results.tar"
    with tarfile.open(tar_path, "w") as t:
        t.add(src / "oof_A.parquet", arcname="oof_A.parquet")
    dest = E.unpack(tar_path)
    assert (dest / "oof_A.parquet").read_bytes() == b"one"
    assert E.unpack(tar_path) == dest                         # same content: idempotent
    (dest / "oof_A.parquet").write_bytes(b"different")
    with pytest.raises(FileExistsError):
        E.unpack(tar_path)
