"""Stage 3 evaluator (rule v2) on a synthetic world: neutral scores, S_new, the 4-test rule, the test gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import eval_stage2 as E
from strategies.momentum_expansion.ablation import eval_stage3 as V

TEST_START = "2026-05-15"
N_TICKERS = 60


def _arms() -> list[dict]:
    out = []
    for name, h in (("L0", 13), ("L2_10", 10), ("L2_20", 20)):
        for suffix in ("", "__perm"):
            out.append({"name": f"{name}__pit{suffix}", "target": f"y_{name}{suffix}", "features": "pit_no_cal",
                        "horizon_sessions": h, "embargo_days": 21, "control": bool(suffix)})
    return out


def _world(tmp_path: Path, *, early_sign: float = 1.0, range_end: str = "2026-09-30") -> tuple[Path, Path]:
    """L2_10__pit / L2_20__pit carry a vol-independent signal (times ``early_sign`` before
    2022-04, the S_new period); L0__pit ranks volatility only; every __perm arm is noise.
    OOF starts 2021-06."""
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2021-01-04", "2026-09-30") + pd.Timedelta(hours=15)
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
    for c in df.columns[2:]:
        df[c] = df[c].astype("float32")
    df.to_parquet(tmp_path / "matrix.parquet", index=False)
    manifest = {"matrix": "matrix.parquet", "arms": _arms(), "range": ["2021-01-04 15:00:00+00:00", range_end]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    oof = df[df["timestamp"] >= pd.Timestamp("2021-06-01", tz="UTC")]
    idx = oof.index.to_numpy()
    sign = np.where((oof["timestamp"] < V.S_NEW_END).to_numpy(), early_sign, 1.0)
    planted = {"L2_10__pit": sig10, "L2_20__pit": sig20}
    for a in _arms():
        name = a["name"]
        if name in planted:
            score = sign * planted[name][idx] + rng.normal(size=len(idx))
        elif name == "L0__pit":
            score = oof["daily_atr_pct"].to_numpy() + 0.1 * rng.normal(size=len(idx))
        else:
            score = rng.normal(size=len(idx))
        fold = ((oof["timestamp"] - pd.Timestamp("2021-06-01", tz="UTC")).dt.days // 182).to_numpy()
        pd.DataFrame({"timestamp": oof["timestamp"].to_numpy(), "ticker": oof["ticker"].to_numpy(),
                      "score": score.astype("float32"), "fold": fold.astype("int16")}
                     ).to_parquet(res / f"oof_{name}.parquet", index=False)
    return tmp_path / "manifest.json", res


def _run(tmp_path, res, manifest, **kw):
    return V.run(results=res, manifest_path=manifest, out_dir=tmp_path / "out", n_boot=60,
                 test_start=TEST_START, **kw)


def test_neutral_names_keep_twins_matched():
    assert V.neutral_name("y_L2_10") == "y_L2_10~n"
    assert V.neutral_name("y_L2_10__perm") == "y_L2_10~n__perm"
    arms = [{"name": "A", "target": "y_A", "control": False}, {"name": "A__perm", "target": "y_A__perm", "control": True}]
    ext = arms + [{**a, "name": V.neutral_name(a["name"]), "target": V.neutral_name(a["target"])} for a in arms]
    assert E.twins(ext)["A~n"] == "A~n__perm"


def test_neutral_score_removes_a_vol_ranker_and_keeps_a_real_signal(tmp_path):
    manifest, res = _world(tmp_path)
    p = E.load_panel(res, tmp_path / "matrix.parquet", [a["name"] for a in _arms()])
    p, _ = V.with_neutral(p, _arms())
    ic = {a: np.nanmean(E.bar_ic(p.bar, p.n_bars, p.score[a], p.col["L2_10"])) for a in p.score}
    atr = {a: np.nanmean(E.bar_ic(p.bar, p.n_bars, p.score[a], p.col["daily_atr_pct"])) for a in p.score}
    assert atr["L0__pit"] > 0.9 and abs(atr["L0__pit~n"]) < 0.02
    assert ic["L2_10__pit~n"] > 0.2 and ic["L2_10__pit~n"] == pytest.approx(ic["L2_10__pit"], abs=0.02)


def test_rule_v2_picks_the_planted_arms_and_records_four_tests(tmp_path):
    manifest, res = _world(tmp_path)
    out = _run(tmp_path, res, manifest)
    assert out["picks"]["momentum"]["pick"] == "L2_10__pit"
    assert out["picks"]["htf"]["pick"] == "L2_20__pit"
    sel = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert sel["rule"] == V.RULE
    tests = [c for s in sel["picks"].values() for c in s["candidates"]]
    assert len(tests) == 4 and all(np.isfinite(t["s_new_ic"]) for t in tests)
    assert set(out["ic"]["partition"]) == {"validation"}
    assert not (tmp_path / "out" / "final_test_access.jsonl").exists()


def test_a_signal_reversed_before_2022_04_fails_the_s_new_check(tmp_path):
    manifest, res = _world(tmp_path, early_sign=-1.0)
    out = _run(tmp_path, res, manifest)
    assert out["picks"]["momentum"]["pick"] is None
    why = {c["arm"]: c["why_not"] for c in out["picks"]["momentum"]["candidates"]}
    assert any("S_new" in w for w in why["L2_10__pit"])


def test_the_test_opens_only_with_twelve_matured_months(tmp_path):
    manifest, res = _world(tmp_path)
    _run(tmp_path, res, manifest)
    with pytest.raises(RuntimeError, match="requires 365"):
        _run(tmp_path, res, manifest, open_final_test=True)
    m = json.loads(manifest.read_text())
    m["range"][1] = "2027-07-15 20:00:00+00:00"
    manifest.write_text(json.dumps(m))
    out = _run(tmp_path, res, manifest, open_final_test=True)
    log = [json.loads(x) for x in (tmp_path / "out" / "final_test_access.jsonl").read_text().splitlines()]
    assert len(log) == 1 and log[0]["rule"] == V.RULE and log[0]["projected_test_mde"]["momentum"] > 0
    assert out["picks"]["momentum"]["confirmed"] is True
    assert set(out["ic"]["partition"]) == {"test"}
