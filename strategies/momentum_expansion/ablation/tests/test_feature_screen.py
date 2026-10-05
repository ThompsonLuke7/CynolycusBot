"""Leak / alignment tests for the Stage-1 feature screen building blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation import run_feature_screen as R
from strategies.momentum_expansion.ablation.candidate_features import CANDIDATE_BLOCKS, daily_candidate_features
from strategies.momentum_expansion.ablation.label_arms import build_label_arms


def _bars(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * np.exp(rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, n))
    ts = pd.bdate_range("2023-01-02", periods=n, tz="UTC") + pd.Timedelta(hours=4)
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": lo, "close": c,
                         "volume": rng.uniform(1e6, 2e6, n)})


def test_candidate_features_never_read_same_or_future_bar():
    d = _bars()
    base = daily_candidate_features(d)
    cut = 300
    shocked = d.copy()
    # Perturb bar `cut` and everything after it: row `cut` must not change.
    shocked.loc[cut:, ["open", "high", "low", "close"]] *= 1.37
    shocked.loc[cut:, "volume"] *= 5
    after = daily_candidate_features(shocked)
    pd.testing.assert_series_equal(base.iloc[cut], after.iloc[cut], check_names=False)
    assert not base.iloc[cut + 1].equals(after.iloc[cut + 1])


def test_every_block_column_is_produced():
    cols = set(daily_candidate_features(_bars()).columns)
    assert {f for blk in CANDIDATE_BLOCKS.values() for f in blk} == cols


def test_split_gap_nans_features_inside_their_lookback_only():
    d = _bars()
    p = 350
    d.loc[p:, ["open", "high", "low", "close"]] /= 10  # 10:1 forward split, no volume jump
    f = daily_candidate_features(d)
    # shift(1): the gap at row p first reaches the value stamped on p+1
    assert np.isnan(f.loc[p + 1, "d_ret_63"])
    assert np.isnan(f.loc[min(p + 60, len(d) - 1), "d_ret_63"])
    assert not np.isnan(f.loc[p - 1, "d_ret_63"])
    # 1-bar candle features recover right after the gap bar
    assert not np.isnan(f.loc[p + 3, "d_body_range"])


def test_label_arms_neutralise_volatility():
    rng = np.random.default_rng(1)
    rows = []
    for t in range(40):
        vol = rng.uniform(0.01, 0.10, 500)
        mfe = vol * rng.uniform(1, 3, 500)  # MFE purely proportional to vol
        ret = vol * rng.normal(0, 1, 500) + 0.5 * vol  # vol premium in returns
        rows.append(pd.DataFrame({"timestamp": t, "daily_atr_pct": vol, "mfe": mfe, "ret": ret}))
    panel = pd.concat(rows, ignore_index=True)
    arms = build_label_arms(panel, mfe_col="mfe", ret_col="ret")
    corr = lambda a: pd.concat([panel, arms], axis=1).groupby("timestamp").apply(
        lambda g: g[a].corr(g["daily_atr_pct"], method="spearman")).mean()
    assert corr("L0") > 0.5            # baseline label is a vol proxy
    assert abs(corr("L1")) < 0.1       # vol-bucketed rank is not
    assert abs(corr("L2")) < 0.15      # vol-matched excess is not
    assert corr("R") > 0.1             # raw return still carries the vol premium


def test_forward_return_skips_split_inside_window(tmp_path, monkeypatch):
    d = _bars(120)
    p = 60
    d.loc[p:, ["open", "high", "low", "close"]] /= 8
    d.to_parquet(tmp_path / "XYZ.parquet")
    monkeypatch.setattr(R, "DAILY_DIR", tmp_path)
    out = R._daily_block("XYZ")
    h = 5
    # entries D with a gap at rows D+2..D+h are dropped: D in [p-h, p-2]
    assert out.loc[p - h:p - 2, f"fret_{h}"].isna().all()
    # D = p-1 enters at row p's open (after the gap): valid
    assert out.loc[p - 1, f"fret_{h}"] == d.loc[p - 1 + h, "close"] / d.loc[p, "open"] - 1
    assert out.loc[p - h - 1, f"fret_{h}"] == d.loc[p - 1, "close"] / d.loc[p - h, "open"] - 1


def test_filtered_oof_is_rejected():
    import pytest
    ts = pd.Series(pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC"))
    assert R.assert_oof_unfiltered(ts, ts, model="m") == 1.0
    with pytest.raises(ValueError, match="filtered sample"):
        R.assert_oof_unfiltered(ts.iloc[::2], ts, model="htf")  # 50%: pivot-only style OOF
