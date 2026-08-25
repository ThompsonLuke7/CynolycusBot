"""Staleness and coverage invariants for the Meta Ranker matrix inputs.

Both defects here share a shape: a job reported success while serving data that
was not what it claimed to be.

  * `ticker_theme_features.parquet` froze at 2026-08-10 when weekly_refresh
    stage 4 failed twice, and the merge_asof join carried those values forward
    indefinitely with nothing marking them stale.
  * A one-ticker append set the matrix's max timestamp for the whole universe,
    so later runs reported "no new reference-market 4H bar to add" and exited 0
    with the real bar never incorporated (2026-07-09 18:00 UTC holds one row
    against 2,874 at 14:00).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.meta_context import build_meta_ranker_matrix as B
from signals.meta_context.meta_ranker.update_meta_matrix import last_covered_timestamp


def _spine(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": pd.to_datetime(dates),
            "mom_score": np.arange(len(dates), dtype=float),
        }
    )


def _theme_features(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime([date]),
            "primary_theme": ["enterprise_saas"],
            "theme_heat_score": [0.8],
            "theme_breadth": [0.5],
        }
    )


def test_theme_context_nulled_beyond_max_carry(tmp_path, monkeypatch):
    """A feed that stopped updating must not keep being served as current."""
    feats = tmp_path / "ticker_theme_features.parquet"
    _theme_features("2026-08-10").to_parquet(feats, index=False)
    monkeypatch.setattr(B, "DYNAMIC_THEME_FEATURES", feats)
    monkeypatch.setattr(B, "DYNAMIC_THEME_FEATURES_HISTORY", tmp_path / "missing.parquet")

    # One bar just inside the carry window, one well beyond it.
    fresh, stale = "2026-08-20", "2026-09-30"
    out, theme_ctx = B.join_theme_context(_spine([fresh, stale]), verbose=False)
    out = out.set_index("date")

    assert "theme_days_since_refresh" in theme_ctx
    fresh_row = out.loc[pd.Timestamp(fresh)]
    assert fresh_row["theme"] == "enterprise_saas"
    assert fresh_row["theme_heat_score"] == pytest.approx(0.8)
    assert fresh_row["theme_days_since_refresh"] == 10

    stale_row = out.loc[pd.Timestamp(stale)]
    assert pd.isna(stale_row["theme_heat_score"])
    assert pd.isna(stale_row["theme_days_since_refresh"])
    # the primary theme is nulled too, or it keeps driving the cross-context
    # ranks off a taxonomy that no longer describes the market
    assert pd.isna(stale_row["theme"])


def test_theme_context_never_looks_ahead(tmp_path, monkeypatch):
    """Same-day theme aggregates must not be visible to that day's bar."""
    feats = tmp_path / "ticker_theme_features.parquet"
    _theme_features("2026-08-20").to_parquet(feats, index=False)
    monkeypatch.setattr(B, "DYNAMIC_THEME_FEATURES", feats)
    monkeypatch.setattr(B, "DYNAMIC_THEME_FEATURES_HISTORY", tmp_path / "missing.parquet")

    out, _ = B.join_theme_context(_spine(["2026-08-20", "2026-08-21"]), verbose=False)
    out = out.set_index("date")
    assert pd.isna(out.loc[pd.Timestamp("2026-08-20"), "theme"])
    assert out.loc[pd.Timestamp("2026-08-21"), "theme"] == "enterprise_saas"


def _matrix(bars: dict[str, int]) -> pd.DataFrame:
    rows = []
    for ts, n in bars.items():
        for i in range(n):
            rows.append({"timestamp": pd.Timestamp(ts, tz="UTC"), "ticker": f"T{i}"})
    return pd.DataFrame(rows)


def test_last_covered_ignores_a_single_ticker_bar():
    """The 2026-07-09 shape: a 1-row bar must not pass as the universe's max."""
    existing = _matrix(
        {
            "2026-07-07 14:00": 2800,
            "2026-07-08 14:00": 2800,
            "2026-07-09 14:00": 2874,
            "2026-07-09 18:00": 1,
        }
    )
    assert existing["timestamp"].max() == pd.Timestamp("2026-07-09 18:00", tz="UTC")
    assert last_covered_timestamp(existing) == pd.Timestamp("2026-07-09 14:00", tz="UTC")


def test_last_covered_accepts_a_full_bar():
    existing = _matrix(
        {"2026-07-08 14:00": 2800, "2026-07-09 14:00": 2874, "2026-07-09 18:00": 2850}
    )
    assert last_covered_timestamp(existing) == pd.Timestamp("2026-07-09 18:00", tz="UTC")


def test_last_covered_degrades_to_newest_on_a_uniformly_thin_matrix():
    """A small test/bootstrap matrix must not be judged against nothing."""
    existing = _matrix({"2026-07-08 14:00": 3, "2026-07-09 14:00": 3})
    assert last_covered_timestamp(existing) == pd.Timestamp("2026-07-09 14:00", tz="UTC")


def test_last_covered_on_empty_matrix_is_none():
    assert last_covered_timestamp(pd.DataFrame(columns=["timestamp", "ticker"])) is None
