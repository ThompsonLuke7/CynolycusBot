"""Membership publication invariants.

Two separate defects put a corrupted `ticker_theme_membership.parquet` in front
of every consumer on 2026-08-17 and 2026-08-24:

  1. step05 lets two clusters legitimately carry the same theme name (Claude is
     told to reuse a known name rather than mint a near-duplicate), and step08
     emitted one row per (ticker, theme-INSTANCE) — 435,450 of 563,182 rows were
     duplicate (ticker, theme) pairs with conflicting scores.
  2. step08 published the file BEFORE validating it, so the guard that caught
     (1) raised only after the bad artifact was already on disk. The run
     reported exit=1 and looked like it had changed nothing.

The second is the one that turns a bug into a silent bad artifact, so it is
tested independently of the first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from themes.dynamic_theme.stages import step08_memberships as step08


def test_collapse_merges_clusters_sharing_a_theme_name():
    """Two clusters under one name are one theme, scored by best-matching lobe."""
    sim = np.array(
        [
            [0.10, 0.90, 0.30],  # ticker A is close to the second lobe
            [0.70, 0.20, 0.40],  # ticker B is close to the first
        ],
        dtype=np.float32,
    )
    collapsed, names = step08._collapse_duplicate_themes(
        sim, ["mortgage_reits", "mortgage_reits", "enterprise_saas"]
    )

    assert names == ["mortgage_reits", "enterprise_saas"]
    # max, not mean: a ticker belongs to the theme if it is close to ANY lobe.
    assert collapsed[0, 0] == pytest.approx(0.90)
    assert collapsed[1, 0] == pytest.approx(0.70)
    # the untouched column survives unchanged
    assert collapsed[0, 1] == pytest.approx(0.30)


def test_collapse_is_a_no_op_when_names_are_unique():
    sim = np.array([[0.1, 0.2]], dtype=np.float32)
    names = ["a_theme", "b_theme"]
    collapsed, out_names = step08._collapse_duplicate_themes(sim, names)
    assert out_names == names
    assert collapsed is sim


def test_assert_publishable_rejects_conflicting_scores():
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "theme": ["cloud_cybersecurity", "cloud_cybersecurity"],
            "membership_score": [0.55, 0.73],
        }
    )
    with pytest.raises(ValueError, match="conflicting immutable membership rows"):
        step08.assert_publishable_memberships(frame)


def test_assert_publishable_collapses_identical_duplicates():
    """Identical repeats are redundant, not contradictory — dedupe, don't raise."""
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "theme": ["cloud_cybersecurity", "cloud_cybersecurity", "cloud_cybersecurity"],
            "membership_score": [0.55, 0.55, 0.61],
        }
    )
    out = step08.assert_publishable_memberships(frame)
    assert len(out) == 2
    assert set(out["ticker"]) == {"AAPL", "MSFT"}


def test_invalid_frame_is_not_published(tmp_path, monkeypatch):
    """A run that fails validation must leave the prior artifact untouched.

    This is the regression that matters: before the fix the write happened
    first, so a raising run still replaced the file consumers read. Simulated by
    disabling the collapse so the conflicting frame reaches the publish step.
    """
    membership_path = tmp_path / "ticker_theme_membership.parquet"
    history_path = tmp_path / "ticker_theme_membership_history.parquet"
    sentinel = pd.DataFrame(
        {
            "ticker": ["PRIOR"],
            "theme": ["prior_theme"],
            "membership_score": [0.42],
            "date": [pd.Timestamp("2026-08-10")],
        }
    )
    sentinel.to_parquet(membership_path, index=False)

    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_PATH", membership_path)
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(step08, "ensure_outputs", lambda: None)
    # Re-introduce the pre-fix behaviour: sibling clusters keep their own column.
    monkeypatch.setattr(
        step08, "_collapse_duplicate_themes", lambda sim, names: (sim, names)
    )

    tickers = ["AAA", "BBB"]
    # Deliberately NOT orthogonal: every ticker must score positively against
    # both cluster centroids, or the score > 0 filter hides the duplicate rows
    # and the fixture stops reproducing the production condition.
    matrix = np.array([[1.0, 0.2], [0.3, 1.0]], dtype=np.float32)
    embeddings = pd.DataFrame({"ticker": tickers, "embedding": [row.tolist() for row in matrix]})
    clusters = pd.DataFrame({"ticker": tickers, "cluster_id": [0, 1]})
    # Both clusters carry the SAME theme name — the 2026-08-24 condition.
    registry = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "theme_name": ["shared_theme", "shared_theme"],
        }
    )

    with pytest.raises(ValueError, match="conflicting immutable membership rows"):
        step08.compute_memberships(
            embeddings_df=embeddings,
            clusters_df=clusters,
            registry_df=registry,
            as_of=pd.Timestamp("2026-08-24", tz="UTC"),
        )

    survived = pd.read_parquet(membership_path)
    assert survived["ticker"].tolist() == ["PRIOR"], (
        "a failed run overwrote the published membership artifact"
    )


def test_shared_theme_name_produces_one_row_per_ticker_theme(tmp_path, monkeypatch):
    """End to end: the 2026-08-24 input now publishes a clean artifact."""
    membership_path = tmp_path / "ticker_theme_membership.parquet"
    history_path = tmp_path / "ticker_theme_membership_history.parquet"
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_PATH", membership_path)
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(step08, "ensure_outputs", lambda: None)

    tickers = ["AAA", "BBB"]
    # Deliberately NOT orthogonal: every ticker must score positively against
    # both cluster centroids, or the score > 0 filter hides the duplicate rows
    # and the fixture stops reproducing the production condition.
    matrix = np.array([[1.0, 0.2], [0.3, 1.0]], dtype=np.float32)
    embeddings = pd.DataFrame({"ticker": tickers, "embedding": [row.tolist() for row in matrix]})
    clusters = pd.DataFrame({"ticker": tickers, "cluster_id": [0, 1]})
    registry = pd.DataFrame(
        {"cluster_id": [0, 1], "theme_name": ["shared_theme", "shared_theme"]}
    )

    out = step08.compute_memberships(
        embeddings_df=embeddings,
        clusters_df=clusters,
        registry_df=registry,
        as_of=pd.Timestamp("2026-08-24", tz="UTC"),
    )

    assert not out.duplicated(["ticker", "theme"]).any()
    assert out["theme"].nunique() == 1
    assert history_path.exists(), "history must be written on a successful run"
