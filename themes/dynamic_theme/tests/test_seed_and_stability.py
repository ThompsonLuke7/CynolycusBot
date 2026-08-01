"""Unit tests for seeded themes + label-stability helpers (no pipeline run)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def test_seed_registry_rows_present_and_pinned():
    from themes.dynamic_theme.seed_themes import seed_registry_rows
    from themes.dynamic_theme.config import SEED_THEMES

    as_of = pd.Timestamp("2026-06-28")
    rows = seed_registry_rows(as_of)
    assert len(rows) == len(SEED_THEMES)
    mem = rows[rows["theme_name"] == "memory_storage"]
    assert len(mem) == 1
    r = mem.iloc[0]
    assert r["confidence"] == 1.0           # hand-pinned
    assert r["cluster_id"] < 0              # reserved range, never collides with HDBSCAN
    assert json.loads(r["related_themes"])  # related list survives round-trip


def test_seed_centroid_makes_anchor_its_primary_theme():
    """A memory name must score higher to the memory_storage anchor than to a
    competing 'semi cap equipment' centroid — i.e. the seed fixes the misroute."""
    from themes.dynamic_theme.seed_themes import seed_centroids, seed_cluster_id

    rng = np.random.default_rng(0)
    dim = 16
    # Two latent groups: memory cluster and semicap cluster.
    mem_dir = rng.normal(size=dim); mem_dir /= np.linalg.norm(mem_dir)
    cap_dir = rng.normal(size=dim); cap_dir /= np.linalg.norm(cap_dir)

    def near(direction, jitter=0.05):
        v = direction + rng.normal(scale=jitter, size=dim)
        return v / np.linalg.norm(v)

    tickers = ["MU", "WDC", "STX", "SNDK", "AMAT", "LRCX", "KLAC"]
    vecs = {t: near(mem_dir) for t in ["MU", "WDC", "STX", "SNDK"]}
    vecs.update({t: near(cap_dir) for t in ["AMAT", "LRCX", "KLAC"]})
    matrix = np.array([vecs[t] for t in tickers], dtype=np.float32)

    cents, names = seed_centroids(tickers, matrix)
    mem_cid = seed_cluster_id(0)
    assert mem_cid in cents and names[mem_cid] == "memory_storage"

    # A semicap centroid to compete against.
    cap_centroid = matrix[[tickers.index(t) for t in ["AMAT", "LRCX", "KLAC"]]].mean(0)
    cap_centroid /= np.linalg.norm(cap_centroid)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    wdc = matrix[tickers.index("WDC")]
    assert cos(wdc, cents[mem_cid]) > cos(wdc, cap_centroid)


def test_seed_centroid_skipped_when_no_anchor_present():
    from themes.dynamic_theme.seed_themes import seed_centroids

    cents, names = seed_centroids(["AAPL", "MSFT"], np.eye(2, dtype=np.float32))
    assert cents == {} and names == {}  # no memory anchors in this universe


def test_match_prior_theme_reuses_above_threshold_only():
    from themes.dynamic_theme.stages.step05_claude_labeling import _match_prior_theme
    from themes.dynamic_theme.config import LABEL_STABILITY_THRESHOLD

    v = np.array([1.0, 0.0, 0.0])
    near = v + np.array([0.0, 0.02, 0.0])          # ~identical → reuse
    far = np.array([0.0, 1.0, 0.0])                # orthogonal → relabel
    cur = {5: v}
    assert _match_prior_theme(5, cur, {"memory_storage": near}) == "memory_storage"
    assert _match_prior_theme(5, cur, {"memory_storage": far}) is None
    # 'cluster_N' junk names are never carried forward
    assert _match_prior_theme(5, cur, {"cluster_9": near}) is None
    # unknown cluster id → no match
    assert _match_prior_theme(999, cur, {"memory_storage": near}) is None
    assert LABEL_STABILITY_THRESHOLD > 0.5


def test_write_registry_dedups_seed_against_emergent(tmp_path, monkeypatch):
    """If HDBSCAN already produced 'memory_storage', the seed must NOT duplicate it."""
    import themes.dynamic_theme.stages.step05_claude_labeling as s5
    monkeypatch.setattr(s5, "THEME_REGISTRY_PATH", tmp_path / "reg.parquet")
    as_of = pd.Timestamp("2026-06-28")
    rows = [{"cluster_id": 7, "theme_name": "memory_storage", "parent_theme": "semiconductors",
             "description": "emergent", "related_themes": "[]", "confidence": 0.9, "date": as_of}]
    out = s5._write_registry(rows, as_of)
    mem = out[out["theme_name"] == "memory_storage"]
    assert len(mem) == 1 and int(mem.iloc[0]["cluster_id"]) == 7  # emergent kept, seed skipped


def test_write_registry_adds_seed_when_missing(tmp_path, monkeypatch):
    """If the emergent taxonomy lacks 'memory_storage', the seed fills the gap."""
    import themes.dynamic_theme.stages.step05_claude_labeling as s5
    monkeypatch.setattr(s5, "THEME_REGISTRY_PATH", tmp_path / "reg.parquet")
    as_of = pd.Timestamp("2026-06-28")
    rows = [{"cluster_id": 3, "theme_name": "solar_energy", "parent_theme": "energy",
             "description": "x", "related_themes": "[]", "confidence": 0.8, "date": as_of}]
    out = s5._write_registry(rows, as_of)
    assert "memory_storage" in set(out["theme_name"])
    assert (out[out["theme_name"] == "memory_storage"]["cluster_id"] < 0).all()  # seed id


def test_taxonomy_version_ignores_cluster_numbers_paths_order_and_timestamps():
    from themes.dynamic_theme.stages.step08_memberships import compute_taxonomy_version

    first = pd.DataFrame(
        [
            {
                "cluster_id": 17,
                "theme_name": "beta_theme",
                "date": pd.Timestamp("2026-07-30"),
                "local_path": "/tmp/one.parquet",
            },
            {
                "cluster_id": 3,
                "theme_name": "alpha_theme",
                "date": pd.Timestamp("2026-07-30"),
                "local_path": "/tmp/one.parquet",
            },
        ]
    )
    second = first.iloc[::-1].copy()
    second["cluster_id"] = [900, 401]
    second["date"] = pd.Timestamp("2099-01-01")
    second["local_path"] = "/different/machine/taxonomy.parquet"

    assert compute_taxonomy_version(first) == compute_taxonomy_version(second)


def test_taxonomy_version_rejects_ephemeral_cluster_theme_ids():
    from themes.dynamic_theme.stages.step08_memberships import compute_taxonomy_version

    registry = pd.DataFrame({"theme_name": ["cluster_17"]})
    with pytest.raises(ValueError, match="ephemeral"):
        compute_taxonomy_version(registry)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
