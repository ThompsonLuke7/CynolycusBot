"""The incremental news pass must not load the whole embeddings file.

Each vector is stored as a ~8.5 KB JSON string, so the old pass held ~8 GB of
Python strings per read, read the file in three of its four steps and rewrote
it twice: 18.4 GB peak on a 19.5 GB box, and WSL died inside it on 2026-10-12
and 10-16. These tests pin the batched replacements to the old results.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from sklearn.cluster import KMeans

from signals.news import pipeline

pytestmark = pytest.mark.safe


def _vec(seed: int) -> str:
    return json.dumps(np.random.default_rng(seed).standard_normal(4).astype(float).tolist())


def _prior(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "ticker": ["AAA", "BBB"] * (n // 2),
            "timestamp": pd.date_range("2026-10-01", periods=n, freq="h", tz="UTC"),
            "text": [f"headline {i}" for i in range(n)],
            "catalyst_family": ["earnings", "mna"] * (n // 2),
            "embedding": [_vec(i) for i in range(n)],
            "embedding_available": 1.0,
            "finbert_positive_score": 0.1,
            "finbert_negative_score": 0.2,
            "finbert_neutral_score": 0.7,
            "finbert_available": 1.0,
            "news_cluster_id": [float(i % 2) for i in range(n)],
            "news_cluster_key": [f"fam:{i % 2}" for i in range(n)],
            "_text_hash": [pipeline._text_hash(f"headline {i}") for i in range(n)],
        }
    )


def _records(prior: pd.DataFrame, extra: int = 2, changed: str | None = "r1") -> pd.DataFrame:
    rec = prior[["record_id", "ticker", "timestamp", "text", "catalyst_family"]].copy()
    if changed is not None:
        rec.loc[rec["record_id"] == changed, "text"] = "rewritten body"
    new = pd.DataFrame(
        {
            "record_id": [f"n{i}" for i in range(extra)],
            "ticker": "CCC",
            "timestamp": pd.Timestamp("2026-10-02", tz="UTC"),
            "text": [f"fresh {i}" for i in range(extra)],
            "catalyst_family": "earnings",
        }
    )
    return pd.concat([rec, new], ignore_index=True)


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    monkeypatch.setattr(pipeline, "ensure_data_dirs", lambda: None)
    monkeypatch.setattr(pipeline, "embed_texts_bge", lambda texts: [np.full(4, len(t), dtype=np.float32) for t in texts])
    monkeypatch.setattr(
        pipeline, "finbert_scores_batch",
        lambda texts: [{"finbert_positive_score": 0.5, "finbert_negative_score": 0.25, "finbert_neutral_score": 0.25} for _ in texts],
    )


def test_append_never_loads_prior_vectors_and_matches_the_old_concat(tmp_path, monkeypatch):
    prior = _prior()
    emb, rec = tmp_path / "emb.parquet", tmp_path / "rec.parquet"
    prior.to_parquet(emb, index=False)
    _records(prior).to_parquet(rec, index=False)

    schema_before = pq.read_schema(emb)
    real_read = pd.read_parquet

    def _guard(path, *args, **kwargs):
        if str(path) == str(emb):
            assert kwargs.get("columns") is not None, "the full embeddings file was loaded"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.pd, "read_parquet", _guard)
    out = pipeline.build_news_embeddings(rec, output_path=emb)

    saved = real_read(emb)
    # Old semantics: prior minus re-embedded rows, then the new rows at the end.
    assert saved["record_id"].tolist() == ["r0", "r2", "r3", "r4", "r5", "r1", "n0", "n1"]
    assert set(out["record_id"]) == {"r1", "n0", "n1"}
    untouched = saved.set_index("record_id").loc[["r0", "r2"], "embedding"]
    assert untouched.tolist() == prior.set_index("record_id").loc[["r0", "r2"], "embedding"].tolist()
    assert saved.loc[saved["record_id"] == "n0", "news_cluster_id"].isna().all()
    assert pq.read_schema(emb).equals(schema_before)


def test_nothing_new_leaves_the_file_byte_identical(tmp_path):
    prior = _prior()
    emb, rec = tmp_path / "emb.parquet", tmp_path / "rec.parquet"
    prior.to_parquet(emb, index=False)
    _records(prior, extra=0, changed=None).to_parquet(rec, index=False)
    before = emb.read_bytes()

    out = pipeline.build_news_embeddings(rec, output_path=emb)

    assert out.empty
    assert emb.read_bytes() == before


def test_a_miscounted_rewrite_keeps_the_original_file(tmp_path):
    emb = tmp_path / "emb.parquet"
    _prior().to_parquet(emb, index=False)
    before = emb.read_bytes()

    with pytest.raises(RuntimeError, match="expected"):
        pipeline._rewrite_parquet_in_batches(emb, expected_rows=999)

    assert emb.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def _models(tmp_path) -> tuple:
    x = np.vstack([np.asarray(json.loads(_vec(i)), dtype=np.float32) for i in range(10)])
    km = KMeans(n_clusters=2, random_state=0, n_init=3).fit(x)
    path = tmp_path / "kmeans.pkl"
    path.write_bytes(pickle.dumps({"earnings": (km, 10), "mna": (None, 20)}))
    return path, km


def test_cluster_assigns_only_pending_rows_with_the_old_rules(tmp_path):
    frame = _prior(8)
    frame["catalyst_family"] = ["earnings", "mna", "brand_new", None, "earnings", "earnings", "mna", "mna"]
    frame.loc[[0, 1, 2, 3], "news_cluster_id"] = np.nan
    frame.loc[[0, 1, 2, 3], "news_cluster_key"] = None
    emb = tmp_path / "emb.parquet"
    frame.to_parquet(emb, index=False)
    models, km = _models(tmp_path)

    pipeline.cluster_news_embeddings(emb, output_path=emb, models_path=models)

    saved = pd.read_parquet(emb).set_index("record_id")
    expect = float(10 + km.predict(np.asarray([json.loads(frame.at[0, "embedding"])], dtype=np.float32))[0])
    assert saved.at["r0", "news_cluster_id"] == expect           # saved model predicts
    assert saved.at["r1", "news_cluster_id"] == 20.0             # single-row family -> base id
    assert saved.at["r1", "news_cluster_key"] == "mna:0"
    assert saved.at["r2", "news_cluster_id"] == -1.0             # unseen family -> placeholder
    assert saved.at["r2", "news_cluster_key"] == "brand_new:unassigned"
    assert np.isnan(saved.at["r3", "news_cluster_id"])           # no family -> left unassigned
    assert saved.loc[["r4", "r5", "r6", "r7"], "news_cluster_id"].tolist() == frame.loc[4:7, "news_cluster_id"].tolist()


def test_cluster_with_nothing_pending_does_not_rewrite(tmp_path):
    emb = tmp_path / "emb.parquet"
    _prior().to_parquet(emb, index=False)
    models, _ = _models(tmp_path)
    before = emb.read_bytes()

    pipeline.cluster_news_embeddings(emb, output_path=emb, models_path=models)

    assert emb.read_bytes() == before


def test_libraries_stream_to_the_same_rows_as_one_merge(tmp_path):
    prior = _prior(6)
    emb, lab = tmp_path / "emb.parquet", tmp_path / "labels.parquet"
    prior.to_parquet(emb, index=False)
    labels = prior[["record_id", "ticker", "timestamp"]].iloc[[5, 0, 3, 2]].assign(
        forward_5d_return=[0.2, -0.1, 0.0, 0.3], expansion_label=[1.0, 0.0, 0.0, 1.0]
    )
    labels.to_parquet(lab, index=False)
    win, lose = tmp_path / "win.parquet", tmp_path / "lose.parquet"

    pipeline.build_winner_loser_libraries(emb, lab, winner_path=win, loser_path=lose)

    merged = prior.merge(labels, on=["record_id", "ticker", "timestamp"], how="inner")
    for path, value in ((win, 1.0), (lose, 0.0)):
        expect = merged.loc[merged["expansion_label"].eq(value)].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.read_parquet(path), expect)
