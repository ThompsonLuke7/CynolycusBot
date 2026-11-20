"""Fixed theme groups (2026-11-18, research/theme_stability_2026-11-18).

Re-clustering identical inputs keeps only ~0.57 of the grouping, so groups are
re-fit only when due and reused otherwise, and each ticker's primary theme is
frozen for the life of a re-fit. Identity = sector + industry + description,
plus a 0.2 share of price co-movement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from themes.dynamic_theme.stages import step02_embed as s2
from themes.dynamic_theme.stages import step03_cluster as s3
from themes.dynamic_theme.stages import step08_memberships as s8

RECIPE = s3.clustering_recipe()


def _anchor(refit_id="2026-11-01", recipe=RECIPE, rows=(("AAA", 0), ("BBB", 0), ("CCC", 1))):
    return pd.DataFrame([{"ticker": t, "cluster_id": c, "probability": 0.9, "refit_id": refit_id,
                          "recipe": recipe} for t, c in rows])


# ── when to re-fit ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("anchor, refit, expect", [
    (None, False, "no anchor"),
    (_anchor(), True, "requested"),
    (_anchor(recipe="old|recipe"), False, "recipe changed"),
    (_anchor(refit_id="2026-08-01"), False, "days old"),
    (_anchor(), False, None),
])
def test_refit_reason(anchor, refit, expect):
    reason = s3.refit_reason(anchor, refit=refit, as_of=pd.Timestamp("2026-11-16"), recipe=RECIPE)
    assert reason is None if expect is None else expect in reason


def _embeddings(tickers):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ticker": tickers, "embedding": [rng.normal(size=8).tolist() for _ in tickers],
                         "date": pd.Timestamp("2026-11-16")})


def test_fresh_anchor_is_reused_and_new_tickers_wait_unassigned(tmp_path, monkeypatch):
    monkeypatch.setattr(s3, "TICKER_CLUSTER_ANCHOR_PATH", tmp_path / "anchor.parquet")
    monkeypatch.setattr(s3, "TICKER_CLUSTERS_PATH", tmp_path / "clusters.parquet")
    _anchor().to_parquet(tmp_path / "anchor.parquet", index=False)

    def no_umap():
        raise AssertionError("a fresh anchor must not be re-fit")

    monkeypatch.setattr(s3, "load_umap", no_umap)
    out = s3.cluster_tickers(_embeddings(["AAA", "CCC", "NEW"]))
    assert out.set_index("ticker")["cluster_id"].to_dict() == {"AAA": 0, "CCC": 1, "NEW": -1}
    assert set(out["refit_id"]) == {"2026-11-01"}
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "clusters.parquet"), out, check_dtype=False)
    assert len(pd.read_parquet(tmp_path / "anchor.parquet")) == 3  # anchor untouched


def test_refit_writes_a_new_anchor_with_recipe_and_date(tmp_path, monkeypatch):
    monkeypatch.setattr(s3, "TICKER_CLUSTER_ANCHOR_PATH", tmp_path / "anchor.parquet")
    monkeypatch.setattr(s3, "TICKER_CLUSTERS_PATH", tmp_path / "clusters.parquet")

    class FakeUMAP:
        def __init__(self, **kw):
            pass

        def fit_transform(self, m):
            return np.asarray(m)[:, :2]

    class FakeHDBSCAN:
        def __init__(self, **kw):
            pass

        def fit(self, x):
            self.labels_ = np.array([0, 0, 1, -1])
            self.probabilities_ = np.array([1.0, 0.9, 0.8, 0.0])
            return self

    monkeypatch.setattr(s3, "load_umap", lambda: type("U", (), {"UMAP": FakeUMAP}))
    monkeypatch.setattr(s3, "load_hdbscan", lambda: type("H", (), {"HDBSCAN": FakeHDBSCAN}))
    out = s3.cluster_tickers(_embeddings(["AAA", "BBB", "CCC", "DDD"]))
    anchor = pd.read_parquet(tmp_path / "anchor.parquet")
    assert out["cluster_id"].tolist() == [0, 0, 1, -1] and set(out["refit_id"]) == {"2026-11-16"}
    assert anchor["ticker"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert set(anchor["recipe"]) == {RECIPE} and set(anchor["refit_id"]) == {"2026-11-16"}


# ── frozen primary themes ─────────────────────────────────────────────────────
def _registry(names=None):
    names = names or {0: "semis", 1: "regional_banks", 2: "grab_bag"}
    conf = {2: 0.3}
    return pd.DataFrame([{"cluster_id": c, "theme_name": n, "confidence": conf.get(c, 0.9),
                          "date": pd.Timestamp("2026-11-16")} for c, n in names.items()])


def _memberships():
    rows = []
    for t in ("AAA", "BBB", "NEW", "LOOSE"):
        for theme, score in (("semis", 0.80), ("regional_banks", 0.79), ("software", 0.5)):
            rows.append({"ticker": t, "theme": theme, "membership_score": score})
    return pd.DataFrame(rows)


def _primary(picks):
    return pd.DataFrame([{"ticker": t, "primary_theme": th, "membership_score": 0.8, "source": "cluster"}
                         for t, th in picks.items()])


def _clusters(refit_id="2026-11-01"):
    return pd.DataFrame({"ticker": ["AAA"], "cluster_id": [0], "refit_id": [refit_id]})


def test_anchored_ticker_keeps_its_theme_when_the_fresh_pick_wobbles(tmp_path):
    path = tmp_path / "primary_anchor.parquet"
    pd.DataFrame({"ticker": ["AAA"], "cluster_id": [0], "refit_id": ["2026-11-01"]}).to_parquet(path)
    fresh = _primary({"AAA": "regional_banks", "NEW": "regional_banks", "LOOSE": "unclassified"})
    out, anchor = s8.anchor_primary_themes(fresh, _memberships(), _clusters(), _registry(), anchor_path=path)
    got = out.set_index("ticker")
    assert got.loc["AAA", "primary_theme"] == "semis" and got.loc["AAA", "source"] == "anchored"
    assert got.loc["AAA", "membership_score"] == pytest.approx(0.80)
    assert got.loc["NEW", "primary_theme"] == "regional_banks"
    assert anchor.set_index("ticker")["cluster_id"].to_dict() == {"AAA": 0, "NEW": 1}  # LOOSE re-tried weekly
    assert set(anchor["refit_id"]) == {"2026-11-01"}


def test_anchor_follows_the_cluster_when_its_name_is_corrected(tmp_path):
    path = tmp_path / "primary_anchor.parquet"
    pd.DataFrame({"ticker": ["AAA"], "cluster_id": [1], "refit_id": ["2026-11-01"]}).to_parquet(path)
    registry = _registry({0: "semis", 1: "software"})  # cluster 1 renamed regional_banks -> software
    out, _ = s8.anchor_primary_themes(_primary({"AAA": "semis"}), _memberships(), _clusters(), registry,
                                      anchor_path=path)
    assert out.set_index("ticker").loc["AAA", "primary_theme"] == "software"


def test_unpublished_anchor_or_new_refit_falls_back_to_the_fresh_pick(tmp_path):
    path = tmp_path / "primary_anchor.parquet"
    pd.DataFrame({"ticker": ["AAA", "BBB"], "cluster_id": [2, 0], "refit_id": ["2026-11-01", "2026-08-01"]}
                 ).to_parquet(path)
    fresh = _primary({"AAA": "semis", "BBB": "regional_banks"})
    out, anchor = s8.anchor_primary_themes(fresh, _memberships(), _clusters(), _registry(), anchor_path=path)
    got = out.set_index("ticker")["primary_theme"].to_dict()
    assert got == {"AAA": "semis", "BBB": "regional_banks"}  # grab-bag anchor / older re-fit ignored
    assert anchor.set_index("ticker")["cluster_id"].to_dict() == {"AAA": 0, "BBB": 1}


def test_without_a_refit_id_nothing_is_anchored():
    fresh = _primary({"AAA": "semis"})
    out, anchor = s8.anchor_primary_themes(fresh, _memberships(), pd.DataFrame({"ticker": ["AAA"], "cluster_id": [0]}),
                                           _registry())
    assert anchor is None and out.equals(fresh)


# ── identity text ─────────────────────────────────────────────────────────────
def test_identity_text_is_sector_industry_then_business():
    assert s2.identity_text("Makes GPUs.", "Technology", "Semiconductors") == \
        "Sector: Technology. Industry: Semiconductors. Makes GPUs."
    assert s2.identity_text("", "Energy", None, "Exxon Mobil") == "Sector: Energy. Exxon Mobil"
    assert s2.identity_text(None, "nan", float("nan"), None) == ""


def test_news_is_not_read_when_every_ticker_has_identity_text(tmp_path, monkeypatch):
    monkeypatch.setattr(s2, "TICKER_EMBEDDINGS_PATH", tmp_path / "emb.parquet")
    monkeypatch.setattr(s2, "DESC_BLEND_WEIGHT", 1.0)
    monkeypatch.setattr("themes.dynamic_theme.profiles.load_profiles", lambda: pd.DataFrame(
        [{"ticker": "AAA", "sector": "Technology", "industry": "Semiconductors", "shortName": "A"}]))
    seen_texts = []

    def fake_embed(texts, **kw):
        seen_texts.extend(texts)
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr("signals.news.nlp.embed_texts_bge", fake_embed)

    def no_news(tickers):
        raise AssertionError("news must not be loaded when every ticker has identity text")

    monkeypatch.setattr(s2, "_load_article_embeddings_by_ticker", no_news)
    monkeypatch.setattr(s2, "_build_comovement_vectors", lambda tickers, as_of=None: {})
    docs = pd.DataFrame([{"ticker": "AAA", "description": "Makes GPUs.", "date": pd.Timestamp("2026-11-16")}])
    out = s2.generate_embeddings(docs)
    assert seen_texts == ["Sector: Technology. Industry: Semiconductors. Makes GPUs."]
    assert out["ticker"].tolist() == ["AAA"]


def test_comovement_share_is_the_configured_weight():
    from themes.dynamic_theme import config

    assert s2._COMOVEMENT_WEIGHT == config.COMOVEMENT_WEIGHT == 0.2
