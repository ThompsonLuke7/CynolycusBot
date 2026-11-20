from __future__ import annotations

import pandas as pd

from themes.dynamic_theme.emerging import build_outputs, rank_pending_candidates


def test_rank_pending_candidates_filters_and_prioritizes_catalysts():
    pending = pd.DataFrame(
        [
            {
                "ticker": "CAT",
                "status": "pending",
                "source": "catalyst",
                "passes_price": True,
                "passes_adv": True,
                "passes_history": True,
                "last_price": 10,
                "avg_dollar_volume_20d": 8e6,
                "market_cap": 500e6,
                "history_days": 250,
                "catalyst_mentions": 8,
            },
            {
                "ticker": "LIQ",
                "status": "pending",
                "source": "screener",
                "passes_price": True,
                "passes_adv": True,
                "passes_history": True,
                "last_price": 30,
                "avg_dollar_volume_20d": 20e6,
                "market_cap": 5e9,
                "history_days": 300,
                "catalyst_mentions": 0,
            },
            {
                "ticker": "THIN",
                "status": "pending",
                "source": "screener",
                "passes_price": True,
                "passes_adv": False,
                "passes_history": True,
                "last_price": 5,
                "avg_dollar_volume_20d": 1e6,
                "market_cap": 300e6,
                "history_days": 300,
                "catalyst_mentions": 0,
            },
        ]
    )

    ranked = rank_pending_candidates(pending, limit=2)

    assert ranked["ticker"].tolist()[0] == "CAT"
    assert set(ranked["ticker"]) == {"CAT", "LIQ"}
    assert ranked["candidate_rank"].tolist() == [1, 2]


def test_build_outputs_preserves_candidate_score_after_document_merge(tmp_path, monkeypatch):
    candidates = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "candidate_score": 0.8,
                "candidate_rank": 1,
                "catalyst_mentions": 2,
            }
        ]
    )
    docs = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "description": "A test company",
                "sector": "Technology",
                "industry": "Software",
                "quote_type": "EQUITY",
                "recent_news_summary": "New product launch",
                "candidate_score": 0.8,
                "candidate_rank": 1,
                "catalyst_mentions": 2,
                "date": pd.Timestamp("2026-06-19"),
            }
        ]
    )
    assignments = pd.DataFrame(
        [{"ticker": "NEW", "closest_theme": "software", "theme_similarity": 0.8, "assignment_type": "extension"}]
    )
    embeddings = pd.DataFrame([{"ticker": "NEW", "embedding": [1.0, 0.0]}])

    monkeypatch.setattr(
        "themes.dynamic_theme.emerging._price_metrics",
        lambda tickers: pd.DataFrame(
            columns=["ticker", "return_5d", "return_20d", "volume_acceleration"]
        ),
    )
    monkeypatch.setattr("themes.dynamic_theme.emerging.PENDING_THEME_CANDIDATES_PATH", tmp_path / "candidates.parquet")
    monkeypatch.setattr("themes.dynamic_theme.emerging.PENDING_THEME_REGISTRY_PATH", tmp_path / "registry.parquet")
    monkeypatch.setattr("themes.dynamic_theme.emerging.PENDING_THEME_MEMBERSHIP_PATH", tmp_path / "membership.parquet")
    monkeypatch.setattr("themes.dynamic_theme.emerging.PENDING_THEME_HISTORY_PATH", tmp_path / "history.parquet")
    pd.DataFrame(
        [
            {
                "theme_key": "extension::stale",
                "theme_name": "stale",
                "theme_type": "extension",
                "date": pd.Timestamp("2026-06-19"),
            }
        ]
    ).to_parquet(tmp_path / "history.parquet", index=False)

    registry, membership = build_outputs(
        candidates,
        docs,
        embeddings,
        assignments,
        pd.DataFrame(),
        as_of=pd.Timestamp("2026-06-19"),
    )

    assert registry.iloc[0]["emerging_score"] > 0
    assert membership.iloc[0]["ticker"] == "NEW"
    history = pd.read_parquet(tmp_path / "history.parquet")
    assert history["theme_key"].tolist() == ["extension::software"]


def test_provisional_cluster_is_labeled_with_existing_names_and_company_details(tmp_path, monkeypatch):
    """emerging.py used to call _label_cluster(summary): TypeError on every call, so
    no provisional theme was ever named by Claude. It must pass the existing names."""
    tickers = ["P1", "P2", "P3"]
    candidates = pd.DataFrame([
        {"ticker": t, "candidate_score": 0.7, "candidate_rank": i + 1, "catalyst_mentions": 1,
         "date": pd.Timestamp("2026-11-17")} for i, t in enumerate(tickers)])
    docs = pd.DataFrame([
        {"ticker": t, "description": f"{t} builds small modular reactors. More.", "sector": "Utilities",
         "industry": "Utilities - Renewable", "quote_type": "EQUITY", "recent_news_summary": "h1 | h2"}
        for t in tickers])
    assignments = pd.DataFrame([{"ticker": t, "closest_theme": "solar", "theme_similarity": 0.6,
                                 "assignment_type": "novel_residual"} for t in tickers])
    residuals = pd.DataFrame([{"ticker": t, "provisional_cluster": 0, "cluster_probability": 0.9}
                              for t in tickers])
    profiles = pd.DataFrame([{"ticker": "P2", "shortName": "Pee Two", "industry": "Utilities",
                              "marketCap": 5e9}])
    for name in ("CANDIDATES", "REGISTRY", "MEMBERSHIP", "HISTORY"):
        monkeypatch.setattr(f"themes.dynamic_theme.emerging.PENDING_THEME_{name}_PATH", tmp_path / f"{name}.parquet")
    monkeypatch.setattr("themes.dynamic_theme.emerging._price_metrics",
                        lambda t: pd.DataFrame(columns=["ticker", "return_5d", "return_20d", "volume_acceleration"]))
    monkeypatch.setattr("themes.dynamic_theme.emerging._established_theme_names", lambda: {"solar_energy"})
    seen = {}

    def fake_label(summary, existing_names):
        seen.update(summary=summary, existing=existing_names)
        return {"theme_name": "small_modular_reactors", "parent_theme": "nuclear",
                "description": "SMR developers.", "related_themes": [], "confidence": 0.8}

    monkeypatch.setattr("themes.dynamic_theme.emerging._label_cluster", fake_label)
    registry, _ = build_outputs(candidates, docs, pd.DataFrame(), assignments, residuals,
                                as_of=pd.Timestamp("2026-11-17"), profiles=profiles)

    assert seen["existing"] == ["solar_energy"]
    assert seen["summary"]["members"][0] == {"ticker": "P2", "name": "Pee Two", "industry": "Utilities",
                                             "about": "P2 builds small modular reactors."}
    provisional = registry[registry["theme_type"] == "provisional"]
    assert provisional["theme_name"].tolist() == ["small_modular_reactors"]
