"""Business-type theme assignment (2026-10-04).

Grab-bag clusters (the labeler's own "heterogeneous, no single sector theme",
confidence < 0.50) are not themes. A ticker without a confident cluster is
placed with its industry peers or left unclassified -- never by nearest
centroid alone, which matched industry peers 14% of the time for these names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from themes.dynamic_theme import profiles as P
from themes.dynamic_theme.config import MIN_CLUSTER_LABEL_CONFIDENCE, UNCLASSIFIED_THEME
from themes.dynamic_theme.stages import step08_memberships as step08
from themes.dynamic_theme.stages import step09_meta_features as step09


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "theme_name": ["software", "banks", "mixed_small_cap_value"],
            "confidence": [0.90, 0.85, MIN_CLUSTER_LABEL_CONFIDENCE - 0.1],
        }
    )


def _clusters() -> pd.DataFrame:
    # SW1-4 software, BK1-3 banks, GB1-3 in the grab-bag cluster, NZ1 unclustered
    return pd.DataFrame(
        {
            "ticker": ["SW1", "SW2", "SW3", "SW4", "BK1", "BK2", "BK3", "GB1", "GB2", "GB3", "NZ1"],
            "cluster_id": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1],
        }
    )


def _memberships(scores: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """scores: ticker -> (software score, banks score). Grab-bag is not published."""
    rows = []
    for ticker, (sw, bk) in scores.items():
        rows += [(ticker, "software", sw), (ticker, "banks", bk)]
    return pd.DataFrame(rows, columns=["ticker", "theme", "membership_score"])


_SCORES = {
    "SW1": (0.95, 0.60), "SW2": (0.94, 0.61), "SW3": (0.93, 0.62), "SW4": (0.92, 0.63),
    "BK1": (0.60, 0.95), "BK2": (0.61, 0.94), "BK3": (0.62, 0.93),
    "GB1": (0.80, 0.86),   # a software company whose NEAREST theme is banks
    "GB2": (0.70, 0.75),   # a railroad: no theme holds its industry
    "GB3": (0.72, 0.74),   # no industry on file
    "NZ1": (0.70, 0.88),   # unclustered bank
}
_INDUSTRY = {
    "SW1": "Software", "SW2": "Software", "SW3": "Software", "SW4": "Software",
    "BK1": "Banks", "BK2": "Banks", "BK3": "Banks",
    "GB1": "Software", "GB2": "Railroads", "NZ1": "Banks",
}


def test_grab_bag_ids_come_from_label_confidence():
    assert step08.grab_bag_cluster_ids(_registry()) == {2}
    assert step08.grab_bag_cluster_ids(_registry().drop(columns="confidence")) == set()
    unknown = _registry().assign(confidence=[0.9, np.nan, np.nan])
    assert step08.grab_bag_cluster_ids(unknown) == set()  # missing is unknown, not low


def test_placement_by_cluster_then_industry_then_unclassified():
    out = step08.assign_primary_themes(_memberships(_SCORES), _clusters(), _registry(), _INDUSTRY)
    got = out.set_index("ticker")

    assert got.loc[["SW1", "BK2"], "source"].tolist() == ["cluster", "cluster"]
    assert got.loc["SW1", "primary_theme"] == "software"
    # GB1's nearest theme is banks (0.86 > 0.80) but its industry peers are in software
    assert got.loc["GB1", ["primary_theme", "source"]].tolist() == ["software", "industry"]
    assert got.loc["GB1", "membership_score"] == pytest.approx(0.80)
    assert got.loc["NZ1", ["primary_theme", "source"]].tolist() == ["banks", "industry"]
    # no theme holds a railroad / no industry known -> unclassified, never nearest-centroid
    for ticker in ("GB2", "GB3"):
        assert got.loc[ticker, "primary_theme"] == UNCLASSIFIED_THEME
        assert np.isnan(got.loc[ticker, "membership_score"])
    assert len(out) == len(_SCORES) and not out["ticker"].duplicated().any()


def test_one_stray_industry_peer_does_not_make_a_theme_eligible():
    # A single software name sitting in the banks cluster (1 of 4 = 25% of the
    # industry, but under the peer-count floor) must not pull software into banks.
    clusters = _clusters()
    clusters.loc[clusters["ticker"] == "SW4", "cluster_id"] = 1
    scores = dict(_SCORES, SW4=(0.60, 0.93), GB1=(0.50, 0.99))
    out = step08.assign_primary_themes(_memberships(scores), clusters, _registry(), _INDUSTRY)
    assert out.set_index("ticker").loc["GB1", "primary_theme"] == "software"


def test_seed_anchor_counts_as_clustered(monkeypatch):
    monkeypatch.setattr(step08, "_default_seed_members", lambda: {"mega_cap_platforms": ["NZ1"]})
    mem = pd.concat(
        [_memberships(_SCORES), pd.DataFrame([("NZ1", "mega_cap_platforms", 1.0)],
                                             columns=["ticker", "theme", "membership_score"])],
        ignore_index=True,
    )
    got = step08.assign_primary_themes(mem, _clusters(), _registry(), _INDUSTRY).set_index("ticker")
    assert got.loc["NZ1", ["primary_theme", "source"]].tolist() == ["mega_cap_platforms", "cluster"]


def test_without_clusters_it_is_plain_nearest_theme():
    out = step08.assign_primary_themes(_memberships(_SCORES), None, _registry(), _INDUSTRY)
    assert set(out["source"]) == {"similarity"}
    assert out.set_index("ticker").loc["GB1", "primary_theme"] == "banks"


def test_grab_bag_cluster_is_not_published_as_a_theme(tmp_path, monkeypatch):
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_PATH", tmp_path / "m.parquet")
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_HISTORY_PATH", tmp_path / "h.parquet")
    monkeypatch.setattr(step08, "ensure_outputs", lambda: None)
    tickers = ["SW1", "SW2", "BK1", "BK2", "GB1", "GB2"]
    matrix = np.array([[1, .1, .1], [.9, .2, .1], [.1, 1, .1], [.2, .9, .1], [.3, .3, 1], [.2, .4, .9]],
                      dtype=np.float32)
    out = step08.compute_memberships(
        embeddings_df=pd.DataFrame({"ticker": tickers, "embedding": [r.tolist() for r in matrix]}),
        clusters_df=pd.DataFrame({"ticker": tickers, "cluster_id": [0, 0, 1, 1, 2, 2]}),
        registry_df=_registry(),
        as_of=pd.Timestamp("2026-10-05", tz="UTC"),
    )
    assert "mixed_small_cap_value" not in set(out["theme"])
    assert {"software", "banks"} <= set(out["theme"])
    assert UNCLASSIFIED_THEME not in set(out["theme"])  # never a membership row


def test_step09_gives_unclassified_tickers_a_row_without_theme_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(step09, "TICKER_THEME_FEATURES_PATH", tmp_path / "f.parquet")
    monkeypatch.setattr(step09, "ensure_outputs", lambda: None)
    monkeypatch.setattr(step09, "_load_registry_or_empty", lambda: _registry().assign(date=pd.Timestamp("2026-10-05")))
    monkeypatch.setattr(step09, "load_relationships", lambda **kwargs: pd.DataFrame())
    dates = pd.bdate_range("2026-08-01", periods=40)
    rets = pd.DataFrame(0.01, index=dates, columns=list(_SCORES))
    monkeypatch.setattr(step09, "_load_daily_returns", lambda *a, **k: rets)
    mem = _memberships(_SCORES).assign(date=pd.Timestamp("2026-10-05"))
    mem.attrs["taxonomy_version"] = "taxonomy-v1"

    out = step09.build_meta_features(
        memberships_df=mem, as_of=pd.Timestamp("2026-10-05"),
        clusters_df=_clusters(), industries=_INDUSTRY, countries={},
    ).set_index("ticker")

    assert out.loc["GB2", "primary_theme"] == UNCLASSIFIED_THEME
    assert out.loc[["GB2", "GB3"], ["theme_heat_score", "primary_theme_rank", "membership_score"]].isna().all().all()
    assert out.loc["GB1", "primary_theme"] == "software" and np.isfinite(out.loc["GB1", "theme_heat_score"])
    assert set(out["primary_theme_rank"].dropna()) <= {1.0, 2.0}  # only real themes are ranked


# ── profiles ─────────────────────────────────────────────────────────────────

def _point_profiles(monkeypatch, tmp_path):
    shared, extra = tmp_path / "shared.parquet", tmp_path / "extra.parquet"
    monkeypatch.setattr(P, "TICKER_PROFILES_PATH", shared)
    monkeypatch.setattr(P, "TICKER_PROFILES_SUPPLEMENT_PATH", extra)
    return shared, extra


def test_fetch_fills_only_the_supplement_and_is_resumable(tmp_path, monkeypatch):
    shared, extra = _point_profiles(monkeypatch, tmp_path)
    pd.DataFrame({"ticker": ["AAA"], "longBusinessSummary": ["An established shared-table description."],
                  "industry": ["Banks"], "snapshot_date": [pd.Timestamp("2026-06-05")]}).to_parquet(shared)
    shared_before = shared.read_bytes()
    calls: list[list[str]] = []

    def fake_fetch(tickers, **_):
        calls.append(list(tickers))
        ok = [t for t in tickers if t != "DEAD"]
        return pd.DataFrame({"ticker": ok, "snapshot_date": pd.Timestamp("2026-10-04", tz="UTC"),
                             "longBusinessSummary": ["A long enough business description here."] * len(ok),
                             "industry": ["Software"] * len(ok)})

    monkeypatch.setattr("signals.news.sources.fetch_yfinance_profiles", fake_fetch)
    universe = ["AAA", "BBB", "CCC", "DEAD"]
    assert P.tickers_needing_profiles(universe) == ["BBB", "CCC", "DEAD"]

    assert P.fetch_missing_profiles(universe, max_tickers=2, chunk=1) == 2
    assert P.tickers_needing_profiles(universe) == ["DEAD"]          # resumes where it stopped
    assert P.fetch_missing_profiles(universe, max_tickers=5) == 0
    assert P.tickers_needing_profiles(universe) == []                # placeholder: not re-asked every run
    assert calls == [["BBB"], ["CCC"], ["DEAD"]]

    assert shared.read_bytes() == shared_before                      # live news-scorer input untouched
    assert P.industry_by_ticker() == {"AAA": "Banks", "BBB": "Software", "CCC": "Software"}
    later = pd.Timestamp("2026-10-04") + pd.Timedelta(days=45)
    assert P.tickers_needing_profiles(universe, now=later) == ["DEAD"]   # retried after PROFILE_RETRY_DAYS


def test_shared_profile_wins_over_supplement(tmp_path, monkeypatch):
    shared, extra = _point_profiles(monkeypatch, tmp_path)
    pd.DataFrame({"ticker": ["AAA"], "industry": ["Banks"]}).to_parquet(shared)
    pd.DataFrame({"ticker": ["AAA", "BBB"], "industry": ["Software", "Software"]}).to_parquet(extra)
    assert P.industry_by_ticker() == {"AAA": "Banks", "BBB": "Software"}


def test_empty_shared_row_does_not_hide_a_fresh_attempt(tmp_path, monkeypatch):
    # FISV 2026-10-05: an empty June row in the shared table masked the new
    # "tried, nothing there" marker, so the ticker was re-requested every run.
    shared, extra = _point_profiles(monkeypatch, tmp_path)
    pd.DataFrame({"ticker": ["GONE", "AAA"], "longBusinessSummary": [None, None], "industry": [None, "Banks"],
                  "snapshot_date": [pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-05")]}).to_parquet(shared)
    pd.DataFrame({"ticker": ["GONE", "AAA"], "longBusinessSummary": [None, None], "industry": [None, "Software"],
                  "snapshot_date": [pd.Timestamp("2026-10-05", tz="UTC")] * 2}).to_parquet(extra)
    now = pd.Timestamp("2026-10-06")
    assert P.tickers_needing_profiles(["GONE", "AAA"], now=now) == []
    assert P.tickers_needing_profiles(["GONE"], now=now + pd.Timedelta(days=40)) == ["GONE"]
    assert P.industry_by_ticker() == {"AAA": "Banks"}   # usable shared row still wins


def test_old_registry_dates_cannot_unpublish_this_weeks_clusters(tmp_path, monkeypatch):
    """2026-10-05 regression: the weekly pipeline hands step08 the FULL registry
    history. Cluster ids are reused weekly, so last month's grab-bag #0 must not
    unpublish this week's confident cluster #0."""
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_PATH", tmp_path / "m.parquet")
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_HISTORY_PATH", tmp_path / "h.parquet")
    monkeypatch.setattr(step08, "ensure_outputs", lambda: None)
    old = _registry().assign(theme_name=["old_mixed_a", "old_mixed_b", "old_mixed_c"],
                             confidence=[0.20, 0.30, 0.90], date=pd.Timestamp("2026-09-01"))
    new = _registry().assign(date=pd.Timestamp("2026-10-05"))
    history = pd.concat([old, new], ignore_index=True)

    assert step08.grab_bag_cluster_ids(history) == {2}          # this week's only
    assert step08.grab_bag_cluster_ids(old) == {0, 1}

    tickers = ["SW1", "SW2", "BK1", "BK2", "GB1", "GB2"]
    matrix = np.array([[1, .1, .1], [.9, .2, .1], [.1, 1, .1], [.2, .9, .1], [.3, .3, 1], [.2, .4, .9]],
                      dtype=np.float32)
    out = step08.compute_memberships(
        embeddings_df=pd.DataFrame({"ticker": tickers, "embedding": [r.tolist() for r in matrix]}),
        clusters_df=pd.DataFrame({"ticker": tickers, "cluster_id": [0, 0, 1, 1, 2, 2]}),
        registry_df=history,
        as_of=pd.Timestamp("2026-10-05", tz="UTC"),
    )
    assert {"software", "banks"} <= set(out["theme"])
    assert "mixed_small_cap_value" not in set(out["theme"])


def test_industry_singleton_in_a_confident_cluster_is_replaced():
    # ORCL-in-quantum shape, minus company: a software name clustered with banks,
    # the only software company there, belongs with software.
    clusters = pd.concat([_clusters(), pd.DataFrame({"ticker": ["ODD"], "cluster_id": [1]})], ignore_index=True)
    scores = dict(_SCORES, ODD=(0.70, 0.97))
    industry = dict(_INDUSTRY, ODD="Software")
    got = step08.assign_primary_themes(_memberships(scores), clusters, _registry(), industry,
                                       peer_pins={}).set_index("ticker")
    assert got.loc["ODD", ["primary_theme", "source"]].tolist() == ["software", "industry"]
    # two of an industry in one theme are each other's support and stay put
    clusters2 = pd.concat([clusters, pd.DataFrame({"ticker": ["ODD2"], "cluster_id": [1]})], ignore_index=True)
    got2 = step08.assign_primary_themes(
        _memberships(dict(scores, ODD2=(0.71, 0.96))), clusters2, _registry(),
        dict(industry, ODD="Railroads", ODD2="Railroads"), peer_pins={},
    ).set_index("ticker")
    assert got2.loc[["ODD", "ODD2"], "primary_theme"].tolist() == ["banks", "banks"]
    assert got2.loc[["ODD", "ODD2"], "source"].tolist() == ["cluster", "cluster"]


def test_home_market_ticker_is_not_placed_with_foreign_only_peers():
    # GOOGL/META 2026-10-05: the only theme holding their industry was a basket of Chinese ADRs.
    countries = {t: "China" for t in ("BK1", "BK2", "BK3")}
    countries.update({t: "United States" for t in ("SW1", "SW2", "SW3", "SW4", "NZ1", "GB1")})
    got = step08.assign_primary_themes(_memberships(_SCORES), _clusters(), _registry(), _INDUSTRY,
                                       countries, peer_pins={}).set_index("ticker")
    assert got.loc["NZ1", "primary_theme"] == UNCLASSIFIED_THEME      # US bank, only Chinese bank peers
    assert got.loc["GB1", "primary_theme"] == "software"               # US software with US peers
    foreign = dict(countries, NZ1="China")
    got = step08.assign_primary_themes(_memberships(_SCORES), _clusters(), _registry(), _INDUSTRY,
                                       foreign, peer_pins={}).set_index("ticker")
    assert got.loc["NZ1", ["primary_theme", "source"]].tolist() == ["banks", "industry"]


def test_peer_pin_follows_the_peers_theme_and_overrides_the_cluster():
    # BK3 is confidently clustered with banks, but pinned to go where SW1/SW2 go.
    got = step08.assign_primary_themes(_memberships(_SCORES), _clusters(), _registry(), _INDUSTRY,
                                       peer_pins={"BK3": ["SW1", "SW2", "MISSING"]}).set_index("ticker")
    assert got.loc["BK3", ["primary_theme", "source"]].tolist() == ["software", "pinned"]
    assert got.loc["BK3", "membership_score"] == pytest.approx(0.62)   # its own score in that theme
    # peers with no theme leave the ticker where it was
    got = step08.assign_primary_themes(_memberships(_SCORES), _clusters(), _registry(), _INDUSTRY,
                                       peer_pins={"BK3": ["GB2", "GB3"]}).set_index("ticker")
    assert got.loc["BK3", ["primary_theme", "source"]].tolist() == ["banks", "cluster"]
