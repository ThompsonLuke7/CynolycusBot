"""Theme label fixes of 2026-11-12.

1. step05 never carries a low-confidence (grab-bag) prior label onto a new cluster.
2. step08 closes a place-named theme (chinese_*, latam_*) to other countries' companies.
3. step08 puts a seed anchor in its seed theme even when another theme scores higher.
4. step09 gates the week's primary themes (label_checks) before writing features.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from themes.dynamic_theme import label_checks as lc
from themes.dynamic_theme.config import UNCLASSIFIED_THEME
from themes.dynamic_theme.stages import step05_claude_labeling as s5
from themes.dynamic_theme.stages import step08_memberships as step08
from themes.dynamic_theme.stages import step09_meta_features as step09


# ── 1. carry-forward ──────────────────────────────────────────────────────────
def test_low_confidence_prior_label_is_relabeled_not_carried(tmp_path, monkeypatch):
    monkeypatch.setattr(s5, "THEME_REGISTRY_PATH", tmp_path / "reg.parquet")
    monkeypatch.setattr(s5, "_stability_centroids", lambda: ({1: "c1", 2: "c2"}, {"p": "c"}))
    monkeypatch.setattr(s5, "_match_prior_theme_scored",
                        lambda cid, cur, prior: {1: ("emerging_growth_speculative", 0.95), 2: ("semis", 0.95)}[cid])
    monkeypatch.setattr(s5, "_latest_theme_metadata", lambda: {
        "emerging_growth_speculative": {"confidence": 0.38, "parent_theme": "x", "description": "WYFI, GDS, VEON",
                                        "related_themes": "[]"},
        "semis": {"confidence": 0.90, "parent_theme": "tech", "description": "chips", "related_themes": "[]"}})
    sent = {}

    def fake_batch(summaries, known):
        sent.update({s["cluster_id"]: s.get("current_name") for s in summaries})
        return {s["cluster_id"]: {"cluster_id": s["cluster_id"],
                                  "theme_name": s.get("current_name") or "ai_infrastructure",
                                  "parent_theme": "tech", "description": "fresh",
                                  "related_themes": [], "confidence": 0.8} for s in summaries}

    monkeypatch.setattr(s5, "_label_batch", fake_batch)
    out = s5.label_clusters([{"cluster_id": 1}, {"cluster_id": 2}], as_of=pd.Timestamp("2026-11-09"))
    got = out[out["date"] == pd.Timestamp("2026-11-09")].set_index("cluster_id")
    # the 0.38 prior name is not offered as current_name; the 0.90 one is
    assert sent == {1: None, 2: "semis"}
    assert got.loc[1, "theme_name"] == "ai_infrastructure" and got.loc[1, "confidence"] == 0.8
    assert got.loc[2, "theme_name"] == "semis" and got.loc[2, "description"] == "fresh"


# ── 2. place-named themes ─────────────────────────────────────────────────────
def test_theme_countries_reads_the_place_in_the_name():
    assert step08.theme_countries("chinese_internet_stocks") >= {"China", "Hong Kong"}
    assert "Brazil" in step08.theme_countries("latam_banks_financials")
    assert step08.theme_countries("korean_financial_conglomerates") == {"South Korea"}
    for open_theme in ("global_automakers", "precious_metals_mining", "software", UNCLASSIFIED_THEME, None):
        assert step08.theme_countries(open_theme) is None


def test_closed_to_spares_unknown_and_offshore_domiciles():
    themes = pd.Series(["chinese_internet_stocks"] * 5 + ["software"])
    countries = pd.Series(["Japan", "China", "Hong Kong", "Cayman Islands", np.nan, "Japan"])
    assert step08.closed_to(themes, countries).tolist() == [True, False, False, False, False, False]


_REG = pd.DataFrame({"cluster_id": [0, 1], "theme_name": ["chinese_internet_stocks", "global_automakers"],
                     "confidence": [0.9, 0.9]})
_CLU = pd.DataFrame({"ticker": ["CN1", "CN2", "CN3", "JP1", "US1", "US2", "US3", "JP2", "KY1"],
                     "cluster_id": [0, 0, 0, 0, 1, 1, 1, -1, -1]})
_IND = {t: "Auto Manufacturers" for t in _CLU["ticker"]}
_CTY = {"CN1": "China", "CN2": "China", "CN3": "China", "JP1": "Japan", "JP2": "Japan",
        "US1": "United States", "US2": "United States", "US3": "United States", "KY1": "Cayman Islands"}
_AUTO_SCORES = {"CN1": (0.95, 0.6), "CN2": (0.94, 0.6), "CN3": (0.93, 0.6), "JP1": (0.90, 0.85),
                "US1": (0.6, 0.95), "US2": (0.6, 0.94), "US3": (0.6, 0.93), "JP2": (0.92, 0.80), "KY1": (0.92, 0.80)}


def _auto_memberships() -> pd.DataFrame:
    rows = []
    for t, (cn, auto) in _AUTO_SCORES.items():
        rows += [(t, "chinese_internet_stocks", cn), (t, "global_automakers", auto)]
    return pd.DataFrame(rows, columns=["ticker", "theme", "membership_score"])


def test_foreign_names_are_shut_out_of_a_place_named_theme_on_both_routes():
    got = step08.assign_primary_themes(_auto_memberships(), _CLU, _REG, _IND, _CTY, peer_pins={}).set_index("ticker")
    assert got.loc["CN1", "primary_theme"] == "chinese_internet_stocks"
    assert got.loc["JP1", ["primary_theme", "source"]].tolist() == ["global_automakers", "industry"]  # clustered
    assert got.loc["JP2", "primary_theme"] == "global_automakers"                                   # unclustered
    assert got.loc["KY1", "primary_theme"] == "chinese_internet_stocks"                             # offshore
    assert lc.basket_violations(got.reset_index(), _CTY) == []
    old = step08.assign_primary_themes(_auto_memberships(), _CLU, _REG, _IND, None, peer_pins={}).set_index("ticker")
    assert old.loc["JP1", "primary_theme"] == "chinese_internet_stocks"     # no countries on file: no gate


def test_seed_anchor_sits_in_its_seed_theme_even_when_another_scores_higher(monkeypatch):
    monkeypatch.setattr(step08, "_default_seed_members", lambda: {"mega_cap_platforms--abc123": ["US1"]})
    mem = pd.concat([_auto_memberships(), pd.DataFrame([("US1", "mega_cap_platforms", 0.80)],
                                                       columns=["ticker", "theme", "membership_score"])],
                    ignore_index=True)
    got = step08.assign_primary_themes(mem, _CLU, _REG, _IND, _CTY, peer_pins={}).set_index("ticker")
    assert got.loc["US1", ["primary_theme", "membership_score", "source"]].tolist() == \
        ["mega_cap_platforms", 0.80, "cluster"]


# ── 4. the gate ───────────────────────────────────────────────────────────────
def test_regression_file_loads_and_covers_the_known_failures():
    cases = lc.load_regression_cases()
    tickers = [c["ticker"] for c in cases]
    assert len(tickers) == len(set(tickers)) >= 40
    assert {"NBIS", "TM", "HMC", "RACE", "STLA", "GOOGL", "AAPL", "MU"} <= set(tickers)


def test_regression_check_scores_only_what_it_can_judge():
    primary = pd.DataFrame({"ticker": ["TM", "F", "GM", "AAPL", "JPM"],
                            "primary_theme": ["chinese_internet_stocks", "global_automakers", UNCLASSIFIED_THEME,
                                              UNCLASSIFIED_THEME, "regional_banks"]})
    cases = [{"ticker": "TM", "must_match": "auto", "must_not_match": "chin", "unclassified_ok": True},
             {"ticker": "F", "must_match": "auto", "must_not_match": "", "unclassified_ok": True},
             {"ticker": "GM", "must_match": "auto", "must_not_match": "", "unclassified_ok": True},
             {"ticker": "AAPL", "must_match": "mega_cap", "must_not_match": "", "unclassified_ok": False},
             {"ticker": "JPM", "must_match": "bank", "must_not_match": "", "unclassified_ok": True},
             {"ticker": "ZZZ", "must_match": "x", "must_not_match": "y", "unclassified_ok": True}]
    r = lc.regression_check(primary, cases)
    assert len(r["forbidden"]) == 1 and r["forbidden"][0].startswith("TM")
    assert (r["passes"], r["scored"], r["skipped_unclassified"]) == (2, 4, 1)   # TM, AAPL miss; GM skipped


def test_v4_agreement_is_the_share_of_theme_peers_with_a_shared_curated_theme():
    primary = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "primary_theme": ["t"] * 4,
                            "source": ["cluster", "cluster", "cluster", "industry"]})
    v4 = {"A": {"semis"}, "B": {"semis"}, "C": {"banks"}, "D": {"semis"}}
    r = lc.v4_agreement(primary, v4)
    # A: B yes, C no -> 0.5; B: 0.5; C: 0/2; D: A,B yes, C no -> 2/3
    assert r["n"] == 4 and r["agreement"] == pytest.approx((0.5 + 0.5 + 0 + 2 / 3) / 4)
    assert lc.v4_agreement(primary, None)["agreement"] is None


def test_thin_inputs_are_not_measured_and_a_collapse_fails():
    small = pd.DataFrame({"ticker": ["A", "B"], "primary_theme": [UNCLASSIFIED_THEME] * 2, "source": "unclassified"})
    r = lc.run_label_checks(small, countries={}, cases=[], v4={"A": {"x"}}, previous_themes=5)
    assert r["ok"] and r["metrics"]["v4_agreement"] is None   # 2 tickers: no share or v4 judgement
    many = pd.DataFrame({"ticker": [f"T{i}" for i in range(200)], "primary_theme": [f"th{i % 9}" for i in range(200)],
                         "source": "cluster"})
    r = lc.run_label_checks(many, countries={}, cases=[], v4=None, previous_themes=80)
    assert not r["ok"] and "9 themes published vs 80" in r["failures"][0]   # the 2026-10-05 failure


def _step09_inputs(monkeypatch, tmp_path, cases):
    monkeypatch.setattr(step09, "TICKER_THEME_FEATURES_PATH", tmp_path / "f.parquet")
    monkeypatch.setattr(step09, "ensure_outputs", lambda: None)
    monkeypatch.setattr(step09, "_load_registry_or_empty", lambda: _REG.assign(date=pd.Timestamp("2026-11-09")))
    monkeypatch.setattr(step09, "load_relationships", lambda **kwargs: pd.DataFrame())
    rets = pd.DataFrame(0.01, index=pd.bdate_range("2026-09-01", periods=40), columns=list(_AUTO_SCORES))
    monkeypatch.setattr(step09, "_load_daily_returns", lambda *a, **k: rets)
    monkeypatch.setattr(step09, "load_regression_cases", lambda: cases)
    monkeypatch.setattr(step09, "load_v4_map", lambda: None)
    mem = _auto_memberships().assign(date=pd.Timestamp("2026-11-09"))
    mem.attrs["taxonomy_version"] = "taxonomy-v1"
    return mem


def test_failed_gate_raises_before_writing_features(tmp_path, monkeypatch):
    impossible = [{"ticker": "CN1", "must_match": "", "must_not_match": "chin", "unclassified_ok": True}]
    mem = _step09_inputs(monkeypatch, tmp_path, impossible)
    with pytest.raises(lc.LabelGateError, match="forbidden"):
        step09.build_meta_features(memberships_df=mem, as_of=pd.Timestamp("2026-11-09"), clusters_df=_CLU,
                                   industries=_IND, countries=_CTY)
    assert not (tmp_path / "f.parquet").exists()
    report = json.loads((tmp_path / "label_check_latest.json").read_text())
    assert report["ok"] is False and report["enforced"] is True
    out = step09.build_meta_features(memberships_df=mem, as_of=pd.Timestamp("2026-11-09"), clusters_df=_CLU,
                                     industries=_IND, countries=_CTY, enforce_label_gate=False)
    assert (tmp_path / "f.parquet").exists() and len(out) == len(_AUTO_SCORES)


def test_passing_gate_writes_features_and_a_report(tmp_path, monkeypatch):
    mem = _step09_inputs(monkeypatch, tmp_path, [])
    out = step09.build_meta_features(memberships_df=mem, as_of=pd.Timestamp("2026-11-09"), clusters_df=_CLU,
                                     industries=_IND, countries=_CTY).set_index("ticker")
    assert out.loc["JP1", "primary_theme"] == "global_automakers"
    assert json.loads((tmp_path / "label_check_latest.json").read_text())["ok"] is True
