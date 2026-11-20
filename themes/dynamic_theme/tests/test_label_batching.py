"""Clusters are labeled CLAUDE_LABEL_BATCH_SIZE per call, with checked output.

Replaced one call per cluster (~110 a week). What must hold:
  * batches of the configured size, later batches seeing names minted earlier;
  * every cluster gets exactly one usable label -- anything missing or unusable
    in a batch answer is retried one cluster per call;
  * the request uses the label budget, effort and JSON schema.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from themes.dynamic_theme import config
from themes.dynamic_theme.stages import step05_claude_labeling as s5


def _clusters(ids):
    return [{"cluster_id": i, "tickers": [f"T{i}"], "top_keywords": ["k"], "sample_headlines": ["h"]}
            for i in ids]


def _ids_in(prompt: str) -> list[int]:
    return [c["cluster_id"] for c in json.loads(prompt.split("CLUSTERS", 1)[1].split(":", 1)[1])]


def _label(cid, name=None, conf=0.8):
    return {"cluster_id": cid, "theme_name": name or f"theme_{cid}", "parent_theme": "p",
            "description": "d", "related_themes": ["x"], "confidence": conf}


def test_batches_are_sized_and_later_batches_see_earlier_names(monkeypatch):
    monkeypatch.setattr(s5, "CLAUDE_LABEL_BATCH_SIZE", 15)
    prompts = []

    def fake_json(prompt, **kwargs):
        prompts.append(prompt)
        return {"labels": [_label(i) for i in _ids_in(prompt)]}

    monkeypatch.setattr(s5, "call_claude_json", fake_json)
    known = {"semis"}
    labels = s5._label_new_clusters(_clusters(range(32)), known)

    assert [len(_ids_in(p)) for p in prompts] == [15, 15, 2]
    assert sorted(labels) == list(range(32))
    assert '"theme_0"' in prompts[1] and '"theme_0"' not in prompts[0]
    assert {"semis", "theme_31"} <= known


def test_missing_and_unusable_labels_are_retried_one_per_call(monkeypatch):
    prompts = []

    def fake_json(prompt, **kwargs):
        prompts.append(prompt)
        ids = _ids_in(prompt)
        if len(ids) > 1:  # batch answer: 2 missing, 3 placeholder, 4 bad confidence, 99 not asked
            return {"labels": [_label(1), _label(3, "cluster_3"), _label(4, conf=1.7), _label(99)]}
        return {"labels": [_label(ids[0], f"retried_{ids[0]}")]}

    monkeypatch.setattr(s5, "call_claude_json", fake_json)
    labels = s5._label_new_clusters(_clusters([1, 2, 3, 4]), set())

    assert [_ids_in(p) for p in prompts[1:]] == [[2], [3], [4]]
    assert {c: l["theme_name"] for c, l in labels.items()} == {
        1: "theme_1", 2: "retried_2", 3: "retried_3", 4: "retried_4",
    }


def test_labels_are_normalized_to_snake_case(monkeypatch):
    monkeypatch.setattr(s5, "call_claude_json", lambda prompt, **kw: {"labels": [
        {"cluster_id": 1, "theme_name": " AI Infrastructure ", "parent_theme": "Information Technology",
         "description": " d ", "related_themes": ["Cloud Computing", ""], "confidence": 0.7}]})
    label = s5._label_batch(_clusters([1]), [])[1]
    assert label["theme_name"] == "ai_infrastructure"
    assert label["parent_theme"] == "information_technology"
    assert label["related_themes"] == ["cloud_computing"]
    assert label["description"] == "d"


def test_batch_request_uses_label_budget_effort_and_schema(monkeypatch):
    seen = {}

    def fake_json(prompt, **kwargs):
        seen.update(kwargs)
        return {"labels": []}

    monkeypatch.setattr(s5, "call_claude_json", fake_json)
    s5._label_batch(_clusters([1]), ["semis"])
    assert seen["max_tokens"] == config.CLAUDE_LABEL_MAX_TOKENS
    assert seen["effort"] == config.CLAUDE_LABEL_EFFORT
    assert seen["json_schema"] is s5._LABEL_SCHEMA
    assert seen["system"] == s5._LABEL_SYSTEM


def test_schema_is_closed_and_requires_every_field():
    item = s5._LABEL_SCHEMA["properties"]["labels"]["items"]
    assert item["additionalProperties"] is False and s5._LABEL_SCHEMA["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def _carry_semis_to_cluster_2(tmp_path, monkeypatch):
    monkeypatch.setattr(s5, "THEME_REGISTRY_PATH", tmp_path / "reg.parquet")
    monkeypatch.setattr(s5, "_stability_centroids", lambda: ({}, {}))
    monkeypatch.setattr(s5, "_match_prior_theme_scored",
                        lambda cid, cur, prior: ("semis", 0.95) if cid == 2 else None)
    monkeypatch.setattr(s5, "_latest_theme_metadata", lambda: {
        "semis": {"confidence": 0.9, "parent_theme": "tech", "description": "NVDA AMD chips (stale)",
                  "related_themes": '["ai"]'}})


def _clusters_json(prompt):
    return json.loads(prompt.split("CLUSTERS", 1)[1].split(":", 1)[1])


def test_carried_cluster_is_sent_with_its_name_and_gets_a_fresh_description(tmp_path, monkeypatch):
    _carry_semis_to_cluster_2(tmp_path, monkeypatch)
    sent = []

    def fake_json(prompt, **kwargs):
        sent.extend(_clusters_json(prompt))
        return {"labels": [_label(1), _label(2, "semis", conf=0.7) | {"description": "fresh"}, _label(3)]}

    monkeypatch.setattr(s5, "call_claude_json", fake_json)
    out = s5.label_clusters(_clusters([1, 2, 3]), as_of=pd.Timestamp("2026-11-16"))
    got = out[out["date"] == pd.Timestamp("2026-11-16")].set_index("cluster_id")
    assert [c.get("current_name") for c in sent] == [None, "semis", None]  # one call, all three
    assert got.loc[2, "theme_name"] == "semis"
    assert got.loc[2, "description"] == "fresh" and got.loc[2, "confidence"] == 0.7
    assert got.loc[1, "theme_name"] == "theme_1" and got.loc[3, "related_themes"] == '["x"]'


def test_a_clearly_wrong_carried_name_can_be_replaced(tmp_path, monkeypatch):
    _carry_semis_to_cluster_2(tmp_path, monkeypatch)
    monkeypatch.setattr(s5, "call_claude_json", lambda prompt, **kw: {"labels": [
        _label(i, "regional_banks" if i == 2 else None) for i in (1, 2, 3)]})
    out = s5.label_clusters(_clusters([1, 2, 3]), as_of=pd.Timestamp("2026-11-16"))
    assert out.set_index("cluster_id").loc[2, "theme_name"] == "regional_banks"


def test_carried_cluster_keeps_last_weeks_row_when_the_labeler_fails_on_it(tmp_path, monkeypatch):
    _carry_semis_to_cluster_2(tmp_path, monkeypatch)
    monkeypatch.setattr(s5, "call_claude_json", lambda prompt, **kw: {"labels": [
        _label(i) for i in (1, 3) if f'"cluster_id": {i}' in prompt]})
    out = s5.label_clusters(_clusters([1, 2, 3]), as_of=pd.Timestamp("2026-11-16"))
    row = out.set_index("cluster_id").loc[2]
    assert row["theme_name"] == "semis" and row["description"] == "NVDA AMD chips (stale)"
    assert row["related_themes"] == '["ai"]'


def test_prompt_shows_company_details_not_keywords():
    summary = {"cluster_id": 5, "tickers": ["NVDA", "AMD", "XYZ"], "top_keywords": ["yahoo", "finance"],
               "sample_headlines": ["h"], "current_name": "semis",
               "members": [{"ticker": "NVDA", "name": "NVIDIA", "industry": "Semiconductors", "about": "GPUs."},
                           {"ticker": "AMD", "name": "AMD", "industry": "Semiconductors", "about": "CPUs."}]}
    item = _clusters_json(s5._batch_prompt([summary], []))[0]
    assert item["n_members"] == 3 and item["current_name"] == "semis"
    assert [m["ticker"] for m in item["largest_members"]] == ["NVDA", "AMD"]
    assert item["other_tickers"] == ["XYZ"]
    assert "top_keywords" not in item and "tickers" not in item


@pytest.mark.parametrize("bad", [{}, [], {"labels": "nope"}])
def test_malformed_answer_yields_no_labels(monkeypatch, bad):
    monkeypatch.setattr(s5, "call_claude_json", lambda prompt, **kw: bad)
    assert s5._label_batch(_clusters([1]), []) == {}


def test_member_profiles_are_largest_first_with_a_one_line_business():
    from themes.dynamic_theme.stages.step04_cluster_summary import member_profiles

    profiles = pd.DataFrame([
        {"ticker": "SMALL", "shortName": "Small Co", "industry": "Banks", "marketCap": 1e9,
         "longBusinessSummary": "Small Co runs banks. It also does more things."},
        {"ticker": "BIG", "shortName": "Big Co", "industry": "Software", "marketCap": 3e12,
         "longBusinessSummary": "Big Co sells software worldwide. Founded long ago."},
    ])
    docs = pd.DataFrame([{"ticker": "NOPROF", "description": "NoProf makes widgets. More text."}])
    out = member_profiles(["SMALL", "NOPROF", "BIG"], profiles, docs, limit=3)
    assert [m["ticker"] for m in out] == ["BIG", "SMALL", "NOPROF"]  # unknown cap last
    assert out[0] == {"ticker": "BIG", "name": "Big Co", "industry": "Software",
                      "about": "Big Co sells software worldwide."}
    assert out[2] == {"ticker": "NOPROF", "about": "NoProf makes widgets."}
    assert len(member_profiles(["SMALL", "NOPROF", "BIG"], profiles, docs, limit=2)) == 2
