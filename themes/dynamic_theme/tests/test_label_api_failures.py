"""API failures during theme labeling must stop cleanly, not degrade.

2026-11-16: the prepaid API balance ran out after 102 of 110 labels. Each of the
remaining clusters burned three attempts, fell back to a ``cluster_<id>`` name,
and that placeholder was written into theme_registry.parquet before step08
rejected it -- leaving the registry contaminated for the next run.
"""
from __future__ import annotations

import anthropic
import httpx
import pandas as pd
import pytest

from themes.dynamic_theme.client import claude_client as cc
from themes.dynamic_theme.stages import step05_claude_labeling as s5


def _bad_request() -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx.Response(400, request=request),
        body=None,
    )


def _block(kind: str, **fields):
    return type("Block", (), {"type": kind, **fields})()


def _message(text: str = "", *, stop_reason: str = "end_turn", thinking: bool = True):
    content = ([_block("thinking", thinking="")] if thinking else []) + [_block("text", text=text)]
    return type("Msg", (), {"content": content, "stop_reason": stop_reason})()


class _FakeStream:
    def __init__(self, outcome):
        self._outcome = outcome

    def __enter__(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._outcome if not isinstance(self._outcome, str) else _message(self._outcome)


def _fake_client(outcomes: list, calls: list):
    class Messages:
        def stream(self, **kwargs):
            calls.append(kwargs)
            return _FakeStream(outcomes.pop(0))

    beta = type("Beta", (), {"messages": Messages()})()
    return type("Client", (), {"beta": beta})()


# ── client ────────────────────────────────────────────────────────────────────
def test_credit_error_raises_at_once_without_retry(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(cc, "_client", lambda: _fake_client([_bad_request()] * 3, calls))
    monkeypatch.setattr(cc.time, "sleep", sleeps.append)
    with pytest.raises(cc.ClaudeUnavailableError, match="credit balance"):
        cc.call_claude("prompt")
    assert len(calls) == 1 and sleeps == []


def test_transient_errors_are_still_retried(monkeypatch):
    calls = []
    outcomes = [ConnectionError("reset"), ConnectionError("reset"), '{"ok": true}']
    monkeypatch.setattr(cc, "_client", lambda: _fake_client(outcomes, calls))
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    assert cc.call_claude("prompt") == '{"ok": true}'
    assert len(calls) == 3


def test_text_is_read_by_block_type_not_position(monkeypatch):
    """Adaptive thinking puts a thinking block first; content[0].text would crash."""
    monkeypatch.setattr(cc, "_client", lambda: _fake_client(['{"a": 1}'], []))
    assert cc.call_claude_json("prompt") == {"a": 1}


def test_refusal_is_not_retried(monkeypatch):
    calls = []
    refused = _message("", stop_reason="refusal")
    monkeypatch.setattr(cc, "_client", lambda: _fake_client([refused] * 3, calls))
    with pytest.raises(cc.ClaudeResponseError, match="refusal"):
        cc.call_claude("prompt")
    assert len(calls) == 1


def test_truncation_raises_only_for_structured_output(monkeypatch):
    cut = _message('[{"a": 1}, {"b"', stop_reason="max_tokens")
    monkeypatch.setattr(cc, "_client", lambda: _fake_client([cut, cut], []))
    with pytest.raises(cc.ClaudeResponseError, match="max_tokens"):
        cc.call_claude("prompt", json_schema={"type": "object"})
    # plain text: returned as is, so step07 can salvage the complete objects
    assert cc.call_claude("prompt") == '[{"a": 1}, {"b"'


def test_request_carries_system_effort_schema_and_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "_client", lambda: _fake_client(["{}", "{}"], calls))
    cc.call_claude("p", system="sys", effort="medium", json_schema={"type": "object"})
    sent = calls[0]
    assert sent["model"] == cc.CLAUDE_MODEL == "claude-sonnet-5-5"
    assert sent["system"] == "sys"
    assert sent["output_config"] == {
        "effort": "medium", "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert sent["fallbacks"] == [{"model": "claude-sonnet-5"}]
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in sent and "thinking" not in sent
    cc.call_claude("p", model="claude-opus-5-5")  # no fallback mapping for other models
    assert "fallbacks" not in calls[1] and "output_config" not in calls[1]


# ── step05 ────────────────────────────────────────────────────────────────────
def _no_carry_forward(monkeypatch, registry_path):
    monkeypatch.setattr(s5, "THEME_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(s5, "ensure_outputs", lambda: None)
    monkeypatch.setattr(s5, "_stability_centroids", lambda: ({}, {}))
    monkeypatch.setattr(s5, "_match_prior_theme_scored", lambda cid, cur, prior: None)
    monkeypatch.setattr(s5, "_latest_theme_metadata", lambda: {})


def _prior_registry(path) -> pd.DataFrame:
    prior = pd.DataFrame([{
        "cluster_id": 7, "theme_name": "semis", "parent_theme": "tech", "description": "chips",
        "related_themes": "[]", "confidence": 0.9, "date": pd.Timestamp("2026-11-03"),
    }])
    prior.to_parquet(path, index=False)
    return prior


def _clusters(ids):
    return [{"cluster_id": i, "tickers": [f"T{i}"], "top_keywords": [], "sample_headlines": []} for i in ids]


def test_api_unavailable_stops_labeling_and_leaves_registry_untouched(tmp_path, monkeypatch):
    path = tmp_path / "reg.parquet"
    prior = _prior_registry(path)
    _no_carry_forward(monkeypatch, path)
    monkeypatch.setattr(s5, "CLAUDE_LABEL_BATCH_SIZE", 2)
    calls = []

    def fake_json(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise cc.ClaudeUnavailableError("credit balance is too low")
        return {"labels": [
            {"cluster_id": i, "theme_name": f"theme_{i}", "parent_theme": "p",
             "description": "d", "related_themes": [], "confidence": 0.8} for i in (1, 2)
        ]}

    monkeypatch.setattr(s5, "call_claude_json", fake_json)
    with pytest.raises(cc.ClaudeUnavailableError):
        s5.label_clusters(_clusters([1, 2, 3, 4, 5]), as_of=pd.Timestamp("2026-11-16"))
    assert len(calls) == 2  # batch 3 (cluster 5) never sent
    pd.testing.assert_frame_equal(pd.read_parquet(path), prior)


def test_cluster_that_never_gets_a_label_is_not_written(tmp_path, monkeypatch):
    path = tmp_path / "reg.parquet"
    prior = _prior_registry(path)
    _no_carry_forward(monkeypatch, path)

    def fake_json(prompt, **kwargs):  # cluster 2 is never labeled, batch or retry
        return {"labels": [
            {"cluster_id": i, "theme_name": f"theme_{i}", "parent_theme": "p",
             "description": "d", "related_themes": [], "confidence": 0.8}
            for i in (1, 3) if f'"cluster_id": {i}' in prompt
        ]}

    monkeypatch.setattr(s5, "call_claude_json", fake_json)
    with pytest.raises(s5.LabelingIncompleteError, match=r"\[2\]"):
        s5.label_clusters(_clusters([1, 2, 3]), as_of=pd.Timestamp("2026-11-16"))
    pd.testing.assert_frame_equal(pd.read_parquet(path), prior)
