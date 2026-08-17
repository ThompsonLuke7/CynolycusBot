"""Cluster labeling must request its own token budget, not the generic fallback.

`CLAUDE_LABEL_MAX_TOKENS` existed but was never passed, so labeling silently ran
on the 512-token `CLAUDE_MAX_TOKENS` fallback. On 2026-08-17 an ambiguous cluster
drew a reasoning preamble that consumed the whole budget; the response ended
mid-sentence with no JSON object, and the cluster fell back to an ephemeral
`cluster_104` name that the step08 durability guard then rejected.
"""

from __future__ import annotations

import pytest

from themes.dynamic_theme import config
from themes.dynamic_theme.stages import step05_claude_labeling as labeling

pytestmark = pytest.mark.safe


def _summary() -> dict:
    return {
        "cluster_id": 7,
        "tickers": ["AAA", "BBB"],
        "top_keywords": ["solar", "grid"],
        "sample_headlines": ["AAA wins grid contract"],
    }


def test_label_budget_leaves_room_for_a_preamble():
    """512 was sized for the JSON alone; a preamble has to fit alongside it."""
    assert config.CLAUDE_LABEL_MAX_TOKENS >= 1024


def test_label_cluster_requests_the_label_budget(monkeypatch):
    seen: dict[str, object] = {}

    def fake_call(prompt, **kwargs):
        seen.update(kwargs)
        return {"theme_name": "clean_energy", "parent_theme": "energy",
                "description": "d", "related_themes": [], "confidence": 0.9}

    monkeypatch.setattr(labeling, "call_claude_json", fake_call)

    labeling._label_cluster(_summary(), [])

    assert seen.get("max_tokens") == config.CLAUDE_LABEL_MAX_TOKENS, (
        "labeling fell back to the generic CLAUDE_MAX_TOKENS budget"
    )


def test_labeling_failure_is_still_reported_as_ephemeral(monkeypatch):
    """The fallback stays — but it must remain visible, not silently plausible."""
    def boom(prompt, **kwargs):
        raise ValueError("no json")

    monkeypatch.setattr(labeling, "call_claude_json", boom)

    result = labeling._label_cluster(_summary(), [])

    assert result["theme_name"] == "cluster_7"
    assert result["confidence"] == 0.0
