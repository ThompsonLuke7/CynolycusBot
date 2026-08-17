"""Theme labeling must survive a reasoning preamble before the JSON.

On 2026-08-17 twelve of 186 clusters failed to label with
``Expecting value: line 1 column 1 (char 0)`` because the response opened with
prose. Each failure fell back to an ephemeral ``cluster_<id>`` name, which the
step08 durability guard then rejected, aborting the weekly theme run.
"""

from __future__ import annotations

import json

import pytest

from themes.dynamic_theme.client import claude_client

pytestmark = pytest.mark.safe


# Verbatim shape of the responses that failed in production.
PROSE_PREAMBLE = """Looking at this cluster:

- **CMPR** (Cimpress) - mass customization/print-on-demand
- **CNXN** (PC Connection) - IT products/solutions distributor

{"theme_name": "diversified_industrial_tech", "parent_theme": "industrials",
 "description": "Mixed B2B", "related_themes": [], "confidence": 0.55}
"""


def _patch_response(monkeypatch, body: str) -> None:
    monkeypatch.setattr(
        claude_client,
        "call_claude",
        lambda prompt, **kwargs: body,
    )


def test_prose_preamble_before_json_is_recovered(monkeypatch):
    _patch_response(monkeypatch, PROSE_PREAMBLE)

    result = claude_client.call_claude_json("prompt")

    assert result["theme_name"] == "diversified_industrial_tech"
    assert result["confidence"] == 0.55


def test_plain_json_still_parses(monkeypatch):
    _patch_response(monkeypatch, json.dumps({"theme_name": "clean_energy"}))

    assert claude_client.call_claude_json("prompt")["theme_name"] == "clean_energy"


def test_fenced_json_still_parses(monkeypatch):
    _patch_response(monkeypatch, '```json\n{"theme_name": "biotech"}\n```')

    assert claude_client.call_claude_json("prompt")["theme_name"] == "biotech"


def test_braces_inside_strings_do_not_truncate_payload(monkeypatch):
    body = 'Here you go:\n{"theme_name": "odd", "description": "a } brace and { another"}'
    _patch_response(monkeypatch, body)

    result = claude_client.call_claude_json("prompt")

    assert result["description"] == "a } brace and { another"


def test_json_array_payload_is_recovered(monkeypatch):
    _patch_response(monkeypatch, 'Result:\n[{"theme_name": "a"}, {"theme_name": "b"}]')

    assert [r["theme_name"] for r in claude_client.call_claude_json("prompt")] == ["a", "b"]


def test_response_with_no_json_still_raises(monkeypatch):
    """A degraded result must not be invented from prose."""
    _patch_response(monkeypatch, "I cannot label this cluster.")

    with pytest.raises(json.JSONDecodeError):
        claude_client.call_claude_json("prompt")


def test_unbalanced_json_still_raises(monkeypatch):
    _patch_response(monkeypatch, 'Here:\n{"theme_name": "truncated"')

    with pytest.raises(json.JSONDecodeError):
        claude_client.call_claude_json("prompt")
