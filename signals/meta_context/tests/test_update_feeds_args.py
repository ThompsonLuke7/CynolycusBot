"""update_feeds.py job selection: --relabel-themes reaches the theme pipeline only."""
from __future__ import annotations

import sys

import pytest

import signals.meta_context.meta_ranker.update_feeds as uf


def _jobs(monkeypatch, *argv: str) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(uf, "_run", lambda label, cmd, timeout: calls.append(cmd[1:]) or True)
    monkeypatch.setattr(sys, "argv", ["update_feeds.py", *argv])
    uf.main()
    return calls


def test_weekly_carries_labels_forward_by_default(monkeypatch):
    theme = [c for c in _jobs(monkeypatch, "--weekly") if "themes/dynamic_theme/pipeline.py" in c]
    assert theme == [["themes/dynamic_theme/pipeline.py", "--mode", "weekly"]]


def test_relabel_themes_passes_relabel_all_to_theme_pipeline_only(monkeypatch):
    calls = _jobs(monkeypatch, "--weekly", "--relabel-themes")
    assert [c for c in calls if "--relabel-all" in c] == [
        ["themes/dynamic_theme/pipeline.py", "--mode", "weekly", "--relabel-all"]
    ]


def test_relabel_themes_without_weekly_is_rejected(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _jobs(monkeypatch, "--relabel-themes")
    assert exc.value.code == 2
