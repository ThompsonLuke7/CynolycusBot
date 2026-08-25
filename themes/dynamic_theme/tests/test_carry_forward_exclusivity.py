"""A prior theme name may be carried forward by at most one cluster.

Carry-forward matched each cluster to its nearest prior centroid independently,
so nothing stopped several clusters claiming the same name. Membership scores
are per cluster, so the duplicates produced one (ticker, theme) pair with two
different scores and step08's immutable-history guard aborted the run. On
2026-08-17 that collapsed 187 clusters onto 88 names (mortgage_reits x7).
"""

from __future__ import annotations

import pytest

from themes.dynamic_theme.stages.step05_claude_labeling import _resolve_carry_forward

pytestmark = pytest.mark.safe


def test_contested_name_goes_to_the_closest_cluster():
    carried, displaced = _resolve_carry_forward({
        1: ("mortgage_reits", 0.71),
        2: ("mortgage_reits", 0.93),
        3: ("mortgage_reits", 0.80),
    })

    assert carried == {2: "mortgage_reits"}, "the strongest match should win"
    assert displaced == 2


def test_uncontested_names_are_all_kept():
    carried, displaced = _resolve_carry_forward({
        1: ("mortgage_reits", 0.71),
        2: ("cloud_cybersecurity", 0.66),
    })

    assert carried == {1: "mortgage_reits", 2: "cloud_cybersecurity"}
    assert displaced == 0


def test_every_name_is_awarded_at_most_once():
    """The property the step08 guard actually depends on."""
    proposals = {cid: (f"theme_{cid % 7}", 0.5 + cid / 1000) for cid in range(50)}

    carried, displaced = _resolve_carry_forward(proposals)

    assert len(set(carried.values())) == len(carried), "a name was reused"
    assert len(carried) == 7
    assert displaced == 43


def test_resolution_is_deterministic_on_tied_similarity():
    tied = {9: ("energy", 0.80), 4: ("energy", 0.80), 7: ("energy", 0.80)}

    first, _ = _resolve_carry_forward(tied)
    second, _ = _resolve_carry_forward(dict(reversed(list(tied.items()))))

    assert first == second == {4: "energy"}, "ties must break on cluster id"


def test_no_proposals_carries_nothing():
    assert _resolve_carry_forward({}) == ({}, 0)
