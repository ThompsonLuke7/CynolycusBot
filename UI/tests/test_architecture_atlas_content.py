"""Curated-manifest coverage checks for the approved System Atlas."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "UI/architecture_atlas/source/architecture.json"


def test_manifest_covers_the_six_approved_root_domains() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    assert data["schema_version"] == 1
    assert {node_id for node_id, node in nodes.items() if node.get("parent_id") == "system"} == {
        "domain.inputs",
        "domain.fabric",
        "domain.context",
        "domain.strategies",
        "domain.execution",
        "domain.research",
    }


def test_momentum_and_swing_have_complete_variable_depth_paths() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    labels = {node["id"]: node["public"]["label"].lower() for node in nodes}
    strategies = [node_id for node_id, label in labels.items() if label in {"momentum expansion", "multi-ticker swing"}]
    assert len(strategies) == 2

    children: dict[str, list[str]] = {}
    for node in nodes:
        if node.get("parent_id"):
            children.setdefault(node["parent_id"], []).append(node["id"])
    for strategy in strategies:
        descendants: set[str] = set()
        pending = list(children.get(strategy, []))
        while pending:
            current = pending.pop()
            descendants.add(current)
            pending.extend(children.get(current, []))
        descendant_kinds = {next(node["kind"] for node in nodes if node["id"] == node_id) for node_id in descendants}
        for required in ("feature", "model", "signal", "policy", "execution", "audit"):
            assert required in descendant_kinds

        def max_depth(node_id: str) -> int:
            return 1 + max((max_depth(child) for child in children.get(node_id, [])), default=0)

        assert max_depth(strategy) >= 3


def test_large_display_positions_are_collision_free_for_every_scope() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    children: dict[str, list[dict]] = {}
    for node in data["nodes"]:
        if node.get("parent_id"):
            children.setdefault(node["parent_id"], []).append(node)

    # These are deliberately conservative rectangular bounds for the largest
    # graph mode, matching graphScale() and graphSpacingScale() in atlas.js.
    spacing = 1.18
    child_width, child_height = 176 * 1.24, 96 * 1.24
    focus_width = focus_height = 170 * 1.24

    for scope_id, scope_children in children.items():
        boxes = [
            (
                scope_id,
                500 - focus_width / 2,
                330 - focus_height / 2,
                500 + focus_width / 2,
                330 + focus_height / 2,
            )
        ]
        for node in scope_children:
            position = node["position"]
            x = 500 + (position["x"] - 500) * spacing
            y = 330 + (position["y"] - 330) * spacing
            boxes.append(
                (
                    node["id"],
                    x - child_width / 2,
                    y - child_height / 2,
                    x + child_width / 2,
                    y + child_height / 2,
                )
            )

        collisions = []
        for index, left in enumerate(boxes):
            for right in boxes[index + 1 :]:
                overlap_x = min(left[3], right[3]) - max(left[1], right[1])
                overlap_y = min(left[4], right[4]) - max(left[2], right[2])
                if overlap_x > 0 and overlap_y > 0:
                    collisions.append((left[0], right[0]))
        assert not collisions, f"{scope_id} has large-display collisions: {collisions}"
