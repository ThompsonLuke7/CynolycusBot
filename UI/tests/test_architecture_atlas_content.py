"""Curated-manifest coverage checks for the approved System Atlas."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "UI/architecture_atlas/source/architecture.json"


def test_manifest_covers_the_eight_approved_root_domains() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    assert data["schema_version"] == 1
    assert {node_id for node_id, node in nodes.items() if node.get("parent_id") == "system"} == {
        "domain.inputs",
        "domain.fabric",
        "domain.context",
        "domain.strategies",
        "domain.governance",
        "domain.execution",
        "domain.research",
        "domain.runtime",
    }


def test_the_governed_path_and_runtime_harness_are_mapped() -> None:
    """The atlas must show how an intent becomes an order, and what schedules it.

    Both domains were absent while the governed execution path was being built,
    which is exactly when a stale map is most misleading.
    """
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    children: dict[str, set[str]] = {}
    for node in data["nodes"]:
        if node.get("parent_id"):
            children.setdefault(node["parent_id"], set()).add(node["id"])

    assert {"ns.contracts", "ns.context", "ns.policy", "ns.coordinator", "ns.execution"} <= children["domain.governance"]
    # Every state type the contracts package defines needs a node, including the
    # ones no producer publishes yet -- an unpublished state is a gap to see.
    assert len(children["ns.contracts"]) == 7
    assert "ns.state.dealer" in children["ns.contracts"]
    assert {"runtime.supervisor", "runtime.server", "runtime.schedule", "runtime.guards"} <= children["domain.runtime"]


def test_dealer_positioning_records_its_gex_consumers() -> None:
    """Gamma exposure is computed once and read by several modules.

    The manifest previously showed only two of those consumers, which understated
    how widely one nightly capture is trusted.
    """
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    consumers = {edge["target"] for edge in data["edges"] if edge["source"] == "strategy.dealer"}
    assert {
        "strategy.dealer_ranker",
        "strategy.structure",
        "strategy.momentum",
        "strategy.htf",
        "strategy.meta",
        "strategy.swing",
    } <= consumers


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

    # The selected scope is a header, not a center graph node. Bounds match the
    # larger readable graph labels in the largest graph mode.
    spacing = 1.18
    child_width, child_height = 176 * 1.24, 96 * 1.24

    for scope_id, scope_children in children.items():
        boxes = []
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


def test_every_graph_label_wraps_on_word_boundaries() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def wrap(label: str, limit: int) -> list[str]:
        lines: list[str] = []
        line = ""
        for word in label.split():
            candidate = f"{line} {word}".strip()
            if line and len(candidate) > limit:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    for node in data["nodes"]:
        lines = wrap(node["public"]["label"], 18)
        assert all(line.strip() == line and line for line in lines)
        assert " ".join(lines) == node["public"]["label"]
        assert all(len(line) <= 18 or " " not in line for line in lines)
