"""Offline contract tests for the Architecture Atlas builder."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.build_architecture_atlas import AtlasBuildError, build_atlas, check_atlas


def _node(
    node_id: str,
    parent_id: str | None,
    *,
    visibility: str = "public",
    kind: str = "domain",
    evidence: dict | None = None,
) -> dict:
    node = {
        "id": node_id,
        "kind": kind,
        "visibility": visibility,
        "edge_color_role": "data",
        "position": {"x": 10, "y": 20},
        "public": {
            "label": node_id.replace(".", " ").title(),
            "summary": "Safe, causal architecture description.",
            "maturity": "research-ready",
            "mode": "research-only",
            "repo_paths": ["pkg"],
        },
        "evidence": evidence or {"required_paths": ["pkg"]},
    }
    if parent_id is not None:
        node["parent_id"] = parent_id
    if visibility == "local":
        node["local"] = {"details": "Internal scheduler ownership.", "repo_paths": ["private"]}
    return node


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "nodes": [
            _node("system", None),
            _node("domain", "system"),
            _node("strategy", "domain", kind="strategy"),
            _node("feature", "strategy", kind="feature"),
            _node("local.runtime", "domain", visibility="local", kind="runtime"),
        ],
        "edges": [
            {
                "id": "domain-to-strategy",
                "source": "domain",
                "target": "strategy",
                "type": "data",
                "visibility": "public",
                "public": {"label": "provides"},
            },
            {
                "id": "strategy-to-feature",
                "source": "strategy",
                "target": "feature",
                "type": "feature",
                "visibility": "public",
            },
            {
                "id": "strategy-to-local",
                "source": "strategy",
                "target": "local.runtime",
                "type": "control",
                "visibility": "local",
                "local": {"label": "owns"},
            },
        ],
    }


@pytest.fixture
def atlas_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "private").mkdir()
    static = tmp_path / "UI" / "architecture_atlas" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text(
        "<!doctype html><html><head>/*__ATLAS_DATA__*/"
        "<link rel='stylesheet' href='atlas.css?v=__ATLAS_ASSET_REV__'></head>"
        "<body><script src='atlas.js?v=__ATLAS_ASSET_REV__'></script></body></html>",
        encoding="utf-8",
    )
    (static / "atlas.js").write_text("window.started = true;", encoding="utf-8")
    (static / "atlas.css").write_text("body { color: white; }", encoding="utf-8")
    source = tmp_path / "UI" / "architecture_atlas" / "source"
    source.mkdir(parents=True)
    (source / "architecture.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    return tmp_path


def _write_manifest(repo: Path, value: dict) -> None:
    (repo / "UI/architecture_atlas/source/architecture.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


def _embedded_data(index: Path) -> dict:
    text = index.read_text(encoding="utf-8")
    match = re.search(r"window\.ATLAS_DATA\s*=\s*(\{.*?\});</script>", text)
    assert match, "expected classic inline ATLAS_DATA script"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_valid_build_is_deterministic_and_embeds_the_right_datasets(atlas_repo: Path) -> None:
    first = build_atlas(
        repo_root=atlas_repo,
        build_time="2026-07-25T12:00:00Z",
        git_revision="test-revision",
    )
    public = _embedded_data(atlas_repo / "UI/architecture_atlas/dist/public/index.html")
    local = _embedded_data(atlas_repo / "UI/architecture_atlas/dist/local/index.html")

    assert first["source_revision"] == "test-revision"
    assert first["source_manifest_sha256"] == hashlib.sha256(
        (atlas_repo / "UI/architecture_atlas/source/architecture.json").read_bytes()
    ).hexdigest()
    assert list(public["datasets"]) == ["public"]
    assert list(local["datasets"]) == ["public", "local"]
    assert {node["id"] for node in public["datasets"]["public"]["nodes"]} == {
        "system", "domain", "strategy", "feature"
    }
    assert {node["id"] for node in local["datasets"]["local"]["nodes"]} == {
        "system", "domain", "strategy", "feature", "local.runtime"
    }
    assert all("local" not in node for node in public["datasets"]["public"]["nodes"])
    assert (atlas_repo / "UI/architecture_atlas/dist/public/atlas.js").is_file()
    public_html = (atlas_repo / "UI/architecture_atlas/dist/public/index.html").read_text(encoding="utf-8")
    asset_revision = first["source_manifest_sha256"][:12]
    assert "__ATLAS_ASSET_REV__" not in public_html
    assert f"atlas.css?v={asset_revision}" in public_html
    assert f"atlas.js?v={asset_revision}" in public_html

    second = build_atlas(
        repo_root=atlas_repo,
        build_time="2026-07-25T12:00:00Z",
        git_revision="test-revision",
    )
    assert first == second


def test_static_app_includes_large_display_mode() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "architecture_atlas" / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    css = (static_dir / "atlas.css").read_text(encoding="utf-8")
    javascript = (static_dir / "atlas.js").read_text(encoding="utf-8")

    assert 'id="large-text-toggle"' in html
    assert "body.large-display" in css
    assert "min-width: 2560px" in javascript
    assert "cynolycus-atlas-large-display" in javascript
    assert 'id="context-dock"' in html
    assert "text-overflow-wrap" in javascript
    assert '"shape": "cutrectangle"' in javascript
    assert '"shape": "hexagon"' in javascript
    assert "edge.active-flow" in javascript


def test_check_validates_without_writing_dist(atlas_repo: Path) -> None:
    metadata = check_atlas(
        repo_root=atlas_repo,
        git_revision="test-revision",
    )

    assert metadata["validation_counts"] == {"nodes": 5, "edges": 3, "warnings": 0}
    assert metadata["source_revision"] == "test-revision"
    assert not (atlas_repo / "UI/architecture_atlas/dist").exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data["nodes"].append(dict(data["nodes"][0])), "duplicate node id"),
        (lambda data: data["nodes"][1].update(parent_id="not-here"), "missing parent"),
        (
            lambda data: (data["nodes"][1].update(parent_id="strategy"), data["nodes"][2].update(parent_id="domain")),
            "cycle",
        ),
        (lambda data: data["edges"][0].update(target="not-here"), "unknown endpoint"),
        (lambda data: data["edges"][0].update(type="teleport"), "unknown edge type"),
        (lambda data: data["nodes"][0].update(position={"x": "x", "y": 1}), "position"),
    ],
)
def test_invalid_graphs_are_rejected(atlas_repo: Path, mutate, message: str) -> None:
    manifest = _manifest()
    mutate(manifest)
    _write_manifest(atlas_repo, manifest)
    with pytest.raises(AtlasBuildError, match=message):
        build_atlas(repo_root=atlas_repo)


def test_required_paths_symbols_imports_and_exact_text_evidence(atlas_repo: Path) -> None:
    module = atlas_repo / "pkg/module.py"
    module.write_text("from pkg.other import Thing\n\ndef present():\n    return Thing\n", encoding="utf-8")
    (atlas_repo / "pkg/other.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    manifest = _manifest()
    manifest["nodes"][1]["evidence"] = {
        "required_paths": ["pkg/module.py"],
        "symbols": [{"path": "pkg/module.py", "name": "present", "kind": "function"}],
        "imports": [{"path": "pkg/module.py", "module": "pkg.other", "name": "Thing"}],
        "text": [{"path": "pkg/module.py", "pattern": "return Thing"}],
    }
    _write_manifest(atlas_repo, manifest)
    build_atlas(repo_root=atlas_repo)

    manifest["nodes"][1]["evidence"]["symbols"][0]["name"] = "missing"
    _write_manifest(atlas_repo, manifest)
    with pytest.raises(AtlasBuildError, match="symbol"):
        build_atlas(repo_root=atlas_repo)


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ({"required_paths": ["pkg/module.py"], "imports": [{"path": "pkg/module.py", "module": "missing.module"}]}, "import unresolved"),
        ({"required_paths": ["pkg/module.py"], "text": [{"path": "pkg/module.py", "pattern": "not present"}]}, "text evidence unresolved"),
    ],
)
def test_unproven_declared_import_or_text_evidence_fails(atlas_repo: Path, evidence: dict, message: str) -> None:
    (atlas_repo / "pkg/module.py").write_text("from pkg.other import Thing\n", encoding="utf-8")
    (atlas_repo / "pkg/other.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    manifest = _manifest()
    manifest["nodes"][1]["evidence"] = evidence
    _write_manifest(atlas_repo, manifest)
    with pytest.raises(AtlasBuildError, match=message):
        build_atlas(repo_root=atlas_repo)


def test_public_redaction_rejects_absolute_paths_and_credentials(atlas_repo: Path) -> None:
    manifest = _manifest()
    manifest["nodes"][1]["public"]["repo_paths"] = ["/home/not-safe"]
    _write_manifest(atlas_repo, manifest)
    with pytest.raises(AtlasBuildError, match="absolute"):
        build_atlas(repo_root=atlas_repo)

    manifest = _manifest()
    manifest["nodes"][1]["public"]["summary"] = "API_KEY=not-safe"
    _write_manifest(atlas_repo, manifest)
    with pytest.raises(AtlasBuildError, match="credential"):
        build_atlas(repo_root=atlas_repo)

    manifest = _manifest()
    manifest["nodes"][-1]["local"]["details"] = "TOKEN=not-safe"
    _write_manifest(atlas_repo, manifest)
    with pytest.raises(AtlasBuildError, match="credential"):
        build_atlas(repo_root=atlas_repo)


def test_static_inventory_is_allowlisted(atlas_repo: Path) -> None:
    (atlas_repo / "UI/architecture_atlas/static/not-for-deployment.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AtlasBuildError, match="non-allowlisted"):
        build_atlas(repo_root=atlas_repo)


def test_failed_build_preserves_previous_outputs(atlas_repo: Path) -> None:
    build_atlas(repo_root=atlas_repo, build_time="2026-07-25T12:00:00Z")
    before = (atlas_repo / "UI/architecture_atlas/dist/public/index.html").read_bytes()
    manifest = _manifest()
    manifest["nodes"][1]["evidence"] = {"required_paths": ["missing"]}
    _write_manifest(atlas_repo, manifest)

    with pytest.raises(AtlasBuildError, match="required path"):
        build_atlas(repo_root=atlas_repo)
    assert (atlas_repo / "UI/architecture_atlas/dist/public/index.html").read_bytes() == before


def test_variable_depth_and_leaf_nodes_survive_the_dataset(atlas_repo: Path) -> None:
    build_atlas(repo_root=atlas_repo)
    data = _embedded_data(atlas_repo / "UI/architecture_atlas/dist/public/index.html")
    nodes = data["datasets"]["public"]["nodes"]
    by_parent = {node.get("parent_id") for node in nodes}
    assert "feature" not in by_parent  # feature is a real leaf
    assert next(node for node in nodes if node["id"] == "feature")["parent_id"] == "strategy"
