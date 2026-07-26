#!/usr/bin/env python3
"""Validate and build the static Interactive Architecture Atlas artifacts.

This program deliberately only reads text files.  It never imports project
modules, so validating the atlas cannot start a dashboard, touch market data,
or otherwise execute trading code.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
EDGE_TYPES = {"data", "feature", "signal", "policy", "execution", "audit", "research", "control"}
VISIBILITIES = {"public", "local"}
NODE_KINDS = {
    "system", "domain", "input", "data", "fabric", "context", "strategy", "feature",
    "model", "signal", "policy", "execution", "audit", "research", "control", "runtime",
    "integration", "evidence", "portal", "module", "source", "ui",
}
NODE_COLOR_ROLES = EDGE_TYPES | {"input", "shared", "context", "strategy", "runtime"}
MATURITY_VALUES = {"production", "operational", "paper-capable", "research-ready", "research", "experimental", "legacy", "planned"}
MODE_VALUES = {"research-only", "paper-capable", "paper", "broker-integrated", "live", "shared", "not-integrated"}
PUBLIC_FIELDS = {"label", "summary", "maturity", "mode", "repo_paths", "source_links", "tags", "status", "layer"}
NODE_FIELDS = {"id", "parent_id", "kind", "visibility", "edge_color_role", "position", "public", "local", "evidence", "tags", "layout"}
EDGE_FIELDS = {"id", "source", "target", "type", "visibility", "public", "local", "evidence", "layout"}
_ALLOWED_STATIC_FILES = {"index.html", "atlas.css", "atlas.js"}
_CREDENTIAL_ASSIGNMENT = re.compile(r"\b(?:api[_-]?key|secret|token|password|passwd|credential)\b\s*[:=]\s*[^\s]+", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"(?:^|[\s\"'])?[A-Za-z]:\\")
_POSIX_PATH = re.compile(r"(?:^|[\s\"'])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")


class AtlasBuildError(ValueError):
    """The source manifest cannot safely produce an Atlas artifact."""


def _fail(message: str) -> None:
    raise AtlasBuildError(message)


def _is_relative_repo_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and not bool(_WINDOWS_PATH.search(value))


def _repo_path(repo_root: Path, value: Any, context: str) -> Path:
    if not _is_relative_repo_path(value):
        _fail(f"{context} must be a repository-relative path (absolute paths are forbidden): {value!r}")
    return repo_root / Path(value)


def _require_keys(item: dict[str, Any], required: set[str], allowed: set[str], context: str) -> None:
    missing = sorted(required - item.keys())
    unknown = sorted(item.keys() - allowed)
    if missing:
        _fail(f"{context} missing required fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _validate_position(value: Any, context: str) -> None:
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        _fail(f"{context} has malformed position")
    if not all(isinstance(value[axis], (int, float)) and not isinstance(value[axis], bool) and math.isfinite(value[axis]) for axis in ("x", "y")):
        _fail(f"{context} has malformed position")


def _validate_public_block(value: Any, context: str, repo_root: Path) -> None:
    if not isinstance(value, dict):
        _fail(f"{context}.public must be an object")
    _require_keys(value, {"label", "summary", "maturity", "mode", "repo_paths"}, PUBLIC_FIELDS, f"{context}.public")
    if not all(isinstance(value[key], str) and value[key].strip() for key in ("label", "summary")):
        _fail(f"{context}.public label and summary must be non-empty strings")
    if value["maturity"] not in MATURITY_VALUES:
        _fail(f"{context}.public has unknown maturity {value['maturity']!r}")
    if value["mode"] not in MODE_VALUES:
        _fail(f"{context}.public has unknown mode {value['mode']!r}")
    _validate_path_list(value["repo_paths"], f"{context}.public.repo_paths", repo_root)
    for optional in ("source_links", "tags"):
        if optional in value and (not isinstance(value[optional], list) or not all(isinstance(part, str) for part in value[optional])):
            _fail(f"{context}.public.{optional} must be a list of strings")


def _validate_path_list(value: Any, context: str, repo_root: Path, *, optional: bool = False, warnings: list[str] | None = None, require_nonempty: bool = False) -> None:
    if not isinstance(value, list) or (require_nonempty and not value):
        noun = "a non-empty list" if require_nonempty else "a list"
        _fail(f"{context} must be {noun} of paths")
    for declared in value:
        if isinstance(declared, dict):
            path_text = declared.get("path")
            is_optional = bool(declared.get("optional"))
        else:
            path_text, is_optional = declared, optional
        resolved = _repo_path(repo_root, path_text, context)
        if not resolved.exists():
            if is_optional:
                if warnings is not None:
                    warnings.append(f"optional path unresolved: {path_text}")
            else:
                _fail(f"{context} required path does not exist: {path_text}")


def _symbol_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(item.name)
            if isinstance(item, ast.ClassDef):
                for child in item.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(f"{item.name}.{child.name}")
    return names


def _optional_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and bool(entry.get("optional"))


def _warn_or_fail(optional: bool, warnings: list[str], message: str) -> None:
    if optional:
        warnings.append(message)
    else:
        _fail(message)


def _read_python(repo_root: Path, path_text: Any, context: str) -> ast.Module:
    path = _repo_path(repo_root, path_text, context)
    if not path.is_file():
        _fail(f"{context} required path does not exist: {path_text}")
    if path.suffix != ".py":
        _fail(f"{context} must point to a Python file: {path_text}")
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        _fail(f"{context} could not be parsed: {exc}")


def _validate_symbols(repo_root: Path, entries: Any, warnings: list[str], context: str) -> None:
    if not isinstance(entries, list):
        _fail(f"{context}.symbols must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("name"), str):
            _fail(f"{context}.symbols entries require path and name")
        if entry.get("kind") is not None and entry["kind"] not in {"function", "class"}:
            _fail(f"{context}.symbols kind must be function or class")
        tree = _read_python(repo_root, entry["path"], f"{context}.symbols")
        matching_kinds = {
            item.name: "class" if isinstance(item, ast.ClassDef) else "function"
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if entry["name"] not in _symbol_names(tree) or (entry.get("kind") is not None and matching_kinds.get(entry["name"]) != entry["kind"]):
            _warn_or_fail(_optional_entry(entry), warnings, f"symbol unresolved: {entry['path']}:{entry['name']}")


def _validate_imports(repo_root: Path, entries: Any, warnings: list[str], context: str) -> None:
    if not isinstance(entries, list):
        _fail(f"{context}.imports must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("module"), str):
            _fail(f"{context}.imports entries require path and module")
        expected_name = entry.get("name")
        if expected_name is not None and not isinstance(expected_name, str):
            _fail(f"{context}.imports name must be a string")
        tree = _read_python(repo_root, entry["path"], f"{context}.imports")
        matched = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                matched = any(alias.name == entry["module"] and (expected_name is None or alias.asname == expected_name or alias.name.rsplit(".", 1)[-1] == expected_name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module == entry["module"]:
                matched = expected_name is None or any(alias.name == expected_name for alias in node.names)
            if matched:
                break
        if not matched:
            _warn_or_fail(_optional_entry(entry), warnings, f"import unresolved: {entry['path']} -> {entry['module']}")


def _validate_text(repo_root: Path, entries: Any, warnings: list[str], context: str) -> None:
    if not isinstance(entries, list):
        _fail(f"{context}.text must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("pattern"), str):
            _fail(f"{context}.text entries require path and pattern")
        pattern = entry["pattern"]
        if not pattern or len(pattern) > 512:
            _fail(f"{context}.text pattern must be between 1 and 512 characters")
        path = _repo_path(repo_root, entry["path"], f"{context}.text")
        if not path.is_file():
            _warn_or_fail(_optional_entry(entry), warnings, f"text evidence unresolved: {entry['path']}")
            continue
        try:
            found = pattern in path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _fail(f"{context}.text could not be read: {exc}")
        if not found:
            _warn_or_fail(_optional_entry(entry), warnings, f"text evidence unresolved: {entry['path']}:{pattern}")


def _validate_evidence(repo_root: Path, evidence: Any, warnings: list[str], context: str) -> None:
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        _fail(f"{context}.evidence must be an object")
    allowed = {"required_paths", "optional_paths", "symbols", "imports", "text", "text_patterns", "dynamic_dependencies"}
    unknown = sorted(evidence.keys() - allowed)
    if unknown:
        _fail(f"{context}.evidence has unknown fields: {', '.join(unknown)}")
    if "required_paths" in evidence:
        _validate_path_list(evidence["required_paths"], f"{context}.evidence.required_paths", repo_root, require_nonempty=True)
    if "optional_paths" in evidence:
        _validate_path_list(evidence["optional_paths"], f"{context}.evidence.optional_paths", repo_root, optional=True, warnings=warnings)
    if "symbols" in evidence:
        _validate_symbols(repo_root, evidence["symbols"], warnings, context)
    if "imports" in evidence:
        _validate_imports(repo_root, evidence["imports"], warnings, context)
    for key in ("text", "text_patterns"):
        if key in evidence:
            _validate_text(repo_root, evidence[key], warnings, context)
    if "dynamic_dependencies" in evidence:
        if not isinstance(evidence["dynamic_dependencies"], list) or not all(isinstance(value, str) and value for value in evidence["dynamic_dependencies"]):
            _fail(f"{context}.evidence.dynamic_dependencies must be a list of strings")
        warnings.extend(f"dynamic dependency not statically proven: {value}" for value in evidence["dynamic_dependencies"])


def _validate_manifest(manifest: Any, repo_root: Path) -> list[str]:
    if not isinstance(manifest, dict):
        _fail("manifest must be a JSON object")
    _require_keys(manifest, {"schema_version", "nodes", "edges"}, {"schema_version", "nodes", "edges"}, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        _fail(f"unsupported schema_version {manifest['schema_version']!r}; expected {SCHEMA_VERSION}")
    if not isinstance(manifest["nodes"], list) or not manifest["nodes"]:
        _fail("manifest.nodes must be a non-empty list")
    if not isinstance(manifest["edges"], list):
        _fail("manifest.edges must be a list")

    warnings: list[str] = []
    node_ids: set[str] = set()
    for node in manifest["nodes"]:
        if not isinstance(node, dict):
            _fail("node must be an object")
        _require_keys(node, {"id", "kind", "visibility", "edge_color_role", "position", "public"}, NODE_FIELDS, f"node {node.get('id', '<unknown>')}")
        node_id = node["id"]
        if not isinstance(node_id, str) or not node_id:
            _fail("node id must be a non-empty string")
        if node_id in node_ids:
            _fail(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        if node["kind"] not in NODE_KINDS:
            _fail(f"node {node_id} has unknown kind {node['kind']!r}")
        if node["visibility"] not in VISIBILITIES:
            _fail(f"node {node_id} has unknown visibility {node['visibility']!r}")
        if node["edge_color_role"] not in NODE_COLOR_ROLES:
            _fail(f"node {node_id} has unknown edge_color_role {node['edge_color_role']!r}")
        if "parent_id" in node and node["parent_id"] is not None and not isinstance(node["parent_id"], str):
            _fail(f"node {node_id} parent_id must be a string or null")
        _validate_position(node["position"], f"node {node_id}")
        _validate_public_block(node["public"], f"node {node_id}", repo_root)
        if "local" in node and not isinstance(node["local"], dict):
            _fail(f"node {node_id}.local must be an object")
        if node["visibility"] == "local" and "local" not in node:
            _fail(f"local node {node_id} requires a local detail block")
        if "local" in node and "repo_paths" in node["local"]:
            _validate_path_list(node["local"]["repo_paths"], f"node {node_id}.local.repo_paths", repo_root)
        _validate_evidence(repo_root, node.get("evidence"), warnings, f"node {node_id}")

    roots = [node["id"] for node in manifest["nodes"] if node.get("parent_id") is None]
    if len(roots) != 1:
        _fail("manifest must contain exactly one root node")
    for node in manifest["nodes"]:
        parent = node.get("parent_id")
        if parent is not None and parent not in node_ids:
            _fail(f"node {node['id']} has missing parent {parent}")
        if parent is not None and node["visibility"] == "public":
            parent_node = next(candidate for candidate in manifest["nodes"] if candidate["id"] == parent)
            if parent_node["visibility"] != "public":
                _fail(f"public node {node['id']} cannot have a local-only parent")

    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for node in manifest["nodes"]:
        if parent := node.get("parent_id"):
            children[parent].append(node["id"])
    seen: set[str] = set()
    visiting: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visiting:
            _fail(f"parent graph contains a cycle at {node_id}")
        if node_id in seen:
            return
        visiting.add(node_id)
        for child in children[node_id]:
            walk(child)
        visiting.remove(node_id)
        seen.add(node_id)

    # Detect a disconnected parent-cycle before reporting it merely as an
    # orphaned branch; both diagnoses are useful, but a cycle is the root cause.
    for node_id in sorted(node_ids):
        walk(node_id)
    reachable: set[str] = set()

    def mark_reachable(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)
        for child in children[node_id]:
            mark_reachable(child)

    mark_reachable(roots[0])
    if reachable != node_ids:
        _fail(f"unreachable nodes: {', '.join(sorted(node_ids - reachable))}")

    edge_ids: set[str] = set()
    for edge in manifest["edges"]:
        if not isinstance(edge, dict):
            _fail("edge must be an object")
        _require_keys(edge, {"id", "source", "target", "type", "visibility"}, EDGE_FIELDS, f"edge {edge.get('id', '<unknown>')}")
        edge_id = edge["id"]
        if not isinstance(edge_id, str) or not edge_id:
            _fail("edge id must be a non-empty string")
        if edge_id in edge_ids:
            _fail(f"duplicate edge id: {edge_id}")
        edge_ids.add(edge_id)
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            _fail(f"edge {edge_id} has unknown endpoint")
        if edge["type"] not in EDGE_TYPES:
            _fail(f"edge {edge_id} has unknown edge type {edge['type']!r}")
        if edge["visibility"] not in VISIBILITIES:
            _fail(f"edge {edge_id} has unknown visibility {edge['visibility']!r}")
        _validate_evidence(repo_root, edge.get("evidence"), warnings, f"edge {edge_id}")
    return warnings


def _sorted_dataset(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "nodes": sorted(nodes, key=lambda node: node["id"]),
        "edges": sorted(edges, key=lambda edge: edge["id"]),
    }


def _public_dataset(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes = []
    public_ids = {node["id"] for node in manifest["nodes"] if node["visibility"] == "public"}
    for source in manifest["nodes"]:
        if source["id"] not in public_ids:
            continue
        node = {key: copy.deepcopy(source[key]) for key in ("id", "kind", "visibility", "edge_color_role", "position", "public")}
        if "parent_id" in source and source["parent_id"] in public_ids:
            node["parent_id"] = source["parent_id"]
        nodes.append(node)
    edges = []
    for source in manifest["edges"]:
        if source["visibility"] != "public" or source["source"] not in public_ids or source["target"] not in public_ids:
            continue
        edge = {key: copy.deepcopy(source[key]) for key in ("id", "source", "target", "type", "visibility")}
        if "public" in source:
            edge["public"] = copy.deepcopy(source["public"])
        if "layout" in source:
            edge["layout"] = copy.deepcopy(source["layout"])
        edges.append(edge)
    return _sorted_dataset(nodes, edges)


def _local_dataset(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return _sorted_dataset(copy.deepcopy(manifest["nodes"]), copy.deepcopy(manifest["edges"]))


def _validate_safe_output(value: Any, *, public: bool) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if public and key == "local":
                _fail("public output contains a local field")
            _validate_safe_output(child, public=public)
    elif isinstance(value, list):
        for child in value:
            _validate_safe_output(child, public=public)
    elif isinstance(value, str):
        if _CREDENTIAL_ASSIGNMENT.search(value):
            _fail("credential-like assignment found in Atlas output")
        if _WINDOWS_PATH.search(value) or _POSIX_PATH.search(value):
            _fail("absolute workstation path found in Atlas output")


def _revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _inline_data(index: Path, payload: dict[str, Any]) -> None:
    html = index.read_text(encoding="utf-8")
    script = "<script>window.ATLAS_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) .replace("</", "<\\/") + ";</script>"
    marker = "/*__ATLAS_DATA__*/"
    wrapped_marker = f"<script>{marker}</script>"
    if wrapped_marker in html:
        index.write_text(html.replace(wrapped_marker, script, 1), encoding="utf-8")
        return
    if marker in html:
        index.write_text(html.replace(marker, script, 1), encoding="utf-8")
        return
    if "</head>" in html:
        index.write_text(html.replace("</head>", script + "</head>", 1), encoding="utf-8")
        return
    _fail(f"static index has no {marker} marker or closing head tag")


def _validate_static_inventory(static_dir: Path) -> None:
    missing = sorted(name for name in _ALLOWED_STATIC_FILES if not (static_dir / name).is_file())
    if missing:
        _fail(f"static source is missing required application assets: {', '.join(missing)}")
    for path in static_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(static_dir).as_posix()
        if relative in _ALLOWED_STATIC_FILES:
            continue
        if re.fullmatch(r"vendor/(?:cytoscape(?:\.min)?\.js|LICENSE-[A-Za-z0-9_.-]+)", relative):
            continue
        if re.fullmatch(r"fonts/(?:[A-Za-z0-9_.-]+\.woff2|LICENSE-[A-Za-z0-9_.-]+)", relative):
            continue
        _fail(f"static source contains non-allowlisted output file: {relative}")


def _write_artifact(static_dir: Path, target: Path, payload: dict[str, Any]) -> None:
    shutil.copytree(static_dir, target)
    index = target / "index.html"
    if not index.is_file():
        _fail("static directory must contain index.html")
    _inline_data(index, payload)


def _replace_output(temp_output: Path, output_dir: Path) -> None:
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.with_name(f".{output_dir.name}.previous-{uuid4().hex}")
        os.replace(output_dir, backup)
    try:
        os.replace(temp_output, output_dir)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_atlas(
    *,
    repo_root: Path | str,
    manifest_path: Path | str | None = None,
    static_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    build_time: str | None = None,
    git_revision: str | None = None,
) -> dict[str, Any]:
    """Build both safe static artifacts, replacing ``dist`` only on success."""
    root = Path(repo_root).resolve()
    atlas_root = root / "UI" / "architecture_atlas"
    manifest_file = Path(manifest_path) if manifest_path is not None else atlas_root / "source" / "architecture.json"
    static_source = Path(static_dir) if static_dir is not None else atlas_root / "static"
    destination = Path(output_dir) if output_dir is not None else atlas_root / "dist"
    if not manifest_file.is_file():
        _fail(f"source manifest does not exist: {manifest_file}")
    if not static_source.is_dir():
        _fail(f"static source directory does not exist: {static_source}")
    _validate_static_inventory(static_source)
    raw_manifest = manifest_file.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON manifest: {exc}")
    warnings = _validate_manifest(manifest, root)
    public_data = _public_dataset(manifest)
    local_data = _local_dataset(manifest)
    _validate_safe_output(public_data, public=True)
    _validate_safe_output(local_data, public=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "build_time": build_time or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_revision": git_revision if git_revision is not None else _revision(root),
        "source_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "validation_counts": {"nodes": len(manifest["nodes"]), "edges": len(manifest["edges"]), "warnings": len(warnings)},
        "warnings": sorted(warnings),
    }
    public_payload = {"schema_version": SCHEMA_VERSION, "metadata": metadata, "datasets": {"public": public_data}}
    local_payload = {"schema_version": SCHEMA_VERSION, "metadata": metadata, "datasets": {"public": public_data, "local": local_data}}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=".architecture-atlas-", dir=destination.parent))
    temp_output = temp_parent / destination.name
    try:
        _write_artifact(static_source, temp_output / "public", public_payload)
        _write_artifact(static_source, temp_output / "local", local_payload)
        _replace_output(temp_output, destination)
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)
    return metadata


def check_atlas(
    *,
    repo_root: Path | str,
    manifest_path: Path | str | None = None,
    static_dir: Path | str | None = None,
    git_revision: str | None = None,
) -> dict[str, Any]:
    """Validate the manifest and static inputs without writing build output."""
    root = Path(repo_root).resolve()
    atlas_root = root / "UI" / "architecture_atlas"
    manifest_file = Path(manifest_path) if manifest_path is not None else atlas_root / "source" / "architecture.json"
    static_source = Path(static_dir) if static_dir is not None else atlas_root / "static"
    if not manifest_file.is_file():
        _fail(f"source manifest does not exist: {manifest_file}")
    if not static_source.is_dir():
        _fail(f"static source directory does not exist: {static_source}")
    _validate_static_inventory(static_source)
    raw_manifest = manifest_file.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON manifest: {exc}")
    warnings = _validate_manifest(manifest, root)
    _validate_safe_output(_public_dataset(manifest), public=True)
    _validate_safe_output(_local_dataset(manifest), public=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_revision": git_revision if git_revision is not None else _revision(root),
        "source_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "validation_counts": {
            "nodes": len(manifest["nodes"]),
            "edges": len(manifest["edges"]),
            "warnings": len(warnings),
        },
        "warnings": sorted(warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--static-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="validate inputs without writing dist output")
    args = parser.parse_args()
    if args.check and args.output_dir is not None:
        parser.error("--output-dir cannot be used with --check")
    if args.check:
        metadata = check_atlas(
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            static_dir=args.static_dir,
        )
    else:
        metadata = build_atlas(
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            static_dir=args.static_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
