"""The refresher's rebuild/bootstrap branch must be reachable.

2026-09-13: `--rebuild` died with
`UnboundLocalError: cannot access local variable 'ga_predictor'`. The branch was
added above the code that builds the predictor, so it could never run -- and
neither could the bootstrap it shares, which is the only way to recover a matrix
that has gone missing. Nothing caught it, because no test exercised that branch:
the file already existed everywhere it was tried.

This checks the ordering statically, so the guard costs no model loading.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path


def _main_function() -> ast.FunctionDef:
    import scripts.update_live_meta_base_frame as refresher

    tree = ast.parse(Path(inspect.getfile(refresher)).read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _rebuild_branch(main: ast.FunctionDef) -> ast.If:
    for node in ast.walk(main):
        if isinstance(node, ast.If) and "rebuild" in ast.dump(node.test):
            return node
    raise AssertionError("no rebuild branch found in main()")


def test_names_the_rebuild_branch_reads_are_assigned_before_it():
    main = _main_function()
    branch = _rebuild_branch(main)

    assigned_before: set[str] = set()
    for node in ast.walk(main):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if getattr(node, "lineno", 0) >= branch.lineno:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    assigned_before.add(sub.id)

    assigned_anywhere: set[str] = set()
    for node in ast.walk(main):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        assigned_anywhere.add(sub.id)

    # Names the branch binds for itself are fine wherever they sit.
    assigned_in_branch: set[str] = set()
    for node in ast.walk(branch):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    assigned_in_branch.add(sub.id)

    read_in_branch = {
        node.id for node in ast.walk(branch)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    # Only locals defined outside the branch matter; globals and imports resolve
    # at module level, and branch-local names bind as the branch runs.
    locals_read = read_in_branch & (assigned_anywhere - assigned_in_branch)
    unbound = sorted(locals_read - assigned_before)

    assert not unbound, f"read before assignment in the rebuild branch: {unbound}"


def test_the_predictor_is_built_for_both_paths():
    """Hoisting it is what fixes the branch; duplicating it would drift."""
    main = _main_function()
    branch = _rebuild_branch(main)

    predictor_assignments = [
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ga_predictor" for t in node.targets)
    ]

    assert predictor_assignments, "expected ga_predictor to be assigned in main()"
    assert min(predictor_assignments) < branch.lineno, "ga_predictor must be built before the branch"
