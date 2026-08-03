#!/usr/bin/env python3
"""Verify a local directory is fully backed up to GCS before deleting it.

Compares every local file under a path against the corresponding objects in the
bucket by name AND byte size. Exits non-zero if anything is missing or differs,
so it can gate a deletion:

    python3 scripts/verify_gcs_backup.py Data/options_history/trades && rm -rf ...

It also fails on files the Phase 2.4 upload deliberately EXCLUDED (`*.py`,
`__pycache__/`, the two verified duplicates) that are not tracked in git either.
The exclusion is only safe because "source belongs in git" — a gitignored .py
lives in exactly one place on earth, and deleting the directory destroys it.

Size equality is a necessary-but-not-sufficient check (it cannot detect a
same-length rewrite); re-run `gcloud storage rsync` first so the bucket is
current, then use this as the confirmation.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "gs://cynolycusbot-data"

# Local tree -> bucket prefix. Mirrors the Phase 2.4 upload manifest.
PREFIX_MAP = {
    "Data": "Data",
    "signals": "signals",
    "strategies": "strategies",
    "theme_expansion": "theme_expansion",
    "themes": "themes",
    "UI/swing_audit": "UI/swing_audit",
    "research": "research",
    "backtests": "backtests",
}

# Same exclusions the upload applied.
EXCLUDE = re.compile(r"(.*/)?__pycache__/.*|.*\.pyc?$")
EXTRA_EXCLUDE = {
    "Data": re.compile(r"runtime/live_data_jobs\.lock$|runtime/startup_queue\.json$"),
    "strategies": re.compile(
        r"momentum_expansion/data/training_export/training_matrix_4h\.parquet$"
        r"|momentum_expansion/data/training_export_ablation/ablation_colab_bundle\.tgz$"
    ),
}

LIST_LINE = re.compile(r"^\s*(\d+)\s+\S+\s+(gs://\S.*)$")


def bucket_prefix_for(rel_path: str) -> tuple[str, str]:
    """Return (tree_root, bucket_prefix) for a repo-relative path."""
    for tree in sorted(PREFIX_MAP, key=len, reverse=True):
        if rel_path == tree or rel_path.startswith(tree + "/"):
            suffix = rel_path[len(tree):].lstrip("/")
            prefix = PREFIX_MAP[tree]
            return tree, f"{prefix}/{suffix}".rstrip("/")
    raise SystemExit(
        f"'{rel_path}' is not inside any uploaded tree: {', '.join(sorted(PREFIX_MAP))}"
    )


def list_local(abs_dir: str, tree: str, tree_abs: str) -> tuple[dict[str, int], list[str]]:
    """Return (uploaded_files, excluded_files) keyed/relative to abs_dir."""
    extra = EXTRA_EXCLUDE.get(tree)
    out: dict[str, int] = {}
    excluded: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(abs_dir):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                continue
            rel_to_tree = os.path.relpath(full, tree_abs).replace(os.sep, "/")
            rel_to_dir = os.path.relpath(full, abs_dir).replace(os.sep, "/")
            if EXCLUDE.fullmatch(rel_to_tree) or (extra and extra.search(rel_to_tree)):
                excluded.append(rel_to_dir)
                continue
            out[rel_to_dir] = os.path.getsize(full)
    return out, excluded


def tracked_in_git(rel_dir: str, committed_only: bool = False) -> set[str]:
    """Paths (relative to rel_dir) that git knows about under it.

    committed_only=True asks the stronger question — is this file in HEAD? A
    staged-but-uncommitted file still exists only on this disk, so it is not a
    second copy for backup purposes.
    """
    if committed_only:
        cmd = ["git", "-C", REPO_ROOT, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", rel_dir]
    else:
        cmd = ["git", "-C", REPO_ROOT, "ls-files", "-z", "--", rel_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[2:4])} failed:\n{proc.stderr.strip()}")
    prefix = rel_dir.rstrip("/") + "/"
    return {
        p[len(prefix):]
        for p in proc.stdout.split("\0")
        if p.startswith(prefix)
    }


def list_gcs(prefix: str) -> dict[str, int]:
    url = f"{BUCKET}/{prefix}/"
    proc = subprocess.run(
        ["gcloud", "storage", "ls", "-r", "--long", url],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"gcloud storage ls failed for {url}:\n{proc.stderr.strip()}")
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        m = LIST_LINE.match(line)
        if not m:
            continue
        obj = m.group(2)
        if not obj.startswith(url):
            continue
        out[obj[len(url):]] = int(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="repo-relative directory, e.g. Data/options_history/trades")
    ap.add_argument("--show", type=int, default=20, help="max problem files to print")
    args = ap.parse_args()

    rel = args.path.rstrip("/").replace(os.sep, "/")
    abs_dir = os.path.join(REPO_ROOT, rel)
    if not os.path.isdir(abs_dir):
        raise SystemExit(f"not a directory: {abs_dir}")

    tree, prefix = bucket_prefix_for(rel)
    tree_abs = os.path.join(REPO_ROOT, tree)

    local, excluded = list_local(abs_dir, tree, tree_abs)
    remote = list_gcs(prefix)
    tracked = tracked_in_git(rel)                        # index: what rm -rf would destroy
    committed = tracked_in_git(rel, committed_only=True)  # HEAD: what actually has a 2nd copy

    missing = sorted(set(local) - set(remote))
    differing = sorted(p for p in set(local) & set(remote) if local[p] != remote[p])

    # Excluded from the upload AND absent from git == this file exists nowhere else.
    dup_re = EXTRA_EXCLUDE.get(tree)
    orphans, known_dups, staged_only = [], [], []
    for p in excluded:
        if p in committed or p.endswith(".pyc") or "/__pycache__/" in f"/{p}":
            continue
        if p in tracked:
            staged_only.append(p)
            continue
        rel_to_tree = os.path.relpath(os.path.join(abs_dir, p), tree_abs).replace(os.sep, "/")
        (known_dups if dup_re and dup_re.search(rel_to_tree) else orphans).append(p)

    n_tracked = sum(1 for p in tracked if os.path.exists(os.path.join(abs_dir, p)))

    print(f"local  {rel}/")
    print(f"       {len(local):>7} files  {sum(local.values()):>15,} bytes")
    print(f"bucket {BUCKET}/{prefix}/")
    print(f"       {len(remote):>7} files  {sum(remote.values()):>15,} bytes")
    print()

    if missing:
        print(f"MISSING from bucket: {len(missing)}")
        for p in missing[: args.show]:
            print(f"    {p}  ({local[p]:,} bytes)")
        if len(missing) > args.show:
            print(f"    ... and {len(missing) - args.show} more")
    if differing:
        print(f"SIZE DIFFERS: {len(differing)}")
        for p in differing[: args.show]:
            print(f"    {p}  local={local[p]:,}  bucket={remote[p]:,}")
        if len(differing) > args.show:
            print(f"    ... and {len(differing) - args.show} more")

    if known_dups:
        print(f"EXCLUDED as verified duplicates (confirm the twin copy is uploaded): {len(known_dups)}")
        for p in known_dups[: args.show]:
            print(f"    {p}")
    if orphans:
        print(f"NOT IN BUCKET AND NOT IN GIT: {len(orphans)}")
        for p in orphans[: args.show]:
            print(f"    {p}  ({os.path.getsize(os.path.join(abs_dir, p)):,} bytes)")
        if len(orphans) > args.show:
            print(f"    ... and {len(orphans) - args.show} more")

    if staged_only:
        print(f"STAGED BUT NOT COMMITTED: {len(staged_only)}")
        for p in staged_only[: args.show]:
            print(f"    {p}")

    if missing or differing:
        print("\nNOT SAFE TO DELETE — re-run `gcloud storage rsync` for this tree first.")
        return 1
    if staged_only:
        print(
            "\nNOT SAFE TO DELETE — the files above are excluded from the upload and only "
            "staged, so this disk is still their only copy. Commit and push first."
        )
        return 1
    if orphans:
        print(
            "\nNOT SAFE TO DELETE — the files above are gitignored and were excluded "
            "from the upload, so this directory is their only copy. Upload them "
            "explicitly or commit them first."
        )
        return 1

    print("All local files present in the bucket at matching size.")
    if n_tracked:
        print(
            f"\n⚠️  {n_tracked} files here are GIT-TRACKED. A plain `rm -rf` would delete them "
            f"from the working tree.\n"
            f"    Use:  git clean -nXd {rel}    # dry run, gitignored files only\n"
            f"          git clean -fXd {rel}    # then for real"
        )
    else:
        print(
            f"\nNo git-tracked files here — `rm -rf {rel}` would not touch the working tree.\n"
            f"    (Backup-safe only. This says nothing about whether live code still READS "
            f"this path.)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
