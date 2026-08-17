"""Remove ephemeral ``cluster_<id>`` labels from the dynamic theme registry.

``step05_claude_labeling._label_cluster`` falls back to ``cluster_<id>`` with
confidence 0.0 whenever the labeling call raises, and that placeholder was
persisted into ``theme_registry.parquet`` as if it were a durable theme name.
``step08_memberships.canonical_theme_id`` — which reached main in the 2026-08-16
merge — correctly refuses such a label, so every accumulated placeholder now
aborts the weekly theme run at the memberships step.

Cluster ids are ephemeral: cluster 169 in June and cluster 169 today are
unrelated groupings, so these rows cannot be repaired into real identities.
They carry no information either (confidence 0.0, empty description). This
script moves them out of the registry rather than deleting them.

Dry-run by default; pass --apply to write.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path("themes/dynamic_theme/outputs/theme_registry.parquet")
EPHEMERAL_RE = re.compile(r"^cluster_-?\d+$", re.IGNORECASE)


def is_ephemeral(label: object) -> bool:
    """Mirror the step08 guard: a bare ``cluster_<id>`` is not a theme identity."""
    return bool(EPHEMERAL_RE.fullmatch(str(label).strip()))


def select_rows(registry: pd.DataFrame) -> pd.Series:
    """Rows to quarantine.

    Deliberately narrow: an ephemeral name alone is the criterion. A row that
    somehow carries a real description or a non-zero confidence is reported but
    left alone, since that would mean the placeholder assumption does not hold.
    """
    ephemeral = registry["theme_name"].map(is_ephemeral)
    informative = (
        registry["description"].astype(str).str.strip().str.len().gt(0)
        | registry["confidence"].fillna(0.0).ne(0.0)
    )
    return ephemeral & ~informative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()

    registry_path: Path = args.registry
    if not registry_path.exists():
        print(f"registry not found: {registry_path}")
        return 1

    registry = pd.read_parquet(registry_path)
    mask = select_rows(registry)
    doomed = registry[mask]
    kept = registry[~mask]

    ephemeral_total = int(registry["theme_name"].map(is_ephemeral).sum())
    print(f"registry rows:            {len(registry):,}")
    print(f"ephemeral labels:         {ephemeral_total}")
    print(f"selected for quarantine:  {len(doomed)}")
    if ephemeral_total != len(doomed):
        print(
            f"  NOTE: {ephemeral_total - len(doomed)} ephemeral row(s) carry a "
            "description or confidence and were left in place for review"
        )
    if not doomed.empty:
        print(doomed.groupby(doomed["date"].astype(str)).size().to_string())

    if doomed.empty:
        print("nothing to do")
        return 0

    if not args.apply:
        print("\ndry run — pass --apply to write")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = registry_path.with_suffix(f".bak_{stamp}.parquet")
    shutil.copy2(registry_path, backup)

    quarantine = registry_path.with_name("theme_registry.quarantine.parquet")
    if quarantine.exists():
        doomed = pd.concat([pd.read_parquet(quarantine), doomed], ignore_index=True)
    doomed.to_parquet(quarantine, index=False)
    kept.to_parquet(registry_path, index=False)

    # Conservation: the backup must equal kept + removed, with nothing invented.
    restored = pd.read_parquet(backup)
    assert len(restored) == len(kept) + int(mask.sum()), "row conservation failed"
    assert not pd.read_parquet(registry_path)["theme_name"].map(is_ephemeral).any(), (
        "ephemeral labels survived the write"
    )

    print(f"\nbackup:      {backup}")
    print(f"quarantined: {quarantine} ({len(doomed):,} rows)")
    print(f"registry:    {registry_path} ({len(kept):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
