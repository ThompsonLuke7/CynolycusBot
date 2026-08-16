"""Repair NUL-byte holes in append-only JSONL artefacts, preserving the audit trail.

An unclean host kill (this box has a known WSL2 crash history) can leave ext4
delayed-allocation holes: the file's new SIZE is journaled but the data blocks
are not, so recovered records read back as runs of \\x00. Found 2026-08-03 in
four live artefacts, all dating to crashes on 2026-07-21/24/28.

This script never invents data. For each corrupted physical line it:
  * keeps any complete JSON record that survived alongside the NUL run,
  * replaces the zeroed bytes with an explicit `audit_gap` record stating how
    many bytes were lost and roughly how many records that represents (from the
    file's own median record length),
  * writes a byte-for-byte backup first, and
  * refuses to touch the file at all if a surviving fragment does not parse.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/repair_nul_corrupted_jsonl.py --dry-run <paths...>
  PYTHONPATH=. .venv/bin/python scripts/repair_nul_corrupted_jsonl.py <paths...>
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def analyze(path: Path) -> dict:
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    trailing_newline = bool(lines) and lines[-1] == b""
    if trailing_newline:
        lines = lines[:-1]
    clean_lens = [len(l) for l in lines if b"\x00" not in l and l.strip()]
    median = sorted(clean_lens)[len(clean_lens) // 2] if clean_lens else 0
    corrupt = [(i, l) for i, l in enumerate(lines, 1) if b"\x00" in l]
    return {"raw": raw, "lines": lines, "median": median, "corrupt": corrupt,
            "trailing_newline": trailing_newline, "total_nul": raw.count(b"\x00")}


def repair(path: Path, *, dry_run: bool = False) -> dict:
    info = analyze(path)
    if not info["corrupt"]:
        return {"path": str(path), "status": "clean"}

    median = info["median"]
    out: list[bytes] = []
    gaps: list[dict] = []
    for idx, line in enumerate(info["lines"], 1):
        if b"\x00" not in line:
            out.append(line)
            continue
        nuls = line.count(b"\x00")
        surviving = line.replace(b"\x00", b"")
        if surviving.strip():
            # Refuse to guess: a partial fragment is not a record.
            json.loads(surviving.decode("utf-8"))
        # +1 for the newline that separated each lost record.
        est_lost = max(1, round(nuls / (median + 1))) if median else None
        gap = {
            "event": "audit_gap",
            "reason": "nul_block_from_unclean_shutdown",
            "physical_line_in_corrupt_file": idx,
            "nul_bytes": nuls,
            "median_record_bytes": median or None,
            "estimated_lost_records": est_lost,
            "surviving_fragment_recovered": bool(surviving.strip()),
            "detail": (
                f"{nuls} zeroed bytes at physical line {idx}. At this file's median record "
                f"length of {median} bytes, roughly {est_lost} record(s) were lost to an "
                f"unclean host shutdown and are unrecoverable."
            ),
            "repaired_at": datetime.now(timezone.utc).isoformat(),
        }
        gaps.append(gap)
        out.append(json.dumps(gap, separators=(",", ":")).encode("utf-8"))
        if surviving.strip():
            out.append(surviving)

    result = {"path": str(path), "status": "would_repair" if dry_run else "repaired",
              "corrupt_lines": len(info["corrupt"]), "total_nul": info["total_nul"],
              "gaps": gaps}
    if dry_run:
        return result

    backup = path.parent / "_corrupt_backup" / f"{path.name}.nul_corrupt_{datetime.now():%Y%m%d}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    result["backup"] = str(backup)

    path.write_bytes(b"\n".join(out) + (b"\n" if info["trailing_newline"] else b""))

    verify = path.read_bytes()
    assert verify.count(b"\x00") == 0, f"{path}: NUL bytes survived the repair"
    for l in verify.split(b"\n"):
        if l.strip():
            json.loads(l)
    result["verified_lines"] = sum(1 for l in verify.split(b"\n") if l.strip())
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for p in args.paths:
        res = repair(Path(p), dry_run=args.dry_run)
        print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
