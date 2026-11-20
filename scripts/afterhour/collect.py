"""Collect raw AfterHour public posts for a callout-accuracy study.

Subcommands (all read-only GETs, single-threaded, resumable per unit):
  users    full post history per username  -> <out>/users/<username>.jsonl (+ .profile.json)
  scan     global newest-first feed back N days -> <out>/feed_scan/items.jsonl, authors.csv
  details  full post body + comments for selected posts -> <out>/details/<username>/<hash>.json

Raw API items are stored exactly as returned (one JSON object per line). Each
unit is written atomically and skipped on rerun, so a crash loses one unit only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from scripts.afterhour.client import AfterHourClient

PAGE = 50
MIN_PAGE = 10


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028/U+0085 inside post text
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def post_hash(share_url: str) -> str | None:
    """`https://afterhour.com/<user>/<hash>/<slug>` -> `<hash>`."""
    parts = [p for p in urlparse(share_url or "").path.split("/") if p]
    return parts[1] if len(parts) >= 2 else None


def created_at(item: dict) -> datetime:
    raw = (item.get("post") or {}).get("createdAt") or item.get("createdAt")
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def iter_feed(client: AfterHourClient, **filters: Any) -> Iterator[dict]:
    """Yield raw feed pages, shrinking the page size if the gateway keeps timing out."""
    cursor, take = None, PAGE
    while True:
        try:
            page = client.feed_page(take, cursor, **filters)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500 and take > MIN_PAGE:
                take = max(take // 2, MIN_PAGE)
                continue
            raise
        items = page.get("items") or []
        if not items:
            return
        yield page
        nxt = page.get("cursor")
        if nxt is None or nxt == cursor:
            return
        cursor = nxt


def collect_user(client: AfterHourClient, username: str, out: Path) -> dict:
    dest = out / "users" / f"{username}.jsonl"
    if dest.exists():
        return {"username": username, "status": "skipped_existing"}
    profile = client.profile(username)
    pid = profile.get("id")
    if not str(pid).startswith("prf_"):
        raise LookupError(f"no profile id for {username!r}: {str(profile)[:200]}")
    items, seen, total = [], set(), None
    for page in iter_feed(client, authorId=pid):
        total = page.get("totalCount") if total is None else total
        for it in page["items"]:
            if it.get("id") not in seen:
                seen.add(it.get("id"))
                items.append(it)
        print(f"  {username}: {len(items)}/{total}", file=sys.stderr, flush=True)
    profile["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    profile["_feed_total_count"] = total
    atomic_write_text(out / "users" / f"{username}.profile.json", json.dumps(profile, indent=1))
    write_jsonl(dest, items)
    return {"username": username, "status": "ok", "items": len(items), "total_count": total}


def scan_feed(client: AfterHourClient, days: int, out: Path) -> dict:
    dest = out / "feed_scan" / "items.jsonl"
    if dest.exists():
        return {"status": "skipped_existing"}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items, seen = [], set()
    for page in iter_feed(client):
        for it in page["items"]:
            if it.get("id") not in seen:
                seen.add(it.get("id"))
                items.append(it)
        oldest = min(created_at(i) for i in page["items"])
        print(f"  scan: {len(items)} items, oldest {oldest:%Y-%m-%d %H:%M}", file=sys.stderr, flush=True)
        if oldest < cutoff:
            break
    write_jsonl(dest, items)
    authors: dict[str, dict] = {}
    for it in items:
        a = it.get("author") or {}
        row = authors.setdefault(a.get("id"), {"author_id": a.get("id"), "username": a.get("username"),
                                               "num_followers": a.get("numFollowers"), "posts_in_window": 0,
                                               "privacy": a.get("financialDataPrivacyMode")})
        row["posts_in_window"] += 1
    rows = sorted(authors.values(), key=lambda r: -(r["num_followers"] or 0))
    path = out / "feed_scan" / "authors.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"status": "ok", "items": len(items), "authors": len(rows), "cutoff_utc": cutoff.isoformat()}


def collect_details(client: AfterHourClient, selection: Path, out: Path) -> dict:
    counts = {"ok": 0, "skipped_existing": 0, "missing": 0, "error": 0}
    for row in read_jsonl(selection):
        dest = out / "details" / row["username"] / f"{row['hash']}.json"
        if dest.exists():
            counts["skipped_existing"] += 1
            continue
        try:
            detail = client.post(row["hash"])
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            counts["missing" if code == 404 else "error"] += 1
            detail = {"_error_status": code}
        else:
            counts["ok"] += 1
        detail["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_text(dest, json.dumps(detail, ensure_ascii=False))
        done = sum(counts.values())
        if done % 50 == 0:
            print(f"  details: {done} {counts}", file=sys.stderr, flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("users")
    u.add_argument("usernames", nargs="+")
    s = sub.add_parser("scan")
    s.add_argument("--days", type=int, default=90)
    d = sub.add_parser("details")
    d.add_argument("--selection", type=Path, required=True, help="jsonl rows with username, hash")
    args = ap.parse_args()
    client = AfterHourClient()
    if args.cmd == "users":
        results = []
        for name in args.usernames:
            try:
                results.append(collect_user(client, name, args.out))
            except Exception as exc:  # one bad username must not discard the others
                results.append({"username": name, "status": "error", "error": str(exc)[:300]})
            print(json.dumps(results[-1]), flush=True)
    elif args.cmd == "scan":
        print(json.dumps(scan_feed(client, args.days, args.out)), flush=True)
    else:
        print(json.dumps(collect_details(client, args.selection, args.out)), flush=True)
    print(json.dumps({"requests_made": client.requests_made}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
