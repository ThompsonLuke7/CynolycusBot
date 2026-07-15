"""
Meta Ranker 4H loop (step 4) — one pass, designed to be scheduled every 4H bar.

Pipeline per bar:
  1. catch up shared bars (REST, whole universe — no websocket; 200/min tier is fine on 4H)
  2. refresh upstream feeds (news/calendars/guidance/treasury; themes only when --weekly-due)
  3. append new bars to the rolling matrix (live base-model scoring, not OOF)
  4. run the runner (equity or options; dry-run unless --submit)

Run ONE pass (schedule this via cron/systemd timer at ~5 min after each 4H close):
  PYTHONPATH=. python signals/meta_context/meta_ranker/run_4h_loop.py --mode equity
  PYTHONPATH=. python signals/meta_context/meta_ranker/run_4h_loop.py --mode options --submit

Themes: pass --weekly-due (e.g. from a Sunday cron) to also refresh the theme taxonomy.
A step failing is logged but does not abort later steps (the runner's own staleness
guard is the backstop against trading on stale data).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from core.live_job_guard import heavy_job_guard

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = sys.executable


def _step(label: str, argv: list[str], timeout: int) -> bool:
    print(f"\n########## {label} ##########\n  $ {' '.join(argv)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    t0 = time.time()
    try:
        rc = subprocess.run(argv, cwd=str(REPO), env=env, timeout=timeout).returncode
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {label}: {type(exc).__name__}: {exc}")
        rc = 1
    print(f"  {label}: {'OK' if rc == 0 else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["equity", "options"], default="equity")
    ap.add_argument("--submit", action="store_true", help="Place orders (default dry-run).")
    ap.add_argument("--live", action="store_true", help="Live account (default paper).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--weekly-due", action="store_true", help="Also refresh dynamic themes (Claude $).")
    ap.add_argument("--skip-bars", action="store_true")
    ap.add_argument("--skip-feeds", action="store_true")
    ap.add_argument("--skip-matrix", action="store_true",
                    help="Skip the matrix rebuild. With --skip-bars --skip-feeds this makes the "
                         "pass runner-only (~seconds) — use it when a background refresher keeps "
                         "the shared bars + matrix fresh out-of-band.")
    ap.add_argument("--no-1d", action="store_true",
                    help="Skip the per-ticker daily-bar fetch in the catch-up (halves REST calls; "
                         "daily bars only matter at EOD, refreshed by the nightly readiness job).")
    args = ap.parse_args()

    base = HERE
    t0 = time.time()
    print(f"=== Meta Ranker 4H loop pass | mode={args.mode} submit={args.submit} live={args.live} ===")

    needs_heavy = not (args.skip_bars and args.skip_feeds and args.skip_matrix)
    if needs_heavy:
        with heavy_job_guard(
            "meta-ranker-4h-loop",
            block_live_window=True,
            min_available_mb=4096,
            min_swap_free_mb=4096,
        ) as guard:
            if not guard.ok:
                print(f"  ! heavy refresh skipped: {guard.reason}", flush=True)
            else:
                if not args.skip_bars:
                    # Same-day pre-close refresh: scope to the tradeable (is_eligible) set so it
                    # finishes before the MOC entry instead of crawling the full 3k universe and
                    # colliding with the live session. The nightly readiness job backfills the rest.
                    bars_argv = [PY, "scripts/catchup_shared_bars.py", "--workers", "6", "--eligible-only"]
                    if args.no_1d:
                        bars_argv.append("--no-1d")
                    _step("1/4 bars", bars_argv, timeout=2400)
                if not args.skip_feeds:
                    feed_argv = [PY, str(base / "update_feeds.py")] + (["--weekly"] if args.weekly_due else [])
                    _step("2/4 feeds", feed_argv, timeout=3600)
                if not args.skip_matrix:
                    _step("3/4 matrix", [PY, str(base / "update_meta_matrix.py")], timeout=2400)

    run_argv = [PY, str(base / "live_runner.py"), "--mode", args.mode, "--top-k", str(args.top_k)]
    if args.submit:
        run_argv.append("--submit")
    if args.live:
        run_argv.append("--live")
    ok = _step("4/4 runner", run_argv, timeout=900)

    print(f"\n=== loop pass done in {time.time()-t0:.0f}s | runner {'OK' if ok else 'FAILED'} ===")


if __name__ == "__main__":
    main()
