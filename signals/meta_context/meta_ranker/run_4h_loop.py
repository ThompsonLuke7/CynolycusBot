"""
Meta Ranker 4H loop (step 4) — one pass, designed to be scheduled every 4H bar.

Pipeline per bar, in this order:
  1. catch up shared bars (REST, whole universe — no websocket; 200/min tier is fine on 4H)
  2. refresh upstream feeds (news/calendars/guidance/treasury; themes only when --weekly-due)
  3. append new bars to the rolling matrix (live base-model scoring, not OOF)
  4. run the runner (equity or options; dry-run unless --submit)

Run ONE pass (schedule this via cron/systemd timer at ~5 min after each 4H close):
  PYTHONPATH=. python signals/meta_context/meta_ranker/run_4h_loop.py --mode equity
  PYTHONPATH=. python signals/meta_context/meta_ranker/run_4h_loop.py --mode options --submit

Themes: pass --weekly-due (e.g. from a Sunday cron) to also refresh the theme taxonomy.

Fail-fast ordering: a failed or blocked required stage prevents every later
stage, and the runner does not execute. The previous behaviour logged failures
and ran the runner anyway, relying on the runner's own staleness guard as a
backstop; that let a pass trade on stale bars whenever the guard did not fire.
A skipped stage is only accepted with a freshness certificate.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from core.live_job_guard import heavy_job_guard
from core.nervous_system.orchestration.jobs import Stage, StageStatus, run_stages

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = sys.executable

# The matrix the runner scores from; the runner must not run if this is stale.
# This is the same file update_meta_matrix.py writes and live_runner.py reads
# (both resolve it as HERE / "meta_ranker_matrix.parquet").
MATRIX_PATH = HERE / "meta_ranker_matrix.parquet"
MAX_MATRIX_AGE_SEC = 6 * 3600


@dataclass
class LoopResult:
    """One pass, reported rather than printed."""

    stages: list
    submitted: bool
    runner_ok: bool
    blocked_reason: str | None = None

    @property
    def exit_code(self) -> int:
        return 0 if self.runner_ok else 1

    def stage(self, name: str):
        for item in self.stages:
            if item.name == name:
                return item
        return None


def _run_subprocess(label: str, argv: list[str], timeout: int) -> int:
    print(f"\n########## {label} ##########\n  $ {' '.join(argv)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    t0 = time.time()
    try:
        rc = subprocess.run(argv, cwd=str(REPO), env=env, timeout=timeout).returncode
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {label}: {type(exc).__name__}: {exc}")
        rc = 1
    print(f"  {label}: {'OK' if rc == 0 else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
    return rc


def _matrix_is_fresh(now: float | None = None) -> tuple[bool, str, dict]:
    """Postcondition for the matrix stage.

    Exit zero is not evidence: update_meta_matrix.py can return zero for a
    no-op, so the file's age is checked directly.
    """

    if not MATRIX_PATH.exists():
        return False, f"{MATRIX_PATH} does not exist", {}
    age = (now if now is not None else time.time()) - MATRIX_PATH.stat().st_mtime
    counts = {"matrix_age_sec": int(age)}
    if age > MAX_MATRIX_AGE_SEC:
        return False, f"matrix is {int(age)}s old (limit {MAX_MATRIX_AGE_SEC}s)", counts
    return True, f"matrix is {int(age)}s old", counts


def build_stages(
    args: argparse.Namespace,
    *,
    runner: Callable[[], int],
    bars: Callable[[], int] | None = None,
    feeds: Callable[[], int] | None = None,
    matrix: Callable[[], int] | None = None,
    matrix_freshness: Callable[[], tuple[bool, str, dict]] = _matrix_is_fresh,
) -> list[Stage]:
    base = HERE

    def default_bars() -> int:
        argv = [PY, "scripts/catchup_shared_bars.py", "--workers", "6", "--eligible-only"]
        if args.no_1d:
            argv.append("--no-1d")
        return _run_subprocess("1/4 bars", argv, timeout=2400)

    def default_feeds() -> int:
        argv = [PY, str(base / "update_feeds.py")] + (["--weekly"] if args.weekly_due else [])
        return _run_subprocess("2/4 feeds", argv, timeout=3600)

    def default_matrix() -> int:
        return _run_subprocess("3/4 matrix", [PY, str(base / "update_meta_matrix.py")], timeout=2400)

    def certificate() -> tuple[bool, str]:
        fresh, detail, _counts = matrix_freshness()
        return fresh, detail

    return [
        Stage(
            name="bars",
            run=bars or default_bars,
            required=True,
            skip=args.skip_bars,
            skip_certificate=certificate,
        ),
        Stage(
            name="feeds",
            run=feeds or default_feeds,
            required=True,
            skip=args.skip_feeds,
            skip_certificate=certificate,
        ),
        Stage(
            name="matrix",
            run=matrix or default_matrix,
            verify=matrix_freshness,
            required=True,
            skip=args.skip_matrix,
            skip_certificate=certificate,
        ),
        Stage(name="runner", run=runner, required=True),
    ]


def run_once(
    args: argparse.Namespace,
    *,
    runner: Callable[[], int] | None = None,
    guard_factory: Callable[[], object] | None = None,
    **stage_overrides,
) -> LoopResult:
    """Execute one pass and return its result instead of exiting."""

    def default_runner() -> int:
        argv = [PY, str(HERE / "live_runner.py"), "--mode", args.mode, "--top-k", str(args.top_k)]
        if args.submit:
            argv.append("--submit")
        return _run_subprocess("4/4 runner", argv, timeout=900)

    needs_heavy = not (args.skip_bars and args.skip_feeds and args.skip_matrix)
    if needs_heavy:
        factory = guard_factory or (
            lambda: heavy_job_guard(
                "meta-ranker-4h-loop",
                block_live_window=True,
                min_available_mb=4096,
                min_swap_free_mb=4096,
            )
        )
        with factory() as guard:
            if not guard.ok:
                # A blocked guard used to fall through to the runner. It now
                # stops the pass: without the refresh the runner would score a
                # stale matrix.
                print(f"  ! heavy refresh blocked: {guard.reason}", flush=True)
                return LoopResult(
                    stages=[], submitted=False, runner_ok=False,
                    blocked_reason=f"guard blocked: {guard.reason}",
                )
            stages = build_stages(args, runner=runner or default_runner, **stage_overrides)
            results = run_stages(stages)
    else:
        stages = build_stages(args, runner=runner or default_runner, **stage_overrides)
        results = run_stages(stages)

    runner_result = next((item for item in results if item.name == "runner"), None)
    runner_ok = runner_result is not None and runner_result.status is StageStatus.OK
    for item in results:
        if item.status is not StageStatus.OK:
            print(f"  - {item.name}: {item.status.value} ({item.reason})", flush=True)
    return LoopResult(
        stages=list(results),
        submitted=bool(args.submit) and runner_ok,
        runner_ok=runner_ok,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["equity", "options"], default="equity")
    ap.add_argument("--submit", action="store_true", help="Place orders (default dry-run).")
    ap.add_argument(
        "--live",
        action="store_true",
        help=argparse.SUPPRESS,  # rejected; kept only to fail loudly
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--weekly-due", action="store_true", help="Also refresh dynamic themes (Claude $).")
    ap.add_argument("--skip-bars", action="store_true")
    ap.add_argument("--skip-feeds", action="store_true")
    ap.add_argument("--skip-matrix", action="store_true",
                    help="Skip the matrix rebuild. Only accepted when the existing matrix is "
                         "still certifiably fresh.")
    ap.add_argument("--no-1d", action="store_true",
                    help="Skip the per-ticker daily-bar fetch in the catch-up (halves REST calls; "
                         "daily bars only matter at EOD, refreshed by the nightly readiness job).")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.live:
        # The governed path is paper-only. A live flag here would bypass every
        # environment check the nervous system performs.
        print(
            "ERROR: --live is not accepted by the governed 4H loop; "
            "live trading is not enabled in this MVP.",
            file=sys.stderr,
        )
        return 2

    t0 = time.time()
    print(f"=== Meta Ranker 4H loop pass | mode={args.mode} submit={args.submit} ===")
    result = run_once(args)
    if result.blocked_reason:
        print(f"\n=== loop pass blocked: {result.blocked_reason} ===")
        return 1
    print(
        f"\n=== loop pass done in {time.time()-t0:.0f}s | "
        f"runner {'OK' if result.runner_ok else 'FAILED'} ==="
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
