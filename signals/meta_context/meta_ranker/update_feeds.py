"""
Refresh the upstream feeds the Meta Ranker matrix consumes (step 3 of the live loop).

Each feed is an existing pipeline; this just sequences them and continues on error so
one flaky source doesn't block the rest. Network/credentials required (Alpaca/news).

Cadence:
  * per-4H bar (default): news incremental, economic calendar, forward-guidance signal.
  * daily (--daily): news-catalyst rescore. The bounded earnings sweep is owned
    by nightly_market_data.sh and is not part of entry readiness.
  * optional treasury (--include-treasury): FRED rates, non-required by default.
  * weekly (--weekly): ALSO the dynamic-theme pipeline (recluster + Claude labeling).
    This uses the Claude API ($), so it is OFF unless --weekly is passed.

  PYTHONPATH=. python signals/meta_context/meta_ranker/update_feeds.py            # per-bar feeds
  PYTHONPATH=. python signals/meta_context/meta_ranker/update_feeds.py --weekly   # + themes (costs $)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable

# (label, argv) — run with cwd=REPO, PYTHONPATH=REPO
#
# PER_BAR: light feeds safe to refresh every 4H bar (~10 min total). These keep
#   the matrix's fast-moving columns current intraday.
# DAILY:   whole-corpus news-catalyst rescore for explicit/manual maintenance.
#   Production nightly_market_data.sh already rebuilds this signal after its
#   priority news collection. Earnings are separately bounded there.
PER_BAR = [
    ("news: incremental",      [PY, "-m", "signals.news.main", "--stage", "incremental"]),
    ("news: economic-calendar", [PY, "-m", "signals.news.main", "--stage", "economic-calendar"]),
    ("forward-guidance signal", [PY, "signals/meta_context/build_forward_guidance_signal.py"]),
]
DAILY = [
    ("news-catalyst signal",   [PY, "signals/meta_context/build_news_signal.py"]),
]
OPTIONAL_TREASURY = ("treasury rates (FRED, optional)", [PY, "scripts/fetch_treasury_rates.py"])
WEEKLY = [
    ("dynamic theme (weekly, Claude $)", [PY, "themes/dynamic_theme/pipeline.py", "--mode", "weekly"]),
]


def _run(label: str, argv: list[str], timeout: int) -> bool:
    print(f"\n=== {label} ===\n  $ {' '.join(argv)}")
    env = {"PYTHONPATH": str(REPO)}
    import os
    env = {**os.environ, **env}
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=str(REPO), env=env, timeout=timeout)
        ok = r.returncode == 0
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {label} raised {type(exc).__name__}: {exc}")
        ok = False
    print(f"  {'OK' if ok else 'FAILED'} in {time.time()-t0:.0f}s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true", help="Also run the dynamic-theme pipeline (Claude $).")
    ap.add_argument("--daily", action="store_true",
                    help="Run the whole-corpus news-catalyst rescore INSTEAD of the per-bar set. "
                         "Production nightly collection already owns this step.")
    ap.add_argument("--include-treasury", action="store_true",
                    help="Also try FRED treasury rates. Optional; not part of live-readiness.")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-feed timeout (s).")
    args = ap.parse_args()

    if args.daily:
        jobs = DAILY + (WEEKLY if args.weekly else [])
    else:
        jobs = PER_BAR + (WEEKLY if args.weekly else [])
    if args.include_treasury:
        jobs = jobs + [OPTIONAL_TREASURY]
    results = {label: _run(label, argv, args.timeout) for label, argv in jobs}
    print("\n==== feed refresh summary ====")
    for label, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {label}")
    n_ok = sum(results.values())
    print(f"{n_ok}/{len(results)} feeds refreshed")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
