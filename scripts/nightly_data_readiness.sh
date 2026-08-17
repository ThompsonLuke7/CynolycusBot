#!/bin/bash
#
# Nightly data-readiness for the shared-universe swing stack.
#
# Guarantees that the THREE shared-universe modules — HTF Swing, Momentum
# Expansion, and Meta Ranker — wake up every trading day caught up to the exact
# same state. They all read the SAME data:
#
#   Data/shared/universe/shared_universe.csv          (the shared universe)
#   Data/shared/bars/{1h,4h,1d}/{TICKER}.parquet      (the shared bars)
#
# so a single refresh keeps all three in lock-step. This job does the heavy work
# that must NOT collide with the live session or the 15:50 Meta Ranker MOC loop,
# so schedule it off-hours (default: pre-open via the combined server).
#
# Entry-critical steps:
#   1) Catch up shared bars for the WHOLE universe (1H/4H/1D), with 429 backoff.
#   2) Refresh shared context bars.
#   3) Rebuild the daily market-regime / sector-state tables off those fresh 1D
#      bars (non-fatal — see below).
#   4) Rebuild the 4H feature matrix (features_4h.parquet) the HTF Swing dashboard
#      scores off — this is what was going 21 days stale.
#   5) Append the fresh bars to the Meta Ranker rolling matrix.
#
# Step 3 ordering is load-bearing: daily_regime.parquet is computed from the
# Data/shared/bars/1d files that step 1 refreshes, and step 4's feature matrix
# JOINS the regime table. Running it here means the 4H features are built
# against a same-session regime instead of a stale one. Until 2026-07-30 the
# regime build was in no scheduled job at all, so it sat 6 days stale and
# momentum_expansion logged 5,471 stale-regime warnings on 2026-07-29.
#
# Step 3 is deliberately NON-FATAL: consumers already degrade gracefully (the
# momentum 4H feature matrix warns and exposes regime_stale_days as a feature),
# so a regime hiccup must not withhold the readiness stamp and blank an entire
# trading day — the exact failure mode that made 2026-07-29 dark.
#
# Daily enrichment (earnings/news/etc.) is intentionally NOT in this critical
# path.  It is bounded and owned by nightly_market_data.sh; a flaky vendor must
# not prevent fresh bars/features/matrix from authorizing new entries.
#
# Usage (manual):  bash scripts/nightly_data_readiness.sh
# Logs:            signals/news/data/processed/data_readiness.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_FILE="$REPO_ROOT/signals/news/data/processed/data_readiness.log"
export PYTHONPATH="$REPO_ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

mkdir -p "$(dirname "$LOG_FILE")" "$REPO_ROOT/Data/runtime"

# Never let this full-universe readiness job collide with live startup/open.
# It belongs overnight/pre-open; the live server now has a lighter afternoon
# bars+matrix refresher for 4H decision times.
HHMM="$(TZ=America/New_York date +%H%M)"
DOW="$(TZ=America/New_York date +%u)"
if [ "${ALLOW_LIVE_READINESS:-0}" != "1" ] && [ "$DOW" -le 5 ] && [ "$HHMM" -ge 0745 ] && [ "$HHMM" -lt 1640 ]; then
  {
    echo ""
    echo "[$(ts)] nightly_data_readiness.sh skipped: blocked during live window (${HHMM} ET)"
  } >> "$LOG_FILE" 2>&1
  exit 76
fi

# Idempotency: an off-hours repair completed after the latest session already
# satisfies the morning schedule. Avoid rebuilding a 4 GB feature cache twice.
#
# --for-next-session is load-bearing: this job runs at 22:15 to authorize the
# NEXT session, so it must ask whether the stamp will still be good tomorrow,
# not whether it is good now. Without it a same-day stamp satisfies the plain
# gate (prev_trading_day at 22:15 Mon is Friday), the job skips, and Tuesday
# opens on a stamp that has since gone stale — six times: 07-28, 08-05, 08-07,
# 08-10, 08-12, 08-14.
if [ "${FORCE_DATA_READINESS:-0}" != "1" ] && "$PYTHON" -m core.live_readiness --for-next-session >/dev/null 2>&1; then
  {
    echo ""
    echo "[$(ts)] nightly_data_readiness.sh skipped: current session is already ready"
  } >> "$LOG_FILE" 2>&1
  exit 0
fi

LOCK_FILE="$REPO_ROOT/Data/runtime/live_data_jobs.lock"
OWNS_LOCK=0
PARENT_OWNS_LOCK=0
clear_owned_lock() {
  if [ "$OWNS_LOCK" = "1" ]; then
    : > "$LOCK_FILE"
  fi
}
trap clear_owned_lock EXIT

# combined_server acquires this same lock in `heavy_job_guard` and THEN launches
# this script as a child, so our own parent is the holder. That must be detected
# BEFORE waiting on the lock, not after: on 2026-07-30 both stale-stamp catch-ups
# blocked the full 90-minute `READINESS_LOCK_WAIT_SECONDS` on their own parent, began
# ~95 minutes late, and were killed mid-feature-build (the second one at 2,875 of
# 2,888 tickers) with the stamp still stale — a third consecutive dark day for
# every 4H entry. The check below already existed but sat in the timeout branch,
# which is exactly too late to be useful.
#
# Two factors must agree before we skip locking: our parent's PID, and the lock
# file naming that same PID as the combined-server holder.
if grep -q "^combined-server-data-readiness pid=${PPID} " "$LOCK_FILE" 2>/dev/null; then
  PARENT_OWNS_LOCK=1
  {
    echo ""
    echo "[$(ts)] nightly_data_readiness.sh proceeding under parent's heavy-job lock (combined-server pid=${PPID})"
  } >> "$LOG_FILE" 2>&1
elif command -v flock >/dev/null 2>&1; then
  # Append mode matters: a losing contender must not truncate the current
  # owner's metadata.
  exec 9>>"$LOCK_FILE"
  # Wait rather than bail: this job now runs in the evening, right behind
  # nightly_market_data.sh, whose runtime varies by an hour or more. Failing
  # instantly on a still-held lock would silently leave the stamp stale, which
  # blocks every 4H entry the following session.
  if ! flock -w "${READINESS_LOCK_WAIT_SECONDS:-5400}" 9; then
    {
      echo ""
      echo "[$(ts)] nightly_data_readiness.sh skipped: another heavy data job is already running"
    } >> "$LOG_FILE" 2>&1
    exit 75
  fi
  OWNS_LOCK=1
  printf 'nightly_data_readiness pid=%s started=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_FILE"
fi

# Durable record of how far a run got, rewritten after every stage. Without it a
# killed run leaves no trace of its progress at all: on 2026-07-30 stages 1-3
# succeeded twice and nobody could tell from any artifact, because only the
# final all-or-nothing stamp is written on success. Consumed by nothing that
# gates trading — the per-ticker check in core/live_readiness.py does that — this
# exists so the next person can see what completed.
PROGRESS_FILE="$REPO_ROOT/Data/readiness/last_run_progress.json"
record_stage() {
  local stage="$1" rc="$2"
  mkdir -p "$(dirname "$PROGRESS_FILE")"
  printf '{"stage": "%s", "exit_code": %s, "at_utc": "%s", "pid": %s}\n' \
    "$stage" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "$PROGRESS_FILE"
}

run_timed() {
  local label="$1"
  local seconds="$2"
  shift 2
  echo "[$(ts)] $label (timeout=${seconds}s)"
  timeout --signal=TERM --kill-after=60s "${seconds}s" "$@"
  local rc=$?
  echo "[$(ts)] $label exit=$rc"
  record_stage "$label" "$rc"
  return "$rc"
}

# Run a stage inside a memory-capped cgroup so a runaway allocation kills the
# STAGE instead of the machine. On 2026-07-31 the 4H feature build reached
# 17.5 GB RSS on a 19 GB box, drank all 24 GB of swap and drove PSI to 99% —
# the kernel OOM killer never fired because everything was swapping rather than
# failing to allocate, so the whole VM livelocked and the session went dark.
# A cgroup limit converts that unbounded thrash into a clean non-zero exit,
# which leaves the stamp stale (one dark session) instead of taking the box
# down (which also kills the live server and every other job).
#
# MemorySwapMax=0 is the load-bearing half: without it the cgroup just swaps to
# the same standstill. Degrades to an uncapped run if systemd --user isn't up.
#
# Built as a command PREFIX rather than a wrapper function because run_timed
# execs `timeout`, which resolves a real binary and cannot call a shell function.
# The 18G default is measured, not guessed: a full 2,886-ticker rebuild peaked at
# 15.74 GiB on 2026-08-03 (accumulation tops out ~5 GB; the spike is pd.concat's
# unavoidable 2x plus the sort's copy, on top of an input heap glibc will not
# return). That is a tight fit on a 19 GB box — it holds while nothing else heavy
# runs, which is true in the pre-open window this job owns. Lowering the peak
# properly means an out-of-core combine; until then, do not schedule anything
# large alongside stage 4.
MEM_CAP=()
if command -v systemd-run >/dev/null 2>&1 && systemctl --user is-system-running >/dev/null 2>&1; then
  MEM_CAP=(systemd-run --user --scope --quiet --collect
           -p MemoryMax="${READINESS_FEATURES_MEMORY_MAX:-18G}" -p MemorySwapMax=0 --)
fi

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] nightly_data_readiness.sh starting"
  echo "================================================================"

  STATUS=0

  run_timed "1/5 catch up shared bars (1H/4H/1D, full universe, 429 backoff)" \
    "${READINESS_BARS_TIMEOUT_SECONDS:-7200}" \
    "$PYTHON" -u scripts/catchup_shared_bars.py --workers 4
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    run_timed "2/5 refresh shared context bars (including VIXY)" \
      "${READINESS_CONTEXT_TIMEOUT_SECONDS:-1800}" \
      "$PYTHON" -u scripts/refresh_shared_context_bars.py
    STATUS=$?
  fi

  # NON-FATAL by design (see header): rc is logged but never folded into STATUS,
  # so a regime failure degrades feature freshness instead of withholding the
  # stamp and blanking the next session.
  if [ "$STATUS" -eq 0 ]; then
    run_timed "3/5 rebuild daily market-regime + sector-state tables" \
      "${READINESS_REGIME_TIMEOUT_SECONDS:-1800}" \
      "$PYTHON" -u -m signals.market_regime.build
    REGIME_STATUS=$?
    if [ "$REGIME_STATUS" -ne 0 ]; then
      echo "[$(ts)] WARNING: market-regime rebuild failed (exit=$REGIME_STATUS); continuing on stale regime (non-fatal — consumers expose regime_stale_days)"
    fi
  fi

  if [ "$STATUS" -eq 0 ]; then
    echo "[$(ts)] 4/5 rebuild HTF 4H feature matrix (features_4h.parquet)"
  # --refresh-stale, not --force. Plain (no flag) skips entirely when the combined
  # parquet exists, which froze the matrix and had HTF Swing scoring a 3-week-old
  # bar. --force fixed that but rebuilt all ~2,900 tickers every run, so a run
  # killed part-way lost everything: on 2026-07-30 two consecutive catch-ups died
  # inside this stage (the second at 2,875/2,888) and each restarted from zero,
  # keeping the 4H entry gate shut for a third session. --refresh-stale rebuilds
  # only tickers whose features are older than their bars and always rewrites the
  # combined parquet, so it stays correct AND resumes. Use --force by hand after
  # feature-code changes, which leave mtimes untouched.
    if [ "${#MEM_CAP[@]}" -eq 0 ]; then
      echo "[$(ts)] NOTE: systemd --user unavailable; 4/5 runs without a memory cap"
    fi
    run_timed "4/5 build-features" "${READINESS_FEATURES_TIMEOUT_SECONDS:-7200}" \
      "${MEM_CAP[@]}" \
      "$PYTHON" -u -m strategies.momentum_expansion.main --build-features --refresh-stale
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    run_timed "5/5 append fresh bars to the Meta Ranker matrix" \
      "${READINESS_MATRIX_TIMEOUT_SECONDS:-2700}" \
      "$PYTHON" -u signals/meta_context/meta_ranker/update_meta_matrix.py
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" -m core.live_readiness --write-success --job nightly_data_readiness
    echo "[$(ts)] readiness stamp updated"
  else
    echo "[$(ts)] readiness aborted; stamp NOT updated (status=$STATUS)"
  fi
  echo "[$(ts)] nightly_data_readiness.sh complete status=$STATUS"
} >> "$LOG_FILE" 2>&1

exit "$STATUS"
