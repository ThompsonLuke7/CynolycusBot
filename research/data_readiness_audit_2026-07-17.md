# Data collection and readiness audit — 2026-07-17

## Incident finding

The entry block was not a model-signal failure. The readiness stamp last
succeeded 2026-07-12 13:26 ET. A 2026-07-14 23:52 ET readiness run rebuilt the
feature matrix and then stopped inside the serial full-universe Yahoo earnings
calendar sweep. Without a newer success stamp, Meta, HTF, and Momentum could
score fresh intraday data but correctly reject new buys.

The 2026-07-16 05:30 exit 75 had an additional deterministic cause: the combined
server acquired `live_data_jobs.lock`, then its child shell tried to acquire the
same lock and rejected itself. It was not proof that the July 14 Yahoo process
was still alive.

Two readiness starts appear at 23:52 and 23:54 on July 14. Current host inspection
found no user crontab, no custom systemd user timer, and no matching Windows task.
The supervised combined server is the only persistent owner now, so those starts
were most likely manual/agent invocations. Historical logs do not preserve enough
caller identity to attribute them conclusively.

## Job inventory and observed behavior

| Workflow | Observed state before repair | Finding |
|---|---|---|
| 05:30 critical readiness | Stamp stale since July 12 | Yahoo enrichment was incorrectly critical; combined-server lock self-deadlocked |
| Intraday catalyst poller | Five cycles on July 16, about 85–90 minutes each | A nominal 5-minute poll swept all 2,903 names and included serial Yahoo news |
| 13:45–16:40 shared refresher | Meta matrix current through July 16 | Correctly kept bars/matrix fresh, explaining valid new signals despite blocked buys |
| 16:30 post-close pipeline | July 16 completed 23:50; recorded stages succeeded | Healthy but long (4–7 hours); overlapped the refresher for its first ten minutes |
| Weekly manual refresh | Last completed July 12 in about 4.5 hours | Appropriate long-running scope, but previously shared no cross-workflow lock/time bounds |

Pre-repair artifact mtimes: CBOE 17:25 ET, FINRA 18:13 ET, news signal 23:36 ET,
and Meta matrix 15:46 ET on July 16; 4H combined features were July 15 00:43 ET;
the readiness stamp was July 12 13:26 ET.

## Remediation

- Readiness now contains only bars, context bars, forced 4H features, Meta matrix,
  and the success stamp. Required stages are fail-fast and time-bounded.
- Yahoo earnings runs post-close with known ETFs excluded, in a replaceable subprocess with a
  hard per-ticker timeout plus a whole-stage timeout. It cannot withhold entry readiness.
- Readiness recognizes the exact combined-server parent lock instead of rejecting
  itself, and skips idempotently when the latest session is already ready.
- Readiness, post-close, and weekly workflows share one owner-preserving lock.
- The canonical post-close time is 16:45 ET, after the shared refresher closes.
- The live catalyst poller now uses the curated 1,110-name swing universe and
  bounded Google/Fed sources; slow Yahoo news remains post-close.
- Nightly priority news is bounded to swing names plus the top 300 Momentum
  snapshot names (about 1,244 unique names); the weekly job retains all names.
- The combined server is documented as the sole scheduler; cron/Windows helpers
  are not production owners.

See `docs/DATA_READINESS_OPERATIONS.md` for the resulting operating schedule.

## Post-repair validation

The off-hours critical workflow completed all required stages successfully on
2026-07-17: 3,092/3,092 bar catch-ups with zero errors, context exit 0, an
8,423,790-row feature cache through 19:00 UTC, Meta matrix current through the
same timestamp, and a new success stamp at 01:03:08 ET. The final code passed 23
focused tests plus shell syntax, Python compilation, and diff whitespace checks.
