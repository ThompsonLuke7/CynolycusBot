---
name: daily-review
description: Produce CynolycusBot's Daily Live Secretary report — a per-module performance and bug review of the live/paper trading server (SPY daytrader, Multi-ticker swing 30m, Meta ranker, Momentum expansion, HTF swing, Dealer Ranker/Amethyst, Intraday structure, two-sleeve shadow tracker, nightly data pipeline). Use when the user asks for a "daily review", "daily report", "how did the server/modules do today", "daily live secretary", "performance review of the live server", or wants yesterday's/today's trading activity and bugs summarized. Also use to review a specific past date's activity, not just today's.
---

# Daily Review (Daily Live Secretary)

Produces `research/daily_live_reports/YYYY-MM-DD.md`: a read-only, evidence-based
report of what each live/paper module did on a given trading day, plus any
operational issues. This is a research/reporting task, not a trading action —
never place, modify, or cancel orders while doing this.

Two prior reports are the canonical style reference — read them before writing:
`research/daily_live_reports/2026-07-09.md` (full example) and
`research/daily_live_reports/2026-07-13.md` (short example, for a thin data day).
Match their section structure and tone; don't pad sections that have no data —
say "nothing to report" instead.

## Procedure

1. **Pin the date.** Run `date` — never assume or infer today's date. Default
   to today unless the user names a different date. Report path:
   `research/daily_live_reports/YYYY-MM-DD.md`. If that file already exists,
   confirm with the user before overwriting (it may be a completed report from
   earlier in the day).
2. **Check server health first.** `ps aux | grep combined_server` for the live
   PID, then read `logs/live_server/watchdog_<launch-date>.log` for restarts,
   OOM kills (exit 137), or missed-heartbeat alerts.
   **Gotcha:** `scripts/run_live_server.sh` names `server_YYYYMMDD.log` /
   `watchdog_YYYYMMDD.log` once, at process launch — if the server has run
   continuously across a midnight boundary, today's activity lives inside a log
   file named for a *previous* date. Always filter by in-file timestamps, not
   by filename, and confirm the actual process start time before assuming which
   log file(s) hold today's data.
3. **Gather each module's data from source, not from memory of past reports.**
   Paths shift as the system evolves — verify each one exists before relying on
   it. As of the last time this skill was refreshed:

   | Module | State | Signal audit | Other |
   |---|---|---|---|
   | SPY daytrader | — | — | `Data/inference/live_runs/<ts>_live_spy/` session dirs |
   | Multi-ticker swing 30m | — | — | `UI/swing_audit/` and `UI/swing_audit/paper/` `swing_session_*.jsonl`; `UI/swing_audit/forensics_*/` |
   | Meta ranker | `signals/meta_context/meta_ranker/live_state.json` | `Data/inference/meta_ranker/live_signal_audit.jsonl` | |
   | Momentum expansion | `strategies/momentum_expansion/live/momentum_live_state.json` | `Data/inference/momentum_expansion/live_signal_audit.jsonl` | `strategies/momentum_expansion/live/alerts.jsonl` |
   | HTF swing | `strategies/multi_ticker_swing_htf/live/htf_live_state.json` | `Data/inference/multi_ticker_swing_htf/live_signal_audit.jsonl` | |
   | Dealer Ranker / Amethyst | `Data/inference/dealer_ranker/live_state.json` | `Data/inference/dealer_ranker/live_signal_audit.jsonl` | `Data/dealer_positioning/{SPX,QQQ,SMH,SPY,IWM,GLD,SLV}/live_trade_log.csv` — point-based sim, keep separate from dollar P&L |
   | Intraday structure (paper-only) | `Data/inference/intraday_structure/` | | |
   | Two-sleeve shadow tracker (paper-only, no order path) | `Data/inference/shadow_two_sleeve/` | | reads other modules' `live_state.json`; runs at 14:35/16:35 ET inside `combined_server` |
   | Nightly data pipeline | | | `scripts/nightly_market_data.sh`'s `$LOG_DIR/nightly_cron.log`; `signals/news/data/processed/` for news/theme runs |
   | Broker reconciliation | | | `Data/inference/account_snapshots/broker_equity_<date>_<mode>.jsonl` |

   Each 4H module (Meta/Momentum/HTF) also writes a
   `Data/inference/<module>/pending_open_entries.json` for entries deferred to
   next open (e.g. signals fired after market close) — check it so deferred
   opens aren't miscounted as no activity.

   If a path above no longer exists, grep the codebase for its neighbors
   (`STATE_PATH`, `AUDIT_LOG`, `DEFAULT_*_LOG` constants in each module's
   `live/runner.py` or equivalent) rather than guessing a replacement.

   **Lead the Executive Summary with the account-level equity delta** from the
   broker snapshot (prior close -> latest snapshot) when available — it's the
   one number that nets realized + unrealized across every module sharing the
   account, and it doesn't require reconstructing marks module-by-module. Give
   per-module realized/open figures as the detail underneath it, not as a
   substitute for it.
4. **Aggregate, don't dump.** Several of these files are multi-MB
   (`server_*.log`, `live_signal_audit.jsonl`). Use `jq`, `grep -c`,
   `python -c` one-liners, or `awk` to pull counts, timestamps, and specific
   rows instead of reading whole files into context. Prefer delegating this
   step to a background `Agent` (general-purpose) when there are many sources
   to reconcile — pass it this source map and the two style-reference paths
   directly rather than re-deriving them.
5. **Write the report** with these sections, adapted to what data actually
   exists: title + generation-time note, Executive Summary, Module Scorecard
   table, Top Five Closed Trades, Top Five Open Trades, Module Details (one
   subsection per module with activity), Amethyst/dealer simulation detail,
   Operations And Bugs, Nightly Data, Sources (every file actually used).
6. **Follow repo-wide data-integrity rules** (from `AGENTS.md`): never fabricate
   a number — say "unknown" or "not found" instead; label proxy/mark prices
   explicitly as proxies versus real fills; distinguish signal time from order
   time from fill time; do not overwrite or delete raw logs, state files, or
   existing reports — only create the new report file.
7. **Append a `LIVING_SUMMARY.md` entry** per the standard project convention
   (timestamp from `date`, agent, area, 3 lines max) noting the report was
   produced, headline P&L/health, and the biggest bug found, if any.
8. **Summarize back to the user**: headline realized P&L (if any), server
   health, and the top 1-3 bugs/issues — not the full report inline; point them
   at the file.

## What counts as a "bug" worth flagging

Server crashes/restarts, OOM kills, HTTP errors on order attempts (esp.
after-hours 403s), a position that repeatedly fails to close (403/rejected —
distinguish "stuck, needs a human" from a normal retry), stale/rejected state
left in a module, NaN-served features, stream/queue pressure (dropped bars),
broker/local position mismatches, unverified orders that exhausted retry
ladders, stale reference data (old VIXY context, sklearn artifact-version
warnings, etc.), and auth/credential outages — e.g. an expired broker OAuth
token falling into an interactive re-login prompt that nothing in an
unattended run can answer, silently blocking every downstream job that needed
it. Report these even if they didn't affect P&L — operational health is part
of this review.
