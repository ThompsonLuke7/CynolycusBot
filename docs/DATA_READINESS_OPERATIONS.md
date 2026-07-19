# Data readiness operations

The supervised combined server is the sole day-to-day scheduler. Do not add a
second cron, systemd timer, or Windows task for these scripts.

## Production schedule (America/New_York)

| Time | Owner | Job | Purpose |
|---|---|---|---|
| 05:30 weekdays | combined server | `nightly_data_readiness.sh` | Entry-critical full bars, context bars, forced 4H feature rebuild, Meta matrix append, then readiness stamp |
| 09:30–16:00 weekdays | combined server | intraday catalyst poller | Fast Google/Fed polling for the curated swing universe; separate live ledger |
| 13:45–16:40 weekdays | combined server | `SharedDataRefresher` | Eligible 1H/4H bar catch-up and incremental Meta matrix refresh near decision bars |
| 14:20 / 16:20 | combined server | Meta runner | Read/score/trade only |
| 14:25 / 16:25 | combined server | HTF and Momentum runners | Read/score/trade only |
| 15:45 | combined server | Dealer Ranker | Near-close option-chain ranking pass |
| 16:45 weekdays | combined server | `nightly_market_data.sh` | Bounded post-close collection/enrichment: CBOE, dealer snapshots, FINRA, discovery, priority news, themes, earnings |
| Sunday/manual | operator | `weekly_refresh.sh` | Full-universe news, universe refresh, and weekly dynamic-theme work |

All three shell workflows share `Data/runtime/live_data_jobs.lock`. Exit `75`
means another heavy workflow owns the lock; exit `76` means full readiness was
requested during its prohibited live window.

## What authorizes entries

Only successful completion of the four 05:30 entry-critical stages writes
`Data/readiness/latest_success.json`. Earnings, news, dealer, theme, and other
vendor enrichment are valuable context but cannot withhold the readiness stamp.
They have their own bounded post-close workflow and retain their prior valid
snapshots if a vendor fails.

Fresh signals and blocked orders are therefore not contradictory: scoring can
read newly refreshed bars and emit a decision while the independent entry gate
rejects buys when the last full critical-path stamp is stale. Exits remain
allowed.

## Manual checks

```bash
.venv/bin/python -m core.live_readiness
ps -ef | rg 'combined_server|nightly_data_readiness|nightly_market_data|weekly_refresh'
tail -100 signals/news/data/processed/data_readiness.log
tail -100 signals/news/data/processed/nightly_cron.log
```

Manual repair is safe off-hours; the same lock prevents overlap:

```bash
bash scripts/nightly_data_readiness.sh
```

The script exits successfully without rebuilding when the latest session is
already ready. Use `FORCE_DATA_READINESS=1` only for an intentional off-hours
rebuild.

The legacy `nightly_market_data.ps1` is not a production scheduler or the
canonical pipeline. It is retained only as an old manual Windows helper.
