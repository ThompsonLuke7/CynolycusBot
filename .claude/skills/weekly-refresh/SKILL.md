---
name: weekly-refresh
description: Run and verify CynolycusBot's weekend data readiness + refresh so the live stack opens Monday on current data — full-universe bar catch-up, momentum universe snapshot, full-scope catalyst news, Meta Ranker feeds + dynamic themes, then a forced readiness rebuild, plus the Schwab re-auth deadline check. Use when the user asks for the "weekly refresh", "weekend refresh", "data readiness for Monday", "get ready for Monday open", "run the Sunday job", or asks whether the weekly data jobs succeeded.
---

# Weekly Refresh (weekend data readiness)

Gets the shared-universe stack (HTF Swing, Momentum Expansion, Meta Ranker,
themes, dealer) caught up over the weekend so Monday's open is not scored off
stale data. This is a data/ops task — never place, modify, or cancel orders
while doing it.

The heavy lifting already exists in `scripts/weekly_refresh.sh`. This skill is
about running it at the right time, in the right order, and **verifying** it —
a weekly run that exits non-zero mid-stage is the normal case, not the
exception (see Known failure modes).

## Timing

Total runtime is **~5 hours** for a clean run (measured 2026-08-03, 08-10,
08-17: bars ~30 min, universe snapshot <1 min, full news collect ~3 h, news
embed/signal ~25 min, meta feeds + themes ~30-50 min). The forced readiness
rebuild that follows adds ~1 h.

So: start it by **~02:00 ET Monday at the latest**. Started Sunday evening it
finishes around 04:00-05:00 ET, comfortably before the 09:30 open. If it is
already past ~03:00 ET Monday, say so and either run a reduced set (stages 1-2
only) or skip to the readiness rebuild — do not start a 5-hour job that will
still be holding the heavy-job lock at the open.

## Procedure

### 1. Pin the time and check the Schwab deadline FIRST

Run `date`. Then compute the Schwab refresh-token expiry before anything else —
it is the only step that needs a human, it is easy to miss, and it silently
kills dealer gamma levels for the week:

```bash
.venv/bin/python -c "
import datetime, json, zoneinfo
et = zoneinfo.ZoneInfo('America/New_York')
c = json.load(open('core/API/Schwab_API/schwab_token.json'))['creation_timestamp']
exp = datetime.datetime.fromtimestamp(c + 7*86400, et)
print('refresh token created:', datetime.datetime.fromtimestamp(c, et))
print('expires (+7d):', exp)
print('hours left:', round((exp - datetime.datetime.now(et)).total_seconds()/3600, 2))
"
```

The refresh token lives **7 days** from `creation_timestamp`. If it expires
before the next weekend, tell the user **at the top of your response**, with the
command, and do not bury it under the refresh status:

```
.venv/bin/python -m core.API.Schwab_API.schwab_client --reauth
```

It needs an interactive browser login; nothing in an unattended run can answer
it. `core/API/Schwab_API/schwab_token.invalid_<YYYYMMDDTHHMMSS>.json` files are
past tokens renamed by their expiry — useful for confirming the weekly cadence.

### 2. Pre-flight

Run these in parallel and read before launching:

- `ps aux | grep -E "combined_server|catchup_shared|nightly_|weekly_"` — is the
  live server up, is a heavy job already running?
- `cat Data/runtime/live_data_jobs.lock` — `weekly_refresh.sh` takes this lock
  with `flock -n` and **exits 75 immediately** if anything else holds it.
  `nightly_market_data.sh` (16:45 via combined_server) and
  `nightly_data_readiness.sh` (22:15) contend for it.
- `.venv/bin/python -m core.live_readiness --for-next-session` — current stamp
  state. On a Sunday a Friday-evening stamp usually still passes (`max_age` is
  96 h and no bars have printed since Friday's close); that is not a reason to
  skip the refresh, because the refresh is about *universe* and *news*
  freshness, not bar age.
- `grep -aE "weekly_refresh.sh (starting|DONE)|exit=" signals/news/data/processed/weekly_refresh.log | tail -30`
  — what last week's run did, and which stages failed.
- `git diff --stat` / `git status` — the run executes the **working tree**, so
  know what uncommitted changes are about to run in production data jobs.

### 3. Launch

```bash
nohup bash scripts/weekly_refresh.sh > <scratchpad>/weekly_refresh_run.out 2>&1 &
```

Background it and detach — it outlives the session. Confirm it took the lock
(`cat Data/runtime/live_data_jobs.lock` should name the new pid) before
reporting it as started.

Stages, in the order the script runs them (the order is load-bearing):

| # | Stage | Why it must come here |
|---|---|---|
| 1 | `catchup_shared_bars.py --workers 6` (full universe, 1H/4H/1D) | names promoted during the week have no history yet |
| 2 | `momentum_expansion.main --refresh-universe` | scores off daily bars, so needs #1 first |
| 3 | `collect_news_scope --scope full` → `signals.news.main --stage incremental` → `meta_context.build_news_signal` | nightly only collects the ~1.25k PRIORITY scope; this is the ~2.9k full sweep |
| 4 | `meta_ranker/update_feeds.py --weekly` | Meta feeds + dynamic theme taxonomy; **costs Claude API $** |

### 4. Chain the readiness rebuild

`weekly_refresh.sh` does **not** rebuild the 4H feature matrix or the Meta
Ranker matrix, so newly promoted names have bars but no features. Chain a
forced readiness rebuild behind it rather than waiting to do it by hand — write
a small chaser that polls the weekly pid and then runs:

```bash
FORCE_DATA_READINESS=1 bash scripts/nightly_data_readiness.sh
```

`FORCE_DATA_READINESS=1` is required: the stamp from Friday still satisfies the
idempotency gate, so an unforced run exits 0 without rebuilding anything.

Do **not** also set `ALLOW_LIVE_READINESS=1`. If the chain slips past 07:45 ET
the live-window guard should block it — Friday's stamp still covers Monday, and
a full-universe feature rebuild colliding with live startup is the worse
outcome. Stage 4 of that script peaks near 16 GiB on a 19 GB box, so nothing
heavy may run alongside it.

### 5. Decide about `nightly_market_data.sh` — usually skip it

It is a separate ~5 h job (CBOE options snapshot, dealer snapshots/rankings,
FINRA short volume, ticker discovery, priority news, emerging themes, earnings
calendar). Check `grep -aE "nightly_market_data.sh (starting|complete)" signals/news/data/processed/nightly_cron.log | tail`.
If it completed **after Friday's close**, skip it: no new market data exists
over a weekend, the weekly full-scope news collect supersedes its priority-scope
collect, and running it serially would push the chain past the open. Run it only
if Friday's run is missing or failed.

### 6. Verify — do not report success from an exit code alone

After the chain finishes:

- `cat Data/readiness/weekly_refresh_latest.json` — the completion stamp: overall
  status plus every stage's exit code. Read this first; it is a contract, unlike
  grepping the log. Absent means the run never reached the end.
- `grep -aE "weekly_refresh.sh (starting|DONE)|exit=" signals/news/data/processed/weekly_refresh.log | tail -20`
  — the same rcs in context, for a run that died before stamping.
- `grep -aiE "error|traceback|failed" signals/news/data/processed/weekly_refresh.log | tail -30`
  — stage 4 can exit 0 with many per-cluster failures inside it.
- `.venv/bin/python -m core.live_readiness --for-next-session` — must be `ok: true`
  with a stamp newer than the run.
- `cat Data/readiness/last_run_progress.json` — how far the readiness rebuild got.
- Freshness by mtime of the artifacts the modules actually read:
  `Data/shared/universe/shared_universe.csv`,
  `strategies/momentum_expansion/data/processed/features_4h.parquet`,
  `themes/dynamic_theme/outputs/theme_registry.parquet`,
  `signals/meta_context/meta_ranker/meta_ranker_matrix.parquet`.

State plainly which stages failed. A partial refresh is a normal outcome and the
user needs to know which feeds are stale going into the week — do not smooth it
over.

### 7. Append a `LIVING_SUMMARY.md` entry

Standard convention: `{YYYY-MM-DD HH:MM ET} {agent} {area}` + up to 3 lines —
what ran, what failed, Schwab re-auth state, next step. Timestamp from `date`,
never guessed.

## Known failure modes

- **Stage 4 Claude labeling** (`themes/dynamic_theme/stages/step05_claude_labeling.py`):
  `Failed to parse Claude response as JSON` per cluster; on 2026-08-17 this plus a
  carry-forward name collision (187 clusters onto 88 names, seven called
  `mortgage_reits`) aborted step08's immutable-history guard and left the theme
  registry a week stale. Check the theme registry mtime specifically.
- **`meta feeds exit=1`** recurred on 2026-08-03, 08-17 and 08-24. Stage 4 can be
  rerun on its own: `.venv/bin/python signals/meta_context/meta_ranker/update_feeds.py --weekly`.
  When steps 1-7 already succeeded (check the mtimes of `ticker_embeddings.parquet`,
  `ticker_clusters.parquet` and `theme_registry.parquet` for today's date), resume at
  step 8 instead — `compute_memberships()` then `build_meta_features()` read those
  artifacts straight off disk, which skips the ~40 min of paid Claude labeling.
- **`exit=75`** — lock contention; another heavy job held
  `Data/runtime/live_data_jobs.lock`. Reschedule, do not force.
- **`exit=143`** — SIGTERM from the stage `timeout`. Overridable per stage via
  `WEEKLY_{BARS,UNIVERSE,NEWS_COLLECT,NEWS_PROCESS,NEWS_SIGNAL,FEEDS}_TIMEOUT_SECONDS`.
- **`exit=76`** on the readiness script — blocked by the live-window guard
  (Mon-Fri 07:45-16:40 ET). Expected if the chain ran long; see step 4.
