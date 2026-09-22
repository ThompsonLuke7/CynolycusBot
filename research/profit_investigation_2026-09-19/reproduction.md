# Reproduction and scope

The experiment is fixed at September 18, 2026, end of ET day. It reads the paper broker only; nothing submits, cancels, replaces, or edits orders. No existing raw logs, market data, model bundles, or trading configuration were changed.

The [Alpaca account-activity documentation](https://docs.alpaca.markets/us/docs/account-activities) identifies FILL, exercise, expiry, and other activity types. [Order-list documentation](https://docs.alpaca.markets/us/reference/getallorders-1) documents the order pagination. The actual endpoint repeated page-boundary orders despite the documented exclusive cursor. The downloader skips identical order IDs at that boundary; complete filled-order quantities/notionals are independently verified against activity-ID pagination.

Scripts, from repository root:

```bash
.venv/bin/python scripts/profit_investigation/fetch_snapshot.py
.venv/bin/python scripts/profit_investigation/snapshot_local.py
.venv/bin/python scripts/profit_investigation/build_study.py --out run3
.venv/bin/python scripts/profit_investigation/enrich.py
.venv/bin/python scripts/profit_investigation/account_context.py
.venv/bin/python scripts/profit_investigation/analyze.py
.venv/bin/python scripts/profit_investigation/supplement.py
MPLCONFIGDIR=/tmp/cynolycus_profit_matplotlib .venv/bin/python scripts/profit_investigation/plot_summary.py
.venv/bin/python -m pytest -q scripts/profit_investigation/test_study.py
```

**Do not rerun these commands over original research inputs.** The snapshot scripts use exclusive creation, and the broker downloader supports append-only recovery from an interrupted orders download. `build_study` requires a new output directory; `enrich` requires a new analysis directory. To rerun the fixed study, copy its evidence into an isolated checkout and give it a fresh output directory. The statistics/plot scripts update their own derived outputs, not raw evidence. The final source hashes are saved in `analysis/research_code_hashes.json`.

`data/local/spy_events.jsonl` is a frozen extraction of rows with `payload.result.orders` from `Data/inference/live_runs/*_live_spy/trade-events.jsonl`. Each row preserves `source`, `line`, and the original `record`; source hashes are in `data/local/spy_manifest.json`. This extraction was run as a standalone read-only Python step. The other local extraction script retains relevant 30m audit events with the same source/line wrapper. For exact reproduction use these frozen extracts, not later appended operational logs.

Price inputs have SHA-256 hashes in `analysis/price_source_hashes.json`. They were read locally; the study did not download new historical price series or query historical option quotes. If those price files change, a rerun must not be presented as identical to this snapshot. Broker orders/activities and the historical account endpoint snapshot are frozen under `data/`.

Named baseline: **actual observed paper policy as deployed during July 13–September 18**, including its changes/outages and real fills. It is not a homogeneous versioned strategy backtest. No model was trained; no train/validation/test boundary was moved; no policy was selected against a final test set. Overall, last-month, and August comparisons overlap. Feature tests and threshold sensitivity tables are descriptive, not estimated deployable uplift.

Important conventions:

- Economic inventory basis is moving weighted average inside each flat-to-flat lifecycle, with option premium carried into shares on exercise. This is not tax-lot reporting.
- Cash funding is not profit. Exercise transfers are not profit/loss realizations. Worthless expirations require a quantitative broker settlement event. Unresolved inventory is not written off.
- Realized period statistics use exit ET dates; completed-position feature cohorts use final closure ET dates; candidate metrics use decision availability dates. Late-born candidates without complete horizons remain unevaluated.
- A profitable lifecycle/exit must exceed $0.0000001 in reconstructed P&L. Floating-point residuals around zero are classified flat.
- Score comparisons require module/instrument agreement. Conflicting ownership stays separate; matched controls require a score gap no greater than 0.25 cohort SD and entry separation no greater than 14 calendar days.
- Score bootstrap: 1,000 resamples, seed 1909, ticker clusters or entry-date clusters for SPY. Intervals are withheld for fewer than five positive/negative observations or five clusters. Feature-test BH adjustments span all reported comparisons within each of the unpaired/paired families; their underlying tests are unclustered and exploratory.
- The quote-cost primary sample requires an exit quote observed 0–60 seconds before the fill. The 300-second sample is a sensitivity check. Midpoints are references, not guaranteed fills or savings.
- Minute outcome paths require 80% expected RTH minute coverage and fully contained bars. No imputation. Prior daily features require at least 21 observations and a recent previous session; large discontinuities invalidate derived windows. Intraday paths and returns never synthesize option marks.

Known limitations that remain: historical records predating retained broker orders; incomplete exact module ownership and full entry-vector/version lineage; partially observed opportunity populations; open-position maturity bias; unallocated per-trade fees; unresolved CMCO inventory and a $4.78 cash difference. See `report.md` for the implications.
