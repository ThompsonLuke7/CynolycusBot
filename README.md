# CynolycusBot

**A research-first quantitative trading system for US equities and options.**

CynolycusBot ingests market, news, options, and macro data; engineers point-in-time
features; trains and validates ML models; ranks candidates across several independent
strategy modules; and runs the surviving signals through a paper-trading execution stack
with live monitoring dashboards.

It is an active research workspace, not a packaged library — expect saved experiments,
model artifacts, backtest outputs, and live audit traces alongside the code.

> ⚠️ **Paper trading only by default.** Nothing here is financial advice, and saved
> backtests or replay results do not guarantee live performance. See [Disclaimer](#disclaimer).

---

## 🌐 See it live: the 3D Theme Explorer

The dynamic theme pipeline groups the tradable universe into market narratives
(AI infrastructure, GLP-1, uranium, defense primes, …) from news embeddings and
description-anchored ticker profiles, then renders them as an interactive 3D graph.

### ▶ **[thompsonluke7.github.io](https://thompsonluke7.github.io/)**

Currently **156 themes** across **3,199 tickers**, rebuilt and republished automatically
every night. Click a theme to see its members and related themes; click a ticker to see
every theme it belongs to. Runs entirely in the browser — no login, no server.

Built by [`themes/dynamic_theme/viz/build_theme_explorer.py`](themes/dynamic_theme/viz/build_theme_explorer.py)
and published by [`scripts/publish_theme_explorer.py`](scripts/publish_theme_explorer.py)
(allowlisted to a single HTML file — no data, code, or credentials leave the repo).

---

## What's inside

| Layer | What it does | Where |
|---|---|---|
| **Data ingestion** | Alpaca bars/quotes/options, Schwab chains, CBOE options summary, FINRA short volume, SEC filings, news feeds | [core/API/](core/API/), [signals/](signals/) |
| **Signals** | Catalyst news (BGE embeddings + FinBERT tone + clustering), scheduled events, market/sector regime, social attention, parabolic filter | [signals/](signals/) |
| **Themes** | Dynamic theme taxonomy, ticker→theme membership, theme rotation features, 3D explorer | [themes/](themes/) |
| **Strategies** | Six independent ranking/execution modules (below) | [strategies/](strategies/) |
| **Execution** | Order policy, defined-risk option routing, readiness gates, broker reconciliation, audit logs | [core/live_4h_exec.py](core/live_4h_exec.py), [strategies/*/live/](strategies/) |
| **Monitoring** | One combined server hosting 11 dashboards | [UI/](UI/) |
| **Research** | Portfolio sizing, pyramiding, options, confluence labs + daily live post-mortems | [research/](research/) |

### Strategy modules

| Module | Horizon | Status |
|---|---|---|
| [SPY intraday](strategies/spy_intraday/) | 10m signal → 1m confirmation | Paper-live; legacy research line, no longer the leading path |
| [Multi-ticker swing](strategies/multi_ticker_swing/) | 30-minute bars | Paper-live; most mature execution path |
| [HTF swing](strategies/multi_ticker_swing_htf/) | 4-hour bars | Paper-live |
| [Momentum expansion](strategies/momentum_expansion/) | Daily/4H continuation | Paper-live |
| [Meta Ranker](signals/meta_context/meta_ranker/) | Confluence ensemble over the above | Paper-live |
| [Dealer positioning](strategies/dealer_positioning/) | Options/dealer flow (Amethyst, Dealer Ranker) | Paper-live |
| [Intraday structure](strategies/intraday_structure/) | Event-driven 1m confirmation engine; closed-setup ledger, abstention log, pre-open plan | Paper-only, opt-in |

---

## Quickstart

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Python 3.12 is the tested interpreter. Alpaca/Schwab-backed workflows need credentials in
`.env`. Some research paths need extras that are **not** in the base requirements
(`transformers`, `sentence-transformers`, `hdbscan`, `lightgbm`).

### Look around without touching anything

These inspect local CLIs — they do not train models, fetch data, or submit orders:

```bash
python scripts/run_safe_smoke_tests.py
./.venv/bin/python -m strategies.momentum_expansion.main -h
./.venv/bin/python -m strategies.multi_ticker_swing.main -h
./.venv/bin/python -m signals.news.main --stage features -h
```

### Run the tests

```bash
./.venv/bin/python -m pytest
```

`pyproject.toml` scopes `testpaths` to the maintained suites (UI, momentum expansion,
swing, dealer positioning, intraday structure, news, catalysts, events, options lab).

---

## The live/paper stack

One supervised process serves every dashboard and shares a single Alpaca WebSocket
(the IEX free tier allows only one concurrent stream):

```bash
scripts/run_live_server.sh
```

The launcher adds crash-loop restart with backoff, glibc allocator tuning against RSS
growth, and a heartbeat watchdog that alerts when no audit file is written during RTH.
Defaults to **paper**; `LIVE=1` is required to route orders to a live account.

Open the hub at **http://localhost:8764** — it links to every module:

| Port | Dashboard | Port | Dashboard |
|---|---|---|---|
| 8764 | Hub | 8771 | HTF Swing |
| 8765 | SPY Intraday | 8772 | Amethyst |
| 8766 | Swing | 8773 | Dealer Ranker |
| 8768 | Dealer Positioning | 8774 | Intraday Structure |
| 8769 | Meta Ranker | 8775 | Library (news/catalyst search) |
| 8770 | Momentum | | |

### Scheduled jobs

| Script | When | What |
|---|---|---|
| [`scripts/nightly_data_readiness.sh`](scripts/nightly_data_readiness.sh) | 22:15 ET | Keeps HTF Swing, Momentum Expansion, and Meta Ranker in lock-step: whole-universe 1H/4H/1D bar catch-up → context bars → market-regime rebuild → 4H feature matrix → Meta rolling matrix → readiness stamp. **No stamp, no entries the next session.** |
| [`scripts/nightly_market_data.sh`](scripts/nightly_market_data.sh) | 16:45 ET | CBOE options snapshot, FINRA short volume, ticker discovery, catalyst news collect/embed/cluster/label, news-catalyst signal, theme explorer rebuild + publish |
| [`scripts/weekly_refresh.sh`](scripts/weekly_refresh.sh) | Sunday, manual | Full-universe bar backfill, momentum universe snapshot, Meta Ranker feeds, and dynamic theme taxonomy rebuild |

---

## Research conventions

The rules the whole repo is held to live in [AGENTS.md](AGENTS.md). The load-bearing ones:

- **Point-in-time or it doesn't count.** Every feature must be available at its decision
  timestamp. Event time, availability time, signal time, order time, fill time, and
  evaluation time are tracked separately.
- **Validate the data source before building on it.** Check that a derivative's returns
  actually correlate with its underlying; distinguish trade prints from marks; confirm the
  data entitlement actually in use. This rule exists because an entire options-routing
  study reached confident conclusions off stale option trade prints
  (corr with the stock was +0.09) and had to be fully retracted —
  see [research/options_experiment/10_RETRACTION_option_pnl_invalid.md](research/options_experiment/10_RETRACTION_option_pnl_invalid.md).
- **Pre-register, then test.** Hypotheses are written before the run, and null results are
  published as prominently as positive ones — see the pyramiding study
  ([research/pyramid_lab/](research/pyramid_lab/), null once capital-matched), the
  regime-policy study ([research/portfolio_lab/regime_policy/](research/portfolio_lab/regime_policy/),
  no rule beat the constant baseline), and the confluence discovery study
  ([research/confluence_discovery_2026-07-07.md](research/confluence_discovery_2026-07-07.md),
  zero certified cross-signal interactions).
- **Research and live compute features the same way.** Divergence between the two is
  treated as a bug, not a detail.

---

## Where to read next

| Document | For |
|---|---|
| [LIVING_SUMMARY.md](LIVING_SUMMARY.md) | Reverse-chronological engineering log — decisions, validation, retractions |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Which modules are operable vs. research-ready vs. archived |
| [docs/ResearchPaperSummarySoFar.md](docs/ResearchPaperSummarySoFar.md) | Research narrative and best saved results |
| [capstone_final/](capstone_final/) | Full written capstone report with figures and evidence audit |
| [research/daily_live_reports/](research/daily_live_reports/) | Daily paper-trading post-mortems: P&L attribution and bug triage |
| [docs/GCP_MIGRATION_TUTORIAL.md](docs/GCP_MIGRATION_TUTORIAL.md) | In-progress port to Google Cloud |
| [docs/superpowers/specs/](docs/superpowers/specs/) | Design specs, including the in-progress "nervous system" rearchitecture |

---

## Repo map

```
core/            Broker/data plumbing (Alpaca, Schwab), shared universe, plotting, live execution
signals/         news · catalysts · events (incl. forward guidance) · market_regime
                 social_attention · meta_context · parabolic_filter
themes/          dynamic_theme (live pipeline + 3D explorer) · theme_expansion_legacy
strategies/      spy_intraday · multi_ticker_swing · multi_ticker_swing_htf
                 momentum_expansion · momentum_scalper · dealer_positioning
                 intraday_structure · model_training
research/        portfolio_lab · pyramid_lab · options_lab · confluence · capstone
                 daily_live_reports
UI/              combined_server + 11 dashboards, shared stream, nightly scheduler
scripts/         Nightly/weekly jobs, supervised launcher, research one-offs
Data/            Raw → processed → models → inference → live_runs (~11 GB, mostly gitignored)
docs/            Status maps, design specs, operations runbooks, migration tutorial
```

Data conventions: raw market data is immutable under `Data/raw/`; cleaned and engineered
data is versioned separately under `Data/processed/`; model artifacts under `Data/models/`;
live and replay session traces under `Data/inference/live_runs/`. Many outputs are local
artifacts that are not reproducible without the same caches, credentials, and package versions.

---

## Handle with care

- **Training and sweeps are expensive.** GA-XGBoost training, the deep-learning folders,
  and `scripts/run_*`/`sweep_*`/`probe_*` should never be run casually.
- **`core/API/Schwab_API/` can place real orders.** Verify account mode first.
- **Do not overwrite** raw market data, historical labels, experiment outputs, or live
  trading logs.
- Prefer static inspection, narrow smoke runs, and cached artifacts.

---

## Disclaimer

This repository is for research and engineering purposes. It is **not financial advice**.
Backtests, replay results, and paper-trading records do not guarantee live trading
performance. Trading equities and options involves substantial risk of loss.
