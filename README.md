# CynolycusBot

CynolycusBot is a research and execution workspace for machine-learning-assisted trading systems. The main line of work is an intraday SPY setup agent: a 10-minute GA-selected XGBoost model produces long/neutral/short setup probabilities, then a 1-minute rule-based order policy decides whether price action confirms an entry. The repo also contains newer research tracks for multi-ticker swing trading, earnings forward-guidance setups, and momentum expansion.

This is an active research repo, not a packaged library. Many folders contain saved experiments, backtest outputs, live/replay traces, and model artifacts. For the current research narrative and best saved results, start with `docs/ResearchPaperSummarySoFar.md`.

## Current Focus

- Intraday SPY setup detection using `swing_support_single` / GA-XGBoost artifacts.
- 1-minute confirmation order policies, especially body-and-close confirmation.
- Live and replay monitoring through the dashboard in `UI/`.
- Alpaca market data, options plumbing, and live-run audit artifacts.
- Expansion research for multi-ticker swing, earnings guidance, and momentum continuation.

## Repo Map

The repo is organized into four top-level groups plus shared data/UI/scripts:

| Path | Purpose |
|---|---|
| `core/API/Alpaca_API/` | Alpaca market-data, option, live-runner, and replay-runner utilities. |
| `core/API/Schwab_API/` | Schwab client, token refresh, and trading CLI (places real orders — handle with care). |
| `core/shared_universe/`, `core/shared_plotting/` | Cross-strategy universe resolution and plotting helpers. |
| `strategies/spy_intraday/Features/` | SPY feature-engineering, labeling, scaling, and leakage-check utilities. |
| `strategies/spy_intraday/Models/ga_xgboost/` | Primary GA-XGBoost training, feature selection, thresholding, and phase-4 analysis code. |
| `strategies/spy_intraday/Models/meta_xgboost/` | Meta entry/exit model experiments. Entry is experimental; exit had stronger validation but is not the main entry source. |
| `strategies/spy_intraday/Models/{iTransformer,bilstm,tcn}/` | Deep-learning research paths from earlier experiments. |
| `strategies/spy_intraday/Policy/` | Inference pipeline, order policy, regime filters, replay proxy logic, and execution-agent experiments. |
| `strategies/multi_ticker_swing/` | Multi-ticker swing pipeline with 30-minute features, labels, matrices, and model artifacts. |
| `strategies/multi_ticker_swing_htf/` | Higher-time-frame (4h) swing research track. |
| `strategies/momentum_expansion/` | Momentum expansion universe, features, labels, matrices, alerts, and backtest work. |
| `strategies/momentum_scalper/` | Momentum scalper MVP research track. |
| `signals/news/` | Unscheduled catalyst news ingestion, deduplication, BGE embeddings, FinBERT tone, clustering, labels, and features. |
| `signals/events/` | Scheduled macro/earnings event context features; `signals/events/forward_guidance/` is the post-earnings guidance module. |
| `signals/catalysts/` | Catalyst record building, scoring, and features. |
| `signals/social_attention/` | Social/attention signal research. |
| `signals/meta_context/` | Specialist-signal and final meta-model scaffolding. |
| `themes/dynamic_theme/` | Dynamic theme taxonomy pipeline (embeddings, clustering, labeling, features). |
| `themes/theme_expansion_legacy/` | Legacy theme pipeline; still supplies `theme_map_v4.csv` and `outputs/universe_filter.csv` to live code. |
| `Data/` | Local raw data, processed matrices, trained artifacts, inference outputs, plots, and live-run logs. |
| `UI/` | Live/replay dashboards for SPY, swing work, and forward-guidance views. |
| `scripts/` | Research scripts for phase-4 policy sweeps, diagnostics, plotting, probability normalization, and live-order analysis. |
| `docs/ResearchPaperSummarySoFar.md` | Current research summary and best-saved-result narrative. |

## Environment

The repo expects Python plus the packages in `requirements.txt`. A local virtual environment already exists in `.venv` in this workspace, but a fresh setup usually looks like:

```bash
python -m venv .venv
```

```bash
./.venv/bin/pip install -r requirements.txt
```

Alpaca-backed workflows need credentials in `.env` or the environment. Be careful with live/paper mode settings before using anything that can submit orders.

Some optional research paths need extra packages that are not in the base requirements. For example, forward-guidance and news NLP features may use `transformers`, `sentence-transformers`, `hdbscan`, or `lightgbm`.

## Safe Orientation Commands

These commands inspect local CLIs and should not train models or submit orders:

```bash
python scripts/run_safe_smoke_tests.py
```

```bash
./.venv/bin/python main.py -h
```

```bash
./.venv/bin/python -m strategies.spy_intraday.Policy.run_inference -h
```

```bash
./.venv/bin/python -m strategies.multi_ticker_swing.main -h
```

```bash
./.venv/bin/python -m signals.events.forward_guidance.main -h
```

```bash
./.venv/bin/python -m signals.events.main --stage features -h
```

```bash
./.venv/bin/python -m signals.news.main --stage features -h
```

```bash
./.venv/bin/python -m strategies.momentum_expansion.main -h
```

For a compact status map of what is operable, research-ready, or legacy/experimental, see `docs/PROJECT_STATUS.md`. For the loose collection of one-off research scripts, see `scripts/README.md` before running broad sweeps or API-fetching jobs.

## SPY Research Pipeline

The older root entry point is `main.py`. It can fetch data, build features/labels, and render label plots. It still supports ticker, timeframe, cache, label-mode, and plotting arguments.

Example plot-only command:

```bash
./.venv/bin/python main.py --ticker SPY --plot-only --plot-type mfe_mae --plot-timeframe 15T --no-refresh-data
```

The current best SPY research artifacts are mostly under:

- `Data/models/ga_xgboost/10min/`
- `Data/models/ga_xgboost/model_competition_phase4/`
- `Data/models/ga_xgboost/model_competition_phase4_focused/`
- `Data/inference/spy/10min/setup/`
- `Data/inference/live_runs/`

The latest root summary intentionally separates model-validation metrics, ATR policy scoreboards, and live-style replay verification because those outputs use different evaluation units.

## Inference, Replay, And Dashboard

The inference code can score raw bars with existing GA-XGBoost artifacts:

```bash
./.venv/bin/python -m strategies.spy_intraday.Policy.run_inference --ticker SPY --raw-parquet Data/raw/spy/spy_intraday_10min.parquet --skip-eval
```

The live/replay dashboard is the easiest way to inspect runtime behavior:

```bash
./.venv/bin/python -m UI.live_dashboard --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`.

The dashboard supports live and replay modes. In replay mode, option orders are forced to simulated payloads when option ordering is enabled. Use the UI controls to start and stop sessions, and use the terminal to stop the dashboard server.

## Multi-Ticker Swing

The multi-ticker swing project is a separate pipeline for broader equity/ETF swing setups. It fetches 30-minute and 10-minute data, builds 30-minute features, creates swing and triple-barrier labels, and merges them into a training matrix.

```bash
./.venv/bin/python -m strategies.multi_ticker_swing.main --stage matrix
```

Use `--stage fetch`, `--stage features`, `--stage labels`, `--stage matrix`, or `--stage all` depending on the work. Fetching and full recomputation can take time and may call external APIs.

## Forward Guidance

The forward-guidance project studies post-earnings opportunities where the market reaction may disagree with historically bullish guidance traits. V1 is read-only and does not submit orders.

Common stages:

```bash
./.venv/bin/python -m signals.events.forward_guidance.main --stage discover-events --start 2025-01-01 --end 2026-02-01 --discovery-source sec --limit 10
```

```bash
./.venv/bin/python -m signals.events.forward_guidance.main --stage features --events-csv events.csv
```

```bash
./.venv/bin/python -m UI.forward_guidance_dashboard
```

See `signals/events/forward_guidance/README.md` for event CSV format and optional NLP features.

## Momentum Expansion

The momentum expansion project scores a broader universe for continuation setups, builds 1-hour / 4-hour / daily context, and can emit live-evaluation alerts.

```bash
./.venv/bin/python -m strategies.momentum_expansion.main --live-evaluate
```

Other stages include `--refresh-universe`, `--fetch`, `--fetch-context`, `--build-features`, `--build-labels`, `--build-matrix`, and `--export-colab`.

## Training And Long-Running Work

Training scripts and broad sweeps can be expensive. Do not run these casually:

- `strategies/spy_intraday/Models/ga_xgboost/train.py`
- `strategies/spy_intraday/Models/iTransformer/run_train.py`
- `strategies/spy_intraday/Models/iTransformer/itransformer_train.py`
- `strategies/spy_intraday/Models/meta_xgboost/train*.py`
- `strategies/spy_intraday/Policy/Execution_Agent/run_train.py`
- large scripts under `scripts/` with names like `run_*`, `sweep_*`, or `probe_*`

Prefer static inspection, narrow smoke runs, or cached artifacts unless a longer experiment is intentional. GPU-heavy AI training should be avoided unless explicitly planned.

## Data And Artifacts

This repo is artifact-heavy. Large files include model bundles, parquet datasets, live-run traces, and Colab export archives. Important conventions:

- Raw market data lives under `Data/raw/`.
- Processed single-SPY datasets live under `Data/processed/`.
- Saved SPY model artifacts live under `Data/models/`.
- SPY setup verification and replay outputs live under `Data/inference/spy/`.
- Live and replay session metadata lives under `Data/inference/live_runs/`.
- Multi-ticker data and models live under `strategies/multi_ticker_swing/data/` and `strategies/multi_ticker_swing/models/`.

Many outputs are local research artifacts and may not be reproducible without the same data caches, API credentials, and historical package versions.

## Notes For Future Work

- Keep the single `swing_support_single` model as the primary SPY entry source until a cleaner challenger beats it out of sample.
- Continue validating body-and-close order-policy behavior against newer windows.
- Keep probability normalization as an experimental calibration layer, not a default policy.
- Add option flow and GEX behavior carefully as contextual features or filters.
- Improve live trading safety around Alpaca order reconciliation, duplicate-order prevention, stale data, and kill-switch behavior.
- Formalize methodology and citations in `docs/ResearchPaperSummarySoFar.md`.

## Disclaimer

This repository is for research and engineering. It is not financial advice, and saved backtests or replay results do not guarantee live trading performance.
