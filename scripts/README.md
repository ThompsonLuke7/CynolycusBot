# Research Script Index

This folder is a mixed workbench: some scripts are safe local diagnostics, while others run broad sweeps, fetch data, or assume specific cached artifacts. Prefer importing durable helpers into package modules when a script starts being reused by live code.

## Safe Local Diagnostics

These are usually good on this device because they read cached logs/artifacts and produce summaries or plots:

- `analyze_*`
- `plot_*`
- `summarize_*`
- focused `compare_*` scripts that point at existing local CSV/JSONL outputs

## Heavy Or Artifact-Sensitive

These can be useful, but check inputs and output directories before running:

- `run_*`
- `sweep_*`
- `probe_*`
- `experiment_*`
- Colab bundle builders and scripts that read large parquet matrices

## Network Or API-Sensitive

These may call Alpaca, yfinance, SEC, Reddit, or other external services:

- `fetch_*`
- `backfill_*`
- pipeline stages in `signals/news/`, `signals/catalysts/`, `signals/events/`, `signals/social_attention/`, and `themes/theme_expansion_legacy/`

## Live-Risk Rule

Any script that can place, cancel, or reconcile orders belongs in a package-level module with tests and explicit dry-run defaults before it becomes part of a normal workflow.
