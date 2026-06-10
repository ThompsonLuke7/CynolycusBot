# Copilot instructions for CynolycusBot

Summary
- ML trading research/execution workspace. Main line: intraday SPY setup agent (10-min GA-XGBoost probabilities + 1-min rule-based order policy), plus research tracks for multi-ticker swing, momentum expansion, event/news signals, and theme taxonomies.

Layout (top-level groups)
- `core/` — broker integrations (`core/API/Alpaca_API`, `core/API/Schwab_API`) and shared libs (`core/shared_universe`, `core/shared_plotting`).
- `strategies/` — trading strategy tracks: `spy_intraday/` (Features, Models, Policy), `multi_ticker_swing/`, `multi_ticker_swing_htf/`, `momentum_expansion/`, `momentum_scalper/`.
- `signals/` — market context & event signals: `news/`, `events/` (incl. `forward_guidance/`), `catalysts/`, `social_attention/`, `meta_context/`.
- `themes/` — `dynamic_theme/` (current taxonomy pipeline) and `theme_expansion_legacy/` (still supplies universe/theme CSVs to live code).
- `Data/` — datasets, model artifacts, inference/live-run outputs (mostly gitignored).
- `UI/` — live/replay dashboards. `scripts/` — one-off research scripts. `docs/` — research docs and runbooks.

Quick commands (run from repo root)
- Safe smoke tests: `python scripts/run_safe_smoke_tests.py`
- SPY inference: `python -m strategies.spy_intraday.Policy.run_inference -h`
- Swing pipeline: `python -m strategies.multi_ticker_swing.main -h`
- News/events stages: `python -m signals.news.main --stage features -h`, `python -m signals.events.main --stage features -h`
- Dashboard: `python -m UI.live_dashboard --host 127.0.0.1 --port 8765`

Conventions & gotchas
- Always run from the repo root; modules resolve the repo root via `Path(__file__).resolve().parents[N]` and reference data with repo-relative paths.
- Time-series practice: chronological splits only — never shuffle unless intentionally randomizing.
- Tests live in per-module `tests/` dirs; `pyproject.toml` lists pytest testpaths and markers (`safe`, `network`, `slow`, `live`).
- Leakage check after changing features/labels: `strategies/spy_intraday/Features/test_leakage.py`.

Integration & secrets
- Alpaca: env vars `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (or local `.env`).
- Schwab: token files under `core/API/Schwab_API/` are gitignored — never commit them. The trading CLI places real orders; do not run automatically.
- Do not execute trading CLIs or live runners in automated runs; use replay/simulated modes.
