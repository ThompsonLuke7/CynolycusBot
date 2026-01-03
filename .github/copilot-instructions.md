# Copilot instructions for CynolycusBot

Summary
- Small ML trading repo: data ingestion -> feature generation -> label creation -> model experiments. Focused on SPY daily and intraday data.

Quick commands (how to reproduce common workflows)
- Fetch daily history: `python Data/retrieve_data.py` (writes `spy_data.csv`)
- Build features & labels: `python f.py` (calls `Features.feature_engineering.main()`)
- Quick leakage check: `python Features/test_leakage.py`
- Run GA + XGBoost comparison: `python Models/ga_xgboost/ga_xgboost_compare.py`
- Train BiLSTM regressor: `python Models/bilstm/mabilstm_train.py`
- Fetch intraday using Alpaca: `python Alpaca_API/fetch_intraday.py` (requires Alpaca env vars)
- Schwab interactive CLI (dangerous — does real orders): `python Schwab_API/trading_cli.py` (do NOT run against live accounts in CI)

Architecture & data flow (what to read first)
- Data ingest: `Data/retrieve_data.py` -> `Data/spy_data.csv` (yfinance)
- Feature pipeline: `Features/feature_engineering.py` (uses `Features/pandas_ta_indicators.py` and `Features/custom_indicators.py`)
- Labels: `Features/label_generations.py` (see `add_all_labels()` and `add_atr_pivot_swing_labels()` which is the main labeling scheme)
- Processed artifacts: `Data/processed/` (parquet + .npy files: `X_spy_daily.npy`, `y_spy_daily_*.npy`, `close_spy_daily.npy`, `features_spy_daily.txt`)
- Models consume the processed files directly (see `Models/ga_xgboost/*`, `Models/bilstm/*`, `Models/off_policy_ppo_agent/*`)

Project-specific conventions & gotchas (important)
- Absolute path: `Features/feature_engineering.py` sets `global_file_path = "C:/Users/luket/CynolycusBot"` and some scripts use that — update to a relative path or run from that folder.
- Path casing/inconsistency: some model scripts expect `data/processed/` while the repo uses `Data/processed/`. Verify/correct paths when running experiments.
- Time-series practice: most model code uses chronological splits (train = first chunk, later chunk = validation/test) — do NOT shuffle unless you intentionally change to a randomized split.
- Feature formats: both Parquet (keeps column names) and NumPy `.npy` (fast arrays) are used. Use Parquet when you need column metadata.
- Label leakage: there is an ad-hoc leakage check at `Features/test_leakage.py` — use it after adding features/labels.

Integration & secrets
- Alpaca: credentials via env vars `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. See `Alpaca_API/config.py` which reads a local `.env` if present.
- Schwab: token files in `Schwab_API/` (`schwab_tokens.json` / `schwab_token.json`). The CLI places real orders; **do not run** in CI or without manual confirmation.

Where to add changes (good starting points)
- New indicators: add to `Features/custom_indicators.py` and include in `add_all_custom_indicators()` and `feature_engineering.py`.
- New labels: add to `Features/label_generations.py` and include in `add_all_labels()`.
- New model: add under `Models/`, follow patterns from `Models/ga_xgboost/*` or `Models/bilstm/*` for data loading and chronological evaluation.

Testing & debugging notes
- No formal test suite or CI configured. Use existing scripts (`f.py`, `Features/test_leakage.py`, model scripts) as smoke tests.
- Visual debugging in `Features/feature_engineering.py` plots labels and pivots — helpful when verifying labeling logic.
- Watch for NaNs introduced by technical indicators; the pipeline drops all-NaN columns and may require additional cleaning.

PR / agent checklist (when making changes)
- Run full feature pipeline and confirm `Data/processed/` updated.
- Run `Features/test_leakage.py` (or add a targeted check) when changing labels or features.
- Ensure model scripts load the correct path (fix `data/` vs `Data/` inconsistency).
- Do not execute trading CLIs (Schwab) in automatic runs. Use mocks or stubs for integration tests.

If anything here is unclear or you'd like more detail (examples of label behavior, recommended dependency list, or adding tests), tell me which area to expand. Thanks!