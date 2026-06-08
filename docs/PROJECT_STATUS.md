# Project Status Map

This repo is an active trading-research workspace, not a single production package.

## Production-ish / Operable

- `multi_ticker_swing/live/`: most relevant live/paper swing bot path; still needs conservative controls and audit-first operation.
- `API/Alpaca_API/core/` and `API/Alpaca_API/options/`: shared broker/data plumbing used by live workflows.
- `UI/combined_server.py`, `UI/swing_dashboard.py`, `UI/shared_stream.py`: current monitoring surface for paper/live swing sessions.
- `multi_ticker_swing/live/real_account_policy.py`: real-account guardrail layer for new option entries.

## Research-Ready

- `multi_ticker_swing/`: current best candidate research line for broader swing setups.
- `news/` and `catalysts/`: catalyst ingestion/scoring pipeline with useful collectors and labels.
- `theme_expansion/`: rule-based theme rotation and Colab export work.
- `momentum_expansion/`: promising ranker/playbook research, especially broad training plus filtered execution.
- `events/forward_guidance/`: read-only post-earnings guidance module.
- `social_attention/`: Reddit/social attention pipeline MVP.

## Legacy / Experimental

- SPY daytrader and Phase 4 scripts under `Models/ga_xgboost/`, `Policy/`, and many root `scripts/` files remain valuable as research history, but recent live evidence suggests they are not the leading monetization path.
- Deep-learning folders such as `Models/iTransformer/`, `Models/bilstm/`, and `Models/tcn/` should be treated as archived experiments unless intentionally revived.

## Local-Device Priorities

- Run safe smoke tests before and after refactors.
- Add unit tests around trading safety policy, dashboard stream fanout, and live-session time gates.
- Move repeatedly reused script logic into package modules with focused tests.
- Keep GPU-heavy training and broad data backfills for the other machine unless explicitly planned.
