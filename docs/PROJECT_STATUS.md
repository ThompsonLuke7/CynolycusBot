# Project Status Map

This repo is an active trading-research workspace, not a single production package.

## Production-ish / Operable

- `strategies/multi_ticker_swing/live/`: most relevant live/paper swing bot path; still needs conservative controls and audit-first operation.
- `core/API/Alpaca_API/core/` and `core/API/Alpaca_API/options/`: shared broker/data plumbing used by live workflows.
- `UI/combined_server.py`, `UI/swing_dashboard.py`, `UI/shared_stream.py`: current monitoring surface for paper/live swing sessions.
- `strategies/multi_ticker_swing/live/real_account_policy.py`: real-account guardrail layer for new option entries.

## Research-Ready

- `strategies/multi_ticker_swing/`: current best candidate research line for broader swing setups.
- `signals/news/` and `signals/catalysts/`: catalyst ingestion/scoring pipeline with useful collectors and labels.
- `theme_expansion/`: rule-based theme rotation and Colab export work.
- `strategies/momentum_expansion/`: promising ranker/playbook research, especially broad training plus filtered execution.
- `signals/events/forward_guidance/`: read-only post-earnings guidance module.
- `signals/social_attention/`: Reddit/social attention pipeline MVP.
- `strategies/intraday_structure/`: deterministic, paper-only v1 confirmation engine with persistent setup state, structural levels, replay labels, tests, and combined-server monitoring. It writes closed-setup and abstention ledgers, publishes a 09:00 ET pre-open plan, and as of 2026-08-28 discovers top-750 liquid opening gap/acceleration leaders plus strictly validated catalysts. A unified decision funnel records candidate capacity, setup decisions, and 5/15/30/60-minute candidate MFE/MAE. Thresholds remain uncalibrated — execution promotion is still gated on 6-8 weeks and the frozen ablation. The forward 1-minute archive must be verified after restart; live NBBO/OPRA flow remains a data gap.

## Legacy / Experimental

- SPY daytrader and Phase 4 scripts under `strategies/spy_intraday/Models/ga_xgboost/`, `strategies/spy_intraday/Policy/`, and many root `scripts/` files remain valuable as research history, but recent live evidence suggests they are not the leading monetization path.
- Deep-learning folders such as `strategies/spy_intraday/Models/iTransformer/`, `bilstm/`, and `tcn/` should be treated as archived experiments unless intentionally revived.

## Local-Device Priorities

- Run safe smoke tests before and after refactors.
- Add unit tests around trading safety policy, dashboard stream fanout, and live-session time gates.
- Move repeatedly reused script logic into package modules with focused tests.
- Keep GPU-heavy training and broad data backfills for the other machine unless explicitly planned.
