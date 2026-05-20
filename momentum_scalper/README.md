# Momentum Scalper

Fast, simple, reproducible MVP for small-cap momentum replay:

1. Build/download parquet caches.
2. Reconstruct historical premarket scanner state.
3. Generate features and forward labels.
4. Rank setups with deterministic rules or XGBoost.
5. Replay entries/exits minute by minute and report expectancy.

The first version is intentionally parquet-first and broker-neutral. Downloader
modules require vendor keys, while scanner, features, labels, replay, ranker,
and reports all work from local caches.

## Typical Local Flow

```bash
python -m momentum_scalper.scanners.historical_scanner --day 2026-05-01
python -m momentum_scalper.features.build_features
python -m momentum_scalper.labels.build_labels
python -m momentum_scalper.models.build_training_matrix
python -m momentum_scalper.backtests.replay_engine --day 2026-05-01 --output momentum_scalper/data/processed/replay_2026-05-01.parquet
python -m momentum_scalper.plots.report momentum_scalper/data/processed/replay_2026-05-01.parquet
```

Training is optional:

```bash
python -m momentum_scalper.models.train_xgb
```
