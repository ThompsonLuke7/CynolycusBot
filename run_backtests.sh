#!/usr/bin/env bash
# Runs all 4 backtests sequentially (avoids RAM contention from 4 parallel XGB loads)
set -e
REPO=/home/luket/repos/CynolycusBot
export PYTHONPATH=$REPO
PY=$REPO/.venv/bin/python
SCRIPT=$REPO/multi_ticker_swing/backtest/simulate.py
MODELS=$REPO/multi_ticker_swing/models
RESULTS=$REPO/multi_ticker_swing/backtest/results

mkdir -p $RESULTS/oof_1500 $RESULTS/oof_600 $RESULTS/3m_1500 $RESULTS/3m_600

echo "[$(date -u)] Starting oof_1500 ..."
$PY $SCRIPT --test-start 2021-06-01 --test-end 2026-04-15 --force \
  --model $MODELS/swing_xgb_model.json --results-dir $RESULTS/oof_1500

echo "[$(date -u)] Starting oof_600 ..."
$PY $SCRIPT --test-start 2021-06-01 --test-end 2026-04-15 --force \
  --model $MODELS/swing_xgb_model_600.json \
  --features $MODELS/selected_features_600.txt \
  --results-dir $RESULTS/oof_600

echo "[$(date -u)] Starting 3m_1500 ..."
$PY $SCRIPT --test-start 2026-01-15 --test-end 2026-04-15 --force \
  --model $MODELS/swing_xgb_model.json --results-dir $RESULTS/3m_1500

echo "[$(date -u)] Starting 3m_600 ..."
$PY $SCRIPT --test-start 2026-01-15 --test-end 2026-04-15 --force \
  --model $MODELS/swing_xgb_model_600.json \
  --features $MODELS/selected_features_600.txt \
  --results-dir $RESULTS/3m_600

echo "[$(date -u)] ALL 4 BACKTESTS COMPLETE"
