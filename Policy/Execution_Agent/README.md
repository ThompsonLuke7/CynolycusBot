# 1m Execution Agent (Direction-Gated)

This module trains a 1-minute execution policy conditioned on frozen 15-minute intent.

## Design

- HTF command inputs (from 15m agent trace):
  - `htf_dir` in `{-1,0,+1}`
  - `htf_conf`
  - `time_since_flip_min`
  - `htf_atr_pct`
  - `htf_expected_edge` (optional placeholder)
- Action space (discrete, execution-only):
  - `0=WAIT`
  - `1=ENTER`
  - `2=SCALE_IN`
  - `3=SCALE_OUT`
  - `4=EXIT`
- Direction gate:
  - opens/scales are only allowed in current `htf_dir`
- Reward:
  - `(agent_net - baseline_net) - MAE_penalty - churn_penalty - low_conf_penalty`
  - Baseline is dumb execution: immediate position in `htf_dir`, flat on `htf_dir=0`.

## Stage A/B Training

1. Oracle labels (`flip-window` sniper target) + supervised XGBoost classifier.
2. PPO fine-tune using baseline-relative reward.

## OOF Intent

Use `build_intent_oof.py` to build walk-forward OOF HTF commands from a manifest of fold ranges and 15m checkpoints.

## Example

```bash
python Policy/Execution_Agent/run_train.py \
  --ticker SPY \
  --raw-1m-parquet Data/raw/spy/1m_train.parquet \
  --htf-intent-path Data/outputs/agent/agent_trace.csv \
  --total-timesteps 1500000
```

