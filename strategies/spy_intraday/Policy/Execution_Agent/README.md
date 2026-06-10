# 1m Execution Agent (Direction-Gated)

This module trains a 1-minute execution policy conditioned on frozen 15-minute intent.

## Design

- HTF command inputs (from 15m agent trace):
  - `htf_dir` in `{-1,0,+1}`
  - `htf_conf`
  - `time_since_flip_min`
  - `htf_atr_pct`
  - `htf_expected_edge` (optional placeholder)
- Action space (discrete, event-gated):
  - `0=WAIT`
  - `1=EXECUTE` (fire pending HTF event order once)
- Event gate:
  - pending order is created on HTF event changes (enter/exit/opposite flip)
  - `EXECUTE` is one-shot per pending event window
  - opposite flips are treated as switch events (forced exit+enter semantics)
- Reward:
  - Event-timing reward is emitted when `EXECUTE` fires:
  - `reward = (PnL_if_execute_at_agent_time - PnL_if_execute_immediately) - costs - MAE_penalty`
  - Non-execute steps have zero reward.

## Stage A/B Training

1. Oracle event labels (`oracle_enter` + `oracle_exit`) + supervised XGBoost classifiers.
   Training/inference is head-gated to event windows: enter head on `0->±1` windows, exit head on `±1->0` and `±1->∓1` windows.
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
