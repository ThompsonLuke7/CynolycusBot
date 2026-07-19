# Event Replay and Validation Protocol

## Input contract

- True one-minute OHLCV with symbol and UTC timestamp. Higher-timeframe resampling is rejected.
- Candidate records with both the upstream signal timestamp and `available_at` time.
- Optional options snapshots/prints whose timestamps are not after the decision time.
- Optional synchronized SPY, QQQ, VIXY, and sector ETF bars.

Replay processes events chronologically and generates transitions at the exact bar close when evidence exists. A confirmed setup becomes `RUNNING` on the next bar, modeling the default one-bar entry delay. Spread, slippage, commissions, and minimum bar dollar volume are configuration fields; quote-level fill realism remains unavailable from OHLCV.

## Labels

`target_before_invalidation`, target 1/2 hits, VWAP reached after reversal, breakout held bars/failure, reversal failure, MFE, MAE, realized R, and time to target use only the configured forward evaluation window. Same-bar stop/target collisions are scored stop-first. Overlapping events of the same ticker/setup are suppressed for a configurable cooldown.

## Required comparisons

Run the fixed candidate/event set through:

1. Price/structure only.
2. Price/structure plus static options positioning.
3. Static options plus timestamped live flow.
4. Full inputs plus market/sector context.

Compare trade/event count, target-before-invalidation rate, win rate after modeled costs, average R, MFE/MAE, time to target, confidence calibration, setup type, regime, time of day, and options availability. The supplied `run_ablations` helper keeps candidate inputs fixed; it does not prove benefit without representative data.

## Validation gates before execution work

- Freeze train/validation/test dates and candidate universe.
- Verify candidate availability times and options snapshot as-of joins independently.
- Require enough events per setup and regime; report sparse cohorts.
- Select thresholds on validation only, then run one untouched test.
- Paper-run with transition and quote logs; reconcile expected versus actual alert timing.
- Do not connect signals to capital until target/invalidation behavior, costs, restart recovery, and duplicate suppression pass paper validation.

## Live OPRA next step

Implement a provider that normalizes timestamp, OCC contract, expiration, strike, call/put, size, premium, trade price, contemporaneous bid/ask, delta, gamma, and vendor-supported sweep/block indicators into `OptionFlowPrint`. Persist raw immutable prints separately. Validate clock skew, corrections/cancels, crossed quotes, multi-leg prints, late messages, and provider coverage before enabling the live-flow ablation. Continue labeling dealer positioning as an estimate.
