# Intraday Structure v1 Features

All features are computed at one-minute bar close from bars whose event timestamps are at or before the decision timestamp. No centered windows, backward fills, future-confirmed pivots, or whole-session extrema are used.

## Price and structure

- ATR(14), ATR percent, current session VWAP, VWAP slope over up to five prior observations.
- ATR-normalized distance to VWAP, EMA9, and EMA20; duration above/below VWAP.
- Rolling micro swing high/low, causal higher-low/lower-high flags, trend strength from EMA9 versus EMA20.
- Opening-range high/low from observed bars only; prior-day high/low from completed prior sessions.
- Rolling liquidity support/resistance via `signals.location_features.add_liquidity_zone_features`.

## Volume and momentum

- Rolling 1-minute and 5-minute relative volume, volume z-score, range expansion, close location, upper/lower wick ratios.
- One/three-bar returns, selloff acceleration, downside deceleration, rebound velocity, and sign-change momentum divergence.
- The current relative-volume baseline is rolling, not same-minute-of-day seasonally adjusted. This is explicitly a v1 limitation.

## Cross-asset context

- Synchronized SPY and QQQ direction, index VWAP alignment, optional sector ETF relative strength, and VIXY change proxy.
- Directional market alignment is a documented weighted score. Missing context generates a warning.
- Strong market conflict blocks confirmation unless configured exceptional ticker-relative strength exists.

## Options context

- Static: call wall, put wall, gamma flip, gamma regime, magnets, local GEX, and strike congestion when available.
- Live-provider interface: call/put flow acceleration, net delta-weighted premium, gamma-weighted exposure, short-dated ratio, and optional sweep/block flags.
- Execution classification is inferred from bid/ask. Dealer inventory is not observable from OPRA and is never asserted.

## State features

- Bars/setup age, bars in current state, hold/retest/failed-break counts, target failures, extension count, and entry-relative MFE/MAE.
- State and recent histories are persisted so restart does not reset setup age or confirmations.

## Runway formula

The 0–1 runway score returns every component:

| Component | Weight | Interpretation |
|---|---:|---|
| ATR-normalized target distance | 0.25 | More usable movement up to two ATR improves score |
| Intermediate congestion | 0.22 | Sum of obstacle strengths penalizes the path |
| Target strength | 0.13 | A stronger target is harder to traverse |
| Directional trend | 0.18 | Trend aligned with the trade improves score |
| Market/sector alignment | 0.14 | Broad conflict penalizes the path |
| Options structure | 0.08 | A wall before the target penalizes; missing data is neutral 0.5 |

This formula is a versioned hypothesis. Any changed weight/threshold requires a new config version and replay comparison.
