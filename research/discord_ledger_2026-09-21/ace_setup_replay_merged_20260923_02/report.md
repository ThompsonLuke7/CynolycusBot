# ACE source-tagged underlying outcomes and Intraday Structure replay

Research only. The underlying analysis uses retrospective IEX trade bars. Intraday Structure received each ACE alert as an already-selected candidate; therefore replay measures confirmation/management compatibility, not independent universe discovery.

## All alerts vs matched controls

| Horizon | Alert n | Alert mean (%) | Control mean (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| 120m | 74 | -0.0790 | 0.0653 | -0.1443 |
| 15m | 113 | -0.0385 | -0.0075 | -0.0310 |
| 30m | 105 | -0.0538 | -0.0033 | -0.0504 |
| 5m | 119 | -0.0162 | 0.0021 | -0.0184 |
| 60m | 93 | 0.0381 | 0.0136 | 0.0245 |
| session_close | 120 | -0.0432 | 0.1513 | -0.1945 |

## Intraday Structure compatibility

Complete isolated replays: 120; detected: 96.7%; modeled share trade: 80.8%.
Detected setup families: `{'breakout_continuation': 116, 'gamma_structural_level_rejection': 98, 'trend_pullback_continuation': 115, 'vwap_reclaim_continuation': 111, 'v_shaped_capitulation_reversal': 2}`.

## Interpretation boundary

A detector overlap is not validation: ACE had already chosen both the ticker and direction. The replay does not include historical dealer context, full market/sector context where absent from the cached session file, or option execution. Negative matched-control underlying results mean this data does not yet support treating raw alert occurrence as a standalone directional entry policy.
