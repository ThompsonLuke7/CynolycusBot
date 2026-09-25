# ACE source-tagged setup and Intraday Structure replay

Research only. Underlying returns are based on retrospective IEX trade bars; no option prices, fills, or P&L are manufactured.

## All source alerts vs matched controls

| Horizon | Alert n | Alert mean (%) | Control mean (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| 5m | 119 | -0.0162 | 0.0021 | -0.0184 |
| 15m | 113 | -0.0385 | -0.0075 | -0.0310 |
| 30m | 105 | -0.0538 | -0.0033 | -0.0504 |
| 60m | 93 | 0.0381 | 0.0136 | 0.0245 |
| 120m | 74 | -0.0790 | 0.0653 | -0.1443 |
| session_close | 120 | -0.0432 | 0.1513 | -0.1945 |

Intraday Structure replay was intentionally skipped for this outcome-only run.
