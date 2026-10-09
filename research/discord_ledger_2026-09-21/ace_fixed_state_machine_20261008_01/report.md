# ACE fixed state-machine experiment

Research-only, selection-conditioned test of the screenshot-derived daily-level/pullback/confirmation hypothesis. It contains **no fitted model, grid search, option repricing, or trade execution**. Its candidate rows already contain ACE's historical symbol and direction, so it does not recreate candidate selection.

## Point-in-time design

- Daily SMA5, EMA21, SMA34, SMA55 and ATR are calculated only from ET sessions strictly before each decision date. The decision day's daily bar is excluded.
- The hypothesis uses an ordinary 20-day-extreme pullback proxy. It does not claim to reproduce ACE's private Cycle/Due low/D EM or divergence indicators.
- Observed alerts are compared with frozen, same-symbol/session/direction no-nearby-alert candidates. Those controls are unlabeled—not proven bad trades.
- Rules were fixed from the pasted screenshot interpretation before this run and were not outcome-fitted. The hypothesis itself is still exploratory because it was formed after viewing Discord material.

## Daily-data coverage

Rows with missing daily bars or insufficient prior daily history are retained in the candidate file and fail all downstream states; they are not replaced or silently discarded.

| Partition | Daily feature status | Rows |
| --- | --- | ---: |
| train | ok | 593 |
| validation | missing_daily_bars | 12 |
| validation | ok | 486 |
| test | ok | 260 |

## Fixed state rules

- `candidate`: at least 55 completed prior ET daily sessions and valid ATR.
- `trending`: direction-aware SMA5/EMA21/SMA55 ordering diagnostic.
- `pullback_due_state`: trending and 0.25 to 5 ATR retracement from directionally relevant prior 20-day extreme.
- `approaching_level`: pullback state and nearest of SMA5/EMA21/SMA34/SMA55 within 0.75 ATR.
- `at_level`: pullback state and nearest listed daily average within 0.35 ATR.
- `confirmed`: at-level plus favorable completed 5-minute return and VWAP position.
- `trade_hypothesis`: confirmed plus short-volume expansion; it is not an order instruction.

## Held-out state reproduction

The test partition is chronological. Positive pass-rate lift means this state occurred more often at an observed alert than in its matched quiet-window controls; it does not establish profitability.

| State | Groups | Alert pass | Control pass | Difference (pp) | 95% CI (pp) | MDE (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate | 25 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 0.000 |
| trending | 25 | 0.480 | 0.480 | 0.000 | [0.000, 0.000] | 0.000 |
| pullback_due_state | 25 | 0.400 | 0.450 | -5.010 | [-14.465, 0.889] | 11.896 |
| approaching_level | 25 | 0.200 | 0.276 | -7.556 | [-18.020, 1.535] | 14.382 |
| at_level | 25 | 0.080 | 0.127 | -4.727 | [-11.273, 0.000] | 8.203 |
| confirmed | 25 | 0.000 | 0.030 | -2.982 | [-6.255, -0.364] | 4.179 |
| trade_hypothesis | 25 | 0.000 | 0.011 | -1.091 | [-2.909, 0.000] | 2.240 |

## Held-out underlying returns when both sides passed

These are a sparse diagnostic only: a group contributes only when the alert and at least one matching control both pass the named state. Returns are directional underlying returns, not option P&L. An MDE larger than the observed difference means the sample cannot measure a plausible effect.

### `at_level`

| Horizon | Paired groups | Alert mean (%) | Control mean (%) | Difference (%) | 95% CI (%) | MDE (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15m | 2 | -0.229 | 0.000 | -0.229 | [-0.420, -0.038] | 0.536 |
| 30m | 2 | -0.290 | -0.015 | -0.275 | [-0.563, 0.013] | 0.806 |
| 60m | 1 | 0.088 | 0.010 | 0.078 | [0.078, 0.078] | — |

### `confirmed`

| Horizon | Paired groups | Alert mean (%) | Control mean (%) | Difference (%) | 95% CI (%) | MDE (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15m | 0 | — | — | — | — | — |
| 30m | 0 | — | — | — | — | — |
| 60m | 0 | — | — | — | — | — |

### `trade_hypothesis`

| Horizon | Paired groups | Alert mean (%) | Control mean (%) | Difference (%) | 95% CI (%) | MDE (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15m | 0 | — | — | — | — | — |
| 30m | 0 | — | — | — | — | — |
| 60m | 0 | — | — | — | — | — |

## Interpretation boundary

A positive timing association would only support this small state proxy inside the already-known candidate panel. A null with a large MDE is ‘not measurable here,’ not proof that daily levels do not matter. The next testable variant is to collect actual ACE level/cycle screenshots or structured values, then prospectively score an immutable, independently generated candidate tape across both directions before any integration into a live strategy.
