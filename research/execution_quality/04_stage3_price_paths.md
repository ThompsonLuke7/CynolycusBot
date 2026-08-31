# Stage 3 — price paths and the A/B/C decomposition

Run 2026-08-28. Scripts: `scripts/execution_quality/stage3_fetch_1m.py`, `stage3_metrics.py`
Outputs: `data/bars_1m/` (SIP 1-minute, 473 tickers, 251 MB), `data/stage3_{trade,signal}_metrics.jsonl`

## Data

SIP 1-minute bars, split-adjusted, 2026-07-08 → 08-28, for every traded and every
ranked ticker. Restricted to RTH (13:30–20:00 UTC) for all excursion maths: an
after-hours print on a thin name is a stale print rather than a mark and would
inflate MFE/MAE, and with the extended session included "N bars" stops meaning
"N trading minutes", which the fixed horizons depend on.

Normaliser is daily **ATR(14) as of the session before the decision** — no
look-ahead in the denominator, and a $6 name and a $600 name become comparable.

## Three definitions I had to fix before trusting anything

1. **`entry_slip` was manufacturing zeros.** Where a lifecycle had no plan join I
   had defaulted `available_at = fill_time`, which makes the slip identically 0.
   It is now computed only on the 246 rows with a real availability stamp.
2. **The move-start was unanchored.** A plain Kadane over a 20-day hold finds the
   largest run *anywhere* in the window — often starting days after our entry —
   and reported a median "early by 4 hours" with a p10 of "early by 3 days",
   which is not a statement about this entry. It is now anchored on our own
   trade: take the favourable peak actually reached during the hold, then walk
   back to the extreme immediately preceding it. That is the move we were
   positioned for.
3. **Horizon counting assumed RTH bars.** Fixed by the RTH filter above.

## The decomposition, pooled across 513 lifecycles

| metric | n | p10 | median | p90 |
|---|---|---|---|---|
| **B. entry** | | | | |
| `entry_slip_atr` (fill vs signal-time price) | 246 | −0.11 | **−0.01** | 0.10 |
| `entry_vs_oracle_atr` (fill vs best in the wait window) | 512 | 0.03 | **0.14** | 0.55 |
| `oracle_entry_lag_min` | 512 | 2.0 | 50.3 | 1170 |
| `phase_error_min` (+ = late) | 512 | −1248 | **+62** | +302 |
| `missed_leg_atr` (when late) | 405 | 0.00 | **0.20** | 0.90 |
| `pre_entry_adverse_atr` (when early) | 107 | 0.06 | **0.50** | 2.06 |
| **C. exit** | | | | |
| `mfe_hold_atr` | 353 | 0.02 | 0.35 | 2.91 |
| `mae_hold_atr` | 353 | 0.02 | 0.31 | 1.74 |
| `realized_move_atr` | 353 | −1.07 | **−0.01** | 2.08 |
| `giveback_atr` | 353 | 0.02 | **0.36** | 1.68 |
| `hold_efficiency` (realized / MFE) | 350 | −8.11 | **−0.07** | 0.93 |
| `prematurity_1d_atr` | 359 | 0.03 | 0.20 | 0.84 |
| `prematurity_3d_atr` | 359 | 0.04 | 0.42 | 1.74 |
| `prematurity_10d_atr` | 360 | 0.09 | 0.84 | 3.27 |

### Three things this already says

**The 20–31 minute decision lag costs approximately nothing.** Median
`entry_slip` is −0.01 ATR — we fill a hair *better* than the price at the moment
the signal became available. Whatever is wrong with execution, it is not that the
modules are slow off the bar. That kills the most obvious hypothesis before any
effort goes into shaving minutes off the loop.

**We are late far more often than early, and late is much cheaper — exactly as
predicted.** 405 lifecycles entered after the move began, 107 before it. Being
late costs a median 0.20 ATR of move already spent; being early costs a median
0.50 ATR of drawdown first. The intuition that "late beats early" is now measured
rather than assumed: **early is ~2.5x more expensive per occurrence**, and it is
also the rarer case. This is a policy working roughly as intended.

**The problem is not the entry at all — it is that nothing is kept.** Median MFE
during the hold is 0.35 ATR, and median realized move is −0.01 ATR. Median
giveback is 0.36 ATR and median hold efficiency is −0.07: the typical position
reaches a profit, hands all of it back, and exits slightly negative *on the
underlying*. Meanwhile the median position goes on to make another 0.42 ATR in
the 3 days after we exit, and 0.84 ATR within 10 days.

That is the shape of the whole problem, and it points at the exit, not the model
and not the entry clock. Stage 4 separates it by module and exit reason, because
"the exit is wrong" is not yet actionable — a stop that is too tight and a
take-profit that is too greedy both produce this signature and want opposite fixes.

Caveats carried forward: `realized_move_atr` is the **underlying** move, not
option P&L — for option routes the P&L is levered off this and reported
separately. Median hold spans days for the 4H modules, so a single ATR unit is a
smaller share of the available move for them than for the SPY daytrader; all
per-module comparisons in Stage 4 stay within module.
