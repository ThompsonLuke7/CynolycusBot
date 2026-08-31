# Stage 5 — pre-registered decision rules

**Written 2026-08-28, BEFORE Analyses A/B/C were run.** The point is that the
study cannot be talked into a conclusion after the fact. Repo precedent for why
this matters: the confluence-discovery null result and the retracted options
study, where two intermediate corrections changed magnitudes but not signs and
made a wrong answer look robust.

Full disclosure of what was already visible when this was written: the **pooled**
Stage 3 table (median `entry_slip` −0.01 ATR, 405 late vs 107 early, median
giveback 0.36 ATR against median realized −0.01 ATR). Nothing per-module, nothing
per-exit-reason, no control comparison, no delay curve. The rules below are
therefore honest for every per-module and per-mechanism claim, and only partly
blind for the pooled direction, which is stated rather than hidden.

## Split

Explore on fills before **2026-08-15**; verify on **2026-08-15 → 08-28**.

| | explore | holdout |
|---|---|---|
| closed lifecycles with giveback | 285 | 68 |
| signal rows with a forward path | 1,417 | 422 |

The holdout is small for per-module work (7–25 rows per module). It is therefore
used **only** to check the sign and rough magnitude of a pooled effect, never to
certify a per-module parameter. Any rule that needs per-module holdout evidence
is not decidable from this sample and will be labelled as such rather than
asserted.

## A. Does the ranking carry information?

**Test.** Forward MFE/MAE at 1d/3d/10d from `available_at`, by score/rank decile,
per module, against a **matched control**: same module, same decision bar, same
liquidity decile, drawn from tickers the module ranked but placed lower — plus an
absolute check against the universe drift over the same windows. In a tape that
drifts up, "our picks went up" is not evidence.

- **A1** If forward MFE does not increase monotonically with rank, and the top
  decile is statistically indistinguishable from the control →
  **the ranking is not the constraint. Do not retrain.** Effort goes to the
  trigger and the exit. Retraining on the same features would be re-fitting a
  map that already does not predict.
- **A2** If rank does order forward MFE but our fills land late in the run
  (`missed_leg_atr` in the top tercile) → the model is fine and the **4H cadence**
  is the constraint. Route those names through the intraday engine for timing.
- **A3** If rank orders forward MFE and entries are well-placed → the signal and
  entry are both fine and the exit owns the entire gap. Go to C.

## B. Entry policy

- **B1** Median `phase_error < 0` with material `pre_entry_adverse` →
  systematically early; add a confirmation delay, sized from the Stage-4B delay
  curve rather than by intuition.
- **B2** Median `phase_error > 0` and `pre_entry_adverse` (when early) exceeding
  `missed_leg` (when late) → the current bias is **correct**; do not add urgency.
  Recommend *no change* and say so explicitly, so "improve the entry" does not
  become a default work item.
- **B3** The counterfactual delay grid re-prices every entry at
  availability + {0, 5, 15, 30, 60, 120} minutes and at "first close above the
  signal bar's high". A delay is recommended only if it improves median
  `entry_vs_oracle_atr` **and** holds its sign in the holdout.
- **B4** The 8.5% of entries deferred overnight (afternoon 18:00Z bar → next
  open) are evaluated separately. If their `entry_vs_oracle` and forward capture
  are materially worse than same-session entries → recommend the afternoon bar
  stop opening new positions, rather than trying to fill them faster, since the
  session it would need has already ended.

## C. Exit policy

- **C1** `prematurity ≫ giveback`, per module → exits are too tight; widen.
- **C2** `giveback ≫ prematurity` → too loose; tighten or trail.
- **C3** Both large → the exit is firing at the wrong *event*, not the wrong
  *level*; that is a rule-shape problem and no threshold change fixes it.
- **C4** Split by `exit_reason`. A stop and a take-profit producing the same
  signature want opposite fixes, so no exit recommendation is made on the pooled
  number.
- **C5** For the intraday engine specifically, compare invalidation width to the
  1-minute noise band (MAE distribution of setups that ultimately worked). If the
  invalidation sits inside that band → it is being triggered by noise and the
  width, not the direction call, is the defect.

## Global stopping rules

- No recommendation is made on **n < 20** in a cell. It is reported as
  underpowered instead.
- Any effect that reverses sign between explore and holdout is reported as
  **not established**, whatever its explore-set p-value.
- Nothing here authorises a live change. Every rule's output is a proposal that
  still needs its own validation run.
- Option P&L is never used as a timing metric (retraction rule). Underlying only.
