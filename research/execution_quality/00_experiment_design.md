# Execution-Quality Experiment — design proposal

Status: **proposal, nothing built yet.** Written 2026-08-28.
Question: *how far is our realized execution from the best available execution, and
is the gap in the signal, the entry policy, or the exit policy?*

---

## 1. The frame

For one trade there are five clocks and one price path:

```
  T_signal        module emits the target (4H bar close, or intraday confirm)
  T_visible       the order policy can act on it
  T_submit        order sent
  T_fill          entry fill
  T_exit_decide   the 5-min risk pass / 4H runner decides to exit
  T_exit_fill     exit fill
```

Against the underlying's dense price path we define an **oracle**: within the
window `[T_signal - 60m, T_exit_fill + H]`, the best long entry is the lowest
price and the best exit is the highest price after it. The realized-vs-oracle gap
decomposes into three additive, separately-fixable terms, all in ATR units so
tickers are comparable:

| Term | Definition (long; mirror for short) | What a bad value indicts |
|---|---|---|
| **A. Signal value** | forward MFE / MAE over horizon H measured **from `T_signal`**, minus a matched control | the model / rule itself |
| **B. Entry cost** | `(P_fill − P_at_signal)/ATR`, plus `phase_error = T_fill − T_move_start` | the order policy |
| **C. Exit cost** | `giveback = (MFE_since_fill − realized)/ATR`, and `prematurity = MFE in H after T_exit_fill` | the exit policy |

Realized capture ≈ A − B − C − friction. Keeping them separate is the whole
point: today a bad trade is un-attributable, and "results aren't consistent
across groups" is what you get when three different failure modes are pooled
into one P&L number.

**`T_move_start` (the thing that answers early-vs-late).** Over log returns in
the window, take the maximum-gain contiguous subinterval (Kadane). Its left edge
is the start of the real move. `phase_error < 0` means we bought before the move
began; `pre_entry_adverse` = MAE between `T_fill` and `T_move_start` prices the
cost of that. `phase_error > 0` means we bought into a move already underway;
`missed_leg` = the run already spent. Your stated preference — late beats early —
becomes a measurable trade: `pre_entry_adverse` vs `missed_leg`, per module.

**Hard rule, inherited from the 2026-07 retraction:** every timing metric is
computed on the **underlying**, never on option prices. Option P&L is reported
alongside as an outcome, never as the timing signal.

---

## 2. What already exists (and what is missing)

Verified in the repo today:

**Have**
- `Data/inference/<module>/live_signal_audit.jsonl` — `signal_decision` and
  `order_plan` events with per-ticker `signal_audit` (score, rank, rank_pct,
  `signal_ts`, and module-specific `extra`: `trigger_rule`, `dollar_vol_pctile_252`,
  `htf_score`/`mom_score`/`news_catalyst_score`, dealer components).
  Momentum alone has 1,726 `signal_decision` events.
- `order_audits` inside `order_plan` — `underlying_price` at order time, strike,
  premium, delta, **dte**, `breakeven_move_pct`. This is the price reference at
  `T_submit`.
- `Data/inference/<module>/closed_trades.jsonl` — exit wall-clock `ts`, `bar`,
  `entry_avg_price`, `exit_fill_price`, `realized_pnl`, `exit_reason`,
  `entry_bar`, `runs_held`, exit `order_id`, `decision_gain`, `stop_overshoot`.
- `Data/inference/intraday_structure/` — the richest source by far:
  `closed_setups.jsonl` carries `candidate_available_at`, `confirmed_at`,
  `entry_time`, `exit_time`, `mfe_points`, `mae_points`, `risk_points`,
  costs, evidence list, runway/dealer components. `decision_events.jsonl`
  (13,140 rows) already logs `candidate_fixed_horizon_outcome` with 5/15/30/60m
  MFE/MAE **and** `candidate_availability_lag_seconds`.
- `core/API/Alpaca_API/market_data/fetch_intraday.py` — arbitrary symbol/range
  1Min bars, paginated. This is the price-path source.
- `core/API/Alpaca_API/runners/decision_latency.py` — the same lag decomposition
  already built for the SPY path; reuse its vocabulary.
- `Data/shared/market_regime/`, `dollar_vol_pctile_252` in the audits, theme
  membership — the grouping dimensions.

**Missing — these are the Stage-0 blockers, not details**
1. **No entry-fill timestamp or entry order id is persisted anywhere.**
   `order_plan` lines carry no wall clock, and `closed_trades.entry_bar` is the
   *signal bar*, not the fill. `T_submit`/`T_fill` for entries must be recovered
   from Alpaca `GET /v2/orders?status=all&after=…` (`submitted_at`, `filled_at`,
   `filled_avg_price`) and matched by symbol+qty+day, or the historical arm
   collapses to bar resolution. Alpaca order-history retention for this paper
   account is unverified.
2. **`entry_bar` is null on 20–33% of closed trades** (momentum 9/34, HTF 20/61,
   Meta 7/48, dealer 10/40). Cause unknown; must be established before analysis,
   not imputed.
3. **1-minute bar coverage for the actual traded names is unverified.** The
   traded set includes microcaps (LASE, PURR, INHD, CLYM, RFIL). On the IEX feed
   these may be sparse, and a sparse tape makes MFE/MAE understate. Falls under
   the "trade prints vs marks" rule.
4. **Meta Ranker has `realized_pnl` on only 23 of 48 closes** (19 `horizon`
   rows with `order_id: "?"`). Those rows are not evidence of anything and must
   be excluded, not read as breakeven — consistent with the 2026-08-27 finding.

---

## 3. Sample sizes — the binding constraint, and the way around it

| Module | closed trades | window |
|---|---|---|
| momentum_expansion | 34 | 07-23 → 08-28 |
| multi_ticker_swing_htf | 61 | 07-09 → 08-27 |
| meta_ranker | 48 (23 with P&L) | 07-09 → 08-27 |
| dealer_ranker | 40 | 07-21 → 08-27 |
| intraday_structure | 141 setups (paper, no broker fills) | 08-06 → 08-28 |

Per-module, per-cell inference on n≈40 is not going to produce a stable answer —
that is almost certainly *why* the existing group breakdowns disagree with each
other. Two moves fix this:

- **Execution questions pool across modules.** Entry lag and exit giveback are
  properties of the shared order policy, not of a module. Pooled n ≈ 183 fills,
  with module as a covariate rather than a separate experiment.
- **Signal questions use signal rows, not fill rows.** Every ranked target in
  every `signal_decision` is a forward-return observation, whether or not it was
  traded. That is thousands of rows, already logged with score and rank, and it
  is the correct sample for "do the module signals even make sense" — it is also
  free of the selection effect that the order policy imposes on the traded subset.

The traded subset then answers a different, narrower question: *given a signal
we acted on, what did the policy cost us?*

---

## 4. Stages

### Stage 0 — feasibility, before any pipeline (½ day)
Standalone checks, each of which can kill or reshape the study:
- Pull Alpaca order history for 07-01 → today; count entry fills recoverable and
  matchable to ledger rows. Report the match rate.
- Fetch 1m bars for 20 representative traded tickers across their trade windows;
  report bars-per-session, gap share, and the fraction of sessions with < 200
  bars. Compare IEX vs SIP on the same names to confirm the entitlement in use.
- Diagnose null `entry_bar`.
- **Gate:** if entry fills are unrecoverable *and* 1m coverage is poor for
  microcaps, the historical arm is reduced to a 4H/1H-resolution study and the
  precise arm becomes forward-only. Say so before building, not after.

### Stage 1 — trade spine
One row per closed trade: all clocks, all prices, module, route, exit_reason,
option metadata (dte, delta, breakeven_move_pct), and the full `signal_audit`
joined from the `order_plan`/`signal_decision` line at the same bar.

### Stage 2 — signal spine
One row per ranked target per decision event, traded or not, with score, rank_pct,
buckets, `trigger_rule`, `extra`, and a `was_traded` flag. This is the large-n table.

### Stage 3 — price paths and metrics
Cached 1m (fallback 5m) underlying bars per (ticker, window). Compute A/B/C,
`phase_error`, `pre_entry_adverse`, `missed_leg`, `giveback`, `prematurity`,
`hold_efficiency = realized/MFE`, `time_to_peak`, all ATR-normalized.
Horizons H ∈ {30m, 1d, 3d, 10d, 20d} so the 4H and intraday modules are on one
scale. Cache is immutable and keyed by content; re-runs must be reproducible.

### Stage 4 — the three analyses
- **A. Signal quality vs control.** For each module, forward MFE/MAE at signal
  against a matched control drawn from the same universe, same session, same
  time-of-day, same liquidity decile — because in a bull tape everything drifts
  up and an uncontrolled "our signals go up" claim is worthless. Then: does
  score/rank monotonically order forward MFE? *If it does not, retraining is the
  wrong fix and the entry trigger is doing all the work.* This is the ML-vs-rules
  question, answered directly.
- **B. Entry.** Distribution of `phase_error` and `entry_slip_vs_signal_atr`,
  pooled and by module; the early-cost vs late-cost trade; and a **counterfactual
  delay grid** — re-price every entry at signal+{0,5,15,30,60,120} min and at
  "first close above the signal bar high", using the same spine. This turns
  "should the order policy wait?" into a measured curve rather than an opinion.
- **C. Exit.** `giveback` vs `prematurity` by `exit_reason`; the 5-min risk-pass
  cadence effect (does a stop now fire at a systematically different point in the
  path than it did at 4H sampling?); and time-to-peak, which sets whether the
  horizon exit is even in the right neighbourhood.

### Stage 5 — pre-registered decision rules
Written **before** Stage 4 is run, so the study cannot be talked into a
conclusion afterwards:
- Signals rank forward MFE no better than control → **do not retrain**; the
  ranking is not the constraint. Invest in the trigger and the exit instead.
- Signals rank forward MFE, but our fills sit late in the run (`missed_leg`
  large) → the model is fine and the 4H cadence is the constraint → route those
  names through the intraday engine for timing.
- Median `phase_error < 0` with material `pre_entry_adverse` → add a confirmation
  delay; size it from the Stage-4 delay curve, not by intuition.
- `prematurity ≫ giveback` → exits too tight. Reverse → too loose. Per module,
  since they hold different things.
- Every proposed change is validated on a **held-out window** before it goes
  live. Freeze 08-15 → present as the holdout now; fit on everything before it.

---

## 5. Two things the existing data already says

Both computed today from ledgers on disk, both **preliminary and flagged for
Stage-0 verification**, not conclusions:

**(a) 4H candidates reach the intraday engine roughly 11 hours stale.**
`candidate_availability_lag_seconds`, median minutes by source:

```
high_liquidity_universe  n=400  med    0.0
opening_momentum         n=116  med   32.7   p90  83.1
meta_ranker              n= 47  med  652.0   p10 260.2
30m_swing                n= 47  med    0.0
validated_catalyst       n= 39  med   16.7
4h_swing                 n= 18  med  652.0   p10 265.3
```

If real, the intraday engine is being handed 4H ideas after the move they were
named for. The 652-minute figure clusters suspiciously tightly and may be a
timestamp-convention artifact (previous session's 14:00Z bar vs load time) —
which is exactly why it is a Stage-0 check and not a finding.

**(b) The intraday_structure setups die almost immediately.** Across 141 closed
setups: median hold **2 minutes** (p75 = 9), confirm→entry 1 minute, MFE 0.26R vs
MAE 0.38R, and 61% close at or below zero gross. Adverse excursion exceeding
favorable excursion at this hold length is the signature of an invalidation set
inside the noise band, not of a bad direction call. Thresholds there are
explicitly uncalibrated (`docs/PROJECT_STATUS.md`), and this is paper/shadow with
no broker fills — but it means the GEX/intraday upgrades cannot yet be credited
with better timing, and the invalidation width is the first thing the study
should measure against 1m bars.

---

## 6. Scope, cost, and what this does not do

Roughly: Stage 0 half a day, Stages 1–3 a day, Stage 4 a day, Stage 5 written
first. Read-only throughout — no live change, no order action, no retraining.

Deliberately **out of scope**: any option-price-based timing metric; any live
policy change made off this study without a holdout check; retraining. The
retrain decision is an *output* of Analysis A, not an input — if the signals
already rank forward moves and we are simply arriving late, new data will not
help, and if they do not rank, retraining on the same features probably will not
either. That question gets answered before the GPU is booked.

**Instrumentation ask (small, and worth doing regardless):** persist entry
`submitted_at`, `filled_at`, `filled_avg_price`, and the entry `order_id` on the
order-plan audit line. It is a few fields, it removes the single largest blind
spot in this study, and every future execution question needs it.
