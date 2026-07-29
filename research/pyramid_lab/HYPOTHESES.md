# PYRAMIDING ("add to winners") study — pre-registration

Written 2026-07-27, **BEFORE any pyramiding backtest in this study was run**.
This file fixes the arms, the fill/precedence conventions, the metrics, the
statistic, and the FDR family before any result was looked at. No number
produced by `research/pyramid_lab/` existed when this file was written.

## Why this study exists

`research/portfolio_lab/regime_policy/HYPOTHESES.md` §"UNTESTED — engine does
not support this" recorded that three trading maxims ("only add to winners",
"build positions gradually", "never add to losers") could not be tested,
because **no code path anywhere adds notional to an already-open position**:

* `strategies/momentum_expansion/backtest/family_backtest.py::_simulate_signal`
  resolves one signal to exactly one trade;
* `research/portfolio_lab/portfolio_backtest.py::run_policy` skips a candidate
  whose ticker is already in the book ("mirrors live already_held skip");
* `core/live_4h_exec.py::build_mixed_plan` skips with
  `reason="already_held"` / `"already_held_equity"`;
* `signals/meta_context/meta_ranker/backtest_exits.py::simulate` scans forward
  from `i = exit_bar + 1`, so a ticker can be **re-entered after a full exit**
  (this chained re-entry is how prior work captured the long SNDK/AXTI moves)
  but never **added to while held**.

This study builds that missing capability as a separate, opt-in simulator and
tests it. Chained re-entry after a full exit is the BASELINE behaviour here,
not the treatment. The treatment is adding *without* exiting.

## Signal streams and baseline (fixed for every arm below)

Module coverage matches the prior cross-module exit-policy search
(`scripts/capstone/exit_policy_cross_module.py` ->
`research/capstone/exit_policy_cross_module.csv`): each module is replayed on
**its own** out-of-sample top-10 stream, never on another module's.

| module | score source | span | note |
|---|---|---|---|
| `momentum` | `strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet` (`score`) | 2022-11-14 .. 2026-05-14 | walk-forward OOF of the deployed model |
| `htf` | `strategies/multi_ticker_swing_htf/models/oof_preds.parquet` (`score`) | 2022-11-22 .. 2026-05-22 | walk-forward OOF |
| `meta` | per-timestamp rank-mean of `meta_ranker/models/{quality,upside}/oof_preds.parquet`, via `scripts/capstone/build_meta_scored_from_oof.build_oof_combo_scores` (reused unchanged) | 2024-09-16 .. 2026-05-14 | leak-free `s_combo` substitute for `/tmp/meta_scored.parquet` (leakage_audit.md §4.3); that temp file no longer exists |

Nothing is trained in this study. OOF (not deployed) scores are used for all
three modules for exactly the reason `exit_policy_cross_module.py` documents:
deployed scores partially in-sample the holdout.

**Baseline policy = the current live policy**, i.e.
`core.live_4h_exec.ExecPolicy` defaults, replayed by the same bar-walking
mechanic as `backtest_exits.simulate` / `exit_policy_cross_module.simulate`:

```
take_profit  = 0.30      scale out scale_frac at +30% from entry
scale_frac   = 0.16      trim 16%, ride the rest
stop_loss    = 0.39      full exit at -39% from entry
horizon_bars = 53        full exit after 53 managed bars
trail_stop   = None      off
grace_bars   = None      rank drop-out exit off
target_notional = 5000.0 per NEW entry
MAX_HOLD     = 60        outer cap, from backtest_exits.MAX_HOLD
top_k        = 10
entry        = close of the bar at which the name is observed in the top-10
```

Cost: **10 bps applied to every fill's notional**, i.e. exactly
`portfolio_backtest.run_policy`'s `cost_bps_round_trip = 10.0` convention
(`cost = (bps/1e4) * notional` on the entry leg and again on the exit leg),
generalized to per-fill so that adds and trims are charged on the same basis
as entries and exits. Identical for baseline and every arm.

## Primary grid — 16 arms + baseline

Add sizes are a fraction of the **initial** notional ($5,000), not of the
current position value.

* **Add trigger level** (4 levels):
  * `L10` — gain from ORIGINAL entry >= +10%
  * `L20` — gain from ORIGINAL entry >= +20%
  * `L30` — gain from ORIGINAL entry >= +30%
  * `RESEL` — "re-selection": the name is in the module's top-10 again at a
    later bar while still held
* **Add amount** (2): `50%` and `100%` of the initial notional
* **Max adds per position** (2): `1` and `2`

4 x 2 x 2 = **16 arms**, plus baseline (0 adds).

**Ladder rule for max_adds = 2 on a level trigger:** add *k* (k = 1..M) fires
the first time the intrabar gain from the original entry reaches `k * L`
(e.g. `L10`, M=2 -> adds at +10% and +20%). This is the standard pyramid
ladder and is fixed here, not chosen after seeing results.

**Spacing rule for `RESEL`:** add *k* fires at the k-th bar `j > entry_bar` at
which the ticker is in the top-10, subject to a minimum of **6 bars** (2 bars
per trading day in this dataset, so 3 trading days) since the entry or the
previous add. Without a spacing rule a rank-sticky name would exhaust its adds
on the first two bars after entry, which tests nothing. 6 bars is chosen a
priori for that reason and is not tuned.

## HELD FIXED in the primary grid (stated explicitly)

1. **Stop basis after an add = the ORIGINAL entry price.** The stop, the
   take-profit trim level, the trailing stop (off) and the horizon clock are
   all keyed to the original entry, unchanged by adds. The blended-cost-basis
   alternative is a **secondary sensitivity**, not a primary arm.
2. **The existing trim stays ON** (16% @ +30%), so every arm is a clean delta
   against the live policy.
3. All other `ExecPolicy` params at their live defaults (above).
4. **The horizon clock does NOT reset on an add** — it runs from the ORIGINAL
   entry bar.

### Consequence that must be reported, whatever the result

Because every exit trigger is keyed to the original entry price, adds cannot
change *which* trades happen, *when* they exit, or *why*. Under the primary
grid, pyramiding is a **pure sizing overlay on an identical trade stream**.
Therefore any increase in total P&L that is not matched by an increase in
return per dollar of deployed capital is, by construction, more capital
deployed rather than better trading. This is pre-registered as the study's
central interpretive rule.

## Precedence and edge cases (pre-registered)

Evaluated at each bar `j` after entry, in this order:

1. **Hard stop** (`low[j]/entry - 1 <= -0.39`) -> full exit of ALL lots at
   `entry * (1 - 0.39)`. A stop on bar `j` **pre-empts** any add on bar `j`.
2. **Trailing stop** — disabled by default; if enabled it would ratchet on the
   original-entry-based value. Not exercised in this study.
3. **Take-profit trim** (`high[j]/entry - 1 >= 0.30`, once) -> sell
   `scale_frac = 16%` of the **total shares currently held**, pro-rata across
   all open lots, filled at `entry * (1 + 0.30)` (the same level-fill
   convention the baseline engine already uses for its trim).
4. **Rank drop-out / grace** — disabled by default.
5. **Horizon** (`j - entry_bar >= 53`) -> full exit at `close[j]`.
6. **ADD** — evaluated **LAST**, i.e. only on a bar the position SURVIVES.
   Every exit check above therefore pre-empts the add, and **no lot is ever
   opened on the exit bar** (which would burn a round-trip fee for zero
   exposure). The trim at step 3 still executes BEFORE the add on a surviving
   bar, so a trim and an add on the same bar resolve as "trim the pre-add share
   count, then add". Rationale: risk-reduction and profit-taking take
   precedence over exposure-increase, mirroring the live `exit_action` ladder
   in which stop and take-profit precede everything else.

Additional rules:

* **Add causality.** An add on bar `j` is triggered by `high[j]` (level
  triggers) or by top-10 membership at bar `j` (`RESEL`) — both of which are
  known once bar `j` has closed — and **fills at `close[j]`**, the same
  convention the engine already uses for entries (entry = close of the bar at
  which membership is observed). No bar after `j` is consulted. A unit test
  asserts this by truncating the bar array immediately after the add bar and
  requiring identical add fills.
* **Whole-share rounding is NOT applied** — positions are held in continuous
  share units, matching the baseline `%`-return engine this study must
  reproduce exactly. (`portfolio_backtest.shares_for_notional_min1` rounds;
  the exit-policy engine does not. Parity with the baseline wins.)
* **No slot cap / unconstrained concurrency**, matching fixed-notional policy
  (a) in `portfolio_backtest.py`, which is what live does.
* **No add after the position has been trimmed?** — no such restriction. Adds
  and the trim are independent; both are allowed on the same position.
* **Re-entry after a full exit** is unchanged from baseline (next scan starts
  at `exit_bar + 1`) and is available to every arm equally.

## Metrics

**Primary (both required to claim a win):**

1. **Sharpe** — annualized, from the daily mark-to-market net-P&L series
   (per-bar MTM on open lots + realized events, aggregated across tickers,
   resampled daily). Sharpe is scale-invariant, so it is not inflated by
   deploying more capital.
2. **Return per dollar of average deployed capital** — total net P&L divided
   by the time-average of open cost basis on the 4H bar grid.

**Always reported alongside:** total net P&L, average deployed capital, peak
deployed capital, max concurrent notional, max drawdown (dollars and % of
average deployed capital), win rate, trade count, average hold (bars), and
turnover (total fill notional / average deployed capital).

**Interpretive rule (pre-registered):** any arm that wins on total P&L but not
on return-per-dollar-deployed is deploying more capital, not trading better,
and will be reported in exactly those words.

## Capital-matched check

After the primary grid is scored, the best 2-3 arms by return-per-dollar-
deployed are re-run with the **initial** entry notional scaled down by
`baseline_avg_deployed / arm_avg_deployed`, so the arm's average deployed
capital approximately equals the baseline's. Whether the edge survives that
rescaling is reported either way. Note in advance: because the primary grid's
exits are keyed to the original entry, a uniform rescaling of all lots is
exactly linear in P&L, so this check is expected to be arithmetically trivial
in the primary grid and is reported as such rather than dressed up; it becomes
informative only for the blended-basis sensitivity, where exits do change.

## Statistic and FDR family (fixed before running)

For every (arm x module x walk-forward period): weekly net-dollar P&L is
computed for the arm and for the baseline over the identical calendar weeks;
the statistic is the **mean weekly net-$ P&L difference (arm - baseline)**,
with a 90% CI and a two-sided p-value from
`strategies.momentum_expansion.ablation.bootstrap.week_block_bootstrap_ci`
(reused unchanged) resampling whole weeks with replacement. This is the same
statistic and the same reuse as `regime_policy/engine.weekly_diff_bootstrap`.

**Walk-forward periods:** the repo's one fold spec,
`strategies.momentum_expansion.ablation.folds.build_walk_forward_folds`,
evaluated on the momentum training-matrix date range exactly as
`regime_policy/run_study.py` does (using an OOF stream's own narrower range as
the fold input silently collapses 7 folds into 2 — that trap is already
documented there). Fold test windows:

```
2022-11-14..2023-05-14   2023-05-14..2023-11-14   2023-11-14..2024-05-14
2024-05-14..2024-11-14   2024-11-14..2025-05-14   2025-05-14..2025-11-14
2025-11-14..2026-05-14
```

A module is scored only in periods its own OOF stream covers (meta's stream
starts 2024-09-16, so meta has fewer scorable periods; the exact count is
reported, and thin periods are flagged by trade count rather than dropped
silently).

**BH-FDR** (`ablation.bootstrap.bh_fdr`, re-exported from
`scripts.confluence_discovery.search.bh_fdr`, reused unchanged) at
**q <= 0.10** across the **ENTIRE family**: every (arm x module x period) cell
evaluated. The pre-registered primary family is 16 arms x 3 modules x their
scorable periods. Combinations tried vs. surviving is reported. If any
secondary sensitivity is reported, its cells are added to the family and BOTH
the primary-family and expanded-family FDR results are shown.

## Overfitting discipline

* Repo memory records the frozen test window as substantially spent through
  ~2027 by the prior confluence study. This study is **fold-level replay of an
  already-frozen signal stream**, not fold selection: no model is trained, no
  threshold is fit to these results, and the arms above are final as written.
* If the walk-forward pattern warrants it, **at most ONE** additional held-out
  read is spent, it is declared explicitly as such, and its result is reported
  whether or not it agrees.
* **An arm that wins in 2 of 7 periods with flipping signs is noise** and will
  be reported as noise, not as partial confirmation.
* **A null result is a fully acceptable outcome** and will be reported as such.
  The grid will NOT be re-tuned until something wins.

## Research-only

No live-path file is modified. `_simulate_signal`, `select_signals`,
`BarCache`, `run_policy`, `week_block_bootstrap_ci`, `bh_fdr`,
`build_walk_forward_folds` and `build_oof_combo_scores` are all imported
unchanged. All output lands under `research/pyramid_lab/results/` (a NEW
directory). Nothing is committed to git by this study.
