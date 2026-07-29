# Regime-conditional POLICY study — pre-registration

Written 2026-07-27, BEFORE any regime-conditional policy backtest in this
study is run. This file fixes the rules, thresholds, statistic, and FDR
family before looking at results. The two motivating priors quoted in the
task brief (sector_dispersion_21d top-vs-bottom-bucket spread positive 7/7
walk-forward periods; ret_20 rank IC flips sign with risk_appetite_z) were
already on record in `strategies/momentum_expansion/ablation/results/
group1_market_wide_spread.csv` and `LIVING_SUMMARY.md` before this study
started — they are the reason this study exists, not a result discovered by
peeking at the new backtests below. No new-code result was inspected before
this file was written.

## Question

Prior work asked whether regime features improve the cross-sectional
RANKER. This study asks whether they should condition POLICY instead:
whether to trade, how many names, how big, how tight the stop — holding the
signal (top-K selection from the deployed momentum model's real leak-free
walk-forward OOF scores) completely fixed.

## Signal stream and baseline (fixed for every rule below)

- Signal source: `strategies/momentum_expansion/models/expansion_v1/
  oof_preds.parquet` (2022-11-14 .. 2026-05-14, 1,582,685 rows, real
  walk-forward out-of-fold scores of the deployed xgb_classifier seed45
  winner — not retrained here). Long-only (`allow_short=False`, matches the
  live momentum module).
- Selection: per-bar top-K by `score`. Baseline K=10 (matches
  `run_comparison.py`'s documented top-10 comparison point).
- Execution: unchanged `family_backtest._simulate_signal` — entry at next 4H
  bar open, ATR take-profit / stop-loss / time-stop exit. Baseline
  `tp_atr_mult=5.0`, `sl_atr_mult=5.0`, `max_hold=75` bars (same values
  `run_comparison.py` already uses for this comparison point).
- Sizing: fixed $5,000 notional per entry (mirrors
  `core.live_4h_exec.ExecPolicy.target_notional`), unconstrained
  concurrency — i.e. policy (a) from `research/portfolio_lab/
  portfolio_backtest.py`, unchanged. This is "current policy" for this
  study.
- Cost: 10 bps round-trip (matches `run_comparison.py`).
- Regime join: `signals.market_regime` `daily_regime.parquet`, joined
  causally onto each signal's own `signal_ts` via the existing
  `feature_matrix_4h._asof_join_regime` backward as-of join on
  `available_at` (reused unchanged, not reimplemented). A decision at bar
  `t` only ever sees a regime row whose `available_at <= t`.
- Walk-forward folds: `strategies.momentum_expansion.ablation.folds
  .build_walk_forward_folds` (reused unchanged) — the same 7 six-month test
  windows (2022-11-14 .. 2026-05-14) the prior regime screen used. Rules are
  selected/reported on these 7 periods; see "Overfitting discipline" below
  for how many held-out reads this spends.

## Statistic and FDR family (fixed before running)

For every rule variant, in every walk-forward test period: compute the
admitted-trade stream (baseline's fixed policy vs. the rule's modified
admission/sizing/stop), resample calendar weeks of `entry_ts` with
replacement (`ablation.bootstrap.week_block_bootstrap_ci`, reused unchanged)
on the **weekly net-dollar-P&L difference, rule minus baseline** (same
trades, same weeks — only which trades are admitted, how big, or where the
stop sits differs). This one statistic is used for every rule type (pure
admission gates, pure sizing scalers, and stop-distance changes) because it
answers the same underlying question in every case: does this policy change
make or lose money, in the units that matter, on the identical calendar
clock. 90% CI, two-sided bootstrap p-value, reported per rule per period.

BH-FDR (`scripts.confluence_discovery.search.bh_fdr`, reused unchanged) at
q <= 0.10 is applied across **every (rule variant x period) cell below** —
14 rule variants x 7 periods = **98 tests** in one family, not one test per
rule. This is a policy search over many thresholds on data noted in repo
memory as "burned" through ~2027 by a prior confluence study; the FDR
correction must see the whole search or it will manufacture winners.

## H1 — "trade less" / "when in doubt, stay out" (tips 1, 9)

Suspend new entries entirely when a stress regime feature is elevated at
signal time. Absolute z-score thresholds are used (not empirical quantiles
of this sample) specifically to avoid fitting a threshold to the same data
being tested — the regime composites are already self-normalizing trailing
z-scores, so a fixed z cutoff is a regime-agnostic rule, not a tuned one.

| id | rule |
|---|---|
| H1-liq-1.0 | skip entry if `liquidity_stress_z > 1.0` |
| H1-liq-1.5 | skip entry if `liquidity_stress_z > 1.5` |
| H1-rv-1.0  | skip entry if `spy_rv20_z > 1.0` |
| H1-rv-1.5  | skip entry if `spy_rv20_z > 1.5` |

## H2 — "size small" / "avoid leverage" / "cash is your friend" (tips 2, 3, 5)

Position size scales with regime; shrunk capital is left undeployed (never
redeployed into more names), consistent with "cash is your friend" as a
literal reduced-exposure rule rather than a reallocation rule.

| id | rule |
|---|---|
| H2-liq  | `size_mult = clip(1 - 0.5*liquidity_stress_z, 0.25, 1.5)` |
| H2-risk | `size_mult = clip(1 + 0.5*risk_appetite_z, 0.25, 1.75)` |

Baseline for both: constant `size_mult = 1.0`.

## H3 — "respect the trend" / "don't try to time the bottom" (tips 4, 15)

Gates long entries on trend/risk-appetite state. H3-riskapp is the direct
test of the momentum-inversion prior (`ret_20` IC flips sign with
`risk_appetite_z`): the rule under test is "don't take momentum longs when
risk appetite is negative."

| id | rule |
|---|---|
| H3-trend   | admit only if `spy_trend_state > 0` |
| H3-riskapp | admit only if `risk_appetite_z > 0` |
| H3-combo   | admit only if both hold |

## H4 — "small losses are the best losses" (tip 14)

Tighter stop distance (smaller ATR multiple) in high-volatility regimes,
vs. the fixed baseline `sl_atr_mult=5.0`. This changes trade RESOLUTION
(the exit itself), not just sizing — `_simulate_signal` is re-run per rule
with the row's regime-conditional `sl_atr_mult`, entry/TP logic unchanged.

| id | rule |
|---|---|
| H4-step-1.0 | `sl_mult = 3.0` if `spy_rv20_z > 1.0` else `5.0` |
| H4-step-0.5 | `sl_mult = 3.0` if `spy_rv20_z > 0.5` else `5.0` |
| H4-cont     | `sl_mult = clip(5.0 - 1.5*max(spy_rv20_z, 0), 2.0, 5.0)` |

## H5 — "lean in when dispersion is wide" (from the sector_dispersion_21d prior)

| id | rule |
|---|---|
| H5-size    | `size_mult = 1.5` if `sector_dispersion_z > 1.0` else `1.0` |
| H5-breadth | admit ranks 11-15 (in addition to the baseline top-10) only when `sector_dispersion_z > 1.0`; ranks 11-15 are otherwise never traded |

## Explicitly NOT testable propositions (tips 6, 7, 8, 12)

Reported as such, not force-fit to a parameter:

- **Tip 6 (FOMO)** — a psychological-discipline instruction about the
  trader, not a parameterizable policy rule over this signal stream.
- **Tip 7 (respect risk)** — a restatement of "use stops / size for risk",
  which this study already tests concretely as H2/H4; as a standalone
  maxim it has no independent parameter left to test.
- **Tip 8 (patience)** — discipline/behavioral, not a rule with a
  threshold; conflating it with `max_hold` would silently invent a test the
  tip doesn't actually specify.
- **Tip 12 (mental capital)** — psychological/behavioral, not a market
  observable in this data.

## UNTESTED — engine does not support this (tips 10, 11, 13)

"Only add to winners" / "build positions gradually" / "never add to
losers" all require **scaling into an existing position** (pyramiding /
multi-entry). Checked before writing any rule: `family_backtest.py`'s
`_simulate_signal` resolves one signal to exactly one trade;
`portfolio_backtest.py`'s `run_policy` explicitly skips a signal if the
ticker is `already_held` ("mirrors live `already_held` skip"); the same
skip exists in `core/live_4h_exec.py` (`skip: "already_held"`, three call
sites). There is no code path anywhere in the replay engine or the live
execution path that adds notional to an already-open position. These three
tips are marked **UNTESTED** rather than approximated (e.g. by reusing
`size_mult` as a proxy for "conviction to add") because that would silently
test a different, weaker claim than the one the tip makes.

## Overfitting discipline

- All 98 (rule x period) cells above are evaluated on the SAME 7
  walk-forward periods as the prior regime screen — no new held-out data is
  created or consumed by this step; this is fold-level replay, not fold
  selection.
- No thresholds are tuned after seeing results. The thresholds above are
  final as written in this file.
- If, after running, any rule's cross-period pattern looks promising enough
  to warrant one additional held-out check, this study will spend **at
  most one** such read, state explicitly that it is doing so, and report
  the result whether or not it agrees with the walk-forward pattern. It
  will not spend more than one.
- A rule winning in 2 of 7 periods with flipping signs is noise and will be
  reported as noise, not as a partial confirmation.
- A null result (regime-conditional policy does not beat the constant
  baseline) is an acceptable, fully reportable outcome.
