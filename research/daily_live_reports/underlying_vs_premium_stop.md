# Option stops: premium-referenced vs underlying-referenced

**Date:** 2026-08-18 · **Status:** evidence for a change now wired live, NOT yet paper-validated
**Reproduce:** `PYTHONPATH=. .venv/bin/python scripts/analyze_underlying_vs_premium_stop.py`
**Per-trade table:** `underlying_vs_premium_stop.csv`

## Question

`ExecPolicy.stop_loss = 0.39` is a **premium** stop: exit when the contract's mark is
39% below entry. It was selected on a **shares-only** backtest
(`research/capstone/exit_policy_cross_module.csv`, 2026-07-18) whose own docstring
warns "no option-premium path modeled". Applied to option premium, a -39% stop is
roughly a -13% move in the underlying — well inside the noise band of the names
these modules trade. Is it firing on noise rather than on broken theses?

## Data

Every option full-exit stop in `Data/inference/*/closed_trades.jsonl` carrying an
`entry_bar`: **56 stops**, of which **42** resolve against cached 4H underlying bars
(`Data/shared/bars/4h`). Exits span **2026-07-17 → 2026-08-18**, entries
**2026-07-01 → 2026-08-17**, across all four 4H modules. All 42 are long calls
(no puts in the sample).

Entry/exit premia are **broker fill prices** with order IDs, not marks — so the
premium moves are real. This is *not* the stale-print trap that invalidated the
2026-07 options-routing study.

## Finding 1 — the stop fires while the underlying is basically flat

Underlying move at the moment the premium stop fired:

| stat | value |
|---|---|
| median | **-3.09%** |
| mean | -3.88% |
| 25th pct | -7.42% |
| 75th pct | **+0.22%** |
| max | +11.10% |

| underlying was down less than | trades | share | mean premium ret | realized |
|---|---|---|---|---|
| 2% | 18 / 42 | 43% | -59.3% | **-$43,181** |
| 3% | 21 / 42 | 50% | -58.3% | -$50,408 |
| 5% | 25 / 42 | 60% | -60.6% | -$62,933 |
| 8% | 34 / 42 | 81% | -62.2% | -$91,764 |

A quarter of these stops fired with the underlying **up** on the trade.

## Finding 2 — a 1.5×ATR underlying stop would not have fired on 18 of 42

Anchoring at the entry bar's close and ATR(14) on 4H:

- **24 of 42** — underlying genuinely broke 1.5 ATR. The stop was right; both rules agree.
- **18 of 42** — underlying never broke. **-$43,944 realized on pure premium noise.**
  - **13 of those 18 (72%)** saw the underlying trade back **above its entry price** within 40 4H bars (~16 sessions).
  - Mean maximum forward excursion after the stop: **+14.9%**.

## Finding 3 — the specific cases

Stops that fired while the underlying was flat-to-up (**14 trades, -$31,701**):

| module | ticker | entry | exit | underlying at stop | worst dip | premium | realized | underlying after |
|---|---|---|---|---|---|---|---|---|
| momentum | AAOI | 08-10 | 08-14 | **+11.10%** | -3.0% | +7.6% | +$350 | +22.0% |
| dealer | VIAV | 08-06 | 08-11 | **+10.09%** | -5.8% | -61.8% | -$2,730 | +19.1% |
| htf | SLS | 08-10 | 08-11 | **+6.59%** | -2.5% | -45.0% | -$2,205 | +21.7% |
| meta | CRDO | 08-12 | 08-14 | +5.97% | -5.5% | -66.4% | -$2,920 | +6.7% |
| momentum | NEXA | 08-05 | 08-07 | +4.29% | -5.7% | -41.2% | -$1,995 | +5.6% |
| dealer | NTNX | 08-11 | 08-12 | +2.04% | **-0.1%** | -52.0% | -$2,600 | +5.9% |
| dealer | SMTC | 08-06 | 08-11 | +1.93% | -6.7% | -61.3% | -$2,850 | +11.8% |
| momentum | OUST | 08-06 | 08-07 | +0.98% | -0.8% | -51.9% | -$1,820 | **+23.5%** |
| dealer | EW | 08-14 | 08-18 | +0.86% | **-0.1%** | -92.2% | -$4,465 | — |
| dealer | TEL | 08-12 | 08-13 | +0.33% | -0.6% | -64.3% | -$3,150 | +1.4% |
| dealer | PSKY | 07-17 | 07-21 | +0.29% | -3.6% | -54.8% | -$23 | +19.0% |
| dealer | **LITE** | 08-12 | 08-13 | **+0.04%** | -3.8% | **-97.1%** | **-$4,950** | +9.3% |
| htf | COHR | 08-10 | 08-13 | -0.34% | -1.7% | -44.1% | -$2,250 | +11.0% |
| dealer | **FIG** | 07-21 | 07-23 | -0.75% | -6.1% | **-97.9%** | -$93 | **+42.2%** |

The two clearest: **LITE** lost 97% of premium with its underlying **+0.04%** —
that is spread and theta, not a thesis failure. **NTNX** and **EW** stopped out
with maximum underlying dips of **-0.1%**.

**FIG** is the compounding case: stopped 07-23 at -97.9% premium with the
underlying down 0.75%; the underlying then ran +42%. The module re-entered and
that position became the single best closed round trip in the book, **+$6,267**.
The stop did not avoid a loss — it paid for the same move twice.

## Finding 4 — premium is a weak proxy for the thesis

Across the 42, corr(premium return, underlying return) = **+0.32**
(and +0.22 against the underlying's worst dip). Both are attenuated because the
premium series is truncated at the stop threshold, but the signs are right, so
these are real derivative relationships — the point is the *weakness*. Where the
underlying did fall, the median premium move was **12.3×** the underlying move.
That leverage is the intended exposure; measuring the *stop* on it is not.

## What was changed

`ExecPolicy.underlying_stop_atr = 1.5` (options only). An option's hard stop is now
`entry_underlying - 1.5 × entry_ATR(14, 4H)`. Equity is untouched. Any option whose
underlying basis cannot be established keeps the premium stop (fail-safe). Mirrored
into `core/live_risk_pass.py`, which runs every few minutes and is the path that
actually fires most stops. Added `min_dte_exit = 5` because removing the premium
stop removes the only rule that closed a decaying contract early.

## Limits — read before trusting this

1. **No dollar counterfactual.** This shows the underlying did not break and often
   recovered. It does **not** price what the option would have been worth at a later
   exit. `Data/options_history` bars are trade prints, not marks, and were
   invalidated for P&L by `research/options_experiment/10_RETRACTION_option_pnl_invalid.md`.
   The +$43,944 figure is *what the premium stop cost*, not *what the new rule earns*.
2. **n = 42, one month, one regime, long calls only.** No puts, no walk-forward, no
   regime split. This is a defect diagnosis, not a validated policy.
3. **1.5 ATR is inherited, not fitted** — it is `momentum_config.atr_stop_mult`, chosen
   for the shares path. It has not been tuned here, deliberately: tuning it on the same
   42 trades that motivated the change would be fitting noise.
4. **`min_dte_exit = 5` has no evidence behind it at all.** It is a safety rail added
   because the change removed the only exit a decaying contract had. The risk pass's
   `expiring_before_next_session` flatten already covers the terminal case, so this is
   belt-and-braces and can be set to `None` without reopening the ride-to-zero hole.
5. **Survivorship in the forward window.** "Underlying recovered" is measured on names
   still in the 4H cache; the cache is the live universe, so a delisted name would be absent.
   None of the 18 are delisted, but the check is not structurally immune to it.

## Next

Paper-run it. The measurement that settles this is the exit-reason mix in
`closed_trades.jsonl`: `underlying_stop_-1.5atr` should be materially rarer than
`stop_-39%` was (44 stops in ~4 weeks), and the trades it holds through should not
turn into larger losses. Re-run this script in two weeks against the new rows.
