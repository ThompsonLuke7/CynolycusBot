# Let-It-Run Exits + Filter — the thesis test, and the verdict

> **CORRECTIONS (2026-07-27, after user challenge). Read before §2.**
>
> **(a) The wrong exit policy was tested.** §1 used the *backtest's* 2-ATR take-profit. The
> DEPLOYED live policy is `core/live_4h_exec.py` ExecPolicy "tail-rider": premium **stop 39%**,
> **no trail**, **sell 16% at +30%**, **horizon 53 bars**. Its own code comment says it was chosen on
> a shares-only backtest with "no option-premium path modeled ... not yet paper-validated live" —
> so it had never been evaluated on real option paths. Now tested: **−30.5% return on capital, 18%
> win rate**, and its 39% premium stop fires on **48% of trades** (589 stops / 620 expiries / 5
> horizons). It is not better or worse than the alternatives — the ranking in §1 is unchanged.
>
> **(b) §2's claim that "the filter makes options worse" was WRONG.** §2 sorts by the module's own
> *score*, which is not the parabolic filter. Testing the actual trained filter's out-of-sample picks
> (n=349, both modules, hold-to-expiry, pessimistic):
>
> | filter quintile | n | parabolic rate | option ROC | win rate |
> |---|---:|---:|---:|---:|
> | Q1 (lowest) | 70 | 28.6% | −23.8% | 27% |
> | Q3 | 69 | 33.3% | −21.6% | 28% |
> | **TOP 20%** | 70 | **44.3%** | **−24.2%** | 26% |
>
> The filter **does** separate the moves — parabolic rate rises 28.6% → 44.3%, OOS AUC 0.63. But
> option ROC is **flat at −21% to −24% across every quintile.** The correct statement is that the
> filter is roughly **neutral** for option economics, not harmful. §2's monotonic score-based decline
> is a real but *different* effect (signal strength ↔ option expensiveness) and should not be read as
> the filter's behavior.
>
> **(c) A small-sample result was nearly over-claimed.** A momentum-only cut (n=26) showed the filter's
> top quartile at **+2.6% ROC** — apparently rescuing the strategy. Expanding to n=349 across both
> modules erased it. Top-5% ROC is −9.1% with a bootstrap CI of **[−46%, +46%]** — pure noise at n=17.
> Recorded because it is exactly the kind of cell that gets mistaken for a finding.
>
> **Net effect on the verdict:** unchanged in direction, but the mechanism is sharper. Predicting the
> *move* is achievable (AUC 0.63, 44% hit rate). Converting that into option *profit* is not, because
> the move must also exceed premium plus ~25% round-trip spread within the contract's life.

**Author:** Claude, 2026-07-27. Script: `scripts/experiment_let_it_run_exits.py`.
Data: `research/options_experiment/data/let_it_run.parquet` (5,606 rows, 1,214 long-call trades).
Costs: pessimistic (25.6% round-trip spread + $0.65/contract), per the G1 verdict.

This tests the combination the strategy actually intends — long calls, held through the move
rather than sold at the module's 2-ATR take-profit — using cached daily contract bars, so option
prices are real market prints, not model output.

---

## 1. Letting it run does NOT rescue options

| exit policy | n | return on capital | win rate | median P&L |
|---|---:|---:|---:|---:|
| hold_to_expiry | 1,214 | **−29.1%** | 22% | −$175 |
| atr_target_4 | 1,214 | −30.2% | 21% | −$180 |
| time_20d | 1,164 | −30.3% | 19% | −$168 |
| trail_50pct | 1,214 | −30.4% | 18% | −$187 |
| module_exit (Phase 3 baseline) | 800 | −30.6% | 4% | −$145 |

Holding to expiry is the best of the five and beats the module's exit by **1.5 percentage points**
— directionally consistent with the truncation argument in `06_parabolic_filter.md`, but nowhere
near enough to matter. Every policy loses roughly 30% of deployed capital.

**The truncation hypothesis is directionally right and economically irrelevant.** Fixing the exit
does not fix the instrument.

---

## 2. The filter makes options WORSE, not better — and that is the key insight

Sorting the same trades by the module's own signal score (higher = stronger conviction),
under the best exit policy:

| score quintile | n | return on capital | win rate | avg win | avg loss |
|---|---:|---:|---:|---:|---:|
| Q1 (weakest) | 169 | **−4.9%** | 36% | $469 | −$306 |
| Q2 | 168 | −27.9% | 24% | $239 | −$346 |
| Q3 | 169 | −31.5% | 22% | $340 | −$388 |
| Q4 | 168 | −33.1% | 20% | $452 | −$417 |
| **Q5 (strongest)** | 169 | **−39.6%** | **15%** | $446 | −$409 |

**Monotonic, and backwards from the thesis.** The signals we are most confident in produce the
*worst* option returns. Win rate falls from 36% to 15% as conviction rises.

The most likely mechanism: **the market has already priced the momentum the module is detecting.**
A high-conviction momentum name is one that has already expanded, so its options carry elevated
implied volatility. You pay more premium for the same subsequent move. The edge the signal
identifies is real in the underlying (Phase 3: shares are profitable on momentum) but it is
*already in the option price.*

This is why a better parabolic filter cannot save the strategy. The filter selects for exactly the
names whose options are most expensive. Selectivity and cost move together.

---

## 3. How big is the gap? Quantified.

Breakeven win rate implied by each policy's own win/loss sizes:

| policy | actual win rate | breakeven needed | gap |
|---|---:|---:|---:|
| time_20d | 19.3% | 47.8% | **+28.5pp** |
| hold_to_expiry | 23.4% | 49.1% | **+25.7pp** |
| trail_50pct | 20.2% | 49.2% | +29.0pp |
| atr_target_4 | 21.5% | 55.7% | +34.2pp |
| module_exit | 4.4% | 71.0% | +66.6pp |

Long calls on this book need roughly a **doubling** of win rate (23% → 49%) to break even.

The best filter we could build (`06_parabolic_filter.md`) delivers **1.40–1.43× lift** on the
*underlying* parabolic rate at top-5% selectivity. Even taken at face value, that is far short of
doubling the option win rate — and §2 shows the filter's selection is *negatively* correlated with
option profitability anyway.

**The gap is not a tuning problem. It is roughly an order of magnitude away from closing.**

---

## 4. Verdict on the strategy as conceived

The intended strategy is: detect expansion → buy calls → ride the parabolic move.

Every component has now been tested against real option prices:

| component | finding |
|---|---|
| Do parabolic moves happen? | **Yes** — 40% of momentum trades reach ≥4 ATR within 20 bars |
| Do options capture the tail better? | **Yes** — top 5% of moves: 124.8% on capital vs 23.1% for shares |
| Are the exits truncating the tail? | **Yes** — sells at 2 ATR vs median 3.07 / p90 9.58 ATR MFE |
| Does fixing the exits help? | **Barely** — +1.5pp, still −29% on capital |
| Can we predict the parabolic moves? | **Weakly** — 1.4× lift, AUC 0.58 |
| Does the filter make options pay? | **No — it makes them worse** (Q5 −39.6% vs Q1 −4.9%) |
| Is the gap closeable? | **No** — needs +26pp win rate; best lever gives a fraction of that |

The individual intuitions were sound. The tail is real, options do capture it better, and the exits
were indeed truncating it. But the composed strategy still fails, because signal strength and option
cost are positively correlated: **you cannot systematically buy cheap convexity on names the market
already knows are expanding.**

## 5. What is still open

- **Gamma/dealer positioning remains untested** (zero historical overlap; see
  `05_gex_reconstruction.md`). This is the one component of the thesis not yet falsified. If a
  GEX-based signal identifies squeezes *before* implied vol reprices — i.e. selects for cheap
  convexity rather than expensive — it would attack the exact mechanism that defeats the current
  approach. That is a genuinely different bet from the score-based filter tested here, and it is
  the only remaining path worth the effort.
- **Selling premium instead of buying it.** Every result here says option buyers overpay on this
  book. The symmetric implication — that *selling* premium on high-score names may be the edge —
  is untested. Phase 3's `vertical_credit` lost, but it was sized and exited on the module's rules,
  not as a premium-selling strategy. Worth a dedicated test.
- Shares remain the best instrument found for these signals, and momentum shares are genuinely
  profitable (+$507k matched-risk, +$132k matched-notional across the sample).
