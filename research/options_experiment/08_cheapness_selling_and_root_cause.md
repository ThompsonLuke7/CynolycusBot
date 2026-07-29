# Cheapness gate, premium selling — and the actual root cause

> # ⚠ MAJOR CORRECTION (2026-07-27, user-prompted). The −29% figure below is WRONG.
>
> The user asked why spreads looked so enormous, noting real-world spreads are "at most ~30%, on
> most names within 10 cents." That was correct and it exposed a specification error in the cost model.
>
> **Spread is approximately a fixed number of CENTS, not a fixed percentage of premium.** The 25.6%
> round-trip figure was calibrated in Gate G1 on the 575 real live trades, whose **median premium was
> $0.94** (53% under $1.00) with an observed **~8-cent half-spread**. 8 cents on a $0.94 option really
> is ~9% each way. But the Phase-3 / let-it-run contracts have a **median premium of $5.60 — 6× more
> expensive.** Applying the same *percentage* implies a **72-cent half-spread** on them, roughly 9×
> the spread actually observed.
>
> **Corrected buyer economics (hold-to-expiry, same trades):**
>
> | half-spread assumption | round trip (% of premium) | buyer return on capital |
> |---|---:|---:|
> | 5 cents | 1.8% | **−5.7%** |
> | **10 cents** (the user's figure) | 3.6% | **−7.4%** |
> | 15 cents | 5.4% | −9.1% |
> | 72 cents (what was wrongly applied) | 25.7% | −28.2% |
>
> **The honest number is −6% to −9%, not −29%.**
>
> **What this changes:**
> - §1's claim that the spread is "essentially the entire loss" is **wrong**. With a realistic
>   cents-based spread the toll is ~2–5pp, and the **−3.8% gross edge is now most of the loss.**
> - §5's recommendation "do not trade options on this universe, the toll dominates" is **withdrawn.**
>   Options here are *mildly* unprofitable on these signals, not catastrophically so — the gap to
>   break-even is small enough that a better entry or filter could plausibly close it.
> - The buyer/seller symmetry in §2 still holds directionally (both pay the spread) but the magnitudes
>   are far smaller, and the "market maker collects $366k" figure is inflated by the same error.
> - §3's flat cheapness result is unaffected in shape (ROC was flat across buckets) but its level is
>   overstated by the same ~20pp.
> - **Phase 3 and the `07_` verdict inherit this error**, since both used the flat percentage. Their
>   *relative* ranking (shares vs options) is likely preserved because the bias is common to all option
>   strategies, but every absolute option number in this experiment is too negative, and multi-leg
>   structures are penalized worst because the error compounds per leg.
>
> **Root cause of the mistake:** calibrating a *ratio* on one premium regime ($0.94 median) and
> applying it to another ($5.60 median) without checking that the ratio was scale-invariant. It is not.
> Any future cost model here must be expressed in cents-per-contract with a percentage floor/cap, and
> validated across the premium range it will be applied to.

**Author:** Claude, 2026-07-27. Script: `scripts/experiment_iv_gate_and_selling.py`.
Data: `research/options_experiment/data/iv_gate_selling.parquet` (1,214 long-call trades).

---

## 1. The root cause: it is the SPREAD, not the strategy

Decomposing the buyer's result (hold-to-expiry, 1,214 trades):

| | value | as % of premium deployed |
|---|---:|---:|
| **Gross P&L, before any costs** | −$27,670 | **−3.8%** |
| Spread + commission drag | −$182,984 | **−25.3pp** |
| **Net P&L** | −$210,654 | −29.1% |

**Before costs, buying these calls is nearly a coin flip — only −3.8%.** The options were
approximately fairly priced. Essentially the entire −29% loss is the bid/ask spread.

This reframes every earlier finding. The instrument, the exit policy, the filter, and the
signal-strength effect are all second-order. **The first-order fact is a ~25% round-trip toll.**
That toll is not modeled or assumed — it is measured from your own real fills in Gate G1
(entry −8.5%, exit +13.9% vs mid).

---

## 2. "Couldn't we just sell options then?" — No. Both sides lose.

| side | net result | on what base |
|---|---:|---|
| buyer | **−$210,654** | −29.1% of premium |
| seller (naked short, Reg-T BP) | **−$155,314** | −8.71% of buying power |
| **combined** | **−$365,968** | ← what the market maker collects |

The seller loses too, for the same reason: **whoever crosses the spread pays it.** Selling does not
harvest the buyer's loss, because the buyer's loss was never an edge transferred to the seller — it
was a toll paid to the market maker. Both counterparties pay it.

Selling is also worse than the headline suggests:

- **Tail risk is severe.** Worst single short trade: **−$10,851**, against a median win of **+$210**.
  One worst-case loss erases **52 median wins**. Win rate 45% with that payoff shape is not a
  strategy, it is a slow accumulation in front of a fast loss.
- Naked short calls have **unbounded** loss. The Reg-T 20% figure is an approximation; a real broker
  raises margin as the position moves against you, so the effective capital requirement is worse
  than modeled here.
- Selling gets *monotonically worse* as options get pricier (−1.7% on the cheapest quintile,
  −53.2% on the priciest), which is the opposite of the usual "sell expensive premium" intuition.

---

## 3. The cheapness gate does not work either

Bucketing buys by `cheapness = premium / expected move` (strike-free; expected move =
spot × Yang-Zhang realized vol × √T):

| bucket | median cheapness | buyer ROC | win rate |
|---|---:|---:|---:|
| Q1 cheapest | 0.20 | −28.9% | 16% |
| Q2 | 0.72 | −29.7% | 35% |
| Q3 | 1.28 | −26.3% | 29% |
| Q4 | 1.97 | −31.4% | 19% |
| Q5 priciest | 2.97 | −29.2% | 12% |

**Flat at −26% to −31% everywhere.** Even paying only 20% of an expected move loses just as much.
Because the loss is a fixed ~25% toll on the premium, cheapness cannot rescue it — the toll scales
with the premium you pay, so buying cheaper options buys proportionally cheaper losses.

Note Q1's 16% win rate: the "cheapest" options are far-OTM lottery tickets that rarely pay, so this
metric partly proxies moneyness. The flatness of ROC across it is the finding, not the ordering.

---

## 4. Reconciling this with real-world success trading parabolic calls

This experiment tests **this book's contracts**, and those are thinly traded — the trades in the
sample have per-contract daily volumes in the single-to-low-double digits. A ~25% round-trip spread
is the direct consequence.

The arithmetic that matters: **gross edge was −3.8%.** If the same trades were executed on contracts
with a 2–4% round-trip spread instead of 25%, the outcome moves from −29% to roughly −6% to −8% —
still negative on these signals, but no longer catastrophic, and within range of being flipped by a
genuinely better entry. On a 1% spread it would be roughly break-even before any signal improvement.

So the correct statement is **not** "buying calls on parabolic names does not work." It is:

> **On this universe's contracts, the spread is larger than any edge any component of the system
> produces. Options are not tradable here at acceptable cost — regardless of direction, exit policy,
> filter quality, or structure.**

That is fully consistent with profitable discretionary call buying on liquid names, where the toll is
a small fraction of what it is here.

---

## 5. What this implies for the roadmap

1. **Do not trade options on this universe.** Not long, not short, not spreads. The toll dominates.
   Shares remain the correct instrument for these signals (momentum shares are genuinely profitable).
2. **If options are wanted, change the universe, not the strategy.** The requirement is a hard
   option-liquidity gate — real volume and tight quoted spreads — applied *before* a name is eligible
   for option routing at all. That is a universe-construction change, not a routing rule.
3. **The parabolic filter remains useful for shares.** OOS AUC 0.63, parabolic rate 28.6% → 44.3%.
   Applied to share sizing or entry selection it costs nothing in spread and may add real value. That
   is the highest-value place to reuse the work done here.
4. **GEX/dealer positioning is still untested** and remains the one open component — but note it would
   have to overcome the same toll to matter for options. Its more likely use is as a *shares* signal.
