# Stage 3 pick: net-of-cost selectivity (checklist 1.1), validation only

Run 2026-11-19 with `strategies/momentum_expansion/ablation/run_stage3_selectivity.py` on the
Stage 3 out-of-fold scores. Results are in
`ablation/results/stage3_eval/selectivity_validation.csv`.

**Scope:** validation only, meaning bars before 2026-05-15 minus each horizon's embargo; the
sealed test was not read.

**The question:** can the momentum-slot pick (L2_20__pit, neutral IC +0.0093) be turned into a
long-only book that beats SPY after costs?

## Design

- **Scores:** the pick's raw score and its neutral score.
- **Depths:** top 1, 3, 5, 10, 20 and 50 names, and the top 5%, 10% and 20%.
- **Weighting:** equal, or inverse ATR%.
- **Horizons:** 10 and 20 sessions.
- **That gives 36 constructions per horizon.**

Per bar:
- **own_net** = mean(fret − SPY − cost) of the selection, where the cost is the round-trip
  price band (`run_subset_eval.COST_BANDS`, 0.15–1%) and entry is at the next bar's open;
- **vs_univ** = selection fret minus the bar's equal-weight universe, before cost;
- **− perm** = the same construction on the pick's noise-trained twin, subtracted.

Returns are winsorised at 1/99 within the bar, and CIs come from a week-block bootstrap.

## Result: not monetisable long-only at the assumed costs

| | 10 sessions | 20 sessions |
|---|---|---|
| Universe, equal-weight, own_net | −0.58% [−0.84, −0.28] | −0.85% [−1.45, −0.27] |
| Best construction, own_net | −0.44% [−0.73, −0.18] (raw, inverse-vol, top 20%) | −0.42% [−1.63, +0.79] (neutral, inverse-vol, top 5) |
| Constructions with an own_net CI above 0 | **0 of 36** | **0 of 36** |
| Best vs_univ with a CI above 0 | +0.17% [+0.06, +0.29] (raw, inverse-vol, top 20%) | +0.27% [+0.04, +0.49] (raw, inverse-vol, top 20%) |

- **The selection does add value over its universe.**
  - Broad books (top 10–20%, inverse-vol) beat the equal-weight universe by +0.13 to +0.27%
    per tranche gross, with some CIs above 0.
  - The noise twin at the same broad depths gains about as much (own − perm ≈ 0): breadth and
    inverse-vol weighting capture most of it, not the model.
- **The universe itself trails SPY after costs** (−0.58% at 10 sessions, −0.85% at 20).
  - It is the top 3,000 liquid names, including small caps, over 2019–2026.
  - The cost bands bite hardest on cheap names.
- **The pick's increment is smaller than the round-trip cost of the names.** A book hedged
  against the universe would therefore also be about flat or negative before the hedge's own
  cost.
- **Small-k books are noisy.** Their "beats the noise twin" figures (for example raw top 5,
  +1.15%) come from the twin's top-k being very negative (random vol tilts), not from the pick
  being positive. own_net is still negative.
- **Folds:** own_net is positive in only 2 to 7 of 15 folds for every construction.
- **MDE:** at the broad depths the own_net CIs are about ±0.25–0.3% at 10 sessions (MDE about
  0.4%), so these are measured negatives, not underpowered nulls. The small-k books have an MDE
  of about 1.2–3%, so for them this is not measurable here.

## Interpretation and next variants

The Stage 3 label produces a ranking that is weakly better than noise across the
cross-section, but its edge per trade is below transaction cost. That fits Stage 2 and the
rank-depth findings: ordering is real, but the top-k does not pay.

Variants that could still change the answer, each needing its own pre-registration before the
test opens:

1. **Lower cost through the universe:** restrict to $20+ names (0.15% band) and re-measure the
   selection increment there. The test is whether the edge survives where costs are smallest.
2. **Longer holding with less turnover:** hold 40 sessions or rebalance only on rank decay, so
   the same increment pays cost less often. L2_40 already exists as a yardstick.
3. **Use the score as a filter, not a selector:** apply it as a veto on another module's picks
   (for example, momentum's live top-k), where it adds no turnover of its own.

The final-booster shadow scoring (step 4) runs regardless: forward IC is the evidence any of
these variants would need.
