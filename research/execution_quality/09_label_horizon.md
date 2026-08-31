# The label horizon — the models are graded on a question the exit never asks

2026-08-29. Script: `scripts/execution_quality/stage6_label_horizon.py`

## What the labels actually are

Verified in the config, not from memory:

**momentum_expansion** (`LABEL_CONFIG`) — forward window **25 × 4H bars ≈ 15.4
trading days**. Composite of `fwd_max_alpha` (0.40), `fwd_atr_adj_return` (0.25),
`trend_persistence` (0.20), `fwd_max_drawdown` (0.15); binary target = top 20%
of that composite in a rolling 2,000-bar window.

**multi_ticker_swing_htf** (`PIVOT_LABEL_CONFIG`) — fractal pivot-anchored
(3 left / 3 right), forward **13–38 bars ≈ 8–23 trading days**; composite alpha
0.35, atr-adjusted 0.25, drawdown 0.25, persistence 0.15; top 15%.

Neither is triple-barrier and neither is a plain MFE/MAE ratio — both are
**top-quantile composite ranking targets** over a multi-week window. (Note
`trend_persistence` is legitimate here: it is part of the *label*, which is
forward by definition. The 2026-07 retraction was about using it as a *feature*.)

## The mismatch

Hold measured as trading days between entry and exit fill:

| module | median hold | p75 | label horizon | label ÷ hold |
|---|---|---|---|---|
| momentum_expansion | **4 d** | 8 d | 15.4 d | **3.8x** |
| multi_ticker_swing_htf | **5 d** | 31 d | 15.7 d | **3.1x** |
| meta_ranker | 7 d | 18.5 d | (inherits both) | — |
| dealer_ranker | 2 d | 3 d | rules, no ML label | — |
| multi_ticker_swing (30m) | 1 d | 2 d | — | — |

**The models rank names by how far they travel over ~15 trading days. The live
policy exits after 4–5.** Stage 4A tested 1d/3d/10d and found nothing — it never
reached the horizon the models were trained for.

## Re-running the same test out to the label's horizon

Top-3 vs the same bar's lower-ranked names, median forward MFE difference in ATR
(daily bars, so the full sample is usable; `*` = bootstrap 95% CI clear of zero):

| module | n | 1d | 3d | 5d | 10d | 15d | 20d | 25d |
|---|---|---|---|---|---|---|---|---|
| dealer_ranker | 164 | −0.107 | +0.062 | +0.110 | +0.171 | +0.218 | +0.408 | **+0.514** |
| meta_ranker | 608 | +0.001 | +0.071 | +0.205 | +0.217* | +0.203 | +0.208 | +0.187 |
| momentum_expansion | 634 | +0.046 | +0.100 | +0.163 | +0.189* | +0.243 | +0.079 | +0.309 |
| multi_ticker_swing_htf | 656 | +0.006 | +0.106 | +0.050 | +0.094 | +0.075 | −0.036 | −0.163 |

**The signal is not zero — it grows with horizon.** Three of four modules show a
monotone-ish rise from ~0 at 1 day to +0.19–0.51 ATR by 10–25 days. Two cells
clear the CI at 10d. This is a materially different picture from Stage 4A's flat
null, and the reason is simply that Stage 4A stopped at 10 days.

It is still **weak**: only 2 of 28 cells are individually significant, and with
seven horizons per module that is roughly what multiple testing alone would
produce. The honest reading is "a small effect that strengthens with horizon,
not established at any single horizon" — not "the models work".

**HTF is the exception and goes the wrong way** (+0.106 at 3d decaying to −0.163
at 25d). Its label is pivot-anchored rather than horizon-anchored, so a fixed
forward window is a poorer match for it than for momentum; that is a caveat on
the test, not a defence of the module.

## What the exit gives up

Pooled across all ranked names, median favourable excursion in ATR:

| by | MFE | vs 5-day hold |
|---|---|---|
| 5d | 0.743 | — |
| 10d | 0.979 | **+0.242** |
| 15d | 1.228 | **+0.494** |
| 20d | 1.423 | **+0.661** |
| 25d | 1.693 | **+0.911** |

Exiting at 5 days leaves roughly **0.5 ATR of favourable excursion unclaimed by
15 days**, which is the horizon the label was built around.

## The catch that stops this being a recommendation

The drift control has not improved. Median forward *return* for an average ranked
name is **−0.291 ATR at 5d, −0.395 at 15d, −0.434 at 20d**, and MAE exceeds MFE
at every horizon (1.579 vs 1.228 at 15d). So the extra excursion at longer
horizons is available in **both directions**, and simply holding longer harvests
the adverse side too. That is consistent with Stage 5, where every "hold for
more" variant improved the median and destroyed the mean.

So the finding is *not* "hold longer". It is:

1. **The horizon mismatch is real and large (3–4x)** and it is the single most
   concrete defect found in the modelling stack.
2. **The fix is to align them, in whichever direction is cheaper to test** —
   relabel at the horizon actually traded (~5 days) is the cheaper experiment and
   does not require holding risk longer.
3. Retraining on a *5-day* label is a different proposition from retraining on
   more data with the same 15-day label. The first tests a hypothesis; the second
   is what Stage 4A already said not to do.

## Recommended next experiment

Rebuild the momentum label with `forward_window_4h_bars` at **8** (~5 trading
days, matching the observed hold) and at **13** (~8 days), keep every feature and
split boundary fixed, and compare rank-vs-forward-MFE at the matched horizon
against the current 25-bar label. That is an ablation with one moving part, it
reuses `scripts/analyze_momentum_expansion_label_variants.py`, and it answers
whether the mismatch is costing anything before any GPU time is booked.
