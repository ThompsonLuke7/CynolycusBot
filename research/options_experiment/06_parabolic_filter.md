# The Parabolic-Likelihood Filter — results

**Author:** Claude, 2026-07-27. Scripts: `scripts/build_forward_excursion_labels.py`,
`scripts/build_parabolic_filter_dataset.py`, `scripts/train_parabolic_filter.py`.
Eval: `research/options_experiment/data/parabolic_filter_eval.json`.

---

## 1. The label was wrong, and fixing it changed everything

Phase 3 (and section 2 of `04_tail_and_thesis_review.md`) measured "parabolic" using
`realized_move_atr` from the trade ledger. **That column is quantized by each module's exit rule,
not by how far the stock ran:**

| module | quantization |
|---|---|
| momentum_expansion | 2,860 trades at *exactly* +2.0 ATR (tp), 865 at *exactly* −4.0 (sl) |
| multi_ticker_swing_htf | 12,969 at exactly −2.0 (sl), 5,411 at exactly +5.0 (tp) |

A stock that squeezed +12 ATR is recorded as +2.0 ATR, because momentum took profit at 2 ATR.
So the earlier "parabolic tail" analysis was really measuring *"did the take-profit fire"*.

The correct label is **forward maximum favorable excursion (MFE)** computed from underlying 4H bars,
anchored strictly after the decision bar, independent of the module's exit.

### True forward excursion (all 27,049 4H trades)

| module | ≥4 ATR within 10 bars | 20 bars | 40 bars | 60 bars | median MFE_20 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| momentum_expansion | 22.8% | **40.0%** | 56.4% | 65.4% | **3.07 ATR** | 9.58 | 56.3 |
| multi_ticker_swing_htf | 17.4% | **32.6%** | 49.2% | 58.7% | 2.63 ATR | 7.47 | 65.9 |

**Parabolic moves are common — far more common than the exits permit capturing.**

---

## 2. Leakage caught: the first result was fake

The first trained model scored **AUC 0.951 / precision 97–100%** on held-out test data. That is not
plausible for forward price movement, and it wasn't real.

The modules' training matrices have their own **label files merged in**. The model's top features were
`expansion_score`, `expansion_target`, `expansion_survival_score` — forward-looking by construction.
`expansion_target` defeated a prefix-based leak filter because it *ends* with "target". Project memory
already records `trend_persistence` as a forward label that invalidated an earlier study; it was
present here too.

**Fix:** enumerate forbidden columns from the label parquets themselves rather than pattern-matching
names. 12 columns dropped for momentum, 18 for HTF.

**Anything reported below is post-fix.** The 0.951 figure is recorded only as a cautionary artifact.

---

## 3. Honest results — the filter works, modestly

Walk-forward by time (60/20/20), test window scored once.

### momentum_expansion — test 2026-02-20 .. 2026-05-14, base rate 42.0%, AUC 0.583

| selectivity | n | precision | lift |
|---|---:|---:|---:|
| **top 5%** | 39 | **59.0%** | **1.40×** |
| top 10% | 77 | 54.5% | 1.30× |
| top 20% | 154 | 51.9% | 1.24× |
| top 50% | 384 | 47.1% | 1.12× |

### multi_ticker_swing_htf — base rate 38.0%, AUC 0.591

| selectivity | n | precision | lift |
|---|---:|---:|---:|
| **top 5%** | 230 | **54.3%** | **1.43×** |
| top 10% | 460 | 49.6% | 1.30× |
| top 20% | 919 | 46.9% | 1.23× |

**Assessment.** AUC ~0.58–0.59 is weak in absolute terms, but the effect is *consistent*: two
independent modules, different feature sets, different test windows, both landing at ~1.4× lift at
top-5% selectivity, with monotonic decay as selectivity loosens. That consistency is the strongest
evidence it is real rather than noise. Validation-to-test degradation (0.672 → 0.583 on momentum)
means the model is somewhat unstable and the operating point should be re-fit periodically.

Top surviving features are ordinary trend/volatility/cross-sectional ones — `atr_pct_14`,
`rs_sector_20`, `ema_stack_4`, `ema_slope_20`, `atr_expand_14_60`, `drawdown_from_60h`,
`xsec_ret_5_rank` — plus the module's own score. No single dominant feature, consistent with a weak
but genuine signal.

---

## 4. The finding that matters more than the filter

**The exit rule is the binding constraint, not the instrument and not the filter.**

momentum takes profit at **+2.0 ATR**. The median trade's true 20-bar MFE is **3.07 ATR**, and the
90th percentile is **9.58 ATR**. The strategy systematically sells the move it is trying to catch.

This reframes the entire options question:

- Long options are a bet on the tail. Their whole edge is convexity in a large move.
- A +2 ATR take-profit truncates precisely that tail.
- **Buying options and exiting at 2 ATR is the worst combination available**: you pay the premium and
  the ~22% round-trip spread for convexity, then sell before the convexity pays.

That is a coherent explanation for the Phase 3 result *and* for the live book's option P&L, without
needing to conclude that the gamma-squeeze thesis is wrong.

---

## 5. What this does and does not license

**Supported:**
- A ~1.4× improvement in parabolic hit rate at top-5% selectivity is available and out-of-sample.
- Parabolic moves are frequent (33–40% reach 4 ATR within 20 bars).
- The exits, not the instrument, are the first thing to fix.

**Not yet established:**
- Whether filter + options + a *let-it-run* exit is actually profitable. That requires re-running the
  counterfactual with option-native exits (trail, time-stop, partial scale-out) instead of the
  module's 2-ATR take-profit. **This is the single highest-value remaining experiment.**
- Anything about gamma/dealer positioning — still zero data overlap (see `05_gex_reconstruction.md`).
- HTF's underlying expectancy remains negative; a better exit does not fix a losing signal.

**Next step:** re-run Phase 3 on the top-5% filtered cohort with option-native exits, and compare
against both shares and the current 2-ATR-capped baseline.
