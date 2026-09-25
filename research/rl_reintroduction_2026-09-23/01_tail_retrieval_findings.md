# Stage 1 — right-tail retrieval: the funnel does not select for the tail

2026-09-23. Script: `01_tail_retrieval.py`. Output: `data/stage1_retrieval.json`,
`data/stage1_population.parquet` (108MB). Events frozen in
`00_preregistration.md` §5.1. **No model is fitted here and the test block is never read.**

## Answer in one line

The momentum funnel's right-tail retrieval is **lift 1.06x at the gate and 1.18x at the
universe** — i.e. it catches tail events at very nearly the rate it catches everything, and
**loosening the gate does not improve it at any setting tested.** These are measured
absences, not underpowered ones: the MDE on lift is 0.018–0.052 against effects of
0.06–0.18.

---

## 1. Population

| | |
|---|---|
| rows (raw / CA-guarded) | 3,487,233 / **3,486,130** (0.032% flagged) |
| tickers | 4,068 of the 4,104 in the daily cache |
| L1 window | 2018-10-05 .. **2025-08-07** (validation end) |
| L2 window | 2020-09-08 .. 2025-08-07 (the 4H matrix's own span) |
| momentum-pool tickers | 1,081 (matrix) / 1,110 (`shared_universe.csv`), overlap 897 |
| P1 base rate | 1.308% (45,207 events) — cf. 1.44% on the panel, consistent |

One row per (ticker, session): the decision is the prior close, the entry is **this
session's open**, and MFE runs forward from that open. Same convention as the decision panel
(the `23_rank_depth_and_options.md` §0 correction).

---

## 2. L1 — universe retrieval

**Bias direction, stated first:** pool membership is TODAY's list; point-in-time universe
snapshots begin only after 2026-09-10 (`core/shared_universe/universe.py:236`). Today's pool
includes names partly selected *because* they later did well, so this measurement is biased
**in the pool's favour**. A poor result is informative; a good one is not.

P1 = `rmfe_10 ≥ 4R`. `lift = precision / base rate`, which equals `recall / fires` exactly.

| population | fires | recall | lift | 95% CI | MDE |
|---|---|---|---|---|---|
| matrix pool, no liquidity floor | 36.25% | 42.75% | **1.179** | [1.166, 1.192] | 0.018 |
| matrix pool, ≥ $5M dollar volume | 29.15% | 32.66% | **1.120** | [1.093, 1.148] | 0.039 |
| matrix pool, ≥ $25M | 19.19% | 23.45% | 1.222 | [1.148, 1.296] | 0.103 |
| shared universe, no floor | 35.88% | 49.59% | 1.382 | [1.369, 1.395] | 0.018 |

P3 (top-1% `rmfe_20` within bar) behaves the same: 1.128x at no floor, 0.995–1.183x at
floors. So the pool's tail retrieval is **statistically above 1.0 but economically
negligible**, and it is the ceiling, not the estimate, because of the bias above.

### 2.1 The tail we miss is MORE liquid than the tail we keep

Composition of P1 events, CA-guarded, no liquidity floor:

| group | n | median 20d $ volume | median price | median ATR% | share ≥ $5M |
|---|---|---|---|---|---|
| in pool | 19,325 | **$0.97M** | $19.55 | 3.45% | 23.0% |
| **outside pool** | **25,882** | **$2.41M** | $16.35 | 3.51% | **35.4%** |

Same direction for P2 ($2.20M vs $0.82M) and P3 ($1.05M vs $0.35M). **57% of P1 tail events
are outside the pool, and they are 2.5x more liquid than the ones inside it.** The missed
tail is not an illiquid-microcap artifact.

Honest caveat on absolute tradability: the tail population overall is small — median dollar
volume $1–2.4M, and only 23–35% of events clear $5M. So this says the pool's *selection* is
not tail-seeking; it does not say there is a large liquid tail sitting there unexploited.

---

## 3. L2 — the candidate gate

Population = pool rows inside the matrix window where the gate actually had a 4H row.

### 3.1 Coverage: 20% of pool days have no gate row, and 40% of the tail lands there

Pool rows in window 1,230,125; the gate was evaluated on **986,161 (80.17%)**. Of the
in-pool P1 events, only **58.6%** fall on a day the gate evaluated (P2 63.9%, P3 60.9%).

That gap is not the window mismatch — bounding the population to the matrix's own start
moved it by 1.6pp only. The implication is arithmetic: the P1 rate is **1.101% on evaluated
rows** against **1.507%** across all pool rows, so **tail events are ~37% more common on the
rows the feature pipeline has no entry for.** The missing rows are the young, gappy and
halted names, which is exactly where the right tail lives. This is an engineering finding,
independent of RL or of any model.

### 3.2 The gate, and every loosening of it (event P1)

| config | fires | recall | lift | 95% CI | MDE |
|---|---|---|---|---|---|
| **CURRENT (live)** | 21.93% | 23.17% | **1.057** | [1.020, 1.094] | 0.052 |
| open `exclude_low_price` | 22.46% | 25.51% | **1.136** | [1.099, 1.173] | 0.052 |
| open `max_dist_to_52w_high_atr` | 37.46% | 40.92% | 1.092 | [1.067, 1.118] | 0.035 |
| open `min_xsec_near_high_rank` | 37.46% | 40.92% | 1.092 | [1.067, 1.118] | 0.035 |
| open `min_dollar_vol_pctile_252` | 23.90% | 24.48% | 1.024 | [0.990, 1.059] | 0.048 |
| open `min_range_pos_20` | 25.57% | 25.92% | 1.014 | [0.981, 1.046] | 0.046 |
| open `min_rs_spy_20` | 27.26% | 26.48% | 0.971 | [0.940, 1.002] | 0.043 |
| open `min_xsec_ret_20_rank` | 27.26% | 26.48% | 0.971 | [0.940, 1.002] | 0.043 |
| joint ladder 25% open | 40.23% | 38.71% | 0.962 | [0.939, 0.986] | 0.033 |
| joint ladder 50% open | 55.10% | 49.85% | 0.905 | [0.887, 0.922] | 0.024 |
| joint ladder 75% open | 75.80% | 72.33% | 0.954 | [0.943, 0.966] | 0.016 |
| FULLY OPEN | 100% | 100% | 1.000 | — | — |

**No setting reaches 1.14x.** The best single change is opening the low-price exclusion
(1.136x), and the joint ladders are *below* 1.0 — loosening several thresholds at once
dilutes faster than it retrieves, which is the same "recall = firing rate" signature
`research/regime_coverage_2026-09-21/recall_precision.py` found for MA-touch rules.

### 3.3 The gate has 7 knobs but only 5 independent ones

`max_dist_to_52w_high_atr` / `min_xsec_near_high_rank` produce **identical** numbers, and so
do `min_rs_spy_20` / `min_xsec_ret_20_rank`. Both pairs are OR'd inside
`momentum_candidate_mask` (`candidate_filter.py:32-65`), so opening either member releases
the whole clause. Worth knowing before anyone tunes them as if they were separate.

---

## 4. P2 is the one table not to read as a failure

P2 (`mfe_10 ≥ 25%`) shows gate lift **0.614x** and pool lift **0.84–0.92x** at liquidity
floors. That is largely *intentional*: P2 events have median price $10–12 and median ATR
6.5–7.5%, and the gate excludes sub-$5 names by design (`exclude_low_price`). A
volatility-normalized event is the fair test of a gate that deliberately avoids penny
stocks, so **P1 and P3 carry the conclusion and P2 is reported as a diagnostic.** That P2's
number is so far below 1.0 does, however, quantify how much absolute-percentage upside the
low-price rule forgoes.

---

## 5. What this means for the RL plan

* **Retrieval is NOT the binding constraint in the way the thread expected, but not for the
  reason the thread hoped either.** The thread's branch was "if recall is poor, fix
  discovery". Recall is poor *and* loosening the gate does not fix it: the gate is not
  trading precision for recall, it is simply not tail-informative at any setting. Building a
  "high-recall opportunity engine" out of these seven thresholds would add candidates
  without adding monsters.
* **So Stage 2 proceeds on the ungated pool cross-section**, which the plan allowed for and
  this result now justifies: with the gate at 1.06x there is no reason to let it pre-filter
  the rows a selection policy sees.
* **Two findings worth acting on independently of RL**, both cheap:
  1. **Gate coverage.** 20% of pool days have no 4H row and the tail is 37% denser on them.
     Diagnosing why (warm-up length, NaN features, bar gaps) is a data-pipeline task with a
     measurable prize and no modelling risk.
  2. **Universe selection, not the gate, is where the tail is lost.** 57% of P1 events sit
     outside the pool entirely and are 2.5x more liquid than the in-pool ones. The blocker
     on measuring this properly is the absence of point-in-time universe history — and that
     is already being fixed going forward (snapshots from 2026-09-10). Until ~6 months of
     snapshots exist, L1 cannot be measured without the favourable bias.

## 6. Next testable variant

Stage 1 does not terminate on a null; it names its successor. Since the gate's thresholds
are exhausted (11 settings, all ≤ 1.14x, MDE 0.016–0.052), the next variant is **not another
threshold sweep** but the question Stage 2 asks: given that the funnel's *membership*
decision carries ~no tail information, does a *ranking/selection* decision over the same
rows carry any — and can it clear the option route's hurdle.
