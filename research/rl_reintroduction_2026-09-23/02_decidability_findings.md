# Stage 2 — the incumbent is decidable; nothing I fit is. And correction C1 was wrong.

2026-09-23. Scripts: `02_decidability.py`, `02b_route_breakeven.py`. Outputs:
`data/stage2_decidability.json`, `data/stage2b_route_breakeven.json`.
**Validation only — the test block is never read** (it was consumed by the horizon grid and
by confluence; see `00_preregistration.md` §3.2).

Rows: train 915,820 / validation 233,642 position-rows, 244 validation decision bars,
ungated pool cross-section (justified by Stage 1: the gate's tail lift is 1.06x).
Target `r_10` in R units. Validation universe drift is **+0.0595R** — a roughly flat tape in
risk-adjusted terms.

---

## A. Top-k excess in R units, and every arm against its own noise twin

| arm | k=1 | k=2 | **k=3** | k=5 | k=10 |
|---|---|---|---|---|---|
| **incumbent `mom_score`** | 0.193 | **0.490** | **0.441** | 0.388 | 0.270 |
| **incumbent `htf_score`** | 0.282 | 0.249 | 0.237 | 0.213 | 0.199 |
| xgb | −0.037 | 0.001 | 0.031 | 0.017 | 0.006 |
| xgb_null | 0.124 | 0.079 | 0.025 | 0.004 | 0.000 |
| lgbm | −0.105 | 0.019 | 0.007 | 0.028 | −0.000 |
| **lgbm_null** | **0.350** | **0.293** | **0.275** | 0.242 | 0.200 |
| mlp | −0.158 | −0.133 | −0.123 | −0.132 | −0.142 |
| mlp_null | 0.199 | 0.157 | 0.161 | 0.169 | 0.132 |

Both incumbents are significant at every depth (p ≤ 0.0024, most p < 0.0001, decision-day
block bootstrap, 5,000 draws).

**Real minus its own permuted-label null:**

| arm | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| xgb | −0.161 | −0.078 | **+0.005** | +0.013 | +0.005 |
| lgbm | −0.456 | −0.275 | −0.268 | −0.215 | −0.200 |
| mlp | −0.356 | −0.290 | −0.284 | −0.301 | −0.274 |

**Answer to the thread's Experiment B: the incumbent wins and nothing I fit adds anything.**
XGBoost nets ~zero; LightGBM and the MLP are *beaten by their own noise twins* by 0.20–0.46R.
`lgbm_null` — trained on within-bar permuted labels — produces **+0.275R** of top-3 excess at
p < 0.0001. Anyone reading that number without the null arm would have reported a discovery.

Note what this does *not* say: the incumbent `mom_score` has no null arm here, because it is
a walk-forward OOF artifact I cannot retrain inside this script. Its ordering is controlled
elsewhere — `23_rank_depth_and_options.md` §2 (within-bar rank permutation, 9/9 cells,
p ≤ 0.003) and `25_matched_control_and_depth_null.md` §B.2 (beta- and liquidity-residualised
depth gradient survives). So the incumbent's *ordering* is externally certified; its *level in
R units* on this window is not.

## B. Tail capture — the first positive tail result in this series

Picks at k=3, 732 validation picks, universe P1 (`rmfe_10 ≥ 4R`) rate **1.174%**:

| arm | mean R | p99 R | share ≥ +3R | **P1 capture** | vs universe |
|---|---|---|---|---|---|
| incumbent `mom_score` | 0.501 | 8.470 | 6.2% | **10.38%** | **8.8x** |
| incumbent `htf_score` | 0.297 | 4.316 | 3.6% | 5.46% | 4.7x |
| xgb | 0.090 | 4.328 | 2.1% | 3.83% | 3.3x |
| xgb_null | 0.085 | 4.161 | 3.3% | 4.92% | 4.2x |
| lgbm | 0.066 | 4.479 | 2.7% | 2.46% | 2.1x |
| lgbm_null | 0.335 | 5.392 | 3.4% | 3.42% | 2.9x |

The deployed momentum score concentrates right-tail events **8.8x**. Again note the fitted
arms are beaten by their nulls on this metric too (xgb 3.83% vs xgb_null 4.92%).

---

## C. What the deployed score actually picks — and it changes the route arithmetic

| metric | picks (k=3) median | universe median | ratio |
|---|---|---|---|
| `atr_pct` | **11.18%** | 3.61% | **3.10x** |
| daily σ (`past_vol_20`) | 8.13% | 2.71% | 3.00x |
| `beta_60` | 2.05 | 1.19 | 1.72x |
| `log_dollar_vol_20` | 13.65 (≈ **$0.85M**) | 15.62 (≈ $6.1M) | **0.87x → ~7x less liquid** |

Mean `fwdret_10` on the picks is **+13.96%** against **+1.29%** for the universe, while the
R-unit means are 0.501 vs 0.087. Both are true: dividing by a 3x larger ATR shrinks a big
percent move. The R number is the pre-registered, risk-adjusted one.

**Momentum's top-3 are small, illiquid, very high-beta names.** That single fact drives the
rest of this section.

---

## D. Correction — my plan's C1 was wrong

The plan asserted: *"a 4–8pp entry-timing improvement cannot pay a 35% round-trip hurdle;
entry-timing RL on the option route is arithmetically incapable of fixing P&L."*

**That comparison omitted leverage and is wrong.** The ~35%-of-premium hurdle is denominated
in *premium*; an edge measured on the *underlying* has to be multiplied by the option's
elasticity before the two can be compared. Breakeven underlying move = round-trip spread ÷
elasticity:

| IV (ann.) | premium as % of spot (30-DTE ATM) | elasticity | **breakeven underlying move** |
|---|---|---|---|
| 30% | 3.61% | 6.5x | **1.60%** |
| 50% | 5.89% | 4.6x | 2.58% |
| 75% | 8.73% | 3.1x | 3.75% |
| 100% | 11.56% | 2.4x | 4.85% |
| 150% | 17.18% | 1.7x | 6.89% |
| 200% | 22.71% | 1.4x | 8.71% |

At the picks' own median annualised vol (**129%**): elasticity **3.89x**, premium 14.8% of
spot, **breakeven 6.07%** of underlying move. The measured top-3 excess of 0.4412R is
**9.86%** at the picks' own ATR. So it clears the spread — by this arithmetic the option route
is **not** hopeless, and the "5x too small" framing in the plan is retracted.

### D.1 Three reasons not to act on that

1. **The 23.6pp spread is demonstrably not scale-invariant**, and this script shows the
   mechanism: premium ranges from 3.6% to 22.7% of spot across the IV grid. The 23.6pp figure
   was calibrated on our *actual* fills — cheaper, shorter-dated contracts on more liquid
   names. On $0.85M-dollar-volume underlyings the real spread is very likely **wider**, so
   6.07% is a floor on the breakeven, not an estimate. AGENTS.md's own rule names this exact
   failure.
2. **The picks may be untradeable in options at all.** Median underlying dollar volume is
   $0.85M. `momentum_config`'s chain gates (`min_open_interest: 500`, `min_chain_volume: 100`)
   would reject most such chains. **New hypothesis, cheap to check:** the live system may
   rarely get to trade its own top-3, which would explain a large part of why live option
   results diverge from the ranking's measured edge. Checkable against
   `Data/inference/*/live_signal_audit.jsonl` and the 738MB OPRA cache.
3. **Regime.** Validation (2025-02..08) was a hot tape for high-beta small caps. Doc 23
   measured *real* option outcomes at −36% to −57% net across three 2026 expiry cycles. These
   are different windows, not a contradiction — and the model-based levels in
   `02_decidability.py` §D (option net +43.4% for the incumbent) must **not** be quoted as a
   result for exactly that reason. Only the breakeven arithmetic above is portable.

Also: the excess is measured against the bar's own mean, and a long option collects the bar
drift too (+0.0595R ≈ +1.3%). That part is a market-direction bet, not an edge.

---

## E. Verdict and consequences

* **The ENTER/PASS decision is decidable — by the deployed score.** +0.441R at k=3,
  p < 0.0001, 8.8x tail concentration.
* **No model I fit improves on it**, and two of three are beaten by their own permuted-label
  null. On this 20-feature interpreted state, a new selection layer — supervised, bandit or
  RL — has nothing to add. The thread's bake-off is answered, and the answer is "the
  incumbent".
* **Correction C1 is retracted** (§D). The route question is open, not closed, and the
  decision-useful number is a **~6% breakeven underlying move**, floor.
* **Stage 5 is not killed.** Stage 2 rules out improving *which* name to take; entry timing
  asks *when*, which is a different question and still has the incumbent's certified ordering
  underneath it.
* **Next testable variant** (per AGENTS.md): the state representation, not the estimator.
  Every fitted arm shares the same 20 interpreted features, so their common failure is
  evidence about the features. The two concrete candidates already sitting in the repo are
  **forward-RV** (`forward_rv_predictability.py`, OOS R² 0.742) and **decision-time IV**
  (shipped 2026-09-22, accumulating). Neither existed when these features were assembled.
