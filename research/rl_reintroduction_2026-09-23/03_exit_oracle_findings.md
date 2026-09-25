# Stage 3 — exit headroom: the live exit rule is the single largest measured loss

2026-09-23. Script: `03_exit_oracle.py`. Output: `data/stage3_exit_oracle.json`,
`data/stage3_paths.parquet`. **Train+validation only; the test block is never read.**

> ## CORRECTION 2026-09-23 (same day) — `live_approx` IS NOT THE LIVE RULE
>
> The arm labelled `live_approx` below models `momentum_config`'s RISK block (1.5-ATR stop,
> 2.4-ATR ratchet trail arming at 1.0 ATR, 15-session cap). **That block does not bind the live
> path.** Its only consumers are `strategies/momentum_expansion/policy/momentum_option_policy.py`
> and `backtest/simulate.py`, and `live/runner.py:516` states plainly that the governed engine
> "Replaces MomentumOptionPolicy execution."
>
> The actual live exit is `core.live_4h_exec.ExitPolicy`, wired at `live/runner.py:679-681`:
> `take_profit 0.30` / `scale_frac 0.16` (small trim), `horizon_bars 53` (4H bars ≈ 26 trading
> days), **`trail_stop = None` — no trail at all**, `stop_loss 0.39` (premium, governs equity),
> and `underlying_stop_atr 1.5` for options since 2026-08-18.
>
> So `live_approx` is approximately the **superseded** config (the stop50/trail35/tp20/hz25
> family), not the current one. Consequences:
>
> * **The "~0.57R per trade" figure is the value of a fix the repo ALREADY SHIPPED, not money
>   on the table.** `exit_policy_cross_module.csv` replaced the trailing shape with the
>   "tail-rider" shape precisely because it earned 2-2.5x more mean return per trade. Section
>   A.1 below independently reproduces that reasoning; it does not identify a new prize.
> * **What still stands unchanged:** every fixed-policy comparison (hold-30 +0.662R, oracle
>   +2.243R, so **1.58R of headroom over the best fixed policy**), §A.3's finding that 83% of
>   the headroom is generic, and all of §B — the learnability nulls do not depend on which
>   fixed policy is the yardstick.
> * **The open question is now different and sharper:** is `horizon_bars 53` long enough, and is
>   `underlying_stop_atr 1.5` the right width? Both are explicitly "not yet paper-validated" in
>   their own docstrings. The machinery in `03_exit_oracle.py` answers exactly that once
>   `live_approx` is swapped for the real ExitPolicy.
>
> This is the AGENTS.md rule "do not let a deprecated module anchor a conclusion about live
> behavior", and I walked into it. Read §A.1 as corroboration of a shipped change.

## Setup

Entries: top-3 per decision bar by the deployed walk-forward OOF `mom_score` and
`htf_score`, plus a **random-3 control** drawn from the same bars and the same eligible
universe. 12,177 picks → **9,790 with a usable ≥2-session path**, 377,487 position-days,
1,015 tickers, bars 2022-11-14 .. 2025-08-07.

Everything is in R units, 1R = 2.0 × ATR14% at entry, net of a 20bp round trip
(`BACKTEST_CONFIG["commission_pct"]`).

Four policies on identical entries:

| policy | rule |
|---|---|
| `fixed_H` | hold H sessions, exit at that close, H ∈ {5,10,15,20,30} |
| `live_approx` | 1.5-ATR initial stop; trail arms at +1.0 ATR then trails 2.4 ATR off the running high; 15-session time stop — i.e. `momentum_config`'s RISK block |
| `oracle` | the best available CLOSE over sessions 1..30 — full hindsight |

**Conventions that bound the claim, stated up front.**
* Every policy decides at a daily close and exits at that close, so the oracle is
  **attainable in principle by a daily policy**. It is deliberately *not* the intraday high,
  which nothing could take. The stop/trail arm is the one exception and uses intraday
  low/high, as a real stop fills.
* Same-session exits (offset 0) are excluded for **all** arms. That denies the oracle a
  same-day rip, and denies `live_approx` a same-day stop-out — so it flatters `live_approx`
  and therefore **understates** headroom.
* `live_approx` omits `score_decay_exit` (needs a forward score path the panel does not
  carry), so it holds slightly longer than the live rule. Also biased toward `live_approx`.
* This is the **equity** path. The options wrapper's ~35%-of-premium round trip
  (`23_rank_depth_and_options.md` §5) is not in these numbers.

---

## A. Headroom — mean realised R by policy

| policy | mom_score | htf_score | random_k |
|---|---|---|---|
| `fixed_5` | 0.096 | 0.161 | 0.032 |
| `fixed_10` | 0.216 | 0.243 | 0.109 |
| `fixed_15` | 0.301 | 0.343 | 0.163 |
| `fixed_20` | 0.410 | 0.435 | 0.241 |
| **`fixed_30`** | **0.662** | **0.656** | 0.408 |
| **`live_approx`** | **0.091** | **0.161** | 0.083 |
| `oracle` | 2.243 | 2.031 | 1.725 |

Median R tells the same story more sharply — `live_approx` median is **−0.148R** (momentum)
and **−0.156R** (HTF) against `fixed_30`'s +0.039R / +0.040R.

### A.1 The finding that needs no model and no control

**The live exit rule gives up ~0.57R per trade against simply holding 30 sessions**
(0.091 vs 0.662 on 2,430 momentum entries; 0.161 vs 0.656 on HTF). That is a direct
comparison of two fixed policies on identical entries — no fitting, no selection, nothing to
null out. And both bias corrections above push in the same direction, so 0.57R is a floor.

It also reproduces, on 26x the entries and with a proper control arm, what
`research/regime_coverage_2026-09-21` Experiment A found on 316 live entries: widening the
stop monotonically lengthens the hold and improves the mean.

Mechanism is visible in the distribution (momentum arm):

| policy | p5 | median | p95 | share ≥ +3R | share ≤ −1R |
|---|---|---|---|---|---|
| `live_approx` | −0.763 | −0.148 | 1.975 | **1.4%** | **0.0%** |
| `fixed_30` | −1.777 | 0.039 | 5.076 | **10.2%** | 16.6% |
| `oracle` | −0.204 | 1.087 | 8.810 | 21.4% | 0.2% |

The stop works exactly as designed — **0.0%** of trades lose more than 1R, against 16.6% for
hold-30. It buys that by cutting the right tail from 10.2% to 1.4% of trades reaching +3R.
For a strategy whose entire thesis is the fat right tail, that is the wrong trade, and it is
the same conclusion the DTE/hold work reached from the other direction.

### A.2 Headroom over each policy (oracle − policy, mean R)

| vs | mom_score | htf_score | random_k |
|---|---|---|---|
| `live_approx` | **2.152** | 1.869 | 1.641 |
| `fixed_5` | 2.148 | 1.870 | 1.693 |
| `fixed_30` | **1.581** | 1.374 | 1.316 |

Median exit session: oracle 22 (momentum) / 16 (HTF) vs `live_approx` 12 / 10. **The oracle
holds roughly twice as long as the live rule.**

### A.3 Headroom is mostly NOT entry-specific

The random-3 control carries 1.32–1.69R of headroom against the same policies — about 83% of
what the ranked arms carry. So most of the prize is a generic property of holding equity
paths with a tight trailing stop, not something our entries specially create. The ranked arms
do have more absolute room (oracle 2.24 vs 1.73), which is consistent with the ordering edge
already established in `23_rank_depth_and_options.md` §2.

---

## B. Recovery — NULL. The learned exit rule does not survive its own control

Trained on TRAIN position-days (273,267), graded on VALIDATION (64,108 position-days,
1,772 entries). Target at each open position-day is the oracle's remaining value,
`cont = max(r_close[t+1:]) − r_close[t]`; the policy exits the first day it predicts
`cont ≤ 0`, so its **default is hold-to-the-end** — which is why the benchmark has to be the
best fixed horizon, not `live_approx`.

| arm | n | live_R | best fixed_R | imitation_R | **null_R** | oracle_R | gain vs fixed30 | 95% CI | **net of null** |
|---|---|---|---|---|---|---|---|---|---|
| mom_score | 434 | 0.179 | 1.210 | 1.950 | **2.033** | 3.299 | +0.740 | [0.517, 0.969] | **−0.083** |
| htf_score | 606 | 0.247 | 1.482 | 1.605 | **1.645** | 2.890 | +0.123 | [−0.003, 0.255] | **−0.039** |
| random_k | 732 | 0.050 | 0.417 | 0.426 | 0.430 | 1.585 | +0.009 | [−0.020, 0.040] | −0.004 |

**In all three arms the permuted-label model matches or beats the real one.** The momentum
arm's +0.740R gain over hold-30, with a CI comfortably clear of zero, is entirely reproduced
by a model that never saw a true label (+0.823R). Net of its own control the learned policy
is **−0.083R**.

Mechanism: mean exit day is **28.9** for the real policy and the null's early-exit rate
(16.8%) is comparable to the real one's (26.3%). Both are "hold to ~29 sessions with a
scattering of early exits", and that shape is worth +0.7R against a hard day-30 exit
*whatever* trained it. `off` is the #2 feature by importance, behind `regime_spy_ret_20`
which is constant within a decision bar — i.e. the model leans on horizon and on
bar-level context, not on the position's own evolving state.

This is the same failure mode `25_matched_control_and_depth_null.md` §B.1 documented for
top-k excess, in a new place. Without this arm the honest-looking headline would have been
"a supervised exit policy recovers 57% of a perfect exit out of sample", and it would have
been wrong.

---

## C. Verdict

**A stands, B is rejected.**

* **The prize is real and large.** The live exit rule gives up ~**0.57R per trade** against
  simply holding 30 sessions, and ~**2.15R** against a hindsight-perfect exit, on 2,430
  momentum and 2,430 HTF entries, with both stated biases pushing the estimate *down*.
* **It is not captured by learning.** No state-dependent exit rule survives its own control
  at these features. What captures a third of it is **changing the fixed policy**: the 1.5-ATR
  stop with a 2.4-ATR trail and a 15-session time stop is cutting the right tail
  (share of trades reaching +3R falls 10.2% → 1.4%) to buy a loss cap (0.0% of trades below
  −1R) that a fat-tail strategy should not want.
* **Most of the headroom is generic**, not entry-specific: the random-3 control carries 83%
  of the same room, so this is a property of holding equity paths under a tight trailing
  stop rather than something our ranking creates.

### C.1 Pre-registered consequence for Stage 4

`00_preregistration.md` §6.5 and the plan's Stage 3 kill criterion fire here: a negligible
out-of-sample recovery means **the exit decision is not learnable from the current features,
and the named next variant is FEATURES — forward realised volatility first, given
`research/regime_coverage_2026-09-21/forward_rv_predictability.py` measured forward-20d RV at
OOS R² = 0.742 against 0.003 for forward return — not a different optimizer.**

Stage 4 (fitted-Q) is therefore run as a **confirmatory test with a pre-declared expectation
of null**, not as a second attempt at a rejected hypothesis. It contributes one thing the
imitation could not: an **offset-only control arm**, which asks directly whether *any*
state-dependent exit signal exists beyond a holding horizon. A positive Stage 4 result would
be UNCERTIFIED and would require the fresh post-2026-05-14 holdout before anyone acts on it.

### C.2 What is actionable now, with no model

The stop-and-trail geometry, not the optimizer. That is a configuration question for
`momentum_config`'s RISK block and it sits directly on the open roadmap item (the (a)/(b)
DTE-vs-hold decision per module). **Nothing here has been changed** — this is equity-path
evidence and the live book routes much of its risk through options, whose ~35% round-trip
hurdle is not in these numbers.
